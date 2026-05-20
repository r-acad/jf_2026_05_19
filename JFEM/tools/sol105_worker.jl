# Persistent SOL 105 worker.
#
# Start once, then submit single-case or batch commands on stdin. This keeps
# Julia, OpenJFEM, and compiled hot methods alive across multiple jobs.

function _print_cli_usage()
    println("""
Usage:
  julia --threads=auto --startup-file=no --project=JFEM JFEM/tools/sol105_worker.jl [FLAG=val,FLAG2=val2]

Default flags:
  JFEM_EXPORT_BINARY=false,JFEM_MATRIX_ASYMMETRY_CHECK=false,JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES=false,JFEM_SUPPRESS_THREAD_HINT=1

Worker commands:
  run <deck> | <output_dir>
  batch <input> | <output_root> [| --recursive --stop-on-error --ext=.bdf,.dat,.nas --no-gc-between]
  status
  gc
  help
  quit
""")
end

if length(ARGS) == 1 && lowercase(strip(ARGS[1])) in ("-h", "--help", "help")
    _print_cli_usage()
    exit(0)
end

length(ARGS) > 1 && error("usage: sol105_worker.jl [FLAG=val,FLAG2=val2]")

include(joinpath(@__DIR__, "testing", "run_manifest.jl"))

using Dates
using JSON
using Printf

const DEFAULT_FLAGS = "JFEM_EXPORT_BINARY=false,JFEM_MATRIX_ASYMMETRY_CHECK=false,JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES=false,JFEM_SUPPRESS_THREAD_HINT=1"
const FLAGS_RAW = length(ARGS) == 1 ? strip(ARGS[1]) : DEFAULT_FLAGS
const APPLIED_FLAGS = apply_jfem_flags!(FLAGS_RAW)

using OpenJFEM

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const WORKER_SESSION_ID = string(Dates.format(Dates.now(Dates.UTC), dateformat"yyyymmddTHHMMSS"), "-pid", getpid())
const EXPORT_JFEM_BINARY = lowercase(strip(get(ENV, "JFEM_EXPORT_BINARY", "true"))) in
                           ("1", "true", "yes", "on")

mutable struct WorkerCounters
    commands::Int
    completed::Int
    failed::Int
end

mutable struct BatchOptions
    recursive::Bool
    stop_on_error::Bool
    dry_run::Bool
    gc_between::Bool
    extensions::Set{String}
end

function _default_batch_options()
    return BatchOptions(false, false, false, true, Set([".bdf", ".dat", ".nas"]))
end

function _parse_batch_options(raw::AbstractString)
    opts = _default_batch_options()
    for arg in split(strip(raw))
        isempty(arg) && continue
        if startswith(arg, "--ext=")
            ext_raw = strip(arg[length("--ext=")+1:end])
            isempty(ext_raw) && error("--ext requires at least one extension")
            exts = Set{String}()
            for item in split(ext_raw, occursin(";", ext_raw) ? ";" : ",")
                ext = lowercase(strip(item))
                isempty(ext) && continue
                startswith(ext, ".") || (ext = "." * ext)
                push!(exts, ext)
            end
            isempty(exts) && error("--ext requires at least one extension")
            opts.extensions = exts
        elseif arg == "--recursive"
            opts.recursive = true
        elseif arg == "--stop-on-error"
            opts.stop_on_error = true
        elseif arg == "--dry-run"
            opts.dry_run = true
        elseif arg == "--no-gc-between"
            opts.gc_between = false
        else
            error("unknown batch option: $arg")
        end
    end
    return opts
end

function _is_deck(path::AbstractString, extensions::Set{String})
    return lowercase(splitext(path)[2]) in extensions
end

function _resolve_input_path(path::AbstractString, base_dir::AbstractString=pwd())
    p = strip(path)
    isempty(p) && return ""
    return normpath(isabspath(p) ? p : joinpath(base_dir, p))
end

function _read_manifest(path::AbstractString, extensions::Set{String})
    base_dir = dirname(abspath(path))
    decks = String[]
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "#") && continue
        first_col = strip(split(line, ','; limit=2)[1])
        first_col = strip(first_col, ['"', '\''])
        deck = _resolve_input_path(first_col, base_dir)
        isempty(deck) && continue
        isfile(deck) || error("manifest deck not found: $deck")
        _is_deck(deck, extensions) || error("manifest entry is not a supported deck extension: $deck")
        push!(decks, deck)
    end
    return decks
end

function _find_decks(input_path::AbstractString, opts::BatchOptions)
    path = _resolve_input_path(input_path)
    if isdir(path)
        decks = String[]
        if opts.recursive
            for (dir, _, files) in walkdir(path)
                for f in files
                    p = joinpath(dir, f)
                    _is_deck(p, opts.extensions) && push!(decks, normpath(p))
                end
            end
        else
            for f in readdir(path)
                p = joinpath(path, f)
                isfile(p) && _is_deck(p, opts.extensions) && push!(decks, normpath(p))
            end
        end
        return sort!(decks), path
    elseif isfile(path)
        if _is_deck(path, opts.extensions)
            return [path], dirname(path)
        end
        return _read_manifest(path, opts.extensions), nothing
    else
        error("input path not found: $path")
    end
end

function _csv_escape(x)
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s) || occursin('\r', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function _write_csv_row(io, values)
    println(io, join(_csv_escape.(values), ","))
end

function _slug_for_deck(path::AbstractString, base_dir, used::Dict{String,Int})
    stem_source = base_dir === nothing ? basename(path) : relpath(path, base_dir)
    stem = splitext(stem_source)[1]
    slug = replace(stem, '\\' => '_', '/' => '_')
    slug = replace(slug, r"[^A-Za-z0-9_.-]+" => "_")
    slug = strip(slug, ['_', '.', '-'])
    isempty(slug) && (slug = "case")
    n = get(used, slug, 0) + 1
    used[slug] = n
    return n == 1 ? slug : "$(slug)_$(n)"
end

function _split_pipe_payload(payload::AbstractString, expected_min::Int)
    parts = [strip(p) for p in split(payload, '|')]
    length(parts) >= expected_min || error("expected at least $expected_min pipe-separated fields")
    any(isempty, parts[1:expected_min]) && error("empty required field in command")
    return parts
end

function _run_case!(deck_raw::AbstractString, out_dir_raw::AbstractString;
                    command_text::AbstractString,
                    command_index::Integer,
                    extra=Dict{String,Any}())
    deck = _resolve_input_path(deck_raw)
    isfile(deck) || error("deck not found: $deck")
    out_dir = normpath(strip(out_dir_raw))
    isempty(out_dir) && error("output directory is empty")
    mkpath(out_dir)

    manifest_extra = Dict{String,Any}(
        "flags_raw" => FLAGS_RAW,
        "export_jfem_binary" => EXPORT_JFEM_BINARY,
        "worker_session_id" => WORKER_SESSION_ID,
        "worker_command" => command_text,
        "worker_command_index" => command_index,
    )
    for (k, v) in extra
        manifest_extra[string(k)] = v
    end

    write_run_manifest(out_dir;
        repo_root=REPO_ROOT,
        bdf_path=deck,
        script_path=@__FILE__,
        args=[command_text],
        applied_flags=APPLIED_FLAGS,
        extra=manifest_extra)

    t0 = time_ns()
    OpenJFEM.main(deck; output_dir=out_dir, export_jfem_binary=EXPORT_JFEM_BINARY)
    wall = (time_ns() - t0) * 1e-9
    return Dict{String,Any}(
        "deck" => deck,
        "output_dir" => out_dir,
        "wall_s" => wall,
    )
end

function _handle_run!(payload::AbstractString, counters::WorkerCounters, command_text::AbstractString)
    parts = _split_pipe_payload(payload, 2)
    deck, out_dir = parts[1], parts[2]
    result = _run_case!(deck, out_dir;
        command_text=command_text,
        command_index=counters.commands)
    counters.completed += 1
    println(">>> worker run ok wall=$(round(result["wall_s"]; digits=3)) s output=$(result["output_dir"])")
end

function _handle_batch!(payload::AbstractString, counters::WorkerCounters, command_text::AbstractString)
    parts = _split_pipe_payload(payload, 2)
    input_raw, out_root_raw = parts[1], parts[2]
    opts_raw = length(parts) >= 3 ? parts[3] : ""
    opts = _parse_batch_options(opts_raw)

    out_root = normpath(out_root_raw)
    decks, deck_base = _find_decks(input_raw, opts)
    isempty(decks) && error("no supported deck files found")
    mkpath(out_root)

    used_slugs = Dict{String,Int}()
    case_rows = Vector{Dict{String,Any}}()
    for (i, deck) in enumerate(decks)
        slug = _slug_for_deck(deck, deck_base, used_slugs)
        out_dir = joinpath(out_root, slug)
        push!(case_rows, Dict{String,Any}(
            "index" => i,
            "deck" => deck,
            "slug" => slug,
            "output_dir" => out_dir,
        ))
    end

    println(">>> worker batch input=$(normpath(_resolve_input_path(input_raw)))")
    println(">>> worker batch output_root=$out_root")
    println(">>> worker batch cases=$(length(case_rows)) recursive=$(opts.recursive) stop_on_error=$(opts.stop_on_error) gc_between=$(opts.gc_between)")

    if opts.dry_run
        for row in case_rows
            println(">>> [$(row["index"])] $(row["deck"]) -> $(row["output_dir"])")
        end
        return
    end

    summary_csv = joinpath(out_root, "batch_summary.csv")
    summary_json = joinpath(out_root, "batch_summary.json")
    completed = Ref(0)
    failed = Ref(0)
    batch_start = time_ns()

    open(summary_csv, "w") do csv
        _write_csv_row(csv, ("index", "status", "wall_s", "deck", "output_dir", "error"))
        for row in case_rows
            opts.gc_between && row["index"] > 1 && GC.gc()
            idx = row["index"]
            deck = row["deck"]
            out_dir = row["output_dir"]

            println(">>> worker [$(idx)/$(length(case_rows))] running $deck")
            status = "ok"
            err_msg = ""
            wall = 0.0
            try
                result = _run_case!(deck, out_dir;
                    command_text=command_text,
                    command_index=counters.commands,
                    extra=Dict{String,Any}(
                        "batch_input" => input_raw,
                        "batch_index" => idx,
                        "batch_count" => length(case_rows),
                        "batch_output_root" => out_root,
                        "batch_options" => opts_raw,
                    ))
                wall = result["wall_s"]
                completed[] += 1
            catch err
                failed[] += 1
                status = "failed"
                err_msg = sprint(showerror, err, catch_backtrace())
                println(">>> worker [$(idx)/$(length(case_rows))] FAILED: $(sprint(showerror, err))")
            end

            row["status"] = status
            row["wall_s"] = wall
            row["error"] = err_msg
            _write_csv_row(csv, (idx, status, @sprintf("%.6f", wall), deck, out_dir, err_msg))
            flush(csv)

            println(">>> worker [$(idx)/$(length(case_rows))] status=$status wall=$(round(wall; digits=3)) s")
            if status != "ok" && opts.stop_on_error
                break
            end
        end
    end

    total_wall = (time_ns() - batch_start) * 1e-9
    summary = Dict{String,Any}(
        "input" => normpath(_resolve_input_path(input_raw)),
        "output_root" => abspath(out_root),
        "flags_raw" => FLAGS_RAW,
        "export_jfem_binary" => EXPORT_JFEM_BINARY,
        "worker_session_id" => WORKER_SESSION_ID,
        "worker_command" => command_text,
        "recursive" => opts.recursive,
        "stop_on_error" => opts.stop_on_error,
        "gc_between" => opts.gc_between,
        "case_count" => length(case_rows),
        "completed" => completed[],
        "failed" => failed[],
        "total_wall_s" => total_wall,
        "cases" => case_rows,
    )
    open(summary_json, "w") do io
        JSON.print(io, summary, 4)
        println(io)
    end

    counters.completed += completed[]
    counters.failed += failed[]
    println(">>> worker batch_summary_csv=$summary_csv")
    println(">>> worker batch_summary_json=$summary_json")
    println(">>> worker batch completed=$(completed[]) failed=$(failed[]) total_wall=$(round(total_wall; digits=3)) s")
end

function _print_worker_help()
    println("""
Commands:
  run <deck> | <output_dir>
      Solve one input deck and write results to output_dir.

  batch <input> | <output_root> | --stop-on-error
      Solve all decks listed in a manifest, in a directory, or in one deck file.
      The third pipe-separated field is optional batch options.

  status
      Print session id, active flags, completed jobs, and failed jobs.

  gc
      Run Julia garbage collection inside the worker.

  quit
      Stop the worker process.
""")
end

function _print_status(counters::WorkerCounters)
    println(">>> worker_status session=$WORKER_SESSION_ID pid=$(getpid()) threads=$(Threads.nthreads())")
    println(">>> worker_status flags=$FLAGS_RAW export_jfem_binary=$EXPORT_JFEM_BINARY")
    println(">>> worker_status commands=$(counters.commands) completed=$(counters.completed) failed=$(counters.failed)")
end

function _dispatch_command!(line::AbstractString, counters::WorkerCounters)
    stripped = strip(line)
    isempty(stripped) && return true
    startswith(stripped, "#") && return true

    counters.commands += 1
    lower = lowercase(stripped)
    if lower in ("quit", "exit")
        return false
    elseif lower in ("help", "?")
        _print_worker_help()
    elseif lower == "status"
        _print_status(counters)
    elseif lower == "gc"
        GC.gc()
        println(">>> worker gc complete")
    elseif startswith(lower, "run ")
        _handle_run!(stripped[5:end], counters, stripped)
    elseif startswith(lower, "batch ")
        _handle_batch!(stripped[7:end], counters, stripped)
    else
        error("unknown worker command: $stripped")
    end
    return true
end

function main()
    counters = WorkerCounters(0, 0, 0)
    println(">>> OpenJFEM SOL 105 worker ready")
    println(">>> session=$WORKER_SESSION_ID pid=$(getpid()) threads=$(Threads.nthreads())")
    println(">>> flags=$FLAGS_RAW")
    println(">>> type `help` for commands, `quit` to stop")
    flush(stdout)

    keep_running = true
    while keep_running && !eof(stdin)
        print("jfem> ")
        flush(stdout)
        raw = readline(stdin)
        try
            keep_running = _dispatch_command!(raw, counters)
        catch err
            counters.failed += 1
            println(">>> worker command failed: $(sprint(showerror, err))")
        end
        flush(stdout)
    end
    println(">>> OpenJFEM SOL 105 worker stopped")
end

main()
