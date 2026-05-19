# Usage:
#   julia --threads=8 --project=JFEM JFEM/tools/testing/run_bdf_batch.jl <input> <output_root> [FLAG=val,FLAG2=val2] [options]
#
# <input> may be:
#   - one .bdf/.dat/.nas file
#   - a directory containing decks
#   - a text/CSV manifest with one deck path per line or in the first column
#
# Options:
#   --recursive        recurse when <input> is a directory
#   --stop-on-error    abort after the first failed case
#   --dry-run          list cases and output directories without solving
#   --ext=.bdf,.dat    deck extensions for directory scans (default .bdf,.dat,.nas)
#   --no-gc-between    skip the default explicit GC between cases
#
# Runs all cases in one Julia process so package load and method compilation are
# paid once. Each case still receives an independent output directory and
# manifest.

include(joinpath(@__DIR__, "run_manifest.jl"))
using Printf

length(ARGS) < 2 && error("usage: run_bdf_batch.jl <input> <output_root> [FLAG=val,FLAG2=val2] [--recursive] [--stop-on-error] [--dry-run] [--ext=.bdf,.dat,.nas]")

const INPUT_RAW = ARGS[1]
const OUT_ROOT = normpath(ARGS[2])

mutable struct BatchOptions
    flags_raw::String
    recursive::Bool
    stop_on_error::Bool
    dry_run::Bool
    gc_between::Bool
    extensions::Set{String}
end

function _parse_batch_options(args)
    opts = BatchOptions("", false, false, false, true, Set([".bdf", ".dat", ".nas"]))
    for arg in args
        if startswith(arg, "--ext=")
            raw = strip(arg[length("--ext=")+1:end])
            isempty(raw) && error("--ext requires at least one extension")
            exts = Set{String}()
            for item in split(raw, occursin(";", raw) ? ";" : ",")
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
        elseif startswith(arg, "--")
            error("unknown option: $arg")
        elseif isempty(opts.flags_raw)
            opts.flags_raw = arg
        else
            error("unexpected positional argument: $arg")
        end
    end
    return opts
end

const OPTS = _parse_batch_options(ARGS[3:end])
const APPLIED_FLAGS = apply_jfem_flags!(OPTS.flags_raw)
using OpenJFEM
const EXPORT_JFEM_BINARY = lowercase(strip(get(ENV, "JFEM_EXPORT_BINARY", "true"))) in
                           ("1", "true", "yes", "on")

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

decks, deck_base = _find_decks(INPUT_RAW, OPTS)
isempty(decks) && error("no supported deck files found")
mkpath(OUT_ROOT)

used_slugs = Dict{String,Int}()
case_rows = Vector{Dict{String,Any}}()
for (i, deck) in enumerate(decks)
    slug = _slug_for_deck(deck, deck_base, used_slugs)
    out_dir = joinpath(OUT_ROOT, slug)
    push!(case_rows, Dict{String,Any}(
        "index" => i,
        "deck" => deck,
        "slug" => slug,
        "output_dir" => out_dir,
    ))
end

println(">>> batch_input=$(normpath(_resolve_input_path(INPUT_RAW)))")
println(">>> output_root=$OUT_ROOT")
println(">>> cases=$(length(case_rows))")
println(">>> recursive=$(OPTS.recursive), dry_run=$(OPTS.dry_run), stop_on_error=$(OPTS.stop_on_error), gc_between=$(OPTS.gc_between)")

if OPTS.dry_run
    for row in case_rows
        println(">>> [$(row["index"])] $(row["deck"]) -> $(row["output_dir"])")
    end
    exit(0)
end

summary_csv = joinpath(OUT_ROOT, "batch_summary.csv")
summary_json = joinpath(OUT_ROOT, "batch_summary.json")

batch_start = time_ns()
completed = Ref(0)
failed = Ref(0)

open(summary_csv, "w") do csv
    _write_csv_row(csv, ("index", "status", "wall_s", "deck", "output_dir", "error"))
    for row in case_rows
        OPTS.gc_between && row["index"] > 1 && GC.gc()
        idx = row["index"]
        deck = row["deck"]
        out_dir = row["output_dir"]
        mkpath(out_dir)

        println(">>> [$idx/$(length(case_rows))] running $deck")
        write_run_manifest(out_dir;
            repo_root=normpath(joinpath(@__DIR__, "..", "..", "..")),
            bdf_path=deck,
            script_path=@__FILE__,
            args=ARGS,
            applied_flags=APPLIED_FLAGS,
            extra=Dict{String,Any}(
                "flags_raw" => OPTS.flags_raw,
                "batch_input" => INPUT_RAW,
                "batch_index" => idx,
                "batch_count" => length(case_rows),
                "batch_output_root" => OUT_ROOT,
                "export_jfem_binary" => EXPORT_JFEM_BINARY,
            ))

        t0 = time_ns()
        status = "ok"
        err_msg = ""
        try
            OpenJFEM.main(deck; output_dir=out_dir, export_jfem_binary=EXPORT_JFEM_BINARY)
            completed[] += 1
        catch err
            failed[] += 1
            status = "failed"
            err_msg = sprint(showerror, err, catch_backtrace())
            println(">>> [$idx/$(length(case_rows))] FAILED: $(sprint(showerror, err))")
        end
        wall = (time_ns() - t0) * 1e-9
        row["status"] = status
        row["wall_s"] = wall
        row["error"] = err_msg
        _write_csv_row(csv, (idx, status, @sprintf("%.6f", wall), deck, out_dir, err_msg))
        flush(csv)

        println(">>> [$idx/$(length(case_rows))] status=$status wall=$(round(wall; digits=3)) s")
        if status != "ok" && OPTS.stop_on_error
            break
        end
    end
end

total_wall = (time_ns() - batch_start) * 1e-9
summary = Dict{String,Any}(
    "input" => normpath(_resolve_input_path(INPUT_RAW)),
    "output_root" => abspath(OUT_ROOT),
    "flags_raw" => OPTS.flags_raw,
    "export_jfem_binary" => EXPORT_JFEM_BINARY,
    "recursive" => OPTS.recursive,
    "stop_on_error" => OPTS.stop_on_error,
    "gc_between" => OPTS.gc_between,
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

println(">>> batch_summary_csv=$summary_csv")
println(">>> batch_summary_json=$summary_json")
println(">>> batch completed=$(completed[]) failed=$(failed[]) total_wall=$(round(total_wall; digits=3)) s")
exit(failed[] == 0 ? 0 : 1)
