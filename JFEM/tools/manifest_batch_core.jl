# Shared JSON-manifest batch execution helpers.
#
# This file is included by run_batch_manifest.jl and jfem_worker_jsonl.jl.
# The including script must load OpenJFEM before calling run_batch_manifest!.

using Dates
using JSON
using Printf

const MANIFEST_FAST_DEFAULT_FLAGS = Dict{String,String}(
    "JFEM_EXPORT_BINARY" => "false",
    "JFEM_MATRIX_ASYMMETRY_CHECK" => "false",
    "JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES" => "false",
    "JFEM_SUPPRESS_THREAD_HINT" => "1",
)

function _manifest_bool(value, default::Bool=false)
    value === nothing && return default
    value isa Bool && return value
    value isa Number && return value != 0
    raw = lowercase(strip(string(value)))
    isempty(raw) && return default
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    return default
end

function _manifest_get(d::AbstractDict, key::AbstractString, default=nothing)
    return haskey(d, key) ? d[key] : default
end

function _manifest_string_dict(value)
    out = Dict{String,String}()
    value === nothing && return out
    if value isa AbstractDict
        for (k, v) in value
            out[string(k)] = string(v)
        end
    elseif value isa AbstractString
        sep = occursin(";", value) ? ";" : ","
        for item in split(value, sep)
            raw = strip(item)
            isempty(raw) && continue
            parts = split(raw, "="; limit=2)
            length(parts) == 2 || error("invalid flag assignment in manifest: $raw")
            out[strip(parts[1])] = strip(parts[2])
        end
    else
        error("flags must be an object or FLAG=value string")
    end
    return out
end

function manifest_default_flags(manifest::AbstractDict)
    defaults = _manifest_get(manifest, "defaults", Dict{String,Any}())
    flags = copy(MANIFEST_FAST_DEFAULT_FLAGS)
    merge!(flags, _manifest_string_dict(_manifest_get(defaults, "flags", nothing)))
    merge!(flags, _manifest_string_dict(_manifest_get(defaults, "flags_raw", nothing)))
    return flags
end

function manifest_apply_flags!(flags::AbstractDict)
    applied = Dict{String,String}()
    for key in sort(collect(keys(flags)); by=string)
        k = string(key)
        v = string(flags[key])
        ENV[k] = v
        applied[k] = v
    end
    return applied
end

function _manifest_with_env(f::Function, flags::AbstractDict)
    old = Dict{String,Union{Nothing,String}}()
    for (key, value) in flags
        k = string(key)
        old[k] = haskey(ENV, k) ? ENV[k] : nothing
        ENV[k] = string(value)
    end
    try
        return f()
    finally
        for (key, value) in old
            if value === nothing
                delete!(ENV, key)
            else
                ENV[key] = value
            end
        end
    end
end

function _manifest_output_bool(options::AbstractDict, names, default::Bool=false)
    for name in names
        haskey(options, name) && return _manifest_bool(options[name], default)
    end
    return default
end

function _manifest_case_output_options(defaults::AbstractDict, case::AbstractDict)
    default_opts = _manifest_get(defaults, "output_options", Dict{String,Any}())
    case_opts = _manifest_get(case, "output_options", Dict{String,Any}())
    out = Dict{String,Any}()
    default_opts isa AbstractDict && merge!(out, Dict(string(k) => v for (k, v) in default_opts))
    case_opts isa AbstractDict && merge!(out, Dict(string(k) => v for (k, v) in case_opts))
    return out
end

function _manifest_export_options(options::AbstractDict, flags::AbstractDict)
    binary_default = lowercase(strip(string(get(flags, "JFEM_EXPORT_BINARY", "true")))) in
                     ("1", "true", "yes", "on")
    eigenvalues_only = _manifest_output_bool(options,
        ("eigenvalues_only", "sol105_eigenvalues_only", "buckling_factors_only", "values_only"),
        false)
    binary = _manifest_output_bool(options,
        ("binary", "jfem_binary", "export_binary", "export_jfem_binary"),
        binary_default)
    return (
        export_model_json = _manifest_output_bool(options, ("model_json", "export_model_json"), false),
        export_card_inventory = _manifest_output_bool(options, ("card_inventory", "export_card_inventory"), false),
        export_json = _manifest_output_bool(options, ("json", "export_json", "results_json"), false),
        export_vtk = !eigenvalues_only && _manifest_output_bool(options, ("vtk", "export_vtk"), false),
        export_hdf5 = !eigenvalues_only && _manifest_output_bool(options, ("hdf5", "h5", "export_hdf5"), false),
        export_jfem_binary = !eigenvalues_only && binary,
        export_report = _manifest_output_bool(options, ("report", "markdown_report", "export_report"), true),
        eigenvalues_only = eigenvalues_only,
    )
end

function _manifest_abs_path(path::AbstractString, base_dir::AbstractString=pwd())
    raw = strip(path)
    isempty(raw) && error("empty path")
    return normpath(isabspath(raw) ? raw : joinpath(base_dir, raw))
end

function _manifest_case_slug(raw)
    slug = replace(string(raw), '\\' => '_', '/' => '_')
    slug = replace(slug, r"[^A-Za-z0-9_.-]+" => "_")
    slug = strip(slug, ['_', '.', '-'])
    return isempty(slug) ? "case" : slug
end

function _manifest_csv_escape(x)
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s) || occursin('\r', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function _manifest_write_csv_row(io, values)
    println(io, join(_manifest_csv_escape.(values), ","))
end

function load_batch_manifest(path::AbstractString)
    p = abspath(path)
    data = JSON.parsefile(p)
    data isa AbstractDict || error("batch manifest must be a JSON object: $p")
    return Dict{String,Any}(string(k) => v for (k, v) in data), p
end

function _manifest_normalize_cases(manifest::AbstractDict, manifest_path::Union{Nothing,String})
    base_dir = manifest_path === nothing ? pwd() : dirname(abspath(manifest_path))
    output_root_raw = _manifest_get(manifest, "output_root", nothing)
    output_root_raw === nothing && error("batch manifest requires output_root")
    output_root = _manifest_abs_path(string(output_root_raw), base_dir)
    cases_raw = _manifest_get(manifest, "cases", nothing)
    cases_raw isa AbstractVector || error("batch manifest requires cases array")
    isempty(cases_raw) && error("batch manifest cases array is empty")

    used = Dict{String,Int}()
    cases = Vector{Dict{String,Any}}()
    for (idx, item) in enumerate(cases_raw)
        item isa AbstractDict || error("case $idx must be an object")
        case = Dict{String,Any}(string(k) => v for (k, v) in item)
        input_raw = _manifest_get(case, "input", _manifest_get(case, "deck", nothing))
        input_raw === nothing && error("case $idx requires input")
        input_path = _manifest_abs_path(string(input_raw), base_dir)
        isfile(input_path) || error("case $idx input deck not found: $input_path")
        case_id = string(_manifest_get(case, "case_id", splitext(basename(input_path))[1]))
        slug_base = _manifest_case_slug(case_id)
        n = get(used, slug_base, 0) + 1
        used[slug_base] = n
        slug = n == 1 ? slug_base : "$(slug_base)_$(n)"
        out_raw = _manifest_get(case, "output_dir", nothing)
        output_dir = out_raw === nothing ?
            joinpath(output_root, slug) :
            _manifest_abs_path(string(out_raw), base_dir)
        push!(cases, Dict{String,Any}(
            "index" => idx,
            "case_id" => case_id,
            "slug" => slug,
            "input" => input_path,
            "output_dir" => output_dir,
            "raw" => case,
        ))
    end
    return output_root, cases
end

function _manifest_report_path(deck::AbstractString, output_dir::AbstractString)
    return joinpath(output_dir, splitext(basename(deck))[1] * ".REPORT.md")
end

function _manifest_run_one_case!(case::AbstractDict, manifest::AbstractDict;
                                 manifest_path::Union{Nothing,String},
                                 repo_root::AbstractString,
                                 script_path::AbstractString,
                                 args,
                                 default_flags::AbstractDict,
                                 default_applied_flags::AbstractDict,
                                 quiet::Bool,
                                 command_extra::AbstractDict)
    defaults = _manifest_get(manifest, "defaults", Dict{String,Any}())
    raw_case = case["raw"]
    case_flags = copy(default_flags)
    merge!(case_flags, _manifest_string_dict(_manifest_get(raw_case, "flags", nothing)))
    merge!(case_flags, _manifest_string_dict(_manifest_get(raw_case, "flags_raw", nothing)))
    options = _manifest_case_output_options(defaults, raw_case)
    export_opts = _manifest_export_options(options, case_flags)
    if export_opts.eigenvalues_only
        case_flags["JFEM_SOL105_EIGENVALUES_ONLY"] = "true"
        case_flags["JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES"] = "false"
    end
    if export_opts.export_jfem_binary
        case_flags["JFEM_EXPORT_BINARY"] = "true"
    else
        case_flags["JFEM_EXPORT_BINARY"] = "false"
    end

    deck = string(case["input"])
    out_dir = string(case["output_dir"])
    mkpath(out_dir)
    case_log = joinpath(out_dir, "jfem_case_stdout.log")

    applied_case_flags = Dict{String,String}(string(k) => string(v) for (k, v) in case_flags)
    extra = Dict{String,Any}(
        "batch_manifest" => manifest_path === nothing ? "" : abspath(manifest_path),
        "batch_id" => string(_manifest_get(manifest, "batch_id", "")),
        "case_id" => string(case["case_id"]),
        "case_index" => case["index"],
        "flags_raw" => join(["$k=$(applied_case_flags[k])" for k in sort(collect(keys(applied_case_flags)))], ","),
        "export_jfem_binary" => export_opts.export_jfem_binary,
        "export_report" => export_opts.export_report,
        "output_options" => Dict(String(k) => v for (k, v) in pairs(options)),
        "quiet_log" => quiet ? case_log : "",
    )
    for (k, v) in command_extra
        extra[string(k)] = v
    end

    if quiet
        open(case_log, "w") do io
            redirect_stdout(io) do
                write_run_manifest(out_dir;
                    repo_root=repo_root,
                    bdf_path=deck,
                    script_path=script_path,
                    args=args,
                    applied_flags=applied_case_flags,
                    extra=extra)
            end
        end
    else
        write_run_manifest(out_dir;
            repo_root=repo_root,
            bdf_path=deck,
            script_path=script_path,
            args=args,
            applied_flags=applied_case_flags,
            extra=extra)
    end

    t0 = time_ns()
    _manifest_with_env(case_flags) do
        if quiet
            open(case_log, "a") do io
                redirect_stdout(io) do
                    redirect_stderr(io) do
                        OpenJFEM.main(deck;
                            output_dir=out_dir,
                            export_model_json=export_opts.export_model_json,
                            export_card_inventory=export_opts.export_card_inventory,
                            export_json=export_opts.export_json,
                            export_vtk=export_opts.export_vtk,
                            export_hdf5=export_opts.export_hdf5,
                            export_jfem_binary=export_opts.export_jfem_binary,
                            export_report=export_opts.export_report)
                    end
                end
            end
        else
            OpenJFEM.main(deck;
                output_dir=out_dir,
                export_model_json=export_opts.export_model_json,
                export_card_inventory=export_opts.export_card_inventory,
                export_json=export_opts.export_json,
                export_vtk=export_opts.export_vtk,
                export_hdf5=export_opts.export_hdf5,
                export_jfem_binary=export_opts.export_jfem_binary,
                export_report=export_opts.export_report)
        end
    end
    wall = (time_ns() - t0) * 1e-9
    return Dict{String,Any}(
        "index" => case["index"],
        "case_id" => case["case_id"],
        "status" => "ok",
        "wall_s" => wall,
        "input" => deck,
        "output_dir" => out_dir,
        "report" => export_opts.export_report ? _manifest_report_path(deck, out_dir) : "",
        "log" => quiet ? case_log : "",
        "error" => "",
    )
end

function run_batch_manifest!(manifest::AbstractDict;
                             manifest_path::Union{Nothing,String}=nothing,
                             repo_root::AbstractString=normpath(joinpath(@__DIR__, "..", "..")),
                             script_path::AbstractString=@__FILE__,
                             args=String[],
                             quiet::Bool=false,
                             stop_on_error_override=nothing,
                             command_extra=Dict{String,Any}())
    defaults = _manifest_get(manifest, "defaults", Dict{String,Any}())
    output_root, cases = _manifest_normalize_cases(manifest, manifest_path)
    mkpath(output_root)

    default_flags = manifest_default_flags(manifest)
    default_applied_flags = Dict{String,String}(string(k) => string(v) for (k, v) in default_flags)
    gc_between = _manifest_bool(_manifest_get(defaults, "gc_between", true), true)
    stop_on_error = stop_on_error_override === nothing ?
        _manifest_bool(_manifest_get(defaults, "stop_on_error", false), false) :
        Bool(stop_on_error_override)

    summary_csv = joinpath(output_root, "batch_summary.csv")
    summary_json = joinpath(output_root, "batch_summary.json")

    batch_start = time_ns()
    rows = Vector{Dict{String,Any}}()
    completed = 0
    failed = 0

    open(summary_csv, "w") do csv
        _manifest_write_csv_row(csv, ("index", "case_id", "status", "wall_s", "input", "output_dir", "report", "log", "error"))
        for case in cases
            gc_between && Int(case["index"]) > 1 && GC.gc()
            row = Dict{String,Any}()
            try
                row = _manifest_run_one_case!(case, manifest;
                    manifest_path=manifest_path,
                    repo_root=repo_root,
                    script_path=script_path,
                    args=args,
                    default_flags=default_flags,
                    default_applied_flags=default_applied_flags,
                    quiet=quiet,
                    command_extra=command_extra)
                completed += 1
            catch err
                failed += 1
                row = Dict{String,Any}(
                    "index" => case["index"],
                    "case_id" => case["case_id"],
                    "status" => "failed",
                    "wall_s" => 0.0,
                    "input" => case["input"],
                    "output_dir" => case["output_dir"],
                    "report" => "",
                    "log" => quiet ? joinpath(string(case["output_dir"]), "jfem_case_stdout.log") : "",
                    "error" => sprint(showerror, err),
                )
                if stop_on_error
                    push!(rows, row)
                    _manifest_write_csv_row(csv, (row["index"], row["case_id"], row["status"], row["wall_s"], row["input"], row["output_dir"], row["report"], row["log"], row["error"]))
                    rethrow()
                end
            end
            push!(rows, row)
            _manifest_write_csv_row(csv, (row["index"], row["case_id"], row["status"], row["wall_s"], row["input"], row["output_dir"], row["report"], row["log"], row["error"]))
        end
    end

    total_wall = (time_ns() - batch_start) * 1e-9
    summary = Dict{String,Any}(
        "created_utc" => string(Dates.now(Dates.UTC)),
        "batch_id" => string(_manifest_get(manifest, "batch_id", "")),
        "manifest" => manifest_path === nothing ? "" : abspath(manifest_path),
        "output_root" => output_root,
        "summary_csv" => summary_csv,
        "summary_json" => summary_json,
        "completed" => completed,
        "failed" => failed,
        "total" => length(cases),
        "total_wall_s" => total_wall,
        "gc_between" => gc_between,
        "stop_on_error" => stop_on_error,
        "quiet" => quiet,
        "cases" => rows,
    )
    open(summary_json, "w") do io
        JSON.print(io, summary, 4)
        println(io)
    end
    return summary
end
