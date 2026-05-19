# Optional representative precompile workload.
#
# Enable before package precompilation with:
#   JFEM_PRECOMPILE_WORKLOAD=true
#
# Optionally override the BDF list with semicolon-separated absolute/relative
# paths in JFEM_PRECOMPILE_BDF and the flag set with
# JFEM_PRECOMPILE_FLAGS="FLAG=val,FLAG2=val2".
#
# The older JFEM_SOL105_PRECOMPILE_* names are still accepted.

const _JFEM_PRECOMPILE_TRUE = Set(("1", "true", "yes", "on"))

@inline function _jfem_precompile_bool(key::String, default::Bool=false)
    raw = lowercase(strip(get(ENV, key, default ? "true" : "false")))
    return raw in _JFEM_PRECOMPILE_TRUE
end

function _jfem_precompile_split_paths(raw::AbstractString)
    paths = String[]
    for item in split(raw, ';')
        path = strip(item)
        isempty(path) && continue
        push!(paths, normpath(isabspath(path) ? path : abspath(path)))
    end
    return paths
end

function _jfem_default_precompile_bdfs()
    examples_dir = normpath(joinpath(@__DIR__, "..", "examples", "precompile"))
    isdir(examples_dir) || return String[]
    paths = String[]
    for name in sort(readdir(examples_dir))
        ext = lowercase(splitext(name)[2])
        ext in (".bdf", ".dat", ".nas") || continue
        path = joinpath(examples_dir, name)
        isfile(path) && push!(paths, path)
    end
    return paths
end

function _jfem_precompile_bdfs()
    raw = strip(get(ENV, "JFEM_PRECOMPILE_BDF", ""))
    isempty(raw) && (raw = strip(get(ENV, "JFEM_SOL105_PRECOMPILE_BDF", "")))
    if !isempty(raw)
        return filter(isfile, _jfem_precompile_split_paths(raw))
    end
    return _jfem_default_precompile_bdfs()
end

function _jfem_precompile_flags()
    raw = strip(get(ENV, "JFEM_PRECOMPILE_FLAGS", ""))
    isempty(raw) && (raw = strip(get(ENV, "JFEM_SOL105_PRECOMPILE_FLAGS", "")))
    pairs = Pair{String,String}[]
    isempty(raw) && return pairs
    sep = occursin(";", raw) ? ";" : ","
    for kv in split(raw, sep)
        isempty(strip(kv)) && continue
        parts = split(kv, "="; limit=2)
        length(parts) == 2 || continue
        push!(pairs, strip(parts[1]) => strip(parts[2]))
    end
    return pairs
end

function _jfem_with_env(f::Function, pairs)
    old = Dict{String,Union{Nothing,String}}()
    for (key, value) in pairs
        old[key] = haskey(ENV, key) ? ENV[key] : nothing
        ENV[key] = value
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

function _jfem_precompile_solve_bdf(path::AbstractString)
    lines = readlines(path)
    lines = NastranParser.resolve_includes(lines, dirname(abspath(path)))
    lines = NastranParser.convert_mystran_to_nastran(lines)
    cc, bulk = NastranParser.read_bulk_and_case(lines)
    cards = NastranParser.process_cards(bulk)
    model = build_model(cards, cc)
    resolve_nested_coords!(model)
    transform_geometry!(model)
    results = solve_model(model)
    return get(results, "sol_type", nothing)
end

if _jfem_precompile_bool("JFEM_PRECOMPILE_WORKLOAD", false) ||
   _jfem_precompile_bool("JFEM_SOL105_PRECOMPILE_WORKLOAD", false) ||
   haskey(ENV, "JFEM_PRECOMPILE_BDF") ||
   haskey(ENV, "JFEM_SOL105_PRECOMPILE_BDF")
    @setup_workload begin
        bdfs = _jfem_precompile_bdfs()
        flags = _jfem_precompile_flags()
        @compile_workload begin
            _jfem_with_env(flags) do
                redirect_stdout(devnull) do
                    redirect_stderr(devnull) do
                        for bdf in bdfs
                            _jfem_precompile_solve_bdf(bdf)
                        end
                    end
                end
            end
        end
    end
end
