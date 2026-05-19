# One-command setup for fast SOL 105 runs.
#
# Usage:
#   julia --threads=auto --startup-file=no --project=JFEM JFEM/tools/precompile_sol105.jl path/to/representative_sol105.bdf
#
# Optional second argument:
#   "JFEM_EXPORT_BINARY=false,JFEM_SUPPRESS_THREAD_HINT=1"

using Pkg

const DEFAULT_FLAGS = "JFEM_EXPORT_BINARY=false,JFEM_SUPPRESS_THREAD_HINT=1"

function _usage()
    return """
    usage:
      julia --threads=auto --startup-file=no --project=JFEM JFEM/tools/precompile_sol105.jl <representative_sol105.bdf> [FLAG=val,FLAG2=val2]

    The representative deck should be a typical SOL 105 case from the same
    model family you plan to run later. The optional flag string defaults to:
      $(DEFAULT_FLAGS)
    """
end

function _abs_existing_file(path::AbstractString)
    p = normpath(isabspath(path) ? path : joinpath(pwd(), path))
    if !isfile(p)
        println(stderr, "ERROR: representative SOL 105 deck not found: $p")
        exit(1)
    end
    return p
end

function _setenv_preserving!(old::Dict{String,Union{Nothing,String}},
                             key::AbstractString,
                             value::AbstractString)
    k = String(key)
    old[k] = haskey(ENV, k) ? ENV[k] : nothing
    ENV[k] = String(value)
    return nothing
end

function _restore_env!(old::Dict{String,Union{Nothing,String}})
    for (key, value) in old
        if value === nothing
            delete!(ENV, key)
        else
            ENV[key] = value
        end
    end
    return nothing
end

if length(ARGS) == 1 && ARGS[1] in ("-h", "--help", "help")
    print(_usage())
    exit(0)
end

if length(ARGS) < 1 || length(ARGS) > 2
    print(_usage())
    exit(1)
end

const REPRESENTATIVE_BDF = _abs_existing_file(ARGS[1])
const PRECOMPILE_FLAGS = length(ARGS) >= 2 ? strip(ARGS[2]) : DEFAULT_FLAGS

old_env = Dict{String,Union{Nothing,String}}()
try
    _setenv_preserving!(old_env, "JFEM_SOL105_PRECOMPILE_WORKLOAD", "true")
    _setenv_preserving!(old_env, "JFEM_SOL105_PRECOMPILE_BDF", REPRESENTATIVE_BDF)
    _setenv_preserving!(old_env, "JFEM_SOL105_PRECOMPILE_FLAGS", PRECOMPILE_FLAGS)
    _setenv_preserving!(old_env, "JFEM_SUPPRESS_THREAD_HINT", "1")

    println("OpenJFEM SOL 105 precompile setup")
    println("  representative deck: ", REPRESENTATIVE_BDF)
    println("  precompile flags:    ", PRECOMPILE_FLAGS)
    println("  Julia threads:       ", Threads.nthreads())
    println()

    println("Instantiating project dependencies...")
    Pkg.instantiate()

    println("Precompiling OpenJFEM with the representative SOL 105 workload...")
    Pkg.precompile()

    println("Verifying package load...")
    @eval using OpenJFEM

    println()
    println("OpenJFEM SOL 105 precompile setup complete.")
    println("You can now run single cases with run_bdf.jl or batches with run_bdf_batch.jl.")
finally
    _restore_env!(old_env)
end
