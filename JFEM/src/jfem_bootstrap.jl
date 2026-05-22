# jfem_bootstrap.jl -- Idempotent source loader for non-package workflows.
#
# Many repo scripts use `include("src/main.jl")` or directly include source
# files from an already-running Julia session. Without guards, that rebuilds
# the source environment and replaces modules like `FEM` and `Solver` every
# time. Keep the load idempotent so repeated solver calls in the same process
# reuse the existing source environment.

if !isdefined(@__MODULE__, :FEM)
    include("FEMKernels.jl")
end
using .FEM

if !isdefined(@__MODULE__, :NastranParser)
    include(joinpath("parsing", "NastranParser.jl"))
end
using .NastranParser

if !isdefined(@__MODULE__, :Solver)
    include(joinpath("solver", "Solver.jl"))
elseif isdefined(Solver, :load_full_solver_extensions!)
    Solver.load_full_solver_extensions!()
end
using .Solver

if !isdefined(@__MODULE__, :build_model)
    include("ModelBuilder.jl")
end

if !isdefined(@__MODULE__, :solve_model)
    include("JFEMSolver.jl")
end
