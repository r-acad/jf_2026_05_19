# jfem_solver_bootstrap.jl -- Lean idempotent loader for solver-only workflows.
#
# This path deliberately avoids loading adjoint / optimization extensions and
# export dependencies until they are actually requested. It is the fastest
# startup path for scripts that only need `bdf_to_model(...)` / `solve_model(...)`.

if !isdefined(@__MODULE__, :FEM)
    include("FEMKernels.jl")
end
using .FEM

if !isdefined(@__MODULE__, :NastranParser)
    include(joinpath("parsing", "NastranParser.jl"))
end
using .NastranParser

if !isdefined(@__MODULE__, :Solver)
    include(joinpath("solver", "SolverCore.jl"))
end
using .Solver

if !isdefined(@__MODULE__, :build_model)
    include("ModelBuilder.jl")
end

if !isdefined(@__MODULE__, :solve_model)
    include("JFEMSolver.jl")
end
