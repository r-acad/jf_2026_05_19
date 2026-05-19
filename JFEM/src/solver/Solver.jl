# Solver.jl -- Full solver module wrapper.
#
# `SolverCore.jl` builds the lean `Solver` module used by solver-only entry
# points. This wrapper keeps the historic include path working while loading the
# adjoint / optimization extensions on top of that core module.

if !isdefined(@__MODULE__, :Solver)
    include("SolverCore.jl")
end

Solver.load_full_solver_extensions!()
