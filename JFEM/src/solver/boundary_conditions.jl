# boundary_conditions.jl - SPC application, AUTOSPC, and linear solve

@inline function model_autospc_enabled(model)
    raw = get(model, "PARAM_AUTOSPC", true)
    if raw isa AbstractString
        token = uppercase(strip(raw))
        return !(token in ("", "NO", "N", "FALSE", "F", "OFF", "0"))
    elseif raw isa Number
        return abs(Float64(raw)) > 1e-12
    elseif raw === nothing
        return true
    end
    return Bool(raw)
end

@inline function _permanent_grid_components(ps_raw)
    comps = Int[]
    seen = Set{Int}()
    for ch in strip(string(ps_raw))
        if '1' <= ch <= '6'
            comp = Int(ch - '0')
            if !(comp in seen)
                push!(comps, comp)
                push!(seen, comp)
            end
        end
    end
    return comps
end

function _apply_permanent_grid_constraints!(fixed_dofs::Set{Int}, model, id_map)
    permanent_dofs = Set{Int}()
    constrained_grids = 0

    for (grid_key, grid) in get(model, "GRIDs", Dict())
        comps = _permanent_grid_components(get(grid, "PS", ""))
        isempty(comps) && continue

        gid = Int(get(grid, "ID", try parse(Int, string(grid_key)) catch; 0 end))
        idx = get(id_map, gid, 0)
        idx > 0 || continue

        constrained_grids += 1
        for comp in comps
            gdof = (idx - 1) * 6 + comp
            push!(fixed_dofs, gdof)
            push!(permanent_dofs, gdof)
        end
    end

    return length(permanent_dofs), constrained_grids
end

mutable struct LinearSolveCacheEntry
    free_dofs::Vector{Int}
    fixed_dofs::Set{Int}
    spc_dofs::Set{Int}
    enforced_dofs::Vector{Int}
    enforced_values::Vector{Float64}
    K_ff::SparseMatrixCSC{Float64,Int}
    K_fs::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    factor::Any
    diagnostics::Dict{String,Any}
end

create_linear_solve_cache() = Dict{Any,LinearSolveCacheEntry}()

mutable struct EigenSolveCacheEntry
    free_dofs::Vector{Int}
    fixed_dofs::Set{Int}
    bc_diagnostics::Dict{String,Any}
    K_ff::SparseMatrixCSC{Float64,Int}
    factor::Any
    factor_backend::String  # "cholesky" | "lu" | ""
end

create_eigen_solve_cache() = Dict{Any,EigenSolveCacheEntry}()

@inline function linear_solve_cache_min_ndof()
    return max(round(Int, solver_env_float("JFEM_LINEAR_CACHE_MIN_NDOF", 2000.0)), 0)
end

@inline function eigen_solve_cache_min_ndof()
    return max(round(Int, solver_env_float("JFEM_EIGEN_CACHE_MIN_NDOF", Float64(linear_solve_cache_min_ndof()))), 0)
end

@inline function _linear_solve_cache_key(K, ndof::Int, model, spc_id, rbe3_map)
    spc_token = isnothing(spc_id) ? 0 : Int(spc_id)
    rbe3_token = isempty(rbe3_map) ? 0 : objectid(rbe3_map)
    return (
        objectid(K),
        ndof,
        objectid(model),
        spc_token,
        rbe3_token,
        model_autospc_enabled(model),
        autospc_trans_relative_threshold(),
        autospc_rot_relative_threshold(),
    )
end

@inline function _eigen_solve_cache_key(K, ndof::Int, model, spc_id, rbe3_map)
    return _linear_solve_cache_key(K, ndof, model, spc_id, rbe3_map)
end

function prepare_eigen_solve_context(K, ndof, model, id_map, spc_id, rbe3_map; eigen_cache=nothing)
    cache_enabled = eigen_cache !== nothing && ndof >= eigen_solve_cache_min_ndof()
    cache_key = cache_enabled ? _eigen_solve_cache_key(K, ndof, model, spc_id, rbe3_map) : nothing
    cached_entry = (cache_enabled && cache_key !== nothing) ? get(eigen_cache, cache_key, nothing) : nothing

    if cached_entry !== nothing
        return cached_entry, true
    end

    free_dofs, fixed_dofs, bc_diagnostics = compute_free_dofs(
        K, ndof, model, id_map, spc_id, rbe3_map; return_diagnostics=true)
    K_ff = K[free_dofs, free_dofs]
    K_ff = 0.5 * (K_ff + K_ff')

    entry = EigenSolveCacheEntry(
        copy(free_dofs),
        copy(fixed_dofs),
        deepcopy(bc_diagnostics),
        K_ff,
        nothing,
        "",
    )

    if cache_enabled && cache_key !== nothing
        eigen_cache[cache_key] = entry
    end

    return entry, false
end

function ensure_eigen_solve_factorization!(entry::EigenSolveCacheEntry)
    cache_hit = entry.factor !== nothing
    if !cache_hit
        # K_ff is SPD for a well-posed problem → Cholesky is the fast path.
        # If it fails (rigid-body modes, mechanism, bad element), fall back to
        # LU rather than letting PosDefException bubble up and abort the solve.
        backend = "cholesky"
        f = try
            cholesky(entry.K_ff)
        catch e
            if e isa LinearAlgebra.PosDefException ||
               e isa LinearAlgebra.SingularException ||
               e isa LinearAlgebra.ZeroPivotException
                log_msg("[EIGEN] Cholesky failed ($(typeof(e))); falling back to LU. Check model constraints.")
                backend = "lu"
                lu(entry.K_ff)
            else
                rethrow(e)
            end
        end
        entry.factor = f
        entry.factor_backend = backend
    end
    return entry.factor, cache_hit
end

function seed_eigen_solve_cache_from_linear!(eigen_cache, linear_cache, K, ndof::Int, model, spc_id, rbe3_map)
    eigen_cache === nothing && return false
    linear_cache === nothing && return false
    ndof >= eigen_solve_cache_min_ndof() || return false

    linear_key = _linear_solve_cache_key(K, ndof, model, spc_id, rbe3_map)
    linear_entry = get(linear_cache, linear_key, nothing)
    linear_entry === nothing && return false

    eigen_key = _eigen_solve_cache_key(K, ndof, model, spc_id, rbe3_map)
    haskey(eigen_cache, eigen_key) && return false

    solver_diag = get(linear_entry.diagnostics, "linear_solver", Dict{String,Any}())
    linear_backend = lowercase(string(get(solver_diag, "backend", "")))
    factor_backend =
        occursin("cholesky", linear_backend) ? "cholesky" :
        (occursin("lu", linear_backend) ? "lu" : "")
    isempty(factor_backend) && return false

    eigen_cache[eigen_key] = EigenSolveCacheEntry(
        copy(linear_entry.free_dofs),
        copy(linear_entry.fixed_dofs),
        deepcopy(get(linear_entry.diagnostics, "bc_partition", Dict{String,Any}())),
        linear_entry.K_ff,
        linear_entry.factor,
        factor_backend,
    )
    return true
end

function _free_dofs_from_fixed_set(ndof::Int, fixed_dofs::Set{Int})
    n_fixed = count(d -> 1 <= d <= ndof, fixed_dofs)
    free_dofs = Vector{Int}(undef, max(ndof - n_fixed, 0))
    next_idx = 1
    @inbounds for dof in 1:ndof
        if !(dof in fixed_dofs)
            free_dofs[next_idx] = dof
            next_idx += 1
        end
    end
    return free_dofs
end

@inline function autospc_rotational_topology_enabled()
    return solver_env_bool("JFEM_AUTOSPC_ROT_TOPOLOGY", true)
end

@inline function autospc_rot_shell_only_multiplier()
    return max(solver_env_float("JFEM_AUTOSPC_ROT_SHELL_ONLY_MUL", 4.0), 0.0)
end

@inline function autospc_rot_rod_shell_multiplier()
    return max(solver_env_float("JFEM_AUTOSPC_ROT_ROD_SHELL_MUL", 4.0), 0.0)
end

@inline function autospc_rot_bar_shell_multiplier()
    return max(solver_env_float("JFEM_AUTOSPC_ROT_BAR_SHELL_MUL", 0.1), 0.0)
end

@inline function autospc_rot_bar_only_multiplier()
    return max(solver_env_float("JFEM_AUTOSPC_ROT_BAR_ONLY_MUL", 0.25), 0.0)
end

@inline function autospc_trans_relative_threshold()
    return max(solver_env_float("JFEM_AUTOSPC_TRANS_REL", 1e-8), 0.0)
end

@inline function autospc_rot_relative_threshold()
    return max(solver_env_float("JFEM_AUTOSPC_ROT_REL", autospc_trans_relative_threshold()), 0.0)
end

function _build_autospc_node_topology(model, id_map)
    n_nodes = length(id_map)
    node_has_shell = falses(n_nodes)
    node_has_bar = falses(n_nodes)
    node_has_rod = falses(n_nodes)

    for (_, el) in get(model, "CSHELLs", Dict())
        for nid in get(el, "NODES", Int[])
            idx = get(id_map, nid, 0)
            idx > 0 && (node_has_shell[idx] = true)
        end
    end

    for group_name in ("CBARs", "CBEAMs")
        for (_, el) in get(model, group_name, Dict())
            ga = get(id_map, get(el, "GA", 0), 0)
            gb = get(id_map, get(el, "GB", 0), 0)
            ga > 0 && (node_has_bar[ga] = true)
            gb > 0 && (node_has_bar[gb] = true)
        end
    end

    for group_name in ("CRODs", "CONRODs")
        for (_, el) in get(model, group_name, Dict())
            ga = get(id_map, get(el, "GA", 0), 0)
            gb = get(id_map, get(el, "GB", 0), 0)
            ga > 0 && (node_has_rod[ga] = true)
            gb > 0 && (node_has_rod[gb] = true)
        end
    end

    return node_has_shell, node_has_bar, node_has_rod
end

@inline function _autospc_rotational_topology_category(has_shell::Bool, has_bar::Bool, has_rod::Bool)
    # The tuning buckets intentionally collapse bar+rod+shell into :bar_shell
    # and bar+rod into :bar_only. That matches the heuristic sweep used to
    # calibrate NAPA_101 against the Nastran singularity table.
    if has_shell && has_bar
        return :bar_shell
    elseif has_shell && has_rod
        return :rod_shell
    elseif has_shell
        return :shell_only
    elseif has_bar
        return :bar_only
    end
    return :default
end

@inline function _autospc_rotational_topology_multiplier(category::Symbol, multipliers::Dict{String,Float64})
    if category === :shell_only
        return multipliers["shell_only"]
    elseif category === :rod_shell
        return multipliers["rod_shell"]
    elseif category === :bar_shell
        return multipliers["bar_shell"]
    elseif category === :bar_only
        return multipliers["bar_only"]
    end
    return 1.0
end

function _diagonal_autospc!(fixed_dofs::Set{Int}, spc_dofs::Union{Nothing,Set{Int}}, K, ndof, model, id_map)
    autospc_rel_trans = autospc_trans_relative_threshold()
    autospc_rel_rot = autospc_rot_relative_threshold()
    max_K_trans = maximum((abs(K[i, i]) for i in 1:ndof if mod(i - 1, 6) + 1 <= 3); init=1.0)
    max_K_rot = maximum((abs(K[i, i]) for i in 1:ndof if mod(i - 1, 6) + 1 > 3); init=1.0)

    topology_enabled = autospc_rotational_topology_enabled()
    node_has_shell, node_has_bar, node_has_rod = topology_enabled ?
        _build_autospc_node_topology(model, id_map) :
        (falses(length(id_map)), falses(length(id_map)), falses(length(id_map)))
    multipliers = Dict(
        "shell_only" => autospc_rot_shell_only_multiplier(),
        "rod_shell" => autospc_rot_rod_shell_multiplier(),
        "bar_shell" => autospc_rot_bar_shell_multiplier(),
        "bar_only" => autospc_rot_bar_only_multiplier(),
    )
    node_counts = Dict(
        "shell_only" => 0,
        "rod_shell" => 0,
        "bar_shell" => 0,
        "bar_only" => 0,
        "default" => 0,
    )
    for idx in 1:length(node_has_shell)
        category = topology_enabled ?
            _autospc_rotational_topology_category(node_has_shell[idx], node_has_bar[idx], node_has_rod[idx]) :
            :default
        node_counts[String(category)] += 1
    end

    n_autospc = 0
    n_autospc_trans = 0
    n_autospc_rot = 0
    rot_dofs_by_category = Dict(
        "shell_only" => 0,
        "rod_shell" => 0,
        "bar_shell" => 0,
        "bar_only" => 0,
        "default" => 0,
    )

    for i in 1:ndof
        dof_local = mod(i - 1, 6) + 1
        max_K_ref = dof_local <= 3 ? max_K_trans : max_K_rot
        rel_thresh = dof_local <= 3 ? autospc_rel_trans : autospc_rel_rot
        thresh = rel_thresh * max(max_K_ref, 1.0)
        category = :default
        if dof_local > 3 && topology_enabled
            node_idx = div(i - 1, 6) + 1
            if 1 <= node_idx <= length(node_has_shell)
                category = _autospc_rotational_topology_category(
                    node_has_shell[node_idx],
                    node_has_bar[node_idx],
                    node_has_rod[node_idx],
                )
                thresh *= _autospc_rotational_topology_multiplier(category, multipliers)
            end
        end
        if !(i in fixed_dofs) && (abs(K[i, i]) < thresh || K[i, i] < 0)
            push!(fixed_dofs, i)
            !isnothing(spc_dofs) && push!(spc_dofs, i)
            n_autospc += 1
            if dof_local <= 3
                n_autospc_trans += 1
            else
                n_autospc_rot += 1
                rot_dofs_by_category[String(category)] += 1
            end
        end
    end

    return Dict(
        "dofs" => n_autospc,
        "translational_dofs" => n_autospc_trans,
        "rotational_dofs" => n_autospc_rot,
        "rel_threshold" => autospc_rel_trans,
        "rel_threshold_trans" => autospc_rel_trans,
        "rel_threshold_rot" => autospc_rel_rot,
        "max_K_trans" => max_K_trans,
        "max_K_rot" => max_K_rot,
        "rotational_topology" => Dict(
            "enabled" => topology_enabled,
            "node_counts" => node_counts,
            "multipliers" => multipliers,
            "autospc_rotational_dofs_by_category" => rot_dofs_by_category,
        ),
    )
end

function factorization_autospc_free_dofs(K, ndof, fixed_dofs::Set{Int})
    diagnostics = Dict{String,Any}(
        "triggered" => false,
        "mechanism_dofs" => 0,
        "mechanism_translational_dofs" => 0,
        "mechanism_rotational_dofs" => 0,
        "shift_exponent" => nothing,
        "skipped_as_too_aggressive" => false,
    )

    free_dofs = _free_dofs_from_fixed_set(ndof, fixed_dofs)
    isempty(free_dofs) && return free_dofs, 0, diagnostics

    K_ff = K[free_dofs, free_dofs]
    try
        cholesky(Symmetric(K_ff))
        return free_dofs, 0, diagnostics
    catch
    end

    max_diag = maximum((abs(K[d, d]) for d in free_dofs); init=1.0)
    max_diag = max(max_diag, 1.0)
    n_mechanism_total = 0

    for shift_exp in (-12, -10, -8, -6)
        shift_val = max_diag * 10.0^shift_exp
        try
            diagnostics["triggered"] = true
            diagnostics["shift_exponent"] = shift_exp

            F_chol_probe = cholesky(Symmetric(K_ff); shift=shift_val)
            L_sparse = sparse(F_chol_probe.L)
            L_diag = abs.(diag(L_sparse))
            L_median = median(L_diag)
            pivot_threshold = min(sqrt(shift_val) * 3.0, L_median * 1e-4)
            small_pivot_mask = L_diag .< pivot_threshold
            n_mechanism = count(small_pivot_mask)

            if n_mechanism > length(free_dofs) * 0.5
                diagnostics["mechanism_dofs"] = n_mechanism
                diagnostics["skipped_as_too_aggressive"] = true
                log_msg("[BUCKLING] Factorization AUTOSPC (shift=1e$shift_exp): $n_mechanism DOFs (>50% of $(length(free_dofs)) free) - threshold too aggressive, skipping")
                break
            end

            if n_mechanism > 0
                perm = F_chol_probe.p
                mechanism_local = findall(small_pivot_mask)
                mechanism_original = perm[mechanism_local]
                mechanism_global = free_dofs[mechanism_original]
                n_mech_trans = count(d -> mod(d - 1, 6) + 1 <= 3, mechanism_global)
                n_mech_rot = n_mechanism - n_mech_trans
                diagnostics["mechanism_dofs"] = n_mechanism
                diagnostics["mechanism_translational_dofs"] = n_mech_trans
                diagnostics["mechanism_rotational_dofs"] = n_mech_rot
                log_msg("[BUCKLING] Factorization AUTOSPC (shift=1e$shift_exp): Found $n_mechanism DOFs ($n_mech_trans trans + $n_mech_rot rot)")
                for d in mechanism_global
                    push!(fixed_dofs, d)
                end
                n_mechanism_total = n_mechanism
                free_dofs = _free_dofs_from_fixed_set(ndof, fixed_dofs)
            end
            break
        catch
            continue
        end
    end

    return free_dofs, n_mechanism_total, diagnostics
end

# Compute free DOFs without solving (for eigenvalue problems that need the same BC partition).
function compute_free_dofs(K, ndof, model, id_map, spc_id, rbe3_map; return_diagnostics::Bool=false)
    fixed_dofs = Set{Int}()
    diagnostics = Dict{String,Any}(
        "mpc_dependent_dofs" => length(rbe3_map),
        "permanent_grid_dofs" => 0,
        "permanent_grid_constraints" => 0,
        "explicit_spc_dofs" => 0,
        "autospc_enabled" => model_autospc_enabled(model),
        "autospc_diagonal_dofs" => 0,
        "autospc_diagonal_translational_dofs" => 0,
        "autospc_diagonal_rotational_dofs" => 0,
        "autospc_rotational_topology" => Dict{String,Any}(),
        "factorization_autospc" => Dict{String,Any}(),
        "fixed_dofs" => 0,
        "free_dofs" => 0,
    )

    # MPC dependent DOFs
    for dep_dof in keys(rbe3_map)
        push!(fixed_dofs, dep_dof)
    end

    permanent_grid_dofs, permanent_grid_constraints = _apply_permanent_grid_constraints!(fixed_dofs, model, id_map)
    diagnostics["permanent_grid_dofs"] = permanent_grid_dofs
    diagnostics["permanent_grid_constraints"] = permanent_grid_constraints

    # SPC DOFs
    sets = Set{Int}()
    if !isnothing(spc_id)
        sid = Int(spc_id)
        if haskey(model["SPCADDs"], sid)
            union!(sets, model["SPCADDs"][sid])
        else
            push!(sets, sid)
        end
    end
    fixed_before_spc = length(fixed_dofs)
    for spc in model["SPC1s"]
        if Int(spc["SID"]) in sets
            for n in spc["NODES"]
                idx = get(id_map, n, 0)
                if idx > 0
                    for c in spc["C"]
                        push!(fixed_dofs, (idx - 1) * 6 + parse(Int, c))
                    end
                end
            end
        end
    end
    diagnostics["explicit_spc_dofs"] = max(length(fixed_dofs) - fixed_before_spc, 0)

    if model_autospc_enabled(model)
        autospc_diag = _diagonal_autospc!(fixed_dofs, nothing, K, ndof, model, id_map)
        diagnostics["autospc_diagonal_dofs"] = autospc_diag["dofs"]
        diagnostics["autospc_diagonal_translational_dofs"] = autospc_diag["translational_dofs"]
        diagnostics["autospc_diagonal_rotational_dofs"] = autospc_diag["rotational_dofs"]
        diagnostics["autospc_rotational_topology"] = autospc_diag["rotational_topology"]
    end

    free_dofs, n_fact_autospc, fact_diag = factorization_autospc_free_dofs(K, ndof, fixed_dofs)
    if n_fact_autospc > 0
        log_msg("[BUCKLING] Added $n_fact_autospc factorization AUTOSPC DOFs to stabilize eigen partition")
    end
    diagnostics["factorization_autospc"] = fact_diag
    diagnostics["fixed_dofs"] = length(fixed_dofs)
    diagnostics["free_dofs"] = length(free_dofs)

    if return_diagnostics
        return free_dofs, fixed_dofs, diagnostics
    end
    return free_dofs, fixed_dofs
end

function apply_bc_and_solve(K, ndof, model, id_map, F_applied, node_R, rbe3_map, max_elem_stiff, orig_diag;
                            linear_cache=nothing)
    log_msg("[SOLVER] Processing Boundary Conditions...")

    spc_id = get(model, "_spc_id", nothing)
    cache_enabled = linear_cache !== nothing && ndof >= linear_solve_cache_min_ndof()
    cache_key = cache_enabled ? _linear_solve_cache_key(K, ndof, model, spc_id, rbe3_map) : nothing
    cached_entry = (cache_enabled && cache_key !== nothing) ? get(linear_cache, cache_key, nothing) : nothing

    diagnostics = Dict{String,Any}(
        "bc_partition" => Dict{String,Any}(
            "mpc_dependent_dofs" => length(rbe3_map),
            "permanent_grid_dofs" => 0,
            "permanent_grid_constraints" => 0,
            "explicit_spc_dofs" => 0,
            "enforced_displacement_dofs" => 0,
            "autospc_enabled" => model_autospc_enabled(model),
            "autospc_diagonal_dofs" => 0,
            "autospc_diagonal_translational_dofs" => 0,
            "autospc_diagonal_rotational_dofs" => 0,
            "autospc_rotational_topology" => Dict{String,Any}(),
            "post_factorization_singular_dofs" => 0,
            "post_factorization_singular_translational_dofs" => 0,
            "post_factorization_singular_rotational_dofs" => 0,
            "fixed_dofs" => 0,
            "free_dofs" => 0,
        ),
        "linear_solver" => Dict{String,Any}(
            "backend" => "unknown",
            "strategy" => "unknown",
            "cache_hit" => false,
            "used_enforced_displacement_correction" => false,
            "used_lu_fallback" => false,
            "used_factorization_autospc" => false,
            "factorization_autospc" => Dict{String,Any}(
                "triggered" => false,
                "mechanism_dofs" => 0,
                "mechanism_translational_dofs" => 0,
                "mechanism_rotational_dofs" => 0,
                "shift_exponent" => nothing,
                "skipped_as_too_aggressive" => false,
            ),
            "force_norm" => 0.0,
            "force_max" => 0.0,
            "force_nonzero_dofs" => 0,
            "residual_norm" => 0.0,
            "relative_residual" => 0.0,
        ),
    )

    F_norm = norm(F_applied)
    F_max = mapreduce(abs, max, F_applied; init=0.0)
    n_nonzero = count(x -> abs(x) > 1e-10, F_applied)

    if cached_entry !== nothing
        diagnostics = deepcopy(cached_entry.diagnostics)
        diagnostics["linear_solver"]["cache_hit"] = true
        diagnostics["linear_solver"]["force_norm"] = F_norm
        diagnostics["linear_solver"]["force_max"] = F_max
        diagnostics["linear_solver"]["force_nonzero_dofs"] = n_nonzero
        log_msg("[SOLVER] Force vector: |F|=$(F_norm), max=$(F_max), nonzero DOFs=$n_nonzero")
        log_msg("[SOLVER] Reusing BC partition/factorization cache: Fixed DOFs=$(length(cached_entry.fixed_dofs)), Free DOFs=$(length(cached_entry.free_dofs))")

        F_ff = F_applied[cached_entry.free_dofs]
        if cached_entry.K_fs !== nothing
            F_ff = F_ff - cached_entry.K_fs * cached_entry.enforced_values
            log_msg("[SOLVER] Enforced displacement RHS correction applied ($(length(cached_entry.enforced_dofs)) DOFs)")
        end

        u_ff = cached_entry.factor \ F_ff
        r_solve = cached_entry.K_ff * u_ff - F_ff
        r_norm = norm(r_solve)
        rel_residual = r_norm / max(norm(F_ff), 1e-30)
        diagnostics["linear_solver"]["residual_norm"] = r_norm
        diagnostics["linear_solver"]["relative_residual"] = rel_residual
        log_msg("[SOLVER] Residual: |r|=$(r_norm), |r|/|F|=$rel_residual")

        log_msg("[SOLVER] Post-Processing...")
        u_global = zeros(ndof)
        u_global[cached_entry.free_dofs] = u_ff
        for (gdof, dval) in zip(cached_entry.enforced_dofs, cached_entry.enforced_values)
            u_global[gdof] = dval
        end

        n_rbe3_recovered = 0
        for (dep_dof, pairs) in rbe3_map
            u_avg = 0.0
            for (ind_dof, coeff) in pairs
                u_avg += coeff * u_global[ind_dof]
            end
            u_global[dep_dof] = u_avg
            n_rbe3_recovered += 1
        end
        if n_rbe3_recovered > 0
            log_msg("[SOLVER] RBE3: Recovered $n_rbe3_recovered dependent DOFs")
        end

        return u_global, copy(cached_entry.fixed_dofs), copy(cached_entry.spc_dofs), diagnostics
    end

    fixed_dofs = Set{Int}()
    spc_dofs = Set{Int}()  # True SPC DOFs only (SPC1 + AUTOSPC), excludes MPC-dependent
    enforced_disp = Dict{Int,Float64}()  # global_dof => enforced value (non-zero)

    # Fix MPC dependent DOFs (RBE2/RBE3/RBE1/RSPLINE/MPC)
    for dep_dof in keys(rbe3_map)
        push!(fixed_dofs, dep_dof)
    end
    if !isempty(rbe3_map)
        log_msg("[SOLVER] MPC: Fixed $(length(rbe3_map)) dependent DOFs")
    end

    permanent_grid_dofs, permanent_grid_constraints = _apply_permanent_grid_constraints!(fixed_dofs, model, id_map)
    diagnostics["bc_partition"]["permanent_grid_dofs"] = permanent_grid_dofs
    diagnostics["bc_partition"]["permanent_grid_constraints"] = permanent_grid_constraints
    if permanent_grid_dofs > 0
        log_msg("[SOLVER] Permanent GRID/GRDSET constraints: $permanent_grid_dofs DOFs across $permanent_grid_constraints grid(s)")
    end

    sets = Set{Int}()
    spc_id = get(model, "_spc_id", nothing)
    if !isnothing(spc_id)
        sid = Int(spc_id)
        if haskey(model["SPCADDs"], sid)
            union!(sets, model["SPCADDs"][sid])
        else
            push!(sets, sid)
        end
    end
    fixed_before_spc = length(fixed_dofs)
    for spc in model["SPC1s"]
        if Int(spc["SID"]) in sets
            d_val = Float64(get(spc, "D", 0.0))
            for n in spc["NODES"]
                idx = get(id_map, n, 0)
                if idx > 0
                    for c in spc["C"]
                        gdof = (idx - 1) * 6 + parse(Int, c)
                        push!(fixed_dofs, gdof)
                        push!(spc_dofs, gdof)
                        if abs(d_val) > 0.0
                            enforced_disp[gdof] = d_val
                        end
                    end
                end
            end
        end
    end
    log_msg("[SOLVER] SPC: $(length(spc_dofs)) constrained DOFs from SPC1 cards")
    diagnostics["bc_partition"]["explicit_spc_dofs"] = max(length(fixed_dofs) - fixed_before_spc, 0)
    if !isempty(enforced_disp)
        log_msg("[SOLVER] Enforced displacements: $(length(enforced_disp)) DOFs")
    end
    diagnostics["bc_partition"]["enforced_displacement_dofs"] = length(enforced_disp)
    diagnostics["linear_solver"]["used_enforced_displacement_correction"] = !isempty(enforced_disp)

    if model_autospc_enabled(model)
        autospc_diag = _diagonal_autospc!(fixed_dofs, spc_dofs, K, ndof, model, id_map)
        diagnostics["bc_partition"]["autospc_diagonal_dofs"] = autospc_diag["dofs"]
        diagnostics["bc_partition"]["autospc_diagonal_translational_dofs"] = autospc_diag["translational_dofs"]
        diagnostics["bc_partition"]["autospc_diagonal_rotational_dofs"] = autospc_diag["rotational_dofs"]
        diagnostics["bc_partition"]["autospc_rotational_topology"] = autospc_diag["rotational_topology"]
        rel_trans = autospc_diag["rel_threshold_trans"]
        rel_rot = autospc_diag["rel_threshold_rot"]
        rel_msg = rel_trans == rel_rot ? string(rel_trans) : "trans=$(rel_trans), rot=$(rel_rot)"
        log_msg("[SOLVER] AUTOSPC: $(autospc_diag["dofs"]) DOFs ($(autospc_diag["translational_dofs"]) trans + $(autospc_diag["rotational_dofs"]) rot, rel_thresh=$rel_msg, max_K_trans=$(round(autospc_diag["max_K_trans"], sigdigits=3)), max_K_rot=$(round(autospc_diag["max_K_rot"], sigdigits=3)))")
        if get(autospc_diag["rotational_topology"], "enabled", false)
            topology = autospc_diag["rotational_topology"]
            multipliers = topology["multipliers"]
            node_counts = topology["node_counts"]
            log_msg("[SOLVER] AUTOSPC rotational topology: shell_only=$(multipliers["shell_only"]) ($(node_counts["shell_only"]) nodes), rod_shell=$(multipliers["rod_shell"]) ($(node_counts["rod_shell"]) nodes), bar_shell=$(multipliers["bar_shell"]) ($(node_counts["bar_shell"]) nodes), bar_only=$(multipliers["bar_only"]) ($(node_counts["bar_only"]) nodes)")
        end
    else
        log_msg("[SOLVER] AUTOSPC: disabled by PARAM,AUTOSPC")
    end

    F_norm = norm(F_applied)
    F_max = mapreduce(abs, max, F_applied; init=0.0)
    n_nonzero = count(x -> abs(x) > 1e-10, F_applied)
    diagnostics["linear_solver"]["force_norm"] = F_norm
    diagnostics["linear_solver"]["force_max"] = F_max
    diagnostics["linear_solver"]["force_nonzero_dofs"] = n_nonzero
    log_msg("[SOLVER] Force vector: |F|=$(F_norm), max=$(F_max), nonzero DOFs=$n_nonzero")
    log_msg("[SOLVER] Slicing Matrix (Reducing System)...")
    free_dofs = _free_dofs_from_fixed_set(ndof, fixed_dofs)
    diagnostics["bc_partition"]["fixed_dofs"] = length(fixed_dofs)
    diagnostics["bc_partition"]["free_dofs"] = length(free_dofs)
    log_msg("[SOLVER] Fixed DOFs: $(length(fixed_dofs)), Free DOFs: $(length(free_dofs))")

    K_ff = K[free_dofs, free_dofs]
    F_ff = F_applied[free_dofs]
    enforced_dofs = Int[]
    enforced_values = Float64[]
    K_fs = nothing

    # Enforced displacement correction: F_ff -= K_fs * u_s
    if !isempty(enforced_disp)
        enforced_dofs = sort(collect(keys(enforced_disp)))
        enforced_values = [enforced_disp[d] for d in enforced_dofs]
        K_fs = K[free_dofs, enforced_dofs]
        F_ff = F_ff - K_fs * enforced_values
        log_msg("[SOLVER] Enforced displacement RHS correction applied ($(length(enforced_dofs)) DOFs)")
    end

    n_free = length(free_dofs)
    solve_factor = nothing
    if n_free <= 2000000
        diagnostics["linear_solver"]["strategy"] = "direct"
        log_msg("[SOLVER] Using Direct Solver (Cholesky) for $n_free DOFs...")
        u_ff = try
            F_chol = cholesky(Symmetric(K_ff))

            L_sparse = sparse(F_chol.L)
            L_diag = abs.(diag(L_sparse))
            K_diag = [abs(K_ff[i, i]) for i in 1:n_free]
            pivot_ratios = zeros(n_free)
            for i in 1:n_free
                if K_diag[i] > 1e-30
                    pivot_ratios[i] = L_diag[i]^2 / K_diag[i]
                else
                    pivot_ratios[i] = 1.0
                end
            end
            perm = F_chol.p
            sing_threshold = 1e-7
            singular_local = findall(pivot_ratios .< sing_threshold)
            n_sing = length(singular_local)
            if n_sing > 0
                singular_original = perm[singular_local]
                singular_global = free_dofs[singular_original]
                n_sing_trans = count(d -> mod(d - 1, 6) + 1 <= 3, singular_global)
                n_sing_rot = n_sing - n_sing_trans
                diagnostics["bc_partition"]["post_factorization_singular_dofs"] = n_sing
                diagnostics["bc_partition"]["post_factorization_singular_translational_dofs"] = n_sing_trans
                diagnostics["bc_partition"]["post_factorization_singular_rotational_dofs"] = n_sing_rot
                log_msg("[SOLVER] Post-factorization singularity: $n_sing DOFs ($n_sing_trans trans + $n_sing_rot rot, threshold=$sing_threshold)")

                for d in singular_global
                    push!(fixed_dofs, d)
                end
                free_dofs = _free_dofs_from_fixed_set(ndof, fixed_dofs)
                K_ff = K[free_dofs, free_dofs]
                F_ff = F_applied[free_dofs]
                if !isempty(enforced_disp)
                    K_fs = K[free_dofs, enforced_dofs]
                    F_ff = F_ff - K_fs * enforced_values
                end
                n_free = length(free_dofs)
                diagnostics["bc_partition"]["fixed_dofs"] = length(fixed_dofs)
                diagnostics["bc_partition"]["free_dofs"] = n_free
                log_msg("[SOLVER] Re-solving with $(length(fixed_dofs)) fixed, $n_free free DOFs")
                F_chol = cholesky(Symmetric(K_ff))
            end

            diagnostics["linear_solver"]["backend"] = "direct_cholesky"
            solve_factor = F_chol
            F_chol \ F_ff
        catch e
            log_msg("[SOLVER] Cholesky failed: $(typeof(e)). Running factorization AUTOSPC...")

            local u_result
            mechanism_found = false

            # Try shifted Cholesky with progressively larger shifts (limited range to avoid false positives)
            for shift_exp in [-12, -10, -8, -6]
                shift_val = max(max_elem_stiff, 1.0) * 10.0^shift_exp
                try
                    diagnostics["linear_solver"]["used_factorization_autospc"] = true
                    diagnostics["linear_solver"]["factorization_autospc"]["triggered"] = true
                    diagnostics["linear_solver"]["factorization_autospc"]["shift_exponent"] = shift_exp

                    F_chol_probe = cholesky(Symmetric(K_ff); shift=shift_val)
                    L_sparse = sparse(F_chol_probe.L)
                    L_diag = abs.(diag(L_sparse))
                    L_median = median(L_diag)
                    # Use ratio-based threshold: mechanisms have L[i] close to sqrt(shift),
                    # regular DOFs have L[i] much larger than sqrt(shift).
                    pivot_threshold = min(sqrt(shift_val) * 3.0, L_median * 1e-4)
                    small_pivot_mask = L_diag .< pivot_threshold
                    n_mechanism = count(small_pivot_mask)

                    # Sanity check: if >50% of DOFs flagged, threshold is too aggressive for this shift
                    if n_mechanism > n_free * 0.5
                        diagnostics["linear_solver"]["factorization_autospc"]["mechanism_dofs"] = n_mechanism
                        diagnostics["linear_solver"]["factorization_autospc"]["skipped_as_too_aggressive"] = true
                        log_msg("[SOLVER] Factorization AUTOSPC (shift=1e$shift_exp): $n_mechanism DOFs (>50% of $n_free free) - threshold too aggressive, skipping")
                        mechanism_found = true
                        break
                    end

                    if n_mechanism > 0
                        perm = F_chol_probe.p
                        mechanism_local = findall(small_pivot_mask)
                        mechanism_original = perm[mechanism_local]
                        mechanism_global = free_dofs[mechanism_original]
                        n_mech_trans = count(d -> mod(d - 1, 6) + 1 <= 3, mechanism_global)
                        n_mech_rot = n_mechanism - n_mech_trans
                        diagnostics["linear_solver"]["factorization_autospc"]["mechanism_dofs"] = n_mechanism
                        diagnostics["linear_solver"]["factorization_autospc"]["mechanism_translational_dofs"] = n_mech_trans
                        diagnostics["linear_solver"]["factorization_autospc"]["mechanism_rotational_dofs"] = n_mech_rot
                        log_msg("[SOLVER] Factorization AUTOSPC (shift=1e$shift_exp): Found $n_mechanism DOFs ($n_mech_trans trans + $n_mech_rot rot)")

                        for d in mechanism_global
                            push!(fixed_dofs, d)
                            push!(spc_dofs, d)
                        end
                        free_dofs = _free_dofs_from_fixed_set(ndof, fixed_dofs)
                        K_ff = K[free_dofs, free_dofs]
                        F_ff = F_applied[free_dofs]
                        if !isempty(enforced_disp)
                            K_fs = K[free_dofs, enforced_dofs]
                            F_ff = F_ff - K_fs * enforced_values
                        end
                        n_free = length(free_dofs)
                        diagnostics["bc_partition"]["fixed_dofs"] = length(fixed_dofs)
                        diagnostics["bc_partition"]["free_dofs"] = n_free
                        log_msg("[SOLVER] Rebuilt system: Fixed=$(length(fixed_dofs)), Free=$n_free")
                    end

                    mechanism_found = true
                    break
                catch
                    continue
                end
            end

            if mechanism_found
                # Try clean Cholesky on reduced system
                try
                    F_chol_clean = cholesky(Symmetric(K_ff))
                    solve_factor = F_chol_clean
                    u_result = F_chol_clean \ F_ff
                    diagnostics["linear_solver"]["backend"] = "direct_cholesky_after_factorization_autospc"
                    log_msg("[SOLVER] Clean Cholesky succeeded after factorization AUTOSPC")
                catch e2
                    log_msg("[SOLVER] Clean Cholesky still failed: $(typeof(e2)). Using LU factorization...")
                    F_lu = lu(K_ff)
                    solve_factor = F_lu
                    u_result = F_lu \ F_ff
                    diagnostics["linear_solver"]["backend"] = "direct_lu_after_factorization_autospc"
                    diagnostics["linear_solver"]["used_lu_fallback"] = true
                end
            else
                log_msg("[SOLVER] All shifted Cholesky attempts failed. Using LU factorization directly...")
                F_lu = lu(K_ff)
                solve_factor = F_lu
                u_result = F_lu \ F_ff
                diagnostics["linear_solver"]["backend"] = "direct_lu"
                diagnostics["linear_solver"]["used_lu_fallback"] = true
            end
            u_result
        end
    else
        # Ensure perfect symmetry for iterative solver
        K_sym = Symmetric(K_ff)
        diagnostics["linear_solver"]["strategy"] = "iterative"
        log_msg("[SOLVER] Computing Preconditioner (Smoothed Aggregation)...")
        ml = smoothed_aggregation(K_sym)
        P = aspreconditioner(ml)
        log_msg("[SOLVER] Solving Linear System (CG + AMG)...")
        u_ff = try
            diagnostics["linear_solver"]["backend"] = "iterative_cg_amg"
            cg(K_sym, F_ff; reltol=1e-8, maxiter=5000, Pl=P)
        catch e
            log_msg("[SOLVER] CG Failed ($e). Trying MINRES...")
            diagnostics["linear_solver"]["backend"] = "iterative_minres"
            minres(K_sym, F_ff; reltol=1e-8, maxiter=5000)
        end
    end

    # Report residual
    r_solve = K_ff * u_ff - F_ff
    r_norm = norm(r_solve)
    rel_residual = r_norm / max(norm(F_ff), 1e-30)
    diagnostics["linear_solver"]["residual_norm"] = r_norm
    diagnostics["linear_solver"]["relative_residual"] = rel_residual
    log_msg("[SOLVER] Residual: |r|=$(r_norm), |r|/|F|=$rel_residual")

    if cache_enabled && cache_key !== nothing && solve_factor !== nothing &&
       diagnostics["linear_solver"]["strategy"] == "direct"
        linear_cache[cache_key] = LinearSolveCacheEntry(
            copy(free_dofs),
            copy(fixed_dofs),
            copy(spc_dofs),
            copy(enforced_dofs),
            copy(enforced_values),
            K_ff,
            K_fs,
            solve_factor,
            deepcopy(diagnostics),
        )
    end

    log_msg("[SOLVER] Post-Processing...")
    u_global = zeros(ndof)
    u_global[free_dofs] = u_ff

    # Apply enforced displacement values
    for (gdof, dval) in zip(enforced_dofs, enforced_values)
        u_global[gdof] = dval
    end

    # RBE3 displacement recovery
    n_rbe3_recovered = 0
    for (dep_dof, pairs) in rbe3_map
        u_avg = 0.0
        for (ind_dof, coeff) in pairs
            u_avg += coeff * u_global[ind_dof]
        end
        u_global[dep_dof] = u_avg
        n_rbe3_recovered += 1
    end
    if n_rbe3_recovered > 0
        log_msg("[SOLVER] RBE3: Recovered $n_rbe3_recovered dependent DOFs")
    end

    return u_global, fixed_dofs, spc_dofs, diagnostics
end
