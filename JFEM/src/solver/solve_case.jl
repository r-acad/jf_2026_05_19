# solve_case.jl — Subcase solver orchestrator

function _assemble_applied_force(ndof, model, id_map, X, load_id, node_R, rbe3_map;
                                 load_scale::Float64=1.0,
                                 temp_load_id=nothing,
                                 log_rbe3::Bool=true)
    F_applied = zeros(ndof)
    if !isnothing(load_id)
        elem_map = Dict{Int, Any}()
        F_global_accum = zeros(ndof)
        had_disable = haskey(model, "_disable_thermal_in_resolve_loads")
        prev_disable = had_disable ? model["_disable_thermal_in_resolve_loads"] : nothing
        model["_disable_thermal_in_resolve_loads"] = true
        try
            resolve_loads(model, Int(load_id), load_scale, id_map, elem_map, X, F_global_accum)
        finally
            if had_disable
                model["_disable_thermal_in_resolve_loads"] = prev_disable
            else
                delete!(model, "_disable_thermal_in_resolve_loads")
            end
        end
        resolve_thermal_loads(model, temp_load_id, load_scale, id_map, elem_map, X, F_global_accum; node_R=node_R)
        _rotate_global_force_to_analysis!(F_applied, F_global_accum, node_R, length(id_map))
    elseif !isnothing(temp_load_id)
        elem_map = Dict{Int, Any}()
        F_global_accum = zeros(ndof)
        resolve_thermal_loads(model, temp_load_id, load_scale, id_map, elem_map, X, F_global_accum; node_R=node_R)
        _rotate_global_force_to_analysis!(F_applied, F_global_accum, node_R, length(id_map))
    end

    if !isempty(rbe3_map)
        n_force_redist = 0
        for (dep_dof, pairs) in rbe3_map
            f_dep = F_applied[dep_dof]
            if abs(f_dep) > 1e-30
                for (ind_dof, coeff) in pairs
                    F_applied[ind_dof] += f_dep * coeff
                end
                F_applied[dep_dof] = 0.0
                n_force_redist += 1
            end
        end
        if n_force_redist > 0 && log_rbe3
            log_msg("[SOLVER] RBE3: Redistributed forces from $n_force_redist dependent DOFs")
        end
    end
    return F_applied
end

function _rotate_global_force_to_analysis!(F_applied, F_global_accum, node_R, n_nodes::Int)
    @inbounds for i in 1:n_nodes
        idx = (i - 1) * 6
        R = node_R[i]

        f1 = F_global_accum[idx + 1]
        f2 = F_global_accum[idx + 2]
        f3 = F_global_accum[idx + 3]
        F_applied[idx + 1] = R[1, 1] * f1 + R[2, 1] * f2 + R[3, 1] * f3
        F_applied[idx + 2] = R[1, 2] * f1 + R[2, 2] * f2 + R[3, 2] * f3
        F_applied[idx + 3] = R[1, 3] * f1 + R[2, 3] * f2 + R[3, 3] * f3

        m1 = F_global_accum[idx + 4]
        m2 = F_global_accum[idx + 5]
        m3 = F_global_accum[idx + 6]
        F_applied[idx + 4] = R[1, 1] * m1 + R[2, 1] * m2 + R[3, 1] * m3
        F_applied[idx + 5] = R[1, 2] * m1 + R[2, 2] * m2 + R[3, 2] * m3
        F_applied[idx + 6] = R[1, 3] * m1 + R[2, 3] * m2 + R[3, 3] * m3
    end
    return F_applied
end

@inline function _matrix_column_max_abs(A::AbstractMatrix, col::Int)
    max_val = 0.0
    @inbounds for row in axes(A, 1)
        v = abs(A[row, col])
        if v > max_val
            max_val = v
        end
    end
    return max_val
end

function _scale_matrix_column!(A::AbstractMatrix, col::Int, scale::Float64)
    @inbounds for row in axes(A, 1)
        A[row, col] *= scale
    end
    return A
end

function _deterministic_buckling_start_vector(n::Int, ordinal::Int)
    seed = buckling_rng_seed()
    v = Vector{Float64}(undef, n)
    phase = 0.137 * (seed + 1) + 0.731 * ordinal
    @inbounds for i in 1:n
        x = i + phase
        v[i] = sin(0.7548776662466927 * x) + 0.5 * cos(1.246979603717467 * x)
    end
    return v
end

function _buckling_raw_csv_cell(x)
    replace(string(x), "," => ";", "\n" => " ", "\r" => " ")
end

function _write_buckling_raw_eigen_csv(eigenvalues::AbstractVector, path::AbstractString;
                                       buckling_subcase=nothing,
                                       static_subcase=nothing,
                                       backend="",
                                       phase="pre_positive_filter",
                                       requested_modes::Int=0,
                                       requested_modes_internal::Int=0,
                                       eigrl_v1::Float64=0.0,
                                       eigrl_v2::Float64=0.0,
                                       positive_tol::Float64=1e-10)
    isempty(strip(path)) && return
    try
        dir = dirname(path)
        if !isempty(dir) && dir != "." && !isdir(dir)
            mkpath(dir)
        end
        vals = collect(Float64, eigenvalues)
        abs_rank = Dict{Int,Int}()
        for (rank, idx) in enumerate(sortperm(abs.(vals)))
            abs_rank[idx] = rank
        end
        has_range = (eigrl_v1 != 0.0 || eigrl_v2 != 0.0) && eigrl_v2 > eigrl_v1
        range_abs_tol = max(solver_env_float("JFEM_SOL105_RANGE_ABS_TOL", 0.0), 0.0)
        range_rel_tol = max(solver_env_float("JFEM_SOL105_RANGE_REL_TOL", 0.0), 0.0)
        v1_positive_eff = max(eigrl_v1, positive_tol)
        v2_eff = has_range ? eigrl_v2 + max(range_abs_tol, abs(eigrl_v2) * range_rel_tol) : eigrl_v2
        write_header = !isfile(path) || filesize(path) == 0
        open(path, write_header ? "w" : "a") do io
            if write_header
                println(io, "phase,buckling_subcase,static_subcase,backend,raw_index,abs_rank,lambda,abs_lambda,is_positive,in_signed_eigrl_range,in_positive_eigrl_range,eigrl_v1,eigrl_v2,eigrl_v2_eff,requested_modes,requested_modes_internal")
            end
            for (i, lam) in enumerate(vals)
                in_signed_range = !has_range || (lam >= eigrl_v1 && lam <= v2_eff)
                in_positive_range = lam > positive_tol && (!has_range || (lam >= v1_positive_eff && lam <= v2_eff))
                println(io,
                    _buckling_raw_csv_cell(phase), ",",
                    _buckling_raw_csv_cell(buckling_subcase === nothing ? "" : buckling_subcase), ",",
                    _buckling_raw_csv_cell(static_subcase === nothing ? "" : static_subcase), ",",
                    _buckling_raw_csv_cell(backend), ",",
                    i, ",", abs_rank[i], ",", lam, ",", abs(lam), ",",
                    lam > positive_tol, ",", in_signed_range, ",", in_positive_range, ",",
                    eigrl_v1, ",", eigrl_v2, ",", v2_eff, ",",
                    requested_modes, ",", requested_modes_internal)
            end
        end
        log_msg("[BUCKLING] Raw signed eigenvalue trace: $(length(vals)) rows -> $path")
    catch e
        log_msg("[BUCKLING] WARNING: failed to write JFEM_BUCKLING_RAW_EIGEN_CSV -> $path: $(sprint(showerror, e))")
    end
    return
end

function _build_results_from_state(ndof, model, id_map, X, node_R, u_global, residual_vector,
                                   snorm_normals, solver_diagnostics)
    results_json = Dict(
        "displacements" => [],
        "spc_forces" => [],
        "forces" => Dict("cbar" => [], "quad4" => [], "tria3" => [], "crod" => [], "conrod" => [], "celas1" => []),
        "forces_bilin" => Dict("quad4" => [], "tria3" => []),
        "stresses" => Dict("cbar" => [], "quad4" => [], "tria3" => [], "crod" => [], "conrod" => [], "celas1" => [], "ctetra" => [], "chexa" => [], "cpenta" => []),
        "strains" => Dict("cbar" => [], "quad4" => [], "tria3" => [], "crod" => [], "conrod" => [], "celas1" => [], "ctetra" => [], "chexa" => [], "cpenta" => []),
        "solver_diagnostics" => solver_diagnostics,
    )

    u_out = zeros(ndof)
    sorted_nodes = sort(collect(keys(id_map)))
    for nid in sorted_nodes
        idx = id_map[nid]
        base = (idx - 1) * 6
        u_loc = view(u_global, base+1:base+6)

        t_glob = node_R[idx] * u_loc[1:3]
        r_glob = node_R[idx] * u_loc[4:6]

        u_out[base+1:base+3] = t_glob
        u_out[base+4:base+6] = r_glob

        push!(results_json["displacements"], Dict(
            "grid_id" => nid,
            "t1" => t_glob[1], "t2" => t_glob[2], "t3" => t_glob[3],
            "r1" => r_glob[1], "r2" => r_glob[2], "r3" => r_glob[3],
        ))

        r_loc = view(residual_vector, base+1:base+6)
        r_reac_glob = vcat(node_R[idx] * r_loc[1:3], node_R[idx] * r_loc[4:6])
        if norm(r_reac_glob) > 1e-20
            push!(results_json["spc_forces"], Dict(
                "grid_id" => nid,
                "t1" => r_reac_glob[1], "t2" => r_reac_glob[2], "t3" => r_reac_glob[3],
                "r1" => r_reac_glob[4], "r2" => r_reac_glob[5], "r3" => r_reac_glob[6],
            ))
        end
    end

    spc_fx = isempty(results_json["spc_forces"]) ? 0.0 : sum(s["t1"] for s in results_json["spc_forces"])
    spc_fy = isempty(results_json["spc_forces"]) ? 0.0 : sum(s["t2"] for s in results_json["spc_forces"])
    spc_fz = isempty(results_json["spc_forces"]) ? 0.0 : sum(s["t3"] for s in results_json["spc_forces"])
    solver_diagnostics["equilibrium"] = Dict(
        "spc_reaction_sum" => Dict("fx" => spc_fx, "fy" => spc_fy, "fz" => spc_fz),
        "residual" => sqrt(spc_fx^2 + spc_fy^2 + spc_fz^2),
        "relative_residual" => 0.0,
    )

    stresses = Dict{Int, Float64}()
    recover_shell_stresses!(model, id_map, X, node_R, u_global, snorm_normals, stresses, results_json)
    recover_bar_stresses!(model, id_map, X, node_R, u_global, stresses, results_json)
    recover_rod_stresses!(model, id_map, X, node_R, u_global, stresses, results_json)
    recover_spring_forces!(model, id_map, u_global, stresses, results_json)
    recover_solid_stresses!(model, id_map, X, node_R, u_global, stresses, results_json)

    return u_out, stresses, results_json
end

@inline function _quad4_is_coplanar(p1::SVector{3,Float64}, p2::SVector{3,Float64},
                                    p3::SVector{3,Float64}, p4::SVector{3,Float64};
                                    tol::Float64=1e-8)
    n = cross(p2 - p1, p3 - p1)
    nn = norm(n)
    nn <= 1e-12 && return false
    return abs(dot(p4 - p1, n / nn)) <= tol * max(norm(p2-p1), norm(p3-p1), norm(p4-p1), 1.0)
end

function _formal_von_karman_shell_supported(model, id_map, X)
    return _formal_von_karman_shell_support_report(model, id_map, X).supported
end

function _formal_von_karman_shell_support_report(model, id_map, X)
    has_shell = false
    total_shell_count = 0
    quad4_count = 0
    tria3_count = 0
    planar_quad_count = 0
    warped_quad_count = 0
    for (_, el) in get(model, "CSHELLs", Dict())
        nids = get(el, "NODES", Int[])
        n = length(nids)
        n in (3, 4) || return (
            supported=false,
            support_class="unsupported",
            unsupported_reason="unsupported_shell_node_count_$n",
            total_shell_count=total_shell_count,
            quad4_count=quad4_count,
            tria3_count=tria3_count,
            planar_quad_count=planar_quad_count,
            warped_quad_count=warped_quad_count,
        )
        any(nid -> !haskey(id_map, nid), nids) && return (
            supported=false,
            support_class="unsupported",
            unsupported_reason="missing_grid_reference",
            total_shell_count=total_shell_count,
            quad4_count=quad4_count,
            tria3_count=tria3_count,
            planar_quad_count=planar_quad_count,
            warped_quad_count=warped_quad_count,
        )
        pid = string(get(el, "PID", 0))
        haskey(model["PSHELLs"], pid) || return (
            supported=false,
            support_class="unsupported",
            unsupported_reason="missing_pshell_property",
            total_shell_count=total_shell_count,
            quad4_count=quad4_count,
            tria3_count=tria3_count,
            planar_quad_count=planar_quad_count,
            warped_quad_count=warped_quad_count,
        )
        prop = model["PSHELLs"][pid]
        if get(prop, "TYPE", "") == "PCOMP_CLT"
            haskey(prop, "Cm") || return (
                supported=false,
                support_class="unsupported",
                unsupported_reason="missing_pcomp_clt_membrane_matrix",
                total_shell_count=total_shell_count,
                quad4_count=quad4_count,
                tria3_count=tria3_count,
                planar_quad_count=planar_quad_count,
                warped_quad_count=warped_quad_count,
            )
        else
            mid = string(get(prop, "MID", 0))
            haskey(model["MATs"], mid) || return (
                supported=false,
                support_class="unsupported",
                unsupported_reason="missing_material",
                total_shell_count=total_shell_count,
                quad4_count=quad4_count,
                tria3_count=tria3_count,
                planar_quad_count=planar_quad_count,
                warped_quad_count=warped_quad_count,
            )
        end
        total_shell_count += 1
        if n == 4
            quad4_count += 1
            i1, i2, i3, i4 = id_map[nids[1]], id_map[nids[2]], id_map[nids[3]], id_map[nids[4]]
            p1 = SVector{3}(X[i1,1], X[i1,2], X[i1,3])
            p2 = SVector{3}(X[i2,1], X[i2,2], X[i2,3])
            p3 = SVector{3}(X[i3,1], X[i3,2], X[i3,3])
            p4 = SVector{3}(X[i4,1], X[i4,2], X[i4,3])
            if _quad4_is_coplanar(p1, p2, p3, p4)
                planar_quad_count += 1
            else
                warped_quad_count += 1
            end
        else
            tria3_count += 1
        end
        has_shell = true
    end
    has_shell || return (
        supported=false,
        support_class="unsupported",
        unsupported_reason="no_shell_elements",
        total_shell_count=0,
        quad4_count=0,
        tria3_count=0,
        planar_quad_count=0,
        warped_quad_count=0,
    )
    return (
        supported=true,
        support_class=warped_quad_count > 0 ? "warped_fallback" : "planar",
        unsupported_reason="",
        total_shell_count=total_shell_count,
        quad4_count=quad4_count,
        tria3_count=tria3_count,
        planar_quad_count=planar_quad_count,
        warped_quad_count=warped_quad_count,
    )
end

function _formal_shell_membrane_constitutive(prop, el, mat, v1, v2, v3, p1, p2; tri::Bool=false)
    if get(prop, "TYPE", "") == "PCOMP_CLT" && haskey(prop, "Cm")
        Cm = copy(prop["Cm"])
        beta =
            if tri
                shell_pcomp_material_rotation(
                    q4_pcomp_axis_mode("JFEM_Q4_PCOMP_AXIS_MODE_STATIC"),
                    v1, v2, v3, p1, p2,
                    deg2rad(Float64(get(el, "THETA", 0.0))),
                    Int(get(el, "MCID", 0)),
                    model["CORDs"],
                )
            else
                theta_rad = deg2rad(Float64(get(el, "THETA", 0.0)))
                shell_pcomp_material_rotation(
                    q4_pcomp_axis_mode("JFEM_Q4_PCOMP_AXIS_MODE_STATIC"),
                    v1, v2, v3, p1, p2, theta_rad,
                    Int(get(el, "MCID", 0)),
                    model["CORDs"],
                )
            end
        if abs(beta) > 1e-10
            cb = cos(beta); sb = sin(beta)
            c2 = cb^2; s2 = sb^2; cs = cb * sb
            _rotate_constitutive_3x3!(Cm, c2, s2, cs, s2, c2, -cs, -2cs, 2cs, c2-s2)
        end
        return Cm
    end

    h = Float64(get(prop, "T", 0.0))
    E = Float64(get(mat, "E", 0.0))
    nu = Float64(get(mat, "NU", 0.0))
    const_mem = E / max(1 - nu^2, 1e-12)
    return (const_mem .* [1 nu 0; nu 1 0; 0 0 (1-nu)/2]) * h
end

function _nonlinear_internal_force(K_linear, Kg, u_state; residual_model::Symbol=:tangent_operator)
    linear_force = K_linear * u_state
    geometric_force = Kg * u_state
    internal_force =
        if residual_model === :secant_geometric
            linear_force .+ 0.5 .* geometric_force
        else
            linear_force .+ geometric_force
        end
    return internal_force, linear_force, geometric_force
end

function _nonlinear_residual_metrics(K_linear, Kg, K_eff, F_applied, ndof, model, id_map, spc_id, rbe3_map, u_state;
                                     residual_model::Symbol=:tangent_operator)
    free_dofs, fixed_dofs, bc_diagnostics = compute_free_dofs(
        K_eff, ndof, model, id_map, spc_id, rbe3_map; return_diagnostics=true)
    internal_force, linear_force, geometric_force = _nonlinear_internal_force(
        K_linear, Kg, u_state; residual_model=residual_model)
    residual = internal_force - F_applied
    active_dofs = isempty(free_dofs) ? collect(1:ndof) : free_dofs
    residual_norm = norm(residual[active_dofs])
    internal_force_norm = norm(internal_force[active_dofs])
    linear_force_norm = norm(linear_force[active_dofs])
    geometric_force_norm = norm(geometric_force[active_dofs])
    force_norm = norm(F_applied[active_dofs])
    relative_residual = residual_norm / max(force_norm, 1e-30)
    return (
        residual_vector=residual,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        internal_force_norm=internal_force_norm,
        linear_force_norm=linear_force_norm,
        geometric_force_norm=geometric_force_norm,
        free_dofs=free_dofs,
        fixed_dofs=fixed_dofs,
        bc_diagnostics=bc_diagnostics,
    )
end

function _evaluate_nonlinear_state(K_linear, F_applied, ndof, model, id_map, X, spc_id, node_R, u_state,
                                   snorm_normals, rbe3_map;
                                   residual_model::Symbol=:tangent_operator)
    Kg = assemble_geometric_stiffness(model, id_map, X, node_R, ndof, u_state, snorm_normals, rbe3_map)
    K_eff = K_linear + Kg
    residual = _nonlinear_residual_metrics(
        K_linear, Kg, K_eff, F_applied, ndof, model, id_map, spc_id, rbe3_map, u_state;
        residual_model=residual_model)
    return (
        Kg=Kg,
        K_eff=K_eff,
        residual_vector=residual.residual_vector,
        residual_norm=residual.residual_norm,
        relative_residual=residual.relative_residual,
        internal_force_norm=residual.internal_force_norm,
        linear_force_norm=residual.linear_force_norm,
        geometric_force_norm=residual.geometric_force_norm,
        free_dofs=residual.free_dofs,
        fixed_dofs=residual.fixed_dofs,
        bc_diagnostics=residual.bc_diagnostics,
    )
end

function _formal_vk_quad4_extra_energy(coords::AbstractMatrix, u_loc::AbstractVector{T}, Cm::AbstractMatrix) where {T}
    pt = 1.0 / sqrt(3.0)
    gauss_pts = (-pt, -pt, pt, -pt, pt, pt, -pt, pt)
    energy_extra = zero(T)

    for gp in 1:4
        r = gauss_pts[2gp-1]
        s = gauss_pts[2gp]
        dNr, dNs = FEM.shape_derivs_quad(r, s)
        J = [dNr'; dNs'] * coords
        detJ = max(abs(det(J)), T(1e-12))
        dN = inv(J) * [dNr'; dNs']

        ux = zero(T); uy = zero(T); vx = zero(T); vy = zero(T); wx = zero(T); wy = zero(T)
        for a in 1:4
            base = (a-1) * 6
            dNx = dN[1, a]
            dNy = dN[2, a]
            ux += dNx * u_loc[base+1]
            uy += dNy * u_loc[base+1]
            vx += dNx * u_loc[base+2]
            vy += dNy * u_loc[base+2]
            wx += dNx * u_loc[base+3]
            wy += dNy * u_loc[base+3]
        end

        eps_lin = [ux, vy, uy + vx]
        half = T(0.5)
        eps_nl = [half * wx^2, half * wy^2, wx * wy]
        N_lin = Cm * eps_lin
        N_tot = Cm * (eps_lin + eps_nl)
        energy_extra += (half * dot(eps_lin + eps_nl, N_tot) - half * dot(eps_lin, N_lin)) * detJ
    end

    return energy_extra
end

function _formal_vk_tria3_extra_energy(coords::AbstractMatrix, u_loc::AbstractVector{T}, Cm::AbstractMatrix) where {T}
    x1, y1 = coords[1,1], coords[1,2]
    x2, y2 = coords[2,1], coords[2,2]
    x3, y3 = coords[3,1], coords[3,2]
    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = abs(area2) / 2.0
    area < 1e-12 && return zero(T)

    dNdx = [(y2 - y3) / area2, (y3 - y1) / area2, (y1 - y2) / area2]
    dNdy = [(x3 - x2) / area2, (x1 - x3) / area2, (x2 - x1) / area2]

    ux = zero(T); uy = zero(T); vx = zero(T); vy = zero(T); wx = zero(T); wy = zero(T)
    for a in 1:3
        base = (a-1) * 6
        ux += dNdx[a] * u_loc[base+1]
        uy += dNdy[a] * u_loc[base+1]
        vx += dNdx[a] * u_loc[base+2]
        vy += dNdy[a] * u_loc[base+2]
        wx += dNdx[a] * u_loc[base+3]
        wy += dNdy[a] * u_loc[base+3]
    end

    eps_lin = [ux, vy, uy + vx]
    half = T(0.5)
    eps_nl = [half * wx^2, half * wy^2, wx * wy]
    N_lin = Cm * eps_lin
    N_tot = Cm * (eps_lin + eps_nl)
    return area * (half * dot(eps_lin + eps_nl, N_tot) - half * dot(eps_lin, N_lin))
end

function _formal_vk_quad4_extra_local(coords::AbstractMatrix, u_loc::AbstractVector, Cm::AbstractMatrix)
    energy_fun = u -> _formal_vk_quad4_extra_energy(coords, u, Cm)
    f_loc = ForwardDiff.gradient(energy_fun, u_loc)
    K_loc = ForwardDiff.hessian(energy_fun, u_loc)
    K_loc .= 0.5 .* (K_loc .+ K_loc')
    energy_extra = energy_fun(u_loc)
    return f_loc, K_loc, energy_extra
end

function _formal_vk_tria3_extra_local(coords::AbstractMatrix, u_loc::AbstractVector, Cm::AbstractMatrix)
    energy_fun = u -> _formal_vk_tria3_extra_energy(coords, u, Cm)
    f_loc = ForwardDiff.gradient(energy_fun, u_loc)
    K_loc = ForwardDiff.hessian(energy_fun, u_loc)
    K_loc .= 0.5 .* (K_loc .+ K_loc')
    energy_extra = energy_fun(u_loc)
    return f_loc, K_loc, energy_extra
end

function _formal_vk_accumulate_local_contribution!(I_idx::Vector{Int}, J_idx::Vector{Int}, V_val::Vector{Float64},
                                                   f_extra::Vector{Float64}, dofs::Vector{Int},
                                                   f_glob::AbstractVector, K_glob::AbstractMatrix)
    for a in eachindex(dofs)
        f_extra[dofs[a]] += f_glob[a]
    end
    for cidx in eachindex(dofs), ridx in eachindex(dofs)
        val = K_glob[ridx, cidx]
        abs(val) <= 0.0 && continue
        push!(I_idx, dofs[ridx])
        push!(J_idx, dofs[cidx])
        push!(V_val, val)
    end
end

function _formal_vk_tria3_global_contribution(prop, el, mat, tri_indices::NTuple{3,Int},
                                              tri_points::NTuple{3,SVector{3,Float64}},
                                              node_R, u_state, snorm_normals;
                                              constitutive_tri::Bool=true,
                                              constitutive_edge::Tuple{SVector{3,Float64},SVector{3,Float64}}=(tri_points[1], tri_points[2]))
    i1, i2, i3 = tri_indices
    p1, p2, p3 = tri_points
    v1, v2, v3 = shell_element_frame_fast(p1, p2, p3, p3, 3)
    v1, v2, v3 = apply_snorm_to_frame(v1, v2, v3, [i1, i2, i3], snorm_normals)
    Rel_t = Matrix(vcat(v1', v2', v3'))
    c = (p1 + p2 + p3) / 3.0
    lc = zeros(3, 2)
    lc[1,1] = dot(p1-c, v1); lc[1,2] = dot(p1-c, v2)
    lc[2,1] = dot(p2-c, v1); lc[2,2] = dot(p2-c, v2)
    lc[3,1] = dot(p3-c, v1); lc[3,2] = dot(p3-c, v2)
    u_loc = zeros(18)
    T_buf = zeros(18, 18)
    for (k, idx) in enumerate((i1, i2, i3))
        base = (k-1) * 6
        TR = Rel_t * node_R[idx]
        T_buf[base+1:base+3, base+1:base+3] .= TR
        T_buf[base+4:base+6, base+4:base+6] .= TR
        u_loc[base+1:base+3] .= TR * u_state[(idx-1)*6+1:(idx-1)*6+3]
        u_loc[base+4:base+6] .= TR * u_state[(idx-1)*6+4:(idx-1)*6+6]
    end

    ref_p1, ref_p2 = constitutive_edge
    Cm = _formal_shell_membrane_constitutive(prop, el, mat, v1, v2, v3, ref_p1, ref_p2; tri=constitutive_tri)
    f_loc, K_loc, e_extra = _formal_vk_tria3_extra_local(lc, u_loc, Cm)
    f_glob = T_buf' * f_loc
    K_glob = T_buf' * K_loc * T_buf
    dofs = vcat([(idx-1)*6 .+ collect(1:6) for idx in (i1, i2, i3)]...)
    return dofs, f_glob, K_glob, e_extra
end

function _assemble_formal_shell_von_karman_extra(model, id_map, X, node_R, ndof, u_state, snorm_normals)
    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]
    f_extra = zeros(ndof)
    q4_frame_mode = q4_frame_mode_from_env("JFEM_Q4_FRAME_MODE_STATIC")
    energy_extra = 0.0
    shell_count = 0
    warped_quad_fallback_count = 0
    warped_quad_split_triangle_count = 0

    for (_, el) in get(model, "CSHELLs", Dict())
        pid = string(get(el, "PID", 0))
        haskey(model["PSHELLs"], pid) || continue
        prop = model["PSHELLs"][pid]
        nids = get(el, "NODES", Int[])
        any(nid -> !haskey(id_map, nid), nids) && continue
        n = length(nids)

        if n == 4
            i1, i2, i3, i4 = id_map[nids[1]], id_map[nids[2]], id_map[nids[3]], id_map[nids[4]]
            p1 = SVector{3}(X[i1,1], X[i1,2], X[i1,3])
            p2 = SVector{3}(X[i2,1], X[i2,2], X[i2,3])
            p3 = SVector{3}(X[i3,1], X[i3,2], X[i3,3])
            p4 = SVector{3}(X[i4,1], X[i4,2], X[i4,3])
            mid = string(get(prop, "MID", 0))
            mat = _effective_mat1_for_nodes(model, mid, nids)
            if _quad4_is_coplanar(p1, p2, p3, p4)
                v1, v2, v3 = shell_element_frame_quad4(p1, p2, p3, p4, q4_frame_mode)
                v1, v2, v3 = apply_snorm_to_frame(v1, v2, v3, [i1, i2, i3, i4], snorm_normals)
                Rel_t = Matrix(vcat(v1', v2', v3'))
                c = (p1 + p2 + p3 + p4) / 4.0
                lc = zeros(4, 2)
                lc[1,1]=dot(p1-c,v1); lc[1,2]=dot(p1-c,v2)
                lc[2,1]=dot(p2-c,v1); lc[2,2]=dot(p2-c,v2)
                lc[3,1]=dot(p3-c,v1); lc[3,2]=dot(p3-c,v2)
                lc[4,1]=dot(p4-c,v1); lc[4,2]=dot(p4-c,v2)
                u_loc = zeros(24)
                T_buf = zeros(24, 24)
                for (k, idx) in enumerate((i1, i2, i3, i4))
                    base = (k-1) * 6
                    TR = Rel_t * node_R[idx]
                    T_buf[base+1:base+3, base+1:base+3] .= TR
                    T_buf[base+4:base+6, base+4:base+6] .= TR
                    u_loc[base+1:base+3] .= TR * u_state[(idx-1)*6+1:(idx-1)*6+3]
                    u_loc[base+4:base+6] .= TR * u_state[(idx-1)*6+4:(idx-1)*6+6]
                end
                Cm = _formal_shell_membrane_constitutive(prop, el, mat, v1, v2, v3, p1, p2; tri=false)
                f_loc, K_loc, e_extra = _formal_vk_quad4_extra_local(lc, u_loc, Cm)
                energy_extra += e_extra
                f_glob = T_buf' * f_loc
                K_glob = T_buf' * K_loc * T_buf
                dofs = vcat([(idx-1)*6 .+ collect(1:6) for idx in (i1, i2, i3, i4)]...)
                _formal_vk_accumulate_local_contribution!(I_idx, J_idx, V_val, f_extra, dofs, f_glob, K_glob)
            else
                warped_quad_fallback_count += 1
                constitutive_edge = (p1, p2)
                for (tri_indices, tri_points) in (
                    ((i1, i2, i3), (p1, p2, p3)),
                    ((i1, i3, i4), (p1, p3, p4)),
                )
                    dofs, f_glob, K_glob, e_extra = _formal_vk_tria3_global_contribution(
                        prop, el, mat, tri_indices, tri_points, node_R, u_state, snorm_normals;
                        constitutive_tri=false,
                        constitutive_edge=constitutive_edge,
                    )
                    energy_extra += e_extra
                    _formal_vk_accumulate_local_contribution!(I_idx, J_idx, V_val, f_extra, dofs, f_glob, K_glob)
                    warped_quad_split_triangle_count += 1
                end
            end
            shell_count += 1
        elseif n == 3
            i1, i2, i3 = id_map[nids[1]], id_map[nids[2]], id_map[nids[3]]
            p1 = SVector{3}(X[i1,1], X[i1,2], X[i1,3])
            p2 = SVector{3}(X[i2,1], X[i2,2], X[i2,3])
            p3 = SVector{3}(X[i3,1], X[i3,2], X[i3,3])
            mid = string(get(prop, "MID", 0))
            mat = get(model["MATs"], mid, Dict{String,Any}())
            dofs, f_glob, K_glob, e_extra = _formal_vk_tria3_global_contribution(
                prop, el, mat, (i1, i2, i3), (p1, p2, p3), node_R, u_state, snorm_normals;
                constitutive_tri=true,
                constitutive_edge=(p1, p2),
            )
            energy_extra += e_extra
            _formal_vk_accumulate_local_contribution!(I_idx, J_idx, V_val, f_extra, dofs, f_glob, K_glob)
            shell_count += 1
        end
    end

    K_extra = sparse(I_idx, J_idx, V_val, ndof, ndof)
    K_extra = 0.5 * (K_extra + K_extra')
    return f_extra, K_extra, Dict(
        "shell_count" => shell_count,
        "warped_quad_fallback_count" => warped_quad_fallback_count,
        "warped_quad_split_triangle_count" => warped_quad_split_triangle_count,
        "extra_energy" => energy_extra,
        "operator_nnz" => nnz(K_extra),
        "operator_norm" => nnz(K_extra) == 0 ? 0.0 : norm(K_extra.nzval),
        "operator_backend" => "forwarddiff_hessian",
    )
end

function _evaluate_nonlinear_state_formal(K_linear, F_applied, ndof, model, id_map, X, spc_id, node_R, u_state,
                                          snorm_normals, rbe3_map)
    f_extra, K_extra, formal_diag = _assemble_formal_shell_von_karman_extra(
        model, id_map, X, node_R, ndof, u_state, snorm_normals)
    K_eff = K_linear + K_extra
    free_dofs, fixed_dofs, bc_diagnostics = compute_free_dofs(
        K_eff, ndof, model, id_map, spc_id, rbe3_map; return_diagnostics=true)
    linear_force = K_linear * u_state
    internal_force = linear_force + f_extra
    residual_vector = internal_force - F_applied
    active_dofs = isempty(free_dofs) ? collect(1:ndof) : free_dofs
    residual_norm = norm(residual_vector[active_dofs])
    force_norm = norm(F_applied[active_dofs])
    relative_residual = residual_norm / max(force_norm, 1e-30)
    nonlinear_force_norm = norm(f_extra[active_dofs])
    linear_strain_energy = 0.5 * dot(u_state, linear_force)
    internal_energy = linear_strain_energy + formal_diag["extra_energy"]
    potential_energy = internal_energy - dot(F_applied, u_state)
    return (
        Kg=K_extra,
        K_eff=K_eff,
        residual_vector=residual_vector,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        internal_force_norm=norm(internal_force[active_dofs]),
        linear_force_norm=norm(linear_force[active_dofs]),
        geometric_force_norm=nonlinear_force_norm,
        nonlinear_force_norm=nonlinear_force_norm,
        linear_strain_energy=linear_strain_energy,
        internal_energy=internal_energy,
        potential_energy=potential_energy,
        free_dofs=free_dofs,
        fixed_dofs=fixed_dofs,
        bc_diagnostics=bc_diagnostics,
        formal_diagnostics=formal_diag,
    )
end

_state_field(state, name::Symbol, default) =
    name in propertynames(state) ? getproperty(state, name) : default

@inline function _metric_reduced(after::Real, before::Real; rtol::Float64=1e-12, atol::Float64=1e-30)
    slack = max(atol, rtol * max(abs(before), abs(after), 1.0))
    return after <= before + slack
end

@inline function _normalized_residual_improvement(current_relative_residual::Real,
                                                  trial_relative_residual::Real)
    current = max(Float64(current_relative_residual), eps(Float64))
    return (Float64(current_relative_residual) - Float64(trial_relative_residual)) / current
end

@inline function _step_contraction_cost(step_scale::Real)
    return max(-log2(max(Float64(step_scale), eps(Float64))), 0.0)
end

@inline function _formal_cleanup_improvement_sufficient(current_relative_residual::Real,
                                                        trial_relative_residual::Real,
                                                        step_scale::Real,
                                                        efficiency_lambda::Float64)
    improvement = _normalized_residual_improvement(current_relative_residual, trial_relative_residual)
    return improvement >= efficiency_lambda * _step_contraction_cost(step_scale)
end

@inline function _line_search_merit(state)
    residual_norm = Float64(state.residual_norm)
    return 0.5 * residual_norm * residual_norm
end

@inline function _line_search_merit_directional_derivative(state)
    residual_norm = Float64(state.residual_norm)
    return -(residual_norm * residual_norm)
end

function _step_growth_quality(step_records)
    if isempty(step_records)
        return (
            eligible=false,
            all_full_step_iterations=false,
            monotone_residual_acceptance=false,
            used_best_available_trial=false,
            max_line_search_backtracks=0,
            max_accepted_trial_index=0,
        )
    end

    all_full_step_iterations = true
    monotone_residual_acceptance = true
    used_best_available_trial = false
    max_line_search_backtracks = 0
    max_accepted_trial_index = 0

    for iter in step_records
        all_full_step_iterations &= Float64(get(iter, "accepted_step_scale", 0.0)) >= 0.99
        monotone_residual_acceptance &= (
            Bool(get(iter, "residual_reduced", false)) ||
            String(get(iter, "accepted_by", "")) == "residual_tolerance"
        )
        used_best_available_trial |= Bool(get(iter, "used_best_available_trial", false))
        max_line_search_backtracks = max(
            max_line_search_backtracks,
            max(length(get(iter, "line_search_trials", Any[])) - 1, 0),
        )
        max_accepted_trial_index = max(max_accepted_trial_index, Int(get(iter, "accepted_trial_index", 0)))
    end

    eligible =
        all_full_step_iterations &&
        monotone_residual_acceptance &&
        !used_best_available_trial &&
        max_line_search_backtracks == 0

    return (
        eligible=eligible,
        all_full_step_iterations=all_full_step_iterations,
        monotone_residual_acceptance=monotone_residual_acceptance,
        used_best_available_trial=used_best_available_trial,
        max_line_search_backtracks=max_line_search_backtracks,
        max_accepted_trial_index=max_accepted_trial_index,
    )
end

function _cutback_recovery_quality(initial_relative_residual::Real, final_relative_residual::Real,
                                   attempted_increment::Real, requested_increment::Real)
    safe_initial = max(Float64(initial_relative_residual), 1e-30)
    safe_final = max(Float64(final_relative_residual), 1e-30)
    safe_attempted = max(Float64(attempted_increment), 1e-30)
    reference_ratio = max(Float64(requested_increment) / safe_attempted, 1.0)
    contraction_ratio = max(safe_initial / safe_final, 1.0)
    proposed_ratio = sqrt(contraction_ratio)
    bracket_midpoint_ratio = sqrt(reference_ratio)
    accepted_ratio = min(bracket_midpoint_ratio, max(1.0, proposed_ratio))
    return (
        contraction_ratio=contraction_ratio,
        reference_ratio=reference_ratio,
        bracket_midpoint_ratio=bracket_midpoint_ratio,
        proposed_ratio=proposed_ratio,
        accepted_ratio=accepted_ratio,
    )
end

function _solve_nonlinear_correction(K_eff, residual_rhs, ndof, model, id_map, spc_id, rbe3_map;
                                     free_dofs=nothing, fixed_dofs=nothing, bc_diagnostics=nothing)
    reused_state_partition = !(isnothing(free_dofs) || isnothing(fixed_dofs) || isnothing(bc_diagnostics))
    if !reused_state_partition
        free_dofs, fixed_dofs, bc_diagnostics = compute_free_dofs(
            K_eff, ndof, model, id_map, spc_id, rbe3_map; return_diagnostics=true)
    end
    correction = zeros(ndof)

    if isempty(free_dofs)
        return correction, Dict{String,Any}(
            "backend" => "empty_system",
            "free_dofs" => 0,
            "fixed_dofs" => length(fixed_dofs),
            "rhs_norm" => 0.0,
            "relative_residual" => 0.0,
            "bc_partition" => bc_diagnostics,
            "reused_state_partition" => reused_state_partition,
        )
    end

    K_ff = K_eff[free_dofs, free_dofs]
    K_ff = 0.5 * (K_ff + K_ff')
    rhs_ff = residual_rhs[free_dofs]
    backend = "direct_cholesky"
    used_lu_fallback = false

    delta_ff = if norm(rhs_ff) <= 1e-30
        zeros(length(free_dofs))
    else
        try
            cholesky(Symmetric(K_ff)) \ rhs_ff
        catch
            backend = "direct_lu"
            used_lu_fallback = true
            lu(K_ff) \ rhs_ff
        end
    end

    correction[free_dofs] = delta_ff
    if !isempty(rbe3_map)
        for (dep_dof, pairs) in rbe3_map
            correction[dep_dof] = sum(coeff * correction[ind_dof] for (ind_dof, coeff) in pairs)
        end
    end

    solve_residual = K_ff * delta_ff - rhs_ff
    rel_residual = norm(solve_residual) / max(norm(rhs_ff), 1e-30)
    diagnostics = Dict{String,Any}(
        "backend" => backend,
        "used_lu_fallback" => used_lu_fallback,
        "free_dofs" => length(free_dofs),
        "fixed_dofs" => length(fixed_dofs),
        "rhs_norm" => norm(rhs_ff),
        "residual_norm" => norm(solve_residual),
        "relative_residual" => rel_residual,
        "bc_partition" => bc_diagnostics,
        "reused_state_partition" => reused_state_partition,
    )
    return correction, diagnostics
end

function solve_case_state(K, ndof, model, id_map, X, load_id, spc_id, node_R;
                          max_elem_stiff=0.0,
                          rbe3_map=Dict{Int,Vector{Tuple{Int,Float64}}}(),
                          orig_diag=Float64[],
                          load_scale::Float64=1.0,
                          temp_load_id=nothing,
                          linear_cache=nothing,
                          log_rbe3::Bool=true)
    F_applied = _assemble_applied_force(
        ndof, model, id_map, X, load_id, node_R, rbe3_map;
        load_scale=load_scale,
        temp_load_id=temp_load_id,
        log_rbe3=log_rbe3,
    )

    had_spc_id = haskey(model, "_spc_id")
    prev_spc_id = had_spc_id ? model["_spc_id"] : nothing
    model["_spc_id"] = spc_id
    local u_global, fixed_dofs, solver_diagnostics
    try
        u_global, fixed_dofs, _, solver_diagnostics = apply_bc_and_solve(
            K, ndof, model, id_map, F_applied, node_R, rbe3_map, max_elem_stiff, orig_diag;
            linear_cache=linear_cache)
    finally
        if had_spc_id
            model["_spc_id"] = prev_spc_id
        else
            delete!(model, "_spc_id")
        end
    end

    return u_global, fixed_dofs, solver_diagnostics, F_applied
end

function solve_case(K, ndof, model, id_map, X, load_id, spc_id, node_R;
                    max_elem_stiff=0.0,
                    rbe3_map=Dict{Int,Vector{Tuple{Int,Float64}}}(),
                    snorm_normals=Dict{Int,SVector{3,Float64}}(),
                    orig_diag=Float64[],
                    load_scale::Float64=1.0,
                    temp_load_id=nothing,
                    linear_cache=nothing)
    u_global, fixed_dofs, solver_diagnostics, F_applied = solve_case_state(
        K, ndof, model, id_map, X, load_id, spc_id, node_R;
        max_elem_stiff=max_elem_stiff,
        rbe3_map=rbe3_map,
        orig_diag=orig_diag,
        load_scale=load_scale,
        temp_load_id=temp_load_id,
        linear_cache=linear_cache,
        log_rbe3=true,
    )

    R = K * u_global - F_applied
    u_out, stresses, results_json = _build_results_from_state(
        ndof, model, id_map, X, node_R, u_global, R, snorm_normals, solver_diagnostics)

    return u_out, stresses, results_json, u_global, fixed_dofs
end

function solve_nonlinear_static(K_linear, ndof, model, id_map, X, load_id, spc_id, node_R;
                                load_steps::Int=4,
                                max_iter::Int=8,
                                tol::Float64=1e-6,
                                relaxation::Float64=1.0,
                                residual_tol::Float64=1e-6,
                                residual_model::Symbol=:tangent_operator,
                                nonlinear_method::Symbol=:auto,
                                line_search_max_backtracks::Int=6,
                                line_search_reduction::Float64=0.5,
                                max_cutbacks::Int=6,
                                cutback_reduction::Float64=0.5,
                                step_growth::Float64=1.25,
                                max_elem_stiff=0.0,
                                rbe3_map=Dict{Int,Vector{Tuple{Int,Float64}}}(),
                                snorm_normals=Dict{Int,SVector{3,Float64}}(),
                                orig_diag=Float64[],
                                temp_load_id=nothing)
    load_steps = max(load_steps, 1)
    max_iter = max(max_iter, 1)
    relaxation = clamp(relaxation, 1e-3, 1.0)
    residual_tol = max(residual_tol, 1e-12)
    residual_model = residual_model in (:secant_geometric, :tangent_operator) ? residual_model : :tangent_operator
    formal_support = _formal_von_karman_shell_support_report(model, id_map, X)
    formal_supported = formal_support.supported
    nonlinear_method =
        if nonlinear_method === :formal_shell_von_karman
            formal_supported || error("[SOLVER] PARAM_NLMETHOD=formal_shell_von_karman requested, but the model is outside the currently supported flat-shell subset")
            :formal_shell_von_karman
        else
            :legacy_geometric
        end
    line_search_max_backtracks = max(line_search_max_backtracks, 0)
    line_search_reduction = clamp(line_search_reduction, 0.1, 0.9)
    max_cutbacks = max(max_cutbacks, 0)
    cutback_reduction = clamp(cutback_reduction, 0.1, 0.9)
    step_growth = max(step_growth, 1.0)
    formal_armijo_c1 = 1e-4
    near_converged_residual_band = 5.0 * residual_tol
    near_converged_efficiency_lambda = 1e-3

    u_committed = zeros(ndof)
    load_history = Any[]
    final_bundle = nothing
    final_state_summary = nothing
    nominal_increment = 1.0 / load_steps
    next_increment = nominal_increment
    current_load_scale = 0.0
    accepted_step = 0
    terminated_early = false
    state_evaluation_count = 0
    correction_partition_reuse_count = 0
    accepted_state_reuse_count = 0
    step_growth_applied_count = 0
    evaluate_state = nonlinear_method === :formal_shell_von_karman ?
        (u_state, F_step) -> _evaluate_nonlinear_state_formal(
            K_linear, F_step, ndof, model, id_map, X, spc_id, node_R, u_state, snorm_normals, rbe3_map
        ) :
        (u_state, F_step) -> _evaluate_nonlinear_state(
            K_linear, F_step, ndof, model, id_map, X, spc_id, node_R, u_state, snorm_normals, rbe3_map;
            residual_model=residual_model,
        )

    while current_load_scale < 1.0 - 1e-12
        remaining = 1.0 - current_load_scale
        attempted_increment = min(next_increment, remaining)
        requested_increment = attempted_increment
        recovery_reference_increment = requested_increment
        attempted_scale = current_load_scale + attempted_increment
        cutback_count = 0
        attempt_history = Any[]
        step_start_state = copy(u_committed)
        step_accepted = false

        while !step_accepted
            F_step = _assemble_applied_force(
                ndof, model, id_map, X, load_id, node_R, rbe3_map;
                load_scale=attempted_scale,
                temp_load_id=temp_load_id,
                log_rbe3=false,
            )
            u_iter = copy(step_start_state)
            step_records = Any[]
            step_converged = false
            final_rel_change = Inf
            final_rel_residual = Inf
            final_relative_incremental_work = Inf
            initial_rel_residual = Inf
            last_accepted_state = nothing
            for iter in 1:max_iter
                current_state = evaluate_state(u_iter, F_step)
                state_evaluation_count += 1
                if iter == 1
                    initial_rel_residual = current_state.relative_residual
                end

                residual_rhs = -current_state.residual_vector
                delta_u, correction_diagnostics = _solve_nonlinear_correction(
                    current_state.K_eff, residual_rhs, ndof, model, id_map, spc_id, rbe3_map,
                    free_dofs=current_state.free_dofs,
                    fixed_dofs=current_state.fixed_dofs,
                    bc_diagnostics=current_state.bc_diagnostics,
                )
                if get(correction_diagnostics, "reused_state_partition", false)
                    correction_partition_reuse_count += 1
                end

                line_search_trials = Any[]
                accepted_alpha = 0.0
                accepted_candidate = copy(u_iter)
                accepted_state = nothing
                accepted_reason = "rejected"
                accepted_trial_index = 0
                best_accepted_alpha = 0.0
                best_accepted_candidate = copy(u_iter)
                best_accepted_state = nothing
                best_accepted_reason = "rejected"
                best_accepted_trial_index = 0
                best_accepted_score = nothing
                best_accepted_is_residual_based = false
                best_alpha = 0.0
                best_candidate = copy(u_iter)
                best_state = nothing
                best_metric = Inf
                best_trial_index = 0
                alpha = relaxation
                current_potential = _state_field(current_state, :potential_energy, nothing)
                directional_derivative = dot(current_state.residual_vector, delta_u)
                current_line_search_merit = _line_search_merit(current_state)
                line_search_merit_directional_derivative = _line_search_merit_directional_derivative(current_state)

                for ls_trial in 1:(line_search_max_backtracks + 1)
                    candidate = u_iter .+ alpha .* delta_u
                    candidate_state = evaluate_state(candidate, F_step)
                    state_evaluation_count += 1
                    candidate_rel_change = norm(alpha .* delta_u) / max(norm(candidate), 1e-30)
                    candidate_formal = _state_field(candidate_state, :formal_diagnostics, Dict{String,Any}())
                    candidate_potential = _state_field(candidate_state, :potential_energy, nothing)
                    candidate_merit = _line_search_merit(candidate_state)
                    armijo_satisfied = false
                    if nonlinear_method === :formal_shell_von_karman &&
                       !(isnothing(current_potential) || isnothing(candidate_potential)) &&
                       directional_derivative < 0.0
                        armijo_satisfied =
                            candidate_potential <= current_potential + formal_armijo_c1 * alpha * directional_derivative
                    end
                    residual_reduced = _metric_reduced(candidate_state.relative_residual, current_state.relative_residual)
                    internal_force_reduced = _metric_reduced(candidate_state.internal_force_norm, current_state.internal_force_norm)
                    potential_reduced =
                        if isnothing(current_potential) || isnothing(candidate_potential)
                            missing
                        else
                            _metric_reduced(candidate_potential, current_potential)
                        end
                    acceptance_reason =
                        if candidate_state.relative_residual <= residual_tol
                            "residual_tolerance"
                        elseif residual_reduced
                            "residual_reduction"
                        elseif armijo_satisfied
                            "armijo"
                        else
                            "rejected"
                        end
                    push!(line_search_trials, Dict(
                        "trial" => ls_trial,
                        "step_scale" => alpha,
                        "merit" => candidate_merit,
                        "relative_change" => candidate_rel_change,
                        "internal_force_norm" => candidate_state.internal_force_norm,
                        "linear_force_norm" => candidate_state.linear_force_norm,
                        "geometric_force_norm" => candidate_state.geometric_force_norm,
                        "potential_energy" => isnothing(candidate_potential) ? missing : candidate_potential,
                        "armijo_satisfied" => armijo_satisfied,
                        "formal_operator_nnz" => get(candidate_formal, "operator_nnz", 0),
                        "formal_operator_norm" => get(candidate_formal, "operator_norm", 0.0),
                        "formal_extra_energy" => get(candidate_formal, "extra_energy", 0.0),
                        "formal_shell_count" => get(candidate_formal, "shell_count", 0),
                        "residual_norm" => candidate_state.residual_norm,
                        "relative_residual" => candidate_state.relative_residual,
                        "residual_reduced" => residual_reduced,
                        "internal_force_reduced" => internal_force_reduced,
                        "potential_reduced" => potential_reduced,
                        "acceptance_reason" => acceptance_reason,
                        "kg_nnz" => nnz(candidate_state.Kg),
                        "kg_norm" => norm(candidate_state.Kg.nzval),
                    ))

                    if best_state === nothing || candidate_merit < best_metric
                        best_alpha = alpha
                        best_candidate = copy(candidate)
                        best_state = candidate_state
                        best_metric = candidate_merit
                        best_trial_index = ls_trial
                    end

                    if acceptance_reason != "rejected"
                        if nonlinear_method === :formal_shell_von_karman
                            near_converged_cleanup_active =
                                current_state.relative_residual <= near_converged_residual_band
                            if acceptance_reason == "residual_tolerance"
                                accepted_alpha = alpha
                                accepted_candidate = candidate
                                accepted_state = candidate_state
                                accepted_reason = acceptance_reason
                                accepted_trial_index = ls_trial
                                break
                            elseif residual_reduced
                                candidate_score = (
                                    Float64(candidate_state.relative_residual),
                                    -Float64(alpha),
                                    Int(ls_trial),
                                )
                                if ls_trial == 1
                                    accepted_alpha = alpha
                                    accepted_candidate = candidate
                                    accepted_state = candidate_state
                                    accepted_reason = acceptance_reason
                                    accepted_trial_index = ls_trial
                                    break
                                elseif near_converged_cleanup_active
                                    if _formal_cleanup_improvement_sufficient(
                                           current_state.relative_residual,
                                           candidate_state.relative_residual,
                                           alpha,
                                           near_converged_efficiency_lambda,
                                       )
                                        accepted_alpha = alpha
                                        accepted_candidate = candidate
                                        accepted_state = candidate_state
                                        accepted_reason = acceptance_reason
                                        accepted_trial_index = ls_trial
                                        break
                                    elseif best_accepted_state === nothing
                                        best_accepted_alpha = alpha
                                        best_accepted_candidate = copy(candidate)
                                        best_accepted_state = candidate_state
                                        best_accepted_reason = acceptance_reason
                                        best_accepted_trial_index = ls_trial
                                    end
                                elseif !best_accepted_is_residual_based ||
                                       isnothing(best_accepted_score) ||
                                       candidate_score < best_accepted_score
                                    best_accepted_alpha = alpha
                                    best_accepted_candidate = copy(candidate)
                                    best_accepted_state = candidate_state
                                    best_accepted_reason = acceptance_reason
                                    best_accepted_trial_index = ls_trial
                                    best_accepted_score = candidate_score
                                    best_accepted_is_residual_based = true
                                end
                            elseif !best_accepted_is_residual_based && best_accepted_state === nothing
                                best_accepted_alpha = alpha
                                best_accepted_candidate = copy(candidate)
                                best_accepted_state = candidate_state
                                best_accepted_reason = acceptance_reason
                                best_accepted_trial_index = ls_trial
                            end
                        else
                            accepted_alpha = alpha
                            accepted_candidate = candidate
                            accepted_state = candidate_state
                            accepted_reason = acceptance_reason
                            accepted_trial_index = ls_trial
                            break
                        end
                    end

                    alpha *= line_search_reduction
                end

                if accepted_state === nothing &&
                   nonlinear_method === :formal_shell_von_karman &&
                   best_accepted_state !== nothing
                    accepted_alpha = best_accepted_alpha
                    accepted_candidate = best_accepted_candidate
                    accepted_state = best_accepted_state
                    accepted_reason = best_accepted_reason
                    accepted_trial_index = best_accepted_trial_index
                elseif accepted_state === nothing
                    accepted_alpha = best_alpha
                    accepted_candidate = best_candidate
                    accepted_state = best_state
                    accepted_reason = "best_available_trial"
                    accepted_trial_index = best_trial_index
                end

                u_iter .= accepted_candidate
                last_accepted_state = accepted_state
                rel_change = norm(accepted_alpha .* delta_u) / max(norm(u_iter), 1e-30)
                reference_work = max(abs(dot(u_iter, F_step)), 1e-30)
                relative_incremental_work =
                    abs(dot(accepted_alpha .* delta_u, current_state.residual_vector)) / reference_work
                final_rel_change = rel_change
                final_rel_residual = accepted_state.relative_residual
                final_relative_incremental_work = relative_incremental_work
                current_formal = _state_field(current_state, :formal_diagnostics, Dict{String,Any}())
                accepted_formal = _state_field(accepted_state, :formal_diagnostics, Dict{String,Any}())
                accepted_residual_reduced = _metric_reduced(accepted_state.relative_residual, current_state.relative_residual)
                accepted_internal_force_reduced = _metric_reduced(accepted_state.internal_force_norm, current_state.internal_force_norm)
                accepted_potential = _state_field(accepted_state, :potential_energy, nothing)
                accepted_potential_reduced =
                    if isnothing(current_potential) || isnothing(accepted_potential)
                        missing
                    else
                        _metric_reduced(accepted_potential, current_potential)
                    end
                push!(step_records, Dict(
                    "iteration" => iter,
                    "load_scale" => attempted_scale,
                    "load_increment" => attempted_increment,
                    "relative_change" => rel_change,
                    "directional_derivative" => directional_derivative,
                    "line_search_merit_before" => current_line_search_merit,
                    "line_search_merit_directional_derivative" => line_search_merit_directional_derivative,
                    "correction_rhs_norm" => norm(residual_rhs),
                    "correction_solver" => correction_diagnostics,
                    "internal_force_norm_before" => current_state.internal_force_norm,
                    "linear_force_norm_before" => current_state.linear_force_norm,
                    "geometric_force_norm_before" => current_state.geometric_force_norm,
                    "potential_energy_before" => isnothing(current_potential) ? missing : current_potential,
                    "formal_operator_nnz_before" => get(current_formal, "operator_nnz", 0),
                    "formal_operator_norm_before" => get(current_formal, "operator_norm", 0.0),
                    "formal_extra_energy_before" => get(current_formal, "extra_energy", 0.0),
                    "formal_shell_count_before" => get(current_formal, "shell_count", 0),
                    "residual_norm_before" => current_state.residual_norm,
                    "relative_residual_before" => current_state.relative_residual,
                    "internal_force_norm_after" => accepted_state.internal_force_norm,
                    "linear_force_norm_after" => accepted_state.linear_force_norm,
                    "geometric_force_norm_after" => accepted_state.geometric_force_norm,
                    "potential_energy_after" => let pe = _state_field(accepted_state, :potential_energy, nothing)
                        isnothing(pe) ? missing : pe
                    end,
                    "formal_operator_nnz_after" => get(accepted_formal, "operator_nnz", 0),
                    "formal_operator_norm_after" => get(accepted_formal, "operator_norm", 0.0),
                    "formal_extra_energy_after" => get(accepted_formal, "extra_energy", 0.0),
                    "formal_shell_count_after" => get(accepted_formal, "shell_count", 0),
                    "residual_norm_after" => accepted_state.residual_norm,
                    "relative_residual_after" => accepted_state.relative_residual,
                    "accepted_step_scale" => accepted_alpha,
                    "accepted_trial_index" => accepted_trial_index,
                    "accepted_by" => accepted_reason,
                    "used_best_available_trial" => accepted_reason == "best_available_trial",
                    "line_search_merit_after" => _line_search_merit(accepted_state),
                    "relative_incremental_work" => relative_incremental_work,
                    "line_search_selection_policy" => nonlinear_method === :formal_shell_von_karman ?
                        "prefer_non_worsening_residual_then_larger_armijo_step" :
                        "first_acceptable_trial",
                    "near_converged_cleanup_gate_active" =>
                        nonlinear_method === :formal_shell_von_karman &&
                        current_state.relative_residual <= near_converged_residual_band,
                    "near_converged_cleanup_efficiency_lambda" =>
                        nonlinear_method === :formal_shell_von_karman ? near_converged_efficiency_lambda : 0.0,
                    "minimum_merit_trial_index" => best_trial_index,
                    "minimum_merit_step_scale" => best_alpha,
                    "minimum_merit_value" => best_metric,
                    "minimum_acceptable_trial_index" =>
                        nonlinear_method === :formal_shell_von_karman && best_accepted_state !== nothing ?
                        best_accepted_trial_index :
                        accepted_trial_index,
                    "minimum_acceptable_step_scale" =>
                        nonlinear_method === :formal_shell_von_karman && best_accepted_state !== nothing ?
                        best_accepted_alpha :
                        accepted_alpha,
                    "accepted_is_minimum_merit_trial" => accepted_trial_index == best_trial_index,
                    "accepted_is_minimum_acceptable_trial" =>
                        accepted_trial_index ==
                        (nonlinear_method === :formal_shell_von_karman && best_accepted_state !== nothing ?
                         best_accepted_trial_index :
                         accepted_trial_index),
                    "residual_reduced" => accepted_residual_reduced,
                    "internal_force_reduced" => accepted_internal_force_reduced,
                    "potential_reduced" => accepted_potential_reduced,
                    "line_search_trials" => line_search_trials,
                    "kg_nnz" => nnz(accepted_state.Kg),
                    "kg_norm" => norm(accepted_state.Kg.nzval),
                ))

                log_msg("[SOLVER] NL target=$(round(attempted_scale, sigdigits=4)) iter $iter: rel_change=$(round(rel_change, sigdigits=4)), rel_res_before=$(round(current_state.relative_residual, sigdigits=4)), rel_res_after=$(round(accepted_state.relative_residual, sigdigits=4)), alpha=$(round(accepted_alpha, sigdigits=4))")

                if accepted_state.relative_residual < residual_tol &&
                   (rel_change < tol || relative_incremental_work < tol)
                    step_converged = true
                    break
                end
            end

            reused_final_state = last_accepted_state !== nothing
            final_state = if reused_final_state
                accepted_state_reuse_count += 1
                last_accepted_state
            else
                state_evaluation_count += 1
                evaluate_state(u_iter, F_step)
            end
            final_solver_diagnostics = Dict{String,Any}(
                "backend" => "nonlinear_postprocess",
                "bc_partition" => final_state.bc_diagnostics,
                "residual_norm" => final_state.residual_norm,
                "relative_residual" => final_state.relative_residual,
            )
            u_out, stresses, sub_res = _build_results_from_state(
                ndof, model, id_map, X, node_R, u_iter, final_state.residual_vector, snorm_normals, final_solver_diagnostics
            )
            recovery_relative_change = 0.0
            final_rel_residual = final_state.relative_residual
            step_converged = step_converged || (
                final_rel_residual < residual_tol &&
                (final_rel_change < tol || final_relative_incremental_work < tol)
            )
            attempt_termination_reason = step_converged ? "converged" :
                ((cutback_count < max_cutbacks && attempted_increment > 1e-12) ? "cutback_retry" : "cutback_exhausted")

            attempt_record = Dict(
                "attempt" => cutback_count + 1,
                "load_scale" => attempted_scale,
                "load_increment" => attempted_increment,
                "cutback_count" => cutback_count,
                "converged" => step_converged,
                "iterations" => step_records,
                "initial_relative_residual" => initial_rel_residual,
                "final_relative_change" => final_rel_change,
                "final_relative_residual" => final_rel_residual,
                "final_relative_incremental_work" => final_relative_incremental_work,
                "recovery_relative_change" => recovery_relative_change,
                "reused_final_state" => reused_final_state,
                "termination_reason" => attempt_termination_reason,
            )
            push!(attempt_history, attempt_record)

            if step_converged
                accepted_step += 1
                current_load_scale = attempted_scale
                u_committed .= u_iter
                final_bundle = (u_out, stresses, sub_res, copy(u_committed), final_state.fixed_dofs, final_state.Kg)
                final_state_summary = final_state
                remaining_after_accept = max(1.0 - current_load_scale, 0.0)
                tiny_increment_threshold = max(1e-12, nominal_increment * 1e-4)
                tiny_increment_step = attempted_increment <= tiny_increment_threshold
                growth_quality = _step_growth_quality(step_records)
                fast_convergence = growth_quality.eligible
                next_increment_reason = "nominal_schedule"
                candidate_next_increment = nominal_increment
                cutback_recovery_quality = nothing
                push!(load_history, Dict(
                    "step" => accepted_step,
                    "load_scale" => current_load_scale,
                    "load_increment" => attempted_increment,
                    "accepted" => true,
                    "converged" => true,
                    "cutback_count" => cutback_count,
                    "attempts" => attempt_history,
                    "iterations" => step_records,
                    "initial_relative_residual" => initial_rel_residual,
                    "final_relative_change" => final_rel_change,
                    "final_relative_residual" => final_rel_residual,
                    "final_relative_incremental_work" => final_relative_incremental_work,
                    "recovery_relative_change" => recovery_relative_change,
                    "reused_final_state" => reused_final_state,
                    "termination_reason" => "converged",
                    "fast_convergence" => fast_convergence,
                    "growth_eligible" => growth_quality.eligible,
                    "growth_all_full_step_iterations" => growth_quality.all_full_step_iterations,
                    "growth_monotone_residual_acceptance" => growth_quality.monotone_residual_acceptance,
                    "growth_used_best_available_trial" => growth_quality.used_best_available_trial,
                    "growth_max_line_search_backtracks" => growth_quality.max_line_search_backtracks,
                    "growth_max_accepted_trial_index" => growth_quality.max_accepted_trial_index,
                    "tiny_increment_step" => tiny_increment_step,
                ))
                if cutback_count > 0
                    cutback_recovery_quality = _cutback_recovery_quality(
                        initial_rel_residual,
                        final_rel_residual,
                        attempted_increment,
                        requested_increment,
                    )
                    recovered_increment = attempted_increment * cutback_recovery_quality.accepted_ratio
                    candidate_next_increment = min(nominal_increment, recovered_increment)
                    next_increment_reason =
                        candidate_next_increment > attempted_increment + 1e-12 ?
                        "cutback_recovery" :
                        "nominal_schedule"
                elseif fast_convergence && step_growth > 1.0
                    candidate_next_increment = max(nominal_increment, attempted_increment * step_growth)
                    next_increment_reason = "fast_convergence_growth"
                else
                    candidate_next_increment = nominal_increment
                    next_increment_reason = "nominal_schedule"
                end
                next_increment = min(remaining_after_accept, candidate_next_increment)
                if next_increment_reason == "fast_convergence_growth" && next_increment > nominal_increment + 1e-12
                    step_growth_applied_count += 1
                    load_history[end]["step_growth_applied"] = true
                else
                    load_history[end]["step_growth_applied"] = false
                end
                load_history[end]["next_load_increment"] = next_increment
                load_history[end]["next_increment_reason"] = next_increment_reason
                load_history[end]["next_increment_ratio"] =
                    attempted_increment > 0.0 ? next_increment / attempted_increment : 0.0
                load_history[end]["recovery_reference_increment"] = recovery_reference_increment
                if cutback_count > 0
                    load_history[end]["recovery_reference_ratio"] =
                        attempted_increment > 0.0 ? recovery_reference_increment / attempted_increment : 0.0
                    load_history[end]["cutback_recovery_contraction_ratio"] =
                        cutback_recovery_quality === nothing ? 1.0 : cutback_recovery_quality.contraction_ratio
                    load_history[end]["cutback_recovery_proposed_ratio"] =
                        cutback_recovery_quality === nothing ? 1.0 : cutback_recovery_quality.proposed_ratio
                    load_history[end]["cutback_recovery_accepted_ratio"] =
                        cutback_recovery_quality === nothing ? 1.0 : cutback_recovery_quality.accepted_ratio
                end
                step_accepted = true
            elseif cutback_count < max_cutbacks && attempted_increment > 1e-12
                recovery_reference_increment = attempted_increment
                attempted_increment *= cutback_reduction
                attempted_scale = current_load_scale + attempted_increment
                cutback_count += 1
                log_msg("[SOLVER] NL cutback: retrying with reduced load increment $(round(attempted_increment, sigdigits=4)) (target scale $(round(attempted_scale, sigdigits=4)))")
            else
                accepted_step += 1
                u_committed .= u_iter
                final_bundle = (u_out, stresses, sub_res, copy(u_committed), final_state.fixed_dofs, final_state.Kg)
                final_state_summary = final_state
                push!(load_history, Dict(
                    "step" => accepted_step,
                    "load_scale" => attempted_scale,
                    "load_increment" => attempted_increment,
                    "accepted" => false,
                    "converged" => false,
                    "cutback_count" => cutback_count,
                    "attempts" => attempt_history,
                    "iterations" => step_records,
                    "initial_relative_residual" => initial_rel_residual,
                    "final_relative_change" => final_rel_change,
                    "final_relative_residual" => final_rel_residual,
                    "final_relative_incremental_work" => final_relative_incremental_work,
                    "recovery_relative_change" => recovery_relative_change,
                    "reused_final_state" => reused_final_state,
                    "terminated_early" => true,
                    "termination_reason" => "cutback_exhausted",
                    "tiny_increment_step" => attempted_increment <= max(1e-12, nominal_increment * 1e-4),
                ))
                terminated_early = true
                step_accepted = true
            end
        end

        if terminated_early
            break
        end
    end

    isnothing(final_bundle) && error("[SOLVER] Nonlinear static solve did not produce any iterate")
    u_out, stresses, sub_res, u_analysis, fixed_dofs, Kg = final_bundle
    final_formal_diagnostics = isnothing(final_state_summary) ?
        Dict{String,Any}() :
        _state_field(final_state_summary, :formal_diagnostics, Dict{String,Any}())
    final_potential_energy = isnothing(final_state_summary) ? nothing :
        _state_field(final_state_summary, :potential_energy, nothing)
    final_internal_energy = isnothing(final_state_summary) ? nothing :
        _state_field(final_state_summary, :internal_energy, nothing)
    line_search_acceptance_counts = Dict(
        "residual_tolerance" => 0,
        "residual_reduction" => 0,
        "armijo" => 0,
        "best_available_trial" => 0,
    )
    line_search_trial_count = 0
    line_search_backtrack_count = 0
    iterations_with_residual_reduction = 0
    iterations_with_internal_force_reduction = 0
    iterations_with_potential_reduction = 0
    iterations_with_nonminimum_merit_acceptance = 0
    iterations_with_nonminimum_acceptable_acceptance = 0
    iterations_with_minimum_acceptable_distinct_from_minimum_merit = 0
    max_minimum_merit_trial_gap = 0
    max_minimum_acceptable_trial_gap = 0
    max_minimum_acceptable_vs_merit_trial_gap = 0
    tiny_increment_threshold = max(1e-12, nominal_increment * 1e-4)
    tiny_increment_step_count = 0
    tiny_increment_cutback_step_count = 0
    tiny_increment_nominal_recovery_count = 0
    tiny_increment_plateau_count = 0
    tiny_increment_plateaus = Any[]
    max_consecutive_tiny_increment_steps = 0
    max_consecutive_tiny_cutback_steps = 0
    current_tiny_increment_streak = 0
    current_tiny_cutback_streak = 0
    current_tiny_plateau_start_step = 0
    current_tiny_plateau_end_step = 0
    current_tiny_plateau_entry_load_scale = 0.0
    current_tiny_plateau_exit_load_scale = 0.0
    current_tiny_plateau_entry_increment = 0.0
    current_tiny_plateau_exit_increment = 0.0
    current_tiny_plateau_exit_next_increment = 0.0
    current_tiny_plateau_exit_next_reason = ""
    current_tiny_plateau_cutback_step_count = 0
    current_tiny_plateau_max_cutback_recovery_ratio = 0.0
    current_tiny_plateau_max_recovery_reference_ratio = 0.0
    current_tiny_plateau_iteration_count = 0
    current_tiny_plateau_line_search_trial_count = 0
    current_tiny_plateau_line_search_backtrack_count = 0
    current_tiny_plateau_residual_satisfied_retry_count = 0
    current_tiny_plateau_min_residual_satisfied_retry_relative_change_ratio = Inf
    max_next_increment_ratio = 0.0
    max_cutback_recovery_ratio = 0.0
    max_cutback_recovery_contraction_ratio = 0.0
    max_cutback_recovery_proposed_ratio = 0.0
    max_tiny_increment_recovery_ratio = 0.0
    tiny_increment_plateau_iteration_count = 0
    tiny_increment_plateau_line_search_trial_count = 0
    tiny_increment_plateau_line_search_backtrack_count = 0
    tiny_increment_plateau_residual_satisfied_retry_count = 0
    residual_satisfied_retry_count = 0
    tiny_increment_residual_satisfied_retry_count = 0
    min_residual_satisfied_retry_relative_change_ratio = Inf
    min_tiny_increment_residual_satisfied_retry_relative_change_ratio = Inf
    max_tiny_plateau_iteration_count = 0
    max_tiny_plateau_line_search_trial_count = 0
    max_tiny_plateau_line_search_backtrack_count = 0
    max_tiny_plateau_residual_satisfied_retry_count = 0
    for step in load_history
        accepted = get(step, "accepted", false)
        is_tiny_step = accepted && Float64(get(step, "load_increment", Inf)) <= tiny_increment_threshold
        is_tiny_cutback_step = is_tiny_step && Int(get(step, "cutback_count", 0)) > 0
        step_number = Int(get(step, "step", 0))
        step_load_scale = Float64(get(step, "load_scale", 0.0))
        step_load_increment = Float64(get(step, "load_increment", 0.0))
        step_next_increment = Float64(get(step, "next_load_increment", 0.0))
        step_next_increment_reason = String(get(step, "next_increment_reason", ""))
        step_next_increment_ratio =
            accepted && haskey(step, "next_increment_ratio") ?
            Float64(get(step, "next_increment_ratio", 0.0)) :
            0.0
        step_recovery_reference_ratio =
            accepted && haskey(step, "recovery_reference_ratio") ?
            Float64(get(step, "recovery_reference_ratio", 0.0)) :
            0.0
        step_iteration_count = 0
        step_line_search_trial_count = 0
        step_line_search_backtrack_count = 0
        step_residual_satisfied_retry_count = 0
        step_min_residual_satisfied_retry_relative_change_ratio = Inf
        for attempt in get(step, "attempts", Any[])
            iter_records = get(attempt, "iterations", Any[])
            step_iteration_count += length(iter_records)
            for iter_record in iter_records
                n_trials = length(get(iter_record, "line_search_trials", Any[]))
                step_line_search_trial_count += n_trials
                step_line_search_backtrack_count += max(n_trials - 1, 0)
            end
            if String(get(attempt, "termination_reason", "")) == "cutback_retry" &&
               Float64(get(attempt, "final_relative_residual", Inf)) < residual_tol
                step_residual_satisfied_retry_count += 1
                step_min_residual_satisfied_retry_relative_change_ratio = min(
                    step_min_residual_satisfied_retry_relative_change_ratio,
                    Float64(get(attempt, "final_relative_change", Inf)) / max(tol, 1e-30),
                )
            end
        end
        residual_satisfied_retry_count += step_residual_satisfied_retry_count
        if isfinite(step_min_residual_satisfied_retry_relative_change_ratio)
            min_residual_satisfied_retry_relative_change_ratio = min(
                min_residual_satisfied_retry_relative_change_ratio,
                step_min_residual_satisfied_retry_relative_change_ratio,
            )
        end

        if is_tiny_step
            current_tiny_increment_streak += 1
            if current_tiny_increment_streak == 1
                tiny_increment_plateau_count += 1
                current_tiny_plateau_start_step = step_number
                current_tiny_plateau_entry_load_scale = max(step_load_scale - step_load_increment, 0.0)
                current_tiny_plateau_entry_increment = step_load_increment
                current_tiny_plateau_cutback_step_count = 0
                current_tiny_plateau_max_cutback_recovery_ratio = 0.0
                current_tiny_plateau_max_recovery_reference_ratio = 0.0
                current_tiny_plateau_iteration_count = 0
                current_tiny_plateau_line_search_trial_count = 0
                current_tiny_plateau_line_search_backtrack_count = 0
                current_tiny_plateau_residual_satisfied_retry_count = 0
                current_tiny_plateau_min_residual_satisfied_retry_relative_change_ratio = Inf
            end
            current_tiny_plateau_end_step = step_number
            current_tiny_plateau_exit_load_scale = step_load_scale
            current_tiny_plateau_exit_increment = step_load_increment
            current_tiny_plateau_exit_next_increment = step_next_increment
            current_tiny_plateau_exit_next_reason = step_next_increment_reason
            is_tiny_cutback_step && (current_tiny_plateau_cutback_step_count += 1)
            current_tiny_plateau_iteration_count += step_iteration_count
            current_tiny_plateau_line_search_trial_count += step_line_search_trial_count
            current_tiny_plateau_line_search_backtrack_count += step_line_search_backtrack_count
            current_tiny_plateau_residual_satisfied_retry_count += step_residual_satisfied_retry_count
            if isfinite(step_min_residual_satisfied_retry_relative_change_ratio)
                current_tiny_plateau_min_residual_satisfied_retry_relative_change_ratio = min(
                    current_tiny_plateau_min_residual_satisfied_retry_relative_change_ratio,
                    step_min_residual_satisfied_retry_relative_change_ratio,
                )
            end
            if step_next_increment_reason == "cutback_recovery"
                current_tiny_plateau_max_cutback_recovery_ratio = max(
                    current_tiny_plateau_max_cutback_recovery_ratio,
                    step_next_increment_ratio,
                )
            end
            current_tiny_plateau_max_recovery_reference_ratio = max(
                current_tiny_plateau_max_recovery_reference_ratio,
                step_recovery_reference_ratio,
            )
            max_consecutive_tiny_increment_steps = max(max_consecutive_tiny_increment_steps, current_tiny_increment_streak)
        else
            if current_tiny_increment_streak > 0
                push!(tiny_increment_plateaus, Dict(
                    "start_step" => current_tiny_plateau_start_step,
                    "end_step" => current_tiny_plateau_end_step,
                    "step_count" => current_tiny_increment_streak,
                    "cutback_step_count" => current_tiny_plateau_cutback_step_count,
                    "nominal_recovery_exit" => current_tiny_plateau_exit_next_reason == "nominal_schedule",
                    "entry_load_scale" => current_tiny_plateau_entry_load_scale,
                    "exit_load_scale" => current_tiny_plateau_exit_load_scale,
                    "entry_load_increment" => current_tiny_plateau_entry_increment,
                    "exit_load_increment" => current_tiny_plateau_exit_increment,
                    "exit_next_increment" => current_tiny_plateau_exit_next_increment,
                    "exit_next_increment_reason" => current_tiny_plateau_exit_next_reason,
                    "max_cutback_recovery_ratio" => current_tiny_plateau_max_cutback_recovery_ratio,
                    "max_recovery_reference_ratio" => current_tiny_plateau_max_recovery_reference_ratio,
                    "iteration_count" => current_tiny_plateau_iteration_count,
                    "line_search_trial_count" => current_tiny_plateau_line_search_trial_count,
                    "line_search_backtrack_count" => current_tiny_plateau_line_search_backtrack_count,
                    "residual_satisfied_retry_count" => current_tiny_plateau_residual_satisfied_retry_count,
                    "min_residual_satisfied_retry_relative_change_ratio" =>
                        isfinite(current_tiny_plateau_min_residual_satisfied_retry_relative_change_ratio) ?
                        current_tiny_plateau_min_residual_satisfied_retry_relative_change_ratio : 0.0,
                    "load_scale_span" => max(current_tiny_plateau_exit_load_scale - current_tiny_plateau_entry_load_scale, 0.0),
                ))
                tiny_increment_plateau_iteration_count += current_tiny_plateau_iteration_count
                tiny_increment_plateau_line_search_trial_count += current_tiny_plateau_line_search_trial_count
                tiny_increment_plateau_line_search_backtrack_count += current_tiny_plateau_line_search_backtrack_count
                tiny_increment_plateau_residual_satisfied_retry_count += current_tiny_plateau_residual_satisfied_retry_count
                max_tiny_plateau_iteration_count = max(max_tiny_plateau_iteration_count, current_tiny_plateau_iteration_count)
                max_tiny_plateau_line_search_trial_count = max(max_tiny_plateau_line_search_trial_count, current_tiny_plateau_line_search_trial_count)
                max_tiny_plateau_line_search_backtrack_count = max(max_tiny_plateau_line_search_backtrack_count, current_tiny_plateau_line_search_backtrack_count)
                max_tiny_plateau_residual_satisfied_retry_count = max(max_tiny_plateau_residual_satisfied_retry_count, current_tiny_plateau_residual_satisfied_retry_count)
            end
            current_tiny_increment_streak = 0
            current_tiny_plateau_start_step = 0
            current_tiny_plateau_end_step = 0
            current_tiny_plateau_entry_load_scale = 0.0
            current_tiny_plateau_exit_load_scale = 0.0
            current_tiny_plateau_entry_increment = 0.0
            current_tiny_plateau_exit_increment = 0.0
            current_tiny_plateau_exit_next_increment = 0.0
            current_tiny_plateau_exit_next_reason = ""
            current_tiny_plateau_cutback_step_count = 0
            current_tiny_plateau_max_cutback_recovery_ratio = 0.0
            current_tiny_plateau_max_recovery_reference_ratio = 0.0
            current_tiny_plateau_iteration_count = 0
            current_tiny_plateau_line_search_trial_count = 0
            current_tiny_plateau_line_search_backtrack_count = 0
            current_tiny_plateau_residual_satisfied_retry_count = 0
            current_tiny_plateau_min_residual_satisfied_retry_relative_change_ratio = Inf
        end

        if is_tiny_cutback_step
            current_tiny_cutback_streak += 1
        else
            current_tiny_cutback_streak = 0
        end
        max_consecutive_tiny_cutback_steps = max(max_consecutive_tiny_cutback_steps, current_tiny_cutback_streak)

        if accepted && haskey(step, "next_increment_ratio")
            max_next_increment_ratio = max(max_next_increment_ratio, Float64(get(step, "next_increment_ratio", 0.0)))
        end
        if accepted && get(step, "next_increment_reason", "") == "cutback_recovery" && haskey(step, "next_increment_ratio")
            ratio = Float64(get(step, "next_increment_ratio", 0.0))
            max_cutback_recovery_ratio = max(max_cutback_recovery_ratio, ratio)
            if haskey(step, "cutback_recovery_contraction_ratio")
                max_cutback_recovery_contraction_ratio = max(
                    max_cutback_recovery_contraction_ratio,
                    Float64(get(step, "cutback_recovery_contraction_ratio", 0.0)),
                )
            end
            if haskey(step, "cutback_recovery_proposed_ratio")
                max_cutback_recovery_proposed_ratio = max(
                    max_cutback_recovery_proposed_ratio,
                    Float64(get(step, "cutback_recovery_proposed_ratio", 0.0)),
                )
            end
            is_tiny_step && (max_tiny_increment_recovery_ratio = max(max_tiny_increment_recovery_ratio, ratio))
        end

        if get(step, "accepted", false) &&
           Float64(get(step, "load_increment", Inf)) <= tiny_increment_threshold
            tiny_increment_step_count += 1
            Int(get(step, "cutback_count", 0)) > 0 && (tiny_increment_cutback_step_count += 1)
            step_next_increment_reason == "nominal_schedule" &&
                (tiny_increment_nominal_recovery_count += 1)
            tiny_increment_residual_satisfied_retry_count += step_residual_satisfied_retry_count
            if isfinite(step_min_residual_satisfied_retry_relative_change_ratio)
                min_tiny_increment_residual_satisfied_retry_relative_change_ratio = min(
                    min_tiny_increment_residual_satisfied_retry_relative_change_ratio,
                    step_min_residual_satisfied_retry_relative_change_ratio,
                )
            end
        end
        for attempt in get(step, "attempts", Any[])
            for iter_record in get(attempt, "iterations", Any[])
                reason = String(get(iter_record, "accepted_by", ""))
                if haskey(line_search_acceptance_counts, reason)
                    line_search_acceptance_counts[reason] += 1
                end
                get(iter_record, "residual_reduced", false) === true && (iterations_with_residual_reduction += 1)
                get(iter_record, "internal_force_reduced", false) === true && (iterations_with_internal_force_reduction += 1)
                get(iter_record, "potential_reduced", false) === true && (iterations_with_potential_reduction += 1)
                accepted_trial_idx = Int(get(iter_record, "accepted_trial_index", 0))
                minimum_merit_trial_idx = Int(get(iter_record, "minimum_merit_trial_index", accepted_trial_idx))
                minimum_acceptable_trial_idx = Int(get(iter_record, "minimum_acceptable_trial_index", accepted_trial_idx))
                accepted_trial_idx != minimum_merit_trial_idx &&
                    (iterations_with_nonminimum_merit_acceptance += 1)
                accepted_trial_idx != minimum_acceptable_trial_idx &&
                    (iterations_with_nonminimum_acceptable_acceptance += 1)
                minimum_acceptable_trial_idx != minimum_merit_trial_idx &&
                    (iterations_with_minimum_acceptable_distinct_from_minimum_merit += 1)
                max_minimum_merit_trial_gap = max(
                    max_minimum_merit_trial_gap,
                    abs(minimum_merit_trial_idx - accepted_trial_idx),
                )
                max_minimum_acceptable_trial_gap = max(
                    max_minimum_acceptable_trial_gap,
                    abs(minimum_acceptable_trial_idx - accepted_trial_idx),
                )
                max_minimum_acceptable_vs_merit_trial_gap = max(
                    max_minimum_acceptable_vs_merit_trial_gap,
                    abs(minimum_acceptable_trial_idx - minimum_merit_trial_idx),
                )
                n_trials = length(get(iter_record, "line_search_trials", Any[]))
                line_search_trial_count += n_trials
                line_search_backtrack_count += max(n_trials - 1, 0)
            end
        end
    end
    if current_tiny_increment_streak > 0
        push!(tiny_increment_plateaus, Dict(
            "start_step" => current_tiny_plateau_start_step,
            "end_step" => current_tiny_plateau_end_step,
            "step_count" => current_tiny_increment_streak,
            "cutback_step_count" => current_tiny_plateau_cutback_step_count,
            "nominal_recovery_exit" => current_tiny_plateau_exit_next_reason == "nominal_schedule",
            "entry_load_scale" => current_tiny_plateau_entry_load_scale,
            "exit_load_scale" => current_tiny_plateau_exit_load_scale,
            "entry_load_increment" => current_tiny_plateau_entry_increment,
            "exit_load_increment" => current_tiny_plateau_exit_increment,
            "exit_next_increment" => current_tiny_plateau_exit_next_increment,
            "exit_next_increment_reason" => current_tiny_plateau_exit_next_reason,
            "max_cutback_recovery_ratio" => current_tiny_plateau_max_cutback_recovery_ratio,
            "max_recovery_reference_ratio" => current_tiny_plateau_max_recovery_reference_ratio,
            "iteration_count" => current_tiny_plateau_iteration_count,
            "line_search_trial_count" => current_tiny_plateau_line_search_trial_count,
            "line_search_backtrack_count" => current_tiny_plateau_line_search_backtrack_count,
            "residual_satisfied_retry_count" => current_tiny_plateau_residual_satisfied_retry_count,
            "min_residual_satisfied_retry_relative_change_ratio" =>
                isfinite(current_tiny_plateau_min_residual_satisfied_retry_relative_change_ratio) ?
                current_tiny_plateau_min_residual_satisfied_retry_relative_change_ratio : 0.0,
            "load_scale_span" => max(current_tiny_plateau_exit_load_scale - current_tiny_plateau_entry_load_scale, 0.0),
        ))
        tiny_increment_plateau_iteration_count += current_tiny_plateau_iteration_count
        tiny_increment_plateau_line_search_trial_count += current_tiny_plateau_line_search_trial_count
        tiny_increment_plateau_line_search_backtrack_count += current_tiny_plateau_line_search_backtrack_count
        tiny_increment_plateau_residual_satisfied_retry_count += current_tiny_plateau_residual_satisfied_retry_count
        max_tiny_plateau_iteration_count = max(max_tiny_plateau_iteration_count, current_tiny_plateau_iteration_count)
        max_tiny_plateau_line_search_trial_count = max(max_tiny_plateau_line_search_trial_count, current_tiny_plateau_line_search_trial_count)
        max_tiny_plateau_line_search_backtrack_count = max(max_tiny_plateau_line_search_backtrack_count, current_tiny_plateau_line_search_backtrack_count)
        max_tiny_plateau_residual_satisfied_retry_count = max(max_tiny_plateau_residual_satisfied_retry_count, current_tiny_plateau_residual_satisfied_retry_count)
    end
    converged = !terminated_early && !isempty(load_history) && current_load_scale >= 1.0 - 1e-12 &&
        all(get(step, "converged", false) for step in load_history)
    sub_res["nonlinear_diagnostics"] = Dict(
        "scheme" => nonlinear_method === :formal_shell_von_karman ?
            "formal_shell_von_karman_line_search" :
            "experimental_geometric_residual_line_search",
        "nonlinear_method" => String(nonlinear_method),
        "load_steps" => load_history,
        "requested_load_step_count" => load_steps,
        "load_step_count" => length(load_history),
        "accepted_load_step_count" => count(step -> get(step, "accepted", false), load_history),
        "base_load_increment" => nominal_increment,
        "max_iterations_per_step" => max_iter,
        "tolerance" => tol,
        "residual_tolerance" => residual_tol,
        "residual_model" => nonlinear_method === :formal_shell_von_karman ? "formal_internal_force" : String(residual_model),
        "correction_tangent_model" => nonlinear_method === :formal_shell_von_karman ? "formal_consistent_tangent" : "linear_plus_geometric",
        "relaxation" => relaxation,
        "line_search_max_backtracks" => line_search_max_backtracks,
        "line_search_reduction" => line_search_reduction,
        "max_cutbacks" => max_cutbacks,
        "cutback_reduction" => cutback_reduction,
        "step_growth" => step_growth,
        "formal_shell_supported" => formal_supported,
        "formal_support_class" => formal_support.support_class,
        "formal_support_unsupported_reason" =>
            isempty(formal_support.unsupported_reason) ? missing : formal_support.unsupported_reason,
        "formal_support_total_shell_count" => formal_support.total_shell_count,
        "formal_support_quad4_count" => formal_support.quad4_count,
        "formal_support_tria3_count" => formal_support.tria3_count,
        "formal_support_planar_quad_count" => formal_support.planar_quad_count,
        "formal_support_warped_quad_count" => formal_support.warped_quad_count,
        "final_formal_diagnostics" => final_formal_diagnostics,
        "final_potential_energy" => isnothing(final_potential_energy) ? missing : final_potential_energy,
        "final_internal_energy" => isnothing(final_internal_energy) ? missing : final_internal_energy,
        "state_evaluation_count" => state_evaluation_count,
        "correction_partition_reuse_count" => correction_partition_reuse_count,
        "accepted_state_reuse_count" => accepted_state_reuse_count,
        "step_growth_applied_count" => step_growth_applied_count,
        "line_search_acceptance_counts" => line_search_acceptance_counts,
        "line_search_trial_count" => line_search_trial_count,
        "line_search_backtrack_count" => line_search_backtrack_count,
        "iterations_with_residual_reduction" => iterations_with_residual_reduction,
        "iterations_with_internal_force_reduction" => iterations_with_internal_force_reduction,
        "iterations_with_potential_reduction" => iterations_with_potential_reduction,
        "iterations_with_nonminimum_merit_acceptance" => iterations_with_nonminimum_merit_acceptance,
        "iterations_with_nonminimum_acceptable_acceptance" => iterations_with_nonminimum_acceptable_acceptance,
        "iterations_with_minimum_acceptable_distinct_from_minimum_merit" =>
            iterations_with_minimum_acceptable_distinct_from_minimum_merit,
        "max_minimum_merit_trial_gap" => max_minimum_merit_trial_gap,
        "max_minimum_acceptable_trial_gap" => max_minimum_acceptable_trial_gap,
        "max_minimum_acceptable_vs_merit_trial_gap" => max_minimum_acceptable_vs_merit_trial_gap,
        "tiny_increment_threshold" => tiny_increment_threshold,
        "tiny_increment_step_count" => tiny_increment_step_count,
        "tiny_increment_cutback_step_count" => tiny_increment_cutback_step_count,
        "tiny_increment_nominal_recovery_count" => tiny_increment_nominal_recovery_count,
        "tiny_increment_plateau_count" => tiny_increment_plateau_count,
        "tiny_increment_plateaus" => tiny_increment_plateaus,
        "tiny_increment_plateau_iteration_count" => tiny_increment_plateau_iteration_count,
        "tiny_increment_plateau_line_search_trial_count" => tiny_increment_plateau_line_search_trial_count,
        "tiny_increment_plateau_line_search_backtrack_count" => tiny_increment_plateau_line_search_backtrack_count,
        "tiny_increment_plateau_residual_satisfied_retry_count" => tiny_increment_plateau_residual_satisfied_retry_count,
        "max_consecutive_tiny_increment_steps" => max_consecutive_tiny_increment_steps,
        "max_consecutive_tiny_cutback_steps" => max_consecutive_tiny_cutback_steps,
        "max_tiny_plateau_iteration_count" => max_tiny_plateau_iteration_count,
        "max_tiny_plateau_line_search_trial_count" => max_tiny_plateau_line_search_trial_count,
        "max_tiny_plateau_line_search_backtrack_count" => max_tiny_plateau_line_search_backtrack_count,
        "max_tiny_plateau_residual_satisfied_retry_count" => max_tiny_plateau_residual_satisfied_retry_count,
        "max_next_increment_ratio" => max_next_increment_ratio,
        "max_cutback_recovery_ratio" => max_cutback_recovery_ratio,
        "max_cutback_recovery_contraction_ratio" => max_cutback_recovery_contraction_ratio,
        "max_cutback_recovery_proposed_ratio" => max_cutback_recovery_proposed_ratio,
        "max_tiny_increment_recovery_ratio" => max_tiny_increment_recovery_ratio,
        "residual_satisfied_retry_count" => residual_satisfied_retry_count,
        "tiny_increment_residual_satisfied_retry_count" => tiny_increment_residual_satisfied_retry_count,
        "min_residual_satisfied_retry_relative_change_ratio" =>
            isfinite(min_residual_satisfied_retry_relative_change_ratio) ?
            min_residual_satisfied_retry_relative_change_ratio : 0.0,
        "min_tiny_increment_residual_satisfied_retry_relative_change_ratio" =>
            isfinite(min_tiny_increment_residual_satisfied_retry_relative_change_ratio) ?
            min_tiny_increment_residual_satisfied_retry_relative_change_ratio : 0.0,
        "converged" => converged,
        "termination_reason" => converged ? "full_load_converged" :
            (terminated_early ? "cutback_exhausted" : "stopped_before_full_load"),
        "final_load_scale" => current_load_scale,
        "final_relative_change" => isempty(load_history) ? Inf : get(last(load_history), "final_relative_change", Inf),
        "final_relative_residual" => isempty(load_history) ? Inf : get(last(load_history), "final_relative_residual", Inf),
        "final_relative_incremental_work" =>
            isempty(load_history) ? Inf : get(last(load_history), "final_relative_incremental_work", Inf),
        "final_kg_nnz" => nnz(Kg),
        "final_kg_norm" => norm(Kg.nzval),
    )
    return u_out, stresses, sub_res, u_analysis, fixed_dofs, Kg
end

# =============================================================================
# SOL105 BUCKLING EIGENVALUE SOLVER
# Solves: [K + lambda * Kg] * phi = 0  →  K*phi = lambda*(-Kg)*phi
# =============================================================================
function _buckling_krylovdim(nev_request::Int, n_free::Int)
    factor = max(solver_env_float("JFEM_SOL105_KRYLOV_DIM_FACTOR", 2.0), 1.0)
    offset = max(solver_env_int("JFEM_SOL105_KRYLOV_DIM_OFFSET", 20), 0)
    min_dim = max(solver_env_int("JFEM_SOL105_KRYLOV_DIM_MIN", 40), 1)
    kd = max(ceil(Int, factor * nev_request) + offset, min_dim, nev_request + 2)
    return min(kd, n_free)
end

function _buckling_krylovtol()
    return max(solver_env_float("JFEM_SOL105_KRYLOV_TOL", 1e-12), 0.0)
end

function _buckling_krylovmaxiter()
    return max(solver_env_int("JFEM_SOL105_KRYLOV_MAXITER", 1000), 1)
end

function solve_buckling(K, Kg, ndof, model, id_map, X, spc_id, node_R, num_modes;
                        rbe3_map=Dict{Int,Vector{Tuple{Int,Float64}}}(),
                        max_elem_stiff=0.0, orig_diag=Float64[],
                        eigrl_v1::Float64=0.0, eigrl_v2::Float64=0.0,
                        eigen_cache=nothing,
                        buckling_subcase=nothing,
                        static_subcase=nothing,
                        return_diagnostics::Bool=false)

    t_buckling_total = time_ns()
    buckling_timings = Dict{String,Any}()
    log_msg("[BUCKLING] Computing free DOFs...")
    t_prepare_context = time_ns()
    eigen_ctx, eigen_cache_hit = prepare_eigen_solve_context(
        K, ndof, model, id_map, spc_id, rbe3_map; eigen_cache=eigen_cache)
    buckling_timings["prepare_context"] = (time_ns() - t_prepare_context) * 1e-9
    free_dofs = eigen_ctx.free_dofs
    fixed_dofs = eigen_ctx.fixed_dofs
    bc_diagnostics = eigen_ctx.bc_diagnostics
    n_free = length(free_dofs)
    log_msg("[BUCKLING] Free DOFs: $n_free, Fixed DOFs: $(length(fixed_dofs))")
    if eigen_cache_hit
        log_msg("[BUCKLING] Reusing eigen BC partition cache: Fixed DOFs=$(length(fixed_dofs)), Free DOFs=$n_free")
    end
    diagnostics = Dict{String,Any}(
        "bc_partition" => deepcopy(bc_diagnostics),
        "requested_modes" => num_modes,
        "requested_modes_internal" => 0,
        "free_dofs" => n_free,
        "fixed_dofs" => length(fixed_dofs),
        "eigrl_range" => Dict("v1" => eigrl_v1, "v2" => eigrl_v2),
        "eigen_cache" => Dict{String,Any}(
            "enabled" => eigen_cache !== nothing,
            "cache_hit" => eigen_cache_hit,
            "factorization_cache_hit" => false,
        ),
        "solver_backend" => "unsolved",
        "solver_attempts" => Any[],
        "returned_modes" => 0,
        "mode_shapes_omitted" => false,
    )

    t_slice = time_ns()
    K_ff  = eigen_ctx.K_ff
    Kg_ff = Kg[free_dofs, free_dofs]
    buckling_timings["slice_free_matrices"] = (time_ns() - t_slice) * 1e-9

    t_symmetry = time_ns()
    # K_ff is symmetrized once when the eigen solve context is prepared/cached.
    # Only Kg changes per buckling subcase. The asymmetry diagnostic is useful
    # while developing new operators, but it builds another sparse matrix, so
    # production batch runs can disable it without changing the solved matrix.
    asymmetry_check = solver_env_bool("JFEM_MATRIX_ASYMMETRY_CHECK", true)
    if asymmetry_check
        Kg_norm_inf = max(norm(Kg_ff, Inf), 1e-30)
        K_asym_rel = 0.0
        Kg_asym_rel = norm(Kg_ff - Kg_ff', Inf) / Kg_norm_inf
        diagnostics["matrix_asymmetry"] = Dict(
            "checked" => true,
            "K_inf_rel" => K_asym_rel,
            "Kg_inf_rel" => Kg_asym_rel,
        )
        asym_warn_rel = solver_env_float("JFEM_MATRIX_ASYMMETRY_WARN_REL", 1e-10)
        if K_asym_rel > asym_warn_rel || Kg_asym_rel > asym_warn_rel
            log_msg("[BUCKLING] Matrix asymmetry before symmetrization: K=$(K_asym_rel), Kg=$(Kg_asym_rel)")
        end
    else
        diagnostics["matrix_asymmetry"] = Dict(
            "checked" => false,
            "K_inf_rel" => nothing,
            "Kg_inf_rel" => nothing,
        )
    end

    # Symmetrize after recording diagnostics; the generalized buckling solver
    # expects the conservative symmetric Hessian pair.
    Kg_ff = 0.5 * (Kg_ff + Kg_ff')
    buckling_timings["symmetry_checks"] = (time_ns() - t_symmetry) * 1e-9
    start_vector_ordinal = Ref(0)
    next_start_vector() = begin
        start_vector_ordinal[] += 1
        _deterministic_buckling_start_vector(n_free, start_vector_ordinal[])
    end

    # When EIGRL range is specified, request extra modes to allow filtering.
    # For dense systems (≤4000 DOFs), all eigenvalues are computed anyway — return 3× modes
    # so the comparison can use subset matching to skip spurious bar/shell modes.
    has_range = (eigrl_v1 != 0.0 || eigrl_v2 != 0.0) && eigrl_v2 > eigrl_v1
    return_all_range = has_range && solver_env_bool("JFEM_SOL105_RETURN_ALL_IN_RANGE", false)
    nd_limited_range_output = has_range && !return_all_range
    will_use_dense = n_free <= 4000
    range_mode_factor_default = nd_limited_range_output ? 3.0 : 4.0
    range_mode_factor = max(solver_env_float("JFEM_SOL105_RANGE_MODE_FACTOR", range_mode_factor_default), 1.0)
    num_modes_request = has_range ? ceil(Int, num_modes * range_mode_factor) :
                        (will_use_dense ? num_modes * 3 : num_modes)

    # Clamp num_modes to system size
    max_modes = max(n_free - 2, 1)
    if num_modes_request > max_modes
        log_msg("[BUCKLING] Reducing num_modes_request from $num_modes_request to $max_modes (system size limit)")
        num_modes_request = max_modes
    end
    diagnostics["requested_modes_internal"] = num_modes_request
    diagnostics["range_mode_factor"] = has_range ? range_mode_factor : nothing
    diagnostics["range_nd_limited_output"] = nd_limited_range_output

    log_msg("[BUCKLING] Solving eigenvalue problem ($num_modes modes, $n_free DOFs)...")

    local eigenvalues, eigenvectors
    solved = false
    t_eigen_search = time_ns()

    function attempt_shifted_buckling_search(sigma::Float64, attempt_name::String)
        push!(diagnostics["solver_attempts"], Dict("name" => attempt_name, "status" => "attempted", "sigma" => sigma))
        try
            log_msg("[BUCKLING] Range-targeted shift-invert at sigma=$sigma ...")
            B = -Kg_ff
            M = K_ff - sigma * B
            M = 0.5 * (M + M')

            factor_backend = "cholesky"
            M_factor =
                try
                    cholesky(M)
                catch
                    factor_backend = "lu"
                    lu(M)
                end

            nd_limited_range_augmentation =
                nd_limited_range_output &&
                solver_env_bool("JFEM_SOL105_RANGE_AUGMENTATION_ND_LIMIT", true)
            range_aug_buffer_default = nd_limited_range_augmentation ? 8 : 16
            range_aug_buffer = max(solver_env_int("JFEM_SOL105_RANGE_AUGMENTATION_BUFFER", range_aug_buffer_default), 0)
            shifted_modes_request = nd_limited_range_augmentation ?
                min(num_modes_request, max(num_modes + range_aug_buffer, num_modes)) :
                num_modes_request
            nev_request = min(shifted_modes_request + 5, n_free - 1)
            kd = _buckling_krylovdim(nev_request, n_free)
            krylov_tol = _buckling_krylovtol()
            krylov_maxiter = _buckling_krylovmaxiter()
            vals_kk, vecs_kk, info = eigsolve(
                x -> M_factor \ (B * x), next_start_vector(), nev_request, :LM;
                krylovdim=kd, maxiter=krylov_maxiter, tol=krylov_tol, eager=true)

            actual_lambdas = Float64[]
            actual_vecs = Vector{Float64}[]
            for (i, theta) in enumerate(vals_kk)
                theta_r = real(theta)
                theta_i = abs(imag(theta))
                if theta_i > 1e-6 * max(abs(theta_r), 1e-20) || abs(theta_r) < 1e-14
                    continue
                end
                lam = sigma + 1.0 / theta_r
                if abs(lam) > 1e-6
                    push!(actual_lambdas, lam)
                    push!(actual_vecs, real.(vecs_kk[i]))
                end
            end

            if isempty(actual_lambdas)
                diagnostics["solver_attempts"][end] = Dict(
                    "name" => attempt_name,
                    "status" => "no_valid_eigenvalues",
                    "sigma" => sigma,
                    "converged" => info.converged,
                    "factorization" => factor_backend,
                )
                return nothing
            end

            perm = sortperm(abs.(actual_lambdas .- sigma))
            n_out = min(shifted_modes_request, length(perm))
            lambdas = [actual_lambdas[perm[i]] for i in 1:n_out]
            vecs = hcat([actual_vecs[perm[i]] for i in 1:n_out]...)

            diagnostics["solver_attempts"][end] = Dict(
                "name" => attempt_name,
                "status" => "succeeded",
                "sigma" => sigma,
                "returned_modes" => n_out,
                "requested_modes_internal" => shifted_modes_request,
                "nd_limited_range_output" => nd_limited_range_augmentation,
                "range_augmentation_buffer" => range_aug_buffer,
                "krylovdim" => kd,
                "krylov_tol" => krylov_tol,
                "krylov_maxiter" => krylov_maxiter,
                "converged" => info.converged,
                "factorization" => factor_backend,
            )
            return lambdas, vecs
        catch e
            diagnostics["solver_attempts"][end] = Dict(
                "name" => attempt_name,
                "status" => "failed",
                "sigma" => sigma,
                "error" => sprint(showerror, e),
            )
            log_msg("[BUCKLING] Range-targeted shift-invert failed: $(sprint(showerror, e))")
            return nothing
        end
    end

    function merge_unique_eigenpairs(base_vals::Vector{Float64}, base_vecs::AbstractMatrix,
                                     add_vals::Vector{Float64}, add_vecs::AbstractMatrix;
                                     rel_tol::Float64=1e-6, abs_tol::Float64=1e-8)
        merged_vals = copy(base_vals)
        merged_vec_cols = [Vector{Float64}(base_vecs[:, i]) for i in 1:size(base_vecs, 2)]
        added = 0
        for j in eachindex(add_vals)
            lam = Float64(add_vals[j])
            duplicate = any(existing ->
                abs(existing - lam) <= max(abs_tol, rel_tol * max(abs(existing), abs(lam), 1.0)),
                merged_vals)
            if duplicate
                continue
            end
            push!(merged_vals, lam)
            push!(merged_vec_cols, Vector{Float64}(add_vecs[:, j]))
            added += 1
        end
        merged_vecs = isempty(merged_vec_cols) ? zeros(Float64, size(base_vecs, 1), 0) : hcat(merged_vec_cols...)
        return merged_vals, merged_vecs, added
    end

    function range_augmentation_sigmas(range_idx_current)
        raw = strip(get(ENV, "JFEM_SOL105_RANGE_AUGMENTATION_SIGMAS", ""))
        sigmas = Float64[]
        if !isempty(raw)
            for part in split(raw, ",")
                sigma = tryparse(Float64, strip(part))
                sigma === nothing && continue
                if sigma >= eigrl_v1 && sigma <= eigrl_v2
                    push!(sigmas, sigma)
                end
            end
        elseif solver_env_bool("JFEM_SOL105_RANGE_AUGMENTATION_MULTI", false)
            count_raw = strip(get(ENV, "JFEM_SOL105_RANGE_AUGMENTATION_MULTI_COUNT", "4"))
            n_mid = tryparse(Int, count_raw)
            n_mid = n_mid === nothing ? 4 : clamp(n_mid, 1, 12)
            positives = [eigenvalues[i] for i in range_idx_current if eigenvalues[i] > max(eigrl_v1, 0.0)]
            low = isempty(positives) ? max(eigrl_v1, eps(Float64)) : maximum(positives)
            low = max(low, eps(Float64))
            front_factor_raw = strip(get(ENV, "JFEM_SOL105_RANGE_AUGMENTATION_FRONT_FACTOR", "1.25"))
            front_factor = tryparse(Float64, front_factor_raw)
            front_factor = front_factor === nothing ? 1.25 : clamp(front_factor, 1.01, 10.0)
            front = min(eigrl_v2, low * front_factor)
            if front > low * (1.0 + 1e-8)
                push!(sigmas, front)
            end
            if eigrl_v2 > front * (1.0 + 1e-8)
                ratio = (eigrl_v2 / front)^(1.0 / n_mid)
                for k in 1:n_mid
                    push!(sigmas, front * ratio^k)
                end
            else
                push!(sigmas, eigrl_v2)
            end
        else
            push!(sigmas, eigrl_v2)
        end

        push!(sigmas, eigrl_v2)
        sort!(sigmas)
        unique_sigmas = Float64[]
        for sigma in sigmas
            if sigma < eigrl_v1 || sigma > eigrl_v2
                continue
            end
            if isempty(unique_sigmas) ||
               abs(sigma - unique_sigmas[end]) > max(1e-8, 1e-6 * max(abs(sigma), abs(unique_sigmas[end]), 1.0))
                push!(unique_sigmas, sigma)
            end
        end
        return unique_sigmas
    end

    # Strategy 1: Dense symmetric-definite eigensolver for small-medium systems.
    # K is positive definite after SPC elimination, so solve the Cholesky-reduced
    # symmetric problem C*y = θ*y with C = L⁻¹*B*L⁻ᵀ, θ = 1/λ, B = -Kg.
    if n_free <= 4000
        push!(diagnostics["solver_attempts"], Dict("name" => "dense_symmetric_definite", "status" => "attempted"))
        try
            log_msg("[BUCKLING] Using dense symmetric-definite eigensolver ($n_free DOFs)...")
            Kd = Matrix(K_ff)
            Bgd = Matrix(-Kg_ff)
            K_factor = cholesky(Symmetric(Kd))
            L = Matrix(K_factor.L)
            C = L \ (Bgd / L')
            C = 0.5 * (C + C')
            theta_vals, theta_vecs = eigen(Symmetric(C))
            valid = findall(x -> isfinite(x) && abs(x) > 1e-12, theta_vals)
            if !isempty(valid)
                thetas = theta_vals[valid]
                lambdas = 1.0 ./ thetas
                vecs = K_factor.U \ theta_vecs[:, valid]
                perm = sortperm(abs.(lambdas))
                # Dense solves already computed the whole spectrum. For ranged
                # SOL105 extraction, keep it until the positive/range filter
                # below; indefinite prestress can put many load-reversal roots
                # ahead of the requested positive branch in |lambda| ordering.
                n_out = has_range ? length(perm) : min(num_modes_request, length(perm))
                eigenvalues = real.(lambdas[perm[1:n_out]])
                eigenvectors = real.(vecs[:, perm[1:n_out]])
                solved = true
                diagnostics["solver_backend"] = "dense_symmetric_definite"
                diagnostics["solver_attempts"][end] = Dict("name" => "dense_symmetric_definite", "status" => "succeeded", "returned_modes" => n_out)
                log_msg("[BUCKLING] Dense symmetric-definite eigensolver converged ($n_out modes)")
            else
                diagnostics["solver_attempts"][end] = Dict("name" => "dense_symmetric_definite", "status" => "no_valid_eigenvalues")
                log_msg("[BUCKLING] Dense symmetric-definite eigensolver: no valid eigenvalues found")
            end
        catch e
            diagnostics["solver_attempts"][end] = Dict("name" => "dense_symmetric_definite", "status" => "failed", "error" => sprint(showerror, e))
            log_msg("[BUCKLING] Dense symmetric-definite eigensolver failed: $e")
        end

        if !solved
            push!(diagnostics["solver_attempts"], Dict("name" => "dense_generalized", "status" => "attempted"))
            try
                log_msg("[BUCKLING] Fallback dense generalized eigensolver ($n_free DOFs)...")
                Kd = Matrix(K_ff)
                Bgd = Matrix(-Kg_ff)
                all_vals, all_vecs = eigen(Kd, Bgd)
                valid = findall(x -> isfinite(x) && isreal(x) && abs(real(x)) > 1e-6, all_vals)
                if !isempty(valid)
                    lambdas = real.(all_vals[valid])
                    vecs = real.(all_vecs[:, valid])
                    perm = sortperm(abs.(lambdas))
                    n_out = has_range ? length(perm) : min(num_modes_request, length(perm))
                    eigenvalues = lambdas[perm[1:n_out]]
                    eigenvectors = vecs[:, perm[1:n_out]]
                    solved = true
                    diagnostics["solver_backend"] = "dense_generalized"
                    diagnostics["solver_attempts"][end] = Dict("name" => "dense_generalized", "status" => "succeeded", "returned_modes" => n_out)
                    log_msg("[BUCKLING] Fallback dense generalized eigensolver converged ($n_out modes)")
                else
                    diagnostics["solver_attempts"][end] = Dict("name" => "dense_generalized", "status" => "no_valid_eigenvalues")
                    log_msg("[BUCKLING] Fallback dense generalized eigensolver: no valid eigenvalues found")
                end
            catch e
                diagnostics["solver_attempts"][end] = Dict("name" => "dense_generalized", "status" => "failed", "error" => sprint(showerror, e))
                log_msg("[BUCKLING] Fallback dense generalized eigensolver failed: $e")
            end
        end
    end

    # Strategy 2: KrylovKit inverse iteration (pure Julia, no Fortran deps)
    # Generalized problem: K*x = λ*B*x where B = -Kg
    # Zero-shift inverse iteration: K⁻¹*B*x = θ*x where θ = 1/λ
    # Largest |θ| from KrylovKit → smallest |λ| (lowest buckling load)
    if !solved
        push!(diagnostics["solver_attempts"], Dict("name" => "krylov_inverse_iteration", "status" => "attempted"))
        try
            log_msg("[BUCKLING] Using KrylovKit inverse iteration ($n_free DOFs)...")
            B = -Kg_ff

            # Factorize K (Cholesky when SPD, LU fallback otherwise).
            K_factor, factor_cache_hit = ensure_eigen_solve_factorization!(eigen_ctx)
            diagnostics["eigen_cache"]["factorization_cache_hit"] = factor_cache_hit
            diagnostics["eigen_cache"]["factor_backend"] = eigen_ctx.factor_backend
            if factor_cache_hit
                log_msg("[BUCKLING] Reusing eigen K factorization cache ($(eigen_ctx.factor_backend))")
            else
                log_msg("[BUCKLING] K factorization succeeded ($(eigen_ctx.factor_backend))")
            end

            # Request extra eigenvalues for robustness. Buckling modes on aircraft
            # shell structures often have nearly-degenerate pairs (symmetric/
            # antisymmetric about a plane of near-symmetry); tight tol + eager=false
            # ensures consistent ordering of those pairs across formulation changes.
            nev_request = min(num_modes_request + 5, n_free - 1)
            kd = _buckling_krylovdim(nev_request, n_free)
            krylov_tol = _buckling_krylovtol()
            krylov_maxiter = _buckling_krylovmaxiter()

            # K⁻¹*B operator: find largest magnitude eigenvalues θ = 1/λ
            # tol tightened from 1e-10 → 1e-13 (2026-04-21): HTP_launch has mode-1/2
            # pairs separated by <0.1%; loose convergence lets order wobble between
            # solver runs and between formulation changes, masking as a "bug".
            vals_kk, vecs_kk, info = eigsolve(
                x -> K_factor \ (B * x), next_start_vector(), nev_request, :LM;
                krylovdim=kd, maxiter=krylov_maxiter, tol=krylov_tol, eager=true)

            log_msg("[BUCKLING] KrylovKit returned $(length(vals_kk)) eigenvalues (converged=$(info.converged))")

            # Convert θ → λ = 1/θ
            actual_lambdas = Float64[]
            actual_vecs = Vector{Float64}[]
            for (i, theta) in enumerate(vals_kk)
                theta_r = real(theta)
                theta_i = abs(imag(theta))
                # Skip complex and near-zero eigenvalues
                if theta_i > 1e-6 * max(abs(theta_r), 1e-20) || abs(theta_r) < 1e-14
                    continue
                end
                lam = 1.0 / theta_r
                if abs(lam) > 1e-6
                    push!(actual_lambdas, lam)
                    push!(actual_vecs, real.(vecs_kk[i]))
                end
            end
            if !isempty(actual_lambdas)
                perm = sortperm(abs.(actual_lambdas))
                n_out = min(num_modes_request, length(perm))
                eigenvalues = [actual_lambdas[perm[i]] for i in 1:n_out]
                eigenvectors = hcat([actual_vecs[perm[i]] for i in 1:n_out]...)
                solved = true
                diagnostics["solver_backend"] = "krylov_inverse_iteration"
                diagnostics["solver_attempts"][end] = Dict(
                    "name" => "krylov_inverse_iteration",
                    "status" => "succeeded",
                    "returned_modes" => n_out,
                    "converged" => info.converged,
                    "krylovdim" => kd,
                    "krylov_tol" => krylov_tol,
                    "krylov_maxiter" => krylov_maxiter,
                )
                log_msg("[BUCKLING] KrylovKit converged ($n_out modes)")
                if solver_env_bool("JFEM_BUCKLING_LOG_RAW_EIGENVALUES", false)
                    for lam in eigenvalues
                        log_msg("[BUCKLING]   θ=$(1.0/lam) → λ=$lam")
                    end
                end
                # Diagnostic: residual + Sturm count (env-gated, single case use).
                # JFEM_DEBUG_STURM_SIGMAS=0.1,0.5,1.0,1.2 → print Sturm counts
                # Residuals are always logged when the flag is set.
                if solver_env_bool("JFEM_DEBUG_BUCKLING", false)
                    # Residual per mode: ||K u - λ (-Kg) u|| / ||K u||
                    B_op = -Kg_ff
                    for m in 1:n_out
                        u_m = eigenvectors[:, m]
                        Ku  = K_ff * u_m
                        Bu  = B_op * u_m
                        lam = eigenvalues[m]
                        nrm_Ku = norm(Ku)
                        rel_res = norm(Ku - lam * Bu) / max(nrm_Ku, 1e-30)
                        log_msg("[BUCKLING][DBG] mode $m: λ=$lam, ||r||/||Ku||=$(rel_res)")
                    end
                    sigmas_env = get(ENV, "JFEM_DEBUG_STURM_SIGMAS", "")
                    if !isempty(strip(sigmas_env))
                        for s_str in split(sigmas_env, ",")
                            sigma = tryparse(Float64, strip(s_str))
                            isnothing(sigma) && continue
                            M = K_ff + sigma * Kg_ff
                            M = 0.5 * (M + M')
                            try
                                F = lu(M)
                                n_neg = count(d -> d < 0.0, diag(F.U))
                                log_msg("[BUCKLING][DBG] Sturm count at σ=$sigma: $n_neg negative diag(U) ⇒ ≈$n_neg eigenvalues below σ")
                            catch e
                                log_msg("[BUCKLING][DBG] Sturm count at σ=$sigma: LU failed ($(sprint(showerror, e)))")
                            end
                        end
                    end
                end
            else
                diagnostics["solver_attempts"][end] = Dict("name" => "krylov_inverse_iteration", "status" => "no_valid_eigenvalues", "converged" => info.converged)
                log_msg("[BUCKLING] KrylovKit: no valid real eigenvalues found")
            end
        catch e
            diagnostics["solver_attempts"][end] = Dict("name" => "krylov_inverse_iteration", "status" => "failed", "error" => sprint(showerror, e))
            log_msg("[BUCKLING] KrylovKit inverse iteration failed: $(sprint(showerror, e))")
        end
    end

    # Strategy 3: KrylovKit with shift near first mode (refine if Strategy 2 failed)
    if !solved
        push!(diagnostics["solver_attempts"], Dict("name" => "krylov_shifted", "status" => "attempted"))
        try
            log_msg("[BUCKLING] Fallback: KrylovKit with small shift...")
            B = -Kg_ff
            sigma = 1.0  # small positive shift

            M = K_ff - sigma * B
            M = 0.5 * (M + M')
            local M_factor
            try
                M_factor = cholesky(M)
            catch
                M_factor = lu(M)
            end

            nev_request = min(num_modes_request + 5, n_free - 1)
            kd = _buckling_krylovdim(nev_request, n_free)
            krylov_tol = _buckling_krylovtol()
            krylov_maxiter = _buckling_krylovmaxiter()
            vals_kk, vecs_kk, info = eigsolve(
                x -> M_factor \ (B * x), next_start_vector(), nev_request, :LM;
                krylovdim=kd, maxiter=krylov_maxiter, tol=krylov_tol, eager=true)

            actual_lambdas = Float64[]
            actual_vecs = Vector{Float64}[]
            for (i, theta) in enumerate(vals_kk)
                theta_r = real(theta)
                if abs(imag(theta)) > 1e-6 * max(abs(theta_r), 1e-20) || abs(theta_r) < 1e-14
                    continue
                end
                lam = sigma + 1.0 / theta_r
                if abs(lam) > 1e-6
                    push!(actual_lambdas, lam)
                    push!(actual_vecs, real.(vecs_kk[i]))
                end
            end
            if !isempty(actual_lambdas)
                perm = sortperm(abs.(actual_lambdas))
                n_out = min(num_modes_request, length(perm))
                eigenvalues = [actual_lambdas[perm[i]] for i in 1:n_out]
                eigenvectors = hcat([actual_vecs[perm[i]] for i in 1:n_out]...)
                solved = true
                diagnostics["solver_backend"] = "krylov_shifted"
                diagnostics["solver_attempts"][end] = Dict(
                    "name" => "krylov_shifted",
                    "status" => "succeeded",
                    "returned_modes" => n_out,
                    "converged" => info.converged,
                    "krylovdim" => kd,
                    "krylov_tol" => krylov_tol,
                    "krylov_maxiter" => krylov_maxiter,
                )
                log_msg("[BUCKLING] Fallback KrylovKit converged ($n_out modes)")
            end
        catch e
            diagnostics["solver_attempts"][end] = Dict("name" => "krylov_shifted", "status" => "failed", "error" => sprint(showerror, e))
            log_msg("[BUCKLING] Fallback KrylovKit failed: $(sprint(showerror, e))")
        end
    end

    if !solved
        log_msg("[BUCKLING] ERROR: All eigenvalue solvers failed")
        diagnostics["solver_backend"] = "failed"
        buckling_timings["eigensolver_search"] = (time_ns() - t_eigen_search) * 1e-9
        buckling_timings["total"] = (time_ns() - t_buckling_total) * 1e-9
        diagnostics["timings"] = buckling_timings
        return return_diagnostics ? (Float64[], zeros(ndof, 0), diagnostics) : (Float64[], zeros(ndof, 0))
    end
    buckling_timings["eigensolver_search"] = (time_ns() - t_eigen_search) * 1e-9
    t_postprocess = time_ns()

    # Post-process: per Nastran SOL105 convention, only POSITIVE eigenvalues are
    # physical buckling load factors. Negatives correspond to load reversal and
    # are reported by Nastran as a tensile-direction critical solution but are
    # not part of the EIGRL-requested buckling spectrum on a compressive deck.
    # We drop them outright. The eigenvalue range is then [max(V1, +tol), V2]
    # capped at EIGRL ND.
    n_found = length(eigenvalues)
    log_msg("[BUCKLING] Found $n_found eigenvalues (raw, pre-filter)")

    positive_tol = 1e-10
    raw_eigen_csv_path = strip(get(ENV, "JFEM_BUCKLING_RAW_EIGEN_CSV", ""))
    if !isempty(raw_eigen_csv_path)
        _write_buckling_raw_eigen_csv(eigenvalues, raw_eigen_csv_path;
            buckling_subcase=buckling_subcase,
            static_subcase=static_subcase,
            backend=get(diagnostics, "solver_backend", ""),
            phase="pre_positive_filter",
            requested_modes=num_modes,
            requested_modes_internal=num_modes_request,
            eigrl_v1=eigrl_v1,
            eigrl_v2=eigrl_v2,
            positive_tol=positive_tol)
        diagnostics["raw_eigen_csv"] = raw_eigen_csv_path
    end
    valid_idx = findall(x -> x > positive_tol, eigenvalues)
    if length(valid_idx) < n_found
        log_msg("[BUCKLING] Dropped $(n_found - length(valid_idx)) non-positive eigenvalues (Nastran convention)")
    end

    # Apply EIGRL V1/V2 range filter if specified. V1 is clamped to >+tol so a
    # negative V1 (commonly -1e-4 in MSC decks) does not re-admit non-positive
    # roots after the positivity filter above.
    if has_range
        v1_eff = max(eigrl_v1, positive_tol)
        range_abs_tol = max(solver_env_float("JFEM_SOL105_RANGE_ABS_TOL", 0.0), 0.0)
        range_rel_tol = max(solver_env_float("JFEM_SOL105_RANGE_REL_TOL", 0.0), 0.0)
        v2_eff = eigrl_v2 + max(range_abs_tol, abs(eigrl_v2) * range_rel_tol)
        range_idx = filter(i -> eigenvalues[i] >= v1_eff && eigenvalues[i] <= v2_eff, valid_idx)
        if v2_eff > eigrl_v2
            log_msg("[BUCKLING] EIGRL range [$eigrl_v1, $eigrl_v2] with upper tolerance -> $v2_eff: $(length(range_idx)) of $(length(valid_idx)) positive eigenvalues in range")
        else
            log_msg("[BUCKLING] EIGRL range [$eigrl_v1, $eigrl_v2]: $(length(range_idx)) of $(length(valid_idx)) positive eigenvalues in range")
        end
        valid_idx = range_idx

        # Zero-shift inverse iteration naturally favors the smallest-|lambda| roots.
        # When a buckling range upper bound is present, augment the in-range spectrum
        # with a targeted shift-invert search near V2 so upper-branch modes are not
        # missed behind lower-|lambda| or negative clusters.
        if !will_use_dense
            # --- Gate: should we run the augmentation at all? ---
            # Safe auto-skip: Strategy 2 has already produced eigenvalues BOTH
            # above V2 and below V1, AND returned enough in-range modes. That
            # proves the zero-shift search spanned the range (no "upper-branch
            # modes hiding behind low-|λ| clusters" can exist). Augmentation
            # would only rediscover duplicates. This branch is intentionally
            # strict so that large-deck validation cases (where Strategy 2 only
            # reaches near zero) never takes it — parity preserved by default.
            #
            # Env opt-out: users may set JFEM_SOL105_SKIP_RANGE_AUGMENTATION=true
            # to force-skip even when the auto gate wouldn't — trades
            # in-range spectrum completeness for ~30-50% faster eigensolves.
            env_skip_aug = solver_env_bool("JFEM_SOL105_SKIP_RANGE_AUGMENTATION", false)
            explicit_aug_sigmas = !isempty(strip(get(ENV, "JFEM_SOL105_RANGE_AUGMENTATION_SIGMAS", "")))
            # Default ON (2026-04-29): the zero-shift inverse iteration only finds
            # smallest-|λ| roots, which for bending-dominant prestress (e.g. 3-point
            # bending) are spurious low/negative modes near 0, NOT the physical
            # buckling modes higher up in the [V1,V2] range. Without this shift-
            # invert near V2 the solver returns the wrong spectrum and reports
            # "no positive modes". The auto-skip gate below still prevents
            # redundant work when zero-shift already brackets the range.
            range_aug_requested =
                solver_env_bool("JFEM_SOL105_RANGE_AUGMENTATION", true) ||
                solver_env_bool("JFEM_SOL105_RANGE_AUGMENTATION_MULTI", false) ||
                explicit_aug_sigmas
            spans_above_V2 = any(v -> real(v) > eigrl_v2, eigenvalues)
            spans_below_V1 = any(v -> real(v) < eigrl_v1, eigenvalues)
            sufficient_in_range = length(range_idx) >= num_modes_request
            auto_skip_aug = spans_above_V2 && spans_below_V1 && sufficient_in_range
            do_augmentation = range_aug_requested && !(env_skip_aug || auto_skip_aug)

            if !do_augmentation
                reason = !range_aug_requested ? "not_requested" :
                         (env_skip_aug ? "env_override" : "auto_brackets")
                log_msg("[BUCKLING] Skipping range-targeted shift-invert ($reason, $(length(range_idx)) in-range modes already)")
                diagnostics["range_augmentation"] = Dict{String,Any}(
                    "status" => "skipped",
                    "reason" => reason,
                    "requested" => range_aug_requested,
                    "in_range_modes_from_strategy2" => length(range_idx),
                )
                buckling_timings["range_augmentation"] = 0.0
            else
                t_aug_start = time_ns()
                aug_added = 0
                aug_result = "no_shifted_modes"
                aug_shift_count = 0
                aug_details = Any[]
                sigmas = range_augmentation_sigmas(valid_idx)
                for (isigma, sigma) in enumerate(sigmas)
                    shifted_modes = attempt_shifted_buckling_search(
                        sigma,
                        isigma == length(sigmas) && sigma == eigrl_v2 ?
                            "krylov_range_shifted" :
                            "krylov_range_shifted_mid",
                    )
                    shifted_modes === nothing && continue
                    aug_shift_count += 1
                    shifted_eigenvalues, shifted_eigenvectors = shifted_modes
                    # Match the main filter: positive-only, lower bound clamped to +tol.
                    shifted_valid_idx = findall(x -> x > positive_tol, shifted_eigenvalues)
                    shifted_range_idx = filter(i -> shifted_eigenvalues[i] >= max(eigrl_v1, positive_tol) && shifted_eigenvalues[i] <= v2_eff, shifted_valid_idx)
                    if !isempty(shifted_range_idx)
                        shifted_vals = shifted_eigenvalues[shifted_range_idx]
                        shifted_vecs = shifted_eigenvectors[:, shifted_range_idx]
                        base_in_range = length(valid_idx)
                        if isempty(valid_idx)
                            eigenvalues = shifted_vals
                            eigenvectors = shifted_vecs
                            valid_idx = collect(eachindex(eigenvalues))
                            diagnostics["solver_backend"] = "krylov_range_shifted"
                            added = length(valid_idx)
                            aug_added += added
                            aug_result = "recovered_from_empty"
                            log_msg("[BUCKLING] EIGRL range [$eigrl_v1, $eigrl_v2]: recovered $(length(valid_idx)) in-range eigenvalues with targeted shift sigma=$sigma")
                        else
                            eigenvalues, eigenvectors, added = merge_unique_eigenpairs(
                                eigenvalues, eigenvectors, shifted_vals, shifted_vecs)
                            valid_idx = findall(x -> x > positive_tol && x >= max(eigrl_v1, positive_tol) && x <= v2_eff, eigenvalues)
                            aug_added += added
                            if added > 0
                                diagnostics["solver_backend"] = "$(diagnostics["solver_backend"])+range_shifted"
                                aug_result = "augmented"
                                log_msg("[BUCKLING] EIGRL range [$eigrl_v1, $eigrl_v2]: augmented in-range spectrum from $base_in_range to $(length(valid_idx)) modes with targeted shift sigma=$sigma")
                            elseif aug_result == "no_shifted_modes"
                                aug_result = "no_new_modes"
                                log_msg("[BUCKLING] EIGRL range [$eigrl_v1, $eigrl_v2]: targeted shift sigma=$sigma added no new in-range eigenvalues")
                            end
                        end
                        push!(aug_details, Dict(
                            "sigma" => sigma,
                            "shifted_in_range_modes" => length(shifted_range_idx),
                            "added_modes" => added,
                            "in_range_modes_after" => length(valid_idx),
                        ))
                    elseif isempty(valid_idx)
                        aug_result = "no_in_range_modes_found"
                        log_msg("[BUCKLING] EIGRL range [$eigrl_v1, $eigrl_v2]: targeted shift sigma=$sigma found no in-range eigenvalues, using all $(length(valid_idx))")
                    else
                        push!(aug_details, Dict(
                            "sigma" => sigma,
                            "shifted_in_range_modes" => 0,
                            "added_modes" => 0,
                            "in_range_modes_after" => length(valid_idx),
                        ))
                    end
                end
                if aug_shift_count == 0 && isempty(range_idx)
                    aug_result = "no_shifted_modes_empty_range"
                    log_msg("[BUCKLING] EIGRL range [$eigrl_v1, $eigrl_v2]: no eigenvalues in range, using all $(length(valid_idx))")
                end
                diagnostics["range_augmentation"] = Dict{String,Any}(
                    "status" => "ran",
                    "result" => aug_result,
                    "added_modes" => aug_added,
                    "sigmas" => sigmas,
                    "shift_count" => aug_shift_count,
                    "shifts" => aug_details,
                    "wall_seconds" => (time_ns() - t_aug_start) * 1e-9,
                )
                buckling_timings["range_augmentation"] =
                    diagnostics["range_augmentation"]["wall_seconds"]
            end
        end
    end

    # Sort ascending by signed eigenvalue (all positives now, smallest λ first =
    # smallest critical load = lowest buckling mode, matches Nastran).
    sorted_idx = valid_idx[sortperm(eigenvalues[valid_idx])]

    # JFEM_BUCKLING_LOCALIZATION_FILTER (2026-05-14 evening): drop modes whose
    # translation-energy participation is dominated by a single element
    # (max element share > threshold). Mechanism: certain kernel-stiffening
    # flag combinations (notably `JFEM_PCOMP_RIGID_TS_LITERAL=true` with low
    # `JFEM_PCOMP_RIGID_TS_CS_SCALE`) produce spurious low-magnitude modes
    # localized on tight clusters of 4-8 adjacent elements. Empirically, the
    # single-element top share for these modes is 25-50%; physical global
    # buckling modes are ≤7% top share (verified on large validation runs:
    # HTP_launch mode 1 = 4.14% top share, VTP_3wp_strain mode 1 = 6.78%).
    # The cluster filter's spectral-gap rule misses these spurious modes
    # because they form a dense low-magnitude band with no clean gap below
    # the physical cluster.
    loc_filter_raw = strip(get(ENV, "JFEM_BUCKLING_LOCALIZATION_FILTER", "true"))
    loc_filter_enabled = isempty(loc_filter_raw) ? true :
        lowercase(loc_filter_raw) in ("1", "true", "yes", "on")
    loc_max_share_raw = strip(get(ENV, "JFEM_BUCKLING_LOCALIZATION_MAX_SHARE", ""))
    # Default 0.10 (2026-05-14 evening): physical buckling modes on the
    # large validation set have top-element share 4-7% (representative mode 1 4.14%,
    # VTP_3wp_strain mode 1 6.78%); intermediate spurious modes that survive
    # the 0.20 threshold sit at 13-14% (verified VTP_3wp_strain Cs=2 modes
    # 3-4). 0.10 catches the intermediate spurious without endangering
    # physical modes. Verified safe on the large validation cases: same mode 1
    # eigenvalue as without the filter (cluster filter handles fallback when
    # locfilter is aggressive). Override via env to relax (0.20) or tighten
    # (0.05) per deck.
    loc_max_share = isempty(loc_max_share_raw) ? 0.10 :
        (tryparse(Float64, loc_max_share_raw) === nothing ? 0.10 : parse(Float64, loc_max_share_raw))

    # Mesh-size guard (2026-05-14 evening, post-probe-regression): the
    # localization filter assumes physical buckling modes are distributed
    # across many elements, so a single-element top share > threshold flags
    # spurious. On a SMALL mesh (e.g., probe BDFs with 8-50 elements) a
    # physical mode can have top share 10-30% just because there are few
    # elements participating in the global shape, with no kernel pathology.
    # Skip the filter when N_elements is below the threshold; the cluster
    # filter still operates as the fallback.
    loc_min_elem_raw = strip(get(ENV, "JFEM_BUCKLING_LOCALIZATION_MIN_ELEMENTS", ""))
    loc_min_elements = isempty(loc_min_elem_raw) ? 100 :
        (tryparse(Int, loc_min_elem_raw) === nothing ? 100 : parse(Int, loc_min_elem_raw))
    n_cshells_total = haskey(model, "CSHELLs") ? length(model["CSHELLs"]) : 0
    loc_top_kappa_raw = strip(get(ENV, "JFEM_BUCKLING_LOCALIZATION_TOP_KAPPA_L_MIN", ""))
    loc_top_kappa_l_min_raw = isempty(loc_top_kappa_raw) ? 0.0 :
        (tryparse(Float64, loc_top_kappa_raw) === nothing ? 0.0 : max(parse(Float64, loc_top_kappa_raw), 0.0))
    loc_top_kappa_modes_min = max(
        solver_env_int("JFEM_BUCKLING_LOCALIZATION_TOP_KAPPA_L_MIN_NMODES_MIN", 0),
        0,
    )
    loc_top_kappa_v2_max = solver_env_float(
        "JFEM_BUCKLING_LOCALIZATION_TOP_KAPPA_L_MIN_V2_MAX",
        0.0,
    )
    loc_top_kappa_l_min =
        (loc_top_kappa_modes_min > 0 && num_modes < loc_top_kappa_modes_min) ||
        (loc_top_kappa_v2_max > 0.0 && (!has_range || eigrl_v2 > loc_top_kappa_v2_max)) ?
        0.0 :
        loc_top_kappa_l_min_raw
    broad_strip_filter_enabled = solver_env_bool("JFEM_BUCKLING_BROAD_STRIP_FILTER", false)
    broad_strip_top_n = max(
        something(tryparse(Int, strip(get(ENV, "JFEM_BUCKLING_BROAD_STRIP_TOP_N", "24"))), 24),
        1,
    )
    broad_strip_pid_count = max(
        something(tryparse(Int, strip(get(ENV, "JFEM_BUCKLING_BROAD_STRIP_PID_COUNT", "4"))), 4),
        1,
    )
    broad_strip_pid_share_min = clamp(
        solver_env_float("JFEM_BUCKLING_BROAD_STRIP_PID_SHARE_MIN", 0.50),
        0.0,
        1.0,
    )
    broad_strip_aspect_min = max(solver_env_float("JFEM_BUCKLING_BROAD_STRIP_ASPECT_MIN", 3.8), 1.0)
    broad_strip_max_top_share = clamp(
        solver_env_float("JFEM_BUCKLING_BROAD_STRIP_MAX_TOP_SHARE", 0.10),
        0.0,
        1.0,
    )
    broad_strip_lambda_min = solver_env_float("JFEM_BUCKLING_BROAD_STRIP_LAMBDA_MIN", 0.0)
    broad_strip_lambda_max = solver_env_float("JFEM_BUCKLING_BROAD_STRIP_LAMBDA_MAX", 1.0e99)
    patch_keep_enabled = solver_env_bool("JFEM_BUCKLING_LOCALIZATION_PATCH_KEEP", false)
    patch_keep_top_n = max(
        something(tryparse(Int, strip(get(ENV, "JFEM_BUCKLING_LOCALIZATION_PATCH_TOP_N", "4"))), 4),
        2,
    )
    patch_keep_top2_share_max = clamp(
        solver_env_float("JFEM_BUCKLING_LOCALIZATION_PATCH_TOP2_SHARE_MAX", 0.75),
        0.0,
        1.0,
    )
    patch_keep_topn_share_max = clamp(
        solver_env_float("JFEM_BUCKLING_LOCALIZATION_PATCH_TOPN_SHARE_MAX", 0.90),
        0.0,
        1.0,
    )
    patch_keep_kappa_l_min = max(
        solver_env_float("JFEM_BUCKLING_LOCALIZATION_PATCH_KAPPA_L_MIN", 0.05),
        0.0,
    )
    patch_keep_lambda_min = solver_env_float("JFEM_BUCKLING_LOCALIZATION_PATCH_LAMBDA_MIN", 0.0)
    patch_keep_lambda_max = solver_env_float("JFEM_BUCKLING_LOCALIZATION_PATCH_LAMBDA_MAX", 1.0e99)

    if loc_filter_enabled && haskey(model, "CSHELLs") && !isempty(sorted_idx) && length(free_dofs) > 0 &&
       n_cshells_total >= loc_min_elements
        t_localization = time_ns()
        loc_scan_all =
            solver_env_bool("JFEM_BUCKLING_LOCALIZATION_SCAN_ALL", false) ||
            solver_env_bool("JFEM_SOL105_RETURN_ALL_IN_RANGE", false)
        loc_scan_buffer = max(solver_env_int("JFEM_BUCKLING_LOCALIZATION_SCAN_BUFFER", 16), 0)
        loc_scan_limit = loc_scan_all ? length(sorted_idx) :
            min(length(sorted_idx), max(num_modes + loc_scan_buffer, num_modes))
        loc_eval_idx = loc_scan_limit >= length(sorted_idx) ?
            sorted_idx : sorted_idx[1:loc_scan_limit]
        max_nid = isempty(id_map) ? 0 : maximum(keys(id_map))
        id_vec = zeros(Int, max_nid)
        for (nid, idx) in id_map
            id_vec[nid] = idx
        end
        elem_nids = Vector{Vector{Int}}()
        elem_pids = Int[]
        elem_aspects = Float64[]
        function elem_edge_aspect(nids)
            pts = Tuple{Float64,Float64,Float64}[]
            for nid in nids
                idx = id_vec[nid]
                push!(pts, (X[idx, 1], X[idx, 2], X[idx, 3]))
            end
            d(a, b) = sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2 + (a[3] - b[3])^2)
            lens = if length(pts) == 4
                (d(pts[1], pts[2]), d(pts[2], pts[3]), d(pts[3], pts[4]), d(pts[4], pts[1]))
            else
                (d(pts[1], pts[2]), d(pts[2], pts[3]), d(pts[3], pts[1]))
            end
            return maximum(lens) / max(minimum(lens), 1e-12)
        end
        for (_, el) in model["CSHELLs"]
            nids = el["NODES"]
            n = length(nids)
            (n == 3 || n == 4) || continue
            valid = true
            for k in 1:n
                nid = nids[k]
                if nid < 1 || nid > max_nid || id_vec[nid] == 0
                    valid = false; break
                end
            end
            if valid
                push!(elem_nids, nids)
                push!(elem_pids, something(tryparse(Int, string(get(el, "PID", 0))), 0))
                push!(elem_aspects, elem_edge_aspect(nids))
            end
        end
        if !isempty(elem_nids)
            elem_top_kappa_l = zeros(Float64, length(elem_nids))
            if loc_top_kappa_l_min > 0.0 || patch_keep_enabled
                elem_normals = zeros(Float64, length(elem_nids), 3)
                node_normal_sum = zeros(Float64, length(id_map), 3)
                for (ei, nids) in enumerate(elem_nids)
                    i1 = id_vec[nids[1]]
                    i2 = id_vec[nids[2]]
                    i3 = id_vec[nids[3]]
                    x1 = X[i1, 1]; y1 = X[i1, 2]; z1 = X[i1, 3]
                    x2 = X[i2, 1]; y2 = X[i2, 2]; z2 = X[i2, 3]
                    x3 = X[i3, 1]; y3 = X[i3, 2]; z3 = X[i3, 3]
                    if length(nids) == 4
                        i4 = id_vec[nids[4]]
                        ax = x3 - x1; ay = y3 - y1; az = z3 - z1
                        bx = X[i4, 1] - x2; by = X[i4, 2] - y2; bz = X[i4, 3] - z2
                    else
                        ax = x2 - x1; ay = y2 - y1; az = z2 - z1
                        bx = x3 - x1; by = y3 - y1; bz = z3 - z1
                    end
                    nx = ay * bz - az * by
                    ny = az * bx - ax * bz
                    nz = ax * by - ay * bx
                    nn = sqrt(nx * nx + ny * ny + nz * nz)
                    nn <= 1e-12 && continue
                    nx /= nn; ny /= nn; nz /= nn
                    elem_normals[ei, 1] = nx
                    elem_normals[ei, 2] = ny
                    elem_normals[ei, 3] = nz
                    for nid in nids
                        idx = id_vec[nid]
                        node_normal_sum[idx, 1] += nx
                        node_normal_sum[idx, 2] += ny
                        node_normal_sum[idx, 3] += nz
                    end
                end
                for idx in axes(node_normal_sum, 1)
                    nx = node_normal_sum[idx, 1]
                    ny = node_normal_sum[idx, 2]
                    nz = node_normal_sum[idx, 3]
                    nn = sqrt(nx * nx + ny * ny + nz * nz)
                    if nn > 1e-12
                        node_normal_sum[idx, 1] = nx / nn
                        node_normal_sum[idx, 2] = ny / nn
                        node_normal_sum[idx, 3] = nz / nn
                    end
                end
                for (ei, nids) in enumerate(elem_nids)
                    ex = elem_normals[ei, 1]
                    ey = elem_normals[ei, 2]
                    ez = elem_normals[ei, 3]
                    enn = sqrt(ex * ex + ey * ey + ez * ez)
                    enn <= 1e-12 && continue
                    acc = 0.0
                    nacc = 0
                    for nid in nids
                        idx = id_vec[nid]
                        nx = node_normal_sum[idx, 1]
                        ny = node_normal_sum[idx, 2]
                        nz = node_normal_sum[idx, 3]
                        nn = sqrt(nx * nx + ny * ny + nz * nz)
                        nn <= 1e-12 && continue
                        acc += 1.0 - clamp(abs(nx * ex + ny * ey + nz * ez), 0.0, 1.0)
                        nacc += 1
                    end
                    nacc > 0 && (elem_top_kappa_l[ei] = acc / nacc)
                end
            end
            surviving = Int[]
            dropped_info = Tuple{Int, Float64, Float64, Float64}[]   # (orig_idx, lambda, share, top kappa_L)
            kept_high_info = Tuple{Int, Float64, Float64, Float64}[]
            patch_kept_info = Tuple{Int, Float64, Float64, Float64, Float64, Float64}[]
            broad_dropped_info = Tuple{Int, Float64, Float64, Float64, Float64}[]
            full_u = zeros(ndof)
            elem_energy = broad_strip_filter_enabled ? zeros(Float64, length(elem_nids)) : Float64[]
            for k_idx in loc_eval_idx
                fill!(full_u, 0.0)
                broad_strip_filter_enabled && fill!(elem_energy, 0.0)
                patch_top_e = patch_keep_enabled ? zeros(Float64, patch_keep_top_n) : Float64[]
                evec = view(eigenvectors, :, k_idx)
                for (i, dof_idx) in enumerate(free_dofs)
                    full_u[dof_idx] = evec[i]
                end
                max_e = 0.0
                max_ei = 0
                total_e = 0.0
                for (ei, nids) in enumerate(elem_nids)
                    n = length(nids)
                    e = 0.0
                    for nid in nids
                        idx = id_vec[nid]
                        base = (idx - 1) * 6
                        tx = full_u[base + 1]
                        ty = full_u[base + 2]
                        tz = full_u[base + 3]
                        e += tx*tx + ty*ty + tz*tz
                    end
                    e /= n
                    broad_strip_filter_enabled && (elem_energy[ei] = e)
                    if patch_keep_enabled && e > patch_top_e[end]
                        patch_top_e[end] = e
                        sort!(patch_top_e; rev=true)
                    end
                    total_e += e
                    if e > max_e
                        max_e = e
                        max_ei = ei
                    end
                end
                if total_e > 0
                    share = max_e / total_e
                    if share > loc_max_share
                        top_kappa_l = max_ei > 0 ? elem_top_kappa_l[max_ei] : 0.0
                        patch_keep = false
                        patch_top2_share = 0.0
                        patch_topn_share = 0.0
                        if patch_keep_enabled &&
                           eigenvalues[k_idx] >= patch_keep_lambda_min &&
                           eigenvalues[k_idx] <= patch_keep_lambda_max &&
                           top_kappa_l >= patch_keep_kappa_l_min
                            patch_top2_share = sum(@view patch_top_e[1:min(2, length(patch_top_e))]) / total_e
                            patch_topn_share = sum(patch_top_e) / total_e
                            patch_keep =
                                patch_top2_share <= patch_keep_top2_share_max &&
                                patch_topn_share <= patch_keep_topn_share_max
                        end
                        if patch_keep
                            push!(patch_kept_info, (
                                k_idx,
                                eigenvalues[k_idx],
                                share,
                                top_kappa_l,
                                patch_top2_share,
                                patch_topn_share,
                            ))
                        elseif loc_top_kappa_l_min <= 0.0 || top_kappa_l >= loc_top_kappa_l_min
                            push!(dropped_info, (k_idx, eigenvalues[k_idx], share, top_kappa_l))
                            continue
                        else
                            push!(kept_high_info, (k_idx, eigenvalues[k_idx], share, top_kappa_l))
                        end
                    end
                    if broad_strip_filter_enabled &&
                       eigenvalues[k_idx] >= broad_strip_lambda_min &&
                       eigenvalues[k_idx] <= broad_strip_lambda_max &&
                       share <= broad_strip_max_top_share &&
                       length(elem_energy) == length(elem_nids)
                        perm = sortperm(elem_energy; rev=true)
                        top_count = min(broad_strip_top_n, length(perm))
                        pid_weight = Dict{Int,Float64}()
                        aspect_sum = 0.0
                        top_weight = 0.0
                        for pos in 1:top_count
                            ei = perm[pos]
                            w = elem_energy[ei] / total_e
                            w <= 0.0 && continue
                            pid = elem_pids[ei]
                            pid_weight[pid] = get(pid_weight, pid, 0.0) + w
                            aspect_sum += w * elem_aspects[ei]
                            top_weight += w
                        end
                        pid_rows = collect(pid_weight)
                        sort!(pid_rows; by=p -> p.second, rev=true)
                        n_pid = min(broad_strip_pid_count, length(pid_rows))
                        pid_share = n_pid == 0 ? 0.0 : sum(pid_rows[i].second for i in 1:n_pid)
                        aspect_w = top_weight > 1e-30 ? aspect_sum / top_weight : 0.0
                        if pid_share >= broad_strip_pid_share_min && aspect_w >= broad_strip_aspect_min
                            push!(broad_dropped_info, (k_idx, eigenvalues[k_idx], share, pid_share, aspect_w))
                            continue
                        end
                    end
                end
                push!(surviving, k_idx)
            end
            if loc_scan_limit < length(sorted_idx)
                append!(surviving, @view sorted_idx[(loc_scan_limit + 1):end])
            end
            if !isempty(dropped_info) || !isempty(broad_dropped_info) || !isempty(patch_kept_info)
                if !isempty(dropped_info)
                    log_msg("[BUCKLING] localization filter: dropped $(length(dropped_info)) of " *
                            "$(length(loc_eval_idx)) scanned modes (top-element share > " *
                            "$(round(100*loc_max_share; digits=1))%)")
                end
                for (_k_idx, lam, sh, kap) in dropped_info[1:min(5, length(dropped_info))]
                    suffix = loc_top_kappa_l_min > 0.0 ? ", top_kappa_L=$(round(kap; sigdigits=4))" : ""
                    log_msg("[BUCKLING]   skip λ=$(round(lam; sigdigits=5)) (share=$(round(100*sh; digits=1))%$suffix)")
                end
                if !isempty(broad_dropped_info)
                    log_msg("[BUCKLING] broad-strip filter: dropped $(length(broad_dropped_info)) of " *
                            "$(length(loc_eval_idx)) scanned modes (top $(broad_strip_pid_count) PID share >= " *
                            "$(round(100*broad_strip_pid_share_min; digits=1))%, weighted aspect >= " *
                            "$(round(broad_strip_aspect_min; digits=3)))")
                    for (_k_idx, lam, sh, psh, asp) in broad_dropped_info[1:min(5, length(broad_dropped_info))]
                        log_msg("[BUCKLING]   skip Î»=$(round(lam; sigdigits=5)) " *
                                "(share=$(round(100*sh; digits=1))%, pid_share=$(round(100*psh; digits=1))%, aspect=$(round(asp; digits=3)))")
                    end
                end
                if loc_top_kappa_l_min > 0.0 && !isempty(kept_high_info)
                    log_msg("[BUCKLING] localization filter: kept $(length(kept_high_info)) high-share mode(s) " *
                            "with top_kappa_L < $(round(loc_top_kappa_l_min; sigdigits=4))")
                    for (_k_idx, lam, sh, kap) in kept_high_info[1:min(5, length(kept_high_info))]
                        log_msg("[BUCKLING]   keep λ=$(round(lam; sigdigits=5)) " *
                                "(share=$(round(100*sh; digits=1))%, top_kappa_L=$(round(kap; sigdigits=4)))")
                    end
                end
                if !isempty(patch_kept_info)
                    log_msg("[BUCKLING] localization patch keep: kept $(length(patch_kept_info)) high-share mode(s) " *
                            "with top2 share <= $(round(100*patch_keep_top2_share_max; digits=1))% and " *
                            "top$(patch_keep_top_n) share <= $(round(100*patch_keep_topn_share_max; digits=1))%")
                    for (_k_idx, lam, sh, kap, top2, topn) in patch_kept_info[1:min(5, length(patch_kept_info))]
                        log_msg("[BUCKLING]   keep Î»=$(round(lam; sigdigits=5)) " *
                                "(share=$(round(100*sh; digits=1))%, top2=$(round(100*top2; digits=1))%, " *
                                "top$(patch_keep_top_n)=$(round(100*topn; digits=1))%, top_kappa_L=$(round(kap; sigdigits=4)))")
                    end
                end
                sorted_idx = surviving
                diagnostics["localization_filter"] = Dict{String,Any}(
                    "dropped" => length(dropped_info),
                    "max_share_threshold" => loc_max_share,
                    "top_kappa_l_min" => loc_top_kappa_l_min,
                    "top_kappa_l_min_raw" => loc_top_kappa_l_min_raw,
                    "top_kappa_l_min_nmodes_min" => loc_top_kappa_modes_min,
                    "top_kappa_l_min_v2_max" => loc_top_kappa_v2_max,
                    "kept_high_share_low_kappa" => length(kept_high_info),
                    "patch_keep_enabled" => patch_keep_enabled,
                    "patch_kept" => length(patch_kept_info),
                    "broad_strip_enabled" => broad_strip_filter_enabled,
                    "broad_strip_dropped" => length(broad_dropped_info),
                    "scanned_modes" => length(loc_eval_idx),
                    "available_modes" => length(sorted_idx),
                    "scan_all" => loc_scan_all,
                    "scan_buffer" => loc_scan_buffer,
                )
            end
            buckling_timings["localization_filter"] = (time_ns() - t_localization) * 1e-9
        end
    end

    # By default, honor EIGRL ND even when V1/V2 is present. Range augmentation
    # may discover many more in-range roots than MSC reports for an ND-limited
    # deck; returning all of them is useful for completeness diagnostics but
    # makes first-N parity and exported mode numbering drift away from Nastran.
    #
    # Opt in to the expanded diagnostic output with
    # JFEM_SOL105_RETURN_ALL_IN_RANGE=true. The hard cap avoids accidentally
    # exporting a huge mode set from a broad range.
    if has_range
        if return_all_range
            cap_raw = strip(get(ENV, "JFEM_SOL105_RETURN_ALL_IN_RANGE_MAX", "256"))
            cap = tryparse(Int, cap_raw)
            cap = cap === nothing ? 256 : clamp(cap, 1, max(length(sorted_idx), 1))
            n_out = min(length(sorted_idx), cap)
            diagnostics["range_output"] = Dict{String,Any}(
                "mode" => "all_in_range",
                "available_in_range_modes" => length(sorted_idx),
                "cap" => cap,
            )
        else
            n_out = min(num_modes, length(sorted_idx))
            diagnostics["range_output"] = Dict{String,Any}(
                "mode" => "eigrl_nd",
                "available_in_range_modes" => length(sorted_idx),
                "requested_modes" => num_modes,
            )
        end
    else
        n_out = min(num_modes_request, length(sorted_idx))
    end
    if n_out == 0
        log_msg("[BUCKLING] WARNING: No valid eigenvalues found")
        diagnostics["solver_backend"] = "no_valid_modes"
        buckling_timings["postprocess_filter_expand"] = (time_ns() - t_postprocess) * 1e-9
        buckling_timings["total"] = (time_ns() - t_buckling_total) * 1e-9
        diagnostics["timings"] = buckling_timings
        return return_diagnostics ? (Float64[], zeros(ndof, 0), diagnostics) : (Float64[], zeros(ndof, 0))
    end

    # JFEM_BUCKLING_CLUSTER_FILTER: spectral-gap filter for spurious low modes
    # produced by the legacy MITC + 3D Jacobian over-stiffening pattern (see
    # 2026-05-01 entry in the SOL105 parity TODO). On 3wp-style decks, JFEM's
    # spectrum has 16 spurious low-magnitude modes between 0 and the actual
    # buckling cluster, with a clean ~1.25× spectral gap separating them.
    # Detect that gap on the FULL in-range spectrum and skip the pre-gap modes,
    # then re-apply the EIGRL ND cap.
    #
    # Conservativeness: only fires when the post-jump cluster has at least
    # N_DENSE eigenvalues within a small relative spread (default 30%). This
    # prevents misfiring on launch-style decks where mode 1 is naturally far
    # below mode 2 (no dense post-jump cluster).
    cluster_filter_raw = strip(get(ENV, "JFEM_BUCKLING_CLUSTER_FILTER", "true"))
    cluster_filter_enabled = isempty(cluster_filter_raw) ?
        true :
        lowercase(cluster_filter_raw) in ("1", "true", "yes", "on")
    cluster_jump_raw = strip(get(ENV, "JFEM_BUCKLING_CLUSTER_FILTER_RATIO", ""))
    # Default 1.25 (2026-05-01) — empirically detects the spectral gap between
    # JFEM's spurious low cluster and the actual buckling cluster on
    # multi-point bending cases (jump 0.903 -> 1.151, ratio 1.27).
    cluster_jump_threshold = isempty(cluster_jump_raw) ? 1.25 :
        (tryparse(Float64, cluster_jump_raw) === nothing ? 1.25 : parse(Float64, cluster_jump_raw))
    cluster_jump_max_raw = strip(get(ENV, "JFEM_BUCKLING_CLUSTER_FILTER_RATIO_MAX", ""))
    cluster_jump_max = isempty(cluster_jump_max_raw) ? 8.0 :
        (tryparse(Float64, cluster_jump_max_raw) === nothing ? 8.0 : parse(Float64, cluster_jump_max_raw))
    cluster_min_v2_raw = strip(get(ENV, "JFEM_BUCKLING_CLUSTER_FILTER_MIN_V2", ""))
    cluster_min_v2 = isempty(cluster_min_v2_raw) ? 1.0 :
        (tryparse(Float64, cluster_min_v2_raw) === nothing ? 1.0 : parse(Float64, cluster_min_v2_raw))
    cluster_dense_n_raw = strip(get(ENV, "JFEM_BUCKLING_CLUSTER_FILTER_DENSE_N", ""))
    cluster_dense_n = isempty(cluster_dense_n_raw) ? 3 :
        (tryparse(Int, cluster_dense_n_raw) === nothing ? 3 : parse(Int, cluster_dense_n_raw))
    cluster_dense_spread_raw = strip(get(ENV, "JFEM_BUCKLING_CLUSTER_FILTER_DENSE_SPREAD", ""))
    cluster_dense_spread = isempty(cluster_dense_spread_raw) ? 0.30 :
        (tryparse(Float64, cluster_dense_spread_raw) === nothing ? 0.30 : parse(Float64, cluster_dense_spread_raw))
    cluster_singleton_raw = strip(get(ENV, "JFEM_BUCKLING_CLUSTER_FILTER_SINGLETON_RATIO", ""))
    cluster_singleton_threshold = isempty(cluster_singleton_raw) ? max(cluster_jump_threshold, 3.0) :
        (tryparse(Float64, cluster_singleton_raw) === nothing ? max(cluster_jump_threshold, 3.0) : parse(Float64, cluster_singleton_raw))
    cluster_filter_range_ok = has_range && eigrl_v2 >= cluster_min_v2
    # JFEM_BUCKLING_CLUSTER_FILTER_SELECT_RULE: "last" (default, 2026-05-11) picks
    # the latest qualifying gap; "first" picks the earliest (legacy behaviour).
    # Mechanism: spurious-mode clusters from kernel over-stiffening can themselves
    # contain internal sub-cluster gaps > the 1.25× threshold (see VTP_3wp_strain
    # 511003 diagnostic, 2026-05-11). The first-gap rule then commits to an
    # internal sub-gap and reports a still-spurious mode as λ₁. The latest-gap
    # rule scans all qualifying gaps and picks the last — since the dense-N
    # criterion (≥3 modes within 30% spread after the gap) already restricts
    # firing to true buckling-cluster boundaries, the latest one is the
    # spurious→physical transition.
    cluster_select_raw = lowercase(strip(get(ENV, "JFEM_BUCKLING_CLUSTER_FILTER_SELECT_RULE", "")))
    cluster_select_rule = isempty(cluster_select_raw) ? "last" : cluster_select_raw
    cluster_select_last = cluster_select_rule != "first"
    # JFEM_BUCKLING_CLUSTER_FILTER_MAX_PRE_SPREAD: reject candidate gaps whose
    # pre-jump cluster spans more than this ratio in eigenvalue magnitude.
    # Mechanism: spurious modes from kernel over-stiffening are all small local
    # instabilities of similar magnitude (observed pre-jump ratio 1.7-3× on the
    # multi-point bending cases). If the pre-jump cluster spans >5×, it is not a spurious
    # cluster — it is a physical low-frequency buckling band followed by higher-
    # frequency buckling modes (seen on probe BDFs with EIGRL V2=1e8 returning
    # many physical bands). Default 5× is generous vs observed spurious spreads.
    cluster_max_pre_raw = strip(get(ENV, "JFEM_BUCKLING_CLUSTER_FILTER_MAX_PRE_SPREAD", ""))
    cluster_max_pre_spread = isempty(cluster_max_pre_raw) ? 5.0 :
        (tryparse(Float64, cluster_max_pre_raw) === nothing ? 5.0 : parse(Float64, cluster_max_pre_raw))
    # JFEM_BUCKLING_CLUSTER_FILTER_MIN_ELEMENTS (2026-05-13): mesh-size guard.
    # The cluster filter was tuned on large validation meshes (10K+ elements) where the
    # spectrum is dense and "spurious clusters" are clearly identifiable.
    # On small meshes (4×4=16 elements, etc.) the spectrum is sparse: pairs of
    # physical modes can look like a "post-jump cluster" to the heuristic
    # (verified on kg_4x4_pcomp_curved_R200 — JFEM's first 4 modes match
    # Nastran's first 4 modes, but the cluster filter classifies them as
    # spurious because a 1.77× jump exists to mode 5). Skip the filter
    # entirely when n_elements < 100; the localization filter has the same guard.
    cluster_min_elem_raw = strip(get(ENV, "JFEM_BUCKLING_CLUSTER_FILTER_MIN_ELEMENTS", ""))
    cluster_min_elements = isempty(cluster_min_elem_raw) ? 100 :
        (tryparse(Int, cluster_min_elem_raw) === nothing ? 100 : parse(Int, cluster_min_elem_raw))
    n_cshells_filter = haskey(model, "CSHELLs") ? length(model["CSHELLs"]) : 0
    cluster_filter_mesh_ok = n_cshells_filter >= cluster_min_elements

    # Run the cluster detection on the FULL in-range positive spectrum
    # (sorted_idx, all of it), not the ND-capped output. Spurious low modes
    # often fill the slots ND would otherwise expose, so the filter must see
    # past the ND cap to detect the spectral gap. After detection, the ND cap
    # is re-applied to the post-filter spectrum.
    n_skip = 0
    full_n = length(sorted_idx)
    if cluster_filter_enabled && cluster_filter_range_ok &&
       cluster_filter_mesh_ok &&
       full_n >= cluster_dense_n + 2 &&
       all(v -> v > 0.0, eigenvalues[sorted_idx])
        full_eigs = eigenvalues[sorted_idx]
        for i in 1:(full_n - cluster_dense_n)
            ei = full_eigs[i]
            ej = full_eigs[i + 1]
            ei <= 0 && continue
            ej <= 0 && continue
            jump = ej / ei
            local_jump_threshold = i == 1 ? cluster_singleton_threshold : cluster_jump_threshold
            jump < local_jump_threshold && continue
            jump > cluster_jump_max && continue   # very large jumps = post-buckling band, NOT spurious
            # Spurious-cluster magnitude-bound check: reject if the pre-jump
            # cluster spans more than cluster_max_pre_spread×. Auto-passes for i==1
            # (singleton pre-jump). Mechanism: spurious modes from kernel over-
            # stiffening are constrained to a narrow magnitude range; a wide
            # pre-jump span indicates the pre-jump cluster is actually a physical
            # low-frequency band, not spurious clutter.
            if i >= 2
                pre_min = full_eigs[1]
                pre_max = full_eigs[i]
                pre_min > 0 || continue
                if pre_max / pre_min > cluster_max_pre_spread
                    continue
                end
            end
            cluster_max = maximum(full_eigs[(i + 1):(i + cluster_dense_n)])
            cluster_min = minimum(full_eigs[(i + 1):(i + cluster_dense_n)])
            if (cluster_max - cluster_min) / cluster_min <= cluster_dense_spread
                n_skip = i
                log_msg("[BUCKLING] cluster filter ($(cluster_select_last ? "last" : "first")): " *
                        "jump $(round(ej/ei; digits=3))× " *
                        "at mode $i ($(round(ei; digits=5)) → $(round(ej; digits=5))) " *
                        "starts a dense cluster of $cluster_dense_n modes within " *
                        "$(round(100*cluster_dense_spread; digits=1))% spread; " *
                        "skipping $n_skip spurious mode(s)")
                cluster_select_last || break
            end
        end
    end
    # Re-apply ND cap after cluster filtering
    if n_skip > 0
        post_filter_idx = sorted_idx[(n_skip + 1):end]
        if has_range
            if return_all_range
                cap_raw = strip(get(ENV, "JFEM_SOL105_RETURN_ALL_IN_RANGE_MAX", "256"))
                cap = tryparse(Int, cap_raw)
                cap = cap === nothing ? 256 : clamp(cap, 1, max(length(post_filter_idx), 1))
                n_out = min(length(post_filter_idx), cap)
            else
                n_out = min(num_modes, length(post_filter_idx))
            end
        else
            n_out = min(num_modes_request, length(post_filter_idx))
        end
        output_sorted_idx = post_filter_idx[1:n_out]
    else
        output_sorted_idx = sorted_idx[1:n_out]
    end

    t_expand_modes = time_ns()
    final_eigenvalues = eigenvalues[output_sorted_idx]
    eigenvalues_only = solver_env_bool("JFEM_SOL105_EIGENVALUES_ONLY", false)
    if eigenvalues_only
        diagnostics["returned_modes"] = n_out
        diagnostics["mode_shapes_omitted"] = true
        diagnostics["mode_shapes_omitted_reason"] = "JFEM_SOL105_EIGENVALUES_ONLY=true"
        buckling_timings["expand_modes"] = (time_ns() - t_expand_modes) * 1e-9
        log_msg("[BUCKLING] Skipping full mode-shape expansion (JFEM_SOL105_EIGENVALUES_ONLY=true)")
        log_msg("[BUCKLING] Eigenvalues (buckling load factors):")
        for (i, lam) in enumerate(final_eigenvalues)
            log_msg("  Mode $i: lambda = $(round(lam, digits=6))")
        end
        buckling_timings["postprocess_filter_expand"] = (time_ns() - t_postprocess) * 1e-9
        buckling_timings["total"] = (time_ns() - t_buckling_total) * 1e-9
        diagnostics["timings"] = buckling_timings
        return return_diagnostics ? (final_eigenvalues, zeros(ndof, 0), diagnostics) : (final_eigenvalues, zeros(ndof, 0))
    end

    final_eigenvectors = eigenvectors[:, output_sorted_idx]

    # Expand to full DOF set
    mode_shapes = zeros(ndof, n_out)
    for m in 1:n_out
        @inbounds for (row, dof) in pairs(free_dofs)
            mode_shapes[dof, m] = final_eigenvectors[row, m]
        end
    end

    # Recover RBE3 dependent DOFs
    for (dep_dof, pairs) in rbe3_map
        for m in 1:n_out
            u_avg = 0.0
            for (ind_dof, coeff) in pairs
                u_avg += coeff * mode_shapes[ind_dof, m]
            end
            mode_shapes[dep_dof, m] = u_avg
        end
    end

    # Transform mode shapes to global coordinates via node_R
    mode_shapes_global = zeros(ndof, n_out)
    for idx in values(id_map)
        base = (idx-1)*6
        R = node_R[idx]
        for m in 1:n_out
            u1 = mode_shapes[base + 1, m]
            u2 = mode_shapes[base + 2, m]
            u3 = mode_shapes[base + 3, m]
            r1 = mode_shapes[base + 4, m]
            r2 = mode_shapes[base + 5, m]
            r3 = mode_shapes[base + 6, m]
            mode_shapes_global[base + 1, m] = R[1, 1] * u1 + R[1, 2] * u2 + R[1, 3] * u3
            mode_shapes_global[base + 2, m] = R[2, 1] * u1 + R[2, 2] * u2 + R[2, 3] * u3
            mode_shapes_global[base + 3, m] = R[3, 1] * u1 + R[3, 2] * u2 + R[3, 3] * u3
            mode_shapes_global[base + 4, m] = R[1, 1] * r1 + R[1, 2] * r2 + R[1, 3] * r3
            mode_shapes_global[base + 5, m] = R[2, 1] * r1 + R[2, 2] * r2 + R[2, 3] * r3
            mode_shapes_global[base + 6, m] = R[3, 1] * r1 + R[3, 2] * r2 + R[3, 3] * r3
        end
    end

    # Normalize mode shapes (max component = 1.0)
    for m in 1:n_out
        max_val = _matrix_column_max_abs(mode_shapes_global, m)
        if max_val > 1e-30
            _scale_matrix_column!(mode_shapes_global, m, 1.0 / max_val)
        end
    end
    buckling_timings["expand_modes"] = (time_ns() - t_expand_modes) * 1e-9

    log_msg("[BUCKLING] Eigenvalues (buckling load factors):")
    for (i, lam) in enumerate(final_eigenvalues)
        log_msg("  Mode $i: lambda = $(round(lam, digits=6))")
    end

    diagnostics["returned_modes"] = n_out
    buckling_timings["postprocess_filter_expand"] = (time_ns() - t_postprocess) * 1e-9
    buckling_timings["total"] = (time_ns() - t_buckling_total) * 1e-9
    diagnostics["timings"] = buckling_timings
    return return_diagnostics ? (final_eigenvalues, mode_shapes_global, diagnostics) : (final_eigenvalues, mode_shapes_global)
end

# =============================================================================
# SOL103 MASS MATRIX ASSEMBLY
# =============================================================================
function assemble_mass(model, id_map, node_coords, node_R, ndof)
    log_msg("[SOLVER] Assembling Mass Matrix (SOL103)...")
    n_nodes = length(id_map)
    max_nid = maximum(keys(id_map))
    id_vec = zeros(Int, max_nid)
    for (nid, idx) in id_map; id_vec[nid] = idx; end

    # Flat node_R for transformation
    node_R_flat = zeros(3, 3, n_nodes)
    for i in 1:n_nodes
        for r in 1:3, c in 1:3; node_R_flat[r,c,i] = node_R[i][r,c]; end
    end

    I_idx = Vector{Int}(); J_idx = Vector{Int}(); V_val = Vector{Float64}()

    pshells = model["PSHELLs"]; mats = model["MATs"]
    cshells = model["CSHELLs"]; cbars = model["CBARs"]
    cbeams = get(model, "CBEAMs", Dict()); crods = model["CRODs"]
    conrods = get(model, "CONRODs", Dict())
    csolids = get(model, "CSOLIDs", Dict())
    psolids = get(model, "PSOLIDs", Dict())
    conm2s = get(model, "CONM2s", Dict())

    T_buf = zeros(24, 24)
    Me_global = zeros(24, 24)
    tmp24 = zeros(24, 24)
    dofs_buf24 = Vector{Int}(undef, 24)
    lc_buf4 = zeros(4, 2)
    coords_buf_solid = zeros(8, 3)
    T_buf_solid = zeros(24, 24)
    dofs_buf_solid = Vector{Int}(undef, 24)

    # --- Shell elements ---
    for (_, el) in cshells
        pid = string(el["PID"])
        !haskey(pshells, pid) && continue
        prop = pshells[pid]; mid = string(prop["MID"])
        !haskey(mats, mid) && continue
        nids = el["NODES"]; n = length(nids)
        mat = _effective_mat1_for_nodes(model, mid, nids)
        h = Float64(prop["T"])
        rho = Float64(get(mat, "RHO", 0.0))
        nsm = Float64(get(prop, "NSM", 0.0))  # non-structural mass per unit area
        # Skip if no mass source at all
        (rho < 1e-30 && nsm < 1e-30) && continue
        # Effective mass/area = rho*h + NSM. Pass equivalent rho to kernel: rho_eff = (rho*h + NSM)/h
        rho_eff = h > 1e-30 ? (rho * h + nsm) / h : rho

        valid = true
        for k in 1:n
            nid = nids[k]
            (nid < 1 || nid > max_nid || id_vec[nid] == 0) && (valid = false; break)
        end
        !valid && continue

        if n == 4
            i1,i2,i3,i4 = id_vec[nids[1]], id_vec[nids[2]], id_vec[nids[3]], id_vec[nids[4]]
            p1 = SVector{3}(node_coords[i1,:])
            p2 = SVector{3}(node_coords[i2,:])
            p3 = SVector{3}(node_coords[i3,:])
            p4 = SVector{3}(node_coords[i4,:])
            v1, v2, v3 = shell_element_frame_quad4(p1, p2, p3, p4, :bisect)
            Rel_t = @SMatrix [v1[1] v1[2] v1[3]; v2[1] v2[2] v2[3]; v3[1] v3[2] v3[3]]

            c_ctr = (p1+p2+p3+p4)/4.0
            for k in 1:4
                pk = k==1 ? p1 : k==2 ? p2 : k==3 ? p3 : p4
                lc_buf4[k,1] = dot(pk-c_ctr, v1); lc_buf4[k,2] = dot(pk-c_ctr, v2)
            end

            Me_loc = FEM.consistent_mass_quad4(lc_buf4, rho_eff, h)

            # Build T (24×24)
            fill!(T_buf, 0.0)
            for k in 1:4
                idx = k==1 ? i1 : k==2 ? i2 : k==3 ? i3 : i4
                base = (k-1)*6
                TR = Rel_t * node_R[idx]
                for rr in 1:3, cc in 1:3
                    T_buf[base+rr, base+cc] = TR[rr,cc]
                    T_buf[base+3+rr, base+3+cc] = TR[rr,cc]
                end
            end

            # Me_global = T' * Me_loc * T
            fill!(Me_global, 0.0); fill!(tmp24, 0.0)
            @inbounds for jj in 1:24, ll in 1:24
                val = T_buf[ll, jj]
                val == 0.0 && continue
                for ii in 1:24; tmp24[ii, jj] += Me_loc[ii, ll] * val; end
            end
            @inbounds for jj in 1:24, ll in 1:24
                val = tmp24[ll, jj]
                val == 0.0 && continue
                for ii in 1:24; Me_global[ii, jj] += T_buf[ll, ii] * val; end
            end

            for k in 1:4
                idx = k==1 ? i1 : k==2 ? i2 : k==3 ? i3 : i4
                b = (idx-1)*6
                for d in 1:6; dofs_buf24[(k-1)*6+d] = b+d; end
            end
            for c in 1:24, r in 1:24
                push!(I_idx, dofs_buf24[r]); push!(J_idx, dofs_buf24[c]); push!(V_val, Me_global[r,c])
            end

        elseif n == 3
            i1,i2,i3 = id_vec[nids[1]], id_vec[nids[2]], id_vec[nids[3]]
            p1 = SVector{3}(node_coords[i1,:])
            p2 = SVector{3}(node_coords[i2,:])
            p3 = SVector{3}(node_coords[i3,:])
            v1, v2, v3 = shell_element_frame_fast(p1, p2, p3, p3, 3)
            Rel_t = @SMatrix [v1[1] v1[2] v1[3]; v2[1] v2[2] v2[3]; v3[1] v3[2] v3[3]]

            c_ctr = (p1+p2+p3)/3.0
            lc3 = zeros(3,2)
            for k in 1:3
                pk = k==1 ? p1 : k==2 ? p2 : p3
                lc3[k,1] = dot(pk-c_ctr, v1); lc3[k,2] = dot(pk-c_ctr, v2)
            end

            Me_loc = FEM.consistent_mass_tria3(lc3, rho_eff, h)

            T18 = zeros(18, 18)
            for k in 1:3
                idx = k==1 ? i1 : k==2 ? i2 : i3
                base = (k-1)*6
                TR = Rel_t * node_R[idx]
                for rr in 1:3, cc in 1:3
                    T18[base+rr, base+cc] = TR[rr,cc]
                    T18[base+3+rr, base+3+cc] = TR[rr,cc]
                end
            end
            Me18 = T18' * Me_loc * T18

            dofs_t3 = Vector{Int}(undef, 18)
            for k in 1:3
                idx = k==1 ? i1 : k==2 ? i2 : i3
                b = (idx-1)*6
                for d in 1:6; dofs_t3[(k-1)*6+d] = b+d; end
            end
            for c in 1:18, r in 1:18
                push!(I_idx, dofs_t3[r]); push!(J_idx, dofs_t3[c]); push!(V_val, Me18[r,c])
            end
        end
    end

    # --- CBAR elements ---
    for (_, bar) in cbars
        ga, gb = bar["GA"], bar["GB"]
        (!haskey(id_map, ga) || !haskey(id_map, gb)) && continue
        i1, i2 = id_map[ga], id_map[gb]
        pid = string(bar["PID"])
        !haskey(pshells, pid) && continue  # PBAR stored in PSHELLs dict
        prop = pshells[pid]; mid = string(prop["MID"])
        !haskey(mats, mid) && continue
        mat = _effective_mat1_for_nodes(model, mid, [ga, gb])
        rho = Float64(get(mat, "RHO", 0.0))
        nsm_bar = Float64(get(prop, "NSM", 0.0))  # non-structural mass per unit length
        (rho < 1e-30 && nsm_bar < 1e-30) && continue

        p1 = SVector{3}(node_coords[i1,:]); p2 = SVector{3}(node_coords[i2,:])
        L = norm(p2 - p1)
        L < 1e-12 && continue

        A_bar = Float64(get(prop, "A", 0.0))
        Iy = Float64(get(prop, "I1", get(prop, "Iy", 0.0)))
        Iz = Float64(get(prop, "I2", get(prop, "Iz", 0.0)))

        # Effective density: rho_eff = rho + NSM/A (NSM is mass per unit length)
        rho_bar = A_bar > 1e-30 ? rho + nsm_bar / A_bar : rho
        Me_loc = FEM.consistent_mass_frame3d(L, rho_bar, A_bar, Iy, Iz)

        # Transformation
        e1 = (p2 - p1) / L
        vbar = haskey(bar, "V") ? SVector{3}(Float64.(bar["V"])) : SVector(0.0, 0.0, 1.0)
        e2_raw = vbar - dot(vbar, e1)*e1
        e2_len = norm(e2_raw)
        e2 = e2_len > 1e-12 ? e2_raw/e2_len : SVector(0.0, 1.0, 0.0)
        e3 = cross(e1, e2)
        Rel_t = @SMatrix [e1[1] e1[2] e1[3]; e2[1] e2[2] e2[3]; e3[1] e3[2] e3[3]]

        T12 = zeros(12, 12)
        for k in 1:2
            idx = k==1 ? i1 : i2
            base = (k-1)*6
            TR = Rel_t * node_R[idx]
            for rr in 1:3, cc in 1:3
                T12[base+rr, base+cc] = TR[rr,cc]
                T12[base+3+rr, base+3+cc] = TR[rr,cc]
            end
        end
        Me12 = T12' * Me_loc * T12

        dofs12 = Vector{Int}(undef, 12)
        for k in 1:2
            idx = k==1 ? i1 : i2
            b = (idx-1)*6
            for d in 1:6; dofs12[(k-1)*6+d] = b+d; end
        end
        for c in 1:12, r in 1:12
            push!(I_idx, dofs12[r]); push!(J_idx, dofs12[c]); push!(V_val, Me12[r,c])
        end
    end

    # --- CROD elements ---
    for (_, rod) in crods
        ga, gb = rod["GA"], rod["GB"]
        (!haskey(id_map, ga) || !haskey(id_map, gb)) && continue
        i1, i2 = id_map[ga], id_map[gb]
        pid = string(rod["PID"])
        !haskey(pshells, pid) && continue
        prop = pshells[pid]; mid = string(prop["MID"])
        !haskey(mats, mid) && continue
        mat = _effective_mat1_for_nodes(model, mid, [ga, gb])
        rho = Float64(get(mat, "RHO", 0.0))
        nsm_rod = Float64(get(prop, "NSM", 0.0))  # non-structural mass per unit length
        (rho < 1e-30 && nsm_rod < 1e-30) && continue

        p1 = SVector{3}(node_coords[i1,:]); p2 = SVector{3}(node_coords[i2,:])
        L = norm(p2 - p1)
        L < 1e-12 && continue
        A_rod = Float64(get(prop, "A", 0.0))

        # Effective density: rho_eff = rho + NSM/A
        rho_rod = A_rod > 1e-30 ? rho + nsm_rod / A_rod : rho
        Me_loc = FEM.consistent_mass_rod(L, rho_rod, A_rod)

        e1 = (p2 - p1) / L
        e2 = abs(e1[3]) < 0.9 ? normalize(cross(e1, SVector(0.0,0.0,1.0))) : normalize(cross(e1, SVector(1.0,0.0,0.0)))
        e3 = cross(e1, e2)
        Rel_t = @SMatrix [e1[1] e1[2] e1[3]; e2[1] e2[2] e2[3]; e3[1] e3[2] e3[3]]

        T12 = zeros(12, 12)
        for k in 1:2
            idx = k==1 ? i1 : i2
            base = (k-1)*6
            TR = Rel_t * node_R[idx]
            for rr in 1:3, cc in 1:3
                T12[base+rr, base+cc] = TR[rr,cc]
                T12[base+3+rr, base+3+cc] = TR[rr,cc]
            end
        end
        Me12 = T12' * Me_loc * T12

        dofs12 = Vector{Int}(undef, 12)
        for k in 1:2
            idx = k==1 ? i1 : i2
            b = (idx-1)*6
            for d in 1:6; dofs12[(k-1)*6+d] = b+d; end
        end
        for c in 1:12, r in 1:12
            push!(I_idx, dofs12[r]); push!(J_idx, dofs12[c]); push!(V_val, Me12[r,c])
        end
    end

    # --- CONROD elements (rod with integrated properties, no PROD card) ---
    for (_, rod) in conrods
        ga, gb = rod["GA"], rod["GB"]
        (!haskey(id_map, ga) || !haskey(id_map, gb)) && continue
        i1, i2 = id_map[ga], id_map[gb]

        mid = string(rod["MID"])
        !haskey(mats, mid) && continue
        mat = _effective_mat1_for_nodes(model, mid, [ga, gb])
        rho = Float64(get(mat, "RHO", 0.0))
        rho < 1e-30 && continue

        p1 = SVector{3}(node_coords[i1,:]); p2 = SVector{3}(node_coords[i2,:])
        L = norm(p2 - p1)
        L < 1e-12 && continue
        A_rod = Float64(get(rod, "A", 0.0))

        # Add non-structural mass (NSM) per unit length
        nsm = Float64(get(rod, "NSM", 0.0))

        Me_loc = FEM.consistent_mass_rod(L, rho, A_rod)

        # Add NSM contribution (lumped at both ends, translational only)
        if nsm > 0
            nsm_total = nsm * L
            for d in [1, 2, 3]
                Me_loc[d, d] += nsm_total / 2.0
                Me_loc[6+d, 6+d] += nsm_total / 2.0
            end
        end

        e1 = (p2 - p1) / L
        e2 = abs(e1[3]) < 0.9 ? normalize(cross(e1, SVector(0.0,0.0,1.0))) : normalize(cross(e1, SVector(1.0,0.0,0.0)))
        e3 = cross(e1, e2)
        Rel_t = @SMatrix [e1[1] e1[2] e1[3]; e2[1] e2[2] e2[3]; e3[1] e3[2] e3[3]]

        T12 = zeros(12, 12)
        for k in 1:2
            idx = k==1 ? i1 : i2
            base = (k-1)*6
            TR = Rel_t * node_R[idx]
            for rr in 1:3, cc in 1:3
                T12[base+rr, base+cc] = TR[rr,cc]
                T12[base+3+rr, base+3+cc] = TR[rr,cc]
            end
        end
        Me12 = T12' * Me_loc * T12

        dofs12 = Vector{Int}(undef, 12)
        for k in 1:2
            idx = k==1 ? i1 : i2
            b = (idx-1)*6
            for d in 1:6; dofs12[(k-1)*6+d] = b+d; end
        end
        for c in 1:12, r in 1:12
            push!(I_idx, dofs12[r]); push!(J_idx, dofs12[c]); push!(V_val, Me12[r,c])
        end
    end

    # --- CBEAM elements (same mass formulation as CBAR) ---
    for (_, beam) in cbeams
        ga, gb = beam["GA"], beam["GB"]
        (!haskey(id_map, ga) || !haskey(id_map, gb)) && continue
        i1, i2 = id_map[ga], id_map[gb]
        pid = string(beam["PID"])
        !haskey(pshells, pid) && continue
        prop = pshells[pid]; mid = string(prop["MID"])
        !haskey(mats, mid) && continue
        mat = _effective_mat1_for_nodes(model, mid, [ga, gb])
        rho = Float64(get(mat, "RHO", 0.0))
        nsm_beam = Float64(get(prop, "NSM", 0.0))
        (rho < 1e-30 && nsm_beam < 1e-30) && continue

        p1 = SVector{3}(node_coords[i1,:]); p2 = SVector{3}(node_coords[i2,:])
        L = norm(p2 - p1)
        L < 1e-12 && continue

        A_beam = Float64(get(prop, "A", 0.0))
        Iy = Float64(get(prop, "I1", get(prop, "Iy", 0.0)))
        Iz = Float64(get(prop, "I2", get(prop, "Iz", 0.0)))

        # Effective density including NSM
        rho_beam = A_beam > 1e-30 ? rho + nsm_beam / A_beam : rho
        Me_loc = FEM.consistent_mass_frame3d(L, rho_beam, A_beam, Iy, Iz)

        # Transformation
        e1 = (p2 - p1) / L
        vbar = haskey(beam, "V") ? SVector{3}(Float64.(beam["V"])) : SVector(0.0, 0.0, 1.0)
        e2_raw = vbar - dot(vbar, e1)*e1
        e2_len = norm(e2_raw)
        e2 = e2_len > 1e-12 ? e2_raw/e2_len : SVector(0.0, 1.0, 0.0)
        e3 = cross(e1, e2)
        Rel_t = @SMatrix [e1[1] e1[2] e1[3]; e2[1] e2[2] e2[3]; e3[1] e3[2] e3[3]]

        T12 = zeros(12, 12)
        for k in 1:2
            idx = k==1 ? i1 : i2
            base = (k-1)*6
            TR = Rel_t * node_R[idx]
            for rr in 1:3, cc in 1:3
                T12[base+rr, base+cc] = TR[rr,cc]
                T12[base+3+rr, base+3+cc] = TR[rr,cc]
            end
        end
        Me12 = T12' * Me_loc * T12

        dofs12 = Vector{Int}(undef, 12)
        for k in 1:2
            idx = k==1 ? i1 : i2
            b = (idx-1)*6
            for d in 1:6; dofs12[(k-1)*6+d] = b+d; end
        end
        for c in 1:12, r in 1:12
            push!(I_idx, dofs12[r]); push!(J_idx, dofs12[c]); push!(V_val, Me12[r,c])
        end
    end

    # --- SOLID elements ---
    for (_, el) in csolids
        pid = string(el["PID"])
        !haskey(psolids, pid) && continue
        prop = psolids[pid]
        mid = string(prop["MID"])
        !haskey(mats, mid) && continue

        nids = el["NODES"]
        nn = length(nids)
        etype = get(el, "TYPE", "")
        mat = _effective_mat1_for_nodes(model, mid, nids)
        rho = Float64(get(mat, "RHO", 0.0))
        rho < 1e-30 && continue

        valid = true
        for k in 1:nn
            nid = nids[k]
            if !haskey(id_map, nid)
                valid = false
                break
            end
        end
        !valid && continue

        for k in 1:nn
            idx = id_map[nids[k]]
            coords_buf_solid[k,1] = node_coords[idx,1]
            coords_buf_solid[k,2] = node_coords[idx,2]
            coords_buf_solid[k,3] = node_coords[idx,3]
        end

        local Me_loc
        local ndof_el::Int
        if etype == "CTETRA" && nn == 4
            Me_loc = FEM.consistent_mass_tetra4(view(coords_buf_solid, 1:4, :), rho)
            ndof_el = 12
        elseif etype == "CHEXA" && nn == 8
            Me_loc = FEM.consistent_mass_hexa8(view(coords_buf_solid, 1:8, :), rho)
            ndof_el = 24
        elseif etype == "CPENTA" && nn == 6
            Me_loc = FEM.consistent_mass_cpenta6(view(coords_buf_solid, 1:6, :), rho)
            ndof_el = 18
        else
            continue
        end

        fill!(view(T_buf_solid, 1:ndof_el, 1:ndof_el), 0.0)
        for k in 1:nn
            idx = id_map[nids[k]]
            base = (k - 1) * 3
            TR = node_R[idx]
            for rr in 1:3, cc in 1:3
                T_buf_solid[base + rr, base + cc] = TR[rr, cc]
            end
        end
        T_sub = view(T_buf_solid, 1:ndof_el, 1:ndof_el)
        Me = T_sub' * Me_loc * T_sub

        for k in 1:nn
            idx = id_map[nids[k]]
            base = (idx - 1) * 6
            dofs_buf_solid[(k - 1) * 3 + 1] = base + 1
            dofs_buf_solid[(k - 1) * 3 + 2] = base + 2
            dofs_buf_solid[(k - 1) * 3 + 3] = base + 3
        end

        for c in 1:ndof_el, r in 1:ndof_el
            push!(I_idx, dofs_buf_solid[r]); push!(J_idx, dofs_buf_solid[c]); push!(V_val, Me[r,c])
        end
    end

    # --- CONM2 concentrated mass ---
    for (_, cm) in conm2s
        gid = cm["GID"]
        !haskey(id_map, gid) && continue
        idx = id_map[gid]
        m = Float64(cm["M"])
        m < 1e-30 && continue
        base = (idx-1)*6

        # CONM2 offset vector (X1, X2, X3) in basic coordinate system
        x_off = get(cm, "X", [0.0, 0.0, 0.0])
        x1, x2, x3 = Float64(x_off[1]), Float64(x_off[2]), Float64(x_off[3])
        has_offset = (abs(x1) + abs(x2) + abs(x3)) > 1e-30

        # Translational mass (diagonal 3×3)
        for d in 1:3
            push!(I_idx, base+d); push!(J_idx, base+d); push!(V_val, m)
        end

        # Rotational inertia (if provided)
        inertia = get(cm, "I", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        I11, I21, I22, I31, I32, I33 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if length(inertia) >= 6
            I11, I21, I22, I31, I32, I33 = Float64.(inertia)
        end

        # Parallel axis theorem: transfer inertia from CG offset to grid point
        # I_total = I_cg + m * [y²+z², -xy, -xz; -xy, x²+z², -yz; -xz, -yz, x²+y²]
        if has_offset
            I11 += m * (x2^2 + x3^2)
            I22 += m * (x1^2 + x3^2)
            I33 += m * (x1^2 + x2^2)
            I21 -= m * x1 * x2
            I31 -= m * x1 * x3
            I32 -= m * x2 * x3
        end

        # Diagonal rotational inertia
        if abs(I11) > 0; push!(I_idx, base+4); push!(J_idx, base+4); push!(V_val, I11); end
        if abs(I22) > 0; push!(I_idx, base+5); push!(J_idx, base+5); push!(V_val, I22); end
        if abs(I33) > 0; push!(I_idx, base+6); push!(J_idx, base+6); push!(V_val, I33); end

        # Off-diagonal rotational inertia (symmetric)
        if abs(I21) > 0
            push!(I_idx, base+4); push!(J_idx, base+5); push!(V_val, I21)
            push!(I_idx, base+5); push!(J_idx, base+4); push!(V_val, I21)
        end
        if abs(I31) > 0
            push!(I_idx, base+4); push!(J_idx, base+6); push!(V_val, I31)
            push!(I_idx, base+6); push!(J_idx, base+4); push!(V_val, I31)
        end
        if abs(I32) > 0
            push!(I_idx, base+5); push!(J_idx, base+6); push!(V_val, I32)
            push!(I_idx, base+6); push!(J_idx, base+5); push!(V_val, I32)
        end

        # Translation-rotation coupling from offset (Nastran CONM2 formulation)
        # Couples translational DOFs to rotational DOFs via mass × offset
        if has_offset
            # M_tr = m * [0, z, -y; -z, 0, x; y, -x, 0]  (skew-symmetric)
            coupling = [( 0.0,    m*x3,  -m*x2),   # row 4 couples to DOFs 1,2,3
                        (-m*x3,   0.0,    m*x1),   # row 5
                        ( m*x2,  -m*x1,   0.0 )]   # row 6
            for r in 1:3
                for c in 1:3
                    val = coupling[r][c]
                    abs(val) < 1e-30 && continue
                    push!(I_idx, base+3+r); push!(J_idx, base+c); push!(V_val, val)
                    push!(I_idx, base+c); push!(J_idx, base+3+r); push!(V_val, val)
                end
            end
        end
    end

    # --- CONM1 concentrated mass (full 6×6 diagonal mass matrix) ---
    conm1s = get(model, "CONM1s", Dict())
    for (_, cm) in conm1s
        gid = cm["GID"]
        !haskey(id_map, gid) && continue
        idx = id_map[gid]
        base = (idx-1)*6
        m_diag = get(cm, "M_DIAG", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for d in 1:min(6, length(m_diag))
            if abs(m_diag[d]) > 1e-30
                push!(I_idx, base+d); push!(J_idx, base+d); push!(V_val, m_diag[d])
            end
        end
    end

    # --- CMASS2 scalar mass (mass value on the card itself) ---
    cmass2s = get(model, "CMASS2s", Dict())
    for (_, cm) in cmass2s
        mass = Float64(get(cm, "M", 0.0))
        abs(mass) < 1e-30 && continue
        g1 = get(cm, "G1", 0); c1 = get(cm, "C1", 0)
        if g1 > 0 && c1 > 0 && haskey(id_map, g1)
            dof1 = (id_map[g1]-1)*6 + c1
            push!(I_idx, dof1); push!(J_idx, dof1); push!(V_val, mass)
        end
        g2 = get(cm, "G2", 0); c2 = get(cm, "C2", 0)
        if g2 > 0 && c2 > 0 && haskey(id_map, g2)
            dof2 = (id_map[g2]-1)*6 + c2
            push!(I_idx, dof2); push!(J_idx, dof2); push!(V_val, mass)
        end
    end

    # --- CMASS1 scalar mass (mass value from PMASS property) ---
    cmass1s = get(model, "CMASS1s", Dict())
    pmasses = get(model, "PMASSs", Dict())
    for (_, cm) in cmass1s
        pid = string(get(cm, "PID", 0))
        pm = get(pmasses, pid, nothing)
        pm === nothing && continue
        mass = Float64(get(pm, "M", 0.0))
        abs(mass) < 1e-30 && continue
        g1 = get(cm, "G1", 0); c1 = get(cm, "C1", 0)
        if g1 > 0 && c1 > 0 && haskey(id_map, g1)
            dof1 = (id_map[g1]-1)*6 + c1
            push!(I_idx, dof1); push!(J_idx, dof1); push!(V_val, mass)
        end
        g2 = get(cm, "G2", 0); c2 = get(cm, "C2", 0)
        if g2 > 0 && c2 > 0 && haskey(id_map, g2)
            dof2 = (id_map[g2]-1)*6 + c2
            push!(I_idx, dof2); push!(J_idx, dof2); push!(V_val, mass)
        end
    end

    log_msg("[SOLVER] Mass matrix: $(length(V_val)) triplets assembled")
    M = sparse(I_idx, J_idx, V_val, ndof, ndof)

    # Apply WTMASS parameter (unit conversion: weight-density → mass-density)
    # Nastran: M_effective = M * WTMASS  (default WTMASS = 1.0)
    # Common: WTMASS = 1/g = 0.00259 (lb-in-s), 0.001 (kg-mm-s → tonnes)
    wtmass = Float64(get(model, "PARAM_WTMASS", 1.0))
    if wtmass != 1.0 && wtmass > 0.0
        log_msg("[SOLVER] Applying WTMASS = $wtmass to mass matrix")
        M .*= wtmass
    end

    return M
end

# =============================================================================
# SOL103 NORMAL MODES EIGENVALUE SOLVER
# Solves: K*phi = omega^2 * M * phi
# =============================================================================
function solve_modes(K, M, ndof, model, id_map, X, spc_id, node_R, num_modes;
                     rbe3_map=Dict{Int,Vector{Tuple{Int,Float64}}}(),
                     max_elem_stiff=0.0, orig_diag=Float64[],
                     eigrl_v1::Float64=0.0, eigrl_v2::Float64=0.0,
                     eigrl_norm::AbstractString="MASS",
                     eigen_cache=nothing,
                     return_diagnostics::Bool=false)

    log_msg("[MODES] Computing free DOFs...")
    eigen_ctx, eigen_cache_hit = prepare_eigen_solve_context(
        K, ndof, model, id_map, spc_id, rbe3_map; eigen_cache=eigen_cache)
    free_dofs = eigen_ctx.free_dofs
    fixed_dofs = eigen_ctx.fixed_dofs
    bc_diagnostics = eigen_ctx.bc_diagnostics
    n_free = length(free_dofs)
    log_msg("[MODES] Free DOFs: $n_free, Fixed DOFs: $(length(fixed_dofs))")
    if eigen_cache_hit
        log_msg("[MODES] Reusing eigen BC partition cache: Fixed DOFs=$(length(fixed_dofs)), Free DOFs=$n_free")
    end
    diagnostics = Dict{String,Any}(
        "bc_partition" => deepcopy(bc_diagnostics),
        "requested_modes" => num_modes,
        "requested_modes_internal" => 0,
        "free_dofs" => n_free,
        "fixed_dofs" => length(fixed_dofs),
        "eigrl_range" => Dict("v1" => eigrl_v1, "v2" => eigrl_v2),
        "eigrl_norm" => uppercase(strip(eigrl_norm)),
        "eigen_cache" => Dict{String,Any}(
            "enabled" => eigen_cache !== nothing,
            "cache_hit" => eigen_cache_hit,
            "factorization_cache_hit" => false,
        ),
        "solver_backend" => "unsolved",
        "solver_attempts" => Any[],
        "returned_modes" => 0,
    )

    K_ff = eigen_ctx.K_ff
    M_ff = M[free_dofs, free_dofs]
    M_ff = 0.5 * (M_ff + M_ff')

    num_modes_request = min(num_modes * 3, max(n_free - 2, 1))
    diagnostics["requested_modes_internal"] = num_modes_request

    log_msg("[MODES] Solving eigenvalue problem ($num_modes modes, $n_free DOFs)...")

    local eigenvalues, eigenvectors
    solved = false

    # Strategy 1: Dense eigensolver for small/medium systems
    if n_free <= 4000
        push!(diagnostics["solver_attempts"], Dict("name" => "dense_symmetric_definite", "status" => "attempted"))
        try
            log_msg("[MODES] Using dense symmetric-definite eigensolver ($n_free DOFs)...")
            Kd = Matrix(K_ff); Md = Matrix(M_ff)
            vals, vecs = eigen(Symmetric(Kd), Symmetric(Md))
            # Filter: positive real eigenvalues (ω²)
            valid = findall(x -> isfinite(x) && x > 1e-6, vals)
            if !isempty(valid)
                n_out = min(num_modes_request, length(valid))
                eigenvalues = vals[valid[1:n_out]]
                eigenvectors = vecs[:, valid[1:n_out]]
                solved = true
                diagnostics["solver_backend"] = "dense_symmetric_definite"
                diagnostics["solver_attempts"][end] = Dict("name" => "dense_symmetric_definite", "status" => "succeeded", "returned_modes" => n_out)
                log_msg("[MODES] Dense eigensolver converged ($n_out modes)")
            end
        catch e
            diagnostics["solver_attempts"][end] = Dict("name" => "dense_symmetric_definite", "status" => "failed", "error" => sprint(showerror, e))
            log_msg("[MODES] Dense eigensolver failed: $e")
        end
    end

    # Strategy 2: KrylovKit shift-invert for larger systems
    if !solved
        push!(diagnostics["solver_attempts"], Dict("name" => "krylov_shift_invert", "status" => "attempted"))
        try
            log_msg("[MODES] Using KrylovKit shift-invert ($n_free DOFs)...")
            K_factor, factor_cache_hit = ensure_eigen_solve_factorization!(eigen_ctx)
            diagnostics["eigen_cache"]["factorization_cache_hit"] = factor_cache_hit
            diagnostics["eigen_cache"]["factor_backend"] = eigen_ctx.factor_backend
            if factor_cache_hit
                log_msg("[MODES] Reusing eigen K factorization cache ($(eigen_ctx.factor_backend))")
            else
                log_msg("[MODES] K factorization succeeded ($(eigen_ctx.factor_backend))")
            end
            nev = min(num_modes_request + 5, n_free - 1)
            kd = min(max(2*nev + 10, 30), n_free)

            # K⁻¹M operator: largest θ = 1/ω² → smallest ω
            vals_kk, vecs_kk, info = eigsolve(
                x -> K_factor \ (M_ff * x), randn(n_free), nev, :LM;
                krylovdim=kd, maxiter=500, tol=1e-10, eager=true)

            actual_omegas_sq = Float64[]
            actual_vecs = Vector{Float64}[]
            for (i, theta) in enumerate(vals_kk)
                theta_r = real(theta)
                theta_r < 1e-14 && continue
                abs(imag(theta)) > 1e-6 * abs(theta_r) && continue
                omega_sq = 1.0 / theta_r
                omega_sq > 1e-6 || continue
                push!(actual_omegas_sq, omega_sq)
                push!(actual_vecs, real.(vecs_kk[i]))
            end
            if !isempty(actual_omegas_sq)
                perm = sortperm(actual_omegas_sq)
                n_out = min(num_modes_request, length(perm))
                eigenvalues = [actual_omegas_sq[perm[i]] for i in 1:n_out]
                eigenvectors = hcat([actual_vecs[perm[i]] for i in 1:n_out]...)
                solved = true
                diagnostics["solver_backend"] = "krylov_shift_invert"
                diagnostics["solver_attempts"][end] = Dict("name" => "krylov_shift_invert", "status" => "succeeded", "returned_modes" => n_out, "converged" => info.converged)
                log_msg("[MODES] KrylovKit converged ($n_out modes)")
            end
        catch e
            diagnostics["solver_attempts"][end] = Dict("name" => "krylov_shift_invert", "status" => "failed", "error" => sprint(showerror, e))
            log_msg("[MODES] KrylovKit failed: $(sprint(showerror, e))")
        end
    end

    if !solved
        log_msg("[MODES] ERROR: All eigenvalue solvers failed")
        diagnostics["solver_backend"] = "failed"
        return return_diagnostics ? (Float64[], Float64[], zeros(ndof, 0), diagnostics) : (Float64[], Float64[], zeros(ndof, 0))
    end

    n_out = length(eigenvalues)
    frequencies = sqrt.(abs.(eigenvalues)) ./ (2π)

    # --- EIGRL V1/V2 frequency range filtering ---
    # V1 and V2 are frequency bounds in Hz (Nastran convention)
    has_range = (eigrl_v1 > 0.0 || eigrl_v2 > 0.0)
    if has_range
        v1 = eigrl_v1 > 0.0 ? eigrl_v1 : 0.0
        v2 = eigrl_v2 > 0.0 ? eigrl_v2 : Inf
        range_idx = findall(f -> f >= v1 && f <= v2, frequencies)
        if !isempty(range_idx)
            log_msg("[MODES] EIGRL frequency filter: V1=$v1 Hz, V2=$v2 Hz → $(length(range_idx)) modes in range")
            eigenvalues = eigenvalues[range_idx]
            frequencies = frequencies[range_idx]
            eigenvectors = eigenvectors[:, range_idx]
            n_out = length(eigenvalues)
        else
            log_msg("[MODES] WARNING: No modes found in frequency range [$v1, $v2] Hz, returning all $(n_out) modes")
        end
    end

    # --- Trim to requested number of modes ---
    if n_out > num_modes
        log_msg("[MODES] Trimming from $n_out to $num_modes requested modes")
        eigenvalues = eigenvalues[1:num_modes]
        frequencies = frequencies[1:num_modes]
        eigenvectors = eigenvectors[:, 1:num_modes]
        n_out = num_modes
    end

    # Expand to full DOF set
    mode_shapes = zeros(ndof, n_out)
    for m in 1:n_out
        @inbounds for (row, dof) in pairs(free_dofs)
            mode_shapes[dof, m] = eigenvectors[row, m]
        end
    end

    # Transform to global coordinates
    mode_shapes_global = zeros(ndof, n_out)
    for idx in values(id_map)
        base = (idx-1)*6
        R = node_R[idx]
        for m in 1:n_out
            u1 = mode_shapes[base + 1, m]
            u2 = mode_shapes[base + 2, m]
            u3 = mode_shapes[base + 3, m]
            r1 = mode_shapes[base + 4, m]
            r2 = mode_shapes[base + 5, m]
            r3 = mode_shapes[base + 6, m]
            mode_shapes_global[base + 1, m] = R[1, 1] * u1 + R[1, 2] * u2 + R[1, 3] * u3
            mode_shapes_global[base + 2, m] = R[2, 1] * u1 + R[2, 2] * u2 + R[2, 3] * u3
            mode_shapes_global[base + 3, m] = R[3, 1] * u1 + R[3, 2] * u2 + R[3, 3] * u3
            mode_shapes_global[base + 4, m] = R[1, 1] * r1 + R[1, 2] * r2 + R[1, 3] * r3
            mode_shapes_global[base + 5, m] = R[2, 1] * r1 + R[2, 2] * r2 + R[2, 3] * r3
            mode_shapes_global[base + 6, m] = R[3, 1] * r1 + R[3, 2] * r2 + R[3, 3] * r3
        end
    end

    norm_mode = uppercase(strip(eigrl_norm))
    isempty(norm_mode) && (norm_mode = "MASS")
    if norm_mode != "MASS" && norm_mode != "MAX"
        log_msg("[MODES] WARNING: Unsupported EIGRL NORM=$norm_mode, using MASS normalization")
        norm_mode = "MASS"
    end

    # Normalize mode shapes according to the requested EIGRL NORM.
    for m in 1:n_out
        if norm_mode == "MAX"
            max_val = _matrix_column_max_abs(mode_shapes_global, m)
            if max_val > 1e-30
                _scale_matrix_column!(mode_shapes_global, m, 1.0 / max_val)
            end
        else
            phi = mode_shapes[free_dofs, m]  # use analysis-frame eigenvector
            gen_mass = dot(phi, M_ff * phi)
            if gen_mass > 1e-30
                scale = 1.0 / sqrt(gen_mass)
                _scale_matrix_column!(mode_shapes_global, m, scale)
            else
                max_val = _matrix_column_max_abs(mode_shapes_global, m)
                if max_val > 1e-30
                    _scale_matrix_column!(mode_shapes_global, m, 1.0 / max_val)
                end
            end
        end
    end

    log_msg("[MODES] Natural frequencies:")
    for (i, f) in enumerate(frequencies)
        log_msg("  Mode $i: f = $(round(f, digits=4)) Hz (ω² = $(round(eigenvalues[i], sigdigits=6)))")
    end

    diagnostics["returned_modes"] = n_out
    diagnostics["returned_modes"] = n_out
    return return_diagnostics ? (eigenvalues, frequencies, mode_shapes_global, diagnostics) : (eigenvalues, frequencies, mode_shapes_global)
end
