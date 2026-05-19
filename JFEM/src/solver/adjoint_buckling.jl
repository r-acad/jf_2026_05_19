# adjoint_buckling.jl — Adjoint sensitivity for buckling eigenvalues (SOL 105)
#
# Sensitivity of buckling load factor λ to design variables x:
#
#   dλ/dx = -1/(φᵀ·Kg·φ) · [φᵀ·dK/dx·φ + λ·φᵀ·∂Kg/∂x|u_fixed·φ - ψᵀ·dK/dx·u]
#
# where ψ solves the adjoint equation:
#   K·ψ = λ · ∂(φᵀ·Kg(σ(u))·φ)/∂u
#
# The adjoint RHS ∂(φᵀ·Kg·φ)/∂u exploits that Kg is linear in stress σ:
#   For shells: Kg(σ) = σ₁·Kg₁ + σ₂·Kg₂ + σ₃·Kg₃  (unit stress geometric stiffnesses)
#   For bars:   Kg(P) = P·Kg_unit                      (unit axial force)

# ============================================================================
# Full-model FD: reassemble K at perturbed parameter for exact dK/dx bilinear forms
# ============================================================================

"""
Compute φᵀ·dK/dx·φ and ψᵀ·dK/dx·u via full model reassembly FD.
Returns Dict{group_label => (phi_dK_phi, psi_dK_u)}.
"""
function _full_model_dKdx_matrix_diffs(dv, model;
                                       bending_incomp::Bool=true,
                                       shear_center_only::Bool=false,
                                       membrane_incomp::Bool=true,
                                       pcomp_membrane_incomp::Bool=false,
                                       snorm_angle_override::Union{Nothing,Float64}=nothing,
                                       iso_no_incomp::Bool=false)
    dv_type = dv["type"]
    result = Dict{String, SparseMatrixCSC{Float64, Int}}()

    function _assemble_diff!(label, apply_plus!, apply_minus!, restore!)
        apply_plus!()
        K_plus, = assemble_stiffness(
            model;
            bending_incomp=bending_incomp,
            shear_center_only=shear_center_only,
            membrane_incomp=membrane_incomp,
            pcomp_membrane_incomp=pcomp_membrane_incomp,
            snorm_angle_override=snorm_angle_override,
            iso_no_incomp=iso_no_incomp,
        )

        apply_minus!()
        K_minus, = assemble_stiffness(
            model;
            bending_incomp=bending_incomp,
            shear_center_only=shear_center_only,
            membrane_incomp=membrane_incomp,
            pcomp_membrane_incomp=pcomp_membrane_incomp,
            snorm_angle_override=snorm_angle_override,
            iso_no_incomp=iso_no_incomp,
        )

        restore!()
        result[label] = K_plus - K_minus
    end

    if dv_type == "shell_thickness"
        pshells = model["PSHELLs"]
        for pid in dv["pids"]
            pid_str = string(Int(pid))
            h0 = pshells[pid_str]["T"]
            delta = max(abs(h0) * 1e-5, 1e-12)
            _assemble_diff!(
                "PID_$pid_str",
                () -> (pshells[pid_str]["T"] = h0 + delta),
                () -> (pshells[pid_str]["T"] = h0 - delta),
                () -> (pshells[pid_str]["T"] = h0),
            )
            result["PID_$pid_str"] ./= (2 * delta)
        end
    elseif dv_type == "material_NU"
        mats = model["MATs"]
        for mid in dv["mids"]
            mid_str = string(Int(mid))
            nu0 = mats[mid_str]["NU"]
            G0 = mats[mid_str]["G"]
            E0 = mats[mid_str]["E"]
            delta = max(abs(nu0) * 1e-5, 1e-8)
            _assemble_diff!(
                "MID_$mid_str",
                () -> begin
                    mats[mid_str]["NU"] = nu0 + delta
                    mats[mid_str]["G"] = E0 / (2 * (1 + nu0 + delta))
                end,
                () -> begin
                    mats[mid_str]["NU"] = nu0 - delta
                    mats[mid_str]["G"] = E0 / (2 * (1 + nu0 - delta))
                end,
                () -> begin
                    mats[mid_str]["NU"] = nu0
                    mats[mid_str]["G"] = G0
                end,
            )
            result["MID_$mid_str"] ./= (2 * delta)
        end
    elseif dv_type == "bar_area"
        pbarls = get(model, "PBARLs", Dict())
        for pid in dv["pids"]
            pid_str = string(Int(pid))
            A0 = pbarls[pid_str]["A"]
            delta = max(abs(A0) * 1e-5, 1e-12)
            _assemble_diff!(
                "PID_$pid_str",
                () -> (pbarls[pid_str]["A"] = A0 + delta),
                () -> (pbarls[pid_str]["A"] = A0 - delta),
                () -> (pbarls[pid_str]["A"] = A0),
            )
            result["PID_$pid_str"] ./= (2 * delta)
        end
    elseif dv_type == "node_coord"
        grid = Int(dv["grid"]); comp = Int(dv["comp"])
        grid_str = string(grid)
        coords_arr = model["GRIDs"][grid_str]["X"]
        x0 = Float64(coords_arr[comp])
        delta = max(abs(x0) * 1e-5, 1e-8)
        label = "GRID_$(grid)_$(comp)"
        _assemble_diff!(
            label,
            () -> (coords_arr[comp] = x0 + delta),
            () -> (coords_arr[comp] = x0 - delta),
            () -> (coords_arr[comp] = x0),
        )
        result[label] ./= (2 * delta)
    end

    return result
end

function _full_model_dKdx_bilinear(dv, model, phi, u_static, psi, ndof)
    diffs = _full_model_dKdx_matrix_diffs(dv, model)
    result = Dict{String, Tuple{Float64, Float64}}()
    for (group_label, dKdx) in diffs
        result[group_label] = (dot(phi, dKdx * phi), dot(psi, dKdx * u_static))
    end
    return result
end

@inline function _sol105_eig_stiffness_fd_kwargs()
    return (
        bending_incomp=sol105_eig_bending_incomp_enabled(),
        shear_center_only=true,
        membrane_incomp=solver_env_bool("JFEM_SOL105_EIG_MEMBRANE_INCOMP", false),
        pcomp_membrane_incomp=solver_env_bool("JFEM_SOL105_EIG_PCOMP_MEMBRANE_INCOMP", false),
        snorm_angle_override=sol105_snorm_angle_override(),
        iso_no_incomp=true,
    )
end

function _contract_full_model_dKdx_split_bilinear(static_diffs, eig_diffs, phi, u_static, psi)
    result = Dict{String, Tuple{Float64, Float64}}()
    group_labels = Set{String}()
    union!(group_labels, keys(static_diffs))
    union!(group_labels, keys(eig_diffs))

    for group_label in group_labels
        phi_dK_phi = haskey(eig_diffs, group_label) ? dot(phi, eig_diffs[group_label] * phi) : 0.0
        psi_dK_u = haskey(static_diffs, group_label) ? dot(psi, static_diffs[group_label] * u_static) : 0.0
        result[group_label] = (phi_dK_phi, psi_dK_u)
    end

    return result
end

function _full_model_dKdx_split_bilinear(dv, model, phi, u_static, psi, ndof)
    static_diffs = _full_model_dKdx_matrix_diffs(dv, model)
    eig_diffs = _full_model_dKdx_matrix_diffs(dv, model; _sol105_eig_stiffness_fd_kwargs()...)
    return _contract_full_model_dKdx_split_bilinear(static_diffs, eig_diffs, phi, u_static, psi)
end

@inline function _buckling_quad4_ctx_requires_full_model_dKdx(ctx)
    return ctx.use_iso_exact_membrane || ctx.use_covariant_operator || ctx.use_covariant_membrane
end

function _buckling_dKdx_group_labels(dv)
    dv_type = dv["type"]
    if dv_type == "shell_thickness"
        return Set("PID_$(Int(pid))" for pid in dv["pids"])
    elseif dv_type == "material_E"
        return Set("MID_$(Int(mid))" for mid in dv["mids"])
    elseif dv_type == "material_NU"
        return Set("MID_$(Int(mid))" for mid in dv["mids"])
    elseif dv_type == "bar_area"
        return Set("PID_$(Int(pid))" for pid in dv["pids"])
    elseif dv_type == "topology_density"
        return Set("EID_$(Int(eid))" for eid in dv["eids"])
    end
    return Set{String}()
end

function _subset_dv_by_group_labels(dv, keep_labels::Set{String})
    isempty(keep_labels) && return nothing

    dv_type = dv["type"]
    registry = get(DV_REGISTRY, dv_type, nothing)
    if isnothing(registry) || isempty(registry.key_field)
        return Dict{String, Any}(String(k) => v for (k, v) in pairs(dv))
    end

    key_field = registry.key_field
    prefix = registry.prefix
    values = get(dv, key_field, nothing)
    if isnothing(values)
        return nothing
    end

    kept_values = [val for val in values if "$(prefix)_$(Int(val))" in keep_labels]
    isempty(kept_values) && return nothing

    dv_subset = Dict{String, Any}(String(k) => v for (k, v) in pairs(dv))
    dv_subset[key_field] = kept_values
    return dv_subset
end

"""
Identify DV groups whose buckling dK/dx term should use full-model FD so the
static stiffness derivative stays synchronized with the active specialized
quadrilateral-shell branch used by the forward SOL 105 solve.
"""
function _buckling_dKdx_full_model_group_labels(dv, model, id_map, node_coords, node_R, u_global)
    dv_type = dv["type"]
    if !(dv_type in ("shell_thickness", "material_NU"))
        return Set{String}()
    end

    cshells = get(model, "CSHELLs", Dict())
    isempty(cshells) && return Set{String}()

    labels = Set{String}()
    support = _covariant_iso_quad4_support(model, id_map, node_coords)
    if dv_type == "shell_thickness"
        target_pids = Set(string.(Int.(dv["pids"])))
        for (_, el) in cshells
            pid_str = string(el["PID"])
            pid_str in target_pids || continue
            ctx = _covariant_iso_quad4_kg_context(
                el, model, id_map, node_coords, node_R, u_global, support
            )
            if !isnothing(ctx) && _buckling_quad4_ctx_requires_full_model_dKdx(ctx)
                push!(labels, "PID_$pid_str")
            end
        end
    else
        target_mids = Set(string.(Int.(dv["mids"])))
        for (_, el) in cshells
            ctx = _covariant_iso_quad4_kg_context(
                el, model, id_map, node_coords, node_R, u_global, support
            )
            if !isnothing(ctx) &&
               ctx.mid_str in target_mids &&
               _buckling_quad4_ctx_requires_full_model_dKdx(ctx)
                push!(labels, "MID_$(ctx.mid_str)")
            end
        end
    end

    return labels
end

@inline function _buckling_mode_mac(phi_ref, phi_cur)
    num = abs(dot(phi_ref, phi_cur))
    den = max(norm(phi_ref) * norm(phi_cur), 1e-30)
    return num / den
end

function _track_buckling_eigenvalues_by_mac(base_mode_shapes, current_mode_shapes, current_eigenvalues)
    n_ref = size(base_mode_shapes, 2)
    n_cur = size(current_mode_shapes, 2)
    n_match = min(n_ref, n_cur)
    tracked = fill(NaN, n_match)
    available = collect(1:n_cur)

    for i in 1:n_match
        phi_ref = base_mode_shapes[:, i]
        best_idx = 0
        best_mac = -1.0
        for j in available
            mac = _buckling_mode_mac(phi_ref, current_mode_shapes[:, j])
            if mac > best_mac
                best_mac = mac
                best_idx = j
            end
        end
        tracked[i] = current_eigenvalues[best_idx]
        filter!(x -> x != best_idx, available)
    end

    return tracked
end

@inline function _buckling_quad4_ctx_requires_full_sensitivity_fd(ctx)
    return ctx.use_iso_exact_membrane
end

function _buckling_dKg_full_model_group_labels(dv, model, id_map, node_coords, node_R, u_global)
    dv_type = dv["type"]
    if !(dv_type in ("shell_thickness", "material_NU"))
        return Set{String}()
    end

    cshells = get(model, "CSHELLs", Dict())
    isempty(cshells) && return Set{String}()

    labels = Set{String}()
    support = _covariant_iso_quad4_support(model, id_map, node_coords)
    if dv_type == "shell_thickness"
        target_pids = Set(string.(Int.(dv["pids"])))
        for (_, el) in cshells
            pid_str = string(el["PID"])
            pid_str in target_pids || continue
            ctx = _covariant_iso_quad4_kg_context(
                el, model, id_map, node_coords, node_R, u_global, support
            )
            if !isnothing(ctx) && ctx.use_iso_exact_membrane
                push!(labels, "PID_$pid_str")
            end
        end
    else
        target_mids = Set(string.(Int.(dv["mids"])))
        for (_, el) in cshells
            ctx = _covariant_iso_quad4_kg_context(
                el, model, id_map, node_coords, node_R, u_global, support
            )
            if !isnothing(ctx) && ctx.mid_str in target_mids && ctx.use_iso_exact_membrane
                push!(labels, "MID_$(ctx.mid_str)")
            end
        end
    end

    return labels
end

function _buckling_full_sensitivity_fd_group_labels(dv, model, id_map, node_coords, node_R, u_global)
    dv_type = dv["type"]
    if !(dv_type in ("shell_thickness", "material_NU"))
        return Set{String}()
    end

    cshells = get(model, "CSHELLs", Dict())
    isempty(cshells) && return Set{String}()

    labels = Set{String}()
    support = _covariant_iso_quad4_support(model, id_map, node_coords)
    if dv_type == "shell_thickness"
        target_pids = Set(string.(Int.(dv["pids"])))
        for (_, el) in cshells
            pid_str = string(el["PID"])
            pid_str in target_pids || continue
            ctx = _covariant_iso_quad4_kg_context(
                el, model, id_map, node_coords, node_R, u_global, support
            )
            if !isnothing(ctx) && _buckling_quad4_ctx_requires_full_sensitivity_fd(ctx)
                push!(labels, "PID_$pid_str")
            end
        end
    else
        target_mids = Set(string.(Int.(dv["mids"])))
        for (_, el) in cshells
            ctx = _covariant_iso_quad4_kg_context(
                el, model, id_map, node_coords, node_R, u_global, support
            )
            if !isnothing(ctx) &&
               ctx.mid_str in target_mids &&
               _buckling_quad4_ctx_requires_full_sensitivity_fd(ctx)
                push!(labels, "MID_$(ctx.mid_str)")
            end
        end
    end

    return labels
end

function _full_model_buckling_fd_group_sensitivities(dv, results, group_labels::Set{String})
    isempty(group_labels) && return Dict{String, Vector{Float64}}()

    base_model = results["model"]
    base_mode_shapes = results["_raw_mode_shapes"]
    n_modes = length(results["eigenvalues"])
    sensitivities = Dict{String, Vector{Float64}}()

    function _solve_tracked(model_pert)
        # `solve_model` lives in the parent of the Solver submodule
        # (Main when this file is include()-loaded, OpenJFEM when it's loaded
        # as part of the OpenJFEM package).
        pert_results = parentmodule(@__MODULE__).solve_model(model_pert)
        return _track_buckling_eigenvalues_by_mac(
            base_mode_shapes,
            pert_results["_raw_mode_shapes"],
            pert_results["eigenvalues"],
        )
    end

    if dv["type"] == "shell_thickness"
        for group_label in group_labels
            pid_str = split(group_label, "_", limit=2)[2]
            h0 = Float64(base_model["PSHELLs"][pid_str]["T"])
            delta = max(abs(h0) * 1e-5, 1e-12)

            model_plus = deepcopy(base_model)
            model_plus["PSHELLs"][pid_str]["T"] = h0 + delta
            lam_plus = _solve_tracked(model_plus)

            model_minus = deepcopy(base_model)
            model_minus["PSHELLs"][pid_str]["T"] = h0 - delta
            lam_minus = _solve_tracked(model_minus)

            sensitivities[group_label] = [
                (lam_plus[i] - lam_minus[i]) / (2.0 * delta) for i in 1:n_modes
            ]
        end
    elseif dv["type"] == "material_NU"
        for group_label in group_labels
            mid_str = split(group_label, "_", limit=2)[2]
            nu0 = Float64(base_model["MATs"][mid_str]["NU"])
            E0 = Float64(base_model["MATs"][mid_str]["E"])
            delta = max(abs(nu0) * 1e-5, 1e-8)

            model_plus = deepcopy(base_model)
            model_plus["MATs"][mid_str]["NU"] = nu0 + delta
            model_plus["MATs"][mid_str]["G"] = E0 / (2.0 * (1.0 + nu0 + delta))
            lam_plus = _solve_tracked(model_plus)

            model_minus = deepcopy(base_model)
            model_minus["MATs"][mid_str]["NU"] = nu0 - delta
            model_minus["MATs"][mid_str]["G"] = E0 / (2.0 * (1.0 + nu0 - delta))
            lam_minus = _solve_tracked(model_minus)

            sensitivities[group_label] = [
                (lam_plus[i] - lam_minus[i]) / (2.0 * delta) for i in 1:n_modes
            ]
        end
    end

    return sensitivities
end


function _full_model_dKg_dx_matrix_diffs(dv, model, u_global)
    dv_type = dv["type"]
    result = Dict{String, SparseMatrixCSC{Float64, Int}}()
    snorm_angle = sol105_snorm_angle_override()

    function _kg_diff(label, apply_plus!, apply_minus!, restore!, delta)
        apply_plus!()
        _, id_map_p, nc_p, ndof_p, nr_p, _, rbe3_p, snorm_p, _ = assemble_stiffness(
            model; snorm_angle_override=snorm_angle
        )
        Kg_plus = assemble_geometric_stiffness(
            model, id_map_p, nc_p, nr_p, ndof_p, u_global, snorm_p, rbe3_p;
            snorm_angle_override=snorm_angle,
        )

        apply_minus!()
        _, id_map_m, nc_m, ndof_m, nr_m, _, rbe3_m, snorm_m, _ = assemble_stiffness(
            model; snorm_angle_override=snorm_angle
        )
        Kg_minus = assemble_geometric_stiffness(
            model, id_map_m, nc_m, nr_m, ndof_m, u_global, snorm_m, rbe3_m;
            snorm_angle_override=snorm_angle,
        )

        restore!()
        result[label] = (Kg_plus - Kg_minus) / (2.0 * delta)
    end

    if dv_type == "shell_thickness"
        pshells = model["PSHELLs"]
        for pid in dv["pids"]
            pid_str = string(Int(pid))
            h0 = Float64(pshells[pid_str]["T"])
            delta = max(abs(h0) * 1e-5, 1e-12)
            _kg_diff(
                "PID_$pid_str",
                () -> (pshells[pid_str]["T"] = h0 + delta),
                () -> (pshells[pid_str]["T"] = h0 - delta),
                () -> (pshells[pid_str]["T"] = h0),
                delta,
            )
        end
    elseif dv_type == "material_NU"
        mats = model["MATs"]
        for mid in dv["mids"]
            mid_str = string(Int(mid))
            nu0 = Float64(mats[mid_str]["NU"])
            E0 = Float64(mats[mid_str]["E"])
            G0 = Float64(mats[mid_str]["G"])
            delta = max(abs(nu0) * 1e-5, 1e-8)
            _kg_diff(
                "MID_$mid_str",
                () -> begin
                    mats[mid_str]["NU"] = nu0 + delta
                    mats[mid_str]["G"] = E0 / (2.0 * (1.0 + nu0 + delta))
                end,
                () -> begin
                    mats[mid_str]["NU"] = nu0 - delta
                    mats[mid_str]["G"] = E0 / (2.0 * (1.0 + nu0 - delta))
                end,
                () -> begin
                    mats[mid_str]["NU"] = nu0
                    mats[mid_str]["G"] = G0
                end,
                delta,
            )
        end
    end

    return result
end

function _full_model_dKg_dx_phi(dv, model, u_global, phi)
    diffs = _full_model_dKg_dx_matrix_diffs(dv, model, u_global)
    result = Dict{String, Float64}()
    for (group_label, dKgdx) in diffs
        result[group_label] = dot(phi, dKgdx * phi)
    end
    return result
end

# ============================================================================
# Adjoint RHS: ∂(φᵀ·Kg(σ(u))·φ)/∂u  (assembled over all elements)
# ============================================================================

"""
    compute_buckling_adjoint_rhs(model, id_map, node_coords, node_R, u_global, phi, ndof) -> Vector{Float64}

Compute the adjoint RHS vector f = ∂(φᵀ·Kg(σ(u))·φ)/∂u.
This is assembled element-by-element using the linearity of Kg in stress.
"""
function compute_buckling_adjoint_rhs(model, id_map, node_coords, node_R, u_global, phi, ndof)
    f_adj = zeros(ndof)

    # Shell elements (QUAD4, TRIA3)
    _buckling_rhs_shells!(f_adj, model, id_map, node_coords, node_R, u_global, phi)

    # Bar/beam elements
    _buckling_rhs_bars!(f_adj, model, id_map, node_coords, node_R, u_global, phi)

    # Rod elements
    _buckling_rhs_rods!(f_adj, model, id_map, node_coords, node_R, u_global, phi)

    return f_adj
end

# ============================================================================
# Shell elements: ∂(φᵀ·Kg·φ)/∂u via unit-stress geometric stiffness
# ============================================================================

@inline function _quad4_is_flat_for_buckling_rhs(p1::SVector{3,Float64},
                                                 p2::SVector{3,Float64},
                                                 p3::SVector{3,Float64},
                                                 p4::SVector{3,Float64})
    d13 = p3 - p1
    d24 = p4 - p2
    v3_geom_raw = cross(d13, d24)
    v3_geom_len = norm(v3_geom_raw)
    if v3_geom_len <= 1e-12
        return true
    end
    c_geom = (p1 + p2 + p3 + p4) / 4.0
    v3g = v3_geom_raw / v3_geom_len
    max_dev = maximum((
        abs(dot(p1 - c_geom, v3g)),
        abs(dot(p2 - c_geom, v3g)),
        abs(dot(p3 - c_geom, v3g)),
        abs(dot(p4 - c_geom, v3g)),
    ))
    L_diag = max(norm(d13), norm(d24))
    return max_dev < 1e-6 * max(L_diag, 1e-12)
end

function _quad4_buckling_local_data(el, model, id_map, node_coords, node_R)
    nids = get(el, "NODES", Int[])
    length(nids) == 4 || return nothing
    if !all(n -> haskey(id_map, n), nids)
        return nothing
    end

    pid_str = string(el["PID"])
    prop = get(model["PSHELLs"], pid_str, nothing)
    isnothing(prop) && return nothing

    mid_str = string(prop["MID"])
    mat = get(model["MATs"], mid_str, nothing)
    isnothing(mat) && return nothing

    idxs = [id_map[n] for n in nids]
    p1 = SVector{3}(node_coords[idxs[1], 1], node_coords[idxs[1], 2], node_coords[idxs[1], 3])
    p2 = SVector{3}(node_coords[idxs[2], 1], node_coords[idxs[2], 2], node_coords[idxs[2], 3])
    p3 = SVector{3}(node_coords[idxs[3], 1], node_coords[idxs[3], 2], node_coords[idxs[3], 3])
    p4 = SVector{3}(node_coords[idxs[4], 1], node_coords[idxs[4], 2], node_coords[idxs[4], 3])

    q4_frame_mode = q4_frame_mode_from_env("JFEM_Q4_FRAME_MODE_KG")
    v1, v2, v3 = shell_element_frame_quad4(p1, p2, p3, p4, q4_frame_mode)
    c_center = (p1 + p2 + p3 + p4) / 4.0

    lc = zeros(4, 2)
    lc[1, 1] = dot(p1 - c_center, v1); lc[1, 2] = dot(p1 - c_center, v2)
    lc[2, 1] = dot(p2 - c_center, v1); lc[2, 2] = dot(p2 - c_center, v2)
    lc[3, 1] = dot(p3 - c_center, v1); lc[3, 2] = dot(p3 - c_center, v2)
    lc[4, 1] = dot(p4 - c_center, v1); lc[4, 2] = dot(p4 - c_center, v2)

    ndof_elem = 24
    T_mat = zeros(ndof_elem, ndof_elem)
    Rel = [v1[1] v1[2] v1[3]; v2[1] v2[2] v2[3]; v3[1] v3[2] v3[3]]
    for k in 1:4
        TR = Rel * node_R[idxs[k]]
        base = (k - 1) * 6
        T_mat[base+1:base+3, base+1:base+3] = TR
        T_mat[base+4:base+6, base+4:base+6] = TR
    end

    dofs = Vector{Int}(undef, ndof_elem)
    for k in 1:4
        base_g = (idxs[k] - 1) * 6
        base_e = (k - 1) * 6
        for d in 1:6
            dofs[base_e + d] = base_g + d
        end
    end

    return (
        n_nodes = 4,
        lc = lc,
        T_mat = T_mat,
        dofs = dofs,
        ndof_elem = ndof_elem,
        prop = prop,
        mat = mat,
        pid_str = pid_str,
        mid_str = mid_str,
        idxs = idxs,
        p1 = p1,
        p2 = p2,
        p3 = p3,
        p4 = p4,
        v1 = v1,
        v2 = v2,
        v3 = v3,
    )
end

function _rotate_pcomp_kg_constitutive(prop, el, qd)
    Cm_kg = copy(prop["Cm"])
    Cb_kg = copy(prop["Cb"])
    Cs_kg = copy(prop["Cs"])

    theta_rad = deg2rad(Float64(get(el, "THETA", 0.0)))
    beta = shell_pcomp_kg_rotation(
        q4_pcomp_kg_axis_mode(),
        qd.v1, qd.v2, qd.v3,
        qd.p1, qd.p2, qd.p3, qd.p4,
        theta_rad,
        Int(get(el, "MCID", 0)),
        model["CORDs"],
    )

    if abs(beta) > 1e-10
        cb = cos(beta)
        sb = sin(beta)
        c2 = cb^2
        s2 = sb^2
        cs = cb * sb
        _rotate_constitutive_3x3!(Cm_kg, c2, s2, cs, s2, c2, -cs, -2cs, 2cs, c2 - s2)
        _rotate_constitutive_3x3!(Cb_kg, c2, s2, cs, s2, c2, -cs, -2cs, 2cs, c2 - s2)

        a11 = Cs_kg[1, 1]
        a12 = Cs_kg[1, 2]
        a22 = Cs_kg[2, 2]
        Cs_kg[1, 1] = cb^2 * a11 + 2.0 * cb * sb * a12 + sb^2 * a22
        Cs_kg[1, 2] = -cb * sb * a11 + (cb^2 - sb^2) * a12 + cb * sb * a22
        Cs_kg[2, 1] = Cs_kg[1, 2]
        Cs_kg[2, 2] = sb^2 * a11 - 2.0 * cb * sb * a12 + cb^2 * a22
    end

    return Cm_kg, Cb_kg, Cs_kg, beta
end

function _flat_pcomp_quad4_sigma_input(lc,
                                       u_local,
                                       E::Float64,
                                       nu::Float64,
                                       h::Float64,
                                       Cm_kg,
                                       beta::Float64;
                                       compatible_only::Bool,
                                       use_incompatible_modes::Bool,
                                       kg_membrane_assumed_mode::Symbol,
                                       membrane_incomp_center_jacobian::Bool,
                                       force_use_gp_sigma::Union{Nothing,Bool}=nothing,
                                       kg_shell_nxy::Float64=1.0,
                                       kg_shell_pcomp_nxy::Float64=1.0,
                                       kg_shell_pcomp_nxy_compression_only::Bool=false)
    N_gp, N_res, _ = FEM.quad4_membrane_force_field(
        lc, u_local, E, nu, h;
        Cm_override = Cm_kg,
        compatible_only = compatible_only,
        use_incompatible_modes = use_incompatible_modes,
        curvature_membrane = nothing,
        membrane_shear_center_row = false,
        material_shear_rotation = beta,
        membrane_assumed_mode = kg_membrane_assumed_mode,
        membrane_incomp_center_jacobian = membrane_incomp_center_jacobian,
    )

    use_gp_sigma = isnothing(force_use_gp_sigma) ? kg_quad4_use_gp_field(N_gp, N_res) : force_use_gp_sigma
    sigma_input = use_gp_sigma ? N_gp ./ h : N_res ./ h

    if kg_shell_nxy != 1.0
        if sigma_input isa AbstractMatrix
            @inbounds for gp in 1:size(sigma_input, 1)
                sigma_input[gp, 3] *= kg_shell_nxy
            end
        else
            sigma_input[3] *= kg_shell_nxy
        end
    end
    kg_shell_apply_pcomp_nxy_scale!(sigma_input, kg_shell_pcomp_nxy, kg_shell_pcomp_nxy_compression_only)

    return sigma_input, use_gp_sigma
end

function _flat_pcomp_quad4_kg_context(el, model, id_map, node_coords, node_R, u_global)
    qd = _quad4_buckling_local_data(el, model, id_map, node_coords, node_R)
    isnothing(qd) && return nothing

    prop = qd.prop
    is_pcomp_clt = get(prop, "TYPE", "") == "PCOMP_CLT" && haskey(prop, "Cm")
    pcomp_is_isotropic = is_pcomp_clt && get(prop, "IS_ISOTROPIC", false)
    if !is_pcomp_clt || pcomp_is_isotropic || get(prop, "Bmb", nothing) !== nothing
        return nothing
    end
    if !_quad4_is_flat_for_buckling_rhs(qd.p1, qd.p2, qd.p3, qd.p4)
        return nothing
    end
    if maximum(abs, prop["Cb"]) <= 1e-30
        return nothing
    end

    E = Float64(get(qd.mat, "E", 0.0))
    nu = Float64(get(qd.mat, "NU", 0.0))
    h = Float64(prop["T"])
    theta_rad = deg2rad(Float64(get(el, "THETA", 0.0)))
    shear_ratio, d16_ratio, _ = pcomp_metric_ratios(prop, theta_rad)

    kg_flat_dkmq_branch = q4_sol105_flat_pcomp_dkmq_enabled()
    kg_flat_plate_auto = q4_sol105_flat_pcomp_plate_auto_enabled() &&
                         FEM.quad4_is_axis_aligned_rectangle(qd.lc) &&
                         d16_ratio <= q4_sol105_flat_pcomp_plate_auto_d16_ratio_max() &&
                         shear_ratio <= q4_sol105_flat_pcomp_plate_auto_shear_ratio_max()
    kg_flat_plate_branch = q4_sol105_flat_pcomp_plate_branch_enabled() || kg_flat_plate_auto
    operator = if kg_flat_dkmq_branch
        :plate_dkmq
    elseif q4_sol105_flat_pcomp_rect_adini_enabled() &&
           kg_flat_plate_branch &&
           FEM.quad4_is_axis_aligned_rectangle(qd.lc)
        :plate_adini
    elseif kg_flat_plate_branch
        :plate_dkq
    elseif q4_sol105_flat_pcomp_plate_like_kg_enabled()
        :generic_normal_only
    else
        :generic_full
    end

    Cm_kg, Cb_kg, Cs_kg, beta = _rotate_pcomp_kg_constitutive(prop, el, qd)

    compatible_only = kg_use_compatible_membrane_stress()
    use_incompatible_modes = solver_env_bool("JFEM_SOL105_EIG_PCOMP_MEMBRANE_INCOMP", false)
    kg_membrane_assumed_mode = q4_flat_pcomp_eig_membrane_assumed_mode()
    membrane_incomp_center_jacobian = q4_sol105_membrane_incomp_center_jacobian_enabled()
    kg_shell_nxy = kg_shell_nxy_scale()
    kg_shell_pcomp_nxy = kg_shell_pcomp_nxy_scale()
    kg_shell_pcomp_nxy_compression_only_v = kg_shell_pcomp_nxy_compression_only()

    u_elem = zeros(qd.ndof_elem)
    for i in 1:qd.ndof_elem
        u_elem[i] = u_global[qd.dofs[i]]
    end
    u_local = qd.T_mat * u_elem
    sigma_input, use_gp_sigma = _flat_pcomp_quad4_sigma_input(
        qd.lc,
        u_local,
        E,
        nu,
        h,
        Cm_kg,
        beta;
        compatible_only = compatible_only,
        use_incompatible_modes = use_incompatible_modes,
        kg_membrane_assumed_mode = kg_membrane_assumed_mode,
        membrane_incomp_center_jacobian = membrane_incomp_center_jacobian,
        kg_shell_nxy = kg_shell_nxy,
        kg_shell_pcomp_nxy = kg_shell_pcomp_nxy,
        kg_shell_pcomp_nxy_compression_only = kg_shell_pcomp_nxy_compression_only_v,
    )

    return (
        qd = qd,
        h = h,
        E = E,
        nu = nu,
        operator = operator,
        sigma_input = sigma_input,
        use_gp_sigma = use_gp_sigma,
        Cm_kg = Cm_kg,
        Cb_kg = Cb_kg,
        Cs_kg = Cs_kg,
        beta = beta,
        compatible_only = compatible_only,
        use_incompatible_modes = use_incompatible_modes,
        kg_membrane_assumed_mode = kg_membrane_assumed_mode,
        membrane_incomp_center_jacobian = membrane_incomp_center_jacobian,
        kg_shell_nxy = kg_shell_nxy,
        kg_shell_pcomp_nxy = kg_shell_pcomp_nxy,
        kg_shell_pcomp_nxy_compression_only = kg_shell_pcomp_nxy_compression_only_v,
        trans_mode = operator === :generic_normal_only ? :normal_only : kg_shell_trans_mode(),
        rot_grad_scale = 0.0,
        curvature_sign = kg_shell_curvature_sign(),
    )
end

function _flat_pcomp_quad4_kg_local(ctx, sigma_input)
    if ctx.operator === :plate_dkmq
        return FEM.geometric_stiffness_quad4_plate_dkmq(
            ctx.qd.lc, sigma_input, ctx.h, ctx.Cb_kg, ctx.Cs_kg
        )
    elseif ctx.operator === :plate_adini
        return FEM.geometric_stiffness_quad4_plate_adini(ctx.qd.lc, sigma_input, ctx.h)
    elseif ctx.operator === :plate_dkq
        return FEM.geometric_stiffness_quad4_plate_dkq(
            ctx.qd.lc, sigma_input, ctx.h, ctx.Cb_kg, ctx.Cs_kg
        )
    end

    return FEM.geometric_stiffness_quad4(
        ctx.qd.lc, sigma_input, ctx.h;
        trans_mode = ctx.trans_mode,
        curvature = nothing,
        curvature_sign = ctx.curvature_sign,
        rot_grad_scale = ctx.rot_grad_scale,
        membrane_shear_center_row = false,
        Cm = ctx.Cm_kg,
        membrane_incomp = false,
        material_shear_rotation = ctx.beta,
        membrane_assumed_mode = ctx.kg_membrane_assumed_mode,
        membrane_incomp_center_jacobian = ctx.membrane_incomp_center_jacobian,
    )
end

function _flat_pcomp_quad4_rhs_local(ctx, phi_local)
    f_local = zeros(ctx.qd.ndof_elem)
    basis = zeros(ctx.qd.ndof_elem)

    for j in 1:ctx.qd.ndof_elem
        fill!(basis, 0.0)
        basis[j] = 1.0
        sigma_basis, _ = _flat_pcomp_quad4_sigma_input(
            ctx.qd.lc,
            basis,
            ctx.E,
            ctx.nu,
            ctx.h,
            ctx.Cm_kg,
            ctx.beta;
            compatible_only = ctx.compatible_only,
            use_incompatible_modes = ctx.use_incompatible_modes,
            kg_membrane_assumed_mode = ctx.kg_membrane_assumed_mode,
            membrane_incomp_center_jacobian = ctx.membrane_incomp_center_jacobian,
            force_use_gp_sigma = ctx.use_gp_sigma,
            kg_shell_nxy = ctx.kg_shell_nxy,
            kg_shell_pcomp_nxy = ctx.kg_shell_pcomp_nxy,
            kg_shell_pcomp_nxy_compression_only = ctx.kg_shell_pcomp_nxy_compression_only,
        )
        Kg_basis = _flat_pcomp_quad4_kg_local(ctx, sigma_basis)
        f_local[j] = dot(phi_local, Kg_basis * phi_local)
    end

    return f_local
end

@inline function _quad4_shell_is_isotropic(prop, mat)
    is_pcomp_clt = get(prop, "TYPE", "") == "PCOMP_CLT" && haskey(prop, "Cm")
    pcomp_is_isotropic = is_pcomp_clt && get(prop, "IS_ISOTROPIC", false)
    is_ortho = !is_pcomp_clt && get(mat, "TYPE", "") == "MAT8" && haskey(mat, "E1") && haskey(mat, "E2")
    is_mat2 = !is_pcomp_clt && !is_ortho && get(mat, "TYPE", "") == "MAT2" && haskey(mat, "G11")
    is_iso = pcomp_is_isotropic || (!is_pcomp_clt && !is_ortho && !is_mat2)
    return is_iso, is_pcomp_clt, pcomp_is_isotropic
end

@inline function _flat_iso_quad4_Cm(E::Float64, nu::Float64, h::Float64)
    const_mem = E / (1.0 - nu^2)
    return (const_mem .* [1.0 nu 0.0; nu 1.0 0.0; 0.0 0.0 (1.0 - nu) / 2.0]) * h
end

function _flat_iso_quad4_sigma_input(ctx,
                                     u_local;
                                     nu_val::Float64=ctx.nu,
                                     force_use_gp_sigma::Union{Nothing,Bool}=nothing)
    Cm_kg = _flat_iso_quad4_Cm(ctx.E, nu_val, ctx.h)
    N_gp, N_res, _ = FEM.quad4_membrane_force_field(
        ctx.qd.lc, u_local, ctx.E, nu_val, ctx.h;
        Cm_override = Cm_kg,
        compatible_only = ctx.compatible_only,
        use_incompatible_modes = ctx.use_incompatible_modes,
        use_enhanced_modes = ctx.use_enhanced_modes,
        curvature_membrane = nothing,
        membrane_shear_center_row = ctx.kg_membrane_shear_center_row,
        material_shear_rotation = 0.0,
        membrane_assumed_mode = ctx.kg_membrane_assumed_mode,
        membrane_incomp_center_jacobian = ctx.membrane_incomp_center_jacobian,
    )

    use_gp_sigma = isnothing(force_use_gp_sigma) ? kg_quad4_use_gp_field(N_gp, N_res) : force_use_gp_sigma
    sigma_input = use_gp_sigma ? N_gp ./ ctx.h : N_res ./ ctx.h

    if ctx.kg_shell_nxy != 1.0
        if sigma_input isa AbstractMatrix
            @inbounds for gp in 1:size(sigma_input, 1)
                sigma_input[gp, 3] *= ctx.kg_shell_nxy
            end
        else
            sigma_input[3] *= ctx.kg_shell_nxy
        end
    end

    return sigma_input, use_gp_sigma
end

function _flat_iso_quad4_kg_context(el, model, id_map, node_coords, node_R, u_global)
    qd = _quad4_buckling_local_data(el, model, id_map, node_coords, node_R)
    isnothing(qd) && return nothing

    prop = qd.prop
    mat = qd.mat
    is_iso, is_pcomp_clt, _ = _quad4_shell_is_isotropic(prop, mat)
    if !is_iso || is_pcomp_clt || !_quad4_is_flat_for_buckling_rhs(qd.p1, qd.p2, qd.p3, qd.p4)
        return nothing
    end

    E = Float64(get(mat, "E", 0.0))
    nu = Float64(get(mat, "NU", 0.0))
    h = Float64(prop["T"])
    compatible_only = kg_use_compatible_membrane_stress()
    membrane_incomp = solver_env_bool("JFEM_SOL105_EIG_MEMBRANE_INCOMP", false)
    flat_iso_membrane_incomp = q4_flat_iso_eig_membrane_incomp_enabled()
    model_has_line_elements =
        !isempty(get(model, "CBARs", Dict())) ||
        !isempty(get(model, "CBEAMs", Dict())) ||
        !isempty(get(model, "CRODs", Dict())) ||
        !isempty(get(model, "CONRODs", Dict()))
    use_enhanced_modes = q4_sol105_flat_iso_dkmq_enabled() &&
                         !model_has_line_elements &&
                         get(prop, "Bmb", nothing) === nothing
    use_incompatible_modes = (membrane_incomp || flat_iso_membrane_incomp) && !use_enhanced_modes
    kg_membrane_shear_center_row = q4_flat_iso_eig_membrane_shear_center_row_enabled()
    kg_membrane_assumed_mode = q4_flat_iso_eig_membrane_assumed_mode()
    membrane_incomp_center_jacobian = q4_sol105_membrane_incomp_center_jacobian_enabled()
    kg_shell_nxy = kg_shell_nxy_scale()
    trans_mode = kg_shell_trans_mode()
    trans_mode === :curvature && (trans_mode = :all)

    u_elem = zeros(qd.ndof_elem)
    for i in 1:qd.ndof_elem
        u_elem[i] = u_global[qd.dofs[i]]
    end
    u_local = qd.T_mat * u_elem
    sigma_input, use_gp_sigma = _flat_iso_quad4_sigma_input(
        (
            qd = qd,
            E = E,
            nu = nu,
            h = h,
            compatible_only = compatible_only,
            use_incompatible_modes = use_incompatible_modes,
            use_enhanced_modes = use_enhanced_modes,
            kg_membrane_shear_center_row = kg_membrane_shear_center_row,
            kg_membrane_assumed_mode = kg_membrane_assumed_mode,
            membrane_incomp_center_jacobian = membrane_incomp_center_jacobian,
            kg_shell_nxy = kg_shell_nxy,
        ),
        u_local;
        nu_val = nu,
    )

    return (
        qd = qd,
        mid_str = qd.mid_str,
        E = E,
        nu = nu,
        h = h,
        compatible_only = compatible_only,
        use_incompatible_modes = use_incompatible_modes,
        use_enhanced_modes = use_enhanced_modes,
        kg_membrane_shear_center_row = kg_membrane_shear_center_row,
        kg_membrane_assumed_mode = kg_membrane_assumed_mode,
        membrane_incomp_center_jacobian = membrane_incomp_center_jacobian,
        kg_shell_nxy = kg_shell_nxy,
        trans_mode = trans_mode,
        curvature_sign = kg_shell_curvature_sign(),
        u_local = u_local,
        sigma_input = sigma_input,
        use_gp_sigma = use_gp_sigma,
    )
end

function _flat_iso_quad4_kg_local(ctx, sigma_input; nu_val::Float64=ctx.nu)
    return FEM.geometric_stiffness_quad4(
        ctx.qd.lc, sigma_input, ctx.h;
        trans_mode = ctx.trans_mode,
        curvature = nothing,
        curvature_sign = ctx.curvature_sign,
        rot_grad_scale = 0.0,
        membrane_shear_center_row = ctx.kg_membrane_shear_center_row,
        Cm = _flat_iso_quad4_Cm(ctx.E, nu_val, ctx.h),
        membrane_incomp = ctx.use_incompatible_modes,
        membrane_enhanced = ctx.use_enhanced_modes,
        material_shear_rotation = 0.0,
        membrane_assumed_mode = ctx.kg_membrane_assumed_mode,
        membrane_incomp_center_jacobian = ctx.membrane_incomp_center_jacobian,
    )
end

function _flat_iso_quad4_rhs_local(ctx, phi_local)
    f_local = zeros(ctx.qd.ndof_elem)
    basis = zeros(ctx.qd.ndof_elem)

    for j in 1:ctx.qd.ndof_elem
        fill!(basis, 0.0)
        basis[j] = 1.0
        sigma_basis, _ = _flat_iso_quad4_sigma_input(
            ctx,
            basis;
            nu_val = ctx.nu,
            force_use_gp_sigma = ctx.use_gp_sigma,
        )
        Kg_basis = _flat_iso_quad4_kg_local(ctx, sigma_basis; nu_val = ctx.nu)
        f_local[j] = dot(phi_local, Kg_basis * phi_local)
    end

    return f_local
end

function _flat_iso_quad4_phi_Kg_phi_at_nu(ctx, phi_local, nu_val::Float64)
    sigma_input, _ = _flat_iso_quad4_sigma_input(
        ctx,
        ctx.u_local;
        nu_val = nu_val,
        force_use_gp_sigma = ctx.use_gp_sigma,
    )
    Kg_local = _flat_iso_quad4_kg_local(ctx, sigma_input; nu_val = nu_val)
    return dot(phi_local, Kg_local * phi_local)
end

function _flat_iso_quad4_phi_dKg_dnu(ctx, phi_local)
    delta = max(abs(ctx.nu) * 1e-6, 1e-8)
    val_plus = _flat_iso_quad4_phi_Kg_phi_at_nu(ctx, phi_local, ctx.nu + delta)
    val_minus = _flat_iso_quad4_phi_Kg_phi_at_nu(ctx, phi_local, ctx.nu - delta)
    return (val_plus - val_minus) / (2.0 * delta)
end

function _flat_iso_quad4_phi_Kg_phi_at_h(ctx, phi_local, h_val::Float64)
    ctx_h = (; ctx..., h = h_val)
    sigma_input, _ = _flat_iso_quad4_sigma_input(
        ctx_h,
        ctx_h.u_local;
        nu_val = ctx_h.nu,
        force_use_gp_sigma = ctx.use_gp_sigma,
    )
    Kg_local = _flat_iso_quad4_kg_local(ctx_h, sigma_input; nu_val = ctx_h.nu)
    return dot(phi_local, Kg_local * phi_local)
end

function _flat_iso_quad4_phi_dKg_dh(ctx, phi_local)
    delta = max(abs(ctx.h) * 1e-6, 1e-8)
    val_plus = _flat_iso_quad4_phi_Kg_phi_at_h(ctx, phi_local, ctx.h + delta)
    val_minus = _flat_iso_quad4_phi_Kg_phi_at_h(ctx, phi_local, ctx.h - delta)
    return (val_plus - val_minus) / (2.0 * delta)
end

@inline function _covariant_iso_quad4_branch_supported()
    membrane_mode = kg_quad4_membrane_recovery_mode()
    return (
        membrane_mode === :covariant ||
        membrane_mode === :auto ||
        kg_shell_surface_operator_mode() === :covariant ||
        q4_sol105_flat_iso_dkmq_enabled()
    )
end

function _quad4_coords3d(qd)
    coords3d = zeros(4, 3)
    for (k, p) in enumerate((qd.p1, qd.p2, qd.p3, qd.p4))
        coords3d[k, 1] = p[1]
        coords3d[k, 2] = p[2]
        coords3d[k, 3] = p[3]
    end
    return coords3d
end

function _quad4_local_to_global_translation_nodes(qd, node_R, u_local)
    u_elem = qd.T_mat' * u_local
    u_nodes_global = zeros(4, 3)
    for (k, idx) in enumerate(qd.idxs)
        base = (k - 1) * 6
        u1 = u_elem[base + 1]
        u2 = u_elem[base + 2]
        u3 = u_elem[base + 3]
        u_nodes_global[k, 1] = node_R[idx][1,1] * u1 + node_R[idx][1,2] * u2 + node_R[idx][1,3] * u3
        u_nodes_global[k, 2] = node_R[idx][2,1] * u1 + node_R[idx][2,2] * u2 + node_R[idx][2,3] * u3
        u_nodes_global[k, 3] = node_R[idx][3,1] * u1 + node_R[idx][3,2] * u2 + node_R[idx][3,3] * u3
    end
    return u_nodes_global
end

function _covariant_iso_quad4_support(model, id_map, node_coords)
    _covariant_iso_quad4_branch_supported() || return nothing

    n_nodes = size(node_coords, 1)
    snorm_normals = compute_snorm_normals(model, id_map, node_coords)
    geom_normals = compute_geometric_nodal_normals(model, id_map, node_coords)
    geom_vec = fill(SVector(0.0, 0.0, 0.0), n_nodes)
    geom_has = falses(n_nodes)
    for (idx, nrm) in geom_normals
        geom_vec[idx] = nrm
        geom_has[idx] = true
    end

    shell_valence = zeros(Int, n_nodes)
    for (_, shell) in get(model, "CSHELLs", Dict())
        for nid in get(shell, "NODES", Int[])
            idx = get(id_map, nid, 0)
            idx > 0 && (shell_valence[idx] += 1)
        end
    end

    return (;
        snorm_normals = snorm_normals,
        geom_normals = geom_normals,
        geom_vec = geom_vec,
        geom_has = geom_has,
        shell_valence = shell_valence,
    )
end

function _quad4_buckling_local_data_with_frame(
    qd,
    node_R,
    v1::SVector{3,Float64},
    v2::SVector{3,Float64},
    v3::SVector{3,Float64};
    nodal_geomnormal_transform::Bool=false,
    geom_normals::Dict{Int,SVector{3,Float64}}=Dict{Int,SVector{3,Float64}}(),
)
    c_center = (qd.p1 + qd.p2 + qd.p3 + qd.p4) / 4.0
    lc = zeros(4, 2)
    for (k, p) in enumerate((qd.p1, qd.p2, qd.p3, qd.p4))
        lc[k, 1] = dot(p - c_center, v1)
        lc[k, 2] = dot(p - c_center, v2)
    end

    T_mat = zeros(qd.ndof_elem, qd.ndof_elem)
    for k in 1:4
        idx = qd.idxs[k]
        vk1, vk2, vk3 =
            nodal_geomnormal_transform && haskey(geom_normals, idx) ?
            shell_project_frame_to_normal(v1, v2, v3, geom_normals[idx]) :
            (v1, v2, v3)
        Rel = [vk1[1] vk1[2] vk1[3]; vk2[1] vk2[2] vk2[3]; vk3[1] vk3[2] vk3[3]]
        TR = Rel * node_R[idx]
        base = (k - 1) * 6
        T_mat[base+1:base+3, base+1:base+3] = TR
        T_mat[base+4:base+6, base+4:base+6] = TR
    end

    return merge(qd, (; lc = lc, T_mat = T_mat, v1 = v1, v2 = v2, v3 = v3))
end

function _covariant_iso_quad4_sigma_input(ctx,
                                          u_local;
                                          nu_val::Float64=ctx.nu,
                                          force_use_gp_sigma::Union{Nothing,Bool}=nothing,
                                          force_sigma_field_mode::Union{Nothing,Symbol}=nothing,
                                          force_gp_blend_alpha::Union{Nothing,Float64}=nothing)
    Cm_kg = _flat_iso_quad4_Cm(ctx.E, nu_val, ctx.h)
    N_gp, N_res, _ = FEM.quad4_membrane_force_field(
        ctx.qd.lc, u_local, ctx.E, nu_val, ctx.h;
        Cm_override = Cm_kg,
        compatible_only = ctx.compatible_only,
        use_incompatible_modes = ctx.use_incompatible_modes,
        use_enhanced_modes = ctx.use_enhanced_modes,
        curvature_membrane = ctx.curvature_membrane,
        membrane_shear_center_row = ctx.recovery_membrane_shear_center_row,
        material_shear_rotation = 0.0,
        membrane_assumed_mode = ctx.kg_membrane_assumed_mode,
        membrane_incomp_center_jacobian = ctx.membrane_incomp_center_jacobian,
    )

    if ctx.use_covariant_membrane
        u_nodes_global = _quad4_local_to_global_translation_nodes(ctx.qd, ctx.node_R, u_local)
        N_gp_cov, N_res_cov, _ = FEM.quad4_membrane_force_field_covariant(
            ctx.coords3d, u_nodes_global, ctx.qd.v1, ctx.qd.v2, Cm_kg
        )
        if ctx.covariant_blend >= 1.0
            N_gp = N_gp_cov
            N_res = N_res_cov
        else
            @inbounds for gp in 1:4, comp in 1:3
                N_gp[gp, comp] =
                    (1.0 - ctx.covariant_blend) * N_gp[gp, comp] +
                    ctx.covariant_blend * N_gp_cov[gp, comp]
            end
            @inbounds for comp in 1:3
                N_res[comp] =
                    (1.0 - ctx.covariant_blend) * N_res[comp] +
                    ctx.covariant_blend * N_res_cov[comp]
            end
        end
    end

    gp_blend_alpha = 0.0
    use_gp_sigma = false
    if isnothing(force_sigma_field_mode)
        use_gp_sigma = isnothing(force_use_gp_sigma) ? kg_quad4_use_gp_field(N_gp, N_res) : force_use_gp_sigma
        if !use_gp_sigma && ctx.auto_gp_patch_candidate
            use_gp_sigma = true
        end
        if !use_gp_sigma && ctx.auto_gp_spread && !isnothing(ctx.geom_curvature)
            gp_mean_norm = 0.0
            @inbounds for gp in 1:size(N_gp, 1)
                gp_mean_norm += sqrt(N_gp[gp,1]^2 + N_gp[gp,2]^2 + N_gp[gp,3]^2)
            end
            gp_mean_norm /= max(size(N_gp, 1), 1)
            if gp_mean_norm > 1e-12
                gp_spread = 0.0
                @inbounds for gp in 1:size(N_gp, 1)
                    dn1 = N_gp[gp,1] - N_res[1]
                    dn2 = N_gp[gp,2] - N_res[2]
                    dn3 = N_gp[gp,3] - N_res[3]
                    gp_spread = max(gp_spread, sqrt(dn1^2 + dn2^2 + dn3^2) / gp_mean_norm)
                end
                k1_gp, _ = q4_curvature_principal_abs(ctx.geom_curvature)
                kappa_l_gp = k1_gp * q4_curvature_characteristic_length(ctx.qd.lc)
                cyl_ratio_gp = q4_curvature_cyl_ratio(ctx.geom_curvature)
                if gp_spread >= kg_quad4_auto_gp_spread_min() &&
                   kappa_l_gp >= kg_quad4_auto_gp_spread_kappa_l_min() &&
                   cyl_ratio_gp >= kg_quad4_auto_gp_spread_cyl_ratio_min()
                    gp_blend_scale = kg_quad4_auto_gp_spread_blend_scale()
                    if gp_blend_scale > 0.0
                        avg_norm = sqrt(N_res[1]^2 + N_res[2]^2 + N_res[3]^2)
                        avg_ratio = avg_norm / gp_mean_norm
                        gp_blend_alpha = clamp(
                            gp_blend_scale * gp_spread * max(0.0, 1.0 - avg_ratio),
                            0.0,
                            1.0,
                        )
                    else
                        use_gp_sigma = true
                    end
                end
            end
        end
    else
        if force_sigma_field_mode === :gp
            use_gp_sigma = true
        elseif force_sigma_field_mode === :blend
            use_gp_sigma = false
            isnothing(force_gp_blend_alpha) &&
                error("force_gp_blend_alpha must be provided when freezing a blended sigma field")
            gp_blend_alpha = force_gp_blend_alpha
        else
            use_gp_sigma = false
        end
    end

    sigma_input = if use_gp_sigma
        N_gp ./ ctx.h
    elseif gp_blend_alpha > 0.0
        kg_quad4_blend_gp_field!(zeros(size(N_gp)), N_gp, N_res, gp_blend_alpha) ./ ctx.h
    else
        N_res ./ ctx.h
    end

    if ctx.kg_shell_nxy_auto > 0.0 && !isnothing(ctx.geom_curvature)
        k1_auto, _ = q4_curvature_principal_abs(ctx.geom_curvature)
        kappa_l_auto = k1_auto * q4_curvature_characteristic_length(ctx.qd.lc)
        cyl_ratio_auto = q4_curvature_cyl_ratio(ctx.geom_curvature)
        if kappa_l_auto > max(ctx.kg_shell_nxy_auto_kappa_l_min, 1e-12) &&
           cyl_ratio_auto <= ctx.kg_shell_nxy_auto_cyl_ratio_max
            if sigma_input isa AbstractMatrix
                @inbounds for gp in 1:size(sigma_input, 1)
                    sigma_input[gp, 3] *= kg_shell_nxy_auto_scale(
                        sigma_input[gp, 1],
                        sigma_input[gp, 2],
                        sigma_input[gp, 3],
                        ctx.kg_shell_nxy_auto,
                        ctx.kg_shell_nxy_auto_ratio_min,
                        ctx.kg_shell_nxy_auto_ratio_full,
                    )
                end
            else
                sigma_input[3] *= kg_shell_nxy_auto_scale(
                    sigma_input[1],
                    sigma_input[2],
                    sigma_input[3],
                    ctx.kg_shell_nxy_auto,
                    ctx.kg_shell_nxy_auto_ratio_min,
                    ctx.kg_shell_nxy_auto_ratio_full,
                )
            end
        end
    end

    if ctx.kg_shell_nxy != 1.0
        if sigma_input isa AbstractMatrix
            @inbounds for gp in 1:size(sigma_input, 1)
                sigma_input[gp, 3] *= ctx.kg_shell_nxy
            end
        else
            sigma_input[3] *= ctx.kg_shell_nxy
        end
    end

    sigma_field_mode = use_gp_sigma ? :gp : (gp_blend_alpha > 0.0 ? :blend : :res)
    return sigma_input, use_gp_sigma, gp_blend_alpha, sigma_field_mode
end

function _covariant_iso_quad4_kg_context(el, model, id_map, node_coords, node_R, u_global, support)
    _covariant_iso_quad4_branch_supported() || return nothing

    qd_base = _quad4_buckling_local_data(el, model, id_map, node_coords, node_R)
    isnothing(qd_base) && return nothing

    prop = qd_base.prop
    mat = qd_base.mat
    is_iso, is_pcomp_clt, _ = _quad4_shell_is_isotropic(prop, mat)
    if !is_iso || is_pcomp_clt
        return nothing
    end

    support = isnothing(support) ? _covariant_iso_quad4_support(model, id_map, node_coords) : support

    qd = qd_base
    if !isnothing(support) && !isempty(support.snorm_normals)
        v1, v2, v3 = apply_snorm_to_frame(
            qd.v1, qd.v2, qd.v3, qd.idxs, support.snorm_normals
        )
        qd = _quad4_buckling_local_data_with_frame(qd, node_R, v1, v2, v3)
    end

    compatible_only = kg_use_compatible_membrane_stress()
    covariant_blend = kg_quad4_covariant_blend()

    elem_is_flat = _quad4_is_flat_for_buckling_rhs(qd.p1, qd.p2, qd.p3, qd.p4)
    d13_geom = qd.p3 - qd.p1
    d24_geom = qd.p4 - qd.p2
    v3_geom_raw = cross(d13_geom, d24_geom)
    v3_geom_len = norm(v3_geom_raw)
    if v3_geom_len > 1e-12
        c_geom = (qd.p1 + qd.p2 + qd.p3 + qd.p4) / 4.0
        v3g = v3_geom_raw / v3_geom_len
        max_dev = max(
            abs(dot(qd.p1 - c_geom, v3g)),
            abs(dot(qd.p2 - c_geom, v3g)),
            abs(dot(qd.p3 - c_geom, v3g)),
            abs(dot(qd.p4 - c_geom, v3g)),
        )
        L_diag = max(norm(d13_geom), norm(d24_geom))
    else
        max_dev = 0.0
        L_diag = max(norm(d13_geom), norm(d24_geom))
    end
    warp_ratio = max_dev / max(L_diag, 1e-12)

    aspect_ratio = q4_local_edge_aspect_ratio(qd.lc)
    valence_sum = isnothing(support) ? 0 : sum(support.shell_valence[idx] for idx in qd.idxs)
    have_geom_normals =
        !isnothing(support) &&
        all(support.geom_has[idx] for idx in qd.idxs)
    use_geom_snorm_kg = false
    if !isnothing(support) &&
       q4_curved_iso_geomnormal_frame_enabled() &&
       aspect_ratio >= q4_curved_iso_geomnormal_frame_aspect_ratio_min() &&
       valence_sum <= 10 &&
       have_geom_normals
        geom_probe = estimate_quad4_curvature_membrane(
            qd.lc,
            support.geom_vec[qd.idxs[1]],
            support.geom_vec[qd.idxs[2]],
            support.geom_vec[qd.idxs[3]],
            support.geom_vec[qd.idxs[4]],
            qd.v1,
            qd.v2,
            qd.v3,
        )
        k1_probe, _ = q4_curvature_principal_abs(geom_probe)
        kappa_l_probe = k1_probe * q4_curvature_characteristic_length(qd.lc)
        cyl_ratio_probe = q4_curvature_cyl_ratio(geom_probe)
        if elem_is_flat
            use_geom_snorm_kg =
                kappa_l_probe >= q4_flat_curved_iso_geomnormal_frame_kappa_l_min() &&
                kappa_l_probe <= q4_flat_curved_iso_geomnormal_frame_kappa_l_max() &&
                cyl_ratio_probe <= q4_flat_curved_iso_geomnormal_frame_cyl_ratio_max()
        else
            use_geom_snorm_kg =
                kappa_l_probe >= q4_curved_iso_geomnormal_frame_kappa_l_min() &&
                kappa_l_probe <= q4_curved_iso_geomnormal_frame_kappa_l_max() &&
                cyl_ratio_probe <= q4_curved_iso_geomnormal_frame_cyl_ratio_max()
        end
    end
    if use_geom_snorm_kg
        n_avg_g = SVector(0.0, 0.0, 0.0)
        for idx in qd.idxs
            n_avg_g += support.geom_vec[idx]
        end
        if norm(n_avg_g) > 1e-12
            v3n = normalize(n_avg_g)
            if dot(v3n, qd.v3) < 0.0
                v3n = -v3n
            end
            v1p = qd.v1 - dot(qd.v1, v3n) * v3n
            if norm(v1p) > 1e-12
                v1n = normalize(v1p)
            else
                v2p = qd.v2 - dot(qd.v2, v3n) * v3n
                v1n = normalize(v2p)
            end
            qd = _quad4_buckling_local_data_with_frame(
                qd, node_R, v1n, cross(v3n, v1n), v3n
            )
            aspect_ratio = q4_local_edge_aspect_ratio(qd.lc)
        end
    end

    elem_flat_curved_iso_nodal_geomnormal_transform =
        !isnothing(support) &&
        q4_flat_curved_iso_nodal_geomnormal_transform_enabled() &&
        elem_is_flat &&
        use_geom_snorm_kg &&
        aspect_ratio >= q4_flat_curved_iso_nodal_geomnormal_transform_aspect_ratio_min() &&
        valence_sum <= q4_flat_curved_iso_nodal_geomnormal_transform_valence_sum_max() &&
        have_geom_normals
    if elem_flat_curved_iso_nodal_geomnormal_transform
        qd = _quad4_buckling_local_data_with_frame(
            qd,
            node_R,
            qd.v1,
            qd.v2,
            qd.v3;
            nodal_geomnormal_transform = true,
            geom_normals = support.geom_normals,
        )
    end

    model_has_line_elements =
        !isempty(get(model, "CBARs", Dict())) ||
        !isempty(get(model, "CBEAMs", Dict())) ||
        !isempty(get(model, "CRODs", Dict())) ||
        !isempty(get(model, "CONRODs", Dict()))
    E = Float64(get(mat, "E", 0.0))
    nu = Float64(get(mat, "NU", 0.0))
    h = Float64(prop["T"])
    membrane_incomp = solver_env_bool("JFEM_SOL105_EIG_MEMBRANE_INCOMP", false)
    flat_iso_membrane_incomp = q4_flat_iso_eig_membrane_incomp_enabled()
    use_enhanced_modes =
        q4_sol105_flat_iso_dkmq_enabled() &&
        elem_is_flat &&
        !model_has_line_elements &&
        get(prop, "Bmb", nothing) === nothing
    kg_membrane_shear_center_row =
        elem_is_flat && q4_flat_iso_eig_membrane_shear_center_row_enabled()
    kg_membrane_assumed_mode = elem_is_flat ? q4_flat_iso_eig_membrane_assumed_mode() : :none
    membrane_incomp_center_jacobian = q4_sol105_membrane_incomp_center_jacobian_enabled()
    kg_shell_nxy = kg_shell_nxy_scale()
    kg_shell_nxy_auto_v = kg_shell_nxy_auto_relax()
    kg_shell_nxy_auto_ratio_min_v = kg_shell_nxy_auto_ratio_min()
    kg_shell_nxy_auto_ratio_full_v = kg_shell_nxy_auto_ratio_full()
    kg_shell_nxy_auto_cyl_ratio_max_v = kg_shell_nxy_auto_cyl_ratio_max()
    kg_shell_nxy_auto_kappa_l_min_v = kg_shell_nxy_auto_kappa_l_min()
    kg_trans_mode_eff = kg_shell_trans_mode()
    kg_auto_curvature_iso = kg_trans_mode_eff !== :curvature && q4_shell_kg_auto_curvature_iso_enabled()
    kg_rot_grad_scale_eff = elem_is_flat ? 0.0 : kg_shell_rot_grad_scale()

    u_elem = zeros(qd.ndof_elem)
    for i in 1:qd.ndof_elem
        u_elem[i] = u_global[qd.dofs[i]]
    end
    u_local = qd.T_mat * u_elem

    geom_curvature = nothing
    if have_geom_normals
        geom_curvature = estimate_quad4_curvature_membrane(
            qd.lc,
            support.geom_vec[qd.idxs[1]],
            support.geom_vec[qd.idxs[2]],
            support.geom_vec[qd.idxs[3]],
            support.geom_vec[qd.idxs[4]],
            qd.v1,
            qd.v2,
            qd.v3,
        )
    end

    flat_curved_iso_center_candidate = false
    if elem_is_flat && !isnothing(geom_curvature)
        k1_flat_curved, _ = q4_curvature_principal_abs(geom_curvature)
        kappa_l_flat_curved = k1_flat_curved * q4_curvature_characteristic_length(qd.lc)
        cyl_ratio_flat_curved = q4_curvature_cyl_ratio(geom_curvature)
        flat_curved_iso_center_candidate =
            q4_flat_curved_iso_eig_center_only_enabled() &&
            kappa_l_flat_curved >= q4_flat_curved_iso_eig_center_only_kappa_l_min() &&
            cyl_ratio_flat_curved <= q4_flat_curved_iso_eig_center_only_cyl_ratio_max()
    end

    iso_corner_curvature = nothing
    if q4_sol105_flat_iso_dkmq_enabled() &&
       !elem_is_flat &&
       get(prop, "Bmb", nothing) === nothing
        iso_corner_curvature = estimate_quad4_corner_curvature_membrane(
            qd.lc,
            qd.p1,
            qd.p2,
            qd.p3,
            qd.p4,
            qd.v1,
            qd.v2,
            qd.v3,
        )
    end

    iso_auto_curvature_resolution_ok =
        aspect_ratio <= q4_shell_kg_auto_curvature_iso_aspect_ratio_max()
    auto_curved_iso_membrane_incomp = false
    auto_warped_iso_membrane_incomp = false
    if !elem_is_flat &&
       q4_curved_iso_eig_auto_membrane_incomp_enabled() &&
       iso_auto_curvature_resolution_ok &&
       !isnothing(geom_curvature)
        k1_iso, _ = q4_curvature_principal_abs(geom_curvature)
        kappa_l_iso = k1_iso * q4_curvature_characteristic_length(qd.lc)
        cyl_ratio_iso = q4_curvature_cyl_ratio(geom_curvature)
        auto_curved_iso_membrane_incomp =
            kappa_l_iso >= q4_curved_iso_eig_auto_membrane_incomp_kappa_l_min() &&
            cyl_ratio_iso <= q4_curved_iso_eig_auto_membrane_incomp_cyl_ratio_max()
        auto_warped_iso_membrane_incomp =
            q4_curved_iso_warp_membrane_incomp_enabled() &&
            warp_ratio >= q4_curved_iso_warp_membrane_incomp_ratio_min() &&
            kappa_l_iso <= q4_curved_iso_warp_membrane_incomp_kappa_l_max()
    end
    auto_elongated_iso_membrane_incomp =
        q4_curved_iso_elongated_membrane_incomp_enabled() &&
        !elem_is_flat &&
        aspect_ratio >= q4_curved_iso_elongated_membrane_incomp_aspect_ratio_min()
    # This helper is limited to isotropic QUAD4 shells, so the PCOMP auto-
    # membrane branch is intentionally inactive here.
    auto_pcomp_membrane_incomp = false
    use_incompatible_modes =
        (
            membrane_incomp ||
            auto_curved_iso_membrane_incomp ||
            auto_warped_iso_membrane_incomp ||
            auto_elongated_iso_membrane_incomp ||
            auto_pcomp_membrane_incomp ||
            (flat_iso_membrane_incomp && elem_is_flat)
        ) &&
        !use_enhanced_modes

    curvature_membrane = nothing
    if !isnothing(support)
        curvature_scale = q4_curvature_membrane_scale("JFEM_Q4_CURVATURE_MEMBRANE_SCALE_KG")
        has_curv_normals =
            use_geom_snorm_kg ||
            all(haskey(support.snorm_normals, idx) for idx in qd.idxs)
        if curvature_scale > 0.0 && has_curv_normals
            n1_curv = use_geom_snorm_kg ? support.geom_vec[qd.idxs[1]] : support.snorm_normals[qd.idxs[1]]
            n2_curv = use_geom_snorm_kg ? support.geom_vec[qd.idxs[2]] : support.snorm_normals[qd.idxs[2]]
            n3_curv = use_geom_snorm_kg ? support.geom_vec[qd.idxs[3]] : support.snorm_normals[qd.idxs[3]]
            n4_curv = use_geom_snorm_kg ? support.geom_vec[qd.idxs[4]] : support.snorm_normals[qd.idxs[4]]
            curvature_raw = estimate_quad4_curvature_membrane(
                qd.lc, n1_curv, n2_curv, n3_curv, n4_curv, qd.v1, qd.v2, qd.v3
            )
            curvature_weight = q4_curvature_filter_weight(
                curvature_raw,
                q4_curvature_filter_mode("JFEM_Q4_CURVATURE_FILTER_MODE_KG"),
                q4_curvature_cyl_ratio_max("JFEM_Q4_CURVATURE_CYL_RATIO_MAX_KG"),
            )
            curvature_weight *= q4_curvature_resolution_weight(
                curvature_raw,
                qd.lc,
                q4_curvature_resolution_min("JFEM_Q4_CURVATURE_RESOLUTION_MIN_KG"),
                q4_curvature_resolution_full("JFEM_Q4_CURVATURE_RESOLUTION_FULL_KG"),
            )
            if curvature_weight > 0.0
                curvature_membrane = curvature_raw * (curvature_scale * curvature_weight)
            end
        end
    end

    covariant_membrane_candidate = false
    kg_curvature = nothing
    curvature_sign = kg_shell_curvature_sign()
    if kg_auto_curvature_iso && !isnothing(geom_curvature) && iso_auto_curvature_resolution_ok
        k1, _ = q4_curvature_principal_abs(geom_curvature)
        kappa_l = k1 * q4_curvature_characteristic_length(qd.lc)
        cyl_ratio = q4_curvature_cyl_ratio(geom_curvature)
        covariant_membrane_candidate =
            kappa_l >= kg_quad4_covariant_auto_kappa_l_min() &&
            cyl_ratio <= kg_quad4_covariant_auto_cyl_ratio_max() &&
            aspect_ratio <= q4_shell_kg_auto_curvature_iso_aspect_ratio_max()
        if kg_auto_curvature_iso_cyl_candidate(kappa_l, cyl_ratio, aspect_ratio)
            kg_trans_mode_eff = :curvature
            kg_curvature = geom_curvature * q4_shell_kg_auto_curvature_iso_cyl_scale()
            curvature_sign = q4_shell_kg_auto_curvature_iso_cyl_sign()
        elseif kg_auto_curvature_iso_candidate(kappa_l, cyl_ratio, aspect_ratio)
            kg_trans_mode_eff = :curvature
            kg_curvature = geom_curvature * q4_shell_kg_auto_curvature_iso_effective_scale(cyl_ratio, kappa_l)
            curvature_sign = q4_shell_kg_auto_curvature_iso_sign()
        end
        if kg_shell_rot_grad_auto_iso_scale() > 0.0 &&
           kappa_l >= kg_shell_rot_grad_auto_kappa_l_min() &&
           cyl_ratio >= kg_shell_rot_grad_auto_cyl_ratio_min() &&
           aspect_ratio <= q4_shell_kg_auto_curvature_iso_aspect_ratio_max()
            kg_rot_grad_scale_eff = max(kg_rot_grad_scale_eff, kg_shell_rot_grad_auto_iso_scale())
        end
    end

    auto_gp_patch_candidate = false
    if !isnothing(geom_curvature) && kg_quad4_auto_gp_patch_enabled()
        k1_patch, _ = q4_curvature_principal_abs(geom_curvature)
        kappa_l_patch = k1_patch * q4_curvature_characteristic_length(qd.lc)
        max_valence = maximum(support.shell_valence[idx] for idx in qd.idxs)
        auto_gp_patch_candidate =
            max_valence >= kg_quad4_auto_gp_patch_valence_min() &&
            kappa_l_patch >= kg_quad4_auto_gp_patch_kappa_l_min()
    end

    membrane_recovery_mode = kg_quad4_membrane_recovery_mode()
    use_covariant_membrane =
        compatible_only &&
        covariant_blend > 0.0 &&
        (
            membrane_recovery_mode === :covariant ||
            (membrane_recovery_mode === :auto && covariant_membrane_candidate)
        )
    use_covariant_operator = kg_shell_surface_operator_mode() === :covariant
    use_flat_curved_iso_exact_membrane =
        q4_sol105_flat_iso_dkmq_enabled() &&
        elem_is_flat &&
        get(prop, "Bmb", nothing) === nothing &&
        flat_curved_iso_center_candidate
    use_cyl_iso_exact_membrane =
        q4_sol105_flat_iso_dkmq_enabled() &&
        !elem_is_flat &&
        get(prop, "Bmb", nothing) === nothing &&
        iso_corner_curvature !== nothing &&
        abs(q4_curvature_gaussian(iso_corner_curvature)) <= 1e-10 &&
        first(q4_curvature_principal_abs(iso_corner_curvature)) > 1e-8
    use_iso_exact_membrane =
        use_flat_curved_iso_exact_membrane || use_cyl_iso_exact_membrane
    if curvature_membrane === nothing && use_cyl_iso_exact_membrane && iso_corner_curvature !== nothing
        curvature_membrane = iso_corner_curvature
    end
    use_enhanced_modes = use_enhanced_modes || use_iso_exact_membrane
    use_incompatible_modes = use_incompatible_modes && !use_iso_exact_membrane
    if !use_covariant_membrane && !use_covariant_operator && !use_iso_exact_membrane
        return nothing
    end

    trans_mode_flat = kg_trans_mode_eff === :curvature ? :all : kg_trans_mode_eff
    trans_mode_cov = kg_trans_mode_eff === :normal_only ? :normal_only : :all
    recovery_membrane_shear_center_row =
        use_iso_exact_membrane ? false : kg_membrane_shear_center_row

    pre_ctx = (
        qd = qd,
        mid_str = qd.mid_str,
        node_R = node_R,
        coords3d = _quad4_coords3d(qd),
        E = E,
        nu = nu,
        h = h,
        compatible_only = compatible_only,
        use_incompatible_modes = use_incompatible_modes,
        use_enhanced_modes = use_enhanced_modes,
        kg_membrane_shear_center_row = kg_membrane_shear_center_row,
        recovery_membrane_shear_center_row = recovery_membrane_shear_center_row,
        kg_membrane_assumed_mode = kg_membrane_assumed_mode,
        membrane_incomp_center_jacobian = membrane_incomp_center_jacobian,
        kg_shell_nxy = kg_shell_nxy,
        use_covariant_membrane = use_covariant_membrane,
        use_covariant_operator = use_covariant_operator,
        use_flat_curved_iso_exact_membrane = use_flat_curved_iso_exact_membrane,
        use_cyl_iso_exact_membrane = use_cyl_iso_exact_membrane,
        use_iso_exact_membrane = use_iso_exact_membrane,
        covariant_blend = covariant_blend,
        curvature_membrane = curvature_membrane,
        auto_gp_patch_candidate = auto_gp_patch_candidate,
        auto_gp_spread = kg_quad4_auto_gp_spread_enabled(),
        geom_curvature = geom_curvature,
        covariant_membrane_candidate = covariant_membrane_candidate,
        kg_shell_nxy_auto = kg_shell_nxy_auto_v,
        kg_shell_nxy_auto_ratio_min = kg_shell_nxy_auto_ratio_min_v,
        kg_shell_nxy_auto_ratio_full = kg_shell_nxy_auto_ratio_full_v,
        kg_shell_nxy_auto_cyl_ratio_max = kg_shell_nxy_auto_cyl_ratio_max_v,
        kg_shell_nxy_auto_kappa_l_min = kg_shell_nxy_auto_kappa_l_min_v,
        trans_mode_flat = trans_mode_flat,
        trans_mode_cov = trans_mode_cov,
        kg_curvature = kg_curvature,
        rot_grad_scale = kg_rot_grad_scale_eff,
        curvature_sign = curvature_sign,
    )
    sigma_input, use_gp_sigma, gp_blend_alpha, sigma_field_mode = _covariant_iso_quad4_sigma_input(
        pre_ctx,
        u_local;
        nu_val = nu,
    )

    return (;
        pre_ctx...,
        u_local = u_local,
        sigma_input = sigma_input,
        use_gp_sigma = use_gp_sigma,
        gp_blend_alpha = gp_blend_alpha,
        sigma_field_mode = sigma_field_mode,
    )
end

function _covariant_iso_quad4_kg_local(ctx, sigma_input; nu_val::Float64=ctx.nu)
    if ctx.use_covariant_operator
        return FEM.geometric_stiffness_quad4_covariant(
            ctx.coords3d,
            sigma_input,
            ctx.h,
            ctx.qd.v1,
            ctx.qd.v2;
            trans_mode = ctx.trans_mode_cov,
            rot_grad_scale = ctx.rot_grad_scale,
        )
    end

    return FEM.geometric_stiffness_quad4(
        ctx.qd.lc, sigma_input, ctx.h;
        trans_mode = ctx.trans_mode_flat,
        curvature = ctx.kg_curvature,
        curvature_sign = ctx.curvature_sign,
        rot_grad_scale = ctx.rot_grad_scale,
        membrane_shear_center_row = ctx.kg_membrane_shear_center_row,
        Cm = _flat_iso_quad4_Cm(ctx.E, nu_val, ctx.h),
        membrane_incomp = ctx.use_incompatible_modes,
        membrane_enhanced = ctx.use_enhanced_modes,
        material_shear_rotation = 0.0,
        membrane_assumed_mode = ctx.kg_membrane_assumed_mode,
        membrane_incomp_center_jacobian = ctx.membrane_incomp_center_jacobian,
    )
end

function _covariant_iso_quad4_rhs_local(ctx, phi_local)
    f_local = zeros(ctx.qd.ndof_elem)
    basis = zeros(ctx.qd.ndof_elem)

    for j in 1:ctx.qd.ndof_elem
        fill!(basis, 0.0)
        basis[j] = 1.0
        sigma_basis, _, _, _ = _covariant_iso_quad4_sigma_input(
            ctx,
            basis;
            nu_val = ctx.nu,
            force_use_gp_sigma = ctx.use_gp_sigma,
            force_sigma_field_mode = ctx.sigma_field_mode,
            force_gp_blend_alpha = ctx.gp_blend_alpha,
        )
        Kg_basis = _covariant_iso_quad4_kg_local(ctx, sigma_basis; nu_val = ctx.nu)
        f_local[j] = dot(phi_local, Kg_basis * phi_local)
    end

    return f_local
end

function _covariant_iso_quad4_phi_Kg_phi_at_nu(ctx, phi_local, nu_val::Float64)
    sigma_input, _, _, _ = _covariant_iso_quad4_sigma_input(
        ctx,
        ctx.u_local;
        nu_val = nu_val,
    )
    Kg_local = _covariant_iso_quad4_kg_local(ctx, sigma_input; nu_val = nu_val)
    return dot(phi_local, Kg_local * phi_local)
end

function _covariant_iso_quad4_phi_dKg_dnu(ctx, phi_local)
    delta = max(abs(ctx.nu) * 1e-6, 1e-8)
    val_plus = _covariant_iso_quad4_phi_Kg_phi_at_nu(ctx, phi_local, ctx.nu + delta)
    val_minus = _covariant_iso_quad4_phi_Kg_phi_at_nu(ctx, phi_local, ctx.nu - delta)
    return (val_plus - val_minus) / (2.0 * delta)
end

function _covariant_iso_quad4_phi_Kg_phi_at_h(ctx, phi_local, h_val::Float64)
    ctx_h = (; ctx..., h = h_val)
    sigma_input, _, _, _ = _covariant_iso_quad4_sigma_input(
        ctx_h,
        ctx_h.u_local;
        nu_val = ctx_h.nu,
        force_use_gp_sigma = ctx.use_gp_sigma,
        force_sigma_field_mode = ctx.sigma_field_mode,
        force_gp_blend_alpha = ctx.gp_blend_alpha,
    )
    Kg_local = _covariant_iso_quad4_kg_local(ctx_h, sigma_input; nu_val = ctx_h.nu)
    return dot(phi_local, Kg_local * phi_local)
end

function _covariant_iso_quad4_phi_dKg_dh(ctx, phi_local)
    delta = max(abs(ctx.h) * 1e-6, 1e-8)
    val_plus = _covariant_iso_quad4_phi_Kg_phi_at_h(ctx, phi_local, ctx.h + delta)
    val_minus = _covariant_iso_quad4_phi_Kg_phi_at_h(ctx, phi_local, ctx.h - delta)
    return (val_plus - val_minus) / (2.0 * delta)
end

function _buckling_rhs_shells!(f_adj, model, id_map, node_coords, node_R, u_global, phi)
    covariant_iso_support = _covariant_iso_quad4_support(model, id_map, node_coords)

    for (_, el) in model["CSHELLs"]
        flat_pcomp_ctx = _flat_pcomp_quad4_kg_context(el, model, id_map, node_coords, node_R, u_global)
        if !isnothing(flat_pcomp_ctx)
            qd = flat_pcomp_ctx.qd
            phi_elem = zeros(qd.ndof_elem)
            for i in 1:qd.ndof_elem
                phi_elem[i] = phi[qd.dofs[i]]
            end

            phi_local = qd.T_mat * phi_elem
            f_local = _flat_pcomp_quad4_rhs_local(flat_pcomp_ctx, phi_local)
            f_global = qd.T_mat' * f_local
            for i in 1:qd.ndof_elem
                f_adj[qd.dofs[i]] += f_global[i]
            end
            continue
        end

        covariant_iso_ctx = _covariant_iso_quad4_kg_context(
            el, model, id_map, node_coords, node_R, u_global, covariant_iso_support
        )
        if !isnothing(covariant_iso_ctx)
            qd = covariant_iso_ctx.qd
            phi_elem = zeros(qd.ndof_elem)
            for i in 1:qd.ndof_elem
                phi_elem[i] = phi[qd.dofs[i]]
            end

            phi_local = qd.T_mat * phi_elem
            f_local = _covariant_iso_quad4_rhs_local(covariant_iso_ctx, phi_local)
            f_global = qd.T_mat' * f_local
            for i in 1:qd.ndof_elem
                f_adj[qd.dofs[i]] += f_global[i]
            end
            continue
        end

        flat_iso_ctx = _flat_iso_quad4_kg_context(el, model, id_map, node_coords, node_R, u_global)
        if !isnothing(flat_iso_ctx)
            qd = flat_iso_ctx.qd
            phi_elem = zeros(qd.ndof_elem)
            for i in 1:qd.ndof_elem
                phi_elem[i] = phi[qd.dofs[i]]
            end

            phi_local = qd.T_mat * phi_elem
            f_local = _flat_iso_quad4_rhs_local(flat_iso_ctx, phi_local)
            f_global = qd.T_mat' * f_local
            for i in 1:qd.ndof_elem
                f_adj[qd.dofs[i]] += f_global[i]
            end
            continue
        end

        ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
        if isnothing(ed); continue; end

        h = ed.prop["T"]; E = ed.mat["E"]; nu = ed.mat["NU"]
        n = ed.n_nodes; ndof_e = ed.ndof_elem

        # Extract mode shape DOFs for this element (in analysis frame)
        phi_elem = zeros(ndof_e)
        for i in 1:ndof_e; phi_elem[i] = phi[ed.dofs[i]]; end

        # Transform to local frame
        phi_local = ed.T_mat * phi_elem

        # Compute unit-stress geometric stiffnesses and contract with mode shape
        # Kg is linear in σ: Kg(σ) = σ₁·Kg₁ + σ₂·Kg₂ + σ₃·Kg₃
        # g_i = φ_local^T · Kg_i · φ_local
        unit_sigmas = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        g = zeros(3)
        for (i, sig) in enumerate(unit_sigmas)
            if n == 4
                Kg_i = FEM.geometric_stiffness_quad4(ed.lc, sig, h)
            else
                Kg_i = FEM.geometric_stiffness_tria3(ed.lc, sig, h)
            end
            g[i] = dot(phi_local, Kg_i * phi_local)
        end

        # Stress-displacement relation at centroid: σ = D · Bm · u_local
        # (σ_mem = N/h = (Cm/h)·Bm·u = D·Bm·u since Cm = D·h)
        Bm, _, D = _shell_centroid_B_matrices(n, ed.lc, E, nu)

        # ∂(φᵀ·Kg·φ)/∂u_local = (D·Bm)ᵀ · g
        f_local = (D * Bm)' * g

        # Transform to global and scatter
        f_global = ed.T_mat' * f_local
        for i in 1:ndof_e
            f_adj[ed.dofs[i]] += f_global[i]
        end
    end
end

# ============================================================================
# Bar/beam elements: ∂(φᵀ·Kg·φ)/∂u via axial force derivative
# ============================================================================

function _buckling_rhs_bars!(f_adj, model, id_map, node_coords, node_R, u_global, phi)
    pbarls = get(model, "PBARLs", Dict())
    mats = model["MATs"]

    for (_, bar) in get(model, "CBARs", Dict())
        pid_str = string(bar["PID"])
        prop = get(pbarls, pid_str, nothing)
        if isnothing(prop); continue; end
        mid_str = string(prop["MID"])
        mat = get(mats, mid_str, nothing)
        if isnothing(mat); continue; end
        if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
        i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]

        rd = _rod_local_frame_and_transform(i1, i2, node_coords, node_R)
        if isnothing(rd); continue; end

        E = mat["E"]; A = prop["A"]; L = rd.L

        # Kg_unit = Kg(P=1)
        Kg_unit = FEM.geometric_stiffness_frame3d(L, 1.0)

        phi_elem = [phi[rd.dofs[i]] for i in 1:12]
        phi_local = rd.T12 * phi_elem
        g_P = dot(phi_local, Kg_unit * phi_local)

        # P = E·A/L · (u₂ₓ - u₁ₓ) in local frame → dP/du_local
        dP_du = zeros(12)
        dP_du[1] = -E * A / L
        dP_du[7] = E * A / L

        f_local = g_P .* dP_du
        f_global = rd.T12' * f_local
        for i in 1:12
            f_adj[rd.dofs[i]] += f_global[i]
        end
    end

    # CBEAM (same treatment)
    for (_, bar) in get(model, "CBEAMs", Dict())
        pid_str = string(bar["PID"])
        prop = get(pbarls, pid_str, nothing)
        if isnothing(prop); prop = get(get(model, "PBEAMs", Dict()), pid_str, nothing); end
        if isnothing(prop); continue; end
        mid_str = string(prop["MID"])
        mat = get(mats, mid_str, nothing)
        if isnothing(mat); continue; end
        if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
        i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]

        rd = _rod_local_frame_and_transform(i1, i2, node_coords, node_R)
        if isnothing(rd); continue; end

        E = mat["E"]; A = prop["A"]; L = rd.L
        Kg_unit = FEM.geometric_stiffness_frame3d(L, 1.0)
        phi_elem = [phi[rd.dofs[i]] for i in 1:12]
        phi_local = rd.T12 * phi_elem
        g_P = dot(phi_local, Kg_unit * phi_local)

        dP_du = zeros(12)
        dP_du[1] = -E * A / L; dP_du[7] = E * A / L

        f_local = g_P .* dP_du
        f_global = rd.T12' * f_local
        for i in 1:12; f_adj[rd.dofs[i]] += f_global[i]; end
    end
end

# ============================================================================
# Rod elements: ∂(φᵀ·Kg·φ)/∂u via axial force derivative
# ============================================================================

function _buckling_rhs_rods!(f_adj, model, id_map, node_coords, node_R, u_global, phi)
    prods = get(model, "PRODs", Dict())
    mats = model["MATs"]

    for (_, rod) in get(model, "CRODs", Dict())
        pid_str = string(rod["PID"])
        prop = get(prods, pid_str, nothing)
        if isnothing(prop); continue; end
        mat = get(mats, string(prop["MID"]), nothing)
        if isnothing(mat); continue; end
        if !haskey(id_map, rod["GA"]) || !haskey(id_map, rod["GB"]); continue; end
        i1, i2 = id_map[rod["GA"]], id_map[rod["GB"]]

        rd = _rod_local_frame_and_transform(i1, i2, node_coords, node_R)
        if isnothing(rd); continue; end

        E = mat["E"]; A = prop["A"]; L = rd.L
        Kg_unit = FEM.geometric_stiffness_rod(L, 1.0)
        phi_elem = [phi[rd.dofs[i]] for i in 1:12]
        phi_local = rd.T12 * phi_elem
        g_P = dot(phi_local, Kg_unit * phi_local)

        dP_du = zeros(12)
        dP_du[1] = -E * A / L; dP_du[7] = E * A / L

        f_local = g_P .* dP_du
        f_global = rd.T12' * f_local
        for i in 1:12; f_adj[rd.dofs[i]] += f_global[i]; end
    end

    for (_, rod) in get(model, "CONRODs", Dict())
        mat = get(mats, string(rod["MID"]), nothing)
        if isnothing(mat); continue; end
        if !haskey(id_map, rod["GA"]) || !haskey(id_map, rod["GB"]); continue; end
        i1, i2 = id_map[rod["GA"]], id_map[rod["GB"]]

        rd = _rod_local_frame_and_transform(i1, i2, node_coords, node_R)
        if isnothing(rd); continue; end

        E = mat["E"]; A = rod["A"]; L = rd.L
        Kg_unit = FEM.geometric_stiffness_rod(L, 1.0)
        phi_elem = [phi[rd.dofs[i]] for i in 1:12]
        phi_local = rd.T12 * phi_elem
        g_P = dot(phi_local, Kg_unit * phi_local)

        dP_du = zeros(12)
        dP_du[1] = -E * A / L; dP_du[7] = E * A / L

        f_local = g_P .* dP_du
        f_global = rd.T12' * f_local
        for i in 1:12; f_adj[rd.dofs[i]] += f_global[i]; end
    end
end

# ============================================================================
# ∂Kg/∂x|u_fixed contracted with mode shape: φᵀ·∂Kg/∂x|u_fixed·φ
# ============================================================================

"""
    compute_dKg_dx_phi(dv, model, id_map, node_coords, node_R, u_global, phi, ndof) -> Dict{String, Float64}

Compute φᵀ · ∂Kg/∂x|u_fixed · φ per design variable group.
For shell_thickness: exact ∂Kg/∂h = Kg/h (Kg ∝ h at fixed stress, since σ = D·B·u is h-independent)
For material_E:      ∂Kg/∂E = Kg/E (at fixed u, σ ∝ E, so Kg ∝ E)
For material_NU:     exact shell contraction through ∂σ/∂ν and Kg(σ) linearity
For bar_area:        exact ∂Kg/∂A = Kg/A for bar-column geometric stiffness
"""
function compute_dKg_dx_phi(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    dv_type = dv["type"]

    if dv_type == "shell_thickness"
        return _dKg_dx_phi_thickness(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    elseif dv_type == "material_E"
        return _dKg_dx_phi_material_E(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    elseif dv_type == "material_NU"
        return _dKg_dx_phi_material_NU(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    elseif dv_type == "bar_area"
        return _dKg_dx_phi_bar_area(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    elseif dv_type == "node_coord"
        return _dKg_dx_phi_node_coord(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    elseif dv_type == "topology_density"
        return _dKg_dx_phi_topology_density(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    else
        error("[ADJOINT-BUCK] Unsupported DV type for dKg/dx: $dv_type")
    end
end

function _dKg_dx_phi_thickness(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    # At fixed u, σ = D·Bm·u (no h dependence), but Kg ∝ h (thickness in integration).
    # So ∂Kg/∂h = Kg/h → φᵀ·∂Kg/∂h·φ = (1/h)·φᵀ·Kg·φ
    #
    # Use the full assembled Kg (from forward solve) for accuracy.
    # For a single PID: φᵀ·∂Kg/∂h·φ = (1/h)·φᵀ·Kg_full·φ (all elements share same h)
    # For multiple PIDs: compute per-PID contribution from the exact element geometric stiffness.
    pids = Set(string.(dv["pids"]))
    result = Dict{String, Float64}()

    # Check if all shell elements share one of the DV PIDs (common case)
    all_pids_in_model = Set(string(el["PID"]) for (_, el) in model["CSHELLs"])
    other_pids = setdiff(all_pids_in_model, pids)
    bars_contribute = !isempty(get(model, "CBARs", Dict())) || !isempty(get(model, "CRODs", Dict()))

    if length(pids) == 1 && isempty(other_pids) && !bars_contribute
        # All Kg comes from shell elements with this single PID — use full Kg
        pid_str = first(pids)
        h = model["PSHELLs"][pid_str]["T"]
        result["PID_$pid_str"] = dot(phi, Kg * phi) / h
    else
        covariant_iso_support = _covariant_iso_quad4_support(model, id_map, node_coords)
        # Per-PID: sum exact element contributions using ∂Kg/∂h = Kg_elem / h
        for pid_str in pids
            val = 0.0
            for (_, el) in model["CSHELLs"]
                if string(el["PID"]) != pid_str; continue; end

                covariant_iso_ctx = _covariant_iso_quad4_kg_context(
                    el, model, id_map, node_coords, node_R, u_global, covariant_iso_support
                )
                if !isnothing(covariant_iso_ctx)
                    phi_elem = [phi[covariant_iso_ctx.qd.dofs[i]] for i in 1:covariant_iso_ctx.qd.ndof_elem]
                    phi_local = covariant_iso_ctx.qd.T_mat * phi_elem
                    val += _covariant_iso_quad4_phi_dKg_dh(covariant_iso_ctx, phi_local)
                    continue
                end

                flat_iso_ctx = _flat_iso_quad4_kg_context(
                    el, model, id_map, node_coords, node_R, u_global
                )
                if !isnothing(flat_iso_ctx)
                    phi_elem = [phi[flat_iso_ctx.qd.dofs[i]] for i in 1:flat_iso_ctx.qd.ndof_elem]
                    phi_local = flat_iso_ctx.qd.T_mat * phi_elem
                    val += _flat_iso_quad4_phi_dKg_dh(flat_iso_ctx, phi_local)
                    continue
                end

                ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
                if isnothing(ed); continue; end

                phi_elem = [phi[ed.dofs[i]] for i in 1:ed.ndof_elem]
                u_elem = [u_global[ed.dofs[i]] for i in 1:ed.ndof_elem]
                phi_local = ed.T_mat * phi_elem
                u_local = ed.T_mat * u_elem

                h = ed.prop["T"]; E = ed.mat["E"]; nu = ed.mat["NU"]
                Bm, _, D = _shell_centroid_B_matrices(ed.n_nodes, ed.lc, E, nu)
                sigma_mem = D * Bm * u_local
                Kg_elem = if ed.n_nodes == 4
                    FEM.geometric_stiffness_quad4(ed.lc, sigma_mem, h)
                else
                    FEM.geometric_stiffness_tria3(ed.lc, sigma_mem, h)
                end
                dKg_dh = Kg_elem / h
                val += dot(phi_local, dKg_dh * phi_local)
            end
            result["PID_$pid_str"] = val
        end
    end
    return result
end

function _dKg_dx_phi_material_E(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    # At fixed u: σ = D(E)·Bm·u ∝ E, so Kg ∝ E → ∂Kg/∂E = Kg/E
    mids = Set(string.(dv["mids"]))
    result = Dict{String, Float64}()

    for mid_str in mids
        val = 0.0
        E_mid = model["MATs"][mid_str]["E"]
        for (_, el) in model["CSHELLs"]
            ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
            if isnothing(ed); continue; end
            if ed.mid_str != mid_str; continue; end

            phi_elem = [phi[ed.dofs[i]] for i in 1:ed.ndof_elem]
            u_elem = [u_global[ed.dofs[i]] for i in 1:ed.ndof_elem]
            phi_local = ed.T_mat * phi_elem
            u_local = ed.T_mat * u_elem

            h = ed.prop["T"]; E = ed.mat["E"]; nu = ed.mat["NU"]
            Bm, _, D = _shell_centroid_B_matrices(ed.n_nodes, ed.lc, E, nu)
            sigma_mem = D * Bm * u_local

            if ed.n_nodes == 4
                Kg_elem = FEM.geometric_stiffness_quad4(ed.lc, sigma_mem, h)
            else
                Kg_elem = FEM.geometric_stiffness_tria3(ed.lc, sigma_mem, h)
            end
            val += dot(phi_local, Kg_elem * phi_local) / E
        end

        # Also bars with this MID
        pbarls = get(model, "PBARLs", Dict())
        for (_, bar) in get(model, "CBARs", Dict())
            prop = get(pbarls, string(bar["PID"]), nothing)
            if isnothing(prop); continue; end
            if string(prop["MID"]) != mid_str; continue; end
            mat = get(model["MATs"], mid_str, nothing)
            if isnothing(mat); continue; end
            if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
            i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]
            rd = _rod_local_frame_and_transform(i1, i2, node_coords, node_R)
            if isnothing(rd); continue; end

            E = mat["E"]; A = prop["A"]; L = rd.L
            u_elem = [u_global[rd.dofs[i]] for i in 1:12]
            u_local = rd.T12 * u_elem
            P = E * A / L * (u_local[7] - u_local[1])
            Kg_elem = FEM.geometric_stiffness_frame3d(L, P)
            phi_elem = [phi[rd.dofs[i]] for i in 1:12]
            phi_local = rd.T12 * phi_elem
            # Kg ∝ P ∝ E (at fixed u)
            val += dot(phi_local, Kg_elem * phi_local) / E
        end

        result["MID_$mid_str"] = val
    end
    return result
end

function _dKg_dx_phi_material_NU(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    # At fixed u and geometry for isotropic shells:
    #   σ = D(ν)·Bm·u, and Kg is linear in σ
    # so:
    #   ∂Kg/∂ν = Kg(∂σ/∂ν),   ∂σ/∂ν = (∂D/∂ν)·Bm·u
    mids = Set(string.(dv["mids"]))
    result = Dict{String, Float64}()
    covariant_iso_support = _covariant_iso_quad4_support(model, id_map, node_coords)

    for mid_str in mids
        val = 0.0
        for (_, el) in model["CSHELLs"]
            covariant_iso_ctx = _covariant_iso_quad4_kg_context(
                el, model, id_map, node_coords, node_R, u_global, covariant_iso_support
            )
            if !isnothing(covariant_iso_ctx) && covariant_iso_ctx.mid_str == mid_str
                phi_elem = [phi[covariant_iso_ctx.qd.dofs[i]] for i in 1:covariant_iso_ctx.qd.ndof_elem]
                phi_local = covariant_iso_ctx.qd.T_mat * phi_elem
                val += _covariant_iso_quad4_phi_dKg_dnu(covariant_iso_ctx, phi_local)
                continue
            end

            flat_iso_ctx = _flat_iso_quad4_kg_context(el, model, id_map, node_coords, node_R, u_global)
            if !isnothing(flat_iso_ctx) && flat_iso_ctx.mid_str == mid_str
                phi_elem = [phi[flat_iso_ctx.qd.dofs[i]] for i in 1:flat_iso_ctx.qd.ndof_elem]
                phi_local = flat_iso_ctx.qd.T_mat * phi_elem
                val += _flat_iso_quad4_phi_dKg_dnu(flat_iso_ctx, phi_local)
                continue
            end

            ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
            if isnothing(ed); continue; end
            if ed.mid_str != mid_str; continue; end

            phi_elem = [phi[ed.dofs[i]] for i in 1:ed.ndof_elem]
            u_elem = [u_global[ed.dofs[i]] for i in 1:ed.ndof_elem]
            phi_local = ed.T_mat * phi_elem
            u_local = ed.T_mat * u_elem

            h = ed.prop["T"]; E = ed.mat["E"]; nu = ed.mat["NU"]
            Bm, _, _ = _shell_centroid_B_matrices(ed.n_nodes, ed.lc, E, nu)
            dD_dnu = _shell_plane_stress_dD_dnu(E, nu)
            dsigma_dnu = dD_dnu * Bm * u_local

            dKg_dnu = if ed.n_nodes == 4
                FEM.geometric_stiffness_quad4(ed.lc, dsigma_dnu, h)
            else
                FEM.geometric_stiffness_tria3(ed.lc, dsigma_dnu, h)
            end
            val += dot(phi_local, dKg_dnu * phi_local)
        end
        result["MID_$mid_str"] = val
    end
    return result
end

function _shell_plane_stress_dD_dnu(E::Float64, nu::Float64)
    scale = E / (1.0 - nu^2)
    dscale_dnu = 2.0 * E * nu / (1.0 - nu^2)^2

    A = [1.0 nu 0.0; nu 1.0 0.0; 0.0 0.0 (1.0 - nu) / 2.0]
    dA_dnu = [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 -0.5]
    return dscale_dnu .* A .+ scale .* dA_dnu
end

function _eval_phi_Kg_phi_shells_at_nu(mid_str, nu_val, model, id_map, node_coords, node_R, u_global, phi)
    val = 0.0
    covariant_iso_support = _covariant_iso_quad4_support(model, id_map, node_coords)

    for (_, el) in model["CSHELLs"]
        covariant_iso_ctx = _covariant_iso_quad4_kg_context(
            el, model, id_map, node_coords, node_R, u_global, covariant_iso_support
        )
        if !isnothing(covariant_iso_ctx) && covariant_iso_ctx.mid_str == mid_str
            phi_elem = [phi[covariant_iso_ctx.qd.dofs[i]] for i in 1:covariant_iso_ctx.qd.ndof_elem]
            phi_local = covariant_iso_ctx.qd.T_mat * phi_elem
            val += _covariant_iso_quad4_phi_Kg_phi_at_nu(covariant_iso_ctx, phi_local, nu_val)
            continue
        end

        flat_iso_ctx = _flat_iso_quad4_kg_context(el, model, id_map, node_coords, node_R, u_global)
        if !isnothing(flat_iso_ctx) && flat_iso_ctx.mid_str == mid_str
            phi_elem = [phi[flat_iso_ctx.qd.dofs[i]] for i in 1:flat_iso_ctx.qd.ndof_elem]
            phi_local = flat_iso_ctx.qd.T_mat * phi_elem
            val += _flat_iso_quad4_phi_Kg_phi_at_nu(flat_iso_ctx, phi_local, nu_val)
            continue
        end

        ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
        if isnothing(ed); continue; end
        if ed.mid_str != mid_str; continue; end

        phi_elem = [phi[ed.dofs[i]] for i in 1:ed.ndof_elem]
        u_elem = [u_global[ed.dofs[i]] for i in 1:ed.ndof_elem]
        phi_local = ed.T_mat * phi_elem
        u_local = ed.T_mat * u_elem

        h = ed.prop["T"]; E = ed.mat["E"]
        Bm, _, D_nu = _shell_centroid_B_matrices(ed.n_nodes, ed.lc, E, nu_val)
        sigma_mem = D_nu * Bm * u_local

        if ed.n_nodes == 4
            Kg_elem = FEM.geometric_stiffness_quad4(ed.lc, sigma_mem, h)
        else
            Kg_elem = FEM.geometric_stiffness_tria3(ed.lc, sigma_mem, h)
        end
        val += dot(phi_local, Kg_elem * phi_local)
    end
    return val
end

function _dKg_dx_phi_bar_area(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    # For bars: P = E·A/L·Δu → at fixed u, dP/dA = P/A, so dKg/dA = Kg/A
    pids = Set(string.(dv["pids"]))
    pbarls = get(model, "PBARLs", Dict())
    mats = model["MATs"]
    result = Dict{String, Float64}()

    for pid_str in pids
        val = 0.0
        for (_, bar) in get(model, "CBARs", Dict())
            if string(bar["PID"]) != pid_str; continue; end
            prop = get(pbarls, pid_str, nothing)
            if isnothing(prop); continue; end
            mat = get(mats, string(prop["MID"]), nothing)
            if isnothing(mat); continue; end
            if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
            i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]
            rd = _rod_local_frame_and_transform(i1, i2, node_coords, node_R)
            if isnothing(rd); continue; end

            E = mat["E"]; A = prop["A"]; L = rd.L
            u_elem = [u_global[rd.dofs[i]] for i in 1:12]
            u_local = rd.T12 * u_elem
            P = E * A / L * (u_local[7] - u_local[1])
            Kg_elem = FEM.geometric_stiffness_frame3d(L, P)
            phi_elem = [phi[rd.dofs[i]] for i in 1:12]
            phi_local = rd.T12 * phi_elem
            val += dot(phi_local, Kg_elem * phi_local) / A
        end
        result["PID_$pid_str"] = val
    end
    return result
end

function _dKg_dx_phi_node_coord(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    # Node coordinate affects both σ (through B matrices) and Kg geometry.
    # Use full Kg reassembly FD at fixed u: perturb coord, reassemble K (for new
    # node_coords/node_R), then assemble Kg with new geometry but same u.
    grid = Int(dv["grid"]); comp = Int(dv["comp"])
    grid_str = string(grid)
    coords_arr = model["GRIDs"][grid_str]["X"]
    x0 = Float64(coords_arr[comp])
    delta = max(abs(x0) * 1e-5, 1e-8)
    label = "GRID_$(grid)_$(comp)"
    result = Dict{String, Float64}()

    # Helper: reassemble K to get updated node_coords/node_R, then assemble Kg
    function _Kg_at_perturbed(d)
        coords_arr[comp] = x0 + d
        _, id_map_p, nc_p, ndof_p, nr_p, _, rbe3_p, snorm_p, _ = assemble_stiffness(model)
        Kg_p = assemble_geometric_stiffness(model, id_map_p, nc_p, nr_p, ndof_p, u_global, snorm_p, rbe3_p)
        coords_arr[comp] = x0  # restore
        return Kg_p
    end

    Kg_plus = _Kg_at_perturbed(delta)
    Kg_minus = _Kg_at_perturbed(-delta)

    phi_dKg_phi = dot(phi, (Kg_plus - Kg_minus) * phi) / (2.0 * delta)
    result[label] = phi_dKg_phi
    return result
end

function _dKg_dx_phi_topology_density(dv, model, id_map, node_coords, node_R, u_global, phi, Kg, ndof)
    # SIMP: at fixed u, stress σ = D·B·u doesn't depend on ρ, and Kg depends on σ and h.
    # Neither σ nor h depends on ρ → ∂Kg/∂ρ|u_fixed = 0 for all elements.
    # (The ρ scaling affects K only, not Kg directly.)
    result = Dict{String, Float64}()
    for eid in dv["eids"]
        result["EID_$(string(Int(eid)))"] = 0.0
    end
    return result
end

# ============================================================================
# Main buckling adjoint solver
# ============================================================================

"""
    solve_adjoint_buckling(results::Dict, adjoint_config_path::String) -> Dict

Run adjoint sensitivity analysis on SOL 105 buckling results.
Computes dλ/dx for each eigenvalue and design variable.
"""
function solve_adjoint_buckling(results::Dict, adjoint_config_path::String)
    if results["sol_type"] != 105
        error("[ADJOINT-BUCK] Requires SOL 105 results (got SOL $(results["sol_type"]))")
    end

    config = parse_adjoint_config(adjoint_config_path)
    design_vars = config["design_variables"]

    model = results["model"]
    id_map = results["id_map"]
    X = results["node_coords"]
    K_static = results["K"]
    K_eig = get(results, "K_eig", K_static)
    ndof = results["ndof"]
    node_R = results["node_R"]
    Kg = results["Kg"]
    u_static = results["u_static"]
    eigenvalues = results["eigenvalues"]
    mode_shapes = results["_raw_mode_shapes"]
    fixed_dofs = results["fixed_dofs"]

    n_modes = length(eigenvalues)
    n_dv = length(design_vars)
    log_msg("[ADJOINT-BUCK] Buckling sensitivity: $n_modes modes × $n_dv design variables")

    # Build free DOFs and factorize K once
    free_dofs = sort(collect(setdiff(1:ndof, fixed_dofs)))
    K_ff = K_static[free_dofs, free_dofs]
    log_msg("[ADJOINT-BUCK] Factorizing K_ff ($(length(free_dofs)) free DOFs)...")
    K_fact = cholesky(Symmetric(K_ff))

    sensitivities = Dict{String, Dict{String, Dict{String, Float64}}}()
    eigenvalue_values = Dict{String, Float64}()
    full_sensitivity_fd_cache = Dict{String, Dict{String, Vector{Float64}}}()
    full_static_dKdx_cache = Dict{String, Dict{String, SparseMatrixCSC{Float64, Int}}}()
    full_eig_dKdx_cache = Dict{String, Dict{String, SparseMatrixCSC{Float64, Int}}}()
    full_dKg_cache = Dict{String, Dict{String, SparseMatrixCSC{Float64, Int}}}()
    static_dKdx_u_cache = Dict{String, Dict{String, Vector{Float64}}}()
    target_labels_cache = Dict{String, Set{String}}()
    full_sensitivity_fd_labels_cache = Dict{String, Set{String}}()
    full_dKg_labels_cache = Dict{String, Set{String}}()
    special_labels_cache = Dict{String, Set{String}}()
    local_dKg_dv_cache = Dict{String, Union{Nothing, Dict{String, Any}}}()
    local_dKdx_dv_cache = Dict{String, Union{Nothing, Dict{String, Any}}}()

    for dv in design_vars
        dv_id = dv["id"]
        target_labels_cache[dv_id] = _buckling_dKdx_group_labels(dv)
        full_sensitivity_fd_labels_cache[dv_id] = _buckling_full_sensitivity_fd_group_labels(
            dv, model, id_map, X, node_R, u_static
        )
        full_dKg_labels_cache[dv_id] = _buckling_dKg_full_model_group_labels(
            dv, model, id_map, X, node_R, u_static
        )
        special_labels_cache[dv_id] = _buckling_dKdx_full_model_group_labels(
            dv, model, id_map, X, node_R, u_static
        )
        local_dKg_dv_cache[dv_id] = _subset_dv_by_group_labels(
            dv, setdiff(target_labels_cache[dv_id], full_dKg_labels_cache[dv_id])
        )
        local_dKdx_dv_cache[dv_id] = _subset_dv_by_group_labels(
            dv, setdiff(target_labels_cache[dv_id], special_labels_cache[dv_id])
        )
    end
    for mode_idx in 1:n_modes
        lam = eigenvalues[mode_idx]
        phi = mode_shapes[:, mode_idx]
        mode_id = "mode_$mode_idx"

        log_msg("[ADJOINT-BUCK]   Mode $mode_idx (λ = $(round(lam, sigdigits=6)))")
        eigenvalue_values[mode_id] = lam

        # Denominator: φᵀ·Kg·φ
        phi_Kg_phi = dot(phi, Kg * phi)
        if abs(phi_Kg_phi) < 1e-30
            log_msg("[ADJOINT-BUCK]   WARNING: φᵀ·Kg·φ ≈ 0, skipping mode $mode_idx")
            sensitivities[mode_id] = Dict{String, Dict{String, Float64}}()
            continue
        end

        # Adjoint RHS: f = λ · ∂(φᵀ·Kg(σ(u))·φ)/∂u
        f_adj_full = lam .* compute_buckling_adjoint_rhs(model, id_map, X, node_R, u_static, phi, ndof)

        # Solve adjoint: K·ψ = f_adj
        f_adj_f = f_adj_full[free_dofs]
        psi_f = K_fact \ f_adj_f
        psi = zeros(ndof)
        psi[free_dofs] = psi_f

        # Compute sensitivities for each DV
        sensitivities[mode_id] = Dict{String, Dict{String, Float64}}()
        for dv in design_vars
            dv_id = dv["id"]
            target_labels = target_labels_cache[dv_id]
            full_sensitivity_fd_labels = full_sensitivity_fd_labels_cache[dv_id]
            full_dKg_labels = full_dKg_labels_cache[dv_id]
            full_sensitivity_fd_groups = if isempty(full_sensitivity_fd_labels)
                Dict{String, Vector{Float64}}()
            else
                get!(full_sensitivity_fd_cache, dv_id) do
                    _full_model_buckling_fd_group_sensitivities(
                        dv, results, full_sensitivity_fd_labels
                    )
                end
            end
            use_full_sensitivity_only =
                !isempty(target_labels) &&
                all(label in full_sensitivity_fd_labels for label in target_labels)

            if use_full_sensitivity_only
                group_sens = Dict{String, Float64}()
                for group_label in target_labels
                    if haskey(full_sensitivity_fd_groups, group_label) &&
                       mode_idx <= length(full_sensitivity_fd_groups[group_label])
                        group_sens[group_label] = full_sensitivity_fd_groups[group_label][mode_idx]
                    end
                end
                sensitivities[mode_id][dv_id] = group_sens
                continue
            end

            # Term 2: λ·φᵀ·∂Kg/∂x|u_fixed·φ per group
            local_dKg_dv = local_dKg_dv_cache[dv_id]
            dKg_dx_phi_groups = isnothing(local_dKg_dv) ?
                Dict{String, Float64}() :
                compute_dKg_dx_phi(local_dKg_dv, model, id_map, X, node_R, u_static, phi, Kg, ndof)
            if !isempty(full_dKg_labels)
                full_dKg_diffs = get!(full_dKg_cache, dv_id) do
                    _full_model_dKg_dx_matrix_diffs(dv, model, u_static)
                end
                for group_label in full_dKg_labels
                    if haskey(full_dKg_diffs, group_label)
                        dKg_dx_phi_groups[group_label] = dot(phi, full_dKg_diffs[group_label] * phi)
                    end
                end
            end

            # Terms 1 & 3 via full-model reassembly FD for accuracy.
            # For material_E: K/E is exact, no need for reassembly.
            group_sens = Dict{String, Float64}()
            dv_method = get_dv_method(dv)
            if dv["type"] == "material_E"
                # Analytical: dK/dE = K/E (uses full assembled K — exact)
                phi_K_phi = dot(phi, K_eig * phi)
                psi_K_u = dot(psi, K_static * u_static)
                for mid in dv["mids"]
                    mid_str = string(Int(mid))
                    E_val = model["MATs"][mid_str]["E"]
                    phi_dK_phi = phi_K_phi / E_val
                    psi_dK_u = psi_K_u / E_val
                    lam_dKg = lam * get(dKg_dx_phi_groups, "MID_$mid_str", 0.0)
                    group_sens["MID_$mid_str"] = -(phi_dK_phi + lam_dKg - psi_dK_u) / phi_Kg_phi
                end
            elseif dv_method in (:analytical, :ad_forward) && dv["type"] != "material_E"
                # Analytical and AD-enabled DV types: use per-group dKdx infrastructure
                special_labels = special_labels_cache[dv_id]
                use_full_only = !isempty(target_labels) && all(label in special_labels for label in target_labels)

                dKdx_phi_groups = Dict{String, Vector{Float64}}()
                dKdx_u_groups = Dict{String, Vector{Float64}}()
                if !use_full_only
                    local_dKdx_dv = local_dKdx_dv_cache[dv_id]
                    if !isnothing(local_dKdx_dv)
                        dKdx_phi_groups = compute_dKdx_u_per_group(
                            local_dKdx_dv, model, id_map, X, node_R, phi, ndof
                        )
                    end
                    dKdx_u_groups = get!(static_dKdx_u_cache, dv_id) do
                        if isnothing(local_dKdx_dv)
                            Dict{String, Vector{Float64}}()
                        else
                            compute_dKdx_u_per_group(
                                local_dKdx_dv, model, id_map, X, node_R, u_static, ndof
                            )
                        end
                    end
                end

                bilinear = Dict{String, Tuple{Float64, Float64}}()
                if !isempty(special_labels)
                    static_diffs = get!(full_static_dKdx_cache, dv_id) do
                        _full_model_dKdx_matrix_diffs(dv, model)
                    end
                    eig_diffs = get!(full_eig_dKdx_cache, dv_id) do
                        _full_model_dKdx_matrix_diffs(dv, model; _sol105_eig_stiffness_fd_kwargs()...)
                    end
                    bilinear = _contract_full_model_dKdx_split_bilinear(
                        static_diffs, eig_diffs, phi, u_static, psi
                    )
                end

                group_labels = Set{String}()
                union!(group_labels, keys(dKdx_phi_groups))
                union!(group_labels, keys(dKg_dx_phi_groups))
                union!(group_labels, special_labels)
                for group_label in group_labels
                    if group_label in special_labels && haskey(bilinear, group_label)
                        phi_dK_phi, psi_dK_u = bilinear[group_label]
                    else
                        phi_dK_phi = haskey(dKdx_phi_groups, group_label) ?
                            dot(phi, dKdx_phi_groups[group_label]) : 0.0
                        psi_dK_u = haskey(dKdx_u_groups, group_label) ?
                            dot(psi, dKdx_u_groups[group_label]) : 0.0
                    end
                    lam_dKg = lam * get(dKg_dx_phi_groups, group_label, 0.0)
                    group_sens[group_label] = -(phi_dK_phi + lam_dKg - psi_dK_u) / phi_Kg_phi
                end
            else
                # Full-model FD: reuse cached static/eigen stiffness derivatives
                static_diffs = get!(full_static_dKdx_cache, dv_id) do
                    _full_model_dKdx_matrix_diffs(dv, model)
                end
                eig_diffs = get!(full_eig_dKdx_cache, dv_id) do
                    _full_model_dKdx_matrix_diffs(dv, model; _sol105_eig_stiffness_fd_kwargs()...)
                end
                bilinear = _contract_full_model_dKdx_split_bilinear(
                    static_diffs, eig_diffs, phi, u_static, psi
                )
                for (group_label, (phi_dK_phi, psi_dK_u)) in bilinear
                    lam_dKg = lam * get(dKg_dx_phi_groups, group_label, 0.0)
                    group_sens[group_label] = -(phi_dK_phi + lam_dKg - psi_dK_u) / phi_Kg_phi
                end
            end
            for group_label in full_sensitivity_fd_labels
                if haskey(full_sensitivity_fd_groups, group_label) &&
                   mode_idx <= length(full_sensitivity_fd_groups[group_label])
                    group_sens[group_label] = full_sensitivity_fd_groups[group_label][mode_idx]
                end
            end
            sensitivities[mode_id][dv_id] = group_sens
        end
    end

    # DV current values
    dv_values = Dict{String, Dict{String, Float64}}()
    for dv in design_vars
        dv_values[dv["id"]] = get_design_variable_values(dv, model)
    end

    path_diagnostics = Dict{String, Dict{String, Any}}()
    dvs_with_full_sensitivity_fd = String[]
    dvs_with_full_dKg = String[]
    dvs_with_full_dKdx = String[]
    dvs_with_full_model_paths = String[]
    total_target_groups = 0
    total_full_sensitivity_fd_groups = 0
    total_full_dKg_groups = 0
    total_full_dKdx_groups = 0
    for dv in design_vars
        dv_id = dv["id"]
        target_labels = sort!(collect(target_labels_cache[dv_id]))
        full_sensitivity_fd_labels = sort!(collect(full_sensitivity_fd_labels_cache[dv_id]))
        full_dKg_labels = sort!(collect(full_dKg_labels_cache[dv_id]))
        full_dKdx_labels = sort!(collect(special_labels_cache[dv_id]))
        uses_full_sensitivity_fd = !isempty(full_sensitivity_fd_labels)
        uses_full_dKg = !isempty(full_dKg_labels)
        uses_full_dKdx = !isempty(full_dKdx_labels)

        total_target_groups += length(target_labels)
        total_full_sensitivity_fd_groups += length(full_sensitivity_fd_labels)
        total_full_dKg_groups += length(full_dKg_labels)
        total_full_dKdx_groups += length(full_dKdx_labels)

        if uses_full_sensitivity_fd
            push!(dvs_with_full_sensitivity_fd, dv_id)
        end
        if uses_full_dKg
            push!(dvs_with_full_dKg, dv_id)
        end
        if uses_full_dKdx
            push!(dvs_with_full_dKdx, dv_id)
        end
        if uses_full_sensitivity_fd || uses_full_dKg || uses_full_dKdx
            push!(dvs_with_full_model_paths, dv_id)
        end

        path_diagnostics[dv_id] = Dict(
            "target_labels" => target_labels,
            "full_sensitivity_fd_labels" => full_sensitivity_fd_labels,
            "full_dKg_labels" => full_dKg_labels,
            "full_dKdx_labels" => full_dKdx_labels,
            "uses_full_sensitivity_fd" => uses_full_sensitivity_fd,
            "uses_full_dKg" => uses_full_dKg,
            "uses_full_dKdx" => uses_full_dKdx,
            "n_target_labels" => length(target_labels),
            "n_full_sensitivity_fd_labels" => length(full_sensitivity_fd_labels),
            "n_full_dKg_labels" => length(full_dKg_labels),
            "n_full_dKdx_labels" => length(full_dKdx_labels),
        )
    end

    sort!(dvs_with_full_sensitivity_fd)
    sort!(dvs_with_full_dKg)
    sort!(dvs_with_full_dKdx)
    sort!(dvs_with_full_model_paths)

    path_summary = Dict{String, Any}(
        "n_design_variables" => length(design_vars),
        "n_design_variables_with_full_sensitivity_fd" => length(dvs_with_full_sensitivity_fd),
        "n_design_variables_with_full_dKg" => length(dvs_with_full_dKg),
        "n_design_variables_with_full_dKdx" => length(dvs_with_full_dKdx),
        "n_design_variables_with_full_model_paths" => length(dvs_with_full_model_paths),
        "n_target_groups" => total_target_groups,
        "n_full_sensitivity_fd_groups" => total_full_sensitivity_fd_groups,
        "n_full_dKg_groups" => total_full_dKg_groups,
        "n_full_dKdx_groups" => total_full_dKdx_groups,
        "design_variables_with_full_sensitivity_fd" => dvs_with_full_sensitivity_fd,
        "design_variables_with_full_dKg" => dvs_with_full_dKg,
        "design_variables_with_full_dKdx" => dvs_with_full_dKdx,
        "design_variables_with_full_model_paths" => dvs_with_full_model_paths,
    )

    return Dict(
        "sensitivities" => sensitivities,
        "eigenvalue_values" => eigenvalue_values,
        "design_variable_values" => dv_values,
        "path_diagnostics" => path_diagnostics,
        "path_summary" => path_summary,
    )
end
