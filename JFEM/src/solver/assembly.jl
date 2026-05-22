# assembly.jl — Global stiffness matrix assembly
#
# =============================================================================
# Q4 KERNEL DISPATCH OVERVIEW
# =============================================================================
# `assemble_stiffness` iterates every CQUAD4 element and picks ONE of the
# stiffness kernels defined in FEMKernels.jl. The dispatch chain lives around
# line 3570-3744 (K) and 5800-5920 (K_g). This block documents what the chain
# actually does. Update this block when the dispatch logic changes.
#
# Read the file as: "for each element, walk the chain top-down; the first
# branch whose condition is true wins". The fallback `else` at the end is the
# production default (`stiffness_quad4_matrices` / `geometric_stiffness_quad4`).
#
# Per-element K stiffness dispatch (see line ~3570):
#
#  # | Condition                                       | Kernel called                              | Default? | Notes
# ---+-------------------------------------------------+--------------------------------------------+----------+------
#  1 | elem_mitc4_3d_kernel                            | stiffness_quad4_mitc4_3d_{ply,resultant}   | OFF      | JFEM_Q4_KERNEL=mitc4_3d (default "macneal")
#  2 | elem_shear_center_only && is_iso_ei             | stiffness_quad4_matrices (center+blend)    | ON*      | Iso curved-shell in K_eig path
#  3 | elem_flat_dkmq_branch                           | stiffness_quad4_plate_dkmq_matrices        | OFF      | JFEM_SOL105_EIG_FLAT_PCOMP_DKMQ
#  4 | elem_rect_plate_branch                          | stiffness_quad4_plate_adini_matrices       | OFF      | JFEM_SOL105_EIG_FLAT_PCOMP_RECT_ADINI
#  5 | elem_flat_plate_branch                          | stiffness_quad4_plate_dkq_matrices         | OFF      | JFEM_SOL105_EIG_FLAT_PCOMP_PLATE_BRANCH
#  6 | elem_shear_center_only && is_pcomp_ei (flat)    | stiffness_quad4_matrices (center+blend)    | ON*      | Composite curved-shell in K_eig path
#  7 | elem_curved_iso_blend > 0.0                     | stiffness_quad4_matrices (curved blend)    | partial  | Iso curved blend factor
#  8 | else (fallback)                                 | stiffness_quad4_matrices                   | ON       | Production default — covers everything else
#
# Per-element K_g (geometric) dispatch (see line ~5800):
#
#  # | Condition                                       | Kernel called                              | Default?
# ---+-------------------------------------------------+--------------------------------------------+---------
#  1 | kg_flat_dkmq_branch && is_pcomp_ei              | geometric_stiffness_quad4_plate_dkmq       | OFF (coupled to K#3)
#  2 | elem_rect_plate_branch && is_pcomp_ei           | geometric_stiffness_quad4_plate_adini      | OFF (coupled to K#4)
#  3 | elem_flat_plate_branch                          | geometric_stiffness_quad4_plate_dkq        | OFF (coupled to K#5)
#  4 | JFEM_SOL105_EIG_CURVED_JACOBIAN && coords_3d    | geometric_stiffness_quad4_covariant        | OFF
#  5 | else (fallback)                                 | geometric_stiffness_quad4                  | ON
#
# Live trace (2026-05-22) on HTP_launch GAME deck (10,274 PCOMP CQUAD4) under
# default settings confirms ONLY these 3 kernels actually fire:
#   - stiffness_quad4_matrices       (K, branch #8 fallback)
#   - add_quad4_macneal_shear_rbf!   (internal, called from stiffness_quad4_matrices)
#   - geometric_stiffness_quad4      (K_g, branch #5 fallback)
#
# Everything else in this dispatch chain is RESEARCH / RETAINED — opt-in via
# JFEM_Q4_KERNEL or JFEM_SOL105_EIG_FLAT_PCOMP_* env vars. See FEMKernels.jl
# header banners on each kernel for status and calibration knobs.
#
# *ON in K_eig path only (shear_center_only=true). With JFEM_SOL105_USE_STATIC_K
# default true, K_eig == K_static so branches #2 and #6 reduce to #8.
# =============================================================================

@inline function curved_iso_eig_fullshear_blend()
    raw = get(ENV, "JFEM_CURVED_ISO_EIG_FULLSHEAR_BLEND", "1.0")
    return clamp(something(tryparse(Float64, raw), 1.0), 0.0, 1.0)
end

@inline function curved_pcomp_eig_fullshear_blend()
    raw = get(ENV, "JFEM_CURVED_PCOMP_EIG_FULLSHEAR_BLEND", "1.0")
    return clamp(something(tryparse(Float64, raw), 1.0), 0.0, 1.0)
end

@inline function q4_geom_normals_nearly_constant(
    n1::SVector{3,Float64},
    n2::SVector{3,Float64},
    n3::SVector{3,Float64},
    n4::SVector{3,Float64};
    tol::Float64=1e-6,
)
    return norm(n1 - n2) <= tol &&
           norm(n1 - n3) <= tol &&
           norm(n1 - n4) <= tol &&
           norm(n2 - n3) <= tol &&
           norm(n2 - n4) <= tol &&
           norm(n3 - n4) <= tol
end

@inline function shell_project_frame_to_normal(
    v1::SVector{3,Float64},
    v2::SVector{3,Float64},
    v3::SVector{3,Float64},
    n::SVector{3,Float64},
)
    v3n = dot(n, v3) < 0.0 ? -n : n
    v1p = v1 - dot(v1, v3n) * v3n
    v1l = norm(v1p)
    if v1l <= 1e-12
        v2p = v2 - dot(v2, v3n) * v3n
        v2l = norm(v2p)
        if v2l <= 1e-12
            return v1, v2, v3
        end
        v1n = SVector{3}(v2p / v2l)
    else
        v1n = SVector{3}(v1p / v1l)
    end
    return v1n, SVector{3}(cross(v3n, v1n)), v3n
end


function build_node_has_line_elements(model, id_map, n_nodes)
    node_has_line = falses(n_nodes)
    for group_name in ("CBARs", "CBEAMs", "CRODs", "CONRODs")
        group = get(model, group_name, Dict())
        for (_, el) in group
            ga = get(id_map, get(el, "GA", 0), 0)
            gb = get(id_map, get(el, "GB", 0), 0)
            ga > 0 && (node_has_line[ga] = true)
            gb > 0 && (node_has_line[gb] = true)
        end
    end
    return node_has_line
end

@inline function q4_flat_iso_eig_membrane_incomp_enabled()
    raw = lowercase(strip(get(ENV, "JFEM_SOL105_EIG_FLAT_ISO_MEMBRANE_INCOMP", "true")))
    return raw in ("1", "true", "yes", "on")
end

@inline function q4_sol105_membrane_incomp_center_jacobian_enabled()
    raw = lowercase(strip(get(ENV, "JFEM_SOL105_MEMBRANE_INCOMP_CENTER_JACOBIAN", "false")))
    return raw in ("1", "true", "yes", "on")
end

@inline function q4_flat_iso_eig_membrane_shear_center_row_enabled()
    raw = lowercase(strip(get(ENV, "JFEM_SOL105_EIG_FLAT_ISO_MEMBRANE_SHEAR_CENTER_ROW", "false")))
    return raw in ("1", "true", "yes", "on")
end

@inline function q4_flat_iso_eig_membrane_assumed_mode()
    raw = lowercase(strip(get(ENV, "JFEM_SOL105_EIG_FLAT_ISO_MEMBRANE_ASSUMED_MODE", "none")))
    if raw in ("mitc4plus", "mitc4+", "ans")
        return :mitc4plus
    elseif raw in ("mitc4plus_all", "mitc4+_all", "ans_all", "all")
        return :mitc4plus_all
    else
        return :none
    end
end

@inline function q4_sol105_flat_iso_dkmq_enabled()
    raw = lowercase(strip(get(ENV, "JFEM_SOL105_EIG_FLAT_ISO_DKMQ", "true")))
    return raw in ("1", "true", "yes", "on")
end

@inline function kg_use_compatible_membrane_stress()
    # Keep legacy behavior unless the user opts into the experimental curved
    # shell formulation. In that branch, Kg should recover membrane resultants
    # from the same compatible field that feeds the covariant surface operator,
    # unless the user explicitly pins the old behavior with JFEM_KG_*.
    if !haskey(ENV, "JFEM_KG_USE_COMPATIBLE_MEMBRANE_STRESS") &&
       q4_eig_curved_jacobian_enabled() &&
       !kg_match_static_membrane_operator_enabled()
        return true
    end
    raw = lowercase(strip(get(ENV, "JFEM_KG_USE_COMPATIBLE_MEMBRANE_STRESS", "false")))
    return raw in ("1", "true", "yes", "on")
end

@inline function kg_quad4_stress_field_mode()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_STRESS_FIELD_MODE", "gauss")))
    if raw in ("avg", "average", "mean")
        return :average
    elseif raw in ("gp", "gauss", "field", "gauss_field")
        return :gauss
    else
        return :auto
    end
end

@inline function kg_quad4_consistent_membrane_operator_enabled()
    # If the element stiffness uses statically condensed membrane modes, the
    # SOL105 differential-stiffness operator should use the same condensed
    # displacement-gradient map. This is a formulation-consistency default, not
    # a laminate/test-case correction.
    return solver_env_bool("JFEM_KG_CONSISTENT_MEMBRANE_OPERATOR", true)
end

@inline function kg_quad4_gp_field_avg_ratio_max()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_GP_FIELD_AVG_RATIO_MAX", "0.35"))
    return max(something(tryparse(Float64, raw), 0.35), 0.0)
end

@inline function kg_quad4_auto_avg_shear_ratio_max()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_AUTO_AVG_SHEAR_RATIO_MAX", "0.02"))
    return clamp(something(tryparse(Float64, raw), 0.02), 0.0, 1.0)
end

@inline function kg_quad4_auto_avg_require_compression()
    return solver_env_bool("JFEM_KG_QUAD4_AUTO_AVG_REQUIRE_COMPRESSION", true)
end

@inline function kg_quad4_auto_avg_require_biaxial_compression()
    return solver_env_bool("JFEM_KG_QUAD4_AUTO_AVG_REQUIRE_BIAXIAL_COMPRESSION", true)
end

@inline function kg_quad4_auto_avg_require_geometry()
    return solver_env_bool("JFEM_KG_QUAD4_AUTO_AVG_REQUIRE_GEOMETRY", false)
end

@inline function kg_quad4_auto_avg_kappa_l_min()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_AUTO_AVG_KAPPA_L_MIN", "1.0e-8"))
    return max(something(tryparse(Float64, raw), 1.0e-8), 0.0)
end

@inline function kg_quad4_auto_avg_cyl_ratio_min()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_AUTO_AVG_CYL_RATIO_MIN", "0.9"))
    return clamp(something(tryparse(Float64, raw), 0.9), 0.0, 1.0)
end

@inline function kg_quad4_auto_avg_load_classifier_enabled()
    return solver_env_bool("JFEM_KG_QUAD4_AUTO_AVG_LOAD_CLASSIFIER", true)
end

@inline function kg_quad4_auto_avg_load_axis()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_AUTO_AVG_LOAD_AXIS", "x")))
    raw in ("2", "y") && return 2
    raw in ("3", "z") && return 3
    return 1
end

@inline function kg_quad4_auto_avg_load_sign()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_AUTO_AVG_LOAD_SIGN", "negative")))
    return raw in ("pos", "positive", "+", "tension") ? 1.0 : -1.0
end

@inline function kg_quad4_auto_avg_load_dominance_min()
    return clamp(solver_env_float("JFEM_KG_QUAD4_AUTO_AVG_LOAD_DOMINANCE_MIN", 0.9), 0.0, 1.0)
end

@inline function kg_quad4_auto_avg_load_transverse_ratio_max()
    return clamp(solver_env_float("JFEM_KG_QUAD4_AUTO_AVG_LOAD_TRANSVERSE_RATIO_MAX", 0.1), 0.0, 1.0)
end

function kg_quad4_sid_has_nonforce_load(model, sid::Int)
    for key in ("MOMENTs", "PLOAD4s", "PLOADs", "PLOAD1s", "GRAVs", "RFORCEs")
        for load in get(model, key, Any[])
            if haskey(load, "SID") && Int(load["SID"]) == sid
                return true
            end
        end
    end
    return false
end

function kg_quad4_collect_force_load_components!(
    signed_force::Vector{Float64},
    abs_force::Vector{Float64},
    model,
    sid::Int,
    scale::Float64,
    visited::Set{Int},
)
    sid in visited && return (found=false, unknown=true)
    push!(visited, sid)

    found = false
    unknown = kg_quad4_sid_has_nonforce_load(model, sid)
    for frc in get(model, "FORCEs", Any[])
        Int(frc["SID"]) == sid || continue
        dir = Float64.(frc["Dir"])
        global_dir = get_coord_transform(model, Int(get(frc, "CID", 0)), dir)
        vec = scale * Float64(frc["Mag"]) .* global_dir
        @inbounds for i in 1:3
            signed_force[i] += vec[i]
            abs_force[i] += abs(vec[i])
        end
        found = true
    end

    for combo in get(model, "LOAD_COMBOS", Any[])
        Int(combo["SID"]) == sid || continue
        combo_scale = scale * Float64(combo["S"])
        for sub in get(combo, "COMPS", Any[])
            sub_found, sub_unknown = kg_quad4_collect_force_load_components!(
                signed_force, abs_force, model, Int(sub["LID"]),
                combo_scale * Float64(sub["S"]), visited,
            )
            found |= sub_found
            unknown |= sub_unknown
        end
    end

    delete!(visited, sid)
    return (found=found, unknown=unknown)
end

function kg_quad4_auto_avg_load_classifier(model, static_load_id)
    kg_quad4_auto_avg_load_classifier_enabled() || return nothing
    isnothing(static_load_id) && return nothing
    sid = something(tryparse(Int, string(static_load_id)), nothing)
    isnothing(sid) && return nothing

    signed_force = zeros(3)
    abs_force = zeros(3)
    result = kg_quad4_collect_force_load_components!(
        signed_force, abs_force, model, sid, 1.0, Set{Int}(),
    )
    (!result.found || result.unknown) && return nothing

    total_abs = sum(abs_force)
    total_abs <= 1e-30 && return nothing
    axis = kg_quad4_auto_avg_load_axis()
    axis_abs = abs_force[axis]
    transverse_ratio = (total_abs - axis_abs) / total_abs
    dominance = axis_abs / total_abs
    signed_axis = signed_force[axis]
    sign_ok = kg_quad4_auto_avg_load_sign() * signed_axis > 1e-12 * total_abs
    return sign_ok &&
        dominance >= kg_quad4_auto_avg_load_dominance_min() &&
        transverse_ratio <= kg_quad4_auto_avg_load_transverse_ratio_max()
end

@inline function kg_quad4_membrane_recovery_mode()
    if !haskey(ENV, "JFEM_KG_QUAD4_MEMBRANE_RECOVERY_MODE") &&
       q4_eig_curved_jacobian_enabled()
        return :covariant
    end
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_MEMBRANE_RECOVERY_MODE", "planar")))
    if raw in ("covariant", "surface", "cov")
        return :covariant
    elseif raw in ("auto", "curved_auto", "surface_auto")
        return :auto
    elseif raw in ("tri_aspect", "triangle_aspect", "nastran1976", "nastran_tri")
        return :tri_aspect
    elseif raw in ("tri_center", "tri_center_adj", "triangle_center")
        return :tri_center_adj
    elseif raw in ("tri_incident", "tri_incident_interp", "triangle_incident")
        return :tri_incident_interp
    elseif raw in ("tri_diagavg", "triangle_diagavg", "diagavg")
        return :tri_diagavg
    else
        return :planar
    end
end

@inline function kg_quad4_membrane_tri_aspect_switch()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_MEMBRANE_TRI_ASPECT_SWITCH", "2.0"))
    return max(something(tryparse(Float64, raw), 2.0), 1.0)
end

@inline function kg_quad4_covariant_blend()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_COVARIANT_BLEND", "1.0"))
    return clamp(something(tryparse(Float64, raw), 1.0), 0.0, 1.0)
end

@inline function kg_quad4_covariant_auto_kappa_l_min()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_COVARIANT_AUTO_KAPPA_L_MIN", "0.01"))
    return max(something(tryparse(Float64, raw), 0.01), 0.0)
end

@inline function kg_quad4_covariant_auto_cyl_ratio_max()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_COVARIANT_AUTO_CYL_RATIO_MAX", "0.85"))
    return clamp(something(tryparse(Float64, raw), 0.85), 0.0, 1.0)
end

@inline function kg_quad4_auto_gp_patch_enabled()
    return solver_env_bool("JFEM_KG_QUAD4_AUTO_GP_PATCH", false)
end

@inline function kg_quad4_auto_gp_patch_valence_min()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_AUTO_GP_PATCH_VALENCE_MIN", "4"))
    return max(something(tryparse(Int, raw), 4), 1)
end

@inline function kg_quad4_auto_gp_patch_kappa_l_min()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_AUTO_GP_PATCH_KAPPA_L_MIN", "0.05"))
    return max(something(tryparse(Float64, raw), 0.05), 0.0)
end

@inline function kg_quad4_auto_gp_spread_enabled()
    return solver_env_bool("JFEM_KG_QUAD4_AUTO_GP_SPREAD", false)
end

@inline function kg_quad4_auto_gp_spread_min()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_AUTO_GP_SPREAD_MIN", "0.5"))
    return max(something(tryparse(Float64, raw), 0.5), 0.0)
end

@inline function kg_quad4_auto_gp_spread_kappa_l_min()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_AUTO_GP_SPREAD_KAPPA_L_MIN", "0.05"))
    return max(something(tryparse(Float64, raw), 0.05), 0.0)
end

@inline function kg_quad4_auto_gp_spread_cyl_ratio_min()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_AUTO_GP_SPREAD_CYL_RATIO_MIN", "0.9"))
    return clamp(something(tryparse(Float64, raw), 0.9), 0.0, 1.0)
end

@inline function kg_quad4_auto_gp_spread_blend_scale()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_AUTO_GP_SPREAD_BLEND_SCALE", "0.0"))
    return max(something(tryparse(Float64, raw), 0.0), 0.0)
end

@inline function kg_quad4_gp_field_blend_override()
    haskey(ENV, "JFEM_KG_QUAD4_GP_FIELD_BLEND") || return nothing
    raw = strip(get(ENV, "JFEM_KG_QUAD4_GP_FIELD_BLEND", ""))
    val = something(tryparse(Float64, raw), 1.0)
    return clamp(val, 0.0, 1.0)
end

@inline function kg_quad4_gp_field_pmin_spread_avg_min()
    raw = solver_env_optional_float("JFEM_KG_QUAD4_GP_FIELD_PMIN_SPREAD_AVG_MIN")
    raw === nothing && return Inf
    return max(raw, 0.0)
end

@inline function kg_quad4_gp_field_pmin_spread_avg_alpha()
    return clamp(solver_env_float("JFEM_KG_QUAD4_GP_FIELD_PMIN_SPREAD_AVG_ALPHA", 0.0), 0.0, 1.0)
end

@inline function kg_quad4_gp_field_extrapolate_scale()
    return max(solver_env_float("JFEM_KG_QUAD4_GP_FIELD_EXTRAPOLATE_SCALE", 1.0), 0.0)
end

@inline function kg_quad4_shear_average_operator_enabled()
    return solver_env_bool("JFEM_KG_QUAD4_SHEAR_AVG_OPERATOR", false)
end

@inline function kg_quad4_shear_average_ratio_min()
    return clamp(solver_env_float("JFEM_KG_QUAD4_SHEAR_AVG_RATIO_MIN", 0.9), 0.0, 1.0)
end

@inline function kg_quad4_shear_average_warp_min()
    return max(solver_env_float("JFEM_KG_QUAD4_SHEAR_AVG_WARP_MIN", 0.0), 0.0)
end

@inline function kg_quad4_shear_average_warp_max()
    return max(solver_env_float("JFEM_KG_QUAD4_SHEAR_AVG_WARP_MAX", 1.0e99), 0.0)
end

@inline function kg_quad4_shear_average_aspect_min()
    return max(solver_env_float("JFEM_KG_QUAD4_SHEAR_AVG_ASPECT_MIN", 1.0), 1.0)
end

@inline function kg_quad4_shear_average_aspect_max()
    return max(solver_env_float("JFEM_KG_QUAD4_SHEAR_AVG_ASPECT_MAX", 1.0e99), 1.0)
end

@inline function kg_quad4_shear_average_geometry_mode()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_SHEAR_AVG_GEOM_MODE", "all")))
    return raw in ("any", "or", "either") ? :any : :all
end

@inline function kg_quad4_geometry_gate(
    warp_ratio::Float64,
    aspect_ratio::Float64,
    warp_min::Float64,
    warp_max::Float64,
    aspect_min::Float64,
    aspect_max::Float64,
    mode::Symbol,
)
    upper_ok = warp_ratio <= warp_max && aspect_ratio <= aspect_max
    lower_ok = mode === :any ?
        (warp_ratio >= warp_min || aspect_ratio >= aspect_min) :
        (warp_ratio >= warp_min && aspect_ratio >= aspect_min)
    return upper_ok && lower_ok
end

@inline function kg_quad4_shear_resultant_ratio(N_avg::AbstractVector)
    denom = abs(N_avg[1]) + abs(N_avg[2]) + abs(N_avg[3])
    return denom > 1e-30 ? abs(N_avg[3]) / denom : 0.0
end

@inline function kg_quad4_use_gp_field(
    N_gp::AbstractMatrix,
    N_avg::AbstractVector,
    auto_avg_geom_ok::Bool=false,
    auto_avg_load_ok::Union{Nothing,Bool}=nothing,
)
    mode = kg_quad4_stress_field_mode()
    mode === :gauss && return true
    mode === :average && return false

    auto_avg_load_ok === false && return true

    if kg_quad4_auto_avg_require_geometry() && !auto_avg_geom_ok
        return true
    end
    if auto_avg_load_ok !== true
        if kg_quad4_auto_avg_require_compression()
            min(N_avg[1], N_avg[2]) < 0.0 || return true
            (N_avg[1] + N_avg[2]) < 0.0 || return true
            if kg_quad4_auto_avg_require_biaxial_compression()
                max(N_avg[1], N_avg[2]) < 0.0 || return true
            end
        end
        if kg_quad4_shear_resultant_ratio(N_avg) > kg_quad4_auto_avg_shear_ratio_max()
            return true
        end
    end

    mean_gp_norm = 0.0
    @inbounds for gp in 1:size(N_gp, 1)
        mean_gp_norm += sqrt(N_gp[gp,1]^2 + N_gp[gp,2]^2 + N_gp[gp,3]^2)
    end
    mean_gp_norm /= max(size(N_gp, 1), 1)
    mean_gp_norm <= 1e-12 && return false

    avg_norm = sqrt(N_avg[1]^2 + N_avg[2]^2 + N_avg[3]^2)
    return avg_norm / mean_gp_norm <= kg_quad4_gp_field_avg_ratio_max()
end

@inline function kg_quad4_blend_gp_field!(N_eff::AbstractMatrix, N_gp::AbstractMatrix, N_avg::AbstractVector, alpha::Float64)
    @inbounds for gp in 1:size(N_gp, 1)
        N_eff[gp, 1] = N_avg[1] + alpha * (N_gp[gp, 1] - N_avg[1])
        N_eff[gp, 2] = N_avg[2] + alpha * (N_gp[gp, 2] - N_avg[2])
        N_eff[gp, 3] = N_avg[3] + alpha * (N_gp[gp, 3] - N_avg[3])
    end
    return N_eff
end

@inline function kg_shell_trans_mode()
    # MSC Nastran's CQUAD4 differential stiffness acts on displacement
    # gradients transverse to the local principal membrane-stress directions,
    # not on all three translational components equally. Single-element
    # MATPRN/KDJJ probes confirm this for both flat and warped quads; keep
    # :all as an explicit continuum/geometric-stiffness research option.
    raw = lowercase(strip(get(ENV, "JFEM_KG_SHELL_TRANS_DOF_MODE", "principal_transverse")))
    if raw in ("normal_only", "normal", "w_only")
        return :normal_only
    elseif raw in ("principal_transverse", "stress_transverse", "nastran_flat", "nastran")
        return :principal_transverse
    elseif raw in ("curvature", "curvature_coupled", "shell_curvature")
        return :curvature
    else
        return :all
    end
end

@inline function kg_shell_principal_transverse_flat_only_enabled()
    # Default off: warped CQUAD4 KDJJ probes still match the principal
    # transverse operator far better than the older :all fallback.
    return solver_env_bool("JFEM_KG_SHELL_PRINCIPAL_TRANSVERSE_FLAT_ONLY", false)
end

@inline function kg_shell_principal_transverse_warp_ratio_max()
    return max(solver_env_float("JFEM_KG_SHELL_PRINCIPAL_TRANSVERSE_WARP_RATIO_MAX", 1e-6), 0.0)
end

@inline function q4_pcomp_kg_trans_mode_final(
    requested::Symbol,
    is_pcomp_clt::Bool,
    pcomp_is_isotropic::Bool,
    has_bmb::Bool,
    elem_is_flat_kg::Bool,
    flat_pcomp_plate_like_kg::Bool,
    nonflat_pcomp_normal_only_kg::Bool,
    geom_curvature,
)
    pcomp_normal_only = false
    is_saddle = false
    final = requested
    if is_pcomp_clt && !pcomp_is_isotropic && !has_bmb
        # For PCOMP composites without B-coupling, use normal-only geometric
        # stiffness on flat elements when explicitly requested and on
        # non-saddle curved elements. Saddle surfaces keep in-plane terms.
        is_saddle = !elem_is_flat_kg && geom_curvature !== nothing &&
            q4_curvature_gaussian(geom_curvature) < -1e-10
        pcomp_normal_only = elem_is_flat_kg ? flat_pcomp_plate_like_kg :
                             (nonflat_pcomp_normal_only_kg && !is_saddle)
        if pcomp_normal_only
            final = :normal_only
        end
    end
    return final, pcomp_normal_only, is_saddle
end

@inline function kg_shell_principal_shear_yy_factor()
    return solver_env_float("JFEM_KG_SHELL_PRINCIPAL_SHEAR_YY_FACTOR", 1.0)
end

@inline function kg_shell_principal_shear_xy_factor()
    return solver_env_float("JFEM_KG_SHELL_PRINCIPAL_SHEAR_XY_FACTOR", 1.0)
end

@inline function kg_shell_principal_shear_z_factor()
    return solver_env_float("JFEM_KG_SHELL_PRINCIPAL_SHEAR_Z_FACTOR", 1.0)
end

@inline function kg_shell_principal_shear_ratio_min()
    return clamp(solver_env_float("JFEM_KG_SHELL_PRINCIPAL_SHEAR_RATIO_MIN", 1.0), 0.0, 1.0)
end

@inline function kg_shell_principal_shear_warp_min()
    return max(solver_env_float("JFEM_KG_SHELL_PRINCIPAL_SHEAR_WARP_MIN", 0.0), 0.0)
end

@inline function kg_shell_principal_shear_warp_max()
    return max(solver_env_float("JFEM_KG_SHELL_PRINCIPAL_SHEAR_WARP_MAX", 1.0e99), 0.0)
end

@inline function kg_shell_principal_shear_aspect_min()
    return max(solver_env_float("JFEM_KG_SHELL_PRINCIPAL_SHEAR_ASPECT_MIN", 1.0), 1.0)
end

@inline function kg_shell_principal_shear_aspect_max()
    return max(solver_env_float("JFEM_KG_SHELL_PRINCIPAL_SHEAR_ASPECT_MAX", 1.0e99), 1.0)
end

@inline function kg_shell_principal_shear_geometry_mode()
    raw = lowercase(strip(get(ENV, "JFEM_KG_SHELL_PRINCIPAL_SHEAR_GEOM_MODE", "all")))
    return raw in ("any", "or", "either") ? :any : :all
end

@inline function kg_shell_principal_shear_feature_gate()
    raw = lowercase(strip(get(ENV, "JFEM_KG_SHELL_PRINCIPAL_SHEAR_FEATURE_GATE", "any")))
    if raw in ("positive", "pos", "tension", "+")
        return :positive
    elseif raw in ("negative", "neg", "compression", "-")
        return :negative
    elseif raw in ("positive_or_gp_pmin_spread", "pos_or_gp_pmin_spread", "positive_or_pmin_spread")
        return :positive_or_gp_pmin_spread
    elseif raw in ("positive_or_gp_nxx_spread", "pos_or_gp_nxx_spread", "positive_or_nxx_spread")
        return :positive_or_gp_nxx_spread
    elseif raw in ("positive_or_gp_spread", "pos_or_gp_spread")
        return :positive_or_gp_spread
    else
        return :any
    end
end

@inline function kg_shell_principal_shear_gp_pmin_spread_min()
    return max(solver_env_float("JFEM_KG_SHELL_PRINCIPAL_SHEAR_GP_PMIN_SPREAD_MIN", 0.0), 0.0)
end

@inline function kg_shell_principal_shear_gp_nxx_spread_min()
    return max(solver_env_float("JFEM_KG_SHELL_PRINCIPAL_SHEAR_GP_NXX_SPREAD_MIN", 0.0), 0.0)
end

@inline function kg_shell_principal_shear_gp_spread_factor()
    return clamp(solver_env_float("JFEM_KG_SHELL_PRINCIPAL_SHEAR_GP_SPREAD_FACTOR", 1.0), 0.0, 1.0)
end

@inline function kg_shell_curvature_sign()
    raw = strip(get(ENV, "JFEM_KG_SHELL_CURVATURE_SIGN", "1.0"))
    val = something(tryparse(Float64, raw), 1.0)
    return val < 0.0 ? -1.0 : 1.0
end

@inline function kg_shell_rot_grad_scale()
    return max(solver_env_float("JFEM_KG_SHELL_ROT_GRAD_SCALE", 1.0), 0.0)
end

@inline function kg_shell_rot_grad_auto_iso_scale()
    return max(solver_env_float("JFEM_KG_SHELL_ROT_GRAD_AUTO_ISO_SCALE", 0.0), 0.0)
end

@inline function kg_shell_rot_grad_auto_pcomp_scale()
    return max(solver_env_float("JFEM_KG_SHELL_ROT_GRAD_AUTO_PCOMP_SCALE", 0.0), 0.0)
end

@inline function kg_shell_rot_grad_auto_kappa_l_min()
    return max(solver_env_float("JFEM_KG_SHELL_ROT_GRAD_AUTO_KAPPA_L_MIN", 0.05), 0.0)
end

@inline function kg_shell_rot_grad_auto_cyl_ratio_min()
    return clamp(solver_env_float("JFEM_KG_SHELL_ROT_GRAD_AUTO_CYL_RATIO_MIN", 0.9), 0.0, 1.0)
end

@inline function kg_shell_nxy_scale()
    return solver_env_float("JFEM_KG_SHELL_NXY_SCALE", 1.0)
end

@inline function kg_shell_nxx_scale()
    return solver_env_float("JFEM_KG_SHELL_NXX_SCALE", 1.0)
end

@inline function kg_shell_nyy_scale()
    return solver_env_float("JFEM_KG_SHELL_NYY_SCALE", 1.0)
end

@inline function kg_shell_axial_scale_dominance_min()
    return clamp(solver_env_float("JFEM_KG_SHELL_AXIAL_SCALE_DOMINANCE_MIN", 0.0), 0.0, 1.0)
end

@inline function kg_quad4_membrane_scale_factor()
    return solver_env_float("JFEM_KG_QUAD4_MEMBRANE_SCALE", 1.0)
end

@inline function kg_quad4_feature_membrane_scale_factor()
    return solver_env_float("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE", 1.0)
end

@inline function kg_quad4_feature_membrane_scale_components()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_COMPONENTS", "all")))
    if raw in ("nxx", "xx", "1")
        return :nxx
    elseif raw in ("nyy", "yy", "2")
        return :nyy
    elseif raw in ("nxy", "xy", "shear", "3")
        return :nxy
    elseif raw in ("nxxnyy", "normal", "normals", "axial", "12")
        return :nxxnyy
    else
        return :all
    end
end

@inline function kg_quad4_feature_membrane_scale_pcomp_only()
    return solver_env_bool("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_PCOMP_ONLY", true)
end

function kg_quad4_feature_membrane_scale_pid_list()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_PID_LIST", ""))
    pids = Int[]
    isempty(raw) && return pids
    for part in split(raw, r"[,/\s]+")
        item = strip(part)
        isempty(item) && continue
        pid = tryparse(Int, item)
        pid === nothing || push!(pids, pid)
    end
    sort!(pids)
    unique!(pids)
    return pids
end

function q4_static_component_pid_list()
    raw = strip(get(ENV, "JFEM_Q4_STATIC_COMPONENT_PID_LIST", ""))
    pids = Int[]
    isempty(raw) && return pids
    for part in split(raw, r"[,/\s]+")
        item = strip(part)
        isempty(item) && continue
        pid = tryparse(Int, item)
        pid === nothing || push!(pids, pid)
    end
    sort!(pids)
    unique!(pids)
    return pids
end

function q4_static_component_eid_list()
    raw = strip(get(ENV, "JFEM_Q4_STATIC_COMPONENT_EID_LIST", ""))
    eids = Int[]
    isempty(raw) && return eids
    for part in split(raw, r"[,/\s]+")
        item = strip(part)
        isempty(item) && continue
        eid = tryparse(Int, item)
        eid === nothing || push!(eids, eid)
    end
    sort!(eids)
    unique!(eids)
    return eids
end

function q4_static_component_neighbor_pid_prefixes()
    raw = strip(get(ENV, "JFEM_Q4_STATIC_COMPONENT_REQUIRE_NEIGHBOR_PID_PREFIX", ""))
    prefixes = String[]
    isempty(raw) && return prefixes
    for part in split(raw, r"[,/\s]+")
        item = strip(part)
        isempty(item) && continue
        push!(prefixes, item)
    end
    unique!(prefixes)
    return prefixes
end

@inline function q4_static_component_v2_min()
    return max(solver_env_float("JFEM_Q4_STATIC_COMPONENT_V2_MIN", 0.0), 0.0)
end

@inline function q4_static_component_v2_max()
    return max(solver_env_float("JFEM_Q4_STATIC_COMPONENT_V2_MAX", 0.0), 0.0)
end

@inline function q4_static_component_thickness_min()
    return max(solver_env_float("JFEM_Q4_STATIC_COMPONENT_THICKNESS_MIN", 0.0), 0.0)
end

@inline function q4_static_component_thickness_max()
    return max(solver_env_float("JFEM_Q4_STATIC_COMPONENT_THICKNESS_MAX", 0.0), 0.0)
end

@inline function q4_static_component_pcomp_shear_ratio_min()
    return max(solver_env_float("JFEM_Q4_STATIC_COMPONENT_PCOMP_SHEAR_RATIO_MIN", 0.0), 0.0)
end

@inline function q4_static_component_pcomp_shear_ratio_max()
    return max(solver_env_float("JFEM_Q4_STATIC_COMPONENT_PCOMP_SHEAR_RATIO_MAX", 0.0), 0.0)
end

@inline function q4_static_component_pcomp_d16_ratio_min()
    return max(solver_env_float("JFEM_Q4_STATIC_COMPONENT_PCOMP_D16_RATIO_MIN", 0.0), 0.0)
end

@inline function q4_static_component_pcomp_d16_ratio_max()
    return max(solver_env_float("JFEM_Q4_STATIC_COMPONENT_PCOMP_D16_RATIO_MAX", 0.0), 0.0)
end

@inline function q4_static_component_pcomp_b_ratio_min()
    return max(solver_env_float("JFEM_Q4_STATIC_COMPONENT_PCOMP_B_RATIO_MIN", 0.0), 0.0)
end

@inline function q4_static_component_pcomp_b_ratio_max()
    return max(solver_env_float("JFEM_Q4_STATIC_COMPONENT_PCOMP_B_RATIO_MAX", 0.0), 0.0)
end

@inline function q4_static_component_range_ok(value::Float64, value_min::Float64, value_max::Float64)
    value_min <= 0.0 && value_max <= 0.0 && return true
    return (value_min <= 0.0 || value >= value_min) &&
           (value_max <= 0.0 || value <= value_max)
end

@inline function q4_static_component_pcomp_range_ok(is_pcomp::Bool, value::Float64,
                                                    value_min::Float64, value_max::Float64)
    value_min <= 0.0 && value_max <= 0.0 && return true
    is_pcomp || return false
    return q4_static_component_range_ok(value, value_min, value_max)
end

function q4_static_component_model_eigrl_v2(model)
    cc = get(model, "CASE_CONTROL", nothing)
    cc isa AbstractDict || return 0.0
    subcases = get(cc, "SUBCASES", nothing)
    subcases isa AbstractDict || return 0.0
    eigrls = get(model, "EIGRLs", nothing)
    eigrls isa AbstractDict || return 0.0
    max_v2 = 0.0
    for sub in values(subcases)
        sub isa AbstractDict || continue
        method_id = get(sub, "METHOD", nothing)
        method_id === nothing && continue
        method_sid = tryparse(Int, string(method_id))
        method_sid === nothing && continue
        eigrl = get(eigrls, string(method_sid), nothing)
        eigrl isa AbstractDict || continue
        max_v2 = max(max_v2, Float64(get(eigrl, "V2", 0.0)))
    end
    return max_v2
end

@inline function q4_static_component_v2_ok(eigrl_v2::Float64,
                                           v2_min::Float64,
                                           v2_max::Float64)
    v2_min <= 0.0 && v2_max <= 0.0 && return true
    eigrl_v2 > 0.0 || return false
    return (v2_min <= 0.0 || eigrl_v2 >= v2_min) &&
           (v2_max <= 0.0 || eigrl_v2 <= v2_max)
end

@inline function q4_pid_matches_any_prefix(pid::AbstractString, prefixes)
    for prefix in prefixes
        startswith(pid, prefix) && return true
    end
    return false
end

@inline function kg_quad4_feature_membrane_scale_aspect_min()
    return max(solver_env_float("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_ASPECT_MIN", 1.0), 1.0)
end

@inline function kg_quad4_feature_membrane_scale_aspect_max()
    return max(solver_env_float("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_ASPECT_MAX", 1.0e99), 1.0)
end

@inline function kg_quad4_feature_membrane_scale_warp_min()
    return max(solver_env_float("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_WARP_MIN", 0.0), 0.0)
end

@inline function kg_quad4_feature_membrane_scale_warp_max()
    return max(solver_env_float("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_WARP_MAX", 1.0e99), 0.0)
end

@inline function kg_quad4_feature_membrane_scale_kappa_l_min()
    return max(solver_env_float("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_KAPPA_L_MIN", 0.0), 0.0)
end

@inline function kg_quad4_feature_membrane_scale_kappa_l_max()
    return max(solver_env_float("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_KAPPA_L_MAX", 1.0e99), 0.0)
end

@inline function kg_quad4_feature_membrane_scale_geometry_mode()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_GEOM_MODE", "all")))
    return raw in ("any", "or", "either") ? :any : :all
end

@inline function kg_quad4_feature_membrane_scale_sign_gate()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_NXX_SIGN", "any")))
    if raw in ("positive", "pos", "tension", "+")
        return :positive
    elseif raw in ("negative", "neg", "compression", "-")
        return :negative
    elseif raw in ("positive_or_gp_pmin_spread", "pos_or_gp_pmin_spread", "positive_or_pmin_spread")
        return :positive_or_gp_pmin_spread
    elseif raw in ("positive_or_gp_nxx_spread", "pos_or_gp_nxx_spread", "positive_or_nxx_spread")
        return :positive_or_gp_nxx_spread
    elseif raw in ("positive_or_gp_spread", "pos_or_gp_spread")
        return :positive_or_gp_spread
    else
        return :any
    end
end

@inline function kg_quad4_feature_membrane_scale_nxy_sign_gate()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_NXY_SIGN", "any")))
    if raw in ("positive", "pos", "+")
        return :positive
    elseif raw in ("negative", "neg", "-")
        return :negative
    else
        return :any
    end
end

@inline function kg_quad4_feature_membrane_scale_nxy_stat()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_NXY_STAT", "mean")))
    if raw in ("min", "minimum", "gp_min", "gp-min")
        return :min
    elseif raw in ("max", "maximum", "gp_max", "gp-max")
        return :max
    else
        return :mean
    end
end

@inline function kg_quad4_feature_membrane_scale_abs_nxy_min()
    return max(solver_env_float("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_ABS_NXY_MIN", 0.0), 0.0)
end

@inline function kg_quad4_feature_membrane_scale_nxy_mode()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_NXY_MODE", "gate")))
    return raw in ("extra_component", "extra-component", "extra", "component_extra") ?
        :extra_component : :gate
end

@inline function kg_quad4_feature_membrane_scale_gp_pmin_spread_min()
    return max(solver_env_float("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_GP_PMIN_SPREAD_MIN", 0.0), 0.0)
end

@inline function kg_quad4_feature_membrane_scale_gp_nxx_spread_min()
    return max(solver_env_float("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_GP_NXX_SPREAD_MIN", 0.0), 0.0)
end

@inline function kg_quad4_feature_membrane_scale_gp_spread_factor()
    return clamp(solver_env_float("JFEM_KG_QUAD4_FEATURE_MEMBRANE_SCALE_GP_SPREAD_FACTOR", 1.0), 0.0, 1.0)
end

function kg_quad4_pid_membrane_scale_map()
    raw = strip(get(ENV, "JFEM_KG_QUAD4_PID_MEMBRANE_SCALE", ""))
    scales = Dict{Int,Float64}()
    isempty(raw) && return scales
    for item in split(raw, ",")
        part = strip(item)
        isempty(part) && continue
        kv = split(part, ":"; limit=2)
        length(kv) == 2 || continue
        pid = tryparse(Int, strip(kv[1]))
        scale = tryparse(Float64, strip(kv[2]))
        (pid === nothing || scale === nothing) && continue
        scales[pid] = max(scale, 0.0)
    end
    return scales
end

@inline function kg_quad4_pid_membrane_scale_components()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_PID_MEMBRANE_SCALE_COMPONENTS", "all")))
    if raw in ("nxx", "xx", "1")
        return :nxx
    elseif raw in ("nyy", "yy", "2")
        return :nyy
    elseif raw in ("nxy", "xy", "shear", "3")
        return :nxy
    elseif raw in ("nxxnyy", "normal", "normals", "axial", "12")
        return :nxxnyy
    else
        return :all
    end
end

@inline function kg_quad4_pid_membrane_scale_nxx_sign_gate()
    raw = lowercase(strip(get(ENV, "JFEM_KG_QUAD4_PID_MEMBRANE_SCALE_NXX_SIGN", "any")))
    if raw in ("positive", "pos", "tension", "+")
        return :positive
    elseif raw in ("negative", "neg", "compression", "-")
        return :negative
    elseif raw in ("positive_or_gp_pmin_spread", "pos_or_gp_pmin_spread", "positive_or_pmin_spread")
        return :positive_or_gp_pmin_spread
    elseif raw in ("positive_or_gp_nxx_spread", "pos_or_gp_nxx_spread", "positive_or_nxx_spread")
        return :positive_or_gp_nxx_spread
    elseif raw in ("positive_or_gp_spread", "pos_or_gp_spread")
        return :positive_or_gp_spread
    else
        return :any
    end
end

@inline function kg_quad4_pid_membrane_scale_gp_pmin_spread_min()
    return max(solver_env_float("JFEM_KG_QUAD4_PID_MEMBRANE_SCALE_GP_PMIN_SPREAD_MIN", 0.0), 0.0)
end

@inline function kg_quad4_pid_membrane_scale_gp_nxx_spread_min()
    return max(solver_env_float("JFEM_KG_QUAD4_PID_MEMBRANE_SCALE_GP_NXX_SPREAD_MIN", 0.0), 0.0)
end

@inline function kg_quad4_pid_membrane_scale_gp_spread_factor()
    return clamp(solver_env_float("JFEM_KG_QUAD4_PID_MEMBRANE_SCALE_GP_SPREAD_FACTOR", 1.0), 0.0, 1.0)
end

@inline function kg_quad4_pid_membrane_scale_v2_min()
    return max(solver_env_float("JFEM_KG_QUAD4_PID_MEMBRANE_SCALE_V2_MIN", 0.0), 0.0)
end

@inline function kg_quad4_pid_membrane_scale_v2_max()
    return max(solver_env_float("JFEM_KG_QUAD4_PID_MEMBRANE_SCALE_V2_MAX", 0.0), 0.0)
end

function kg_quad4_buckling_eigrl_v2(model, buckling_subcase)
    buckling_subcase === nothing && return 0.0
    sid = tryparse(Int, string(buckling_subcase))
    sid === nothing && return 0.0
    cc = get(model, "CASE_CONTROL", nothing)
    cc isa AbstractDict || return 0.0
    subcases = get(cc, "SUBCASES", nothing)
    subcases isa AbstractDict || return 0.0
    sub = get(subcases, sid, nothing)
    sub isa AbstractDict || return 0.0
    method_id = get(sub, "METHOD", nothing)
    method_id === nothing && return 0.0
    method_sid = tryparse(Int, string(method_id))
    method_sid === nothing && return 0.0
    eigrls = get(model, "EIGRLs", nothing)
    eigrls isa AbstractDict || return 0.0
    eigrl = get(eigrls, string(method_sid), nothing)
    eigrl isa AbstractDict || return 0.0
    return Float64(get(eigrl, "V2", 0.0))
end

@inline function kg_quad4_pid_membrane_scale_v2_ok(eigrl_v2::Float64,
                                                   v2_min::Float64,
                                                   v2_max::Float64)
    v2_min <= 0.0 && v2_max <= 0.0 && return true
    eigrl_v2 > 0.0 || return false
    return (v2_min <= 0.0 || eigrl_v2 >= v2_min) &&
           (v2_max <= 0.0 || eigrl_v2 <= v2_max)
end

@inline function kg_quad4_sigma_mean_nxx(sigma_mem_input)
    if sigma_mem_input isa AbstractMatrix
        acc = 0.0
        @inbounds for gp in 1:size(sigma_mem_input, 1)
            acc += sigma_mem_input[gp, 1]
        end
        return acc / max(size(sigma_mem_input, 1), 1)
    else
        return sigma_mem_input[1]
    end
end

@inline function kg_quad4_sigma_mean_nxy(sigma_mem_input)
    if sigma_mem_input isa AbstractMatrix
        acc = 0.0
        @inbounds for gp in 1:size(sigma_mem_input, 1)
            acc += sigma_mem_input[gp, 3]
        end
        return acc / max(size(sigma_mem_input, 1), 1)
    else
        return sigma_mem_input[3]
    end
end

@inline function kg_quad4_sigma_nxy_stat(sigma_mem_input, stat::Symbol)
    if !(sigma_mem_input isa AbstractMatrix)
        return sigma_mem_input[3]
    elseif stat === :min
        val = Inf
        @inbounds for gp in 1:size(sigma_mem_input, 1)
            val = min(val, sigma_mem_input[gp, 3])
        end
        return isfinite(val) ? val : 0.0
    elseif stat === :max
        val = -Inf
        @inbounds for gp in 1:size(sigma_mem_input, 1)
            val = max(val, sigma_mem_input[gp, 3])
        end
        return isfinite(val) ? val : 0.0
    else
        return kg_quad4_sigma_mean_nxy(sigma_mem_input)
    end
end

@inline function kg_quad4_component_sign_ok(sign_gate::Symbol, value::Float64)
    return sign_gate === :any ||
           (sign_gate === :positive && value > 0.0) ||
           (sign_gate === :negative && value < 0.0)
end

@inline function kg_quad4_feature_curvature_gate(geom_curvature,
                                                 lc::AbstractMatrix,
                                                 kappa_l_min::Float64,
                                                 kappa_l_max::Float64)
    if kappa_l_min <= 0.0 && kappa_l_max >= 1.0e98
        return true
    end
    kappa_l = 0.0
    if geom_curvature !== nothing
        k1, _ = q4_curvature_principal_abs(geom_curvature)
        kappa_l = k1 * q4_curvature_characteristic_length(lc)
    end
    return kappa_l >= kappa_l_min && kappa_l <= kappa_l_max
end

@inline function kg_quad4_apply_feature_component_scale!(sigma_mem::AbstractMatrix,
                                                         scale::Float64,
                                                         components::Symbol)
    if components === :nxx
        @inbounds for gp in 1:size(sigma_mem, 1); sigma_mem[gp, 1] *= scale; end
    elseif components === :nyy
        @inbounds for gp in 1:size(sigma_mem, 1); sigma_mem[gp, 2] *= scale; end
    elseif components === :nxy
        @inbounds for gp in 1:size(sigma_mem, 1); sigma_mem[gp, 3] *= scale; end
    elseif components === :nxxnyy
        @inbounds for gp in 1:size(sigma_mem, 1)
            sigma_mem[gp, 1] *= scale
            sigma_mem[gp, 2] *= scale
        end
    else
        sigma_mem .*= scale
    end
    return sigma_mem
end

@inline function kg_quad4_apply_feature_component_scale!(sigma_mem::AbstractVector,
                                                         scale::Float64,
                                                         components::Symbol)
    if components === :nxx
        sigma_mem[1] *= scale
    elseif components === :nyy
        sigma_mem[2] *= scale
    elseif components === :nxy
        sigma_mem[3] *= scale
    elseif components === :nxxnyy
        sigma_mem[1] *= scale
        sigma_mem[2] *= scale
    else
        sigma_mem .*= scale
    end
    return sigma_mem
end

@inline function kg_quad4_sigma_principal_min_resultant(sxx::Float64, syy::Float64, sxy::Float64, h::Float64)
    nxx = sxx * h
    nyy = syy * h
    nxy = sxy * h
    mean_n = 0.5 * (nxx + nyy)
    half_d = 0.5 * (nxx - nyy)
    return mean_n - sqrt(half_d * half_d + nxy * nxy)
end

@inline function kg_quad4_sigma_gp_nxx_spread_resultant(sigma_mem_input, h::Float64)
    if sigma_mem_input isa AbstractMatrix
        lo = Inf
        hi = -Inf
        @inbounds for gp in 1:size(sigma_mem_input, 1)
            nxx = sigma_mem_input[gp, 1] * h
            lo = min(lo, nxx)
            hi = max(hi, nxx)
        end
        return isfinite(lo) && isfinite(hi) ? hi - lo : 0.0
    end
    return 0.0
end

@inline function kg_quad4_sigma_gp_pmin_spread_resultant(sigma_mem_input, h::Float64)
    if sigma_mem_input isa AbstractMatrix
        lo = Inf
        hi = -Inf
        @inbounds for gp in 1:size(sigma_mem_input, 1)
            pmin = kg_quad4_sigma_principal_min_resultant(
                sigma_mem_input[gp, 1],
                sigma_mem_input[gp, 2],
                sigma_mem_input[gp, 3],
                h,
            )
            lo = min(lo, pmin)
            hi = max(hi, pmin)
        end
        return isfinite(lo) && isfinite(hi) ? hi - lo : 0.0
    end
    return 0.0
end

@inline function kg_quad4_pid_membrane_scale_sign_ok(sign_gate::Symbol,
                                                     sigma_mem_input,
                                                     h::Float64,
                                                     gp_pmin_spread_min::Float64,
                                                     gp_nxx_spread_min::Float64)
    nxx = kg_quad4_sigma_mean_nxx(sigma_mem_input)
    return sign_gate === :any ||
           (sign_gate === :positive && nxx > 0.0) ||
           (sign_gate === :negative && nxx < 0.0) ||
           (sign_gate === :positive_or_gp_pmin_spread &&
            (nxx > 0.0 || kg_quad4_sigma_gp_pmin_spread_resultant(sigma_mem_input, h) >= gp_pmin_spread_min)) ||
           (sign_gate === :positive_or_gp_nxx_spread &&
            (nxx > 0.0 || kg_quad4_sigma_gp_nxx_spread_resultant(sigma_mem_input, h) >= gp_nxx_spread_min)) ||
           (sign_gate === :positive_or_gp_spread &&
            (nxx > 0.0 ||
             kg_quad4_sigma_gp_pmin_spread_resultant(sigma_mem_input, h) >= gp_pmin_spread_min ||
             kg_quad4_sigma_gp_nxx_spread_resultant(sigma_mem_input, h) >= gp_nxx_spread_min))
end

@inline function kg_quad4_pid_membrane_effective_scale(pid_scale::Float64,
                                                       sign_gate::Symbol,
                                                       sigma_mem_input,
                                                       h::Float64,
                                                       gp_pmin_spread_min::Float64,
                                                       gp_nxx_spread_min::Float64,
                                                       gp_spread_factor::Float64)
    pid_scale == 1.0 && return 1.0
    nxx = kg_quad4_sigma_mean_nxx(sigma_mem_input)
    if sign_gate === :any
        return pid_scale
    elseif sign_gate === :positive
        return nxx > 0.0 ? pid_scale : 1.0
    elseif sign_gate === :negative
        return nxx < 0.0 ? pid_scale : 1.0
    elseif sign_gate === :positive_or_gp_pmin_spread
        nxx > 0.0 && return pid_scale
        spread_ok = kg_quad4_sigma_gp_pmin_spread_resultant(sigma_mem_input, h) >= gp_pmin_spread_min
        return spread_ok ? 1.0 + (pid_scale - 1.0) * gp_spread_factor : 1.0
    elseif sign_gate === :positive_or_gp_nxx_spread
        nxx > 0.0 && return pid_scale
        spread_ok = kg_quad4_sigma_gp_nxx_spread_resultant(sigma_mem_input, h) >= gp_nxx_spread_min
        return spread_ok ? 1.0 + (pid_scale - 1.0) * gp_spread_factor : 1.0
    elseif sign_gate === :positive_or_gp_spread
        nxx > 0.0 && return pid_scale
        pmin_ok = kg_quad4_sigma_gp_pmin_spread_resultant(sigma_mem_input, h) >= gp_pmin_spread_min
        nxx_ok = kg_quad4_sigma_gp_nxx_spread_resultant(sigma_mem_input, h) >= gp_nxx_spread_min
        return (pmin_ok || nxx_ok) ? 1.0 + (pid_scale - 1.0) * gp_spread_factor : 1.0
    end
    return 1.0
end

@inline function kg_shell_pcomp_nxy_scale()
    return solver_env_float("JFEM_KG_SHELL_PCOMP_NXY_SCALE", 1.0)
end

@inline function kg_shell_pcomp_nxy_aspect_scale_enabled()
    return solver_env_bool("JFEM_KG_SHELL_PCOMP_NXY_ASPECT_SCALE", false)
end

@inline function kg_shell_pcomp_nxy_aspect_low()
    return solver_env_float("JFEM_KG_SHELL_PCOMP_NXY_ASPECT_LOW", 0.70)
end

@inline function kg_shell_pcomp_nxy_aspect_high()
    return solver_env_float("JFEM_KG_SHELL_PCOMP_NXY_ASPECT_HIGH", 1.70)
end

@inline function kg_shell_pcomp_nxy_aspect_mid()
    return solver_env_float(
        "JFEM_KG_SHELL_PCOMP_NXY_ASPECT_MID",
        kg_shell_pcomp_nxy_aspect_high(),
    )
end

@inline function kg_shell_pcomp_nxy_aspect_min()
    return max(solver_env_float("JFEM_KG_SHELL_PCOMP_NXY_ASPECT_MIN", 2.5), 1.0)
end

@inline function kg_shell_pcomp_nxy_aspect_peak()
    return max(solver_env_float("JFEM_KG_SHELL_PCOMP_NXY_ASPECT_PEAK", 4.0), 1.0)
end

@inline function kg_shell_pcomp_nxy_aspect_max()
    return max(solver_env_float("JFEM_KG_SHELL_PCOMP_NXY_ASPECT_MAX", 4.0), 1.0)
end

@inline function kg_shell_pcomp_nxy_aspect_mode()
    raw = lowercase(strip(get(ENV, "JFEM_KG_SHELL_PCOMP_NXY_ASPECT_MODE", "ramp")))
    return raw in ("band", "window", "tent") ? :band : :ramp
end

@inline function kg_shell_pcomp_nxy_compression_only()
    return solver_env_bool("JFEM_KG_SHELL_PCOMP_NXY_COMPRESSION_ONLY", false)
end

@inline function kg_shell_pcomp_nxy_shear_dom_relax()
    return clamp(solver_env_float("JFEM_KG_SHELL_PCOMP_NXY_SHEAR_DOM_RELAX", 0.0), 0.0, 1.0)
end

@inline function kg_shell_pcomp_nxy_shear_dom_ratio_min()
    return max(solver_env_float("JFEM_KG_SHELL_PCOMP_NXY_SHEAR_DOM_RATIO_MIN", 1.5), 0.0)
end

@inline function kg_shell_pcomp_nxy_shear_dom_ratio_full()
    return max(solver_env_float("JFEM_KG_SHELL_PCOMP_NXY_SHEAR_DOM_RATIO_FULL", 4.0), 1e-12)
end

@inline function kg_shell_pcomp_nxy_shear_dom_aspect_min()
    return max(solver_env_float("JFEM_KG_SHELL_PCOMP_NXY_SHEAR_DOM_ASPECT_MIN", 1.0), 1.0)
end

@inline function kg_shell_pcomp_nxy_shear_dom_aspect_max()
    return max(
        solver_env_float("JFEM_KG_SHELL_PCOMP_NXY_SHEAR_DOM_ASPECT_MAX", 1e30),
        kg_shell_pcomp_nxy_shear_dom_aspect_min(),
    )
end

@inline function kg_shell_pcomp_nxy_shear_dom_compression_only()
    return solver_env_bool("JFEM_KG_SHELL_PCOMP_NXY_SHEAR_DOM_COMPRESSION_ONLY", false)
end

@inline function kg_shell_pcomp_nxy_aspect_scale(aspect::Float64,
                                                low_scale::Float64,
                                                high_scale::Float64,
                                                aspect_min::Float64,
                                                aspect_max::Float64)
    if aspect_max <= aspect_min
        return aspect >= aspect_min ? high_scale : low_scale
    end
    t = clamp((aspect - aspect_min) / (aspect_max - aspect_min), 0.0, 1.0)
    return (1.0 - t) * low_scale + t * high_scale
end

@inline function kg_shell_pcomp_nxy_aspect_scale(aspect::Float64,
                                                mode::Symbol,
                                                low_scale::Float64,
                                                mid_scale::Float64,
                                                high_scale::Float64,
                                                aspect_min::Float64,
                                                aspect_peak::Float64,
                                                aspect_max::Float64)
    if mode === :band
        if aspect <= aspect_min
            return low_scale
        elseif aspect >= aspect_max
            return high_scale
        elseif aspect <= aspect_peak
            denom = max(aspect_peak - aspect_min, 1e-12)
            t = clamp((aspect - aspect_min) / denom, 0.0, 1.0)
            return (1.0 - t) * low_scale + t * mid_scale
        else
            denom = max(aspect_max - aspect_peak, 1e-12)
            t = clamp((aspect - aspect_peak) / denom, 0.0, 1.0)
            return (1.0 - t) * mid_scale + t * high_scale
        end
    end
    return kg_shell_pcomp_nxy_aspect_scale(
        aspect,
        low_scale,
        high_scale,
        aspect_min,
        aspect_max,
    )
end

@inline function kg_shell_should_scale_axial_component(scomp::Float64,
                                                       sxx::Float64,
                                                       syy::Float64,
                                                       sxy::Float64,
                                                       dominance_min::Float64)
    dominance_min <= 0.0 && return true
    denom = abs(sxx) + abs(syy) + abs(sxy)
    denom > 1e-30 || return false
    return abs(scomp) / denom >= dominance_min
end

@inline function kg_shell_apply_axial_component_scale!(sigma_mem::AbstractMatrix,
                                                       nxx_scale::Float64,
                                                       nyy_scale::Float64,
                                                       dominance_min::Float64)
    (nxx_scale == 1.0 && nyy_scale == 1.0) && return sigma_mem
    @inbounds for gp in 1:size(sigma_mem, 1)
        sxx = sigma_mem[gp, 1]
        syy = sigma_mem[gp, 2]
        sxy = sigma_mem[gp, 3]
        if nxx_scale != 1.0 &&
           kg_shell_should_scale_axial_component(sxx, sxx, syy, sxy, dominance_min)
            sigma_mem[gp, 1] *= nxx_scale
        end
        if nyy_scale != 1.0 &&
           kg_shell_should_scale_axial_component(syy, sxx, syy, sxy, dominance_min)
            sigma_mem[gp, 2] *= nyy_scale
        end
    end
    return sigma_mem
end

@inline function kg_shell_apply_axial_component_scale!(sigma_mem::AbstractMatrix,
                                                       nxx_scale::Float64,
                                                       nyy_scale::Float64,
                                                       dominance_min::Float64,
                                                       gate_mem::AbstractVector)
    (nxx_scale == 1.0 && nyy_scale == 1.0) && return sigma_mem
    gate_xx = kg_shell_should_scale_axial_component(
        gate_mem[1], gate_mem[1], gate_mem[2], gate_mem[3], dominance_min
    )
    gate_yy = kg_shell_should_scale_axial_component(
        gate_mem[2], gate_mem[1], gate_mem[2], gate_mem[3], dominance_min
    )
    @inbounds for gp in 1:size(sigma_mem, 1)
        if nxx_scale != 1.0 && gate_xx
            sigma_mem[gp, 1] *= nxx_scale
        end
        if nyy_scale != 1.0 && gate_yy
            sigma_mem[gp, 2] *= nyy_scale
        end
    end
    return sigma_mem
end

@inline function kg_shell_apply_axial_component_scale!(sigma_mem::AbstractVector,
                                                       nxx_scale::Float64,
                                                       nyy_scale::Float64,
                                                       dominance_min::Float64)
    (nxx_scale == 1.0 && nyy_scale == 1.0) && return sigma_mem
    sxx = sigma_mem[1]
    syy = sigma_mem[2]
    sxy = sigma_mem[3]
    if nxx_scale != 1.0 &&
       kg_shell_should_scale_axial_component(sxx, sxx, syy, sxy, dominance_min)
        sigma_mem[1] *= nxx_scale
    end
    if nyy_scale != 1.0 &&
       kg_shell_should_scale_axial_component(syy, sxx, syy, sxy, dominance_min)
        sigma_mem[2] *= nyy_scale
    end
    return sigma_mem
end

@inline function kg_shell_apply_axial_component_scale!(sigma_mem::AbstractVector,
                                                       nxx_scale::Float64,
                                                       nyy_scale::Float64,
                                                       dominance_min::Float64,
                                                       gate_mem::AbstractVector)
    (nxx_scale == 1.0 && nyy_scale == 1.0) && return sigma_mem
    if nxx_scale != 1.0 &&
       kg_shell_should_scale_axial_component(
           gate_mem[1], gate_mem[1], gate_mem[2], gate_mem[3], dominance_min
       )
        sigma_mem[1] *= nxx_scale
    end
    if nyy_scale != 1.0 &&
       kg_shell_should_scale_axial_component(
           gate_mem[2], gate_mem[1], gate_mem[2], gate_mem[3], dominance_min
       )
        sigma_mem[2] *= nyy_scale
    end
    return sigma_mem
end

@inline function q4_macneal_bending_aspect_scale_enabled()
    return solver_env_bool("JFEM_Q4_MACNEAL_BENDING_ASPECT_SCALE", false)
end

@inline function q4_macneal_bending_aspect_low_scale()
    return solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT_LOW_SCALE", 1.0)
end

@inline function q4_macneal_bending_aspect_high_scale()
    return solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT_HIGH_SCALE", 1.0)
end

@inline function q4_macneal_bending_aspect_mid_scale()
    return solver_env_float(
        "JFEM_Q4_MACNEAL_BENDING_ASPECT_MID_SCALE",
        q4_macneal_bending_aspect_high_scale(),
    )
end

@inline function q4_macneal_bending_aspect_min()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT_MIN", 2.5), 1.0)
end

@inline function q4_macneal_bending_aspect_peak()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT_PEAK", 4.0), 1.0)
end

@inline function q4_macneal_bending_aspect_max()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT_MAX", 4.0), 1.0)
end

@inline function q4_macneal_bending_aspect_warp_min()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT_WARP_MIN", 0.0), 0.0)
end

@inline function q4_macneal_bending_aspect_warp_max()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT_WARP_MAX", 1.0e99), 0.0)
end

@inline function q4_macneal_bending_aspect_kappa_l_min()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT_KAPPA_L_MIN", 0.0), 0.0)
end

@inline function q4_macneal_bending_aspect_kappa_l_max()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT_KAPPA_L_MAX", 1.0e99), 0.0)
end

@inline function q4_macneal_bending_aspect_skew_min()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT_SKEW_MIN", 0.0), 0.0)
end

@inline function q4_macneal_bending_aspect_skew_max()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT_SKEW_MAX", 180.0), 0.0)
end

@inline function q4_macneal_bending_aspect2_scale_enabled()
    return solver_env_bool("JFEM_Q4_MACNEAL_BENDING_ASPECT2_SCALE", false)
end

@inline function q4_macneal_bending_aspect2_low_scale()
    return solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT2_LOW_SCALE", 1.0)
end

@inline function q4_macneal_bending_aspect2_high_scale()
    return solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT2_HIGH_SCALE", 1.0)
end

@inline function q4_macneal_bending_aspect2_mid_scale()
    return solver_env_float(
        "JFEM_Q4_MACNEAL_BENDING_ASPECT2_MID_SCALE",
        q4_macneal_bending_aspect2_high_scale(),
    )
end

@inline function q4_macneal_bending_aspect2_min()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT2_MIN", 2.5), 1.0)
end

@inline function q4_macneal_bending_aspect2_peak()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT2_PEAK", 4.0), 1.0)
end

@inline function q4_macneal_bending_aspect2_max()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT2_MAX", 4.0), 1.0)
end

@inline function q4_macneal_bending_aspect2_warp_min()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT2_WARP_MIN", 0.0), 0.0)
end

@inline function q4_macneal_bending_aspect2_warp_max()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT2_WARP_MAX", 1.0e99), 0.0)
end

@inline function q4_macneal_bending_aspect2_kappa_l_min()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT2_KAPPA_L_MIN", 0.0), 0.0)
end

@inline function q4_macneal_bending_aspect2_kappa_l_max()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT2_KAPPA_L_MAX", 1.0e99), 0.0)
end

@inline function q4_macneal_bending_aspect2_skew_min()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT2_SKEW_MIN", 0.0), 0.0)
end

@inline function q4_macneal_bending_aspect2_skew_max()
    return max(solver_env_float("JFEM_Q4_MACNEAL_BENDING_ASPECT2_SKEW_MAX", 180.0), 0.0)
end

@inline function q4_mitc4_3d_aspect_skew_min()
    return max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_SKEW_MIN", 0.0), 0.0)
end

@inline function q4_mitc4_3d_aspect_skew_max()
    return max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_SKEW_MAX", 180.0), 0.0)
end

@inline function q4_mitc4_3d_aspect_skew_aspect_min()
    return max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_SKEW_ASPECT_MIN", 0.0), 0.0)
end

@inline function q4_macneal_bending_aspect_mode()
    raw = lowercase(strip(get(ENV, "JFEM_Q4_MACNEAL_BENDING_ASPECT_MODE", "ramp")))
    return raw in ("band", "window", "tent") ? :band : :ramp
end

@inline function q4_macneal_bending_aspect2_mode()
    raw = lowercase(strip(get(ENV, "JFEM_Q4_MACNEAL_BENDING_ASPECT2_MODE", "ramp")))
    return raw in ("band", "window", "tent") ? :band : :ramp
end

@inline function q4_macneal_bending_aspect_scale(
    aspect::Float64,
    mode::Symbol,
    low_scale::Float64,
    mid_scale::Float64,
    high_scale::Float64,
    aspect_min::Float64,
    aspect_peak::Float64,
    aspect_max::Float64,
)
    if mode === :band
        if aspect <= aspect_min
            return low_scale
        elseif aspect >= aspect_max
            return high_scale
        elseif aspect <= aspect_peak
            denom = max(aspect_peak - aspect_min, 1e-12)
            t = clamp((aspect - aspect_min) / denom, 0.0, 1.0)
            return (1.0 - t) * low_scale + t * mid_scale
        else
            denom = max(aspect_max - aspect_peak, 1e-12)
            t = clamp((aspect - aspect_peak) / denom, 0.0, 1.0)
            return (1.0 - t) * mid_scale + t * high_scale
        end
    end
    return kg_shell_pcomp_nxy_aspect_scale(
        aspect,
        low_scale,
        high_scale,
        aspect_min,
        aspect_max,
    )
end

@inline function q4_macneal_bending_aspect_geom_ok(
    warp_ratio::Float64,
    kappa_l::Float64,
    edge_skew::Float64,
    warp_min::Float64,
    warp_max::Float64,
    kappa_l_min::Float64,
    kappa_l_max::Float64,
    skew_min::Float64,
    skew_max::Float64,
)
    return warp_ratio >= warp_min &&
           warp_ratio <= warp_max &&
           kappa_l >= kappa_l_min &&
           kappa_l <= kappa_l_max &&
           edge_skew >= skew_min &&
           edge_skew <= skew_max
end

@inline function q4_mitc4_3d_aspect_geom_ok(
    aspect::Float64,
    warp_ratio::Float64,
    kappa_l::Float64,
    edge_skew::Float64,
    aspect_min::Float64,
    aspect_max::Float64,
    warp_min::Float64,
    warp_max::Float64,
    kappa_l_min::Float64,
    kappa_l_max::Float64,
    skew_min::Float64,
    skew_max::Float64,
    skew_aspect_min::Float64,
)
    skew_gate_ok =
        aspect < skew_aspect_min ||
        (edge_skew >= skew_min && edge_skew <= skew_max)
    return aspect >= aspect_min &&
           aspect <= aspect_max &&
           warp_ratio >= warp_min &&
           warp_ratio <= warp_max &&
           kappa_l >= kappa_l_min &&
           kappa_l <= kappa_l_max &&
           skew_gate_ok
end

@inline function kg_shell_surface_operator_mode()
    if !haskey(ENV, "JFEM_KG_SHELL_SURFACE_OPERATOR") &&
       q4_eig_curved_jacobian_enabled()
        return :covariant
    end
    raw = lowercase(strip(get(ENV, "JFEM_KG_SHELL_SURFACE_OPERATOR", "flat")))
    if raw in ("covariant", "surface", "metric")
        return :covariant
    end
    return :flat
end

@inline function kg_shell_nxy_auto_relax()
    return clamp(solver_env_float("JFEM_KG_SHELL_NXY_AUTO_RELAX", 0.0), 0.0, 1.0)
end

@inline function kg_shell_drill_zero_enabled()
    # MSC/Nastran CQUAD4 differential stiffness is a five-DOF shell operator;
    # the drilling direction is stabilized in K, not prestressed in Kg.
    return solver_env_bool("JFEM_KG_SHELL_DRILL_ZERO", true)
end

@inline function kg_shell_nxy_auto_ratio_min()
    return max(solver_env_float("JFEM_KG_SHELL_NXY_AUTO_RATIO_MIN", 1.5), 0.0)
end

@inline function kg_shell_nxy_auto_ratio_full()
    return max(solver_env_float("JFEM_KG_SHELL_NXY_AUTO_RATIO_FULL", 4.0), 1e-12)
end

@inline function kg_shell_nxy_auto_cyl_ratio_max()
    return clamp(solver_env_float("JFEM_KG_SHELL_NXY_AUTO_CYL_RATIO_MAX", 1.0), 0.0, 1.0)
end

@inline function kg_shell_nxy_auto_kappa_l_min()
    return max(solver_env_float("JFEM_KG_SHELL_NXY_AUTO_KAPPA_L_MIN", 0.0), 0.0)
end

@inline function kg_shell_nxy_auto_scale(sxx::Float64, syy::Float64, sxy::Float64,
                                         relax::Float64, ratio_min::Float64, ratio_full::Float64)
    relax <= 0.0 && return 1.0
    denom = abs(sxx) + abs(syy)
    shear_ratio = abs(sxy) / max(denom, 1e-12)
    if ratio_full <= ratio_min
        alpha = shear_ratio >= ratio_min ? 1.0 : 0.0
    else
        alpha = clamp((shear_ratio - ratio_min) / (ratio_full - ratio_min), 0.0, 1.0)
    end
    return max(0.0, 1.0 - relax * alpha)
end

@inline function kg_shell_pcomp_should_scale_nxy(sxx::Float64, syy::Float64, compression_only::Bool)
    return !compression_only || (sxx + syy < 0.0)
end

@inline function kg_shell_apply_pcomp_nxy_scale!(sigma_mem::AbstractMatrix, scale::Float64, compression_only::Bool)
    scale == 1.0 && return
    @inbounds for gp in 1:size(sigma_mem, 1)
        if kg_shell_pcomp_should_scale_nxy(sigma_mem[gp, 1], sigma_mem[gp, 2], compression_only)
            sigma_mem[gp, 3] *= scale
        end
    end
end

@inline function kg_shell_apply_pcomp_nxy_scale!(sigma_mem::AbstractVector, scale::Float64, compression_only::Bool)
    scale == 1.0 && return
    if kg_shell_pcomp_should_scale_nxy(sigma_mem[1], sigma_mem[2], compression_only)
        sigma_mem[3] *= scale
    end
end

@inline function kg_shell_apply_pcomp_nxy_shear_dom_scale!(sigma_mem::AbstractMatrix,
                                                           relax::Float64,
                                                           ratio_min::Float64,
                                                           ratio_full::Float64,
                                                           aspect::Float64,
                                                           aspect_min::Float64,
                                                           aspect_max::Float64,
                                                           compression_only::Bool)
    (relax > 0.0 && aspect >= aspect_min && aspect <= aspect_max) || return
    @inbounds for gp in 1:size(sigma_mem, 1)
        if kg_shell_pcomp_should_scale_nxy(sigma_mem[gp, 1], sigma_mem[gp, 2], compression_only)
            sigma_mem[gp, 3] *= kg_shell_nxy_auto_scale(
                sigma_mem[gp, 1],
                sigma_mem[gp, 2],
                sigma_mem[gp, 3],
                relax,
                ratio_min,
                ratio_full,
            )
        end
    end
end

@inline function kg_shell_apply_pcomp_nxy_shear_dom_scale!(sigma_mem::AbstractVector,
                                                           relax::Float64,
                                                           ratio_min::Float64,
                                                           ratio_full::Float64,
                                                           aspect::Float64,
                                                           aspect_min::Float64,
                                                           aspect_max::Float64,
                                                           compression_only::Bool)
    (relax > 0.0 && aspect >= aspect_min && aspect <= aspect_max) || return
    if kg_shell_pcomp_should_scale_nxy(sigma_mem[1], sigma_mem[2], compression_only)
        sigma_mem[3] *= kg_shell_nxy_auto_scale(
            sigma_mem[1],
            sigma_mem[2],
            sigma_mem[3],
            relax,
            ratio_min,
            ratio_full,
        )
    end
end

@inline function q4_flat_pcomp_phi_metric(coords::AbstractMatrix, Cb::AbstractMatrix, Cs::AbstractMatrix)
    vals = ntuple(4) do e
        i, j = ((1, 2), (2, 3), (3, 4), (4, 1))[e]
        dx = coords[j, 1] - coords[i, 1]
        dy = coords[j, 2] - coords[i, 2]
        L = sqrt(dx * dx + dy * dy)
        if L <= 1e-12
            0.0
        else
            c = dx / L
            s = dy / L
            Cb_loc, Cs_loc = FEM.dkmq_side_local_constitutive(c, s, Cb, Cs)
            12.0 * max(Cb_loc[1, 1], 0.0) / (L * L * max(abs(Cs_loc[1, 1]), 1e-30))
        end
    end
    sorted = sort(collect(vals))
    return 0.5 * (sorted[2] + sorted[3])
end

@inline function solver_k6rot(default::Real, shear_center_only::Bool)
    primary_key = shear_center_only ? "JFEM_PARAM_K6ROT_EIG" : "JFEM_PARAM_K6ROT_STATIC"
    raw = get(ENV, primary_key, get(ENV, "JFEM_PARAM_K6ROT", ""))
    default_val = Float64(default)
    if isempty(strip(raw))
        if shear_center_only
            eig_floor_raw = strip(get(ENV, "JFEM_PARAM_K6ROT_EIG_FLOOR", "300.0"))
            eig_floor = max(something(tryparse(Float64, eig_floor_raw), 300.0), 0.0)
            return max(default_val, eig_floor)
        end
        return default_val
    end
    return something(tryparse(Float64, raw), default_val)
end

@inline function shell_transverse_shear_matrix(mat, h::Real, tst::Real, theta::Real=0.0)
    h_eff = Float64(h)
    tst_eff = Float64(tst)
    theta_eff = Float64(theta)
    mtype = get(mat, "TYPE", "")
    if mtype == "MAT8" && haskey(mat, "G12")
        G1Z = Float64(get(mat, "G1Z", 0.0))
        G2Z = Float64(get(mat, "G2Z", 0.0))
        G12 = Float64(get(mat, "G12", 0.0))
        G1Z <= 0.0 && (G1Z = G12)
        G2Z <= 0.0 && (G2Z = G12)
        if abs(theta_eff) > 1e-10
            ct = cos(theta_eff)
            st = sin(theta_eff)
            return tst_eff * h_eff .* [
                ct^2 * G1Z + st^2 * G2Z  ct * st * (G1Z - G2Z);
                ct * st * (G1Z - G2Z)   st^2 * G1Z + ct^2 * G2Z
            ], max(G1Z, G2Z)
        end
        return tst_eff * h_eff .* [G1Z 0.0; 0.0 G2Z], max(G1Z, G2Z)
    elseif mtype == "MAT2" && haskey(mat, "G11")
        Gxz = Float64(get(mat, "G13", 0.0))
        Gyz = Float64(get(mat, "G23", 0.0))
        if Gxz <= 0.0 && Gyz <= 0.0
            Gxz = Float64(get(mat, "G33", 0.0))
            Gyz = Gxz
        elseif Gxz <= 0.0
            Gxz = Gyz
        elseif Gyz <= 0.0
            Gyz = Gxz
        end
        if abs(theta_eff) > 1e-10
            ct = cos(theta_eff)
            st = sin(theta_eff)
            return tst_eff * h_eff .* [
                ct^2 * Gxz + st^2 * Gyz  ct * st * (Gxz - Gyz);
                ct * st * (Gxz - Gyz)   st^2 * Gxz + ct^2 * Gyz
            ], max(Gxz, Gyz)
        end
        return tst_eff * h_eff .* [Gxz 0.0; 0.0 Gyz], max(Gxz, Gyz)
    end

    G_val = Float64(get(mat, "G", 0.0))
    if G_val <= 0.0
        E_val = Float64(get(mat, "E", 0.0))
        nu_val = Float64(get(mat, "NU", 0.0))
        G_val = E_val > 0.0 ? E_val / (2 * (1 + nu_val)) : 0.0
    end
    return tst_eff * h_eff .* [G_val 0.0; 0.0 G_val], G_val
end

@inline function pcomp_metric_ratios(prop, theta_rad::Float64)
    Cm_metric = copy(prop["Cm"])
    Cb_metric = copy(prop["Cb"])
    Bmb_metric = haskey(prop, "Bmb") && prop["Bmb"] !== nothing ? copy(prop["Bmb"]) : nothing
    if abs(theta_rad) > 1e-10
        cb = cos(theta_rad); sb = sin(theta_rad)
        c2 = cb^2; s2 = sb^2; cs = cb*sb
        T11 = c2;  T12 = s2;  T13 = cs
        T21 = s2;  T22 = c2;  T23 = -cs
        T31 = -2cs; T32 = 2cs; T33 = c2 - s2
        _rotate_constitutive_3x3!(Cm_metric, T11, T12, T13, T21, T22, T23, T31, T32, T33)
        _rotate_constitutive_3x3!(Cb_metric, T11, T12, T13, T21, T22, T23, T31, T32, T33)
        if Bmb_metric !== nothing
            _rotate_constitutive_3x3!(Bmb_metric, T11, T12, T13, T21, T22, T23, T31, T32, T33)
        end
    end
    shear_ratio = abs(Cm_metric[3,3]) / max(0.5 * (abs(Cm_metric[1,1]) + abs(Cm_metric[2,2])), 1e-30)
    d16_ratio = sqrt(Cb_metric[1,3]^2 + Cb_metric[2,3]^2) / max(maximum(abs.(Cb_metric)), 1e-30)
    b_ratio = Bmb_metric === nothing ? 0.0 : maximum(abs.(Bmb_metric)) / max(maximum(abs.(Cm_metric)), 1e-30)
    return shear_ratio, d16_ratio, b_ratio
end

@inline function kg_auto_pcomp_g12_candidate(theta_deg::Float64, shear_ratio::Float64, d16_ratio::Float64,
                                             b_ratio::Float64, kappa_l::Float64, cyl_ratio::Float64)
    if b_ratio > q4_pcomp_kg_auto_g12_b_ratio_max()
        return false
    end
    if shear_ratio < q4_pcomp_kg_auto_g12_shear_ratio_min() && d16_ratio < q4_pcomp_kg_auto_g12_d16_ratio_min()
        return false
    end
    if abs(theta_deg) < q4_pcomp_kg_auto_g12_theta_abs_min() &&
       d16_ratio < q4_pcomp_kg_auto_g12_d16_theta_bypass_min()
        return false
    end
    return kappa_l >= q4_pcomp_kg_auto_g12_kappa_l_min() &&
           cyl_ratio >= q4_pcomp_kg_auto_g12_cyl_ratio_min()
end

@inline function kg_auto_curvature_iso_candidate(kappa_l::Float64, cyl_ratio::Float64, aspect_ratio::Float64)
    return kappa_l >= q4_shell_kg_auto_curvature_iso_kappa_l_min() &&
           cyl_ratio >= q4_shell_kg_auto_curvature_iso_cyl_ratio_min() &&
           aspect_ratio <= q4_shell_kg_auto_curvature_iso_aspect_ratio_max()
end

@inline function kg_auto_curvature_iso_cyl_candidate(kappa_l::Float64, cyl_ratio::Float64, aspect_ratio::Float64)
    return q4_shell_kg_auto_curvature_iso_cyl_enabled() &&
           kappa_l >= q4_shell_kg_auto_curvature_iso_cyl_kappa_l_min() &&
           cyl_ratio <= q4_shell_kg_auto_curvature_iso_cyl_ratio_max() &&
           aspect_ratio <= q4_shell_kg_auto_curvature_iso_aspect_ratio_max()
end

@inline function kg_auto_curvature_pcomp_candidate(theta_deg::Float64, shear_ratio::Float64, d16_ratio::Float64,
                                                   b_ratio::Float64, kappa_l::Float64, cyl_ratio::Float64)
    if b_ratio > q4_pcomp_kg_auto_curvature_b_ratio_max()
        return false
    end
    if shear_ratio < q4_pcomp_kg_auto_curvature_shear_ratio_min() &&
       d16_ratio < q4_pcomp_kg_auto_curvature_d16_ratio_min()
        return false
    end
    if abs(theta_deg) < q4_pcomp_kg_auto_curvature_theta_abs_min() &&
       d16_ratio < q4_pcomp_kg_auto_curvature_d16_theta_bypass_min()
        return false
    end
    return kappa_l >= q4_pcomp_kg_auto_curvature_kappa_l_min() &&
           cyl_ratio >= q4_pcomp_kg_auto_curvature_cyl_ratio_min()
end

# In-place rotation of 3x3 constitutive matrix: C_out = T' * C * T
# where T = [T11 T12 T13; T21 T22 T23; T31 T32 T33] is the strain transformation
@inline function _rotate_constitutive_3x3!(C::Matrix{Float64},
    T11, T12, T13, T21, T22, T23, T31, T32, T33)
    t11 = C[1,1]*T11 + C[1,2]*T21 + C[1,3]*T31
    t12 = C[1,1]*T12 + C[1,2]*T22 + C[1,3]*T32
    t13 = C[1,1]*T13 + C[1,2]*T23 + C[1,3]*T33
    t21 = C[2,1]*T11 + C[2,2]*T21 + C[2,3]*T31
    t22 = C[2,1]*T12 + C[2,2]*T22 + C[2,3]*T32
    t23 = C[2,1]*T13 + C[2,2]*T23 + C[2,3]*T33
    t31 = C[3,1]*T11 + C[3,2]*T21 + C[3,3]*T31
    t32 = C[3,1]*T12 + C[3,2]*T22 + C[3,3]*T32
    t33 = C[3,1]*T13 + C[3,2]*T23 + C[3,3]*T33
    # C = T' * tmp
    C[1,1] = T11*t11 + T21*t21 + T31*t31
    C[1,2] = T11*t12 + T21*t22 + T31*t32
    C[1,3] = T11*t13 + T21*t23 + T31*t33
    C[2,1] = T12*t11 + T22*t21 + T32*t31
    C[2,2] = T12*t12 + T22*t22 + T32*t32
    C[2,3] = T12*t13 + T22*t23 + T32*t33
    C[3,1] = T13*t11 + T23*t21 + T33*t31
    C[3,2] = T13*t12 + T23*t22 + T33*t32
    C[3,3] = T13*t13 + T23*t23 + T33*t33
    return nothing
end

function assemble_stiffness(model; bending_incomp::Bool=true, shear_center_only::Bool=false,
                            membrane_incomp::Bool=true, pcomp_membrane_incomp::Bool=false,
                            snorm_angle_override::Union{Nothing,Float64}=nothing,
                            iso_no_incomp::Bool=false)
    log_msg("[SOLVER] Indexing...")
    ids = sort(collect(keys(model["GRIDs"])), by=x->parse(Int,x))
    n_nodes = length(ids)
    id_map = Dict(parse(Int, k)=>i for (i,k) in enumerate(ids))
    ndof = n_nodes * 6

    node_R = Vector{Matrix{Float64}}(undef, n_nodes)
    node_coords = zeros(n_nodes, 3)

    for (sid, g) in model["GRIDs"]
        idx = id_map[g["ID"]]
        node_coords[idx, :] = g["X"]
        cid = g["CD"]
        if cid == 0
            node_R[idx] = Matrix(1.0I, 3, 3)
        elseif haskey(model["CORDs"], string(cid))
            c = model["CORDs"][string(cid)]
            node_R[idx] = hcat(c["U"], c["V"], c["W"])
        else
             node_R[idx] = Matrix(1.0I, 3, 3)
        end
    end

    snorm_override_key = shear_center_only ? "JFEM_PARAM_SNORM_OVERRIDE_EIG" : "JFEM_PARAM_SNORM_OVERRIDE_STATIC"
    snorm_override = isnothing(snorm_angle_override) ?
        get(ENV, snorm_override_key, get(ENV, "JFEM_PARAM_SNORM_OVERRIDE", "")) :
        string(snorm_angle_override)
    snorm_model = model
    if !isempty(strip(snorm_override))
        snorm_model = copy(model)
        snorm_model["PARAM_SNORM"] = something(tryparse(Float64, snorm_override), get(model, "PARAM_SNORM", 0.0))
    end
    snorm_normals = compute_snorm_normals(snorm_model, id_map, node_coords)
    q4_frame_mode = q4_frame_mode_from_env(shear_center_only ? "JFEM_Q4_FRAME_MODE_EIG" : "JFEM_Q4_FRAME_MODE_STATIC")
    pcomp_axis_mode_primary_key = shear_center_only ? "JFEM_Q4_PCOMP_AXIS_MODE_EIG" : "JFEM_Q4_PCOMP_AXIS_MODE_STATIC"
    pcomp_axis_mode = q4_pcomp_axis_mode(pcomp_axis_mode_primary_key)
    pcomp_axis_mode_override = haskey(ENV, pcomp_axis_mode_primary_key) || haskey(ENV, "JFEM_Q4_PCOMP_AXIS_MODE")
    curved_iso_blend = curved_iso_eig_fullshear_blend()
    curved_iso_square_blend = q4_curved_iso_square_fullshear_blend_enabled() ? q4_curved_iso_square_fullshear_blend_value() : 1.0
    curved_iso_square_blend_aspect_ratio_max = q4_curved_iso_square_fullshear_blend_aspect_ratio_max()
    curved_pcomp_blend = curved_pcomp_eig_fullshear_blend()
    flat_pcomp_no_phi2_override = solver_env_optional_bool("JFEM_SOL105_EIG_FLAT_PCOMP_NO_PHI2")
    flat_pcomp_auto_phi2 = shear_center_only && isnothing(flat_pcomp_no_phi2_override) && q4_flat_pcomp_auto_phi2_enabled()
    flat_pcomp_auto_shear_ratio_max = q4_flat_pcomp_auto_phi2_shear_ratio_max()
    flat_pcomp_auto_d16_ratio_max = q4_flat_pcomp_auto_phi2_d16_ratio_max()
    flat_pcomp_auto_b_ratio_max = q4_flat_pcomp_auto_phi2_b_ratio_max()
    flat_pcomp_auto_cyl_ratio_max = q4_flat_pcomp_auto_phi2_cyl_ratio_max()
    flat_pcomp_auto_kappa_l_min = q4_flat_pcomp_auto_phi2_kappa_l_min()
    flat_pcomp_plate_branch = shear_center_only && q4_sol105_flat_pcomp_plate_branch_enabled()
    flat_pcomp_dkmq_branch = (shear_center_only || q4_sol105_flat_pcomp_dkmq_static_enabled()) &&
                             q4_sol105_flat_pcomp_dkmq_enabled()
    flat_pcomp_plate_auto = shear_center_only && q4_sol105_flat_pcomp_plate_auto_enabled()
    flat_pcomp_plate_auto_d16_ratio_max = q4_sol105_flat_pcomp_plate_auto_d16_ratio_max()
    flat_pcomp_plate_auto_shear_ratio_max = q4_sol105_flat_pcomp_plate_auto_shear_ratio_max()
    flat_pcomp_rect_adini = shear_center_only && q4_sol105_flat_pcomp_rect_adini_enabled()
    flat_pcomp_fullshear_selective = solver_env_bool("JFEM_SOL105_EIG_FLAT_PCOMP_FULLSHEAR_SELECTIVE", false)
    flat_pcomp_shear_scale = shear_center_only ? q4_sol105_flat_pcomp_shear_scale() : 1.0
    flat_pcomp_auto_shear_scale = shear_center_only && q4_sol105_flat_pcomp_auto_shear_scale_enabled()
    flat_pcomp_auto_shear_scale_gain = q4_sol105_flat_pcomp_auto_shear_scale_gain()
    flat_pcomp_auto_shear_scale_max = q4_sol105_flat_pcomp_auto_shear_scale_max()
    flat_pcomp_exact_membrane = shear_center_only && q4_sol105_flat_pcomp_exact_membrane()
    flat_pcomp_exact_side_shear = q4_sol105_flat_pcomp_exact_side_shear()
    flat_curved_pcomp_exact_side_shear = q4_sol105_flat_curved_pcomp_exact_side_shear()
    flat_pcomp_exact_side_rotcorr = q4_sol105_flat_pcomp_exact_side_rotcorr()
    flat_iso_exact_side_shear = q4_sol105_flat_iso_exact_side_shear()
    flat_curved_iso_coarse_exact_side_shear = shear_center_only && q4_flat_curved_iso_coarse_exact_side_shear_enabled()
    flat_curved_iso_coarse_exact_side_shear_aspect_ratio_max = q4_flat_curved_iso_coarse_exact_side_shear_aspect_ratio_max()
    flat_curved_iso_coarse_exact_side_shear_valence_sum_min = q4_flat_curved_iso_coarse_exact_side_shear_valence_sum_min()
    flat_curved_iso_coarse_exact_side_shear_valence_sum_max = q4_flat_curved_iso_coarse_exact_side_shear_valence_sum_max()
    flat_iso_exact_side_rotcorr = q4_sol105_flat_iso_exact_side_rotcorr()
    flat_pcomp_auto_g12 = shear_center_only && !pcomp_axis_mode_override && q4_flat_pcomp_auto_g12_enabled()
    flat_pcomp_auto_g12_kappa_l_max = q4_flat_pcomp_auto_g12_kappa_l_max()
    flat_pcomp_auto_g12_cyl_ratio_max = q4_flat_pcomp_auto_g12_cyl_ratio_max()
    curved_iso_eig_membrane_incomp = shear_center_only && q4_curved_iso_eig_auto_membrane_incomp_enabled()
    curved_iso_eig_membrane_incomp_kappa_l_min = q4_curved_iso_eig_auto_membrane_incomp_kappa_l_min()
    curved_iso_eig_membrane_incomp_cyl_ratio_max = q4_curved_iso_eig_auto_membrane_incomp_cyl_ratio_max()
    curved_iso_warp_membrane_incomp = shear_center_only && q4_curved_iso_warp_membrane_incomp_enabled()
    curved_iso_warp_membrane_incomp_ratio_min = q4_curved_iso_warp_membrane_incomp_ratio_min()
    curved_iso_warp_membrane_incomp_kappa_l_max = q4_curved_iso_warp_membrane_incomp_kappa_l_max()
    curved_iso_elongated_membrane_incomp = shear_center_only && q4_curved_iso_elongated_membrane_incomp_enabled()
    curved_iso_elongated_membrane_incomp_aspect_ratio_min = q4_curved_iso_elongated_membrane_incomp_aspect_ratio_min()
    membrane_incomp_center_jacobian = q4_sol105_membrane_incomp_center_jacobian_enabled()
    static_pcomp_membrane_incomp_aspect =
        !shear_center_only && q4_sol105_static_pcomp_membrane_incomp_aspect_enabled()
    static_pcomp_membrane_incomp_aspect_min = q4_sol105_static_pcomp_membrane_incomp_aspect_min()
    static_pcomp_membrane_incomp_aspect_max = q4_sol105_static_pcomp_membrane_incomp_aspect_max()
    flat_iso_eig_membrane_incomp = q4_flat_iso_eig_membrane_incomp_enabled()
    flat_iso_eig_membrane_shear_center_row = q4_flat_iso_eig_membrane_shear_center_row_enabled()
    flat_iso_eig_membrane_assumed_mode = q4_flat_iso_eig_membrane_assumed_mode()
    flat_iso_dkmq_branch = shear_center_only && q4_sol105_flat_iso_dkmq_enabled()
    flat_iso_fullshear_selective_mode = q4_flat_iso_fullshear_selective_mode()
    flat_pcomp_eig_membrane_assumed_mode = q4_flat_pcomp_eig_membrane_assumed_mode()
    flat_pcomp_taper_membrane_none = q4_flat_pcomp_taper_membrane_none_enabled()
    flat_pcomp_taper_membrane_none_ratio_max = q4_flat_pcomp_taper_membrane_none_ratio_max()
    flat_pcomp_taper_membrane_none_aspect_min = q4_flat_pcomp_taper_membrane_none_aspect_min()
    nonflat_pcomp_eig_membrane_assumed_mode = q4_nonflat_pcomp_eig_membrane_assumed_mode()
    marguerre_static_coupling =
        !shear_center_only &&
        q4_marguerre_coupling_enabled() &&
        solver_env_bool("JFEM_Q4_MARGUERRE_STATIC_COUPLING", false)
    marguerre_coupling_enabled =
        (shear_center_only && q4_marguerre_coupling_enabled()) ||
        marguerre_static_coupling
    marguerre_static_use_geom_normals =
        marguerre_static_coupling &&
        solver_env_bool("JFEM_Q4_MARGUERRE_STATIC_GEOM_NORMALS", true)
    marguerre_coupling_scale = q4_marguerre_coupling_scale()
    marguerre_coupling_convention = q4_marguerre_coupling_convention()
    marguerre_handover_marker = marguerre_coupling_convention === :handover ? 1.0 : 0.0
    static_component_cm_scale =
        shear_center_only ? 1.0 : solver_env_float("JFEM_Q4_STATIC_COMPONENT_CM_SCALE", 1.0)
    static_component_cb_scale =
        shear_center_only ? 1.0 : solver_env_float("JFEM_Q4_STATIC_COMPONENT_CB_SCALE", 1.0)
    static_component_cs_scale =
        shear_center_only ? 1.0 : solver_env_float("JFEM_Q4_STATIC_COMPONENT_CS_SCALE", 1.0)
    static_component_bmb_scale =
        shear_center_only ? 1.0 : solver_env_float("JFEM_Q4_STATIC_COMPONENT_BMB_SCALE", 1.0)
    static_component_drill_scale =
        shear_center_only ? 1.0 : solver_env_float("JFEM_Q4_STATIC_COMPONENT_DRILL_SCALE", 1.0)
    static_component_pid_filter =
        shear_center_only ? Int[] : q4_static_component_pid_list()
    static_component_eid_filter =
        shear_center_only ? Int[] : q4_static_component_eid_list()
    static_component_neighbor_pid_prefixes =
        shear_center_only ? String[] : q4_static_component_neighbor_pid_prefixes()
    static_component_v2_min = shear_center_only ? 0.0 : q4_static_component_v2_min()
    static_component_v2_max = shear_center_only ? 0.0 : q4_static_component_v2_max()
    static_component_v2 =
        (static_component_v2_min <= 0.0 && static_component_v2_max <= 0.0) ?
        0.0 : q4_static_component_model_eigrl_v2(model)
    static_component_v2_gate_ok =
        q4_static_component_v2_ok(static_component_v2,
                                  static_component_v2_min,
                                  static_component_v2_max)
    static_component_thickness_min =
        shear_center_only ? 0.0 : q4_static_component_thickness_min()
    static_component_thickness_max =
        shear_center_only ? 0.0 : q4_static_component_thickness_max()
    static_component_pcomp_shear_ratio_min =
        shear_center_only ? 0.0 : q4_static_component_pcomp_shear_ratio_min()
    static_component_pcomp_shear_ratio_max =
        shear_center_only ? 0.0 : q4_static_component_pcomp_shear_ratio_max()
    static_component_pcomp_d16_ratio_min =
        shear_center_only ? 0.0 : q4_static_component_pcomp_d16_ratio_min()
    static_component_pcomp_d16_ratio_max =
        shear_center_only ? 0.0 : q4_static_component_pcomp_d16_ratio_max()
    static_component_pcomp_b_ratio_min =
        shear_center_only ? 0.0 : q4_static_component_pcomp_b_ratio_min()
    static_component_pcomp_b_ratio_max =
        shear_center_only ? 0.0 : q4_static_component_pcomp_b_ratio_max()
    static_pcomp_nodal_geomnormal_transform =
        !shear_center_only &&
        solver_env_bool("JFEM_Q4_STATIC_PCOMP_NODAL_GEOMNORMAL_TRANSFORM", false)
    static_curvature_membrane_geom_normals =
        !shear_center_only &&
        solver_env_bool("JFEM_Q4_CURVATURE_MEMBRANE_STATIC_GEOM_NORMALS", false)
    # Curved-Jacobian fix path #2 (2026-04-21 PM scaffold). When enabled, the
    # per-element block fills a 4×3 coords_3d buffer and passes it to
    # stiffness_quad4_matrices via the `coords_3d` kwarg. The function itself
    # does not yet consume this input — the main Gauss-loop rewrite + NRM
    # kinematic coupling is the remaining work for next session. See
    # project_htp_curved_scaffold_2026_04_21.md in memory.
    curved_jacobian_enabled = q4_curved_jacobian_enabled(shear_center_only)
    q4_kernel_key = shear_center_only ? "JFEM_Q4_KERNEL_EIG" : "JFEM_Q4_KERNEL_STATIC"
    q4_kernel_mode_static = lowercase(strip(get(ENV, q4_kernel_key, get(ENV, "JFEM_Q4_KERNEL", "macneal"))))
    mitc4_3d_all_kernel = q4_kernel_mode_static in ("mitc4_3d", "mitc4-3d", "mitc3d")
    mitc4_3d_aspect_kernel = q4_kernel_mode_static in (
        "mitc4_3d_aspect", "mitc4-3d-aspect", "mitc3d_aspect", "mitc3d-aspect",
    )
    mitc4_3d_kernel = mitc4_3d_all_kernel || mitc4_3d_aspect_kernel
    mitc4_3d_aspect_min = max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_MIN", 3.0), 1.0)
    mitc4_3d_aspect_max = max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_MAX", 1e30), mitc4_3d_aspect_min)
    mitc4_3d_aspect_warp_min = max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_WARP_MIN", 0.0), 0.0)
    mitc4_3d_aspect_warp_max = max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_WARP_MAX", 1e30), mitc4_3d_aspect_warp_min)
    mitc4_3d_aspect_kappa_l_min = max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KAPPA_L_MIN", 0.0), 0.0)
    mitc4_3d_aspect_kappa_l_max = max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KAPPA_L_MAX", 1e30), mitc4_3d_aspect_kappa_l_min)
    mitc4_3d_aspect_skew_min = q4_mitc4_3d_aspect_skew_min()
    mitc4_3d_aspect_skew_max = max(q4_mitc4_3d_aspect_skew_max(), mitc4_3d_aspect_skew_min)
    mitc4_3d_aspect_skew_aspect_min = q4_mitc4_3d_aspect_skew_aspect_min()
    mitc4_3d_aspect_pcomp_only = solver_env_bool("JFEM_Q4_MITC4_3D_ASPECT_PCOMP_ONLY", true)
    mitc4_3d_ply_integration = solver_env_bool("JFEM_Q4_MITC4_3D_PLY_INTEGRATION", true)
    q4_macneal_bending_scale = solver_env_float("JFEM_Q4_MACNEAL_BENDING_SCALE", 1.0)
    q4_macneal_bending_isolated_scale =
        solver_env_float("JFEM_Q4_MACNEAL_BENDING_ISOLATED_SCALE", q4_macneal_bending_scale)
    q4_macneal_bending_aspect_enabled = q4_macneal_bending_aspect_scale_enabled()
    q4_macneal_bending_aspect_mode_v = q4_macneal_bending_aspect_mode()
    q4_macneal_bending_aspect_low = q4_macneal_bending_aspect_low_scale()
    q4_macneal_bending_aspect_mid = q4_macneal_bending_aspect_mid_scale()
    q4_macneal_bending_aspect_high = q4_macneal_bending_aspect_high_scale()
    q4_macneal_bending_aspect_min_v = q4_macneal_bending_aspect_min()
    q4_macneal_bending_aspect_peak_v = q4_macneal_bending_aspect_peak()
    q4_macneal_bending_aspect_max_v = q4_macneal_bending_aspect_max()
    q4_macneal_bending_aspect_warp_min_v = q4_macneal_bending_aspect_warp_min()
    q4_macneal_bending_aspect_warp_max_v = q4_macneal_bending_aspect_warp_max()
    q4_macneal_bending_aspect_kappa_l_min_v = q4_macneal_bending_aspect_kappa_l_min()
    q4_macneal_bending_aspect_kappa_l_max_v = q4_macneal_bending_aspect_kappa_l_max()
    q4_macneal_bending_aspect_skew_min_v = q4_macneal_bending_aspect_skew_min()
    q4_macneal_bending_aspect_skew_max_v = q4_macneal_bending_aspect_skew_max()
    q4_macneal_bending_aspect2_enabled = q4_macneal_bending_aspect2_scale_enabled()
    q4_macneal_bending_aspect2_mode_v = q4_macneal_bending_aspect2_mode()
    q4_macneal_bending_aspect2_low = q4_macneal_bending_aspect2_low_scale()
    q4_macneal_bending_aspect2_mid = q4_macneal_bending_aspect2_mid_scale()
    q4_macneal_bending_aspect2_high = q4_macneal_bending_aspect2_high_scale()
    q4_macneal_bending_aspect2_min_v = q4_macneal_bending_aspect2_min()
    q4_macneal_bending_aspect2_peak_v = q4_macneal_bending_aspect2_peak()
    q4_macneal_bending_aspect2_max_v = q4_macneal_bending_aspect2_max()
    q4_macneal_bending_aspect2_warp_min_v = q4_macneal_bending_aspect2_warp_min()
    q4_macneal_bending_aspect2_warp_max_v = q4_macneal_bending_aspect2_warp_max()
    q4_macneal_bending_aspect2_kappa_l_min_v = q4_macneal_bending_aspect2_kappa_l_min()
    q4_macneal_bending_aspect2_kappa_l_max_v = q4_macneal_bending_aspect2_kappa_l_max()
    q4_macneal_bending_aspect2_skew_min_v = q4_macneal_bending_aspect2_skew_min()
    q4_macneal_bending_aspect2_skew_max_v = q4_macneal_bending_aspect2_skew_max()
    q4_macneal_curved_bending_scale = solver_env_float("JFEM_Q4_MACNEAL_CURVED_BENDING_SCALE", 1.0)
    q4_macneal_curved_bending_enabled = q4_macneal_curved_bending_scale != 1.0
    q4_macneal_curved_bending_kappa_l_min =
        max(solver_env_float("JFEM_Q4_MACNEAL_CURVED_BENDING_KAPPA_L_MIN", 1e-6), 0.0)
    q4_macneal_curved_bending_cyl_ratio_max =
        max(solver_env_float("JFEM_Q4_MACNEAL_CURVED_BENDING_CYL_RATIO_MAX", 1.0), 0.0)
    # MSC Nastran QRG 2024.1 MAT8: blank/zero G1Z and G2Z mean zero
    # transverse-shear flexibility, i.e. the infinite-stiffness limit.
    mat8_blank_ts_rigid_limit = solver_env_bool(
        "JFEM_MAT8_BLANK_TS_RIGID_LIMIT",
        true,
    )
    q4_kernel_needs_surface_flatness =
        q4_kernel_mode_static in ("macneal", "macneal_pcomp", "macneal-pcomp", "macneal_aniso",
                                  "mitc4_3d_aspect", "mitc4-3d-aspect", "mitc3d_aspect", "mitc3d-aspect")
    q4_macneal_pcomp_surface_kappa_l_max =
        max(solver_env_float("JFEM_Q4_MACNEAL_PCOMP_SURFACE_KAPPA_L_MAX", 1e-4), 0.0)
    nonflat_pcomp_exact_cs_enabled = q4_nonflat_pcomp_exact_cs_enabled()
    flat_curved_iso_eig_center_only = shear_center_only && q4_flat_curved_iso_eig_center_only_enabled()
    flat_curved_iso_eig_center_only_kappa_l_min = q4_flat_curved_iso_eig_center_only_kappa_l_min()
    flat_curved_iso_eig_center_only_cyl_ratio_max = q4_flat_curved_iso_eig_center_only_cyl_ratio_max()
    flat_curved_iso_exact_membrane_aspect_ratio_max = q4_flat_curved_iso_exact_membrane_aspect_ratio_max()
    flat_curved_iso_geomnormal_frame = shear_center_only && q4_flat_curved_iso_geomnormal_frame_enabled()
    flat_curved_iso_geomnormal_frame_aspect_ratio_min = q4_flat_curved_iso_geomnormal_frame_aspect_ratio_min()
    flat_curved_iso_geomnormal_frame_kappa_l_min = q4_flat_curved_iso_geomnormal_frame_kappa_l_min()
    flat_curved_iso_geomnormal_frame_kappa_l_max = q4_flat_curved_iso_geomnormal_frame_kappa_l_max()
    flat_curved_iso_geomnormal_frame_cyl_ratio_max = q4_flat_curved_iso_geomnormal_frame_cyl_ratio_max()
    flat_curved_iso_nodal_geomnormal_transform = shear_center_only && q4_flat_curved_iso_nodal_geomnormal_transform_enabled()
    flat_curved_iso_nodal_geomnormal_transform_aspect_ratio_min = q4_flat_curved_iso_nodal_geomnormal_transform_aspect_ratio_min()
    flat_curved_iso_nodal_geomnormal_transform_valence_sum_max = q4_flat_curved_iso_nodal_geomnormal_transform_valence_sum_max()
    curved_iso_geomnormal_frame = q4_curved_iso_geomnormal_frame_enabled()
    curved_iso_geomnormal_frame_aspect_ratio_min = q4_curved_iso_geomnormal_frame_aspect_ratio_min()
    curved_iso_geomnormal_frame_kappa_l_min = q4_curved_iso_geomnormal_frame_kappa_l_min()
    curved_iso_geomnormal_frame_kappa_l_max = q4_curved_iso_geomnormal_frame_kappa_l_max()
    curved_iso_geomnormal_frame_cyl_ratio_max = q4_curved_iso_geomnormal_frame_cyl_ratio_max()
    flat_curved_pcomp_fullshear = shear_center_only && q4_flat_curved_pcomp_fullshear_enabled()
    flat_curved_pcomp_fullshear_kappa_l_min = q4_flat_curved_pcomp_fullshear_kappa_l_min()
    flat_curved_pcomp_fullshear_cyl_ratio_max = q4_flat_curved_pcomp_fullshear_cyl_ratio_max()
    flat_curved_pcomp_geomnormal_frame = shear_center_only && q4_flat_curved_pcomp_geomnormal_frame_enabled()
    pcomp_auto_global_x = !pcomp_axis_mode_override && q4_pcomp_auto_global_x_enabled()
    pcomp_auto_global_x_shear_ratio_max = q4_pcomp_auto_global_x_shear_ratio_max()
    pcomp_auto_global_x_d16_ratio_max = q4_pcomp_auto_global_x_d16_ratio_max()
    pcomp_auto_global_x_b_ratio_max = q4_pcomp_auto_global_x_b_ratio_max()
    pcomp_auto_global_x_cyl_ratio_min = q4_pcomp_auto_global_x_cyl_ratio_min()
    pcomp_auto_global_x_kappa_l_min = q4_pcomp_auto_global_x_kappa_l_min()

    cshells = model["CSHELLs"]
    cbars = model["CBARs"]
    cbeams = get(model, "CBEAMs", Dict())
    crods = get(model, "CRODs", Dict())
    rbe2s = get(model, "RBE2s", Dict())
    model_has_line_elements = !isempty(cbars) || !isempty(cbeams) || !isempty(crods) || !isempty(get(model, "CONRODs", Dict()))
    model_has_kinematic_constraints =
        !isempty(rbe2s) ||
        !isempty(get(model, "RBE3s", Dict())) ||
        !isempty(get(model, "RSPLINEs", Dict())) ||
        !isempty(get(model, "MPCs", []))
    pshells = model["PSHELLs"]; pbarls = model["PBARLs"]; mats = model["MATs"]
    auto_pcomp_membrane_incomp_model = any(q4_sol105_pcomp_auto_membrane_incomp_candidate, values(pshells))
    k6rot = solver_k6rot(get(model, "PARAM_K6ROT", 100.0), shear_center_only)
    iso_static_k6rot_override = shear_center_only ? nothing : shell_sol101_iso_static_k6rot_override()
    iso_eig_k6rot_override = shear_center_only ? shell_sol105_iso_eig_k6rot_override() : nothing
    iso_eig_k6rot_cyl_ratio_min = shell_sol105_iso_eig_k6rot_cyl_ratio_min()
    iso_static_drill_scale_override = shear_center_only ? nothing : shell_sol101_iso_static_drill_scale_override()
    iso_eig_drill_scale_override = shear_center_only ? shell_sol105_iso_eig_drill_scale_override() : nothing
    iso_eig_drill_scale_cyl_ratio_min = shell_sol105_iso_eig_drill_scale_cyl_ratio_min()

    nt = Threads.maxthreadid()
    log_msg("[SOLVER] Computing Element Stiffness ($(Threads.nthreads()) threads)...")

    # Convert id_map Dict to dense Vector
    if isempty(id_map)
        error("No nodes found in model — is this a standalone BDF or an INCLUDE fragment?")
    end
    max_nid = maximum(keys(id_map))
    id_vec = zeros(Int, max_nid)
    for (nid, idx) in id_map
        id_vec[nid] = idx
    end

    # Convert snorm_normals Dict to arrays
    snorm_vec = fill(SVector(0.0, 0.0, 0.0), n_nodes)
    snorm_has = falses(n_nodes)
    for (idx, nrm) in snorm_normals
        snorm_vec[idx] = nrm
        snorm_has[idx] = true
    end
    auto_global_x_needs_geom = pcomp_auto_global_x && (pcomp_auto_global_x_kappa_l_min > 0.0 || pcomp_auto_global_x_cyl_ratio_min > 0.0)
    needs_geom_normals = flat_pcomp_auto_phi2 || auto_global_x_needs_geom ||
                         curved_iso_eig_membrane_incomp || flat_curved_iso_eig_center_only ||
                         curved_iso_geomnormal_frame || q4_kernel_needs_surface_flatness ||
                         mitc4_3d_kernel || q4_macneal_curved_bending_enabled ||
                         q4_macneal_bending_aspect_enabled || q4_macneal_bending_aspect2_enabled ||
                         marguerre_static_use_geom_normals || static_pcomp_nodal_geomnormal_transform ||
                         static_curvature_membrane_geom_normals
    geom_normals = needs_geom_normals ? compute_geometric_nodal_normals(model, id_map, node_coords) : Dict{Int, SVector{3,Float64}}()
    geom_vec = fill(SVector(0.0, 0.0, 0.0), n_nodes)
    geom_has = falses(n_nodes)
    for (idx, nrm) in geom_normals
        geom_vec[idx] = nrm
        geom_has[idx] = true
    end
    node_has_line = build_node_has_line_elements(model, id_map, n_nodes)

    # Expand higher-order shells (CQUAD8→4×CQUAD4, CTRIA6→4×CTRIA3) into sub-elements
    # that feed into the existing CQUAD4/CTRIA3 pipeline with full accuracy.
    raw_shells = collect(values(cshells))
    shell_list = []
    n_q8_expanded = 0; n_t6_expanded = 0
    for el in raw_shells
        nids = el["NODES"]; n = length(nids)
        if n == 8
            # CQUAD8: corners 1-4, midsides 5-8 (5=mid(1-2), 6=mid(2-3), 7=mid(3-4), 8=mid(4-1))
            # → 4 CQUAD4: [1,5,9,8], [5,2,6,9], [9,6,3,7], [8,9,7,4] with center 9
            # Since there's no center node, use [1,5,6,8], [5,2,6,8] etc → simpler: 4 triangles
            # Actually: split into 4 CQUAD4 using midside nodes as corners:
            #   Q1: [n1, n5, center, n8], Q2: [n5, n2, n6, center], etc.
            # Without a center node, split into 4 CTRIA3 instead (more robust):
            #   T1: [n1, n5, n8], T2: [n5, n2, n6], T3: [n6, n3, n7], T4: [n7, n4, n8]
            #   T5: [n5, n6, n8], T6: [n6, n7, n8]  (center triangles)
            # Standard subdivision: 8-node quad → 6 triangles
            g = nids
            for tri_nodes in [[g[1],g[5],g[8]], [g[5],g[2],g[6]], [g[6],g[3],g[7]], [g[7],g[4],g[8]],
                              [g[5],g[6],g[8]], [g[6],g[7],g[8]]]
                push!(shell_list, Dict(
                    "ID"=>el["ID"],
                    "PID"=>el["PID"],
                    "NODES"=>tri_nodes,
                    "THETA"=>get(el,"THETA",0.0),
                    "MCID"=>get(el,"MCID",0),
                ))
            end
            n_q8_expanded += 1
        elseif n == 6
            # CTRIA6: corners 1-3, midsides 4-6 (4=mid(1-2), 5=mid(2-3), 6=mid(3-1))
            # → 4 CTRIA3: [n1,n4,n6], [n4,n2,n5], [n5,n3,n6], [n4,n5,n6]
            g = nids
            for tri_nodes in [[g[1],g[4],g[6]], [g[4],g[2],g[5]], [g[5],g[3],g[6]], [g[4],g[5],g[6]]]
                push!(shell_list, Dict(
                    "ID"=>el["ID"],
                    "PID"=>el["PID"],
                    "NODES"=>tri_nodes,
                    "THETA"=>get(el,"THETA",0.0),
                    "MCID"=>get(el,"MCID",0),
                ))
            end
            n_t6_expanded += 1
        else
            push!(shell_list, el)
        end
    end
    if n_q8_expanded + n_t6_expanded > 0
        log_msg("[SOLVER] Expanded $n_q8_expanded CQUAD8 + $n_t6_expanded CTRIA6 into sub-elements")
    end
    n_shells = length(shell_list)

    # Pass 1: count QUAD4 and TRIA3 elements
    n_q4 = 0; n_t3 = 0
    for ei in 1:n_shells
        el = shell_list[ei]
        pid = string(el["PID"])
        if !haskey(pshells, pid); continue; end
        prop = pshells[pid]
        mid = string(prop["MID"])
        if !haskey(mats, mid); continue; end
        nids = el["NODES"]; n = length(nids)
        valid = true
        for k in 1:n
            nid = nids[k]
            if nid < 1 || nid > max_nid || id_vec[nid] == 0; valid = false; break; end
        end
        if !valid; continue; end
        if n == 4; n_q4 += 1; elseif n == 3; n_t3 += 1; end
    end

    # Pre-allocate QUAD4 flat arrays
    q4_idx     = Matrix{Int}(undef, n_q4, 4)
    q4_h       = Vector{Float64}(undef, n_q4)
    q4_br      = Vector{Float64}(undef, n_q4)
    q4_Eref    = Vector{Float64}(undef, n_q4)
    q4_Cm_flat = zeros(3, 3, n_q4)
    q4_Cb_flat = zeros(3, 3, n_q4)
    q4_Cs_flat = zeros(2, 2, n_q4)
    q4_Cs_raw_flat = zeros(2, 2, n_q4)
    q4_Bmb_flat = zeros(3, 3, n_q4)
    q4_has_Bmb = falses(n_q4)
    q4_is_pcomp = falses(n_q4)
    q4_is_pcomp_isotropic = falses(n_q4)
    q4_pcomp_rigid_shear = falses(n_q4)
    q4_is_isotropic = falses(n_q4)
    q4_eid_int = zeros(Int, n_q4)
    q4_pid_int = zeros(Int, n_q4)
    q4_pcomp_auto_element_axis_prop = falses(n_q4)
    q4_el_theta = zeros(n_q4)
    q4_el_mcid = zeros(Int, n_q4)
    q4_pcomp_shear_ratio = zeros(n_q4)
    q4_pcomp_d16_ratio = zeros(n_q4)
    q4_pcomp_b_ratio = zeros(n_q4)
    q4_ply_data = Vector{Any}(undef, n_q4)

    # Pre-allocate TRIA3 arrays
    t3_idx     = Matrix{Int}(undef, n_t3, 3)
    t3_h       = Vector{Float64}(undef, n_t3)
    t3_br      = Vector{Float64}(undef, n_t3)
    t3_tst     = Vector{Float64}(undef, n_t3)
    t3_Eref    = Vector{Float64}(undef, n_t3)
    t3_Cm      = Vector{Matrix{Float64}}(undef, n_t3)
    t3_Cb      = Vector{Matrix{Float64}}(undef, n_t3)
    t3_Cs      = Vector{Matrix{Float64}}(undef, n_t3)
    t3_Bmb     = Vector{Union{Nothing, Matrix{Float64}}}(undef, n_t3)
    t3_is_pcomp = falses(n_t3)
    t3_is_pcomp_isotropic = falses(n_t3)
    t3_is_isotropic = falses(n_t3)
    t3_el_theta = zeros(n_t3)
    t3_el_mcid = zeros(Int, n_t3)

    # Pass 2: fill arrays
    iq4 = 0; it3 = 0
    for ei in 1:n_shells
        el = shell_list[ei]
        pid = string(el["PID"])
        if !haskey(pshells, pid); continue; end
        prop = pshells[pid]
        nids = el["NODES"]
        n = length(nids)
        mid = string(prop["MID"])
        if !haskey(mats, mid); continue; end
        is_pcomp_clt = get(prop, "TYPE", "") == "PCOMP_CLT" && haskey(prop, "Cm")
        base_mat = mats[mid]
        mat = is_pcomp_clt ? base_mat : _effective_mat1_for_nodes(model, mid, nids)
        el_theta = deg2rad(Float64(get(el, "THETA", 0.0)))
        mid3 = get(prop, "MID3", 0)
        shear_mat =
            if mid3 != 0 && haskey(mats, string(mid3))
                is_pcomp_clt ? mats[string(mid3)] : _effective_mat1_for_nodes(model, string(mid3), nids)
            else
                mat
            end
        h = prop["T"]
        br = get(prop, "BEND_RATIO", 1.0)
        tst = get(prop, "TS_T", 5.0/6.0)
        pcomp_is_isotropic = is_pcomp_clt && get(prop, "IS_ISOTROPIC", false)
        is_ortho = !is_pcomp_clt && get(mat, "TYPE", "") == "MAT8" && haskey(mat, "E1") && haskey(mat, "E2")
        is_mat2  = !is_pcomp_clt && !is_ortho && get(mat, "TYPE", "") == "MAT2" && haskey(mat, "G11")

        valid = true
        for k in 1:n
            nid = nids[k]
            if nid < 1 || nid > max_nid || id_vec[nid] == 0; valid = false; break; end
        end
        if !valid; continue; end

        local Cm_e::Matrix{Float64}, Cb_e::Matrix{Float64}, Cs_e::Matrix{Float64}
        local Bmb_e::Union{Nothing, Matrix{Float64}}
        local E_ref_e::Float64

        if is_pcomp_clt
            Cm_e = prop["Cm"]; Cb_e = prop["Cb"]; Cs_e = prop["Cs"]
            Bmb_e = get(prop, "Bmb", nothing)
            E_ref_e = n == 4 ? get(prop, "E_ref", mat["E"]) : mat["G"]
            # Store element THETA for per-element material axis rotation
            el_theta_pcomp = el_theta
        elseif is_ortho
            E1 = mat["E1"]; E2 = mat["E2"]; nu12 = mat["NU12"]; G12 = mat["G12"]
            nu21 = nu12 * E2 / max(E1, 1e-30)
            denom = 1.0 - nu12 * nu21
            Q11 = E1/denom; Q22 = E2/denom; Q12 = nu12*E2/denom; Q66 = G12
            Q16 = 0.0; Q26 = 0.0
            if abs(el_theta) > 1e-10
                ct = cos(el_theta); st = sin(el_theta); c2 = ct^2; s2 = st^2
                Q11r = Q11*c2^2 + 2*(Q12+2*Q66)*c2*s2 + Q22*s2^2
                Q22r = Q11*s2^2 + 2*(Q12+2*Q66)*c2*s2 + Q22*c2^2
                Q12r = (Q11+Q22-4*Q66)*c2*s2 + Q12*(c2^2+s2^2)
                Q16 = (Q11-Q12-2*Q66)*ct*st*c2 + (Q12-Q22+2*Q66)*ct*st*s2
                Q26 = (Q11-Q12-2*Q66)*ct*st*s2 + (Q12-Q22+2*Q66)*ct*st*c2
                Q66r = (Q11+Q22-2*Q12-2*Q66)*c2*s2 + Q66*(c2^2+s2^2)
                Q11 = Q11r; Q22 = Q22r; Q12 = Q12r; Q66 = Q66r
            end
            Cm_e = h .* [Q11 Q12 Q16; Q12 Q22 Q26; Q16 Q26 Q66]
            Cb_e = br * (h^3/12.0) .* [Q11 Q12 Q16; Q12 Q22 Q26; Q16 Q26 Q66]
            Cs_e, _ = shell_transverse_shear_matrix(shear_mat, h, tst, el_theta)
            Bmb_e = nothing
            E_ref_e = n == 4 ? max(E1, E2) : G12
        elseif is_mat2
            G11 = mat["G11"]; G12m = mat["G12"]; G13 = mat["G13"]
            G22 = mat["G22"]; G23 = mat["G23"]; G33 = mat["G33"]
            Cm_e = h .* [G11 G12m G13; G12m G22 G23; G13 G23 G33]
            Cb_e = br * (h^3/12.0) .* [G11 G12m G13; G12m G22 G23; G13 G23 G33]
            Cs_e, G_shear_ref = shell_transverse_shear_matrix(shear_mat, h, tst, el_theta)
            Bmb_e = nothing
            E_ref_e = n == 4 ? max(G11, G22) : G_shear_ref
        else
            E_val = mat["E"]; nu_val = mat["NU"]
            const_mem = E_val * h / (1 - nu_val^2)
            Cm_e = const_mem .* [1.0 nu_val 0.0; nu_val 1.0 0.0; 0.0 0.0 (1-nu_val)/2]
            const_bend = br * (E_val * h^3) / (12 * (1 - nu_val^2))
            Cb_e = const_bend .* [1.0 nu_val 0.0; nu_val 1.0 0.0; 0.0 0.0 (1-nu_val)/2]
            Cs_e, G_shear_ref = shell_transverse_shear_matrix(shear_mat, h, tst, el_theta)
            Bmb_e = nothing
            E_ref_e = n == 4 ? E_val : G_shear_ref
        end

        if br <= 1e-12
            # A PSHELL/Pcomp membrane-only shell should not retain transverse shear
            # or drilling stabilization; those artificial out-of-plane terms prevent
            # AUTOSPC from reproducing Nastran's membrane-only mechanism handling.
            fill!(Cs_e, 0.0)
        end

        if n == 4
            iq4 += 1
            i1 = id_vec[nids[1]]; i2 = id_vec[nids[2]]; i3 = id_vec[nids[3]]; i4 = id_vec[nids[4]]
            q4_idx[iq4,1] = i1; q4_idx[iq4,2] = i2; q4_idx[iq4,3] = i3; q4_idx[iq4,4] = i4
            q4_h[iq4] = h; q4_br[iq4] = br; q4_Eref[iq4] = E_ref_e
            q4_eid_int[iq4] = something(tryparse(Int, string(get(el, "ID", 0))), 0)
            q4_pid_int[iq4] = something(tryparse(Int, pid), 0)
            for j in 1:3, i in 1:3
                q4_Cm_flat[i,j,iq4] = Cm_e[i,j]
                q4_Cb_flat[i,j,iq4] = Cb_e[i,j]
            end
            for j in 1:2, i in 1:2
                q4_Cs_flat[i,j,iq4] = Cs_e[i,j]
                q4_Cs_raw_flat[i,j,iq4] = is_pcomp_clt && haskey(prop, "Cs_raw") ? prop["Cs_raw"][i,j] : Cs_e[i,j]
            end
            if Bmb_e !== nothing
                q4_has_Bmb[iq4] = true
                for j in 1:3, i in 1:3; q4_Bmb_flat[i,j,iq4] = Bmb_e[i,j]; end
            end
            if is_pcomp_clt
                q4_is_pcomp[iq4] = true
                q4_ply_data[iq4] = get(prop, "PLY_DATA", nothing)
                q4_is_pcomp_isotropic[iq4] = pcomp_is_isotropic
                q4_pcomp_rigid_shear[iq4] = Bool(get(prop, "TRANSVERSE_SHEAR_RIGID_LIMIT", false))
                q4_pcomp_auto_element_axis_prop[iq4] = q4_sol105_pcomp_auto_membrane_incomp_candidate(prop)
                q4_el_theta[iq4] = el_theta_pcomp
                q4_el_mcid[iq4] = Int(get(el, "MCID", 0))
                q4_pcomp_shear_ratio[iq4], q4_pcomp_d16_ratio[iq4], q4_pcomp_b_ratio[iq4] =
                    pcomp_metric_ratios(prop, el_theta_pcomp)
            else
                q4_ply_data[iq4] = nothing
                q4_is_isotropic[iq4] = !is_ortho && !is_mat2
            end
        elseif n == 3
            it3 += 1
            i1 = id_vec[nids[1]]; i2 = id_vec[nids[2]]; i3 = id_vec[nids[3]]
            t3_idx[it3,1] = i1; t3_idx[it3,2] = i2; t3_idx[it3,3] = i3
            t3_h[it3] = h; t3_br[it3] = br; t3_tst[it3] = tst; t3_Eref[it3] = E_ref_e
            t3_Cm[it3] = Cm_e; t3_Cb[it3] = Cb_e; t3_Cs[it3] = Cs_e; t3_Bmb[it3] = Bmb_e
            if is_pcomp_clt
                t3_is_pcomp[it3] = true
                t3_is_pcomp_isotropic[it3] = pcomp_is_isotropic
                t3_el_theta[it3] = el_theta_pcomp
                t3_el_mcid[it3] = Int(get(el, "MCID", 0))
            else
                t3_is_isotropic[it3] = !is_ortho && !is_mat2
            end
        end
    end

    log_msg("[SOLVER] Pre-extracted $n_q4 QUAD4 + $n_t3 TRIA3 elements (from $n_shells total)")

    # Convert node_R to flat 3D array
    node_R_flat = zeros(3, 3, n_nodes)
    for i in 1:n_nodes
        for r in 1:3, c in 1:3
            node_R_flat[r, c, i] = node_R[i][r, c]
        end
    end
    shell_valence = zeros(Int, n_nodes)
    for ei in 1:n_q4
        shell_valence[q4_idx[ei,1]] += 1
        shell_valence[q4_idx[ei,2]] += 1
        shell_valence[q4_idx[ei,3]] += 1
        shell_valence[q4_idx[ei,4]] += 1
    end
    for ei in 1:n_t3
        shell_valence[t3_idx[ei,1]] += 1
        shell_valence[t3_idx[ei,2]] += 1
        shell_valence[t3_idx[ei,3]] += 1
    end
    q4_static_component_neighbor_ok = trues(n_q4)
    if !isempty(static_component_neighbor_pid_prefixes)
        fill!(q4_static_component_neighbor_ok, false)
        node_pid_sets = [Set{String}() for _ in 1:n_nodes]
        for ei in 1:n_q4
            pid_s = string(q4_pid_int[ei])
            push!(node_pid_sets[q4_idx[ei,1]], pid_s)
            push!(node_pid_sets[q4_idx[ei,2]], pid_s)
            push!(node_pid_sets[q4_idx[ei,3]], pid_s)
            push!(node_pid_sets[q4_idx[ei,4]], pid_s)
        end
        for ei in 1:n_q4
            own_pid_s = string(q4_pid_int[ei])
            ok = false
            for idx in (q4_idx[ei,1], q4_idx[ei,2], q4_idx[ei,3], q4_idx[ei,4])
                for pid_s in node_pid_sets[idx]
                    if pid_s != own_pid_s &&
                       q4_pid_matches_any_prefix(pid_s, static_component_neighbor_pid_prefixes)
                        ok = true
                        break
                    end
                end
                ok && break
            end
            q4_static_component_neighbor_ok[ei] = ok
        end
    end
    # --- PARALLEL QUAD4 ASSEMBLY ---
    per_thread_ws = [FEM.create_quad4_workspace() for _ in 1:nt]
    per_thread_ws_alt = [FEM.create_quad4_workspace() for _ in 1:nt]

    all_I = Vector{Int}(undef, n_q4 * 576)
    all_J = Vector{Int}(undef, n_q4 * 576)
    all_V = Vector{Float64}(undef, n_q4 * 576)

    prev_blas_threads = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)

    sep_T       = [zeros(24,24) for _ in 1:nt]
    sep_tmp     = [zeros(24,24) for _ in 1:nt]
    sep_global  = [zeros(24,24) for _ in 1:nt]
    sep_dofs    = [Vector{Int}(undef, 24) for _ in 1:nt]
    sep_lc      = [zeros(4,2) for _ in 1:nt]
    # Per-thread 4×3 buffer for the 3D corner coords — source for the curved
    # Jacobian integration (fix path #2, env-gated via
    # JFEM_SOL105_EIG_CURVED_JACOBIAN). When the flag is off (default), this
    # buffer is filled but the `coords_3d` kwarg of stiffness_quad4_matrices
    # receives nothing, keeping runtime behavior identical to pre-scaffold.
    sep_coords3d = [zeros(4,3) for _ in 1:nt]
    sep_coords3d_local = [zeros(4,3) for _ in 1:nt]
    sep_directors3d_local = [zeros(4,3) for _ in 1:nt]
    sep_Cm      = [zeros(3,3) for _ in 1:nt]
    sep_Cb      = [zeros(3,3) for _ in 1:nt]
    sep_Cs      = [zeros(2,2) for _ in 1:nt]
    sep_Bmb     = [zeros(3,3) for _ in 1:nt]
    sep_Ke_blend = [zeros(24,24) for _ in 1:nt]
    q4_use_geom_snorm = falses(n_q4)
    if curved_iso_geomnormal_frame && isempty(snorm_normals)
        for ei in 1:n_q4
            q4_is_isotropic[ei] || continue
            i1 = q4_idx[ei,1]; i2 = q4_idx[ei,2]; i3 = q4_idx[ei,3]; i4 = q4_idx[ei,4]
            (geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]) || continue
            p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
            p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
            p3 = SVector{3}(node_coords[i3,1], node_coords[i3,2], node_coords[i3,3])
            p4 = SVector{3}(node_coords[i4,1], node_coords[i4,2], node_coords[i4,3])
            v1p, v2p, v3p = shell_element_frame_quad4(p1, p2, p3, p4, q4_frame_mode)
            c_p = (p1 + p2 + p3 + p4) / 4.0
            lc_p = zeros(4,2)
            lc_p[1,1] = dot(p1-c_p, v1p); lc_p[1,2] = dot(p1-c_p, v2p)
            lc_p[2,1] = dot(p2-c_p, v1p); lc_p[2,2] = dot(p2-c_p, v2p)
            lc_p[3,1] = dot(p3-c_p, v1p); lc_p[3,2] = dot(p3-c_p, v2p)
            lc_p[4,1] = dot(p4-c_p, v1p); lc_p[4,2] = dot(p4-c_p, v2p)
            aspect_ratio_p = q4_local_edge_aspect_ratio(lc_p)
            aspect_ratio_p >= curved_iso_geomnormal_frame_aspect_ratio_min || continue
            (shell_valence[i1] + shell_valence[i2] + shell_valence[i3] + shell_valence[i4]) <= 10 || continue
            d13_p = p3 - p1
            d24_p = p4 - p2
            v3_geom_raw_p = cross(d13_p, d24_p)
            v3_geom_len_p = norm(v3_geom_raw_p)
            elem_is_flat_p = true
            if v3_geom_len_p > 1e-12
                v3g_p = v3_geom_raw_p / v3_geom_len_p
                max_dev_p = max(abs(dot(p1-c_p, v3g_p)), abs(dot(p2-c_p, v3g_p)),
                                abs(dot(p3-c_p, v3g_p)), abs(dot(p4-c_p, v3g_p)))
                L_diag_p = max(norm(d13_p), norm(d24_p))
                elem_is_flat_p = max_dev_p < 1e-6 * max(L_diag_p, 1e-12)
            end
            geom_curv_p = estimate_quad4_curvature_membrane(
                lc_p, geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4], v1p, v2p, v3p
            )
            k1_p, _ = q4_curvature_principal_abs(geom_curv_p)
            kappa_l_p = k1_p * q4_curvature_characteristic_length(lc_p)
            cyl_ratio_p = q4_curvature_cyl_ratio(geom_curv_p)
            if elem_is_flat_p
                if (shell_valence[i1] + shell_valence[i2] + shell_valence[i3] + shell_valence[i4]) <= 10 &&
                   kappa_l_p >= flat_curved_iso_geomnormal_frame_kappa_l_min &&
                   kappa_l_p <= flat_curved_iso_geomnormal_frame_kappa_l_max &&
                   cyl_ratio_p <= flat_curved_iso_geomnormal_frame_cyl_ratio_max
                    q4_use_geom_snorm[ei] = true
                end
            elseif kappa_l_p >= curved_iso_geomnormal_frame_kappa_l_min &&
                   kappa_l_p <= curved_iso_geomnormal_frame_kappa_l_max &&
                   cyl_ratio_p <= curved_iso_geomnormal_frame_cyl_ratio_max
                q4_use_geom_snorm[ei] = true
            end
        end
    end
    Threads.@threads :static for ei in 1:n_q4
        tid = Threads.threadid()

        i1 = q4_idx[ei,1]; i2 = q4_idx[ei,2]; i3 = q4_idx[ei,3]; i4 = q4_idx[ei,4]

        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
        p3 = SVector{3}(node_coords[i3,1], node_coords[i3,2], node_coords[i3,3])
        p4 = SVector{3}(node_coords[i4,1], node_coords[i4,2], node_coords[i4,3])

        v1, v2, v3 = shell_element_frame_quad4(p1, p2, p3, p4, q4_frame_mode)
        elem_use_geom_snorm = q4_use_geom_snorm[ei]

        # SNORM adjustment
        n_avg = SVector(0.0, 0.0, 0.0); nc = 0
        for idx in (i1, i2, i3, i4)
            if elem_use_geom_snorm
                if geom_has[idx]; n_avg = n_avg + geom_vec[idx]; nc += 1; end
            else
                if snorm_has[idx]; n_avg = n_avg + snorm_vec[idx]; nc += 1; end
            end
        end
        if nc > 0
            n_avg_s = n_avg / nc; len_s = norm(n_avg_s)
            if len_s > 1e-12
                v3n = SVector{3}(n_avg_s / len_s)
                if dot(v3n, v3) < 0.0; v3n = -v3n; end
                v1p = v1 - dot(v1, v3n) * v3n; v1l = norm(v1p)
                if v1l > 1e-12
                    v1n = SVector{3}(v1p / v1l)
                else
                    v2p = v2 - dot(v2, v3n) * v3n; v1n = SVector{3}(normalize(v2p))
                end
                v1, v2, v3 = v1n, SVector{3}(cross(v3n, v1n)), v3n
            end
        end
        lc = sep_lc[tid]
        c = (p1 + p2 + p3 + p4) / 4.0
        lc[1,1] = dot(p1-c, v1); lc[1,2] = dot(p1-c, v2)
        lc[2,1] = dot(p2-c, v1); lc[2,2] = dot(p2-c, v2)
        lc[3,1] = dot(p3-c, v1); lc[3,2] = dot(p3-c, v2)
        lc[4,1] = dot(p4-c, v1); lc[4,2] = dot(p4-c, v2)
        c3d_local = sep_coords3d_local[tid]
        dirs_local = sep_directors3d_local[tid]
        c3d_local[1,1] = lc[1,1]; c3d_local[1,2] = lc[1,2]; c3d_local[1,3] = dot(p1-c, v3)
        c3d_local[2,1] = lc[2,1]; c3d_local[2,2] = lc[2,2]; c3d_local[2,3] = dot(p2-c, v3)
        c3d_local[3,1] = lc[3,1]; c3d_local[3,2] = lc[3,2]; c3d_local[3,3] = dot(p3-c, v3)
        c3d_local[4,1] = lc[4,1]; c3d_local[4,2] = lc[4,2]; c3d_local[4,3] = dot(p4-c, v3)
        aspect_ratio_ei = q4_local_edge_aspect_ratio(lc)
        taper_ratio_ei = q4_local_opposite_edge_ratio(lc)
        edge_skew_ei = q4_local_edge_skew_angle(lc)

        # Curved-Jacobian scaffold (fix path #2). Fill the per-thread 4x3 buffer
        # only when the curved-Jacobian path is active.
        coords_3d_arg = nothing
        if curved_jacobian_enabled
            c3d = sep_coords3d[tid]
            c3d[1,1] = p1[1]; c3d[1,2] = p1[2]; c3d[1,3] = p1[3]
            c3d[2,1] = p2[1]; c3d[2,2] = p2[2]; c3d[2,3] = p2[3]
            c3d[3,1] = p3[1]; c3d[3,2] = p3[2]; c3d[3,3] = p3[3]
            c3d[4,1] = p4[1]; c3d[4,2] = p4[2]; c3d[4,3] = p4[3]
            coords_3d_arg = c3d
        end
        curvature_membrane = nothing
        elem_mitc4_3d_candidate =
            mitc4_3d_all_kernel ||
            (mitc4_3d_aspect_kernel &&
             aspect_ratio_ei >= mitc4_3d_aspect_min &&
             aspect_ratio_ei <= mitc4_3d_aspect_max)
        mitc4_3d_use_geom_dirs =
            elem_mitc4_3d_candidate && geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
        marguerre_static_use_geom_dirs =
            marguerre_static_use_geom_normals && geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
        static_curvature_membrane_use_geom_dirs =
            static_curvature_membrane_geom_normals && geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
        has_curv_normals = elem_use_geom_snorm || mitc4_3d_use_geom_dirs ||
                           marguerre_static_use_geom_dirs ||
                           static_curvature_membrane_use_geom_dirs ||
                           (snorm_has[i1] && snorm_has[i2] && snorm_has[i3] && snorm_has[i4])
        use_geom_curv_dirs =
            elem_use_geom_snorm ||
            mitc4_3d_use_geom_dirs ||
            marguerre_static_use_geom_dirs ||
            static_curvature_membrane_use_geom_dirs
        n1_curv = use_geom_curv_dirs ? geom_vec[i1] : snorm_vec[i1]
        n2_curv = use_geom_curv_dirs ? geom_vec[i2] : snorm_vec[i2]
        n3_curv = use_geom_curv_dirs ? geom_vec[i3] : snorm_vec[i3]
        n4_curv = use_geom_curv_dirs ? geom_vec[i4] : snorm_vec[i4]
        if has_curv_normals
            for (row, ncurv) in enumerate((n1_curv, n2_curv, n3_curv, n4_curv))
                nloc = SVector(dot(ncurv, v1), dot(ncurv, v2), dot(ncurv, v3))
                if nloc[3] < 0.0
                    nloc = -nloc
                end
                nlen = norm(nloc)
                if nlen > 1e-12
                    dirs_local[row,1] = nloc[1] / nlen
                    dirs_local[row,2] = nloc[2] / nlen
                    dirs_local[row,3] = nloc[3] / nlen
                else
                    dirs_local[row,1] = 0.0; dirs_local[row,2] = 0.0; dirs_local[row,3] = 1.0
                end
            end
        else
            for row in 1:4
                dirs_local[row,1] = 0.0; dirs_local[row,2] = 0.0; dirs_local[row,3] = 1.0
            end
        end
        curvature_scale = q4_curvature_membrane_scale(shear_center_only ? "JFEM_Q4_CURVATURE_MEMBRANE_SCALE_EIG" : "JFEM_Q4_CURVATURE_MEMBRANE_SCALE_STATIC")
        if curvature_scale > 0.0 && has_curv_normals
            curvature_raw = estimate_quad4_curvature_membrane(
                lc, n1_curv, n2_curv, n3_curv, n4_curv, v1, v2, v3
            )
            curvature_filter_key = shear_center_only ? "JFEM_Q4_CURVATURE_FILTER_MODE_EIG" : "JFEM_Q4_CURVATURE_FILTER_MODE_STATIC"
            ratio_key = shear_center_only ? "JFEM_Q4_CURVATURE_CYL_RATIO_MAX_EIG" : "JFEM_Q4_CURVATURE_CYL_RATIO_MAX_STATIC"
            curvature_weight = q4_curvature_filter_weight(
                curvature_raw,
                q4_curvature_filter_mode(curvature_filter_key),
                q4_curvature_cyl_ratio_max(ratio_key),
            )
            resolution_min_key = shear_center_only ? "JFEM_Q4_CURVATURE_RESOLUTION_MIN_EIG" : "JFEM_Q4_CURVATURE_RESOLUTION_MIN_STATIC"
            resolution_full_key = shear_center_only ? "JFEM_Q4_CURVATURE_RESOLUTION_FULL_EIG" : "JFEM_Q4_CURVATURE_RESOLUTION_FULL_STATIC"
            curvature_weight *= q4_curvature_resolution_weight(
                curvature_raw, lc,
                q4_curvature_resolution_min(resolution_min_key),
                q4_curvature_resolution_full(resolution_full_key),
            )
            if curvature_weight > 0.0
                curvature_membrane = curvature_raw * (curvature_scale * curvature_weight)
            end
        end

        slope_membrane = nothing
        if marguerre_coupling_enabled && has_curv_normals
            # Ibrahimbegović 1994 Eq. 6.14 rotation-column coupling. Geometric
            # slopes derived from SNORM-averaged nodal normals via small-angle
            # projection onto the element tangent plane. No tunable scale needed
            # unless explicitly overridden (scale default 1.0).
            slope_raw = estimate_quad4_slope_membrane(
                n1_curv, n2_curv, n3_curv, n4_curv, v1, v2, v3
            )
            if marguerre_coupling_scale != 1.0
                slope_raw = slope_raw * marguerre_coupling_scale
            end
            # Encode convention as trailing element: 0 = jfem_kl, 1 = handover
            slope_membrane = SVector{9,Float64}(
                slope_raw[1], slope_raw[2], slope_raw[3], slope_raw[4],
                slope_raw[5], slope_raw[6], slope_raw[7], slope_raw[8],
                marguerre_handover_marker,
            )
        end

        Cm_local = sep_Cm[tid]; Cb_local = sep_Cb[tid]; Cs_local = sep_Cs[tid]
        for j in 1:3, i in 1:3; Cm_local[i,j] = q4_Cm_flat[i,j,ei]; Cb_local[i,j] = q4_Cb_flat[i,j,ei]; end
        for j in 1:2, i in 1:2; Cs_local[i,j] = q4_Cs_flat[i,j,ei]; end
        Bmb_local = nothing
        if q4_has_Bmb[ei]
            Bmb_local = sep_Bmb[tid]
            for j in 1:3, i in 1:3; Bmb_local[i,j] = q4_Bmb_flat[i,j,ei]; end
        end

        ws_stiff = per_thread_ws[tid]
        # Element-adaptive shear integration for K_eig (shear_center_only=true globally):
        #   Flat orthotropic PCOMP: center-only MITC4 shear to avoid over-stiff thin-laminate buckling modes
        #   Flat isotropic / isotropic PCOMP: normal 2×2 MITC4 + phi2
        #   Curved orthotropic PCOMP: full 2×2 MITC4 (center-only is too soft)
        #   Curved isotropic / isotropic PCOMP: blend center-only with some full MITC4
        # Static K always uses global settings unchanged.
        # Planarity check using GEOMETRIC normal (not SNORM — SNORM is surface-tangent on curved shells,
        # causing dot(pi-c, v3_snorm)≈0 even for curved elements). Diagonal cross product gives true normal.
        d13_geom = p3 - p1; d24_geom = p4 - p2
        v3_geom_raw = cross(d13_geom, d24_geom)
        v3_geom_len = norm(v3_geom_raw)
        local max_dev_ei::Float64
        if v3_geom_len > 1e-12
            v3g = v3_geom_raw / v3_geom_len
            max_dev_ei = max(abs(dot(p1-c, v3g)), abs(dot(p2-c, v3g)),
                             abs(dot(p3-c, v3g)), abs(dot(p4-c, v3g)))
        else
            max_dev_ei = 0.0
        end
        L_diag_ei  = max(norm(d13_geom), norm(d24_geom))  # diagonal length (≈ √2 × edge)
        elem_is_flat = max_dev_ei < 1e-6 * max(L_diag_ei, 1e-12)
        warp_ratio_ei = max_dev_ei / max(L_diag_ei, 1e-12)
        # MacNeal-permissive planarity (2026-04-30): the flat MacNeal RBF kernel
        # works correctly on mildly-warped quads as long as warp_ratio is small.
        # The strict `elem_is_flat` test (1e-6 of L) classifies almost all real
        # aerodynamic-mesh elements as non-flat, sending them onto the inferior
        # legacy MITC path. Setting the MacNeal eligibility threshold to 1e-4
        # (0.01% warp) recovers HTP_3wp_disp 511002 from 85.9% rel error to
        # 1.87% with no regression on HTP_launch (2.86% unchanged). Strongly-
        # curved meshes like VTP still need a real curved-aware MacNeal kernel.
        macneal_warp_tol = max(solver_env_float("JFEM_Q4_MACNEAL_WARP_TOL", 1e-4), 1e-12)
        # Aspect-ratio gate (2026-05-01): research switch, default off (1e30).
        # Distribution analysis on the GAME meshes showed HTP_launch p90 aspect
        # ratio 10.4 (max 24) vs VTP_launch p99 only 7.7, suggesting a possible
        # discriminator between HTP-breaking and VTP-improving curved elements
        # under MacNeal. Empirical sweep with `JFEM_Q4_MACNEAL_ASPECT_MAX=10`
        # (combined with `KAPPA_L_MAX=1.0`) did NOT cleanly recover the HTP/VTP
        # trade-off — aspect ratio alone is insufficient. Left as an explicit
        # switch for further per-element classifier work.
        macneal_aspect_max = max(solver_env_float("JFEM_Q4_MACNEAL_ASPECT_MAX", 1e30), 1.0)
        elem_is_macneal_eligible = warp_ratio_ei < macneal_warp_tol &&
                                   aspect_ratio_ei <= macneal_aspect_max
        is_pcomp_ei = q4_is_pcomp[ei]
        is_pcomp_iso_ei = q4_is_pcomp_isotropic[ei]
        is_iso_ei = q4_is_isotropic[ei] || is_pcomp_iso_ei
        if flat_curved_iso_geomnormal_frame &&
           q4_is_isotropic[ei] &&
           elem_is_flat &&
           aspect_ratio_ei >= flat_curved_iso_geomnormal_frame_aspect_ratio_min &&
           (shell_valence[i1] + shell_valence[i2] + shell_valence[i3] + shell_valence[i4]) <= 10 &&
           geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
            iso_geom_curvature_probe = estimate_quad4_curvature_membrane(
                lc, geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4], v1, v2, v3
            )
            k1_probe, _ = q4_curvature_principal_abs(iso_geom_curvature_probe)
            kappa_l_probe = k1_probe * q4_curvature_characteristic_length(lc)
            cyl_ratio_probe = q4_curvature_cyl_ratio(iso_geom_curvature_probe)
            if kappa_l_probe >= flat_curved_iso_geomnormal_frame_kappa_l_min &&
               kappa_l_probe <= flat_curved_iso_geomnormal_frame_kappa_l_max &&
               cyl_ratio_probe <= flat_curved_iso_geomnormal_frame_cyl_ratio_max
                v3_geom_sum = geom_vec[i1] + geom_vec[i2] + geom_vec[i3] + geom_vec[i4]
                if norm(v3_geom_sum) > 1e-12
                    v3_geom_frame = normalize(v3_geom_sum)
                    if dot(v3_geom_frame, v3) < 0.0
                        v3_geom_frame = -v3_geom_frame
                    end
                    v1, v2, v3 = shell_element_frame_quad4_with_normal(
                        p1, p2, p3, p4, v3_geom_frame, q4_frame_mode
                    )
                    lc[1,1] = dot(p1-c, v1); lc[1,2] = dot(p1-c, v2)
                    lc[2,1] = dot(p2-c, v1); lc[2,2] = dot(p2-c, v2)
                    lc[3,1] = dot(p3-c, v1); lc[3,2] = dot(p3-c, v2)
                    lc[4,1] = dot(p4-c, v1); lc[4,2] = dot(p4-c, v2)
                    aspect_ratio_ei = q4_local_edge_aspect_ratio(lc)
                    taper_ratio_ei = q4_local_opposite_edge_ratio(lc)
                end
            end
        end
        if curved_iso_geomnormal_frame &&
           q4_is_isotropic[ei] &&
           !elem_is_flat &&
           aspect_ratio_ei >= curved_iso_geomnormal_frame_aspect_ratio_min &&
           (shell_valence[i1] + shell_valence[i2] + shell_valence[i3] + shell_valence[i4]) <= 10 &&
           geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
            iso_geom_curvature_probe = estimate_quad4_curvature_membrane(
                lc, geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4], v1, v2, v3
            )
            k1_probe, _ = q4_curvature_principal_abs(iso_geom_curvature_probe)
            kappa_l_probe = k1_probe * q4_curvature_characteristic_length(lc)
            cyl_ratio_probe = q4_curvature_cyl_ratio(iso_geom_curvature_probe)
            if kappa_l_probe >= curved_iso_geomnormal_frame_kappa_l_min &&
               kappa_l_probe <= curved_iso_geomnormal_frame_kappa_l_max &&
               cyl_ratio_probe <= curved_iso_geomnormal_frame_cyl_ratio_max
                v3_geom_sum = geom_vec[i1] + geom_vec[i2] + geom_vec[i3] + geom_vec[i4]
                if norm(v3_geom_sum) > 1e-12
                    v3_geom_frame = normalize(v3_geom_sum)
                    if dot(v3_geom_frame, v3) < 0.0
                        v3_geom_frame = -v3_geom_frame
                    end
                    v1, v2, v3 = shell_element_frame_quad4_with_normal(
                        p1, p2, p3, p4, v3_geom_frame, q4_frame_mode
                    )
                    lc[1,1] = dot(p1-c, v1); lc[1,2] = dot(p1-c, v2)
                    lc[2,1] = dot(p2-c, v1); lc[2,2] = dot(p2-c, v2)
                    lc[3,1] = dot(p3-c, v1); lc[3,2] = dot(p3-c, v2)
                    lc[4,1] = dot(p4-c, v1); lc[4,2] = dot(p4-c, v2)
                    aspect_ratio_ei = q4_local_edge_aspect_ratio(lc)
                    taper_ratio_ei = q4_local_opposite_edge_ratio(lc)
                end
            end
        end
        pcomp_geom_curvature = nothing
        iso_geom_curvature = nothing
        iso_corner_curvature = nothing
        if is_pcomp_ei && geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4] &&
           (flat_pcomp_auto_phi2 || flat_pcomp_auto_g12 || pcomp_auto_global_x ||
            q4_kernel_needs_surface_flatness || q4_macneal_curved_bending_enabled ||
            q4_macneal_bending_aspect_enabled || q4_macneal_bending_aspect2_enabled)
            pcomp_geom_curvature = estimate_quad4_curvature_membrane(
                lc, geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4], v1, v2, v3
            )
        end
        if (curved_iso_eig_membrane_incomp || flat_curved_iso_eig_center_only) &&
           q4_is_isotropic[ei] &&
           geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
            iso_geom_curvature = estimate_quad4_curvature_membrane(
                lc, geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4], v1, v2, v3
            )
        end
        if flat_iso_dkmq_branch && q4_is_isotropic[ei] && !elem_is_flat
            iso_corner_curvature = estimate_quad4_corner_curvature_membrane(
                lc, p1, p2, p3, p4, v1, v2, v3
            )
        end
        auto_curved_iso_membrane_incomp = false
        auto_warped_iso_membrane_incomp = false
        auto_elongated_iso_membrane_incomp = false
        flat_curved_iso_center_candidate = false
        kappa_l_iso = 0.0
        cyl_ratio_iso = 1.0
        if iso_geom_curvature !== nothing && q4_is_isotropic[ei]
            k1_iso, _ = q4_curvature_principal_abs(iso_geom_curvature)
            kappa_l_iso = k1_iso * q4_curvature_characteristic_length(lc)
            cyl_ratio_iso = q4_curvature_cyl_ratio(iso_geom_curvature)
            auto_curved_iso_membrane_incomp =
                curved_iso_eig_membrane_incomp &&
                kappa_l_iso >= curved_iso_eig_membrane_incomp_kappa_l_min &&
                cyl_ratio_iso <= curved_iso_eig_membrane_incomp_cyl_ratio_max
            auto_warped_iso_membrane_incomp =
                curved_iso_warp_membrane_incomp &&
                !elem_is_flat &&
                warp_ratio_ei >= curved_iso_warp_membrane_incomp_ratio_min &&
                kappa_l_iso <= curved_iso_warp_membrane_incomp_kappa_l_max
            auto_elongated_iso_membrane_incomp =
                curved_iso_elongated_membrane_incomp &&
                !elem_is_flat &&
                aspect_ratio_ei >= curved_iso_elongated_membrane_incomp_aspect_ratio_min
            flat_curved_iso_center_candidate =
                flat_curved_iso_eig_center_only && elem_is_flat &&
                kappa_l_iso >= flat_curved_iso_eig_center_only_kappa_l_min &&
                cyl_ratio_iso <= flat_curved_iso_eig_center_only_cyl_ratio_max
        end
        elem_k6rot = q4_br[ei] <= 1e-12 ? 0.0 : k6rot
        if q4_br[ei] <= 1e-12
            elem_k6rot = 0.0
        elseif shear_center_only &&
           model_has_line_elements &&
           elem_is_flat &&
           q4_is_isotropic[ei] &&
           !flat_curved_iso_center_candidate
            elem_k6rot = 0.0
        elseif iso_static_k6rot_override !== nothing &&
               q4_is_isotropic[ei] &&
               q4_br[ei] > 1e-12
            elem_k6rot = max(0.0, iso_static_k6rot_override)
        elseif iso_eig_k6rot_override !== nothing &&
               q4_is_isotropic[ei] &&
               q4_br[ei] > 1e-12 &&
               !flat_curved_iso_center_candidate &&
               (iso_geom_curvature === nothing || cyl_ratio_iso >= iso_eig_k6rot_cyl_ratio_min)
            elem_k6rot = max(0.0, iso_eig_k6rot_override)
        end
        elem_drill_scale = q4_br[ei] <= 1e-12 ? 0.0 : 1.0
        if q4_br[ei] <= 1e-12
            elem_drill_scale = 0.0
        elseif iso_static_drill_scale_override !== nothing &&
           q4_is_isotropic[ei] &&
           q4_br[ei] > 1e-12
            elem_drill_scale = iso_static_drill_scale_override
        elseif iso_eig_drill_scale_override !== nothing &&
           q4_is_isotropic[ei] &&
           q4_br[ei] > 1e-12 &&
           !flat_curved_iso_center_candidate &&
           (iso_geom_curvature === nothing || cyl_ratio_iso >= iso_eig_drill_scale_cyl_ratio_min)
            elem_drill_scale = iso_eig_drill_scale_override
        end
        # Flat elements on a smoothly curved shell patch can still need curved-shell
        # buckling treatment in K_eig, especially on faceted cylinders.
        flat_pcomp_h_over_l = q4_h[ei] / max(q4_curvature_characteristic_length(lc), 1e-12)
        flat_curved_pcomp_fullshear_candidate = false
        if flat_curved_pcomp_fullshear &&
           elem_is_flat &&
           is_pcomp_ei &&
           !is_pcomp_iso_ei &&
           Bmb_local === nothing &&
           pcomp_geom_curvature !== nothing
            k1_pcomp, _ = q4_curvature_principal_abs(pcomp_geom_curvature)
            kappa_l_pcomp = k1_pcomp * q4_curvature_characteristic_length(lc)
            cyl_ratio_pcomp = q4_curvature_cyl_ratio(pcomp_geom_curvature)
            flat_curved_pcomp_fullshear_candidate =
                kappa_l_pcomp >= flat_curved_pcomp_fullshear_kappa_l_min &&
                cyl_ratio_pcomp <= flat_curved_pcomp_fullshear_cyl_ratio_max
        end
        if flat_curved_pcomp_geomnormal_frame &&
           flat_curved_pcomp_fullshear_candidate &&
           geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
            v3_geom_sum = geom_vec[i1] + geom_vec[i2] + geom_vec[i3] + geom_vec[i4]
            if norm(v3_geom_sum) > 1e-12
                v3_geom_frame = normalize(v3_geom_sum)
                if dot(v3_geom_frame, v3) < 0.0
                    v3_geom_frame = -v3_geom_frame
                end
                v1, v2, v3 = shell_element_frame_quad4_with_normal(
                    p1, p2, p3, p4, v3_geom_frame, q4_frame_mode
                )
                lc[1,1] = dot(p1-c, v1); lc[1,2] = dot(p1-c, v2)
                lc[2,1] = dot(p2-c, v1); lc[2,2] = dot(p2-c, v2)
                lc[3,1] = dot(p3-c, v1); lc[3,2] = dot(p3-c, v2)
                lc[4,1] = dot(p4-c, v1); lc[4,2] = dot(p4-c, v2)
                aspect_ratio_ei = q4_local_edge_aspect_ratio(lc)
                taper_ratio_ei = q4_local_opposite_edge_ratio(lc)
                pcomp_geom_curvature = estimate_quad4_curvature_membrane(
                    lc, geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4], v1, v2, v3
                )
            end
        end
        if curvature_membrane === nothing &&
           shear_center_only &&
           flat_curved_pcomp_fullshear_candidate &&
           pcomp_geom_curvature !== nothing &&
           q4_pcomp_d16_ratio[ei] >= q4_pcomp_kg_auto_curvature_d16_ratio_min() &&
           q4_curvature_cyl_ratio(pcomp_geom_curvature) >= 0.2
            # Curved composite facets need the actual geometric membrane-curvature
            # coupling in K_eig. Reusing the globally downscaled shell curvature
            # factor underestimates this term and leaves the curved laminate
            # buckling modes too soft.
            curvature_membrane = pcomp_geom_curvature
        end
        # MacNeal eligibility uses the looser warp-tolerance gate so users can
        # opt mildly-curved elements onto the MacNeal RBF path without changing
        # `elem_is_flat`, which other heuristics still key off of.
        elem_kernel_planar = elem_is_macneal_eligible
        if q4_kernel_needs_surface_flatness &&
           elem_is_macneal_eligible &&
           is_pcomp_ei &&
           !is_pcomp_iso_ei &&
           pcomp_geom_curvature !== nothing
            k1_kernel, _ = q4_curvature_principal_abs(pcomp_geom_curvature)
            elem_kernel_planar =
                k1_kernel * q4_curvature_characteristic_length(lc) <=
                q4_macneal_pcomp_surface_kappa_l_max
        end
        # JFEM_Q4_MACNEAL_RIGID_SHEAR_FORCE: research switch (default false)
        # introduced 2026-05-12 after Nastran reverse-engineering showed that
        # disabling the residual-bending-flexibility on transverse shear
        # (i.e., applying rigid_shear) recovers ~92-108% of Nastran's
        # bending+shear block on the single-element probe ladder. Set this
        # env to bypass the PCOMP-only / MAT8-blank gate and apply
        # macneal_rigid_shear on every macneal-eligible Q4. Tested on
        # GAME with TBD outcome.
        elem_macneal_rigid_shear =
            (mat8_blank_ts_rigid_limit &&
             elem_is_flat &&
             elem_kernel_planar &&
             is_pcomp_ei &&
             !is_pcomp_iso_ei &&
             q4_pcomp_rigid_shear[ei] &&
             (q4_kernel_mode_static in ("macneal", "macneal_pcomp", "macneal-pcomp", "macneal_aniso",
                                        "mitc4_3d_aspect", "mitc4-3d-aspect", "mitc3d_aspect", "mitc3d-aspect"))) ||
            (solver_env_bool("JFEM_Q4_MACNEAL_RIGID_SHEAR_FORCE", false) &&
             elem_is_macneal_eligible &&
             (q4_kernel_mode_static in ("macneal", "macneal_pcomp", "macneal-pcomp", "macneal_aniso",
                                        "macneal_all", "mitc4_3d_aspect", "mitc4-3d-aspect",
                                        "mitc3d_aspect", "mitc3d-aspect")))
        flat_pcomp_reduced_shear = elem_is_flat &&
                                   is_pcomp_ei &&
                                   !is_pcomp_iso_ei &&
                                   !elem_macneal_rigid_shear &&
                                   !flat_curved_pcomp_fullshear_candidate &&
                                   flat_pcomp_h_over_l <= q4_sol105_flat_pcomp_center_only_h_over_l_max()
        if shear_center_only && elem_is_flat && is_pcomp_ei && !is_pcomp_iso_ei
            # Use the exact ply-integrated laminate transverse shear matrix in flat
            # SOL105 eigen stiffness. The generic 5/6 Timoshenko correction is an
            # isotropic surrogate and was depressing the physical first modes of the
            # elementary flat laminate buckling decks.
            use_exact_flat_dkmq =
                flat_pcomp_dkmq_branch &&
                Bmb_local === nothing &&
                maximum(abs, Cb_local) > 1e-30
            if use_exact_flat_dkmq
                # Keep the experimental exact DKMQ branch analytical:
                # use the directly integrated laminate transverse shear matrix
                # without the empirical SOL105 auto shear scaling.
                @inbounds for j in 1:2, i in 1:2
                    Cs_local[i, j] = q4_Cs_raw_flat[i, j, ei]
                end
            else
                shear_scale_eff = flat_pcomp_shear_scale
                if shear_scale_eff == 1.0 &&
                   flat_pcomp_auto_shear_scale &&
                   !flat_pcomp_reduced_shear &&
                   Bmb_local === nothing
                    phi_med = q4_flat_pcomp_phi_metric(lc, Cb_local, q4_Cs_raw_flat[:, :, ei])
                    phi_weight = phi_med / (1.0 + max(phi_med, 0.0))
                    shear_scale_eff = min(
                        flat_pcomp_auto_shear_scale_max,
                        1.0 + flat_pcomp_auto_shear_scale_gain * phi_weight,
                    )
                end
                @inbounds for j in 1:2, i in 1:2
                    Cs_local[i, j] = q4_Cs_raw_flat[i, j, ei] * shear_scale_eff
                end
            end
        end
        # Experimental: use exact ply-integrated Cs on NON-flat curved PCOMP too.
        # Investigating HTP_launch residual. Default off.
        if nonflat_pcomp_exact_cs_enabled &&
           shear_center_only && !elem_is_flat && is_pcomp_ei && !is_pcomp_iso_ei
            @inbounds for j in 1:2, i in 1:2
                Cs_local[i, j] = q4_Cs_raw_flat[i, j, ei]
            end
        end
        elem_shear_center_only = shear_center_only && (
            !elem_is_flat ||
            flat_curved_iso_center_candidate ||
            flat_pcomp_reduced_shear
        )
        flat_pcomp_no_phi2 = isnothing(flat_pcomp_no_phi2_override) ? true : flat_pcomp_no_phi2_override
        if flat_pcomp_auto_phi2 && elem_is_flat && is_pcomp_ei && !is_pcomp_iso_ei &&
           pcomp_geom_curvature !== nothing
            k1, _ = q4_curvature_principal_abs(pcomp_geom_curvature)
            kappa_l = k1 * q4_curvature_characteristic_length(lc)
            cyl_ratio = q4_curvature_cyl_ratio(pcomp_geom_curvature)
            if kappa_l >= flat_pcomp_auto_kappa_l_min &&
               cyl_ratio <= flat_pcomp_auto_cyl_ratio_max &&
               q4_pcomp_shear_ratio[ei] <= flat_pcomp_auto_shear_ratio_max &&
               q4_pcomp_d16_ratio[ei] <= flat_pcomp_auto_d16_ratio_max &&
               q4_pcomp_b_ratio[ei] <= flat_pcomp_auto_b_ratio_max
                flat_pcomp_no_phi2 = false
            end
        end
        auto_pcomp_element_axis = false
        if is_pcomp_ei && !is_pcomp_iso_ei && q4_pcomp_auto_element_axis_prop[ei] &&
           pcomp_geom_curvature !== nothing
            k1, _ = q4_curvature_principal_abs(pcomp_geom_curvature)
            kappa_l = k1 * q4_curvature_characteristic_length(lc)
            cyl_ratio = q4_curvature_cyl_ratio(pcomp_geom_curvature)
            auto_pcomp_element_axis =
                q4_sol105_pcomp_auto_element_axis_candidate(
                    q4_pcomp_auto_element_axis_prop[ei],
                    kappa_l,
                    cyl_ratio,
                )
        end
        pcomp_axis_mode_eff = pcomp_axis_mode
        if flat_pcomp_auto_g12 && is_pcomp_ei && !is_pcomp_iso_ei &&
           q4_pcomp_shear_ratio[ei] <= pcomp_auto_global_x_shear_ratio_max &&
           q4_pcomp_d16_ratio[ei] <= pcomp_auto_global_x_d16_ratio_max &&
           q4_pcomp_b_ratio[ei] <= pcomp_auto_global_x_b_ratio_max
            pcomp_axis_mode_eff = :g12
        end
        if is_pcomp_ei && pcomp_auto_global_x && !is_pcomp_iso_ei && !elem_is_flat &&
           !auto_pcomp_element_axis &&
           q4_pcomp_shear_ratio[ei] <= pcomp_auto_global_x_shear_ratio_max &&
           q4_pcomp_d16_ratio[ei] <= pcomp_auto_global_x_d16_ratio_max &&
           q4_pcomp_b_ratio[ei] <= pcomp_auto_global_x_b_ratio_max
            if pcomp_auto_global_x_kappa_l_min <= 0.0 && pcomp_auto_global_x_cyl_ratio_min <= 0.0
                pcomp_axis_mode_eff = :global_x
            elseif pcomp_geom_curvature !== nothing
                k1, _ = q4_curvature_principal_abs(pcomp_geom_curvature)
                kappa_l = k1 * q4_curvature_characteristic_length(lc)
                cyl_ratio = q4_curvature_cyl_ratio(pcomp_geom_curvature)
                if kappa_l >= pcomp_auto_global_x_kappa_l_min &&
                   cyl_ratio >= pcomp_auto_global_x_cyl_ratio_min
                    pcomp_axis_mode_eff = :global_x
                end
            end
        end
        elem_material_shear_rotation = 0.0
        if is_pcomp_ei
            # Pass all 4 corners so the rotation function can resolve
            # :warp_switch and keep the K_eig rotation consistent with the
            # Kg rotation for the same element (no 2- vs 4-corner asymmetry).
            beta = shell_pcomp_material_rotation(
                pcomp_axis_mode_eff,
                v1, v2, v3, p1, p2, p3, p4,
                q4_el_theta[ei],
                q4_el_mcid[ei],
                model["CORDs"],
            )
            elem_material_shear_rotation = beta
            if abs(beta) > 1e-10
                cb = cos(beta); sb = sin(beta)
                c2 = cb^2; s2 = sb^2; cs = cb*sb
                T11 = c2;  T12 = s2;  T13 = cs
                T21 = s2;  T22 = c2;  T23 = -cs
                T31 = -2cs; T32 = 2cs; T33 = c2 - s2
                _rotate_constitutive_3x3!(Cm_local, T11, T12, T13, T21, T22, T23, T31, T32, T33)
                _rotate_constitutive_3x3!(Cb_local, T11, T12, T13, T21, T22, T23, T31, T32, T33)
                a11 = Cs_local[1,1]; a12 = Cs_local[1,2]; a22 = Cs_local[2,2]
                Cs_local[1,1] = cb^2*a11 + 2*cb*sb*a12 + sb^2*a22
                Cs_local[1,2] = -cb*sb*a11 + (cb^2-sb^2)*a12 + cb*sb*a22
                Cs_local[2,1] = Cs_local[1,2]
                Cs_local[2,2] = sb^2*a11 - 2*cb*sb*a12 + cb^2*a22
                if Bmb_local !== nothing
                    _rotate_constitutive_3x3!(Bmb_local, T11, T12, T13, T21, T22, T23, T31, T32, T33)
                end
            end
        end
        elem_no_phi2 = shear_center_only && elem_is_flat && is_pcomp_ei && !is_pcomp_iso_ei && flat_pcomp_no_phi2
        # Use the exact pointwise membrane field for flat orthotropic laminates.
        # The constitutive rotation already puts the element into the material
        # frame, so no additional center-row shear projection is applied here.
        elem_membrane_shear_center_row =
            flat_iso_eig_membrane_shear_center_row && elem_is_flat && q4_is_isotropic[ei]
        elem_membrane_assumed_mode =
            if elem_is_flat && is_pcomp_ei && !is_pcomp_iso_ei && Bmb_local === nothing
                if flat_pcomp_taper_membrane_none &&
                   aspect_ratio_ei >= flat_pcomp_taper_membrane_none_aspect_min &&
                   taper_ratio_ei <= flat_pcomp_taper_membrane_none_ratio_max
                    :none
                else
                    flat_pcomp_eig_membrane_assumed_mode
                end
            elseif elem_is_flat && q4_is_isotropic[ei]
                flat_iso_eig_membrane_assumed_mode
            elseif !elem_is_flat && is_pcomp_ei && !is_pcomp_iso_ei && Bmb_local === nothing
                # Ko-Lee-Bathe 2016 MITC4+: non-flat (warped) curved PCOMP quads need
                # ANS membrane to avoid membrane locking on curved shell geometries.
                # Default is :none (legacy MITC4) — enable via
                # JFEM_SOL105_EIG_NONFLAT_PCOMP_MEMBRANE_ASSUMED_MODE=mitc4plus.
                nonflat_pcomp_eig_membrane_assumed_mode
            else
                :none
            end
        bend_const_scale = 1.0
        if shear_center_only && elem_is_flat && q4_is_isotropic[ei] && iso_geom_curvature !== nothing
            bend_const_scale = q4_sol105_flat_cyl_iso_bend_effective_scale(kappa_l_iso, cyl_ratio_iso)
        end
        if q4_macneal_bending_scale != 1.0 &&
           q4_kernel_mode_static in ("macneal", "macneal_all", "macneal-force", "macneal_force",
                                     "macneal_pcomp", "macneal-pcomp", "macneal_aniso",
                                     "mitc4_3d_aspect", "mitc4-3d-aspect", "mitc3d_aspect", "mitc3d-aspect")
            bend_const_scale *= q4_macneal_bending_scale
        end
        if q4_macneal_bending_aspect_enabled &&
           q4_kernel_mode_static in ("macneal", "macneal_all", "macneal-force", "macneal_force",
                                     "macneal_pcomp", "macneal-pcomp", "macneal_aniso",
                                     "mitc4_3d_aspect", "mitc4-3d-aspect", "mitc3d_aspect", "mitc3d-aspect")
            kappa_l_bend_aspect = 0.0
            if pcomp_geom_curvature !== nothing
                k1_bend_aspect, _ = q4_curvature_principal_abs(pcomp_geom_curvature)
                kappa_l_bend_aspect =
                    k1_bend_aspect * q4_curvature_characteristic_length(lc)
            end
            if q4_macneal_bending_aspect_geom_ok(
                warp_ratio_ei,
                kappa_l_bend_aspect,
                edge_skew_ei,
                q4_macneal_bending_aspect_warp_min_v,
                q4_macneal_bending_aspect_warp_max_v,
                q4_macneal_bending_aspect_kappa_l_min_v,
                q4_macneal_bending_aspect_kappa_l_max_v,
                q4_macneal_bending_aspect_skew_min_v,
                q4_macneal_bending_aspect_skew_max_v,
            )
                bend_const_scale *= q4_macneal_bending_aspect_scale(
                    aspect_ratio_ei,
                    q4_macneal_bending_aspect_mode_v,
                    q4_macneal_bending_aspect_low,
                    q4_macneal_bending_aspect_mid,
                    q4_macneal_bending_aspect_high,
                    q4_macneal_bending_aspect_min_v,
                    q4_macneal_bending_aspect_peak_v,
                    q4_macneal_bending_aspect_max_v,
                )
            end
        end
        if q4_macneal_bending_aspect2_enabled &&
           q4_kernel_mode_static in ("macneal", "macneal_all", "macneal-force", "macneal_force",
                                     "macneal_pcomp", "macneal-pcomp", "macneal_aniso",
                                     "mitc4_3d_aspect", "mitc4-3d-aspect", "mitc3d_aspect", "mitc3d-aspect")
            kappa_l_bend_aspect2 = 0.0
            if pcomp_geom_curvature !== nothing
                k1_bend_aspect2, _ = q4_curvature_principal_abs(pcomp_geom_curvature)
                kappa_l_bend_aspect2 =
                    k1_bend_aspect2 * q4_curvature_characteristic_length(lc)
            end
            if q4_macneal_bending_aspect_geom_ok(
                warp_ratio_ei,
                kappa_l_bend_aspect2,
                edge_skew_ei,
                q4_macneal_bending_aspect2_warp_min_v,
                q4_macneal_bending_aspect2_warp_max_v,
                q4_macneal_bending_aspect2_kappa_l_min_v,
                q4_macneal_bending_aspect2_kappa_l_max_v,
                q4_macneal_bending_aspect2_skew_min_v,
                q4_macneal_bending_aspect2_skew_max_v,
            )
                bend_const_scale *= q4_macneal_bending_aspect_scale(
                    aspect_ratio_ei,
                    q4_macneal_bending_aspect2_mode_v,
                    q4_macneal_bending_aspect2_low,
                    q4_macneal_bending_aspect2_mid,
                    q4_macneal_bending_aspect2_high,
                    q4_macneal_bending_aspect2_min_v,
                    q4_macneal_bending_aspect2_peak_v,
                    q4_macneal_bending_aspect2_max_v,
                )
            end
        end
        if q4_macneal_curved_bending_scale != 1.0 &&
           q4_kernel_mode_static in ("macneal", "macneal_all", "macneal-force", "macneal_force",
                                     "macneal_pcomp", "macneal-pcomp", "macneal_aniso",
                                     "mitc4_3d_aspect", "mitc4-3d-aspect", "mitc3d_aspect", "mitc3d-aspect") &&
           pcomp_geom_curvature !== nothing
            k1_macneal, _ = q4_curvature_principal_abs(pcomp_geom_curvature)
            kappa_l_macneal = k1_macneal * q4_curvature_characteristic_length(lc)
            cyl_ratio_macneal = q4_curvature_cyl_ratio(pcomp_geom_curvature)
            if kappa_l_macneal >= q4_macneal_curved_bending_kappa_l_min &&
               cyl_ratio_macneal <= q4_macneal_curved_bending_cyl_ratio_max
                bend_const_scale *= q4_macneal_curved_bending_scale
            end
        end
        if q4_macneal_bending_isolated_scale != q4_macneal_bending_scale &&
           q4_kernel_mode_static in ("macneal", "macneal_all", "macneal-force", "macneal_force",
                                     "macneal_pcomp", "macneal-pcomp", "macneal_aniso",
                                     "mitc4_3d_aspect", "mitc4-3d-aspect", "mitc3d_aspect", "mitc3d-aspect") &&
           shell_valence[i1] == 1 && shell_valence[i2] == 1 &&
           shell_valence[i3] == 1 && shell_valence[i4] == 1
            bend_const_scale *= q4_macneal_bending_isolated_scale / max(q4_macneal_bending_scale, 1e-30)
        end
        if bend_const_scale != 1.0
            @inbounds @fastmath for jj in 1:3, ii in 1:3
                Cb_local[ii, jj] *= bend_const_scale
            end
        end
        elem_curved_iso_blend = curved_iso_blend
        if elem_shear_center_only &&
           is_iso_ei &&
           (!elem_is_flat || flat_curved_iso_center_candidate) &&
           aspect_ratio_ei <= curved_iso_square_blend_aspect_ratio_max
            elem_curved_iso_blend = min(elem_curved_iso_blend, curved_iso_square_blend)
        end
        auto_pcomp_membrane_incomp =
            shear_center_only &&
            is_pcomp_ei &&
            auto_pcomp_membrane_incomp_model
        static_pcomp_aspect_membrane_incomp =
            static_pcomp_membrane_incomp_aspect &&
            is_pcomp_ei &&
            aspect_ratio_ei >= static_pcomp_membrane_incomp_aspect_min &&
            aspect_ratio_ei <= static_pcomp_membrane_incomp_aspect_max
        # iso_no_incomp: disable Wilson incompatible modes for isotropic elements
        # (matches Nastran CQUAD4 standard bilinear formulation for eigenvalue K_eig)
        elem_bending_incomp = bending_incomp && !(iso_no_incomp && q4_is_isotropic[ei])
        # pcomp_membrane_incomp=true: add Wilson membrane modes for PCOMP elements in K_eig
        # (softens curved PCOMP K_eig → reduces overestimate; does not affect PSHELL elements)
        elem_membrane_incomp = membrane_incomp || auto_curved_iso_membrane_incomp ||
                               auto_warped_iso_membrane_incomp ||
                               auto_elongated_iso_membrane_incomp ||
                               auto_pcomp_membrane_incomp ||
                               static_pcomp_aspect_membrane_incomp ||
                               (pcomp_membrane_incomp && is_pcomp_ei) ||
                               (flat_iso_eig_membrane_incomp && elem_is_flat && q4_is_isotropic[ei])
        # Formal flat symmetric-laminate plate/shell regime:
        # use a DKMQ-style stiffness split (membrane + MITC shear + DKQ bending)
        # when the laminate has no membrane-bending coupling and the geometry is flat.
        elem_flat_dkmq_branch = flat_pcomp_dkmq_branch &&
                                elem_is_flat &&
                                is_pcomp_ei &&
                                !is_pcomp_iso_ei &&
                                !elem_macneal_rigid_shear &&
                                Bmb_local === nothing &&
                                maximum(abs, Cb_local) > 1e-30
        # Experimental flat-laminate plate branch: restrict to regular rectangular
        # quads and low-coupling laminates where the DKQ path has shown cleaner
        # buckling parity on the a500 family without harming the main guardrails.
        elem_flat_plate_auto = flat_pcomp_plate_auto &&
                               elem_is_flat &&
                               is_pcomp_ei &&
                               !is_pcomp_iso_ei &&
                               Bmb_local === nothing &&
                               maximum(abs, Cb_local) > 1e-30 &&
                               FEM.quad4_is_axis_aligned_rectangle(lc) &&
                               q4_pcomp_d16_ratio[ei] <= flat_pcomp_plate_auto_d16_ratio_max &&
                               q4_pcomp_shear_ratio[ei] <= flat_pcomp_plate_auto_shear_ratio_max
        elem_flat_iso_exact_membrane = flat_iso_dkmq_branch &&
                                       !model_has_line_elements &&
                                       is_iso_ei &&
                                       Bmb_local === nothing &&
                                       elem_is_flat &&
                                       !node_has_line[i1] && !node_has_line[i2] &&
                                       !node_has_line[i3] && !node_has_line[i4] &&
                                       geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4] &&
                                       q4_geom_normals_nearly_constant(
                                           geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4]
                                       )
        elem_flat_curved_iso_exact_membrane = flat_iso_dkmq_branch &&
                                              is_iso_ei &&
                                              Bmb_local === nothing &&
                                              flat_curved_iso_center_candidate &&
                                              aspect_ratio_ei <= flat_curved_iso_exact_membrane_aspect_ratio_max
        elem_saddle_iso_exact_membrane = false
        elem_cyl_iso_exact_membrane = flat_iso_dkmq_branch &&
                                      is_iso_ei &&
                                      Bmb_local === nothing &&
                                      !elem_is_flat &&
                                      iso_corner_curvature !== nothing &&
                                      abs(q4_curvature_gaussian(iso_corner_curvature)) <= 1e-10 &&
                                      first(q4_curvature_principal_abs(iso_corner_curvature)) > 1e-8
        elem_iso_exact_membrane =
            elem_flat_iso_exact_membrane || elem_flat_curved_iso_exact_membrane ||
            elem_saddle_iso_exact_membrane || elem_cyl_iso_exact_membrane
        elem_pcomp_exact_membrane = flat_pcomp_exact_membrane &&
                                    !model_has_line_elements &&
                                    !model_has_kinematic_constraints &&
                                    elem_is_flat &&
                                    is_pcomp_ei &&
                                    !is_pcomp_iso_ei &&
                                    Bmb_local === nothing
        elem_exact_membrane_operator = elem_iso_exact_membrane || elem_pcomp_exact_membrane
        # Flat facets on a curved shell patch still need the membrane-to-w
        elem_iso_exact_membrane_curvature_w_coupling =
            elem_cyl_iso_exact_membrane
        if curvature_membrane === nothing && elem_cyl_iso_exact_membrane && iso_corner_curvature !== nothing
            # For developable cylindrical shells, the exact membrane splice needs
            # the actual geometric curvature tensor. The generic SNORM-based
            # curvature filter is often inactive on these coarse barrel patches.
            curvature_membrane = iso_corner_curvature
        end
        elem_flat_plate_branch = (flat_pcomp_plate_branch || elem_flat_plate_auto) &&
                                 elem_is_flat &&
                                 is_pcomp_ei &&
                                 !is_pcomp_iso_ei &&
                                 !elem_macneal_rigid_shear &&
                                 Bmb_local === nothing &&
                                 maximum(abs, Cb_local) > 1e-30
        elem_rect_plate_branch = flat_pcomp_rect_adini && elem_flat_plate_branch && FEM.quad4_is_axis_aligned_rectangle(lc)
        elem_fullshear_selective = flat_pcomp_fullshear_selective &&
                                   elem_is_flat &&
                                   is_pcomp_ei &&
                                   !is_pcomp_iso_ei &&
                                   !elem_shear_center_only &&
                                   Bmb_local === nothing
        elem_flat_iso_fullshear_selective_mode =
            shear_center_only &&
            elem_is_flat &&
            q4_is_isotropic[ei] &&
            !elem_shear_center_only &&
            flat_iso_fullshear_selective_mode != :none ? flat_iso_fullshear_selective_mode : :none
        elem_selective_shear = elem_fullshear_selective || elem_flat_iso_fullshear_selective_mode != :none
        elem_selective_shear_mode =
            elem_flat_iso_fullshear_selective_mode != :none ? elem_flat_iso_fullshear_selective_mode : :all
        elem_exact_side_shear = flat_pcomp_exact_side_shear &&
                                elem_is_flat &&
                                is_pcomp_ei &&
                                !is_pcomp_iso_ei &&
                                !elem_shear_center_only &&
                                Bmb_local === nothing
        if !elem_exact_side_shear &&
           flat_curved_pcomp_exact_side_shear &&
           flat_curved_pcomp_fullshear_candidate &&
           !elem_shear_center_only &&
           Bmb_local === nothing
            elem_exact_side_shear = true
        end
        if !elem_exact_side_shear &&
           flat_iso_exact_side_shear &&
           elem_is_flat &&
           is_iso_ei &&
           elem_flat_curved_iso_exact_membrane &&
           Bmb_local === nothing
            elem_exact_side_shear = true
        end
        if !elem_exact_side_shear &&
           flat_curved_iso_coarse_exact_side_shear &&
           elem_is_flat &&
           is_iso_ei &&
           elem_flat_curved_iso_exact_membrane &&
           Bmb_local === nothing
            valence_sum = shell_valence[i1] + shell_valence[i2] + shell_valence[i3] + shell_valence[i4]
            if aspect_ratio_ei <= flat_curved_iso_coarse_exact_side_shear_aspect_ratio_max &&
               valence_sum >= flat_curved_iso_coarse_exact_side_shear_valence_sum_min &&
               valence_sum <= flat_curved_iso_coarse_exact_side_shear_valence_sum_max
                elem_exact_side_shear = true
            end
        end
        elem_exact_side_rotcorr = flat_pcomp_exact_side_rotcorr &&
                                  elem_is_flat &&
                                  is_pcomp_ei &&
                                  !is_pcomp_iso_ei &&
                                  !elem_shear_center_only &&
                                  Bmb_local === nothing
        if !elem_exact_side_rotcorr &&
           flat_iso_exact_side_rotcorr &&
           elem_is_flat &&
           is_iso_ei &&
           elem_flat_curved_iso_exact_membrane &&
           Bmb_local === nothing
            elem_exact_side_rotcorr = true
        end
        elem_flat_curved_iso_nodal_geomnormal_transform =
            flat_curved_iso_nodal_geomnormal_transform &&
            q4_is_isotropic[ei] &&
            elem_is_flat &&
            elem_flat_curved_iso_exact_membrane &&
            aspect_ratio_ei >= flat_curved_iso_nodal_geomnormal_transform_aspect_ratio_min &&
            (shell_valence[i1] + shell_valence[i2] + shell_valence[i3] + shell_valence[i4]) <=
                flat_curved_iso_nodal_geomnormal_transform_valence_sum_max &&
            geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
        elem_static_pcomp_nodal_geomnormal_transform =
            static_pcomp_nodal_geomnormal_transform &&
            is_pcomp_ei &&
            geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
        elem_mitc4_3d_kernel =
            if mitc4_3d_all_kernel
                true
            elseif mitc4_3d_aspect_kernel
                kappa_l_mitc4_3d_aspect = 0.0
                if pcomp_geom_curvature !== nothing
                    k1_mitc4_3d_aspect, _ = q4_curvature_principal_abs(pcomp_geom_curvature)
                    kappa_l_mitc4_3d_aspect =
                        k1_mitc4_3d_aspect * q4_curvature_characteristic_length(lc)
                end
                q4_mitc4_3d_aspect_geom_ok(
                    aspect_ratio_ei,
                    warp_ratio_ei,
                    kappa_l_mitc4_3d_aspect,
                    edge_skew_ei,
                    mitc4_3d_aspect_min,
                    mitc4_3d_aspect_max,
                    mitc4_3d_aspect_warp_min,
                    mitc4_3d_aspect_warp_max,
                    mitc4_3d_aspect_kappa_l_min,
                    mitc4_3d_aspect_kappa_l_max,
                    mitc4_3d_aspect_skew_min,
                    mitc4_3d_aspect_skew_max,
                    mitc4_3d_aspect_skew_aspect_min,
                ) &&
                (!mitc4_3d_aspect_pcomp_only || is_pcomp_ei)
            else
                false
            end
        elem_static_component_scale_ok =
            static_component_v2_gate_ok &&
            (isempty(static_component_pid_filter) ||
             (q4_pid_int[ei] in static_component_pid_filter)) &&
            (isempty(static_component_eid_filter) ||
             (q4_eid_int[ei] in static_component_eid_filter)) &&
            q4_static_component_range_ok(q4_h[ei],
                                         static_component_thickness_min,
                                         static_component_thickness_max) &&
            q4_static_component_pcomp_range_ok(is_pcomp_ei,
                                               q4_pcomp_shear_ratio[ei],
                                               static_component_pcomp_shear_ratio_min,
                                               static_component_pcomp_shear_ratio_max) &&
            q4_static_component_pcomp_range_ok(is_pcomp_ei,
                                               q4_pcomp_d16_ratio[ei],
                                               static_component_pcomp_d16_ratio_min,
                                               static_component_pcomp_d16_ratio_max) &&
            q4_static_component_pcomp_range_ok(is_pcomp_ei,
                                               q4_pcomp_b_ratio[ei],
                                               static_component_pcomp_b_ratio_min,
                                               static_component_pcomp_b_ratio_max) &&
            q4_static_component_neighbor_ok[ei]
        elem_static_component_cm_scale = elem_static_component_scale_ok ? static_component_cm_scale : 1.0
        elem_static_component_cb_scale = elem_static_component_scale_ok ? static_component_cb_scale : 1.0
        elem_static_component_cs_scale = elem_static_component_scale_ok ? static_component_cs_scale : 1.0
        elem_static_component_bmb_scale = elem_static_component_scale_ok ? static_component_bmb_scale : 1.0
        elem_static_component_drill_scale = elem_static_component_scale_ok ? static_component_drill_scale : 1.0
        if elem_static_component_cm_scale != 1.0
            @inbounds @fastmath for jj in 1:3, ii in 1:3
                Cm_local[ii, jj] *= elem_static_component_cm_scale
            end
        end
        if elem_static_component_cb_scale != 1.0
            @inbounds @fastmath for jj in 1:3, ii in 1:3
                Cb_local[ii, jj] *= elem_static_component_cb_scale
            end
        end
        if elem_static_component_cs_scale != 1.0
            @inbounds @fastmath for jj in 1:2, ii in 1:2
                Cs_local[ii, jj] *= elem_static_component_cs_scale
            end
        end
        if Bmb_local !== nothing && elem_static_component_bmb_scale != 1.0
            @inbounds @fastmath for jj in 1:3, ii in 1:3
                Bmb_local[ii, jj] *= elem_static_component_bmb_scale
            end
        end
        if elem_static_component_drill_scale != 1.0
            elem_drill_scale *= elem_static_component_drill_scale
        end
        if elem_mitc4_3d_kernel
            if mitc4_3d_ply_integration && is_pcomp_ei && q4_ply_data[ei] !== nothing
                Ke_t = FEM.stiffness_quad4_mitc4_3d_ply_matrices(
                    c3d_local, dirs_local, q4_ply_data[ei], Cs_local,
                    q4_h[ei], q4_Eref[ei];
                    k6rot=elem_k6rot,
                    drill_scale=elem_drill_scale,
                    shear_center_only=elem_shear_center_only,
                    material_rotation=elem_material_shear_rotation,
                    local_bending_scale=bend_const_scale,
                )
            else
                Ke_t = FEM.stiffness_quad4_mitc4_3d_resultant_matrices(
                    c3d_local, dirs_local, Cm_local, Cb_local, Cs_local,
                    q4_h[ei], q4_Eref[ei];
                    k6rot=elem_k6rot,
                    drill_scale=elem_drill_scale,
                    Bmb=Bmb_local,
                    shear_center_only=elem_shear_center_only,
                    bending_incomp=elem_bending_incomp,
                )
            end
        elseif elem_shear_center_only && is_iso_ei
            Ke_center = FEM.stiffness_quad4_matrices(lc, Cm_local, Cb_local, Cs_local,
                q4_h[ei], q4_Eref[ei]; bend_ratio=q4_br[ei], k6rot=elem_k6rot, Bmb=Bmb_local,
                drill_scale=elem_drill_scale,
                ws=ws_stiff, bending_incomp=elem_bending_incomp, shear_center_only=true,
                no_phi2=elem_no_phi2, membrane_incomp=elem_membrane_incomp, curvature_membrane=curvature_membrane,
                membrane_shear_center_row=elem_membrane_shear_center_row, material_shear_rotation=elem_material_shear_rotation,
                membrane_assumed_mode=elem_membrane_assumed_mode,
                membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
                exact_membrane_operator=elem_exact_membrane_operator,
                exact_membrane_curvature_w_coupling=elem_iso_exact_membrane_curvature_w_coupling,
                selective_shear=elem_selective_shear, selective_shear_mode=elem_selective_shear_mode,
                exact_side_shear=elem_exact_side_shear,
                exact_side_rotcorr=elem_exact_side_rotcorr,
                coords_3d=coords_3d_arg, kernel_planar=elem_kernel_planar,
                macneal_rigid_shear=elem_macneal_rigid_shear)
            Ke_full = FEM.stiffness_quad4_matrices(lc, Cm_local, Cb_local, Cs_local,
                q4_h[ei], q4_Eref[ei]; bend_ratio=q4_br[ei], k6rot=elem_k6rot, Bmb=Bmb_local,
                drill_scale=elem_drill_scale,
                ws=per_thread_ws_alt[tid], bending_incomp=elem_bending_incomp, shear_center_only=false,
                no_phi2=elem_no_phi2, membrane_incomp=elem_membrane_incomp, curvature_membrane=curvature_membrane,
                membrane_shear_center_row=elem_membrane_shear_center_row, material_shear_rotation=elem_material_shear_rotation,
                membrane_assumed_mode=elem_membrane_assumed_mode,
                membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
                exact_membrane_operator=elem_exact_membrane_operator,
                exact_membrane_curvature_w_coupling=elem_iso_exact_membrane_curvature_w_coupling,
                selective_shear=elem_selective_shear, selective_shear_mode=elem_selective_shear_mode,
                exact_side_shear=elem_exact_side_shear,
                exact_side_rotcorr=elem_exact_side_rotcorr,
                coords_3d=coords_3d_arg, kernel_planar=elem_kernel_planar,
                macneal_rigid_shear=elem_macneal_rigid_shear)
            Ke_t = sep_Ke_blend[tid]
            @inbounds @fastmath for jj in 1:24, ii in 1:24
                Ke_t[ii, jj] = (1.0 - elem_curved_iso_blend) * Ke_center[ii, jj] +
                               elem_curved_iso_blend * Ke_full[ii, jj]
            end
        elseif elem_flat_dkmq_branch
            Ke_t = FEM.stiffness_quad4_plate_dkmq_matrices(
                lc, Cm_local, Cb_local, Cs_local, q4_h[ei], q4_Eref[ei];
                k6rot=elem_k6rot,
                drill_scale=elem_drill_scale,
                ws=ws_stiff,
                membrane_incomp=elem_membrane_incomp,
                curvature_membrane=curvature_membrane,
                membrane_shear_center_row=elem_membrane_shear_center_row,
                material_shear_rotation=elem_material_shear_rotation,
                membrane_assumed_mode=elem_membrane_assumed_mode,
            )
        elseif elem_rect_plate_branch
            Ke_t = FEM.stiffness_quad4_plate_adini_matrices(
                lc, Cm_local, Cb_local, Cs_local, q4_h[ei], q4_Eref[ei];
                k6rot=elem_k6rot,
                drill_scale=elem_drill_scale,
                ws=ws_stiff,
                membrane_incomp=elem_membrane_incomp,
                curvature_membrane=curvature_membrane,
                membrane_shear_center_row=elem_membrane_shear_center_row,
                material_shear_rotation=elem_material_shear_rotation,
                membrane_assumed_mode=elem_membrane_assumed_mode,
            )
        elseif elem_flat_plate_branch
            Ke_t = FEM.stiffness_quad4_plate_dkq_matrices(
                lc, Cm_local, Cb_local, Cs_local, q4_h[ei], q4_Eref[ei];
                k6rot=elem_k6rot,
                drill_scale=elem_drill_scale,
                ws=ws_stiff,
                membrane_incomp=elem_membrane_incomp,
                curvature_membrane=curvature_membrane,
                membrane_shear_center_row=elem_membrane_shear_center_row,
                material_shear_rotation=elem_material_shear_rotation,
                membrane_assumed_mode=elem_membrane_assumed_mode,
            )
        elseif elem_shear_center_only && is_pcomp_ei
            if elem_is_flat && !is_pcomp_iso_ei
                Ke_t = FEM.stiffness_quad4_matrices(lc, Cm_local, Cb_local, Cs_local,
                    q4_h[ei], q4_Eref[ei]; bend_ratio=q4_br[ei], k6rot=elem_k6rot, Bmb=Bmb_local,
                    drill_scale=elem_drill_scale,
                    ws=ws_stiff, bending_incomp=elem_bending_incomp, shear_center_only=true,
                    no_phi2=elem_no_phi2, membrane_incomp=elem_membrane_incomp, curvature_membrane=curvature_membrane,
                    membrane_shear_center_row=elem_membrane_shear_center_row, material_shear_rotation=elem_material_shear_rotation,
                    membrane_assumed_mode=elem_membrane_assumed_mode,
                    membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
                    exact_membrane_operator=elem_exact_membrane_operator,
                    exact_membrane_curvature_w_coupling=elem_iso_exact_membrane_curvature_w_coupling,
                    selective_shear=elem_selective_shear, selective_shear_mode=elem_selective_shear_mode,
                    exact_side_shear=elem_exact_side_shear,
                    exact_side_rotcorr=elem_exact_side_rotcorr,
                    slope_membrane=slope_membrane,
                coords_3d=coords_3d_arg, kernel_planar=elem_kernel_planar,
                macneal_rigid_shear=elem_macneal_rigid_shear)
            elseif curved_pcomp_blend < 1.0
                Ke_center = FEM.stiffness_quad4_matrices(lc, Cm_local, Cb_local, Cs_local,
                    q4_h[ei], q4_Eref[ei]; bend_ratio=q4_br[ei], k6rot=elem_k6rot, Bmb=Bmb_local,
                    drill_scale=elem_drill_scale,
                    ws=ws_stiff, bending_incomp=elem_bending_incomp, shear_center_only=true,
                    no_phi2=elem_no_phi2, membrane_incomp=elem_membrane_incomp, curvature_membrane=curvature_membrane,
                    membrane_shear_center_row=elem_membrane_shear_center_row, material_shear_rotation=elem_material_shear_rotation,
                    membrane_assumed_mode=elem_membrane_assumed_mode,
                    membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
                    exact_membrane_operator=elem_exact_membrane_operator,
                    exact_membrane_curvature_w_coupling=elem_iso_exact_membrane_curvature_w_coupling,
                    selective_shear=elem_selective_shear, selective_shear_mode=elem_selective_shear_mode,
                    exact_side_shear=elem_exact_side_shear,
                    exact_side_rotcorr=elem_exact_side_rotcorr,
                coords_3d=coords_3d_arg, kernel_planar=elem_kernel_planar,
                macneal_rigid_shear=elem_macneal_rigid_shear)
                Ke_full = FEM.stiffness_quad4_matrices(lc, Cm_local, Cb_local, Cs_local,
                    q4_h[ei], q4_Eref[ei]; bend_ratio=q4_br[ei], k6rot=elem_k6rot, Bmb=Bmb_local,
                    drill_scale=elem_drill_scale,
                    ws=per_thread_ws_alt[tid], bending_incomp=elem_bending_incomp, shear_center_only=false,
                    no_phi2=elem_no_phi2, membrane_incomp=elem_membrane_incomp, curvature_membrane=curvature_membrane,
                    membrane_shear_center_row=elem_membrane_shear_center_row, material_shear_rotation=elem_material_shear_rotation,
                    membrane_assumed_mode=elem_membrane_assumed_mode,
                    membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
                    exact_membrane_operator=elem_exact_membrane_operator,
                    exact_membrane_curvature_w_coupling=elem_iso_exact_membrane_curvature_w_coupling,
                    selective_shear=elem_selective_shear, selective_shear_mode=elem_selective_shear_mode,
                    exact_side_shear=elem_exact_side_shear,
                    exact_side_rotcorr=elem_exact_side_rotcorr,
                coords_3d=coords_3d_arg, kernel_planar=elem_kernel_planar,
                macneal_rigid_shear=elem_macneal_rigid_shear)
                Ke_t = sep_Ke_blend[tid]
                @inbounds @fastmath for jj in 1:24, ii in 1:24
                    Ke_t[ii, jj] = (1.0 - curved_pcomp_blend) * Ke_center[ii, jj] +
                                   curved_pcomp_blend * Ke_full[ii, jj]
                end
            else
                Ke_t = FEM.stiffness_quad4_matrices(lc, Cm_local, Cb_local, Cs_local,
                    q4_h[ei], q4_Eref[ei]; bend_ratio=q4_br[ei], k6rot=elem_k6rot, Bmb=Bmb_local,
                    drill_scale=elem_drill_scale,
                    ws=ws_stiff, bending_incomp=elem_bending_incomp, shear_center_only=false,
                    no_phi2=elem_no_phi2, membrane_incomp=elem_membrane_incomp, curvature_membrane=curvature_membrane,
                    membrane_shear_center_row=elem_membrane_shear_center_row, material_shear_rotation=elem_material_shear_rotation,
                membrane_assumed_mode=elem_membrane_assumed_mode,
                membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
                exact_membrane_operator=elem_exact_membrane_operator,
                exact_membrane_curvature_w_coupling=elem_iso_exact_membrane_curvature_w_coupling,
                selective_shear=elem_selective_shear, selective_shear_mode=elem_selective_shear_mode,
                exact_side_shear=elem_exact_side_shear,
                exact_side_rotcorr=elem_exact_side_rotcorr,
                slope_membrane=slope_membrane,
                coords_3d=coords_3d_arg, kernel_planar=elem_kernel_planar,
                macneal_rigid_shear=elem_macneal_rigid_shear)
            end
        else
            Ke_t = FEM.stiffness_quad4_matrices(lc, Cm_local, Cb_local, Cs_local,
                q4_h[ei], q4_Eref[ei]; bend_ratio=q4_br[ei], k6rot=elem_k6rot, Bmb=Bmb_local,
                drill_scale=elem_drill_scale,
                ws=ws_stiff, bending_incomp=elem_bending_incomp, shear_center_only=elem_shear_center_only,
                no_phi2=elem_no_phi2, membrane_incomp=elem_membrane_incomp, curvature_membrane=curvature_membrane,
                membrane_shear_center_row=elem_membrane_shear_center_row, material_shear_rotation=elem_material_shear_rotation,
                membrane_assumed_mode=elem_membrane_assumed_mode,
                membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
                exact_membrane_operator=elem_exact_membrane_operator,
                exact_membrane_curvature_w_coupling=elem_iso_exact_membrane_curvature_w_coupling,
                selective_shear=elem_selective_shear, selective_shear_mode=elem_selective_shear_mode,
                exact_side_shear=elem_exact_side_shear,
                exact_side_rotcorr=elem_exact_side_rotcorr,
                slope_membrane=slope_membrane,
                coords_3d=coords_3d_arg, kernel_planar=elem_kernel_planar,
                macneal_rigid_shear=elem_macneal_rigid_shear)
        end

        T_t = sep_T[tid]; fill!(T_t, 0.0)
        @inbounds @fastmath for k in 1:4
            idx = k == 1 ? i1 : k == 2 ? i2 : k == 3 ? i3 : i4
            base = (k-1)*6
                vk1, vk2, vk3 =
                (elem_flat_curved_iso_nodal_geomnormal_transform ||
                 elem_static_pcomp_nodal_geomnormal_transform) ?
                shell_project_frame_to_normal(v1, v2, v3, geom_vec[idx]) :
                (v1, v2, v3)
            Rel_t = @SMatrix [vk1[1] vk1[2] vk1[3]; vk2[1] vk2[2] vk2[3]; vk3[1] vk3[2] vk3[3]]
            for rr in 1:3, cc in 1:3
                val = Rel_t[rr,1]*node_R_flat[1,cc,idx] + Rel_t[rr,2]*node_R_flat[2,cc,idx] + Rel_t[rr,3]*node_R_flat[3,cc,idx]
                T_t[base+rr, base+cc] = val
                T_t[base+3+rr, base+3+cc] = val
            end
        end
        tmp_t = sep_tmp[tid]; fill!(tmp_t, 0.0)
        out_t = sep_global[tid]; fill!(out_t, 0.0)
        @inbounds @fastmath for jj in 1:24, ll in 1:24
            val = T_t[ll, jj]
            if val != 0.0
                for ii in 1:24; tmp_t[ii, jj] += Ke_t[ii, ll] * val; end
            end
        end
        @inbounds @fastmath for jj in 1:24, ll in 1:24
            val = tmp_t[ll, jj]
            if val != 0.0
                for ii in 1:24; out_t[ii, jj] += T_t[ll, ii] * val; end
            end
        end

        dofs = sep_dofs[tid]
        for k in 1:4
            idx = k == 1 ? i1 : k == 2 ? i2 : k == 3 ? i3 : i4
            b = (idx-1)*6
            for d in 1:6; dofs[(k-1)*6+d] = b+d; end
        end
        base = (ei-1)*576; cnt = 0
        for cc in 1:24, rr in 1:24
            cnt += 1
            all_I[base+cnt] = dofs[rr]
            all_J[base+cnt] = dofs[cc]
            all_V[base+cnt] = out_t[rr,cc]
        end
    end

    LinearAlgebra.BLAS.set_num_threads(prev_blas_threads)

    I_idx = Vector{Int}(undef, 0); J_idx = Vector{Int}(undef, 0); V_val = Vector{Float64}(undef, 0)
    est_total = length(all_V) + n_t3*324 + length(cbars)*144 + length(cbeams)*144 + length(crods)*144
    sizehint!(I_idx, est_total); sizehint!(J_idx, est_total); sizehint!(V_val, est_total)
    append!(I_idx, all_I); append!(J_idx, all_J); append!(V_val, all_V)
    all_I = Int[]; all_J = Int[]; all_V = Float64[]

    # --- SEQUENTIAL TRIA3 ASSEMBLY ---
    lc_buf = zeros(3, 2)
    T_buf = zeros(18, 18)
    dofs_t3 = Vector{Int}(undef, 18)
    for ei in 1:n_t3
        i1 = t3_idx[ei,1]; i2 = t3_idx[ei,2]; i3 = t3_idx[ei,3]
        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
        p3 = SVector{3}(node_coords[i3,1], node_coords[i3,2], node_coords[i3,3])
        v1, v2, v3 = shell_element_frame_fast(p1, p2, p3, SVector{3}(0.0,0.0,0.0), 3)

        # SNORM adjustment
        n_avg = SVector(0.0, 0.0, 0.0); nc = 0
        for idx in (i1, i2, i3)
            if snorm_has[idx]; n_avg = n_avg + snorm_vec[idx]; nc += 1; end
        end
        if nc > 0
            n_avg_s = n_avg / nc; len_s = norm(n_avg_s)
            if len_s > 1e-12
                v3n = SVector{3}(n_avg_s / len_s)
                if dot(v3n, v3) < 0.0; v3n = -v3n; end
                v1p = v1 - dot(v1, v3n) * v3n; v1l = norm(v1p)
                if v1l > 1e-12
                    v1n = SVector{3}(v1p / v1l)
                else
                    v2p = v2 - dot(v2, v3n) * v3n; v1n = SVector{3}(normalize(v2p))
                end
                v1, v2, v3 = v1n, SVector{3}(cross(v3n, v1n)), v3n
            end
        end

        # PCOMP laminate-axis rotation for TRIA3 follows the element x-axis
        # plus the shell THETA angle, matching Nastran shell convention.
        Cm_t3 = t3_Cm[ei]; Cb_t3 = t3_Cb[ei]; Cs_t3 = t3_Cs[ei]; Bmb_t3 = t3_Bmb[ei]
        if t3_is_pcomp[ei]
            beta = shell_pcomp_material_rotation(
                pcomp_axis_mode,
                v1, v2, v3, p1, p2,
                t3_el_theta[ei],
                t3_el_mcid[ei],
                model["CORDs"],
            )
            if abs(beta) > 1e-10
                Cm_t3 = copy(Cm_t3); Cb_t3 = copy(Cb_t3); Cs_t3 = copy(Cs_t3)
                cb = cos(beta); sb = sin(beta)
                c2 = cb^2; s2 = sb^2; cs = cb*sb
                T11 = c2;  T12 = s2;  T13 = cs
                T21 = s2;  T22 = c2;  T23 = -cs
                T31 = -2cs; T32 = 2cs; T33 = c2 - s2
                _rotate_constitutive_3x3!(Cm_t3, T11, T12, T13, T21, T22, T23, T31, T32, T33)
                _rotate_constitutive_3x3!(Cb_t3, T11, T12, T13, T21, T22, T23, T31, T32, T33)
                a11 = Cs_t3[1,1]; a12 = Cs_t3[1,2]; a22 = Cs_t3[2,2]
                Cs_t3[1,1] = cb^2*a11 + 2*cb*sb*a12 + sb^2*a22
                Cs_t3[1,2] = -cb*sb*a11 + (cb^2-sb^2)*a12 + cb*sb*a22
                Cs_t3[2,1] = Cs_t3[1,2]
                Cs_t3[2,2] = sb^2*a11 - 2*cb*sb*a12 + cb^2*a22
                if Bmb_t3 !== nothing
                    Bmb_t3 = copy(Bmb_t3)
                    _rotate_constitutive_3x3!(Bmb_t3, T11, T12, T13, T21, T22, T23, T31, T32, T33)
                end
            end
        end

        c = (p1 + p2 + p3) / 3.0
        lc_buf[1,1] = dot(p1-c, v1); lc_buf[1,2] = dot(p1-c, v2)
        lc_buf[2,1] = dot(p2-c, v1); lc_buf[2,2] = dot(p2-c, v2)
        lc_buf[3,1] = dot(p3-c, v1); lc_buf[3,2] = dot(p3-c, v2)
        elem_k6rot_t3 = t3_br[ei] <= 1e-12 ? 0.0 :
                        (iso_eig_k6rot_override !== nothing && shear_center_only &&
                         t3_is_isotropic[ei] && !t3_is_pcomp[ei] && t3_br[ei] > 1e-12 ?
                         max(0.0, iso_eig_k6rot_override) : k6rot)
        Ke_loc = FEM.stiffness_tria3_matrices(lc_buf, Cm_t3, Cb_t3, Cs_t3,
                    t3_h[ei], t3_Eref[ei]; bend_ratio=t3_br[ei], k6rot=elem_k6rot_t3, Bmb=Bmb_t3)

        Rel_t = @SMatrix [v1[1] v1[2] v1[3]; v2[1] v2[2] v2[3]; v3[1] v3[2] v3[3]]
        fill!(T_buf, 0.0)
        for k in 1:3
            idx = k == 1 ? i1 : k == 2 ? i2 : i3
            TR = Rel_t * node_R[idx]
            base = (k-1)*6
            T_buf[base+1:base+3, base+1:base+3] = TR
            T_buf[base+4:base+6, base+4:base+6] = TR
        end
        Ke = T_buf' * Ke_loc * T_buf

        for k in 1:3
            idx = k == 1 ? i1 : k == 2 ? i2 : i3
            b = (idx-1)*6
            for d in 1:6; dofs_t3[(k-1)*6+d] = b+d; end
        end
        for cc in 1:18, rr in 1:18
            push!(I_idx, dofs_t3[rr]); push!(J_idx, dofs_t3[cc]); push!(V_val, Ke[rr,cc])
        end
    end

    log_msg("[SOLVER] Shell assembly: $n_q4 QUAD4 (parallel) + $n_t3 TRIA3 (sequential), NZ=$(length(I_idx))")
    T_buf = zeros(24, 24)

    # --- CBARS ---
    for (id, bar) in cbars
        pid = string(bar["PID"])
        if !haskey(pbarls, pid); continue; end
        prop = pbarls[pid]
        mid = string(prop["MID"])
        if !haskey(mats, mid); continue; end

        if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
        i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]
        mat = _effective_mat1_for_nodes(model, mid, [bar["GA"], bar["GB"]])
        mat === nothing && continue

        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])

        wa, wb, has_offset, p1_eff, p2_eff = bar_offsets_and_endpoints(bar, p1, p2)
        L = norm(p2_eff - p1_eff)
        if L < 1e-9; continue; end
        vx = normalize(p2_eff - p1_eff)
        v_ref = resolve_bar_vref(bar, p1, id_map, node_coords)

        if norm(v_ref) < 1e-6
             v_ref = SVector(0.0,0.0,1.0)
             if abs(dot(vx, v_ref)) > 0.9; v_ref = SVector(0.0,1.0,0.0); end
        end
        vz = normalize(cross(vx, v_ref))
        vy = cross(vz, vx)
        Rel_t = vcat(vx', vy', vz')

        fill!(view(T_buf, 1:12, 1:12), 0.0)
        TR1 = Rel_t * node_R[i1]
        TR2 = Rel_t * node_R[i2]
        T_buf[1:3, 1:3] = TR1; T_buf[4:6, 4:6] = TR1
        T_buf[7:9, 7:9] = TR2; T_buf[10:12, 10:12] = TR2
        if has_offset
            S_wa = skew3(wa); S_wb = skew3(wb)
            T_buf[1:3, 4:6] = -Rel_t * S_wa * node_R[i1]
            T_buf[7:9, 10:12] = -Rel_t * S_wb * node_R[i2]
        end

        Iy = get(prop, "I2", get(prop, "I", 0.0))
        Iz = get(prop, "I1", get(prop, "I", 0.0))
        if Iy == 0.0; Iy = Iz; end
        if Iz == 0.0; Iz = Iy; end
        Iyz = Float64(get(prop, "I12", 0.0))
        K1 = get(prop, "K1", 0.0); K2 = get(prop, "K2", 0.0)
        As_y = (K1 > 0.0) ? K1 * prop["A"] : Inf
        As_z = (K2 > 0.0) ? K2 * prop["A"] : Inf
        Ke_loc = FEM.stiffness_frame3d(L, prop["A"], Iy, Iz, prop["J"], mat["E"], mat["G"]; As_y=As_y, As_z=As_z, I12=Iyz)
        # Apply pin flags (PA/PB) via static condensation on local stiffness
        pa = get(bar, "PA", 0); pb = get(bar, "PB", 0)
        if pa != 0 || pb != 0
            Ke_loc_m = Matrix(Ke_loc)
            apply_bar_pin_flags!(Ke_loc_m, pa, pb)
            Ke_loc = Ke_loc_m
        end
        T_sub = view(T_buf, 1:12, 1:12)
        Ke = T_sub' * Ke_loc * T_sub

        dofs = [(i1-1)*6+k for k in 1:6]
        append!(dofs, [(i2-1)*6+k for k in 1:6])
        for c in 1:12, r in 1:12
            push!(I_idx, dofs[r]); push!(J_idx, dofs[c]); push!(V_val, Ke[r,c])
        end
    end

    # --- CBEAMs (identical stiffness to CBAR) ---
    for (id, bar) in cbeams
        pid = string(bar["PID"])
        if !haskey(pbarls, pid); continue; end
        prop = pbarls[pid]
        mid = string(prop["MID"])
        if !haskey(mats, mid); continue; end

        if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
        i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]
        mat = _effective_mat1_for_nodes(model, mid, [bar["GA"], bar["GB"]])
        mat === nothing && continue

        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])

        wa, wb, has_offset, p1_eff, p2_eff = bar_offsets_and_endpoints(bar, p1, p2)
        L = norm(p2_eff - p1_eff)
        if L < 1e-9; continue; end
        vx = normalize(p2_eff - p1_eff)
        v_ref = resolve_bar_vref(bar, p1, id_map, node_coords)

        if norm(v_ref) < 1e-6
             v_ref = SVector(0.0,0.0,1.0)
             if abs(dot(vx, v_ref)) > 0.9; v_ref = SVector(0.0,1.0,0.0); end
        end
        vz = normalize(cross(vx, v_ref))
        vy = cross(vz, vx)
        Rel_t = vcat(vx', vy', vz')

        fill!(view(T_buf, 1:12, 1:12), 0.0)
        TR1 = Rel_t * node_R[i1]
        TR2 = Rel_t * node_R[i2]
        T_buf[1:3, 1:3] = TR1; T_buf[4:6, 4:6] = TR1
        T_buf[7:9, 7:9] = TR2; T_buf[10:12, 10:12] = TR2
        if has_offset
            S_wa = skew3(wa); S_wb = skew3(wb)
            T_buf[1:3, 4:6] = -Rel_t * S_wa * node_R[i1]
            T_buf[7:9, 10:12] = -Rel_t * S_wb * node_R[i2]
        end

        Iy = get(prop, "I2", get(prop, "I", 0.0))
        Iz = get(prop, "I1", get(prop, "I", 0.0))
        if Iy == 0.0; Iy = Iz; end
        if Iz == 0.0; Iz = Iy; end
        Iyz = Float64(get(prop, "I12", 0.0))
        K1 = get(prop, "K1", 0.0); K2 = get(prop, "K2", 0.0)
        As_y = (K1 > 0.0) ? K1 * prop["A"] : Inf
        As_z = (K2 > 0.0) ? K2 * prop["A"] : Inf
        Ke_loc = FEM.stiffness_frame3d(L, prop["A"], Iy, Iz, prop["J"], mat["E"], mat["G"]; As_y=As_y, As_z=As_z, I12=Iyz)
        # Apply pin flags (PA/PB) via static condensation on local stiffness
        pa = get(bar, "PA", 0); pb = get(bar, "PB", 0)
        if pa != 0 || pb != 0
            Ke_loc_m = Matrix(Ke_loc)
            apply_bar_pin_flags!(Ke_loc_m, pa, pb)
            Ke_loc = Ke_loc_m
        end
        T_sub = view(T_buf, 1:12, 1:12)
        Ke = T_sub' * Ke_loc * T_sub

        dofs = [(i1-1)*6+k for k in 1:6]
        append!(dofs, [(i2-1)*6+k for k in 1:6])
        for c in 1:12, r in 1:12
            push!(I_idx, dofs[r]); push!(J_idx, dofs[c]); push!(V_val, Ke[r,c])
        end
    end

    # --- CRODS ---
    prods = get(model, "PRODs", Dict())
    for (id, rod) in crods
        pid = string(rod["PID"])
        if !haskey(prods, pid); continue; end
        prop = prods[pid]
        mid = string(prop["MID"])
        if !haskey(mats, mid); continue; end

        if !haskey(id_map, rod["GA"]) || !haskey(id_map, rod["GB"]); continue; end
        i1, i2 = id_map[rod["GA"]], id_map[rod["GB"]]
        mat = _effective_mat1_for_nodes(model, mid, [rod["GA"], rod["GB"]])
        mat === nothing && continue

        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
        L = norm(p2-p1)
        if L < 1e-9; continue; end
        vx = normalize(p2-p1)

        ref = abs(vx[3]) < 0.9 ? SVector(0.0,0.0,1.0) : SVector(0.0,1.0,0.0)
        vz = normalize(cross(vx, ref))
        vy = cross(vz, vx)
        Rel_t = vcat(vx', vy', vz')

        Ke_loc = zeros(12, 12)
        EA_L = mat["E"] * prop["A"] / L
        Ke_loc[1,1] = EA_L; Ke_loc[1,7] = -EA_L; Ke_loc[7,1] = -EA_L; Ke_loc[7,7] = EA_L
        GJ_L = mat["G"] * prop["J"] / L
        Ke_loc[4,4] = GJ_L; Ke_loc[4,10] = -GJ_L; Ke_loc[10,4] = -GJ_L; Ke_loc[10,10] = GJ_L

        fill!(view(T_buf, 1:12, 1:12), 0.0)
        TR1 = Rel_t * node_R[i1]; TR2 = Rel_t * node_R[i2]
        T_buf[1:3, 1:3] = TR1; T_buf[4:6, 4:6] = TR1
        T_buf[7:9, 7:9] = TR2; T_buf[10:12, 10:12] = TR2
        T_sub = view(T_buf, 1:12, 1:12)
        Ke = T_sub' * Ke_loc * T_sub

        dofs = [(i1-1)*6+k for k in 1:6]
        append!(dofs, [(i2-1)*6+k for k in 1:6])
        for c in 1:12, r in 1:12
            push!(I_idx, dofs[r]); push!(J_idx, dofs[c]); push!(V_val, Ke[r,c])
        end
    end

    # --- CONROD ---
    conrods = get(model, "CONRODs", Dict())
    for (id, rod) in conrods
        mid = string(rod["MID"])
        if !haskey(mats, mid); continue; end
        if !haskey(id_map, rod["GA"]) || !haskey(id_map, rod["GB"]); continue; end
        i1, i2 = id_map[rod["GA"]], id_map[rod["GB"]]
        mat = _effective_mat1_for_nodes(model, mid, [rod["GA"], rod["GB"]])
        mat === nothing && continue
        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
        L = norm(p2-p1)
        if L < 1e-9; continue; end
        vx = normalize(p2-p1)
        ref = abs(vx[3]) < 0.9 ? SVector(0.0,0.0,1.0) : SVector(0.0,1.0,0.0)
        vz = normalize(cross(vx, ref))
        vy = cross(vz, vx)
        Rel_t = vcat(vx', vy', vz')
        Ke_loc = zeros(12, 12)
        EA_L = mat["E"] * rod["A"] / L
        Ke_loc[1,1] = EA_L; Ke_loc[1,7] = -EA_L; Ke_loc[7,1] = -EA_L; Ke_loc[7,7] = EA_L
        GJ_L = mat["G"] * rod["J"] / L
        Ke_loc[4,4] = GJ_L; Ke_loc[4,10] = -GJ_L; Ke_loc[10,4] = -GJ_L; Ke_loc[10,10] = GJ_L
        fill!(view(T_buf, 1:12, 1:12), 0.0)
        TR1 = Rel_t * node_R[i1]; TR2 = Rel_t * node_R[i2]
        T_buf[1:3, 1:3] = TR1; T_buf[4:6, 4:6] = TR1
        T_buf[7:9, 7:9] = TR2; T_buf[10:12, 10:12] = TR2
        T_sub = view(T_buf, 1:12, 1:12)
        Ke = T_sub' * Ke_loc * T_sub
        dofs = [(i1-1)*6+k for k in 1:6]
        append!(dofs, [(i2-1)*6+k for k in 1:6])
        for c in 1:12, r in 1:12
            push!(I_idx, dofs[r]); push!(J_idx, dofs[c]); push!(V_val, Ke[r,c])
        end
    end

    # --- CELAS1 / CELAS2 ---
    celases = get(model, "CELASs", Dict())
    pelases = get(model, "PELASs", Dict())
    n_springs = 0
    for (id, spring) in celases
        local K_spring::Float64
        stype = get(spring, "TYPE", "CELAS1")
        if stype == "CELAS2"
            K_spring = Float64(get(spring, "K", 0.0))
        else
            pid = string(get(spring, "PID", 0))
            if !haskey(pelases, pid); continue; end
            K_spring = Float64(pelases[pid]["K"])
        end
        g1 = spring["G1"]; c1 = spring["C1"]
        g2 = spring["G2"]; c2 = spring["C2"]
        if g1 > 0 && haskey(id_map, g1) && c1 > 0
            i1 = id_map[g1]
            dof1 = (i1-1)*6 + c1
            push!(I_idx, dof1); push!(J_idx, dof1); push!(V_val, K_spring)
            if g2 > 0 && haskey(id_map, g2) && c2 > 0
                i2 = id_map[g2]
                dof2 = (i2-1)*6 + c2
                push!(I_idx, dof2); push!(J_idx, dof2); push!(V_val, K_spring)
                push!(I_idx, dof1); push!(J_idx, dof2); push!(V_val, -K_spring)
                push!(I_idx, dof2); push!(J_idx, dof1); push!(V_val, -K_spring)
            end
            n_springs += 1
        end
    end
    if n_springs > 0
        log_msg("[SOLVER] Springs: $n_springs CELAS1/CELAS2 elements assembled")
    end

    # --- CBUSH (bushing elements: 6-DOF spring between two grids) ---
    cbushes = get(model, "CBUSHs", Dict())
    pbushes = get(model, "PBUSHs", Dict())
    n_bush = 0
    for (id, bush) in cbushes
        pid = string(get(bush, "PID", 0))
        if !haskey(pbushes, pid); continue; end
        prop = pbushes[pid]
        K_vals = get(prop, "K", zeros(6))
        ga = bush["GA"]; gb = get(bush, "GB", 0)
        if !haskey(id_map, ga); continue; end
        i1 = id_map[ga]
        # CBUSH with GA only (grounded spring)
        if gb <= 0 || !haskey(id_map, gb)
            for k in 1:6
                Kk = length(K_vals) >= k ? Float64(K_vals[k]) : 0.0
                if Kk == 0.0; continue; end
                dof1 = (i1-1)*6 + k
                push!(I_idx, dof1); push!(J_idx, dof1); push!(V_val, Kk)
            end
        else
            # CBUSH connecting two grids — diagonal stiffness in each DOF
            i2 = id_map[gb]
            for k in 1:6
                Kk = length(K_vals) >= k ? Float64(K_vals[k]) : 0.0
                if Kk == 0.0; continue; end
                dof1 = (i1-1)*6 + k
                dof2 = (i2-1)*6 + k
                push!(I_idx, dof1); push!(J_idx, dof1); push!(V_val, Kk)
                push!(I_idx, dof2); push!(J_idx, dof2); push!(V_val, Kk)
                push!(I_idx, dof1); push!(J_idx, dof2); push!(V_val, -Kk)
                push!(I_idx, dof2); push!(J_idx, dof1); push!(V_val, -Kk)
            end
        end
        n_bush += 1
    end
    if n_bush > 0
        log_msg("[SOLVER] CBUSH: $n_bush bushing elements assembled")
    end

    # --- DMIG (Direct Matrix Input at Grid points) ---
    dmigs = get(model, "DMIGs", Dict{String,Dict{String,Any}}())
    n_dmig_entries = 0
    for (dmig_name, dmig_data) in dmigs
        is_sym = get(dmig_data, "type", "square") == "symmetric"
        entries = get(dmig_data, "entries", [])
        for (gi, ci, gj, cj, aij) in entries
            if !haskey(id_map, gi) || !haskey(id_map, gj); continue; end
            row_dof = (id_map[gi]-1)*6 + ci
            col_dof = (id_map[gj]-1)*6 + cj
            push!(I_idx, row_dof); push!(J_idx, col_dof); push!(V_val, aij)
            if is_sym && row_dof != col_dof
                push!(I_idx, col_dof); push!(J_idx, row_dof); push!(V_val, aij)
            end
            n_dmig_entries += 1
        end
    end
    if n_dmig_entries > 0
        log_msg("[SOLVER] DMIG: $n_dmig_entries entries from $(length(dmigs)) matrix/matrices injected")
    end

    # --- SOLID ELEMENTS (CTETRA, CHEXA, CPENTA) ---
    csolids = get(model, "CSOLIDs", Dict())
    psolids = get(model, "PSOLIDs", Dict())
    n_tetra = 0; n_hexa = 0; n_penta = 0
    T_buf_solid = zeros(24, 24)
    coords_buf = zeros(8, 3)  # max 8 nodes
    for (id, el) in csolids
        pid = string(el["PID"])
        if !haskey(psolids, pid); continue; end
        prop = psolids[pid]
        nids = el["NODES"]
        nn = length(nids)
        mid = string(prop["MID"])
        if !haskey(mats, mid); continue; end
        mat = _effective_mat1_for_nodes(model, mid, nids)
        etype = get(el, "TYPE", "")

        # Validate nodes
        valid = true
        for k in 1:nn
            if !haskey(id_map, nids[k]); valid = false; break; end
        end
        if !valid; continue; end

        # Gather coordinates
        for k in 1:nn
            idx = id_map[nids[k]]
            coords_buf[k,1] = node_coords[idx, 1]
            coords_buf[k,2] = node_coords[idx, 2]
            coords_buf[k,3] = node_coords[idx, 3]
        end

        E_mat = Float64(mat["E"]); nu_mat = Float64(mat["NU"])

        local Ke_loc
        local ndof_el::Int
        if etype == "CTETRA" && nn == 4
            Ke_loc = FEM.stiffness_tetra4(view(coords_buf, 1:4, :), E_mat, nu_mat)
            ndof_el = 12; n_tetra += 1
        elseif etype == "CHEXA" && nn == 8
            Ke_loc = FEM.stiffness_hexa8(view(coords_buf, 1:8, :), E_mat, nu_mat)
            ndof_el = 24; n_hexa += 1
        elseif etype == "CPENTA" && nn == 6
            Ke_loc = FEM.stiffness_cpenta6(view(coords_buf, 1:6, :), E_mat, nu_mat)
            ndof_el = 18; n_penta += 1
        else
            continue
        end

        # Transform for displacement coordinate systems (node_R)
        # Solid stiffness is in global coords; transform to node-local: Ke_final = T' * Ke * T
        # T = block_diag(node_R[i1], node_R[i2], ...)
        fill!(view(T_buf_solid, 1:ndof_el, 1:ndof_el), 0.0)
        for k in 1:nn
            idx = id_map[nids[k]]
            r = (k-1)*3
            for a in 1:3, b in 1:3
                T_buf_solid[r+a, r+b] = node_R[idx][a, b]
            end
        end
        T_sub = view(T_buf_solid, 1:ndof_el, 1:ndof_el)
        Ke = T_sub' * Ke_loc * T_sub

        # Build DOF mapping: solid elements use only translational DOFs (1,2,3)
        dofs_solid = Vector{Int}(undef, ndof_el)
        for k in 1:nn
            idx = id_map[nids[k]]
            base = (idx-1)*6
            dofs_solid[(k-1)*3+1] = base + 1
            dofs_solid[(k-1)*3+2] = base + 2
            dofs_solid[(k-1)*3+3] = base + 3
        end

        for c in 1:ndof_el, r in 1:ndof_el
            push!(I_idx, dofs_solid[r]); push!(J_idx, dofs_solid[c]); push!(V_val, Ke[r,c])
        end
    end
    n_solids_total = n_tetra + n_hexa + n_penta
    if n_solids_total > 0
        log_msg("[SOLVER] Solids: $n_tetra CTETRA + $n_hexa CHEXA + $n_penta CPENTA assembled")
    end

    # Save max element stiffness BEFORE constraint processing
    max_elem_stiff = 0.0
    for i in 1:length(V_val)
        if abs(V_val[i]) > max_elem_stiff; max_elem_stiff = abs(V_val[i]); end
    end

    # Compute original diagonal BEFORE MPC redistribution (for AUTOSPC)
    orig_diag = zeros(ndof)
    for k in 1:length(I_idx)
        if I_idx[k] == J_idx[k]
            orig_diag[I_idx[k]] += V_val[k]
        end
    end

    # --- Constraint assembly (RBE2, RBE3, MPC) ---
    rbe3_map, I_idx, J_idx, V_val = assemble_constraints(model, id_map, node_coords, node_R, I_idx, J_idx, V_val)

    actual_nz = length(I_idx)
    log_msg("[SOLVER] Creating Sparse Matrix (NZ: $actual_nz)...")
    K = sparse(I_idx, J_idx, V_val, ndof, ndof)
    I_idx = nothing; J_idx = nothing; V_val = nothing; GC.gc()

    return K, id_map, node_coords, ndof, node_R, max_elem_stiff, rbe3_map, snorm_normals, orig_diag
end

# =============================================================================
# GEOMETRIC STIFFNESS ASSEMBLY FOR SOL105 LINEAR BUCKLING
# Mirrors assemble_stiffness but builds Kg from SOL101 stress state.
# u_global = displacement solution from SOL101 (in local coord-system DOFs).
# =============================================================================
function assemble_geometric_stiffness(model, id_map, node_coords, node_R, ndof, u_global, snorm_normals, rbe3_map;
                                      snorm_angle_override::Union{Nothing,Float64}=nothing,
                                      buckling_subcase=nothing,
                                      static_load_id=nothing,
                                      timings=nothing)
    kg_t_total = time_ns()
    kg_timings = Dict{String,Any}()
    kg_t_setup = time_ns()
    log_msg("[SOLVER] Assembling Geometric Stiffness Matrix (SOL105)...")

    snorm_override = isnothing(snorm_angle_override) ?
        get(ENV, "JFEM_PARAM_SNORM_OVERRIDE_KG", get(ENV, "JFEM_PARAM_SNORM_OVERRIDE", "")) :
        string(snorm_angle_override)
    if !isempty(strip(snorm_override))
        model = copy(model)
        model["PARAM_SNORM"] = something(tryparse(Float64, snorm_override), get(model, "PARAM_SNORM", 0.0))
    end
    q4_frame_mode = q4_frame_mode_from_env("JFEM_Q4_FRAME_MODE_KG")
    kg_compatible_membrane = kg_use_compatible_membrane_stress()
    kg_match_static_membrane_operator = kg_match_static_membrane_operator_enabled()
    membrane_incomp =
        if haskey(ENV, "JFEM_KG_MEMBRANE_INCOMP")
            solver_env_bool("JFEM_KG_MEMBRANE_INCOMP", sol105_static_membrane_incomp_enabled())
        elseif kg_match_static_membrane_operator
            sol105_static_membrane_incomp_enabled()
        else
            solver_env_bool("JFEM_SOL105_EIG_MEMBRANE_INCOMP", false)
        end
    pcomp_membrane_incomp = solver_env_bool("JFEM_SOL105_EIG_PCOMP_MEMBRANE_INCOMP", false)
    # Diagnostic override: recover Kg prestress with PCOMP incompatible modes
    # without changing the static/eigen stiffness assembly path.
    kg_pcomp_membrane_incomp =
        solver_env_bool("JFEM_KG_PCOMP_MEMBRANE_INCOMP", pcomp_membrane_incomp)
    kg_consistent_membrane_operator = kg_quad4_consistent_membrane_operator_enabled()
    kg_pcomp_consistent_membrane_operator =
        solver_env_bool("JFEM_KG_PCOMP_CONSISTENT_MEMBRANE_INCOMP_OPERATOR",
                        kg_consistent_membrane_operator)
    kg_trans_mode = kg_shell_trans_mode()
    kg_principal_transverse_flat_only = kg_shell_principal_transverse_flat_only_enabled()
    kg_principal_transverse_warp_ratio_max = kg_shell_principal_transverse_warp_ratio_max()
    kg_principal_shear_yy_factor_v = kg_shell_principal_shear_yy_factor()
    kg_principal_shear_xy_factor_v = kg_shell_principal_shear_xy_factor()
    kg_principal_shear_z_factor_v = kg_shell_principal_shear_z_factor()
    kg_principal_shear_ratio_min_v = kg_shell_principal_shear_ratio_min()
    kg_principal_shear_warp_min_v = kg_shell_principal_shear_warp_min()
    kg_principal_shear_warp_max_v = kg_shell_principal_shear_warp_max()
    kg_principal_shear_aspect_min_v = kg_shell_principal_shear_aspect_min()
    kg_principal_shear_aspect_max_v = kg_shell_principal_shear_aspect_max()
    kg_principal_shear_geom_mode_v = kg_shell_principal_shear_geometry_mode()
    kg_principal_shear_feature_gate_v = kg_shell_principal_shear_feature_gate()
    kg_principal_shear_gp_pmin_spread_min_v = kg_shell_principal_shear_gp_pmin_spread_min()
    kg_principal_shear_gp_nxx_spread_min_v = kg_shell_principal_shear_gp_nxx_spread_min()
    kg_principal_shear_gp_spread_factor_v = kg_shell_principal_shear_gp_spread_factor()
    kg_curvature_sign = kg_shell_curvature_sign()
    kg_rot_grad_scale = kg_shell_rot_grad_scale()
    kg_rot_grad_auto_iso_scale = kg_shell_rot_grad_auto_iso_scale()
    kg_rot_grad_auto_pcomp_scale = kg_shell_rot_grad_auto_pcomp_scale()
    kg_rot_grad_auto_kappa_l_min = kg_shell_rot_grad_auto_kappa_l_min()
    kg_rot_grad_auto_cyl_ratio_min = kg_shell_rot_grad_auto_cyl_ratio_min()
    kg_shell_nxy = kg_shell_nxy_scale()
    kg_shell_nxx = kg_shell_nxx_scale()
    kg_shell_nyy = kg_shell_nyy_scale()
    kg_shell_axial_dom_min = kg_shell_axial_scale_dominance_min()
    kg_quad4_membrane_scale = kg_quad4_membrane_scale_factor()
    kg_quad4_feature_membrane_scale = kg_quad4_feature_membrane_scale_factor()
    kg_quad4_feature_membrane_scale_components_v = kg_quad4_feature_membrane_scale_components()
    kg_quad4_feature_membrane_scale_pcomp_only_v = kg_quad4_feature_membrane_scale_pcomp_only()
    kg_quad4_feature_membrane_scale_pids = kg_quad4_feature_membrane_scale_pid_list()
    kg_quad4_feature_membrane_scale_aspect_min_v = kg_quad4_feature_membrane_scale_aspect_min()
    kg_quad4_feature_membrane_scale_aspect_max_v = kg_quad4_feature_membrane_scale_aspect_max()
    kg_quad4_feature_membrane_scale_warp_min_v = kg_quad4_feature_membrane_scale_warp_min()
    kg_quad4_feature_membrane_scale_warp_max_v = kg_quad4_feature_membrane_scale_warp_max()
    kg_quad4_feature_membrane_scale_kappa_l_min_v = kg_quad4_feature_membrane_scale_kappa_l_min()
    kg_quad4_feature_membrane_scale_kappa_l_max_v = kg_quad4_feature_membrane_scale_kappa_l_max()
    kg_quad4_feature_membrane_scale_geom_mode_v = kg_quad4_feature_membrane_scale_geometry_mode()
    kg_quad4_feature_membrane_scale_nxx_sign_v = kg_quad4_feature_membrane_scale_sign_gate()
    kg_quad4_feature_membrane_scale_nxy_sign_v = kg_quad4_feature_membrane_scale_nxy_sign_gate()
    kg_quad4_feature_membrane_scale_nxy_stat_v = kg_quad4_feature_membrane_scale_nxy_stat()
    kg_quad4_feature_membrane_scale_abs_nxy_min_v = kg_quad4_feature_membrane_scale_abs_nxy_min()
    kg_quad4_feature_membrane_scale_nxy_mode_v = kg_quad4_feature_membrane_scale_nxy_mode()
    kg_quad4_feature_membrane_scale_gp_pmin_spread_min_v = kg_quad4_feature_membrane_scale_gp_pmin_spread_min()
    kg_quad4_feature_membrane_scale_gp_nxx_spread_min_v = kg_quad4_feature_membrane_scale_gp_nxx_spread_min()
    kg_quad4_feature_membrane_scale_gp_spread_factor_v = kg_quad4_feature_membrane_scale_gp_spread_factor()
    kg_quad4_gp_field_pmin_spread_avg_min_v = kg_quad4_gp_field_pmin_spread_avg_min()
    kg_quad4_gp_field_pmin_spread_avg_alpha_v = kg_quad4_gp_field_pmin_spread_avg_alpha()
    kg_quad4_pid_membrane_scales = kg_quad4_pid_membrane_scale_map()
    kg_quad4_pid_membrane_scale_components_v = kg_quad4_pid_membrane_scale_components()
    kg_quad4_pid_membrane_scale_nxx_sign = kg_quad4_pid_membrane_scale_nxx_sign_gate()
    kg_quad4_pid_membrane_scale_gp_pmin_spread_min_v = kg_quad4_pid_membrane_scale_gp_pmin_spread_min()
    kg_quad4_pid_membrane_scale_gp_nxx_spread_min_v = kg_quad4_pid_membrane_scale_gp_nxx_spread_min()
    kg_quad4_pid_membrane_scale_gp_spread_factor_v = kg_quad4_pid_membrane_scale_gp_spread_factor()
    kg_quad4_pid_membrane_scale_v2_min_v = kg_quad4_pid_membrane_scale_v2_min()
    kg_quad4_pid_membrane_scale_v2_max_v = kg_quad4_pid_membrane_scale_v2_max()
    kg_quad4_pid_membrane_scale_v2_v = kg_quad4_buckling_eigrl_v2(model, buckling_subcase)
    kg_quad4_pid_membrane_scale_v2_ok_v = kg_quad4_pid_membrane_scale_v2_ok(
        kg_quad4_pid_membrane_scale_v2_v,
        kg_quad4_pid_membrane_scale_v2_min_v,
        kg_quad4_pid_membrane_scale_v2_max_v,
    )
    kg_quad4_auto_avg_load_ok = kg_quad4_auto_avg_load_classifier(model, static_load_id)
    kg_shell_pcomp_nxy = kg_shell_pcomp_nxy_scale()
    kg_shell_pcomp_nxy_aspect = kg_shell_pcomp_nxy_aspect_scale_enabled()
    kg_shell_pcomp_nxy_aspect_mode_v = kg_shell_pcomp_nxy_aspect_mode()
    kg_shell_pcomp_nxy_aspect_low_v = kg_shell_pcomp_nxy_aspect_low()
    kg_shell_pcomp_nxy_aspect_mid_v = kg_shell_pcomp_nxy_aspect_mid()
    kg_shell_pcomp_nxy_aspect_high_v = kg_shell_pcomp_nxy_aspect_high()
    kg_shell_pcomp_nxy_aspect_min_v = kg_shell_pcomp_nxy_aspect_min()
    kg_shell_pcomp_nxy_aspect_peak_v = kg_shell_pcomp_nxy_aspect_peak()
    kg_shell_pcomp_nxy_aspect_max_v = kg_shell_pcomp_nxy_aspect_max()
    kg_shell_pcomp_nxy_compression_only_v = kg_shell_pcomp_nxy_compression_only()
    kg_shell_pcomp_nxy_shear_dom_relax_v = kg_shell_pcomp_nxy_shear_dom_relax()
    kg_shell_pcomp_nxy_shear_dom_ratio_min_v = kg_shell_pcomp_nxy_shear_dom_ratio_min()
    kg_shell_pcomp_nxy_shear_dom_ratio_full_v = kg_shell_pcomp_nxy_shear_dom_ratio_full()
    kg_shell_pcomp_nxy_shear_dom_aspect_min_v = kg_shell_pcomp_nxy_shear_dom_aspect_min()
    kg_shell_pcomp_nxy_shear_dom_aspect_max_v = kg_shell_pcomp_nxy_shear_dom_aspect_max()
    kg_shell_pcomp_nxy_shear_dom_compression_only_v = kg_shell_pcomp_nxy_shear_dom_compression_only()
    q4_kernel_mode_kg = lowercase(strip(get(ENV, "JFEM_Q4_KERNEL_KG", get(ENV, "JFEM_Q4_KERNEL", "macneal"))))
    mitc4_3d_kg_all_kernel = q4_kernel_mode_kg in ("mitc4_3d", "mitc4-3d", "mitc3d")
    mitc4_3d_kg_aspect_kernel = q4_kernel_mode_kg in (
        "mitc4_3d_aspect", "mitc4-3d-aspect", "mitc3d_aspect", "mitc3d-aspect",
    )
    mitc4_3d_kg_all_consistent =
        mitc4_3d_kg_all_kernel && solver_env_bool("JFEM_Q4_MITC4_3D_KG_COVARIANT", true)
    mitc4_3d_kg_aspect_consistent =
        mitc4_3d_kg_aspect_kernel &&
        solver_env_bool(
            "JFEM_Q4_MITC4_3D_ASPECT_KG_COVARIANT",
            solver_env_bool("JFEM_Q4_MITC4_3D_KG_COVARIANT", true),
        )
    mitc4_3d_kg_recovery_all =
        mitc4_3d_kg_all_consistent && solver_env_bool("JFEM_Q4_MITC4_3D_KG_RECOVERY", false)
    mitc4_3d_kg_recovery_aspect =
        mitc4_3d_kg_aspect_consistent &&
        solver_env_bool(
            "JFEM_Q4_MITC4_3D_ASPECT_KG_RECOVERY",
            solver_env_bool("JFEM_Q4_MITC4_3D_KG_RECOVERY", false),
        )
    mitc4_3d_kg_recovery = mitc4_3d_kg_recovery_all || mitc4_3d_kg_recovery_aspect
    mitc4_3d_aspect_min_base_kg = max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_MIN", 3.0), 1.0)
    mitc4_3d_aspect_max_base_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_MAX", 1e30), mitc4_3d_aspect_min_base_kg)
    mitc4_3d_aspect_warp_min_base_kg = max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_WARP_MIN", 0.0), 0.0)
    mitc4_3d_aspect_warp_max_base_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_WARP_MAX", 1e30), mitc4_3d_aspect_warp_min_base_kg)
    mitc4_3d_aspect_kappa_l_min_base_kg = max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KAPPA_L_MIN", 0.0), 0.0)
    mitc4_3d_aspect_kappa_l_max_base_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KAPPA_L_MAX", 1e30), mitc4_3d_aspect_kappa_l_min_base_kg)
    mitc4_3d_aspect_skew_min_base_kg = q4_mitc4_3d_aspect_skew_min()
    mitc4_3d_aspect_skew_max_base_kg =
        max(q4_mitc4_3d_aspect_skew_max(), mitc4_3d_aspect_skew_min_base_kg)
    mitc4_3d_aspect_skew_aspect_min_base_kg = q4_mitc4_3d_aspect_skew_aspect_min()
    mitc4_3d_aspect_pcomp_only_base_kg = solver_env_bool("JFEM_Q4_MITC4_3D_ASPECT_PCOMP_ONLY", true)
    mitc4_3d_aspect_min_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KG_ASPECT_MIN", mitc4_3d_aspect_min_base_kg), 1.0)
    mitc4_3d_aspect_max_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KG_ASPECT_MAX", mitc4_3d_aspect_max_base_kg), mitc4_3d_aspect_min_kg)
    mitc4_3d_aspect_warp_min_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KG_WARP_MIN", mitc4_3d_aspect_warp_min_base_kg), 0.0)
    mitc4_3d_aspect_warp_max_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KG_WARP_MAX", mitc4_3d_aspect_warp_max_base_kg), mitc4_3d_aspect_warp_min_kg)
    mitc4_3d_aspect_kappa_l_min_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KG_KAPPA_L_MIN", mitc4_3d_aspect_kappa_l_min_base_kg), 0.0)
    mitc4_3d_aspect_kappa_l_max_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KG_KAPPA_L_MAX", mitc4_3d_aspect_kappa_l_max_base_kg), mitc4_3d_aspect_kappa_l_min_kg)
    mitc4_3d_aspect_skew_min_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KG_SKEW_MIN", mitc4_3d_aspect_skew_min_base_kg), 0.0)
    mitc4_3d_aspect_skew_max_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KG_SKEW_MAX", mitc4_3d_aspect_skew_max_base_kg), mitc4_3d_aspect_skew_min_kg)
    mitc4_3d_aspect_skew_aspect_min_kg =
        max(solver_env_float("JFEM_Q4_MITC4_3D_ASPECT_KG_SKEW_ASPECT_MIN", mitc4_3d_aspect_skew_aspect_min_base_kg), 0.0)
    mitc4_3d_aspect_pcomp_only_kg =
        solver_env_bool("JFEM_Q4_MITC4_3D_ASPECT_KG_PCOMP_ONLY", mitc4_3d_aspect_pcomp_only_base_kg)
    kg_surface_operator_mode = kg_shell_surface_operator_mode()
    if mitc4_3d_kg_all_consistent && !haskey(ENV, "JFEM_KG_SHELL_SURFACE_OPERATOR")
        kg_surface_operator_mode = :covariant
    end
    kg_shell_nxy_auto = kg_shell_nxy_auto_relax()
    kg_shell_drill_zero = kg_shell_drill_zero_enabled()
    kg_shell_nxy_auto_ratio_min_v = kg_shell_nxy_auto_ratio_min()
    kg_shell_nxy_auto_ratio_full_v = kg_shell_nxy_auto_ratio_full()
    kg_shell_nxy_auto_cyl_ratio_max_v = kg_shell_nxy_auto_cyl_ratio_max()
    kg_shell_nxy_auto_kappa_l_min_v = kg_shell_nxy_auto_kappa_l_min()
    kg_membrane_recovery_mode = kg_quad4_membrane_recovery_mode()
    kg_covariant_blend = kg_quad4_covariant_blend()
    kg_covariant_auto_kappa_l_min = kg_quad4_covariant_auto_kappa_l_min()
    kg_covariant_auto_cyl_ratio_max = kg_quad4_covariant_auto_cyl_ratio_max()
    kg_auto_gp_patch = kg_quad4_auto_gp_patch_enabled()
    kg_auto_gp_patch_valence_min = kg_quad4_auto_gp_patch_valence_min()
    kg_auto_gp_patch_kappa_l_min = kg_quad4_auto_gp_patch_kappa_l_min()
    kg_shear_avg_operator = kg_quad4_shear_average_operator_enabled()
    kg_shear_avg_ratio_min = kg_quad4_shear_average_ratio_min()
    kg_shear_avg_warp_min = kg_quad4_shear_average_warp_min()
    kg_shear_avg_warp_max = kg_quad4_shear_average_warp_max()
    kg_shear_avg_aspect_min = kg_quad4_shear_average_aspect_min()
    kg_shear_avg_aspect_max = kg_quad4_shear_average_aspect_max()
    kg_shear_avg_geom_mode = kg_quad4_shear_average_geometry_mode()
    kg_auto_gp_spread = kg_quad4_auto_gp_spread_enabled()
    kg_auto_gp_spread_min = kg_quad4_auto_gp_spread_min()
    kg_auto_gp_spread_kappa_l_min = kg_quad4_auto_gp_spread_kappa_l_min()
    kg_auto_gp_spread_cyl_ratio_min = kg_quad4_auto_gp_spread_cyl_ratio_min()
    kg_gp_extrapolate_scale = kg_quad4_gp_field_extrapolate_scale()
    membrane_incomp_center_jacobian = q4_sol105_membrane_incomp_center_jacobian_enabled()
    flat_iso_eig_membrane_incomp = q4_flat_iso_eig_membrane_incomp_enabled()
    flat_iso_eig_membrane_shear_center_row = q4_flat_iso_eig_membrane_shear_center_row_enabled()
    flat_iso_eig_membrane_assumed_mode = q4_flat_iso_eig_membrane_assumed_mode()
    flat_iso_dkmq_branch = q4_sol105_flat_iso_dkmq_enabled()
    flat_curved_iso_eig_center_only = q4_flat_curved_iso_eig_center_only_enabled()
    flat_curved_iso_eig_center_only_kappa_l_min = q4_flat_curved_iso_eig_center_only_kappa_l_min()
    flat_curved_iso_eig_center_only_cyl_ratio_max = q4_flat_curved_iso_eig_center_only_cyl_ratio_max()
    flat_curved_iso_geomnormal_frame = q4_flat_curved_iso_geomnormal_frame_enabled()
    flat_curved_iso_geomnormal_frame_aspect_ratio_min = q4_flat_curved_iso_geomnormal_frame_aspect_ratio_min()
    flat_curved_iso_geomnormal_frame_kappa_l_min = q4_flat_curved_iso_geomnormal_frame_kappa_l_min()
    flat_curved_iso_geomnormal_frame_kappa_l_max = q4_flat_curved_iso_geomnormal_frame_kappa_l_max()
    flat_curved_iso_geomnormal_frame_cyl_ratio_max = q4_flat_curved_iso_geomnormal_frame_cyl_ratio_max()
    flat_curved_iso_nodal_geomnormal_transform = q4_flat_curved_iso_nodal_geomnormal_transform_enabled()
    flat_curved_iso_nodal_geomnormal_transform_aspect_ratio_min = q4_flat_curved_iso_nodal_geomnormal_transform_aspect_ratio_min()
    flat_curved_iso_nodal_geomnormal_transform_valence_sum_max = q4_flat_curved_iso_nodal_geomnormal_transform_valence_sum_max()
    flat_pcomp_eig_membrane_assumed_mode = q4_flat_pcomp_eig_membrane_assumed_mode()
    flat_pcomp_taper_membrane_none = q4_flat_pcomp_taper_membrane_none_enabled()
    flat_pcomp_taper_membrane_none_ratio_max = q4_flat_pcomp_taper_membrane_none_ratio_max()
    flat_pcomp_taper_membrane_none_aspect_min = q4_flat_pcomp_taper_membrane_none_aspect_min()
    nonflat_pcomp_eig_membrane_assumed_mode = q4_nonflat_pcomp_eig_membrane_assumed_mode()
    kg_pcomp_axis_mode = q4_pcomp_kg_axis_mode()
    kg_pcomp_axis_mode_override = haskey(ENV, "JFEM_Q4_PCOMP_KG_AXIS_MODE")
    flat_pcomp_plate_branch = q4_sol105_flat_pcomp_plate_branch_enabled()
    flat_pcomp_dkmq_branch = q4_sol105_flat_pcomp_dkmq_enabled()
    flat_pcomp_plate_auto = q4_sol105_flat_pcomp_plate_auto_enabled()
    flat_pcomp_plate_auto_d16_ratio_max = q4_sol105_flat_pcomp_plate_auto_d16_ratio_max()
    flat_pcomp_plate_auto_shear_ratio_max = q4_sol105_flat_pcomp_plate_auto_shear_ratio_max()
    flat_pcomp_plate_like_kg = q4_sol105_flat_pcomp_plate_like_kg_enabled()
    nonflat_pcomp_normal_only_kg = q4_sol105_nonflat_pcomp_normal_only_kg_enabled()
    flat_pcomp_rect_adini = q4_sol105_flat_pcomp_rect_adini_enabled()
    curved_iso_eig_membrane_incomp = q4_curved_iso_eig_auto_membrane_incomp_enabled()
    curved_iso_eig_membrane_incomp_kappa_l_min = q4_curved_iso_eig_auto_membrane_incomp_kappa_l_min()
    curved_iso_eig_membrane_incomp_cyl_ratio_max = q4_curved_iso_eig_auto_membrane_incomp_cyl_ratio_max()
    curved_iso_warp_membrane_incomp = q4_curved_iso_warp_membrane_incomp_enabled()
    curved_iso_warp_membrane_incomp_ratio_min = q4_curved_iso_warp_membrane_incomp_ratio_min()
    curved_iso_warp_membrane_incomp_kappa_l_max = q4_curved_iso_warp_membrane_incomp_kappa_l_max()
    curved_iso_elongated_membrane_incomp = q4_curved_iso_elongated_membrane_incomp_enabled()
    curved_iso_geomnormal_frame = q4_curved_iso_geomnormal_frame_enabled()
    curved_iso_geomnormal_frame_aspect_ratio_min = q4_curved_iso_geomnormal_frame_aspect_ratio_min()
    curved_iso_geomnormal_frame_kappa_l_min = q4_curved_iso_geomnormal_frame_kappa_l_min()
    curved_iso_geomnormal_frame_kappa_l_max = q4_curved_iso_geomnormal_frame_kappa_l_max()
    curved_iso_geomnormal_frame_cyl_ratio_max = q4_curved_iso_geomnormal_frame_cyl_ratio_max()
    curved_iso_elongated_membrane_incomp_aspect_ratio_min = q4_curved_iso_elongated_membrane_incomp_aspect_ratio_min()
    kg_flat_pcomp_auto_g12 = !kg_pcomp_axis_mode_override && q4_flat_pcomp_auto_g12_enabled()
    kg_flat_pcomp_auto_g12_kappa_l_max = q4_flat_pcomp_auto_g12_kappa_l_max()
    kg_flat_pcomp_auto_g12_cyl_ratio_max = q4_flat_pcomp_auto_g12_cyl_ratio_max()
    kg_flat_pcomp_auto_g12_shear_ratio_max = q4_pcomp_auto_global_x_shear_ratio_max()
    kg_flat_pcomp_auto_g12_d16_ratio_max = q4_pcomp_auto_global_x_d16_ratio_max()
    kg_flat_pcomp_auto_g12_b_ratio_max = q4_pcomp_auto_global_x_b_ratio_max()
    kg_pcomp_auto_global_x = !kg_pcomp_axis_mode_override && q4_pcomp_auto_global_x_enabled()
    kg_pcomp_auto_global_x_shear_ratio_max = q4_pcomp_auto_global_x_shear_ratio_max()
    kg_pcomp_auto_global_x_d16_ratio_max = q4_pcomp_auto_global_x_d16_ratio_max()
    kg_pcomp_auto_global_x_b_ratio_max = q4_pcomp_auto_global_x_b_ratio_max()
    kg_pcomp_auto_global_x_cyl_ratio_min = q4_pcomp_auto_global_x_cyl_ratio_min()
    kg_pcomp_auto_global_x_kappa_l_min = q4_pcomp_auto_global_x_kappa_l_min()
    kg_pcomp_auto_g12 = !kg_pcomp_axis_mode_override && q4_pcomp_kg_auto_g12_enabled()
    kg_auto_curvature_pcomp = kg_trans_mode !== :curvature && q4_pcomp_kg_auto_curvature_enabled()
    kg_auto_curvature_iso = kg_trans_mode !== :curvature && q4_shell_kg_auto_curvature_iso_enabled()

    cshells = model["CSHELLs"]
    cbars   = model["CBARs"]
    cbeams  = get(model, "CBEAMs", Dict())
    crods   = get(model, "CRODs", Dict())
    conrods = get(model, "CONRODs", Dict())
    model_has_line_elements = !isempty(cbars) || !isempty(cbeams) || !isempty(crods) || !isempty(conrods)
    pshells = model["PSHELLs"]; pbarls = model["PBARLs"]; mats = model["MATs"]
    auto_pcomp_membrane_incomp_model = any(q4_sol105_pcomp_auto_membrane_incomp_candidate, values(pshells))

    n_nodes = length(id_map)
    max_nid = maximum(keys(id_map))
    id_vec = zeros(Int, max_nid)
    for (nid, idx) in id_map; id_vec[nid] = idx; end

    snorm_vec = fill(SVector(0.0, 0.0, 0.0), n_nodes)
    snorm_has = falses(n_nodes)
    snorm_normals_local = isempty(strip(snorm_override)) ? snorm_normals : compute_snorm_normals(model, id_map, node_coords)
    for (idx, nrm) in snorm_normals_local; snorm_vec[idx] = nrm; snorm_has[idx] = true; end
    geom_normals_local =
        (kg_trans_mode === :curvature || kg_pcomp_auto_g12 || kg_pcomp_auto_global_x ||
         kg_auto_curvature_pcomp || kg_auto_curvature_iso || curved_iso_geomnormal_frame ||
         mitc4_3d_kg_recovery) ?
        compute_geometric_nodal_normals(model, id_map, node_coords) :
        Dict{Int,SVector{3,Float64}}()
    geom_vec = fill(SVector(0.0, 0.0, 0.0), n_nodes)
    geom_has = falses(n_nodes)
    for (idx, nrm) in geom_normals_local; geom_vec[idx] = nrm; geom_has[idx] = true; end
    node_has_line = build_node_has_line_elements(model, id_map, n_nodes)

    # Convert node_R to flat 3D array
    node_R_flat = zeros(3, 3, n_nodes)
    for i in 1:n_nodes
        for r in 1:3, c in 1:3; node_R_flat[r,c,i] = node_R[i][r,c]; end
    end
    shell_valence = zeros(Int, n_nodes)
    for (_, el) in cshells
        for nid in el["NODES"]
            idx = get(id_map, nid, 0)
            idx > 0 && (shell_valence[idx] += 1)
        end
    end
    I_idx = Vector{Int}(); J_idx = Vector{Int}(); V_val = Vector{Float64}()

    # --- QUAD4 geometric stiffness ---
    shell_keys = collect(keys(cshells))
    shell_list = [cshells[k] for k in shell_keys]
    shell_eids = [something(tryparse(Int, string(k)), 0) for k in shell_keys]
    n_shells = length(shell_list)
    # Count and pre-extract QUAD4/TRIA3 elements (same logic as assemble_stiffness)
    n_q4 = 0; n_t3 = 0
    for el in shell_list
        pid = string(el["PID"])
        if !haskey(pshells, pid); continue; end
        prop = pshells[pid]; mid = string(prop["MID"])
        if !haskey(mats, mid); continue; end
        nids = el["NODES"]; n = length(nids)
        valid = true
        for k in 1:n
            nid = nids[k]
            if nid < 1 || nid > max_nid || id_vec[nid] == 0; valid = false; break; end
        end
        if !valid; continue; end
        if n == 4; n_q4 += 1; elseif n == 3; n_t3 += 1; end
    end

    est_total = n_q4*576 + n_t3*324 + length(cbars)*144 + length(cbeams)*144 + length(crods)*144 + length(conrods)*144
    sizehint!(I_idx, est_total); sizehint!(J_idx, est_total); sizehint!(V_val, est_total)

    # --- Parallel shell Kg assembly ---
    # Per-thread scratch: each Julia thread that may run an iteration gets
    # its own bank of 24×24 / 4×3 / 4×2 buffers so they don't collide.
    # Sized by maxthreadid() (safe upper bound for threadid()); on a 1-thread
    # run each vector has length 1 and the loop executes sequentially with no
    # synchronization overhead.
    nt_kg = Threads.maxthreadid()
    T_buf_tl         = [zeros(24, 24)           for _ in 1:nt_kg]
    lc_buf4_tl       = [zeros(4, 2)             for _ in 1:nt_kg]
    dofs_buf24_tl    = [Vector{Int}(undef, 24)  for _ in 1:nt_kg]
    u_elem24_tl      = [zeros(24)               for _ in 1:nt_kg]
    Kg_global_tl     = [zeros(24, 24)           for _ in 1:nt_kg]
    tmp24a_tl        = [zeros(24, 24)           for _ in 1:nt_kg]
    N_gp_eff_tl      = [zeros(4, 3)             for _ in 1:nt_kg]
    coords3d_buf4_tl = [zeros(4, 3)             for _ in 1:nt_kg]
    coords3d_local_buf4_tl = [zeros(4, 3)       for _ in 1:nt_kg]
    directors3d_local_buf4_tl = [zeros(4, 3)    for _ in 1:nt_kg]

    # Thread-local COO triplet accumulators — concatenated into the master
    # I/J/V after the loop. Capacity hint is each thread's share of est_total.
    thread_I = [Int[]     for _ in 1:nt_kg]
    thread_J = [Int[]     for _ in 1:nt_kg]
    thread_V = [Float64[] for _ in 1:nt_kg]
    _per_thread_cap = cld(max(est_total, 1), nt_kg) + 1024
    for t in 1:nt_kg
        sizehint!(thread_I[t], _per_thread_cap)
        sizehint!(thread_J[t], _per_thread_cap)
        sizehint!(thread_V[t], _per_thread_cap)
    end

    # Per-thread scalar reductions (summed after the loop).
    diag_Nxx_sum_tl = zeros(nt_kg)
    diag_Nyy_sum_tl = zeros(nt_kg)
    diag_Nxy_sum_tl = zeros(nt_kg)
    diag_count_tl   = zeros(Int, nt_kg)
    n_q4_done_tl    = zeros(Int, nt_kg)
    kg_diag_pid_enabled = solver_env_bool("JFEM_KG_DIAG_PID", false)
    # JFEM_KG_DIAG_EID_CSV: when set to a writable path, the Kg loop dumps
    # per-element membrane forces (Nxx, Nyy, Nxy) to a CSV after Kg assembly.
    # Used to compare σ-recovery across kernel modes for the same external
    # load (see 2026-05-12 entry in SOL105 parity TODO).
    kg_diag_eid_csv_path = strip(get(ENV, "JFEM_KG_DIAG_EID_CSV", ""))
    kg_diag_eid_enabled = !isempty(kg_diag_eid_csv_path)
    kg_diag_subcase = buckling_subcase === nothing ? "" : string(buckling_subcase)
    kg_eid_rows_tl = [NamedTuple[] for _ in 1:nt_kg]
    kg_pid_count_tl = [Dict{Int,Int}() for _ in 1:nt_kg]
    kg_pid_nxx_tl = [Dict{Int,Float64}() for _ in 1:nt_kg]
    kg_pid_nyy_tl = [Dict{Int,Float64}() for _ in 1:nt_kg]
    kg_pid_nxy_tl = [Dict{Int,Float64}() for _ in 1:nt_kg]

    # The per-element kernel only does small (≤24×24) matmuls where BLAS
    # threading is pure overhead. Pin to 1 BLAS thread while the @threads
    # loop is active, then restore so Cholesky/Krylov downstream can still
    # use all cores.
    kg_timings["setup"] = (time_ns() - kg_t_setup) * 1e-9
    kg_t_shells = time_ns()
    _prev_blas_threads_kg = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)

    log_msg("[SOLVER] Assembling Kg shells ($(Threads.nthreads()) Julia thread$(Threads.nthreads()==1 ? "" : "s"))")

    Threads.@threads :static for _shell_ei in 1:length(shell_list)
        tid = Threads.threadid()
        el = shell_list[_shell_ei]
        let T_buf         = T_buf_tl[tid],
            lc_buf4       = lc_buf4_tl[tid],
            dofs_buf24    = dofs_buf24_tl[tid],
            u_elem24      = u_elem24_tl[tid],
            Kg_global     = Kg_global_tl[tid],
            tmp24a        = tmp24a_tl[tid],
            N_gp_eff      = N_gp_eff_tl[tid],
            coords3d_buf4 = coords3d_buf4_tl[tid],
            coords3d_local_buf4 = coords3d_local_buf4_tl[tid],
            directors3d_local_buf4 = directors3d_local_buf4_tl[tid],
            I_idx         = thread_I[tid],
            J_idx         = thread_J[tid],
            V_val         = thread_V[tid]

        pid = string(el["PID"])
        if !haskey(pshells, pid); continue; end
        prop = pshells[pid]; mid = string(prop["MID"])
        nids = el["NODES"]; n = length(nids)
        if !haskey(mats, mid); continue; end
        is_pcomp_clt = get(prop, "TYPE", "") == "PCOMP_CLT" && haskey(prop, "Cm")
        base_mat = mats[mid]
        mat = is_pcomp_clt ? base_mat : _effective_mat1_for_nodes(model, mid, nids)
        h = Float64(prop["T"])
        pcomp_is_isotropic = is_pcomp_clt && get(prop, "IS_ISOTROPIC", false)
        is_ortho = !is_pcomp_clt && get(mat, "TYPE", "") == "MAT8" && haskey(mat, "E1") && haskey(mat, "E2")
        is_mat2  = !is_pcomp_clt && !is_ortho && get(mat, "TYPE", "") == "MAT2" && haskey(mat, "G11")
        is_iso_kg = pcomp_is_isotropic || (!is_pcomp_clt && !is_ortho && !is_mat2)

        valid = true
        for k in 1:n
            nid = nids[k]
            if nid < 1 || nid > max_nid || id_vec[nid] == 0; valid = false; break; end
        end
        if !valid; continue; end

        if n == 4
            # QUAD4 geometric stiffness
            i1 = id_vec[nids[1]]; i2 = id_vec[nids[2]]; i3 = id_vec[nids[3]]; i4 = id_vec[nids[4]]
            p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
            p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
            p3 = SVector{3}(node_coords[i3,1], node_coords[i3,2], node_coords[i3,3])
            p4 = SVector{3}(node_coords[i4,1], node_coords[i4,2], node_coords[i4,3])
            d13_geom = p3 - p1
            d24_geom = p4 - p2
            v3_geom_raw = cross(d13_geom, d24_geom)
            v3_geom_len = norm(v3_geom_raw)
            local elem_is_flat_kg::Bool
            if v3_geom_len > 1e-12
                c_geom = (p1 + p2 + p3 + p4) / 4.0
                v3g = v3_geom_raw / v3_geom_len
                max_dev = max(abs(dot(p1-c_geom, v3g)), abs(dot(p2-c_geom, v3g)),
                              abs(dot(p3-c_geom, v3g)), abs(dot(p4-c_geom, v3g)))
                L_diag = max(norm(d13_geom), norm(d24_geom))
                elem_is_flat_kg = max_dev < 1e-6 * max(L_diag, 1e-12)
            else
                elem_is_flat_kg = true
                max_dev = 0.0
                L_diag = max(norm(d13_geom), norm(d24_geom))
            end
            warp_ratio_kg = max_dev / max(L_diag, 1e-12)
            v1, v2, v3 = shell_element_frame_quad4(p1, p2, p3, p4, q4_frame_mode)

            # SNORM adjustment
            n_avg = SVector(0.0, 0.0, 0.0); nc = 0
            for idx in (i1, i2, i3, i4)
                if snorm_has[idx]; n_avg = n_avg + snorm_vec[idx]; nc += 1; end
            end
            if nc > 0
                n_avg_s = n_avg / nc; len_s = norm(n_avg_s)
                if len_s > 1e-12
                    v3n = SVector{3}(n_avg_s / len_s)
                    if dot(v3n, v3) < 0.0; v3n = -v3n; end
                    v1p = v1 - dot(v1, v3n) * v3n; v1l = norm(v1p)
                    if v1l > 1e-12
                        v1n = SVector{3}(v1p / v1l)
                    else
                        v2p = v2 - dot(v2, v3n) * v3n; v1n = SVector{3}(normalize(v2p))
                    end
                    v1, v2, v3 = v1n, SVector{3}(cross(v3n, v1n)), v3n
                end
            end
            if curved_iso_geomnormal_frame &&
               is_iso_kg &&
               !elem_is_flat_kg &&
               (shell_valence[i1] + shell_valence[i2] + shell_valence[i3] + shell_valence[i4]) <= 10 &&
               geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
                lc_probe = zeros(4,2)
                c_probe = (p1 + p2 + p3 + p4) / 4.0
                lc_probe[1,1] = dot(p1-c_probe, v1); lc_probe[1,2] = dot(p1-c_probe, v2)
                lc_probe[2,1] = dot(p2-c_probe, v1); lc_probe[2,2] = dot(p2-c_probe, v2)
                lc_probe[3,1] = dot(p3-c_probe, v1); lc_probe[3,2] = dot(p3-c_probe, v2)
                lc_probe[4,1] = dot(p4-c_probe, v1); lc_probe[4,2] = dot(p4-c_probe, v2)
                aspect_ratio_probe = q4_local_edge_aspect_ratio(lc_probe)
                geom_curvature_probe = estimate_quad4_curvature_membrane(
                    lc_probe, geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4], v1, v2, v3
                )
                k1_probe, _ = q4_curvature_principal_abs(geom_curvature_probe)
                kappa_l_probe = k1_probe * q4_curvature_characteristic_length(lc_probe)
                cyl_ratio_probe = q4_curvature_cyl_ratio(geom_curvature_probe)
                if aspect_ratio_probe >= curved_iso_geomnormal_frame_aspect_ratio_min &&
                   kappa_l_probe >= curved_iso_geomnormal_frame_kappa_l_min &&
                   kappa_l_probe <= curved_iso_geomnormal_frame_kappa_l_max &&
                   cyl_ratio_probe <= curved_iso_geomnormal_frame_cyl_ratio_max
                    v3_geom_sum = geom_vec[i1] + geom_vec[i2] + geom_vec[i3] + geom_vec[i4]
                    if norm(v3_geom_sum) > 1e-12
                        v3_geom_frame = normalize(v3_geom_sum)
                        if dot(v3_geom_frame, v3) < 0.0
                            v3_geom_frame = -v3_geom_frame
                        end
                        v1, v2, v3 = shell_element_frame_quad4_with_normal(
                            p1, p2, p3, p4, v3_geom_frame, q4_frame_mode
                        )
                    end
                end
            end
            Rel_t = @SMatrix [v1[1] v1[2] v1[3]; v2[1] v2[2] v2[3]; v3[1] v3[2] v3[3]]

            # Local coordinates
            c_ctr = (p1 + p2 + p3 + p4) / 4.0
            lc_buf4[1,1] = dot(p1-c_ctr, v1); lc_buf4[1,2] = dot(p1-c_ctr, v2)
            lc_buf4[2,1] = dot(p2-c_ctr, v1); lc_buf4[2,2] = dot(p2-c_ctr, v2)
            lc_buf4[3,1] = dot(p3-c_ctr, v1); lc_buf4[3,2] = dot(p3-c_ctr, v2)
            lc_buf4[4,1] = dot(p4-c_ctr, v1); lc_buf4[4,2] = dot(p4-c_ctr, v2)
            coords3d_local_buf4[1,1] = lc_buf4[1,1]; coords3d_local_buf4[1,2] = lc_buf4[1,2]; coords3d_local_buf4[1,3] = dot(p1-c_ctr, v3)
            coords3d_local_buf4[2,1] = lc_buf4[2,1]; coords3d_local_buf4[2,2] = lc_buf4[2,2]; coords3d_local_buf4[2,3] = dot(p2-c_ctr, v3)
            coords3d_local_buf4[3,1] = lc_buf4[3,1]; coords3d_local_buf4[3,2] = lc_buf4[3,2]; coords3d_local_buf4[3,3] = dot(p3-c_ctr, v3)
            coords3d_local_buf4[4,1] = lc_buf4[4,1]; coords3d_local_buf4[4,2] = lc_buf4[4,2]; coords3d_local_buf4[4,3] = dot(p4-c_ctr, v3)
            aspect_ratio_kg = q4_local_edge_aspect_ratio(lc_buf4)
            taper_ratio_kg = q4_local_opposite_edge_ratio(lc_buf4)
            use_geom_snorm_kg = false
            if curved_iso_geomnormal_frame &&
               is_iso_kg &&
               aspect_ratio_kg >= curved_iso_geomnormal_frame_aspect_ratio_min &&
               (shell_valence[i1] + shell_valence[i2] + shell_valence[i3] + shell_valence[i4]) <= 10 &&
               geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
                geom_curv_probe = estimate_quad4_curvature_membrane(
                    lc_buf4, geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4], v1, v2, v3
                )
                k1_probe, _ = q4_curvature_principal_abs(geom_curv_probe)
                kappa_l_probe = k1_probe * q4_curvature_characteristic_length(lc_buf4)
                cyl_ratio_probe = q4_curvature_cyl_ratio(geom_curv_probe)
                if elem_is_flat_kg
                    use_geom_snorm_kg =
                        kappa_l_probe >= flat_curved_iso_geomnormal_frame_kappa_l_min &&
                        kappa_l_probe <= flat_curved_iso_geomnormal_frame_kappa_l_max &&
                        cyl_ratio_probe <= flat_curved_iso_geomnormal_frame_cyl_ratio_max
                else
                    use_geom_snorm_kg =
                        kappa_l_probe >= curved_iso_geomnormal_frame_kappa_l_min &&
                        kappa_l_probe <= curved_iso_geomnormal_frame_kappa_l_max &&
                        cyl_ratio_probe <= curved_iso_geomnormal_frame_cyl_ratio_max
                end
            end
            if use_geom_snorm_kg
                n_avg_g = geom_vec[i1] + geom_vec[i2] + geom_vec[i3] + geom_vec[i4]
                len_g = norm(n_avg_g)
                if len_g > 1e-12
                    v3n = SVector{3}(n_avg_g / len_g)
                    if dot(v3n, v3) < 0.0
                        v3n = -v3n
                    end
                    v1p = v1 - dot(v1, v3n) * v3n
                    v1l = norm(v1p)
                    if v1l > 1e-12
                        v1n = SVector{3}(v1p / v1l)
                    else
                        v2p = v2 - dot(v2, v3n) * v3n
                        v1n = SVector{3}(normalize(v2p))
                    end
                    v1, v2, v3 = v1n, SVector{3}(cross(v3n, v1n)), v3n
                    lc_buf4[1,1] = dot(p1-c_ctr, v1); lc_buf4[1,2] = dot(p1-c_ctr, v2)
                    lc_buf4[2,1] = dot(p2-c_ctr, v1); lc_buf4[2,2] = dot(p2-c_ctr, v2)
                    lc_buf4[3,1] = dot(p3-c_ctr, v1); lc_buf4[3,2] = dot(p3-c_ctr, v2)
                    lc_buf4[4,1] = dot(p4-c_ctr, v1); lc_buf4[4,2] = dot(p4-c_ctr, v2)
                    coords3d_local_buf4[1,1] = lc_buf4[1,1]; coords3d_local_buf4[1,2] = lc_buf4[1,2]; coords3d_local_buf4[1,3] = dot(p1-c_ctr, v3)
                    coords3d_local_buf4[2,1] = lc_buf4[2,1]; coords3d_local_buf4[2,2] = lc_buf4[2,2]; coords3d_local_buf4[2,3] = dot(p2-c_ctr, v3)
                    coords3d_local_buf4[3,1] = lc_buf4[3,1]; coords3d_local_buf4[3,2] = lc_buf4[3,2]; coords3d_local_buf4[3,3] = dot(p3-c_ctr, v3)
                    coords3d_local_buf4[4,1] = lc_buf4[4,1]; coords3d_local_buf4[4,2] = lc_buf4[4,2]; coords3d_local_buf4[4,3] = dot(p4-c_ctr, v3)
                    aspect_ratio_kg = q4_local_edge_aspect_ratio(lc_buf4)
                    taper_ratio_kg = q4_local_opposite_edge_ratio(lc_buf4)
                end
            end
            edge_skew_kg = q4_local_edge_skew_angle(lc_buf4)
            elem_mitc4_3d_kg_recovery =
                if mitc4_3d_kg_recovery_all
                    true
                elseif mitc4_3d_kg_recovery_aspect
                    kappa_l_mitc4_3d_aspect_kg = 0.0
                    if geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
                        geom_curv_mitc4_kg = estimate_quad4_curvature_membrane(
                            lc_buf4, geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4], v1, v2, v3
                        )
                        k1_mitc4_kg, _ = q4_curvature_principal_abs(geom_curv_mitc4_kg)
                        kappa_l_mitc4_3d_aspect_kg =
                            k1_mitc4_kg * q4_curvature_characteristic_length(lc_buf4)
                    end
                    q4_mitc4_3d_aspect_geom_ok(
                        aspect_ratio_kg,
                        warp_ratio_kg,
                        kappa_l_mitc4_3d_aspect_kg,
                        edge_skew_kg,
                        mitc4_3d_aspect_min_kg,
                        mitc4_3d_aspect_max_kg,
                        mitc4_3d_aspect_warp_min_kg,
                        mitc4_3d_aspect_warp_max_kg,
                        mitc4_3d_aspect_kappa_l_min_kg,
                        mitc4_3d_aspect_kappa_l_max_kg,
                        mitc4_3d_aspect_skew_min_kg,
                        mitc4_3d_aspect_skew_max_kg,
                        mitc4_3d_aspect_skew_aspect_min_kg,
                    ) &&
                    (!mitc4_3d_aspect_pcomp_only_kg || is_pcomp_clt)
                else
                    false
                end
            mitc4_3d_use_geom_dirs_kg =
                elem_mitc4_3d_kg_recovery && geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
            if elem_mitc4_3d_kg_recovery
                for (row, ncurv) in enumerate((
                    mitc4_3d_use_geom_dirs_kg ? geom_vec[i1] : (snorm_has[i1] ? snorm_vec[i1] : v3),
                    mitc4_3d_use_geom_dirs_kg ? geom_vec[i2] : (snorm_has[i2] ? snorm_vec[i2] : v3),
                    mitc4_3d_use_geom_dirs_kg ? geom_vec[i3] : (snorm_has[i3] ? snorm_vec[i3] : v3),
                    mitc4_3d_use_geom_dirs_kg ? geom_vec[i4] : (snorm_has[i4] ? snorm_vec[i4] : v3),
                ))
                    nloc = SVector(dot(ncurv, v1), dot(ncurv, v2), dot(ncurv, v3))
                    if nloc[3] < 0.0
                        nloc = -nloc
                    end
                    nlen = norm(nloc)
                    if nlen > 1e-12
                        directors3d_local_buf4[row,1] = nloc[1] / nlen
                        directors3d_local_buf4[row,2] = nloc[2] / nlen
                        directors3d_local_buf4[row,3] = nloc[3] / nlen
                    else
                        directors3d_local_buf4[row,1] = 0.0
                        directors3d_local_buf4[row,2] = 0.0
                        directors3d_local_buf4[row,3] = 1.0
                    end
                end
            end
            elem_flat_curved_iso_nodal_geomnormal_transform_kg =
                flat_curved_iso_nodal_geomnormal_transform &&
                is_iso_kg &&
                elem_is_flat_kg &&
                use_geom_snorm_kg &&
                aspect_ratio_kg >= flat_curved_iso_nodal_geomnormal_transform_aspect_ratio_min &&
                (shell_valence[i1] + shell_valence[i2] + shell_valence[i3] + shell_valence[i4]) <=
                    flat_curved_iso_nodal_geomnormal_transform_valence_sum_max &&
                geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
            iso_auto_curvature_resolution_ok = !is_iso_kg ||
                aspect_ratio_kg <= q4_shell_kg_auto_curvature_iso_aspect_ratio_max()
            iso_geom_curvature_kg = nothing
            iso_corner_curvature_kg = nothing
            kappa_l_iso_kg = 0.0
            cyl_ratio_iso_kg = 1.0
            auto_curved_iso_membrane_incomp_kg = false
            auto_warped_iso_membrane_incomp_kg = false
            auto_elongated_iso_membrane_incomp_kg = false
            if curved_iso_eig_membrane_incomp &&
               !is_pcomp_clt &&
               !is_ortho &&
               !is_mat2 &&
               iso_auto_curvature_resolution_ok &&
               geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
                iso_geom_curvature_kg = estimate_quad4_curvature_membrane(
                    lc_buf4, geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4], v1, v2, v3
                )
                k1_iso_kg, _ = q4_curvature_principal_abs(iso_geom_curvature_kg)
                kappa_l_iso_kg = k1_iso_kg * q4_curvature_characteristic_length(lc_buf4)
                cyl_ratio_iso_kg = q4_curvature_cyl_ratio(iso_geom_curvature_kg)
                auto_curved_iso_membrane_incomp_kg =
                    kappa_l_iso_kg >= curved_iso_eig_membrane_incomp_kappa_l_min &&
                    cyl_ratio_iso_kg <= curved_iso_eig_membrane_incomp_cyl_ratio_max
                auto_warped_iso_membrane_incomp_kg =
                    curved_iso_warp_membrane_incomp &&
                    !elem_is_flat_kg &&
                    warp_ratio_kg >= curved_iso_warp_membrane_incomp_ratio_min &&
                    kappa_l_iso_kg <= curved_iso_warp_membrane_incomp_kappa_l_max
            end
            if flat_iso_dkmq_branch && is_iso_kg && !elem_is_flat_kg
                iso_corner_curvature_kg = estimate_quad4_corner_curvature_membrane(
                    lc_buf4, p1, p2, p3, p4, v1, v2, v3
                )
            end
            auto_elongated_iso_membrane_incomp_kg =
                curved_iso_elongated_membrane_incomp &&
                is_iso_kg &&
                !elem_is_flat_kg &&
                aspect_ratio_kg >= curved_iso_elongated_membrane_incomp_aspect_ratio_min
            elem_membrane_incomp_kg = membrane_incomp || auto_curved_iso_membrane_incomp_kg ||
                                      auto_warped_iso_membrane_incomp_kg ||
                                      auto_elongated_iso_membrane_incomp_kg ||
                                      ((kg_pcomp_membrane_incomp ||
                                        (auto_pcomp_membrane_incomp_model && is_pcomp_clt)) && is_pcomp_clt) ||
                                      (flat_iso_eig_membrane_incomp && elem_is_flat_kg && is_iso_kg)
            curvature_membrane = nothing
            kg_curvature = nothing
            kg_trans_mode_eff = kg_trans_mode
            if kg_trans_mode_eff === :principal_transverse &&
               kg_principal_transverse_flat_only &&
               (!elem_is_flat_kg || warp_ratio_kg > kg_principal_transverse_warp_ratio_max)
                kg_trans_mode_eff = :all
            end
            kg_curvature_sign_eff = kg_curvature_sign
            # The rotational-gradient prestress term is a curved-shell correction.
            # On exactly flat shells it should not contribute to the plate buckling operator.
            kg_rot_grad_scale_eff = elem_is_flat_kg ? 0.0 : kg_rot_grad_scale
            covariant_membrane_candidate = false
            curvature_scale = q4_curvature_membrane_scale("JFEM_Q4_CURVATURE_MEMBRANE_SCALE_KG")
            n1_curv_kg = use_geom_snorm_kg ? geom_vec[i1] : snorm_vec[i1]
            n2_curv_kg = use_geom_snorm_kg ? geom_vec[i2] : snorm_vec[i2]
            n3_curv_kg = use_geom_snorm_kg ? geom_vec[i3] : snorm_vec[i3]
            n4_curv_kg = use_geom_snorm_kg ? geom_vec[i4] : snorm_vec[i4]
            has_curv_normals_kg = use_geom_snorm_kg ||
                                  (snorm_has[i1] && snorm_has[i2] && snorm_has[i3] && snorm_has[i4])
            if curvature_scale > 0.0 && has_curv_normals_kg
                curvature_raw = estimate_quad4_curvature_membrane(
                    lc_buf4, n1_curv_kg, n2_curv_kg, n3_curv_kg, n4_curv_kg, v1, v2, v3
                )
                curvature_weight = q4_curvature_filter_weight(
                    curvature_raw,
                    q4_curvature_filter_mode("JFEM_Q4_CURVATURE_FILTER_MODE_KG"),
                    q4_curvature_cyl_ratio_max("JFEM_Q4_CURVATURE_CYL_RATIO_MAX_KG"),
                )
                curvature_weight *= q4_curvature_resolution_weight(
                    curvature_raw, lc_buf4,
                    q4_curvature_resolution_min("JFEM_Q4_CURVATURE_RESOLUTION_MIN_KG"),
                    q4_curvature_resolution_full("JFEM_Q4_CURVATURE_RESOLUTION_FULL_KG"),
                )
                if curvature_weight > 0.0
                    curvature_membrane = curvature_raw * (curvature_scale * curvature_weight)
                end
            end
            geom_curvature = nothing
            if (kg_trans_mode === :curvature ||
                (kg_pcomp_auto_g12 && is_pcomp_clt && !pcomp_is_isotropic) ||
                (kg_auto_curvature_pcomp && is_pcomp_clt && !pcomp_is_isotropic) ||
                (kg_auto_curvature_iso && is_iso_kg)) &&
               geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4]
                geom_curvature = estimate_quad4_curvature_membrane(
                    lc_buf4, geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4], v1, v2, v3
                )
            end
            if is_iso_kg && kg_trans_mode !== :curvature && !iso_auto_curvature_resolution_ok
                geom_curvature = nothing
            end
            if kg_trans_mode === :curvature
                kg_curvature = geom_curvature
            end
            if kg_auto_curvature_iso && is_iso_kg && geom_curvature !== nothing
                k1, _ = q4_curvature_principal_abs(geom_curvature)
                kappa_l = k1 * q4_curvature_characteristic_length(lc_buf4)
                cyl_ratio = q4_curvature_cyl_ratio(geom_curvature)
                aspect_ratio = q4_local_edge_aspect_ratio(lc_buf4)
                covariant_membrane_candidate = covariant_membrane_candidate ||
                    (kappa_l >= kg_covariant_auto_kappa_l_min &&
                     cyl_ratio <= kg_covariant_auto_cyl_ratio_max &&
                     aspect_ratio <= q4_shell_kg_auto_curvature_iso_aspect_ratio_max())
                if kg_auto_curvature_iso_cyl_candidate(kappa_l, cyl_ratio, aspect_ratio)
                    kg_trans_mode_eff = :curvature
                    kg_curvature = geom_curvature * q4_shell_kg_auto_curvature_iso_cyl_scale()
                    kg_curvature_sign_eff = q4_shell_kg_auto_curvature_iso_cyl_sign()
                elseif kg_auto_curvature_iso_candidate(kappa_l, cyl_ratio, aspect_ratio)
                    kg_trans_mode_eff = :curvature
                    kg_curvature = geom_curvature * q4_shell_kg_auto_curvature_iso_effective_scale(cyl_ratio, kappa_l)
                    kg_curvature_sign_eff = q4_shell_kg_auto_curvature_iso_sign()
                end
                if kg_rot_grad_auto_iso_scale > 0.0 &&
                   kappa_l >= kg_rot_grad_auto_kappa_l_min &&
                   cyl_ratio >= kg_rot_grad_auto_cyl_ratio_min &&
                   aspect_ratio <= q4_shell_kg_auto_curvature_iso_aspect_ratio_max()
                    kg_rot_grad_scale_eff = max(kg_rot_grad_scale_eff, kg_rot_grad_auto_iso_scale)
                end
            end
            auto_gp_patch_candidate = false
            if kg_auto_gp_patch && geom_curvature !== nothing
                k1_patch, _ = q4_curvature_principal_abs(geom_curvature)
                kappa_l_patch = k1_patch * q4_curvature_characteristic_length(lc_buf4)
                max_valence = max(shell_valence[i1], shell_valence[i2], shell_valence[i3], shell_valence[i4])
                auto_gp_patch_candidate = max_valence >= kg_auto_gp_patch_valence_min &&
                    kappa_l_patch >= kg_auto_gp_patch_kappa_l_min
            end

            # Build transformation matrix T (24x24)
            fill!(T_buf, 0.0)
            for k in 1:4
                idx = k == 1 ? i1 : k == 2 ? i2 : k == 3 ? i3 : i4
                base = (k-1)*6
                vk1, vk2, vk3 =
                    elem_flat_curved_iso_nodal_geomnormal_transform_kg ?
                    shell_project_frame_to_normal(v1, v2, v3, geom_vec[idx]) :
                    (v1, v2, v3)
                Rel_t = @SMatrix [vk1[1] vk1[2] vk1[3]; vk2[1] vk2[2] vk2[3]; vk3[1] vk3[2] vk3[3]]
                for rr in 1:3, cc in 1:3
                    val = Rel_t[rr,1]*node_R_flat[1,cc,idx] + Rel_t[rr,2]*node_R_flat[2,cc,idx] + Rel_t[rr,3]*node_R_flat[3,cc,idx]
                    T_buf[base+rr, base+cc] = val
                    T_buf[base+3+rr, base+3+cc] = val
                end
            end

            # Extract element displacements in local coordinates
            for k in 1:4
                idx = k == 1 ? i1 : k == 2 ? i2 : k == 3 ? i3 : i4
                b_g = (idx-1)*6
                b_l = (k-1)*6
                for d in 1:6; u_elem24[b_l+d] = 0.0; end
                for d in 1:6
                    ug = u_global[b_g+d]
                    for dd in 1:6; u_elem24[b_l+dd] += T_buf[b_l+dd, b_l+d] * ug; end
                end
            end

            # Get material properties for stress recovery
            E_val = get(mat, "E", 70000.0); nu_val = get(mat, "NU", 0.3)
            t_shell = h

            # For PCOMP, rotate the laminate Cm into the element frame using
            # the shell THETA angle measured from the element x-axis.
            Cm_override = nothing
            Cb_kg = nothing
            Cs_kg = nothing
            Bmb_kg = nothing
            kg_axis_mode_eff = kg_pcomp_axis_mode
            kg_material_shear_rotation = 0.0
            shear_ratio = 0.0
            d16_ratio = 0.0
            b_ratio = 0.0
            if is_pcomp_clt && !pcomp_is_isotropic
                theta_deg_metrics = Float64(get(el, "THETA", 0.0))
                shear_ratio, d16_ratio, b_ratio = pcomp_metric_ratios(prop, deg2rad(theta_deg_metrics))
            end
            kg_flat_curved_iso_exact_membrane = false
            kg_flat_iso_exact_membrane = flat_iso_dkmq_branch &&
                                         !model_has_line_elements &&
                                         is_iso_kg &&
                                         elem_is_flat_kg &&
                                         get(prop, "Bmb", nothing) === nothing &&
                                         !node_has_line[i1] && !node_has_line[i2] &&
                                         !node_has_line[i3] && !node_has_line[i4] &&
                                         geom_has[i1] && geom_has[i2] && geom_has[i3] && geom_has[i4] &&
                                         q4_geom_normals_nearly_constant(
                                             geom_vec[i1], geom_vec[i2], geom_vec[i3], geom_vec[i4]
                                         )
            kg_saddle_iso_exact_membrane = false
            kg_cyl_iso_exact_membrane = flat_iso_dkmq_branch &&
                                        is_iso_kg &&
                                        !elem_is_flat_kg &&
                                        get(prop, "Bmb", nothing) === nothing &&
                                        iso_corner_curvature_kg !== nothing &&
                                        abs(q4_curvature_gaussian(iso_corner_curvature_kg)) <= 1e-10 &&
                                        first(q4_curvature_principal_abs(iso_corner_curvature_kg)) > 1e-8
            if flat_iso_dkmq_branch &&
               is_iso_kg &&
               elem_is_flat_kg &&
               get(prop, "Bmb", nothing) === nothing &&
               iso_geom_curvature_kg !== nothing
                kg_flat_curved_iso_exact_membrane =
                    flat_curved_iso_eig_center_only &&
                    kappa_l_iso_kg >= flat_curved_iso_eig_center_only_kappa_l_min &&
                    cyl_ratio_iso_kg <= flat_curved_iso_eig_center_only_cyl_ratio_max
            end
            kg_iso_exact_membrane =
                kg_flat_iso_exact_membrane || kg_flat_curved_iso_exact_membrane ||
                kg_saddle_iso_exact_membrane || kg_cyl_iso_exact_membrane
            if curvature_membrane === nothing && kg_cyl_iso_exact_membrane && iso_corner_curvature_kg !== nothing
                # Keep the cylindrical exact-membrane branch consistent between
                # the static/eigen stiffness and the geometric stiffness recovery.
                curvature_membrane = iso_corner_curvature_kg
            end
            kg_flat_dkmq_branch = flat_pcomp_dkmq_branch &&
                                  is_pcomp_clt &&
                                  !pcomp_is_isotropic &&
                                  elem_is_flat_kg &&
                                  get(prop, "Bmb", nothing) === nothing &&
                                  maximum(abs, prop["Cb"]) > 1e-30
            kg_flat_plate_auto = is_pcomp_clt &&
                                 flat_pcomp_plate_auto &&
                                 !pcomp_is_isotropic &&
                                 elem_is_flat_kg &&
                                 get(prop, "Bmb", nothing) === nothing &&
                                 maximum(abs, prop["Cb"]) > 1e-30 &&
                                 FEM.quad4_is_axis_aligned_rectangle(lc_buf4) &&
                                 d16_ratio <= flat_pcomp_plate_auto_d16_ratio_max &&
                                 shear_ratio <= flat_pcomp_plate_auto_shear_ratio_max
            kg_flat_plate_branch = is_pcomp_clt &&
                                   (flat_pcomp_plate_branch || kg_flat_plate_auto) &&
                                   !pcomp_is_isotropic &&
                                   elem_is_flat_kg &&
                                   get(prop, "Bmb", nothing) === nothing &&
                                   maximum(abs, prop["Cb"]) > 1e-30
            if is_pcomp_clt
                theta_deg = Float64(get(el, "THETA", 0.0))
                theta_rad = deg2rad(theta_deg)
                if (kg_flat_pcomp_auto_g12 || kg_pcomp_auto_g12 || kg_auto_curvature_pcomp) &&
                   !get(prop, "IS_ISOTROPIC", false) && geom_curvature !== nothing
                    k1, _ = q4_curvature_principal_abs(geom_curvature)
                    kappa_l = k1 * q4_curvature_characteristic_length(lc_buf4)
                    cyl_ratio = q4_curvature_cyl_ratio(geom_curvature)
                    auto_pcomp_element_axis_kg =
                        !kg_pcomp_axis_mode_override &&
                        q4_sol105_pcomp_auto_element_axis_candidate(prop, kappa_l, cyl_ratio)
                    covariant_membrane_candidate = covariant_membrane_candidate ||
                        (kappa_l >= kg_covariant_auto_kappa_l_min && cyl_ratio <= kg_covariant_auto_cyl_ratio_max)
                    if auto_pcomp_element_axis_kg
                        kg_axis_mode_eff = :element
                    elseif kg_pcomp_auto_global_x &&
                           !elem_is_flat_kg &&
                           shear_ratio <= kg_pcomp_auto_global_x_shear_ratio_max &&
                           d16_ratio <= kg_pcomp_auto_global_x_d16_ratio_max &&
                           b_ratio <= kg_pcomp_auto_global_x_b_ratio_max
                        if kg_pcomp_auto_global_x_kappa_l_min <= 0.0 &&
                           kg_pcomp_auto_global_x_cyl_ratio_min <= 0.0
                            kg_axis_mode_eff = :global_x
                        elseif kappa_l >= kg_pcomp_auto_global_x_kappa_l_min &&
                               cyl_ratio >= kg_pcomp_auto_global_x_cyl_ratio_min
                            kg_axis_mode_eff = :global_x
                        end
                    elseif kg_flat_pcomp_auto_g12 &&
                       shear_ratio <= kg_flat_pcomp_auto_g12_shear_ratio_max &&
                       d16_ratio <= kg_flat_pcomp_auto_g12_d16_ratio_max &&
                       b_ratio <= kg_flat_pcomp_auto_g12_b_ratio_max
                        kg_axis_mode_eff = :g12
                    elseif kg_pcomp_auto_g12 &&
                       kg_auto_pcomp_g12_candidate(theta_deg, shear_ratio, d16_ratio, b_ratio, kappa_l, cyl_ratio)
                        kg_axis_mode_eff = :g12
                    end
                    if kg_auto_curvature_pcomp &&
                       kg_auto_curvature_pcomp_candidate(theta_deg, shear_ratio, d16_ratio, b_ratio, kappa_l, cyl_ratio)
                        kg_trans_mode_eff = :curvature
                        kg_curvature = geom_curvature * q4_pcomp_kg_auto_curvature_scale()
                        kg_curvature_sign_eff = q4_pcomp_kg_auto_curvature_sign()
                    end
                    if kg_rot_grad_auto_pcomp_scale > 0.0 &&
                       kappa_l >= kg_rot_grad_auto_kappa_l_min &&
                       cyl_ratio >= kg_rot_grad_auto_cyl_ratio_min
                        kg_rot_grad_scale_eff = max(kg_rot_grad_scale_eff, kg_rot_grad_auto_pcomp_scale)
                    end
                end
                Cm_override = copy(prop["Cm"])
                Bmb_kg = get(prop, "Bmb", nothing) === nothing ? nothing : copy(prop["Bmb"])
                beta_kg = shell_pcomp_kg_rotation(
                    kg_axis_mode_eff,
                    v1, v2, v3, p1, p2, p3, p4,
                    theta_rad,
                    Int(get(el, "MCID", 0)),
                    model["CORDs"],
                )
                kg_material_shear_rotation = beta_kg
                if abs(beta_kg) > 1e-10
                    cb = cos(beta_kg); sb = sin(beta_kg)
                    c2 = cb^2; s2 = sb^2; cs = cb*sb
                    _rotate_constitutive_3x3!(Cm_override,
                        c2, s2, cs, s2, c2, -cs, -2cs, 2cs, c2-s2)
                    if Bmb_kg !== nothing
                        _rotate_constitutive_3x3!(Bmb_kg,
                            c2, s2, cs, s2, c2, -cs, -2cs, 2cs, c2-s2)
                    end
                end
                if kg_flat_plate_branch || kg_flat_dkmq_branch
                    Cb_kg = copy(prop["Cb"])
                    Cs_kg = copy(prop["Cs"])
                    if abs(beta_kg) > 1e-10
                        cb = cos(beta_kg); sb = sin(beta_kg)
                        c2 = cb^2; s2 = sb^2; cs = cb*sb
                        _rotate_constitutive_3x3!(Cb_kg,
                            c2, s2, cs, s2, c2, -cs, -2cs, 2cs, c2-s2)
                        a11 = Cs_kg[1,1]; a12 = Cs_kg[1,2]; a22 = Cs_kg[2,2]
                        Cs_kg[1,1] = cb^2*a11 + 2*cb*sb*a12 + sb^2*a22
                        Cs_kg[1,2] = -cb*sb*a11 + (cb^2-sb^2)*a12 + cb*sb*a22
                        Cs_kg[2,1] = Cs_kg[1,2]
                        Cs_kg[2,2] = sb^2*a11 - 2*cb*sb*a12 + cb^2*a22
                    end
                end
            end
            # Compute membrane resultants from the SOL101 displacement using the
            # same named membrane formulation selected for the eigen/Kg path. A
            # caller can still force compatible-only recovery through
            # JFEM_KG_USE_COMPATIBLE_MEMBRANE_STRESS for diagnostic isolation.
            kg_membrane_assumed_mode =
                if is_pcomp_clt && !pcomp_is_isotropic && elem_is_flat_kg && get(prop, "Bmb", nothing) === nothing
                    if flat_pcomp_taper_membrane_none &&
                       aspect_ratio_kg >= flat_pcomp_taper_membrane_none_aspect_min &&
                       taper_ratio_kg <= flat_pcomp_taper_membrane_none_ratio_max
                        :none
                    else
                        flat_pcomp_eig_membrane_assumed_mode
                    end
                elseif elem_is_flat_kg && is_iso_kg
                    flat_iso_eig_membrane_assumed_mode
                else
                    :none
                end
            if elem_mitc4_3d_kg_recovery
                Cm_mitc4_3d = if Cm_override === nothing
                    const_mem = E_val / (1 - nu_val^2)
                    (const_mem .* [1 nu_val 0; nu_val 1 0; 0 0 (1-nu_val)/2]) * h
                else
                    Cm_override
                end
                N_gp, N_res, _ = FEM.quad4_mitc4_3d_membrane_force_field(
                    coords3d_local_buf4,
                    directors3d_local_buf4,
                    u_elem24,
                    Cm_mitc4_3d,
                    h;
                    Bmb=Bmb_kg,
                )
            else
                N_gp, N_res, _ = FEM.quad4_membrane_force_field(
                    lc_buf4, u_elem24, E_val, nu_val, h;
                    Cm_override=Cm_override,
                    compatible_only=kg_compatible_membrane,
                    use_incompatible_modes=elem_membrane_incomp_kg && !kg_iso_exact_membrane,
                    use_enhanced_modes=kg_iso_exact_membrane,
                    curvature_membrane=curvature_membrane,
                    membrane_shear_center_row=false,
                    material_shear_rotation=kg_material_shear_rotation,
                    membrane_assumed_mode=kg_membrane_assumed_mode,
                    membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
                )
            end
            if !elem_mitc4_3d_kg_recovery &&
               kg_membrane_recovery_mode in (:tri_aspect, :tri_center_adj, :tri_incident_interp, :tri_diagavg)
                Cm_tri = if Cm_override === nothing
                    const_mem = E_val / (1 - nu_val^2)
                    (const_mem .* [1 nu_val 0; nu_val 1 0; 0 0 (1-nu_val)/2]) * h
                else
                    Cm_override
                end
                N_gp = FEM.quad4_membrane_force_field_triangle_recovery(
                    lc_buf4,
                    u_elem24,
                    Cm_tri,
                    N_res;
                    mode=kg_membrane_recovery_mode,
                    aspect_switch=kg_quad4_membrane_tri_aspect_switch(),
                )
            end
            if !elem_mitc4_3d_kg_recovery &&
               kg_compatible_membrane && kg_membrane_recovery_mode !== :planar && kg_covariant_blend > 0.0
                use_covariant = kg_membrane_recovery_mode === :covariant ||
                    (kg_membrane_recovery_mode === :auto && covariant_membrane_candidate)
                if use_covariant
                    coords3d = zeros(4, 3)
                    u_nodes_global = zeros(4, 3)
                    Cm_cov = if Cm_override === nothing
                        const_mem = E_val / (1 - nu_val^2)
                        (const_mem .* [1 nu_val 0; nu_val 1 0; 0 0 (1-nu_val)/2]) * h
                    else
                        Cm_override
                    end
                    for (kk, idx) in enumerate((i1, i2, i3, i4))
                        coords3d[kk, 1] = node_coords[idx, 1]
                        coords3d[kk, 2] = node_coords[idx, 2]
                        coords3d[kk, 3] = node_coords[idx, 3]
                        bg = (idx - 1) * 6
                        for rr in 1:3
                            u_nodes_global[kk, rr] =
                                node_R[idx][rr,1] * u_global[bg+1] +
                                node_R[idx][rr,2] * u_global[bg+2] +
                                node_R[idx][rr,3] * u_global[bg+3]
                        end
                    end
                    N_gp_cov, N_res_cov, _ = FEM.quad4_membrane_force_field_covariant(
                        coords3d, u_nodes_global, v1, v2, Cm_cov
                    )
                    if kg_covariant_blend >= 1.0
                        N_gp = N_gp_cov
                        N_res = N_res_cov
                    else
                        @inbounds for gp in 1:4, comp in 1:3
                            N_gp[gp, comp] = (1.0 - kg_covariant_blend) * N_gp[gp, comp] + kg_covariant_blend * N_gp_cov[gp, comp]
                        end
                        @inbounds for comp in 1:3
                            N_res[comp] = (1.0 - kg_covariant_blend) * N_res[comp] + kg_covariant_blend * N_res_cov[comp]
                        end
                    end
                end
            end
            gp_blend_override = kg_quad4_gp_field_blend_override()
            auto_avg_geom_ok = false
            auto_avg_curvature = geom_curvature
            if auto_avg_curvature === nothing && has_curv_normals_kg
                auto_avg_curvature = estimate_quad4_curvature_membrane(
                    lc_buf4, n1_curv_kg, n2_curv_kg, n3_curv_kg, n4_curv_kg, v1, v2, v3
                )
            end
            if auto_avg_curvature === nothing && !elem_is_flat_kg
                auto_avg_curvature = estimate_quad4_corner_curvature_membrane(
                    lc_buf4, p1, p2, p3, p4, v1, v2, v3
                )
            end
            if auto_avg_curvature !== nothing
                k1_auto_avg, _ = q4_curvature_principal_abs(auto_avg_curvature)
                kappa_l_auto_avg = k1_auto_avg * q4_curvature_characteristic_length(lc_buf4)
                cyl_ratio_auto_avg = q4_curvature_cyl_ratio(auto_avg_curvature)
                auto_avg_geom_ok =
                    kappa_l_auto_avg >= kg_quad4_auto_avg_kappa_l_min() &&
                    cyl_ratio_auto_avg >= kg_quad4_auto_avg_cyl_ratio_min()
            end
            use_gp_sigma = gp_blend_override === nothing &&
                kg_quad4_use_gp_field(N_gp, N_res, auto_avg_geom_ok, kg_quad4_auto_avg_load_ok)
            stress_mode_label = "average"
            gp_blend_alpha = 0.0
            shear_avg_candidate =
                kg_shear_avg_operator &&
                kg_quad4_shear_resultant_ratio(N_res) >= kg_shear_avg_ratio_min &&
                kg_quad4_geometry_gate(
                    warp_ratio_kg,
                    aspect_ratio_kg,
                    kg_shear_avg_warp_min,
                    kg_shear_avg_warp_max,
                    kg_shear_avg_aspect_min,
                    kg_shear_avg_aspect_max,
                    kg_shear_avg_geom_mode,
                )
            if !use_gp_sigma && auto_gp_patch_candidate
                use_gp_sigma = true
                stress_mode_label = "gauss_auto_patch"
            end
            if gp_blend_override === nothing && !use_gp_sigma && kg_auto_gp_spread && geom_curvature !== nothing
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
                    k1_gp, _ = q4_curvature_principal_abs(geom_curvature)
                    kappa_l_gp = k1_gp * q4_curvature_characteristic_length(lc_buf4)
                    cyl_ratio_gp = q4_curvature_cyl_ratio(geom_curvature)
                    if gp_spread >= kg_auto_gp_spread_min &&
                       kappa_l_gp >= kg_auto_gp_spread_kappa_l_min &&
                       cyl_ratio_gp >= kg_auto_gp_spread_cyl_ratio_min
                        gp_blend_scale = kg_quad4_auto_gp_spread_blend_scale()
                        if gp_blend_scale > 0.0
                            avg_norm = sqrt(N_res[1]^2 + N_res[2]^2 + N_res[3]^2)
                            avg_ratio = avg_norm / gp_mean_norm
                            gp_blend_alpha = clamp(gp_blend_scale * gp_spread * max(0.0, 1.0 - avg_ratio), 0.0, 1.0)
                        else
                            use_gp_sigma = true
                        end
                    end
                end
            end
            sigma_mem_input = if gp_blend_override !== nothing
                stress_mode_label = "override_blend"
                kg_quad4_blend_gp_field!(N_gp_eff, N_gp, N_res, gp_blend_override) ./ h
            elseif shear_avg_candidate
                stress_mode_label = "shear_average"
                N_res ./ h
            elseif use_gp_sigma
                if stress_mode_label == "average"
                    stress_mode_label = "gauss"
                end
                if kg_gp_extrapolate_scale != 1.0
                    stress_mode_label = "gauss_extrapolate"
                    kg_quad4_blend_gp_field!(N_gp_eff, N_gp, N_res, kg_gp_extrapolate_scale) ./ h
                else
                    N_gp ./ h
                end
            elseif gp_blend_alpha > 0.0
                stress_mode_label = "auto_blend"
                kg_quad4_blend_gp_field!(N_gp_eff, N_gp, N_res, gp_blend_alpha) ./ h
            else
                N_res ./ h
            end
            if kg_quad4_gp_field_pmin_spread_avg_alpha_v > 0.0 &&
               isfinite(kg_quad4_gp_field_pmin_spread_avg_min_v) &&
               sigma_mem_input isa AbstractMatrix &&
               kg_quad4_sigma_gp_pmin_spread_resultant(sigma_mem_input, h) >=
                   kg_quad4_gp_field_pmin_spread_avg_min_v
                gp_weight = 1.0 - kg_quad4_gp_field_pmin_spread_avg_alpha_v
                sigma_mem_input =
                    kg_quad4_blend_gp_field!(N_gp_eff, N_gp, N_res, gp_weight) ./ h
                stress_mode_label *= "_pminavg"
                gp_blend_alpha = gp_weight
            end
            if kg_shell_nxy_auto > 0.0 && geom_curvature !== nothing
                k1_auto, _ = q4_curvature_principal_abs(geom_curvature)
                kappa_l_auto = k1_auto * q4_curvature_characteristic_length(lc_buf4)
                cyl_ratio_auto = q4_curvature_cyl_ratio(geom_curvature)
                # The automatic Nxy relaxation is a curved-shell correction.
                # Keep it inactive on exactly flat geometry even when the env
                # threshold is zero, otherwise flat plates get an artificial
                # 1-relax reduction in shear prestress.
                if kappa_l_auto > max(kg_shell_nxy_auto_kappa_l_min_v, 1e-12) &&
                   cyl_ratio_auto <= kg_shell_nxy_auto_cyl_ratio_max_v
                    if sigma_mem_input isa AbstractMatrix
                        @inbounds for gp in 1:size(sigma_mem_input, 1)
                            sxx = sigma_mem_input[gp, 1]
                            syy = sigma_mem_input[gp, 2]
                            sxy = sigma_mem_input[gp, 3]
                            sigma_mem_input[gp, 3] *= kg_shell_nxy_auto_scale(
                                sxx, syy, sxy,
                                kg_shell_nxy_auto,
                                kg_shell_nxy_auto_ratio_min_v,
                                kg_shell_nxy_auto_ratio_full_v,
                            )
                        end
                    else
                        sigma_mem_input[3] *= kg_shell_nxy_auto_scale(
                            sigma_mem_input[1], sigma_mem_input[2], sigma_mem_input[3],
                            kg_shell_nxy_auto,
                            kg_shell_nxy_auto_ratio_min_v,
                            kg_shell_nxy_auto_ratio_full_v,
                        )
                    end
                end
            end
            if kg_shell_nxy != 1.0
                if sigma_mem_input isa AbstractMatrix
                    @inbounds for gp in 1:size(sigma_mem_input, 1)
                        sigma_mem_input[gp, 3] *= kg_shell_nxy
                    end
                else
                    sigma_mem_input[3] *= kg_shell_nxy
                end
            end
            if is_pcomp_clt
                kg_shell_pcomp_nxy_eff = kg_shell_pcomp_nxy
                if kg_shell_pcomp_nxy_aspect
                    kg_shell_pcomp_nxy_eff *= kg_shell_pcomp_nxy_aspect_scale(
                        q4_local_edge_aspect_ratio(lc_buf4),
                        kg_shell_pcomp_nxy_aspect_mode_v,
                        kg_shell_pcomp_nxy_aspect_low_v,
                        kg_shell_pcomp_nxy_aspect_mid_v,
                        kg_shell_pcomp_nxy_aspect_high_v,
                        kg_shell_pcomp_nxy_aspect_min_v,
                        kg_shell_pcomp_nxy_aspect_peak_v,
                        kg_shell_pcomp_nxy_aspect_max_v,
                    )
                end
                kg_shell_apply_pcomp_nxy_scale!(
                    sigma_mem_input,
                    kg_shell_pcomp_nxy_eff,
                    kg_shell_pcomp_nxy_compression_only_v,
                )
                kg_shell_apply_pcomp_nxy_shear_dom_scale!(
                    sigma_mem_input,
                    kg_shell_pcomp_nxy_shear_dom_relax_v,
                    kg_shell_pcomp_nxy_shear_dom_ratio_min_v,
                    kg_shell_pcomp_nxy_shear_dom_ratio_full_v,
                    aspect_ratio_kg,
                    kg_shell_pcomp_nxy_shear_dom_aspect_min_v,
                    kg_shell_pcomp_nxy_shear_dom_aspect_max_v,
                    kg_shell_pcomp_nxy_shear_dom_compression_only_v,
                )
            end
            if kg_quad4_membrane_scale != 1.0
                sigma_mem_input .*= kg_quad4_membrane_scale
            end
            feature_scale_diag_eff = 1.0
            feature_scale_diag_nxy_stat = 0.0
            feature_scale_diag_abs_nxy = 0.0
            feature_scale_diag_geom_ok = false
            feature_scale_diag_curv_ok = false
            feature_scale_diag_nxy_ok = false
            feature_scale_diag_abs_nxy_ok = false
            if kg_quad4_feature_membrane_scale != 1.0
                feature_scale_geom_ok = kg_quad4_geometry_gate(
                    warp_ratio_kg,
                    aspect_ratio_kg,
                    kg_quad4_feature_membrane_scale_warp_min_v,
                    kg_quad4_feature_membrane_scale_warp_max_v,
                    kg_quad4_feature_membrane_scale_aspect_min_v,
                    kg_quad4_feature_membrane_scale_aspect_max_v,
                    kg_quad4_feature_membrane_scale_geom_mode_v,
                )
                feature_scale_curv_ok = kg_quad4_feature_curvature_gate(
                    geom_curvature,
                    lc_buf4,
                    kg_quad4_feature_membrane_scale_kappa_l_min_v,
                    kg_quad4_feature_membrane_scale_kappa_l_max_v,
                )
                feature_scale_pcomp_ok = !kg_quad4_feature_membrane_scale_pcomp_only_v || is_pcomp_clt
                feature_scale_pid_ok =
                    isempty(kg_quad4_feature_membrane_scale_pids) ||
                    (something(tryparse(Int, string(pid)), 0) in kg_quad4_feature_membrane_scale_pids)
                feature_scale_nxy_stat_value = kg_quad4_sigma_nxy_stat(
                    sigma_mem_input,
                    kg_quad4_feature_membrane_scale_nxy_stat_v,
                )
                feature_scale_diag_nxy_stat = feature_scale_nxy_stat_value * h
                feature_scale_diag_abs_nxy = abs(feature_scale_diag_nxy_stat)
                feature_scale_nxy_ok = kg_quad4_component_sign_ok(
                    kg_quad4_feature_membrane_scale_nxy_sign_v,
                    feature_scale_nxy_stat_value,
                )
                feature_scale_abs_nxy_ok =
                    kg_quad4_feature_membrane_scale_abs_nxy_min_v <= 0.0 ||
                    abs(feature_scale_nxy_stat_value * h) >= kg_quad4_feature_membrane_scale_abs_nxy_min_v
                feature_scale_diag_geom_ok = feature_scale_geom_ok
                feature_scale_diag_curv_ok = feature_scale_curv_ok
                feature_scale_diag_nxy_ok = feature_scale_nxy_ok
                feature_scale_diag_abs_nxy_ok = feature_scale_abs_nxy_ok
                feature_scale_gate_ok = kg_quad4_feature_membrane_scale_nxy_mode_v === :extra_component ||
                                        feature_scale_nxy_ok
                if feature_scale_geom_ok && feature_scale_curv_ok &&
                   feature_scale_pcomp_ok && feature_scale_pid_ok && feature_scale_gate_ok &&
                   feature_scale_abs_nxy_ok
                    feature_scale_eff = kg_quad4_pid_membrane_effective_scale(
                        kg_quad4_feature_membrane_scale,
                        kg_quad4_feature_membrane_scale_nxx_sign_v,
                        sigma_mem_input,
                        h,
                        kg_quad4_feature_membrane_scale_gp_pmin_spread_min_v,
                        kg_quad4_feature_membrane_scale_gp_nxx_spread_min_v,
                        kg_quad4_feature_membrane_scale_gp_spread_factor_v,
                    )
                    feature_scale_diag_eff = feature_scale_eff
                    if feature_scale_eff != 1.0
                        kg_quad4_apply_feature_component_scale!(
                            sigma_mem_input,
                            feature_scale_eff,
                            kg_quad4_feature_membrane_scale_components_v,
                        )
                        if kg_quad4_feature_membrane_scale_nxy_mode_v === :extra_component &&
                           kg_quad4_feature_membrane_scale_nxy_sign_v !== :any &&
                           feature_scale_nxy_ok
                            kg_quad4_apply_feature_component_scale!(
                                sigma_mem_input,
                                feature_scale_eff,
                                :nxy,
                            )
                        end
                    end
                end
            end
            if kg_quad4_pid_membrane_scale_v2_ok_v && !isempty(kg_quad4_pid_membrane_scales)
                pid_int_scale = something(tryparse(Int, pid), 0)
                pid_scale = get(kg_quad4_pid_membrane_scales, pid_int_scale, 1.0)
                pid_scale_eff = kg_quad4_pid_membrane_effective_scale(
                    pid_scale,
                    kg_quad4_pid_membrane_scale_nxx_sign,
                    sigma_mem_input,
                    h,
                    kg_quad4_pid_membrane_scale_gp_pmin_spread_min_v,
                    kg_quad4_pid_membrane_scale_gp_nxx_spread_min_v,
                    kg_quad4_pid_membrane_scale_gp_spread_factor_v,
                )
                if pid_scale_eff != 1.0
                    kg_quad4_apply_feature_component_scale!(
                        sigma_mem_input,
                        pid_scale_eff,
                        kg_quad4_pid_membrane_scale_components_v,
                    )
                end
            end
            kg_shell_apply_axial_component_scale!(
                sigma_mem_input,
                kg_shell_nxx,
                kg_shell_nyy,
                kg_shell_axial_dom_min,
                N_res,
            )

            has_bmb_kg = get(prop, "Bmb", nothing) !== nothing
            kg_pcomp_normal_only_diag = false
            kg_saddle_diag = false
            kg_trans_mode_diag, kg_pcomp_normal_only_diag, kg_saddle_diag =
                q4_pcomp_kg_trans_mode_final(
                    kg_trans_mode_eff,
                    is_pcomp_clt,
                    pcomp_is_isotropic,
                    has_bmb_kg,
                    elem_is_flat_kg,
                    flat_pcomp_plate_like_kg,
                    nonflat_pcomp_normal_only_kg,
                    geom_curvature,
                )
            geom_curvature_ok_diag = geom_curvature !== nothing
            geom_kappa_l_diag = 0.0
            geom_cyl_ratio_diag = 1.0
            geom_gaussian_diag = 0.0
            if geom_curvature_ok_diag
                k1_diag, _ = q4_curvature_principal_abs(geom_curvature)
                geom_kappa_l_diag = k1_diag * q4_curvature_characteristic_length(lc_buf4)
                geom_cyl_ratio_diag = q4_curvature_cyl_ratio(geom_curvature)
                geom_gaussian_diag = q4_curvature_gaussian(geom_curvature)
            end

            diag_Nxx_sum_tl[tid] += N_res[1]; diag_Nyy_sum_tl[tid] += N_res[2]; diag_Nxy_sum_tl[tid] += N_res[3]; diag_count_tl[tid] += 1
            if kg_diag_pid_enabled
                pid_int = something(tryparse(Int, pid), 0)
                if pid_int != 0
                    kg_pid_count_tl[tid][pid_int] = get(kg_pid_count_tl[tid], pid_int, 0) + 1
                    kg_pid_nxx_tl[tid][pid_int] = get(kg_pid_nxx_tl[tid], pid_int, 0.0) + N_res[1]
                    kg_pid_nyy_tl[tid][pid_int] = get(kg_pid_nyy_tl[tid], pid_int, 0.0) + N_res[2]
                    kg_pid_nxy_tl[tid][pid_int] = get(kg_pid_nxy_tl[tid], pid_int, 0.0) + N_res[3]
                end
            end
            if kg_diag_eid_enabled
                eid_int = shell_eids[_shell_ei]
                pid_int_e = something(tryparse(Int, pid), 0)
                nin_xx = 0.0
                nin_yy = 0.0
                nin_xy = 0.0
                nin_gp1_xx = 0.0; nin_gp1_yy = 0.0; nin_gp1_xy = 0.0
                nin_gp2_xx = 0.0; nin_gp2_yy = 0.0; nin_gp2_xy = 0.0
                nin_gp3_xx = 0.0; nin_gp3_yy = 0.0; nin_gp3_xy = 0.0
                nin_gp4_xx = 0.0; nin_gp4_yy = 0.0; nin_gp4_xy = 0.0
                if sigma_mem_input isa AbstractMatrix
                    ngp_in = size(sigma_mem_input, 1)
                    @inbounds for gp in 1:ngp_in
                        gxx = sigma_mem_input[gp, 1] * h
                        gyy = sigma_mem_input[gp, 2] * h
                        gxy = sigma_mem_input[gp, 3] * h
                        nin_xx += gxx
                        nin_yy += gyy
                        nin_xy += gxy
                        if gp == 1
                            nin_gp1_xx = gxx; nin_gp1_yy = gyy; nin_gp1_xy = gxy
                        elseif gp == 2
                            nin_gp2_xx = gxx; nin_gp2_yy = gyy; nin_gp2_xy = gxy
                        elseif gp == 3
                            nin_gp3_xx = gxx; nin_gp3_yy = gyy; nin_gp3_xy = gxy
                        elseif gp == 4
                            nin_gp4_xx = gxx; nin_gp4_yy = gyy; nin_gp4_xy = gxy
                        end
                    end
                    inv_ngp_in = 1.0 / max(ngp_in, 1)
                    nin_xx *= inv_ngp_in
                    nin_yy *= inv_ngp_in
                    nin_xy *= inv_ngp_in
                else
                    nin_xx = sigma_mem_input[1] * h
                    nin_yy = sigma_mem_input[2] * h
                    nin_xy = sigma_mem_input[3] * h
                    nin_gp1_xx = nin_xx; nin_gp1_yy = nin_yy; nin_gp1_xy = nin_xy
                    nin_gp2_xx = nin_xx; nin_gp2_yy = nin_yy; nin_gp2_xy = nin_xy
                    nin_gp3_xx = nin_xx; nin_gp3_yy = nin_yy; nin_gp3_xy = nin_xy
                    nin_gp4_xx = nin_xx; nin_gp4_yy = nin_yy; nin_gp4_xy = nin_xy
                end
                push!(kg_eid_rows_tl[tid], (
                    subcase=kg_diag_subcase,
                    eid=eid_int,
                    pid=pid_int_e,
                    stress_mode=stress_mode_label,
                    blend_alpha=gp_blend_override === nothing ? gp_blend_alpha : gp_blend_override,
                    nres_xx=N_res[1],
                    nres_yy=N_res[2],
                    nres_xy=N_res[3],
                    nin_xx=nin_xx,
                    nin_yy=nin_yy,
                    nin_xy=nin_xy,
                    nin_gp1_xx=nin_gp1_xx,
                    nin_gp1_yy=nin_gp1_yy,
                    nin_gp1_xy=nin_gp1_xy,
                    nin_gp2_xx=nin_gp2_xx,
                    nin_gp2_yy=nin_gp2_yy,
                    nin_gp2_xy=nin_gp2_xy,
                    nin_gp3_xx=nin_gp3_xx,
                    nin_gp3_yy=nin_gp3_yy,
                    nin_gp3_xy=nin_gp3_xy,
                    nin_gp4_xx=nin_gp4_xx,
                    nin_gp4_yy=nin_gp4_yy,
                    nin_gp4_xy=nin_gp4_xy,
                    feature_scale_eff=feature_scale_diag_eff,
                    feature_nxy_stat=feature_scale_diag_nxy_stat,
                    feature_abs_nxy=feature_scale_diag_abs_nxy,
                    feature_geom_ok=feature_scale_diag_geom_ok,
                    feature_curv_ok=feature_scale_diag_curv_ok,
                    feature_nxy_ok=feature_scale_diag_nxy_ok,
                    feature_abs_nxy_ok=feature_scale_diag_abs_nxy_ok,
                    kg_trans_mode=kg_trans_mode_diag,
                    kg_pcomp_normal_only=kg_pcomp_normal_only_diag,
                    kg_saddle=kg_saddle_diag,
                    elem_is_flat=elem_is_flat_kg,
                    aspect=aspect_ratio_kg,
                    warp_ratio=warp_ratio_kg,
                    geom_curvature_ok=geom_curvature_ok_diag,
                    geom_kappa_l=geom_kappa_l_diag,
                    geom_cyl_ratio=geom_cyl_ratio_diag,
                    geom_gaussian=geom_gaussian_diag,
                    is_pcomp=is_pcomp_clt,
                    pcomp_is_isotropic=pcomp_is_isotropic,
                ))
            end
            n_q4_done_tl[tid] += 1

            # Compute geometric stiffness in local coordinates
            if kg_flat_dkmq_branch
                Kg_loc = FEM.geometric_stiffness_quad4_plate_dkmq(
                    lc_buf4, sigma_mem_input, h, Cb_kg === nothing ? prop["Cb"] : Cb_kg, Cs_kg === nothing ? prop["Cs"] : Cs_kg
                )
            elseif flat_pcomp_rect_adini && kg_flat_plate_branch && FEM.quad4_is_axis_aligned_rectangle(lc_buf4)
                Kg_loc = FEM.geometric_stiffness_quad4_plate_adini(
                    lc_buf4, sigma_mem_input, h
                )
            elseif kg_flat_plate_branch
                Kg_loc = FEM.geometric_stiffness_quad4_plate_dkq(
                    lc_buf4, sigma_mem_input, h, Cb_kg, Cs_kg
                )
            elseif kg_surface_operator_mode === :covariant
                coords3d_buf4[1,1] = p1[1]; coords3d_buf4[1,2] = p1[2]; coords3d_buf4[1,3] = p1[3]
                coords3d_buf4[2,1] = p2[1]; coords3d_buf4[2,2] = p2[2]; coords3d_buf4[2,3] = p2[3]
                coords3d_buf4[3,1] = p3[1]; coords3d_buf4[3,2] = p3[2]; coords3d_buf4[3,3] = p3[3]
                coords3d_buf4[4,1] = p4[1]; coords3d_buf4[4,2] = p4[2]; coords3d_buf4[4,3] = p4[3]
                kg_trans_cov = kg_trans_mode_eff === :normal_only ? :normal_only : :all
                Kg_loc = FEM.geometric_stiffness_quad4_covariant(
                    coords3d_buf4, sigma_mem_input, h, v1, v2;
                    trans_mode=kg_trans_cov,
                    rot_grad_scale=kg_rot_grad_scale_eff,
                )
            else
                kg_membrane_shear_center_row =
                    flat_iso_eig_membrane_shear_center_row && elem_is_flat_kg && is_iso_kg
                Cm_kg =
                    if Cm_override === nothing
                        const_mem = E_val / (1 - nu_val^2)
                        (const_mem .* [1 nu_val 0; nu_val 1 0; 0 0 (1-nu_val)/2]) * h
                    else
                        Cm_override
                    end
                kg_consistent_membrane_incomp =
                    elem_membrane_incomp_kg &&
                    (is_pcomp_clt ? kg_pcomp_consistent_membrane_operator :
                                    kg_consistent_membrane_operator) &&
                    get(prop, "Bmb", nothing) === nothing &&
                    kg_trans_mode_eff !== :curvature
                kg_pcomp_normal_only_actual = false
                kg_trans_mode_eff, kg_pcomp_normal_only_actual, _ =
                    q4_pcomp_kg_trans_mode_final(
                        kg_trans_mode_eff,
                        is_pcomp_clt,
                        pcomp_is_isotropic,
                        has_bmb_kg,
                        elem_is_flat_kg,
                        flat_pcomp_plate_like_kg,
                        nonflat_pcomp_normal_only_kg,
                        geom_curvature,
                    )
                if kg_pcomp_normal_only_actual
                    kg_rot_grad_scale_eff = 0.0
                end
                principal_shear_geom_ok = kg_quad4_geometry_gate(
                    warp_ratio_kg,
                    aspect_ratio_kg,
                    kg_principal_shear_warp_min_v,
                    kg_principal_shear_warp_max_v,
                    kg_principal_shear_aspect_min_v,
                    kg_principal_shear_aspect_max_v,
                    kg_principal_shear_geom_mode_v,
                )
                principal_shear_yy_factor_eff = 1.0
                principal_shear_xy_factor_eff = 1.0
                principal_shear_z_factor_eff = 1.0
                if principal_shear_geom_ok
                    principal_shear_yy_factor_eff = kg_quad4_pid_membrane_effective_scale(
                        kg_principal_shear_yy_factor_v,
                        kg_principal_shear_feature_gate_v,
                        sigma_mem_input,
                        h,
                        kg_principal_shear_gp_pmin_spread_min_v,
                        kg_principal_shear_gp_nxx_spread_min_v,
                        kg_principal_shear_gp_spread_factor_v,
                    )
                    principal_shear_xy_factor_eff = kg_quad4_pid_membrane_effective_scale(
                        kg_principal_shear_xy_factor_v,
                        kg_principal_shear_feature_gate_v,
                        sigma_mem_input,
                        h,
                        kg_principal_shear_gp_pmin_spread_min_v,
                        kg_principal_shear_gp_nxx_spread_min_v,
                        kg_principal_shear_gp_spread_factor_v,
                    )
                    principal_shear_z_factor_eff = kg_quad4_pid_membrane_effective_scale(
                        kg_principal_shear_z_factor_v,
                        kg_principal_shear_feature_gate_v,
                        sigma_mem_input,
                        h,
                        kg_principal_shear_gp_pmin_spread_min_v,
                        kg_principal_shear_gp_nxx_spread_min_v,
                        kg_principal_shear_gp_spread_factor_v,
                    )
                end
                Kg_loc = FEM.geometric_stiffness_quad4(
                    lc_buf4, sigma_mem_input, h;
                    trans_mode=kg_trans_mode_eff,
                    curvature=kg_curvature,
                    curvature_sign=kg_curvature_sign_eff,
                    rot_grad_scale=kg_rot_grad_scale_eff,
                    membrane_shear_center_row=kg_membrane_shear_center_row,
                    Cm=Cm_kg,
                    membrane_incomp=kg_consistent_membrane_incomp && !kg_iso_exact_membrane,
                    membrane_enhanced=kg_iso_exact_membrane,
                    material_shear_rotation=kg_material_shear_rotation,
                    membrane_assumed_mode=kg_membrane_assumed_mode,
                    membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
                    principal_shear_yy_factor=principal_shear_yy_factor_eff,
                    principal_shear_xy_factor=principal_shear_xy_factor_eff,
                    principal_shear_z_factor=principal_shear_z_factor_eff,
                    principal_shear_ratio_min=principal_shear_geom_ok ? kg_principal_shear_ratio_min_v : 1.0,
                )
            end

            # Transform to global: Kg_global = T' * Kg_loc * T
            fill!(Kg_global, 0.0); fill!(tmp24a, 0.0)
            @inbounds @fastmath for jj in 1:24, ll in 1:24
                val = T_buf[ll, jj]
                if val != 0.0
                    for ii in 1:24; tmp24a[ii, jj] += Kg_loc[ii, ll] * val; end
                end
            end
            @inbounds @fastmath for jj in 1:24, ll in 1:24
                val = tmp24a[ll, jj]
                if val != 0.0
                    for ii in 1:24; Kg_global[ii, jj] += T_buf[ll, ii] * val; end
                end
            end

            # Optional: zero the drilling-DOF rows/cols of Kg_global.
            # Matches Nastran's 5-DOF-per-node Kg convention (1976 theoretical
            # manual §5.6): CQUAD4 geometric stiffness is not formed on the
            # drilling direction of each node_R frame. Kg_loc already has zero
            # drill rows/cols in element-local coords, but the T_buf transform
            # into the node_R frame can mix rot_grad_scale contributions from
            # local θx/θy into global θz per node. The gate below removes that
            # leakage when the formulation is not expected to carry any drill
            # contribution in Kg.
            if kg_shell_drill_zero
                @inbounds for k in 1:4
                    d = k*6
                    for r in 1:24; Kg_global[r, d] = 0.0; Kg_global[d, r] = 0.0; end
                end
            end

            # Map DOFs and accumulate triplets
            for k in 1:4
                idx = k == 1 ? i1 : k == 2 ? i2 : k == 3 ? i3 : i4
                b = (idx-1)*6
                for d in 1:6; dofs_buf24[(k-1)*6+d] = b+d; end
            end
            for cc in 1:24, rr in 1:24
                push!(I_idx, dofs_buf24[rr]); push!(J_idx, dofs_buf24[cc]); push!(V_val, Kg_global[rr,cc])
            end

        elseif n == 3
            # TRIA3 geometric stiffness
            i1 = id_vec[nids[1]]; i2 = id_vec[nids[2]]; i3 = id_vec[nids[3]]
            p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
            p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
            p3 = SVector{3}(node_coords[i3,1], node_coords[i3,2], node_coords[i3,3])

            v1, v2, v3 = shell_element_frame_fast(p1, p2, p3, SVector{3}(0.0,0.0,0.0), 3)

            # SNORM adjustment
            n_avg = SVector(0.0, 0.0, 0.0); nc = 0
            for idx in (i1, i2, i3)
                if snorm_has[idx]; n_avg = n_avg + snorm_vec[idx]; nc += 1; end
            end
            if nc > 0
                n_avg_s = n_avg / nc; len_s = norm(n_avg_s)
                if len_s > 1e-12
                    v3n = SVector{3}(n_avg_s / len_s)
                    if dot(v3n, v3) < 0.0; v3n = -v3n; end
                    v1p = v1 - dot(v1, v3n) * v3n; v1l = norm(v1p)
                    if v1l > 1e-12
                        v1n = SVector{3}(v1p / v1l)
                    else
                        v2p = v2 - dot(v2, v3n) * v3n; v1n = SVector{3}(normalize(v2p))
                    end
                    v1, v2, v3 = v1n, SVector{3}(cross(v3n, v1n)), v3n
                end
            end

            Rel_t = @SMatrix [v1[1] v1[2] v1[3]; v2[1] v2[2] v2[3]; v3[1] v3[2] v3[3]]

            # Local coordinates
            c_ctr = (p1 + p2 + p3) / 3.0
            lc3 = zeros(3, 2)
            lc3[1,1] = dot(p1-c_ctr, v1); lc3[1,2] = dot(p1-c_ctr, v2)
            lc3[2,1] = dot(p2-c_ctr, v1); lc3[2,2] = dot(p2-c_ctr, v2)
            lc3[3,1] = dot(p3-c_ctr, v1); lc3[3,2] = dot(p3-c_ctr, v2)

            # Build T (18x18)
            T18 = zeros(18, 18)
            for k in 1:3
                idx = k == 1 ? i1 : k == 2 ? i2 : i3
                TR = Rel_t * node_R[idx]
                base = (k-1)*6
                T18[base+1:base+3, base+1:base+3] = TR
                T18[base+4:base+6, base+4:base+6] = TR
            end

            # Extract element displacements in local coords
            u_elem18 = zeros(18)
            for k in 1:3
                idx = k == 1 ? i1 : k == 2 ? i2 : i3
                b_g = (idx-1)*6; b_l = (k-1)*6
                for d in 1:6
                    for dd in 1:6; u_elem18[b_l+dd] += T18[b_l+dd, b_l+d] * u_global[b_g+d]; end
                end
            end

            E_val = get(mat, "E", 70000.0); nu_val = get(mat, "NU", 0.3)
            br = get(prop, "BEND_RATIO", 1.0)

            # Use the same shell material-axis definition here as in the main
            # shell formulation, including MCID support for composite CTRIA3.
            Cm_override_t3 = nothing
            if is_pcomp_clt
                Cm_override_t3 = copy(prop["Cm"])
                tri_kg_axis_mode =
                    kg_pcomp_axis_mode === :warp_switch ? :element : kg_pcomp_axis_mode
                beta_kg = shell_pcomp_material_rotation(
                    tri_kg_axis_mode,
                    v1, v2, v3, p1, p2,
                    deg2rad(Float64(get(el, "THETA", 0.0))),
                    Int(get(el, "MCID", 0)),
                    model["CORDs"],
                )
                if abs(beta_kg) > 1e-10
                    cb = cos(beta_kg); sb = sin(beta_kg)
                    c2 = cb^2; s2 = sb^2; cs = cb*sb
                    _rotate_constitutive_3x3!(Cm_override_t3,
                        c2, s2, cs, s2, c2, -cs, -2cs, 2cs, c2-s2)
                end
            end

            N_res, _, _, _, _, _, _ = FEM.stress_strain_tria3(lc3, u_elem18, E_val, nu_val, h; bend_ratio=br, Cm_override=Cm_override_t3)
            if kg_diag_pid_enabled
                pid_int = something(tryparse(Int, pid), 0)
                if pid_int != 0
                    kg_pid_count_tl[tid][pid_int] = get(kg_pid_count_tl[tid], pid_int, 0) + 1
                    kg_pid_nxx_tl[tid][pid_int] = get(kg_pid_nxx_tl[tid], pid_int, 0.0) + N_res[1]
                    kg_pid_nyy_tl[tid][pid_int] = get(kg_pid_nyy_tl[tid], pid_int, 0.0) + N_res[2]
                    kg_pid_nxy_tl[tid][pid_int] = get(kg_pid_nxy_tl[tid], pid_int, 0.0) + N_res[3]
                end
            end
            sigma_mem = N_res ./ h
            if kg_shell_nxy != 1.0
                sigma_mem[3] *= kg_shell_nxy
            end
            if is_pcomp_clt
                kg_shell_apply_pcomp_nxy_scale!(
                    sigma_mem,
                    kg_shell_pcomp_nxy,
                    kg_shell_pcomp_nxy_compression_only_v,
                )
            end
            if kg_quad4_membrane_scale != 1.0
                sigma_mem .*= kg_quad4_membrane_scale
            end
            kg_shell_apply_axial_component_scale!(
                sigma_mem,
                kg_shell_nxx,
                kg_shell_nyy,
                kg_shell_axial_dom_min,
            )

            Kg_loc = FEM.geometric_stiffness_tria3(
                lc3, sigma_mem, h;
                trans_mode=kg_trans_mode,
                curvature=nothing,
                curvature_sign=kg_curvature_sign,
            )
            Kg18 = T18' * Kg_loc * T18

            # Optional drill-DOF zero (see QUAD4 branch for rationale).
            if kg_shell_drill_zero
                @inbounds for k in 1:3
                    d = k*6
                    for r in 1:18; Kg18[r, d] = 0.0; Kg18[d, r] = 0.0; end
                end
            end

            dofs_t3 = Vector{Int}(undef, 18)
            for k in 1:3
                idx = k == 1 ? i1 : k == 2 ? i2 : i3
                b = (idx-1)*6
                for d in 1:6; dofs_t3[(k-1)*6+d] = b+d; end
            end
            for cc in 1:18, rr in 1:18
                push!(I_idx, dofs_t3[rr]); push!(J_idx, dofs_t3[cc]); push!(V_val, Kg18[rr,cc])
            end
        end
        end  # let
    end  # Threads.@threads :static for _shell_ei

    # Restore BLAS threads so downstream Cholesky / Krylov can use all cores.
    LinearAlgebra.BLAS.set_num_threads(_prev_blas_threads_kg)

    # Concatenate per-thread COO accumulators into the master triplet arrays.
    # Total allocation is still O(est_total); we just deferred it until after
    # the parallel section so threads didn't fight over the shared push!.
    total_kg_nz = 0
    for t in 1:nt_kg
        total_kg_nz += length(thread_I[t])
    end
    sizehint!(I_idx, length(I_idx) + total_kg_nz)
    sizehint!(J_idx, length(J_idx) + total_kg_nz)
    sizehint!(V_val, length(V_val) + total_kg_nz)
    for t in 1:nt_kg
        append!(I_idx, thread_I[t])
        append!(J_idx, thread_J[t])
        append!(V_val, thread_V[t])
    end

    # Reduce thread-local scalar counters.
    n_q4_done    = sum(n_q4_done_tl)
    diag_Nxx_sum = sum(diag_Nxx_sum_tl)
    diag_Nyy_sum = sum(diag_Nyy_sum_tl)
    diag_Nxy_sum = sum(diag_Nxy_sum_tl)
    diag_count   = sum(diag_count_tl)

    log_msg("[SOLVER] Kg shells: $n_q4 QUAD4 + $n_t3 TRIA3")
    if diag_count > 0
        log_msg("[SOLVER] Kg Q4 avg membrane forces: Nxx=$(round(diag_Nxx_sum/diag_count, digits=4)), Nyy=$(round(diag_Nyy_sum/diag_count, digits=4)), Nxy=$(round(diag_Nxy_sum/diag_count, digits=4))")
    end
    if kg_diag_pid_enabled
        pid_count = Dict{Int,Int}()
        pid_nxx = Dict{Int,Float64}()
        pid_nyy = Dict{Int,Float64}()
        pid_nxy = Dict{Int,Float64}()
        for t in 1:nt_kg
            for (pid, count) in kg_pid_count_tl[t]
                pid_count[pid] = get(pid_count, pid, 0) + count
            end
            for (pid, val) in kg_pid_nxx_tl[t]
                pid_nxx[pid] = get(pid_nxx, pid, 0.0) + val
            end
            for (pid, val) in kg_pid_nyy_tl[t]
                pid_nyy[pid] = get(pid_nyy, pid, 0.0) + val
            end
            for (pid, val) in kg_pid_nxy_tl[t]
                pid_nxy[pid] = get(pid_nxy, pid, 0.0) + val
            end
        end
        rows = NamedTuple{(:pid,:count,:nxx,:nyy,:nxy,:pmin,:pmax),Tuple{Int,Int,Float64,Float64,Float64,Float64,Float64}}[]
        for (pid, count) in pid_count
            count <= 0 && continue
            nxx = get(pid_nxx, pid, 0.0) / count
            nyy = get(pid_nyy, pid, 0.0) / count
            nxy = get(pid_nxy, pid, 0.0) / count
            mean_n = 0.5 * (nxx + nyy)
            half_d = 0.5 * (nxx - nyy)
            radius = sqrt(half_d * half_d + nxy * nxy)
            push!(rows, (pid=pid, count=count, nxx=nxx, nyy=nyy, nxy=nxy,
                         pmin=mean_n - radius, pmax=mean_n + radius))
        end
        sort!(rows; by=r -> r.pmin)
        nshow = min(12, length(rows))
        if nshow > 0
            txt = join([string(r.pid, "(n=", r.count,
                               ",pmin=", round(r.pmin; sigdigits=4),
                               ",pmax=", round(r.pmax; sigdigits=4),
                               ",Nxx=", round(r.nxx; sigdigits=4),
                               ",Nyy=", round(r.nyy; sigdigits=4),
                               ",Nxy=", round(r.nxy; sigdigits=4), ")")
                        for r in rows[1:nshow]], "; ")
            log_msg("[SOLVER] Kg PID most-compressive membrane forces: $txt")
        end
    end
    if kg_diag_eid_enabled
        all_rows = NamedTuple[]
        for t in 1:nt_kg
            append!(all_rows, kg_eid_rows_tl[t])
        end
        sort!(all_rows; by = r -> (r.subcase, r.eid))
        try
            write_header = !isfile(kg_diag_eid_csv_path) || filesize(kg_diag_eid_csv_path) == 0
            open(kg_diag_eid_csv_path, write_header ? "w" : "a") do io
                if write_header
                    println(io, "subcase,eid,pid,stress_mode,blend_alpha,nres_xx,nres_yy,nres_xy,nin_xx,nin_yy,nin_xy,nin_gp1_xx,nin_gp1_yy,nin_gp1_xy,nin_gp2_xx,nin_gp2_yy,nin_gp2_xy,nin_gp3_xx,nin_gp3_yy,nin_gp3_xy,nin_gp4_xx,nin_gp4_yy,nin_gp4_xy,feature_scale_eff,feature_nxy_stat,feature_abs_nxy,feature_geom_ok,feature_curv_ok,feature_nxy_ok,feature_abs_nxy_ok,kg_trans_mode,kg_pcomp_normal_only,kg_saddle,elem_is_flat,aspect,warp_ratio,geom_curvature_ok,geom_kappa_l,geom_cyl_ratio,geom_gaussian,is_pcomp,pcomp_is_isotropic")
                end
                for r in all_rows
                    println(io, r.subcase, ",", r.eid, ",", r.pid, ",", r.stress_mode, ",", r.blend_alpha, ",",
                            r.nres_xx, ",", r.nres_yy, ",", r.nres_xy, ",",
                            r.nin_xx, ",", r.nin_yy, ",", r.nin_xy, ",",
                            r.nin_gp1_xx, ",", r.nin_gp1_yy, ",", r.nin_gp1_xy, ",",
                            r.nin_gp2_xx, ",", r.nin_gp2_yy, ",", r.nin_gp2_xy, ",",
                            r.nin_gp3_xx, ",", r.nin_gp3_yy, ",", r.nin_gp3_xy, ",",
                            r.nin_gp4_xx, ",", r.nin_gp4_yy, ",", r.nin_gp4_xy, ",",
                            r.feature_scale_eff, ",", r.feature_nxy_stat, ",", r.feature_abs_nxy, ",",
                            r.feature_geom_ok, ",", r.feature_curv_ok, ",", r.feature_nxy_ok, ",",
                            r.feature_abs_nxy_ok, ",",
                            r.kg_trans_mode, ",", r.kg_pcomp_normal_only, ",", r.kg_saddle, ",",
                            r.elem_is_flat, ",", r.aspect, ",", r.warp_ratio, ",",
                            r.geom_curvature_ok, ",", r.geom_kappa_l, ",", r.geom_cyl_ratio, ",",
                            r.geom_gaussian, ",", r.is_pcomp, ",", r.pcomp_is_isotropic)
                end
            end
            log_msg("[SOLVER] Kg per-EID σ dump: $(length(all_rows)) rows → $kg_diag_eid_csv_path")
        catch e
            log_msg("[SOLVER] WARNING: failed to write JFEM_KG_DIAG_EID_CSV → $kg_diag_eid_csv_path: $e")
        end
    end

    kg_timings["shells"] = (time_ns() - kg_t_shells) * 1e-9

    # --- CBAR geometric stiffness ---
    kg_t_bars = time_ns()
    n_bars = 0
    T12 = zeros(12, 12)
    for (id, bar) in cbars
        pid = string(bar["PID"])
        if !haskey(pbarls, pid); continue; end
        prop = pbarls[pid]; mid = string(prop["MID"])
        if !haskey(mats, mid); continue; end
        mat = _effective_mat1_for_nodes(model, mid, [bar["GA"], bar["GB"]])

        if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
        i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]

        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])

        wa, wb, has_offset, p1_eff, p2_eff = bar_offsets_and_endpoints(bar, p1, p2)
        L = norm(p2_eff - p1_eff)
        if L < 1e-9; continue; end
        vx = normalize(p2_eff - p1_eff)
        v_ref = resolve_bar_vref(bar, p1, id_map, node_coords)
        if norm(v_ref) < 1e-6
            v_ref = SVector(0.0,0.0,1.0)
            if abs(dot(vx, v_ref)) > 0.9; v_ref = SVector(0.0,1.0,0.0); end
        end
        vz = normalize(cross(vx, v_ref))
        vy = cross(vz, vx)
        Rel = vcat(vx', vy', vz')

        fill!(T12, 0.0)
        TR1 = Rel * node_R[i1]; TR2 = Rel * node_R[i2]
        T12[1:3, 1:3] = TR1; T12[4:6, 4:6] = TR1
        T12[7:9, 7:9] = TR2; T12[10:12, 10:12] = TR2
        if has_offset
            S_wa = skew3(wa); S_wb = skew3(wb)
            T12[1:3, 4:6] = -Rel * S_wa * node_R[i1]
            T12[7:9, 10:12] = -Rel * S_wb * node_R[i2]
        end

        # Extract local displacements
        u_bar = zeros(12)
        for d in 1:6; u_bar[d] = 0.0; u_bar[6+d] = 0.0; end
        b1 = (i1-1)*6; b2 = (i2-1)*6
        for d in 1:6
            for dd in 1:6
                u_bar[dd] += T12[dd, d] * u_global[b1+d]
                u_bar[6+dd] += T12[6+dd, 6+d] * u_global[b2+d]
            end
            # Cross terms from offset
            if has_offset
                for dd in 1:3
                    u_bar[dd] += T12[dd, 3+d] * u_global[b1+d]   # offset coupling
                    u_bar[6+dd] += T12[6+dd, 9+d] * u_global[b2+d]
                end
            end
        end

        # Compute axial force
        Iy = get(prop, "I2", get(prop, "I", 0.0))
        Iz = get(prop, "I1", get(prop, "I", 0.0))
        if Iy == 0.0; Iy = Iz; end
        if Iz == 0.0; Iz = Iy; end
        Iyz = Float64(get(prop, "I12", 0.0))
        K1 = get(prop, "K1", 0.0); K2 = get(prop, "K2", 0.0)
        As_y = (K1 > 0.0) ? K1 * prop["A"] : Inf
        As_z = (K2 > 0.0) ? K2 * prop["A"] : Inf
        forces = FEM.forces_frame3d(u_bar, L, prop["A"], Iy, Iz, prop["J"], mat["E"], mat["G"]; As_y=As_y, As_z=As_z, I12=Iyz)
        P = forces["axial"]  # Axial force (positive = tension)
        Kg_loc = FEM.geometric_stiffness_frame3d(L, P)
        Kg_bar = T12' * Kg_loc * T12

        dofs = [(i1-1)*6+k for k in 1:6]
        append!(dofs, [(i2-1)*6+k for k in 1:6])
        for c in 1:12, r in 1:12
            push!(I_idx, dofs[r]); push!(J_idx, dofs[c]); push!(V_val, Kg_bar[r,c])
        end
        n_bars += 1
    end

    # --- CBEAM geometric stiffness (identical to CBAR) ---
    n_beams = 0
    for (id, bar) in cbeams
        pid = string(bar["PID"])
        if !haskey(pbarls, pid); continue; end
        prop = pbarls[pid]; mid = string(prop["MID"])
        if !haskey(mats, mid); continue; end
        mat = _effective_mat1_for_nodes(model, mid, [bar["GA"], bar["GB"]])

        if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
        i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]

        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])

        wa, wb, has_offset, p1_eff, p2_eff = bar_offsets_and_endpoints(bar, p1, p2)
        L = norm(p2_eff - p1_eff)
        if L < 1e-9; continue; end
        vx = normalize(p2_eff - p1_eff)
        v_ref = resolve_bar_vref(bar, p1, id_map, node_coords)
        if norm(v_ref) < 1e-6
            v_ref = SVector(0.0,0.0,1.0)
            if abs(dot(vx, v_ref)) > 0.9; v_ref = SVector(0.0,1.0,0.0); end
        end
        vz = normalize(cross(vx, v_ref))
        vy = cross(vz, vx)
        Rel = vcat(vx', vy', vz')

        fill!(T12, 0.0)
        TR1 = Rel * node_R[i1]; TR2 = Rel * node_R[i2]
        T12[1:3, 1:3] = TR1; T12[4:6, 4:6] = TR1
        T12[7:9, 7:9] = TR2; T12[10:12, 10:12] = TR2
        if has_offset
            S_wa = skew3(wa); S_wb = skew3(wb)
            T12[1:3, 4:6] = -Rel * S_wa * node_R[i1]
            T12[7:9, 10:12] = -Rel * S_wb * node_R[i2]
        end

        u_bar = zeros(12)
        b1 = (i1-1)*6; b2 = (i2-1)*6
        for d in 1:6
            for dd in 1:6
                u_bar[dd] += T12[dd, d] * u_global[b1+d]
                u_bar[6+dd] += T12[6+dd, 6+d] * u_global[b2+d]
            end
            if has_offset
                for dd in 1:3
                    u_bar[dd] += T12[dd, 3+d] * u_global[b1+d]
                    u_bar[6+dd] += T12[6+dd, 9+d] * u_global[b2+d]
                end
            end
        end

        Iy = get(prop, "I2", get(prop, "I", 0.0))
        Iz = get(prop, "I1", get(prop, "I", 0.0))
        if Iy == 0.0; Iy = Iz; end; if Iz == 0.0; Iz = Iy; end
        Iyz = Float64(get(prop, "I12", 0.0))
        K1 = get(prop, "K1", 0.0); K2 = get(prop, "K2", 0.0)
        As_y = (K1 > 0.0) ? K1 * prop["A"] : Inf
        As_z = (K2 > 0.0) ? K2 * prop["A"] : Inf
        forces = FEM.forces_frame3d(u_bar, L, prop["A"], Iy, Iz, prop["J"], mat["E"], mat["G"]; As_y=As_y, As_z=As_z, I12=Iyz)
        P = forces["axial"]
        Kg_loc = FEM.geometric_stiffness_frame3d(L, P)
        Kg_bar = T12' * Kg_loc * T12

        dofs = [(i1-1)*6+k for k in 1:6]
        append!(dofs, [(i2-1)*6+k for k in 1:6])
        for c in 1:12, r in 1:12
            push!(I_idx, dofs[r]); push!(J_idx, dofs[c]); push!(V_val, Kg_bar[r,c])
        end
        n_beams += 1
    end
    if n_bars + n_beams > 0
        log_msg("[SOLVER] Kg bars: $n_bars CBAR + $n_beams CBEAM")
    end
    kg_timings["bars_beams"] = (time_ns() - kg_t_bars) * 1e-9

    # --- CROD geometric stiffness ---
    kg_t_rods = time_ns()
    prods = get(model, "PRODs", Dict())
    n_rods = 0
    for (id, rod) in crods
        pid = string(rod["PID"])
        if !haskey(prods, pid); continue; end
        prop = prods[pid]; mid = string(prop["MID"])
        if !haskey(mats, mid); continue; end
        mat = _effective_mat1_for_nodes(model, mid, [rod["GA"], rod["GB"]])

        if !haskey(id_map, rod["GA"]) || !haskey(id_map, rod["GB"]); continue; end
        i1, i2 = id_map[rod["GA"]], id_map[rod["GB"]]
        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
        L = norm(p2-p1)
        if L < 1e-9; continue; end
        vx = normalize(p2-p1)
        ref = abs(vx[3]) < 0.9 ? SVector(0.0,0.0,1.0) : SVector(0.0,1.0,0.0)
        vz = normalize(cross(vx, ref)); vy = cross(vz, vx)
        Rel = vcat(vx', vy', vz')

        fill!(T12, 0.0)
        TR1 = Rel * node_R[i1]; TR2 = Rel * node_R[i2]
        T12[1:3, 1:3] = TR1; T12[4:6, 4:6] = TR1
        T12[7:9, 7:9] = TR2; T12[10:12, 10:12] = TR2

        # Axial force P = E*A/L * (u2_x - u1_x) in local coords
        u_rod = zeros(12)
        b1 = (i1-1)*6; b2 = (i2-1)*6
        for d in 1:6
            for dd in 1:6
                u_rod[dd] += T12[dd, d] * u_global[b1+d]
                u_rod[6+dd] += T12[6+dd, 6+d] * u_global[b2+d]
            end
        end
        P = mat["E"] * prop["A"] / L * (u_rod[7] - u_rod[1])

        Kg_loc = FEM.geometric_stiffness_rod(L, P)
        Kg_rod = T12' * Kg_loc * T12

        dofs = [(i1-1)*6+k for k in 1:6]
        append!(dofs, [(i2-1)*6+k for k in 1:6])
        for c in 1:12, r in 1:12
            push!(I_idx, dofs[r]); push!(J_idx, dofs[c]); push!(V_val, Kg_rod[r,c])
        end
        n_rods += 1
    end

    # --- CONROD geometric stiffness ---
    n_conrods = 0
    for (id, rod) in conrods
        mid = string(rod["MID"])
        if !haskey(mats, mid); continue; end
        mat = _effective_mat1_for_nodes(model, mid, [rod["GA"], rod["GB"]])
        if !haskey(id_map, rod["GA"]) || !haskey(id_map, rod["GB"]); continue; end
        i1, i2 = id_map[rod["GA"]], id_map[rod["GB"]]
        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
        L = norm(p2-p1)
        if L < 1e-9; continue; end
        vx = normalize(p2-p1)
        ref = abs(vx[3]) < 0.9 ? SVector(0.0,0.0,1.0) : SVector(0.0,1.0,0.0)
        vz = normalize(cross(vx, ref)); vy = cross(vz, vx)
        Rel = vcat(vx', vy', vz')

        fill!(T12, 0.0)
        TR1 = Rel * node_R[i1]; TR2 = Rel * node_R[i2]
        T12[1:3, 1:3] = TR1; T12[4:6, 4:6] = TR1
        T12[7:9, 7:9] = TR2; T12[10:12, 10:12] = TR2

        u_rod = zeros(12)
        b1 = (i1-1)*6; b2 = (i2-1)*6
        for d in 1:6
            for dd in 1:6
                u_rod[dd] += T12[dd, d] * u_global[b1+d]
                u_rod[6+dd] += T12[6+dd, 6+d] * u_global[b2+d]
            end
        end
        P = mat["E"] * rod["A"] / L * (u_rod[7] - u_rod[1])

        Kg_loc = FEM.geometric_stiffness_rod(L, P)
        Kg_rod = T12' * Kg_loc * T12

        dofs = [(i1-1)*6+k for k in 1:6]
        append!(dofs, [(i2-1)*6+k for k in 1:6])
        for c in 1:12, r in 1:12
            push!(I_idx, dofs[r]); push!(J_idx, dofs[c]); push!(V_val, Kg_rod[r,c])
        end
        n_conrods += 1
    end
    if n_rods + n_conrods > 0
        log_msg("[SOLVER] Kg rods: $n_rods CROD + $n_conrods CONROD")
    end
    kg_timings["rods"] = (time_ns() - kg_t_rods) * 1e-9

    # --- SOLID geometric stiffness ---
    kg_t_solids = time_ns()
    csolids_kg = get(model, "CSOLIDs", Dict())
    psolids_kg = get(model, "PSOLIDs", Dict())
    n_tetra_kg = 0; n_hexa_kg = 0; n_penta_kg = 0
    coords_buf_kg = zeros(8, 3)
    T_buf_kg = zeros(24, 24)
    for (id, el) in csolids_kg
        pid = string(el["PID"])
        if !haskey(psolids_kg, pid); continue; end
        prop = psolids_kg[pid]; mid = string(prop["MID"])
        nids = el["NODES"]; nn = length(nids)
        if !haskey(mats, mid); continue; end
        mat = _effective_mat1_for_nodes(model, mid, nids)
        etype = get(el, "TYPE", "")

        valid = true
        for k in 1:nn
            if !haskey(id_map, nids[k]); valid = false; break; end
        end
        if !valid; continue; end

        for k in 1:nn
            idx = id_map[nids[k]]
            coords_buf_kg[k,1] = node_coords[idx,1]
            coords_buf_kg[k,2] = node_coords[idx,2]
            coords_buf_kg[k,3] = node_coords[idx,3]
        end

        E_mat = Float64(mat["E"]); nu_mat = Float64(mat["NU"])
        D = FEM.iso_3d_constitutive(E_mat, nu_mat)

        # Recover centroid stress from static displacement u_global
        local ndof_el::Int
        local B_cen
        if etype == "CTETRA" && nn == 4
            B_cen = FEM.solid_centroid_B_tetra4(view(coords_buf_kg, 1:4, :)); ndof_el = 12
        elseif etype == "CHEXA" && nn == 8
            B_cen = FEM.solid_centroid_B_hexa8(view(coords_buf_kg, 1:8, :)); ndof_el = 24
        elseif etype == "CPENTA" && nn == 6
            B_cen = FEM.solid_centroid_B_cpenta6(view(coords_buf_kg, 1:6, :)); ndof_el = 18
        else
            continue
        end

        # Extract element displacements (translational DOFs, global frame)
        u_el = zeros(ndof_el)
        for k in 1:nn
            idx = id_map[nids[k]]
            u_loc = u_global[(idx-1)*6+1:(idx-1)*6+3]
            u_el[(k-1)*3+1:(k-1)*3+3] = node_R[idx] * u_loc
        end

        stress_vec = D * (B_cen * u_el)

        local Kg_loc
        if etype == "CTETRA"
            Kg_loc = FEM.geometric_stiffness_tetra4(view(coords_buf_kg, 1:4, :), stress_vec)
            n_tetra_kg += 1
        elseif etype == "CHEXA"
            Kg_loc = FEM.geometric_stiffness_hexa8(view(coords_buf_kg, 1:8, :), stress_vec)
            n_hexa_kg += 1
        else
            Kg_loc = FEM.geometric_stiffness_cpenta6(view(coords_buf_kg, 1:6, :), stress_vec)
            n_penta_kg += 1
        end

        # Transform by node_R
        fill!(view(T_buf_kg, 1:ndof_el, 1:ndof_el), 0.0)
        for k in 1:nn
            idx = id_map[nids[k]]; r = (k-1)*3
            for a in 1:3, b in 1:3; T_buf_kg[r+a, r+b] = node_R[idx][a,b]; end
        end
        T_sub = view(T_buf_kg, 1:ndof_el, 1:ndof_el)
        Kg_el = T_sub' * Kg_loc * T_sub

        dofs_solid = Vector{Int}(undef, ndof_el)
        for k in 1:nn
            idx = id_map[nids[k]]; base = (idx-1)*6
            dofs_solid[(k-1)*3+1] = base+1; dofs_solid[(k-1)*3+2] = base+2; dofs_solid[(k-1)*3+3] = base+3
        end
        for c in 1:ndof_el, r in 1:ndof_el
            push!(I_idx, dofs_solid[r]); push!(J_idx, dofs_solid[c]); push!(V_val, Kg_el[r,c])
        end
    end
    n_solids_kg = n_tetra_kg + n_hexa_kg + n_penta_kg
    if n_solids_kg > 0
        log_msg("[SOLVER] Kg solids: $n_tetra_kg CTETRA + $n_hexa_kg CHEXA + $n_penta_kg CPENTA")
    end
    kg_timings["solids"] = (time_ns() - kg_t_solids) * 1e-9

    # --- Constraint redistribution (same as for K) ---
    kg_t_constraints = time_ns()
    _, I_idx, J_idx, V_val = assemble_constraints(model, id_map, node_coords, node_R, I_idx, J_idx, V_val)
    kg_timings["constraint_redistribution"] = (time_ns() - kg_t_constraints) * 1e-9

    kg_t_sparse = time_ns()
    log_msg("[SOLVER] Creating Sparse Kg (NZ: $(length(I_idx)))...")
    Kg = sparse(I_idx, J_idx, V_val, ndof, ndof)
    kg_timings["sparse_build"] = (time_ns() - kg_t_sparse) * 1e-9
    kg_timings["total"] = (time_ns() - kg_t_total) * 1e-9

    if timings !== nothing
        empty!(timings)
        for (k, v) in kg_timings
            timings[string(k)] = v
        end
    end
    return Kg
end
