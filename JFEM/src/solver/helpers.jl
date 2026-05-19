# helpers.jl — Small utility functions for the solver

function log_msg(message::String)
    println("[$(Dates.format(now(), "HH:MM:SS"))] $message")
end

# CBAR V-vector resolution (G0 grid point or direct vector)
function resolve_bar_vref(bar, p_ga::SVector{3,Float64}, id_map, node_coords)
    g0 = get(bar, "G0", 0)
    if g0 > 0 && haskey(id_map, g0)
        ig0 = id_map[g0]
        p_g0 = SVector{3}(node_coords[ig0,1], node_coords[ig0,2], node_coords[ig0,3])
        return p_g0 - p_ga
    else
        v_raw = bar["V"]
        return SVector{3}(v_raw...)
    end
end

# Skew-symmetric cross-product matrix: skew(w) * v = w × v
@inline function skew3(w)
    return @SMatrix [0.0 -w[3] w[2]; w[3] 0.0 -w[1]; -w[2] w[1] 0.0]
end

# Get CBAR offset vectors and effective endpoints
function bar_offsets_and_endpoints(bar, p1::SVector{3,Float64}, p2::SVector{3,Float64})
    wa_raw = get(bar, "WA", [0.0, 0.0, 0.0])
    wb_raw = get(bar, "WB", [0.0, 0.0, 0.0])
    wa = SVector{3}(wa_raw...)
    wb = SVector{3}(wb_raw...)
    has_offset = (wa[1]^2+wa[2]^2+wa[3]^2) > 1e-20 || (wb[1]^2+wb[2]^2+wb[3]^2) > 1e-20
    p1_eff = has_offset ? p1 + wa : p1
    p2_eff = has_offset ? p2 + wb : p2
    return wa, wb, has_offset, p1_eff, p2_eff
end

@inline function _subcase_temp_load_sid(sub::AbstractDict, cc::AbstractDict)
    if get(sub, "TEMP_MODIFIER", nothing) == "LOAD" && haskey(sub, "TEMP")
        return Int(sub["TEMP"])
    end
    if get(cc, "TEMP_MODIFIER", nothing) == "LOAD" && haskey(cc, "TEMP")
        return Int(cc["TEMP"])
    end
    return nothing
end

@inline function _temperature_field_for_sid(model, temp_sid)
    if isnothing(temp_sid)
        return Dict{Int,Float64}(), 0.0
    end
    sid = Int(temp_sid)
    temps_map = get(model, "TEMPs", Dict{Int,Dict{Int,Float64}}())
    tempd_map = get(model, "TEMPDs", Dict{Int,Float64}())
    node_temps = get(temps_map, sid, Dict{Int,Float64}())
    default_temp = get(tempd_map, sid, 0.0)
    return node_temps, default_temp
end

@inline function _average_temperature_for_nodes(nids, node_temps::AbstractDict, default_temp::Float64)
    isempty(nids) && return default_temp
    total = 0.0
    for nid in nids
        total += Float64(get(node_temps, nid, default_temp))
    end
    return total / length(nids)
end

function _tablem1_interp(table::AbstractDict, x::Float64)
    points = get(table, "POINTS", Any[])
    isempty(points) && return 0.0
    if length(points) == 1
        return Float64(points[1]["Y"])
    end

    x_first = Float64(points[1]["X"])
    if x <= x_first
        return Float64(points[1]["Y"])
    end

    for i in 1:length(points)-1
        p_lo = points[i]
        p_hi = points[i + 1]
        x_lo = Float64(p_lo["X"])
        x_hi = Float64(p_hi["X"])
        if x <= x_hi
            y_lo = Float64(p_lo["Y"])
            y_hi = Float64(p_hi["Y"])
            abs(x_hi - x_lo) <= 1e-30 && return y_hi
            t = (x - x_lo) / (x_hi - x_lo)
            return y_lo + t * (y_hi - y_lo)
        end
    end

    return Float64(points[end]["Y"])
end

@inline function _complete_mat1_triplet(E::Float64, G::Float64, nu::Float64)
    if E > 0.0 && G > 0.0 && nu < 0.0
        nu = E / (2.0 * G) - 1.0
    elseif E > 0.0 && nu >= 0.0 && G <= 0.0
        G = E / (2.0 * (1.0 + nu))
    elseif G > 0.0 && nu >= 0.0 && E <= 0.0
        E = 2.0 * G * (1.0 + nu)
    end
    E <= 0.0 && (E = 0.0)
    G <= 0.0 && (G = 0.0)
    nu < 0.0 && (nu = 0.3)
    return E, G, nu
end

function _effective_mat1_for_nodes(model, mid_raw, nids; temp_sid=nothing)
    mid = string(mid_raw)
    mats = get(model, "MATs", Dict{String,Any}())
    base = get(mats, mid, nothing)
    base === nothing && return nothing

    matt1s = get(model, "MATT1s", Dict{String,Any}())
    matt1 = get(matt1s, mid, nothing)
    matt1 === nothing && return base

    sid = isnothing(temp_sid) ? get(model, "_active_temp_sid", nothing) : temp_sid
    isnothing(sid) && return base

    tablem1s = get(model, "TABLEM1s", Dict{String,Any}())
    isempty(tablem1s) && return base

    node_temps, default_temp = _temperature_field_for_sid(model, sid)
    temperature = _average_temperature_for_nodes(nids, node_temps, default_temp)

    mat = deepcopy(base)
    for (table_key, field) in (("E_TABLE", "E"), ("G_TABLE", "G"), ("NU_TABLE", "NU"),
                               ("RHO_TABLE", "RHO"), ("ALPHA_TABLE", "ALPHA"))
        tid = get(matt1, table_key, nothing)
        isnothing(tid) && continue
        table = get(tablem1s, string(tid), nothing)
        table === nothing && continue
        mat[field] = _tablem1_interp(table, temperature)
    end

    E = Float64(get(mat, "E", 0.0))
    G = Float64(get(mat, "G", 0.0))
    nu = Float64(get(mat, "NU", -1.0))
    E, G, nu = _complete_mat1_triplet(E, G, nu)
    mat["E"] = E
    mat["G"] = G
    mat["NU"] = nu
    return mat
end

# Fast shell element frame computation
@inline function q4_frame_mode_from_env(primary_key::String)
    # Default to the solver's geometric CQUAD4 frame. Alternative element frames
    # are explicit formulation experiments and should not be selected by a
    # validation-suite preset.
    base_default = "diag"
    raw = lowercase(strip(get(ENV, primary_key, get(ENV, "JFEM_Q4_FRAME_MODE", base_default))))
    if raw in ("parametric", "center", "center_tangent", "tangent")
        return :parametric
    elseif raw in ("edge", "g12", "edge12")
        return :edge
    else
        return :diag
    end
end

@inline function q4_stage_key(primary_key::String)
    if endswith(primary_key, "_STATIC")
        return :static
    elseif endswith(primary_key, "_EIG")
        return :eig
    else
        return :other
    end
end

@inline function q4_curvature_stage_default(primary_key::String, static_value, eig_value, other_value)
    stage = q4_stage_key(primary_key)
    if stage === :static
        return static_value
    elseif stage === :eig
        return eig_value
    else
        return other_value
    end
end

@inline function solver_env_bool(primary_key::String, default::Bool)
    raw = lowercase(strip(get(ENV, primary_key, default ? "true" : "false")))
    return raw in ("1", "true", "yes", "on")
end

@inline function solver_env_optional_bool(primary_key::String)
    haskey(ENV, primary_key) || return nothing
    raw = lowercase(strip(ENV[primary_key]))
    return raw in ("1", "true", "yes", "on")
end

@inline function solver_env_float(primary_key::String, default::Float64)
    raw = strip(get(ENV, primary_key, string(default)))
    return something(tryparse(Float64, raw), default)
end

@inline function solver_env_int(primary_key::String, default::Int)
    raw = strip(get(ENV, primary_key, string(default)))
    return something(tryparse(Int, raw), default)
end

@inline function solver_env_optional_float(primary_key::String)
    haskey(ENV, primary_key) || return nothing
    raw = strip(ENV[primary_key])
    isempty(raw) && return nothing
    return tryparse(Float64, raw)
end

@inline function sol105_use_static_k_enabled()
    # Linear buckling should use the tangent stiffness associated with the
    # static preload displacement. The older split K_eig path is retained as an
    # explicit research switch via JFEM_SOL105_USE_STATIC_K=false.
    return solver_env_bool("JFEM_SOL105_USE_STATIC_K", true)
end

@inline function sol105_static_membrane_incomp_enabled()
    # Default OFF (2026-04-29). Wilson-Taylor incompatible membrane modes were
    # turned on in the static K to reduce shear-locking on bending-dominant
    # cases. Empirically they soften the static stiffness too much for the
    # bending-prestress validation cases and hurt the eigenvalue magnitudes
    # against Nastran (HTP_launch first-eigenvalue rel err: 13.3% with bubbles
    # vs 2.9% without; 3wp magnitudes also improve). Nastran's CQUAD4 (MacNeal
    # lineage) does not include Wilson-Taylor membrane bubbles, so removing
    # them brings JFEM closer to Nastran's static K. Re-enable with
    # JFEM_SOL105_STATIC_MEMBRANE_INCOMP=true for the older default.
    return solver_env_bool("JFEM_SOL105_STATIC_MEMBRANE_INCOMP", false)
end

@inline function sol105_static_membrane_incomp_auto_load_enabled()
    # Default OFF. Diagnostic gate for SOL105 static preload K: enable Wilson
    # membrane modes only on simple compression subcases accepted by the same
    # direct-FORCE load classifier used by Kg stress-field auto mode. Broad
    # static membrane bubbles close the curved-compression patch probes but
    # break shear/mixed guardrails, so this remains explicitly opt-in.
    return solver_env_bool("JFEM_SOL105_STATIC_MEMBRANE_INCOMP_AUTO_LOAD", false)
end

@inline function q4_sol105_static_pcomp_membrane_incomp_aspect_enabled()
    # Default OFF. Diagnostic gate for SOL105 static preload K: add Wilson
    # membrane modes only to PCOMP CQUAD4 elements inside an aspect-ratio
    # window. This lets the parity campaign test whether the high-aspect HTP
    # population wants the bubble contribution while lower-aspect VTP/guardrail
    # meshes should stay on the MacNeal-like compatible membrane path.
    return solver_env_optional_float("JFEM_SOL105_STATIC_PCOMP_MEMBRANE_INCOMP_ASPECT_MIN") !== nothing
end

@inline function q4_sol105_static_pcomp_membrane_incomp_aspect_min()
    raw = solver_env_optional_float("JFEM_SOL105_STATIC_PCOMP_MEMBRANE_INCOMP_ASPECT_MIN")
    raw === nothing && return Inf
    return max(raw, 0.0)
end

@inline function q4_sol105_static_pcomp_membrane_incomp_aspect_max()
    raw = solver_env_optional_float("JFEM_SOL105_STATIC_PCOMP_MEMBRANE_INCOMP_ASPECT_MAX")
    raw === nothing && return Inf
    return max(raw, 0.0)
end

@inline function kg_match_static_membrane_operator_enabled()
    # Default OFF (2026-04-29). Previously this cascaded from USE_STATIC_K=true,
    # which silently switched Kg σ-recovery to the bubble-augmented (Wilson-
    # Taylor) kinematic field. For bending-dominant prestress cases, the WT
    # bubble correction has comparable magnitude and opposite
    # sign to the compatible σ, so it flipped the σ sign and hence the buckling
    # eigenvalue sign — Nastran returned λ ≈ +1.14, JFEM returned λ ≈ -0.20.
    # The bubble σ matches K_static in JFEM's own internal element but Nastran's
    # CQUAD4 (MacNeal lineage) does not use the same internal modes, so the
    # "matching" was consistent only with JFEM's element, not Nastran's.
    # Reverting to compatible-only σ restores positive eigenvalue signs across
    # all tested multi-point bending subcases.
    # Re-enable explicitly with JFEM_KG_MATCH_STATIC_MEMBRANE_OPERATOR=true.
    return solver_env_bool("JFEM_KG_MATCH_STATIC_MEMBRANE_OPERATOR", false)
end

@inline function buckling_rng_seed()
    return solver_env_int("JFEM_BUCKLING_RNG_SEED", 0)
end

@inline function sol105_snorm_angle_override()
    return solver_env_optional_float("JFEM_SOL105_SNORM_ANGLE")
end

@inline function sol105_eig_bending_incomp_enabled()
    # Use the enriched bending branch by default for SOL105 K_eig. This is now
    # the safest common baseline across the broad buckling validation set,
    # while still allowing an explicit opt-out through the env override.
    return solver_env_bool("JFEM_SOL105_EIG_BENDING_INCOMP", true)
end

@inline function q4_curvature_membrane_scale(primary_key::String)
    # Default 0.0 (2026-04-21 cleanup): the curvature_membrane B[:, idx+3] term
    # (Koiter-Donnell style -N_k*kappa coupling on w-DOF) was gated off by the
    # resolution threshold on every large-deck validation element anyway; removing it at the
    # default keeps behavior consistent with pure_v1 baseline. A proper
    # Marguerre-style coupling on the rotation DOFs (idx+4/5) is the intended
    # replacement per Ibrahimbegović 1994 Eq. 6.14.
    default = 0.0
    raw = strip(get(ENV, primary_key, get(ENV, "JFEM_Q4_CURVATURE_MEMBRANE_SCALE", string(default))))
    return something(tryparse(Float64, raw), default)
end

@inline function q4_curvature_filter_mode(primary_key::String)
    default = q4_curvature_stage_default(primary_key, "cylindrical", "cylindrical", "none")
    raw = lowercase(strip(get(ENV, primary_key, get(ENV, "JFEM_Q4_CURVATURE_FILTER_MODE", default))))
    if raw in ("cyl", "cylindrical", "developable")
        return :cylindrical
    elseif raw in ("cylsmooth", "cylindrical_smooth", "developable_smooth", "smooth")
        return :cylindrical_smooth
    else
        return :none
    end
end

@inline function q4_curvature_cyl_ratio_max(primary_key::String)
    default = q4_curvature_stage_default(primary_key, 0.05, 0.05, 0.15)
    raw = strip(get(ENV, primary_key, get(ENV, "JFEM_Q4_CURVATURE_CYL_RATIO_MAX", string(default))))
    return clamp(something(tryparse(Float64, raw), default), 1e-6, 1.0)
end

@inline function q4_curvature_filter_weight(curvature::SVector{3,Float64}, mode::Symbol, ratio_max::Float64)
    mode === :none && return 1.0

    k1, k2 = q4_curvature_principal_abs(curvature)
    k1 <= 1e-30 && return 0.0

    ratio = k2 / k1
    if mode === :cylindrical
        return ratio <= ratio_max ? 1.0 : 0.0
    else
        return clamp(1.0 - ratio / ratio_max, 0.0, 1.0)
    end
end

@inline function q4_curvature_principal_abs(curvature::SVector{3,Float64})
    k11, k22, k12 = curvature
    tr = 0.5 * (k11 + k22)
    dev = sqrt(((k11 - k22) / 2.0)^2 + k12^2)
    k1 = abs(tr + dev)
    k2 = abs(tr - dev)
    if k2 > k1
        k1, k2 = k2, k1
    end
    return k1, k2
end

@inline function q4_curvature_cyl_ratio(curvature::SVector{3,Float64})
    k1, k2 = q4_curvature_principal_abs(curvature)
    return k1 <= 1e-30 ? 1.0 : k2 / k1
end

@inline function q4_curvature_resolution_min(primary_key::String)
    default = q4_curvature_stage_default(primary_key, 0.05, 0.05, 0.0)
    raw = strip(get(ENV, primary_key, get(ENV, "JFEM_Q4_CURVATURE_RESOLUTION_MIN", string(default))))
    return max(something(tryparse(Float64, raw), default), 0.0)
end

@inline function q4_curvature_resolution_full(primary_key::String)
    default = q4_curvature_stage_default(primary_key, 0.05, 0.05, 0.0)
    raw = strip(get(ENV, primary_key, get(ENV, "JFEM_Q4_CURVATURE_RESOLUTION_FULL", string(default))))
    return max(something(tryparse(Float64, raw), default), 0.0)
end

@inline function q4_pcomp_kg_axis_mode()
    # Kg must use the same laminate material axis as the preload/static
    # operator unless the user explicitly asks for an experimental axis study.
    raw = lowercase(strip(get(ENV, "JFEM_Q4_PCOMP_KG_AXIS_MODE", "element")))
    if raw in ("element", "theta", "theta_only", "element_x")
        return :element
    elseif raw in ("g12", "edge12", "g1g2")
        return :g12
    elseif raw in ("global", "global_x", "projected_global_x")
        return :global_x
    elseif raw in ("warp_switch", "warp", "auto")
        return :warp_switch
    end
    return :warp_switch
end

@inline function q4_pcomp_kg_auto_g12_enabled()
    # Default off (2026-04-21 cleanup): heuristic auto-G12 axis selector was
    # case-tuned and net-negative on the broad validation set. Env override retained.
    return solver_env_bool("JFEM_Q4_PCOMP_KG_AUTO_G12", false)
end

@inline function q4_pcomp_kg_auto_g12_shear_ratio_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_G12_SHEAR_RATIO_MIN", 0.15)
end

@inline function q4_pcomp_kg_auto_g12_d16_ratio_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_G12_D16_RATIO_MIN", 0.15)
end

@inline function q4_pcomp_kg_auto_g12_b_ratio_max()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_G12_B_RATIO_MAX", 0.05)
end

@inline function q4_pcomp_kg_auto_g12_theta_abs_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_G12_THETA_ABS_MIN", 15.0)
end

@inline function q4_pcomp_kg_auto_g12_d16_theta_bypass_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_G12_D16_THETA_BYPASS_MIN", 0.35)
end

@inline function q4_pcomp_kg_auto_g12_cyl_ratio_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_G12_CYL_RATIO_MIN", 0.0)
end

@inline function q4_pcomp_kg_auto_g12_kappa_l_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_G12_KAPPA_L_MIN", 0.005)
end

@inline function q4_pcomp_kg_auto_curvature_enabled()
    return solver_env_bool("JFEM_Q4_PCOMP_KG_AUTO_CURVATURE", false)
end

@inline function q4_pcomp_kg_auto_curvature_shear_ratio_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_CURVATURE_SHEAR_RATIO_MIN", 0.15)
end

@inline function q4_pcomp_kg_auto_curvature_d16_ratio_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_CURVATURE_D16_RATIO_MIN", 0.15)
end

@inline function q4_pcomp_kg_auto_curvature_b_ratio_max()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_CURVATURE_B_RATIO_MAX", 0.05)
end

@inline function q4_pcomp_kg_auto_curvature_theta_abs_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_CURVATURE_THETA_ABS_MIN", 15.0)
end

@inline function q4_pcomp_kg_auto_curvature_d16_theta_bypass_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_CURVATURE_D16_THETA_BYPASS_MIN", 0.35)
end

@inline function q4_pcomp_kg_auto_curvature_kappa_l_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_CURVATURE_KAPPA_L_MIN", 0.005)
end

@inline function q4_pcomp_kg_auto_curvature_cyl_ratio_min()
    return solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_CURVATURE_CYL_RATIO_MIN", 0.02)
end

@inline function q4_pcomp_kg_auto_curvature_sign()
    raw = strip(get(ENV, "JFEM_Q4_PCOMP_KG_AUTO_CURVATURE_SIGN", "-1.0"))
    val = something(tryparse(Float64, raw), -1.0)
    return val < 0.0 ? -1.0 : 1.0
end

@inline function q4_pcomp_kg_auto_curvature_scale()
    # Default 1.0 neutral (2026-04-21 cleanup): the 7.0 multiplier was a fudge
    # factor tuned to a synthetic laminate family; net-negative on the broad validation set.
    return max(solver_env_float("JFEM_Q4_PCOMP_KG_AUTO_CURVATURE_SCALE", 1.0), 0.0)
end

@inline function q4_shell_kg_auto_curvature_iso_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO", false)
end

@inline function q4_shell_kg_auto_curvature_iso_kappa_l_min()
    return solver_env_float("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_KAPPA_L_MIN", 0.005)
end

@inline function q4_shell_kg_auto_curvature_iso_cyl_ratio_min()
    return solver_env_float("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_CYL_RATIO_MIN", 0.05)
end

@inline function q4_shell_kg_auto_curvature_iso_sign()
    raw = strip(get(ENV, "JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_SIGN", "-1.0"))
    val = something(tryparse(Float64, raw), -1.0)
    return val < 0.0 ? -1.0 : 1.0
end

@inline function q4_shell_kg_auto_curvature_iso_scale()
    # Default 1.0 neutral (2026-04-21 cleanup).
    return max(solver_env_float("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_SCALE", 1.0), 0.0)
end

@inline function q4_shell_kg_auto_curvature_iso_cyl_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_CYL", false)
end

@inline function q4_shell_kg_auto_curvature_iso_cyl_kappa_l_min()
    return max(solver_env_float("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_CYL_KAPPA_L_MIN", 0.05), 0.0)
end

@inline function q4_shell_kg_auto_curvature_iso_cyl_ratio_max()
    return clamp(solver_env_float("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_CYL_RATIO_MAX", 0.05), 0.0, 1.0)
end

@inline function q4_shell_kg_auto_curvature_iso_cyl_sign()
    raw = strip(get(ENV, "JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_CYL_SIGN", "1.0"))
    val = something(tryparse(Float64, raw), 1.0)
    return val < 0.0 ? -1.0 : 1.0
end

@inline function q4_shell_kg_auto_curvature_iso_cyl_scale()
    return max(solver_env_float("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_CYL_SCALE", 1.0), 0.0)
end

@inline function q4_shell_kg_auto_curvature_iso_aspect_ratio_max()
    return max(solver_env_float("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_ASPECT_RATIO_MAX", 1.5), 1.0)
end

@inline function q4_shell_kg_auto_curvature_iso_double_curv_gain()
    return max(solver_env_float("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_DOUBLE_CURV_GAIN", 0.0), 0.0)
end

@inline function q4_shell_kg_auto_curvature_iso_double_curv_kappa_min()
    return max(solver_env_float("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_DOUBLE_CURV_KAPPA_MIN", 0.04), 0.0)
end

@inline function q4_shell_kg_auto_curvature_iso_double_curv_kappa_full()
    return max(solver_env_float("JFEM_Q4_SHELL_KG_AUTO_CURVATURE_ISO_DOUBLE_CURV_KAPPA_FULL", 0.06), 0.0)
end

@inline function q4_shell_kg_auto_curvature_iso_effective_scale(cyl_ratio::Float64, kappa_l::Float64)
    base = q4_shell_kg_auto_curvature_iso_scale()
    gain = q4_shell_kg_auto_curvature_iso_double_curv_gain()
    ratio_min = q4_shell_kg_auto_curvature_iso_cyl_ratio_min()
    if gain <= 0.0 || cyl_ratio <= ratio_min
        return base
    end
    double_curv_weight = clamp((cyl_ratio - ratio_min) / max(1.0 - ratio_min, 1e-12), 0.0, 1.0)
    kappa_min = q4_shell_kg_auto_curvature_iso_double_curv_kappa_min()
    kappa_full = q4_shell_kg_auto_curvature_iso_double_curv_kappa_full()
    coarse_weight = if kappa_full <= kappa_min
        kappa_l >= kappa_min ? 1.0 : 0.0
    else
        clamp((kappa_l - kappa_min) / max(kappa_full - kappa_min, 1e-12), 0.0, 1.0)
    end
    return base * (1.0 + gain * double_curv_weight * coarse_weight)
end

@inline function q4_pcomp_axis_mode(primary_key::String, fallback_key::String="JFEM_Q4_PCOMP_AXIS_MODE")
    raw = lowercase(strip(get(ENV, primary_key, get(ENV, fallback_key, "element"))))
    if raw in ("element", "theta", "theta_only", "element_x")
        return :element
    elseif raw in ("g12", "edge12", "g1g2")
        return :g12
    elseif raw in ("global", "global_x", "projected_global_x")
        return :global_x
    elseif raw in ("warp_switch", "warp", "auto")
        return :warp_switch
    else
        return :element
    end
end

@inline function q4_pcomp_kg_warp_ratio_threshold()
    raw = strip(get(ENV, "JFEM_Q4_PCOMP_KG_WARP_RATIO_THRESHOLD", "5e-5"))
    return max(something(tryparse(Float64, raw), 5e-5), 0.0)
end

@inline function q4_curvature_characteristic_length(lc::AbstractMatrix)
    p1 = @SVector [lc[1,1], lc[1,2]]
    p2 = @SVector [lc[2,1], lc[2,2]]
    p3 = @SVector [lc[3,1], lc[3,2]]
    p4 = @SVector [lc[4,1], lc[4,2]]
    return 0.25 * (norm(p2 - p1) + norm(p3 - p2) + norm(p4 - p3) + norm(p1 - p4))
end

@inline function q4_local_edge_aspect_ratio(lc::AbstractMatrix)
    p1 = @SVector [lc[1,1], lc[1,2]]
    p2 = @SVector [lc[2,1], lc[2,2]]
    p3 = @SVector [lc[3,1], lc[3,2]]
    p4 = @SVector [lc[4,1], lc[4,2]]
    l12 = norm(p2 - p1)
    l23 = norm(p3 - p2)
    l34 = norm(p4 - p3)
    l41 = norm(p1 - p4)
    lmin = max(min(l12, l23, l34, l41), 1e-12)
    lmax = max(l12, l23, l34, l41)
    return lmax / lmin
end

@inline function q4_local_opposite_edge_ratio(lc::AbstractMatrix)
    p1 = @SVector [lc[1,1], lc[1,2]]
    p2 = @SVector [lc[2,1], lc[2,2]]
    p3 = @SVector [lc[3,1], lc[3,2]]
    p4 = @SVector [lc[4,1], lc[4,2]]
    l12 = max(norm(p2 - p1), 1e-12)
    l23 = max(norm(p3 - p2), 1e-12)
    l34 = max(norm(p4 - p3), 1e-12)
    l41 = max(norm(p1 - p4), 1e-12)
    r13 = min(l12, l34) / max(l12, l34)
    r24 = min(l23, l41) / max(l23, l41)
    return min(r13, r24)
end

@inline function q4_local_edge_skew_angle(lc::AbstractMatrix)
    p1 = @SVector [lc[1,1], lc[1,2]]
    p2 = @SVector [lc[2,1], lc[2,2]]
    p3 = @SVector [lc[3,1], lc[3,2]]
    e12 = p2 - p1
    e23 = p3 - p2
    l12 = max(norm(e12), 1e-12)
    l23 = max(norm(e23), 1e-12)
    cos_corner = abs(dot(e12 / l12, e23 / l23))
    return acos(clamp(cos_corner, 0.0, 1.0)) * 180.0 / pi
end

@inline function q4_curvature_resolution_weight(curvature::SVector{3,Float64}, lc::AbstractMatrix,
                                                min_res::Float64, full_res::Float64)
    (min_res <= 0.0 && full_res <= 0.0) && return 1.0

    k1, _ = q4_curvature_principal_abs(curvature)
    kappa_l = k1 * q4_curvature_characteristic_length(lc)
    if full_res <= min_res
        return kappa_l >= min_res ? 1.0 : 0.0
    end
    return clamp((kappa_l - min_res) / (full_res - min_res), 0.0, 1.0)
end

@inline function q4_flat_pcomp_auto_phi2_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_PCOMP_AUTO_PHI2", false)
end

@inline function q4_flat_pcomp_auto_phi2_shear_ratio_max()
    return solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_AUTO_PHI2_SHEAR_RATIO_MAX", 0.12)
end

@inline function q4_flat_pcomp_auto_phi2_d16_ratio_max()
    return solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_AUTO_PHI2_D16_RATIO_MAX", 0.02)
end

@inline function q4_flat_pcomp_auto_phi2_b_ratio_max()
    return solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_AUTO_PHI2_B_RATIO_MAX", 0.02)
end

@inline function q4_flat_pcomp_auto_phi2_cyl_ratio_max()
    return solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_AUTO_PHI2_CYL_RATIO_MAX", 0.005)
end

@inline function q4_flat_pcomp_auto_phi2_kappa_l_min()
    return solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_AUTO_PHI2_KAPPA_L_MIN", 0.20)
end

@inline function q4_pcomp_auto_global_x_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_Q4_PCOMP_AUTO_GLOBAL_X", false)
end

@inline function q4_pcomp_auto_global_x_shear_ratio_max()
    return solver_env_float("JFEM_Q4_PCOMP_AUTO_GLOBAL_X_SHEAR_RATIO_MAX", 0.12)
end

@inline function q4_pcomp_auto_global_x_d16_ratio_max()
    return solver_env_float("JFEM_Q4_PCOMP_AUTO_GLOBAL_X_D16_RATIO_MAX", 0.02)
end

@inline function q4_pcomp_auto_global_x_b_ratio_max()
    return solver_env_float("JFEM_Q4_PCOMP_AUTO_GLOBAL_X_B_RATIO_MAX", 0.02)
end

@inline function q4_pcomp_auto_global_x_cyl_ratio_min()
    return solver_env_float("JFEM_Q4_PCOMP_AUTO_GLOBAL_X_CYL_RATIO_MIN", 0.05)
end

@inline function q4_pcomp_auto_global_x_kappa_l_min()
    return solver_env_float("JFEM_Q4_PCOMP_AUTO_GLOBAL_X_KAPPA_L_MIN", 0.005)
end

@inline function q4_sol105_flat_pcomp_rect_adini_enabled()
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_PCOMP_RECT_ADINI", false)
end

@inline function q4_sol105_flat_pcomp_plate_branch_enabled()
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_PCOMP_PLATE_BRANCH", false)
end

"""
JFEM_NASTRAN_PARITY is deprecated as a formulation preset.

The solver must converge toward a generic Nastran-compatible formulation, not a
bundle of validation-suite decisions. This helper therefore intentionally does
not activate any hidden element, stress-recovery, frame, or eigenvalue filters.
Use explicit formulation switches for controlled experiments, for example:

  JFEM_SOL105_EIG_FLAT_PCOMP_DKMQ=true
  JFEM_KG_QUAD4_STRESS_FIELD_MODE=gauss
  JFEM_Q4_FRAME_MODE=parametric
  JFEM_SOL105_EIG_PCOMP_MEMBRANE_INCOMP=true

Optional EIGRL range-completeness diagnostics remain explicit:
  JFEM_SOL105_RANGE_AUGMENTATION=true
  JFEM_SOL105_RANGE_AUGMENTATION_MULTI=true
  JFEM_SOL105_RANGE_AUGMENTATION_SIGMAS=0.037,0.4
  JFEM_SOL105_RETURN_ALL_IN_RANGE=true


"""
@inline function jfem_nastran_parity_preset()
    return false
end

@inline function q4_sol105_flat_pcomp_dkmq_enabled()
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_PCOMP_DKMQ", false)
end

@inline function q4_sol105_flat_pcomp_dkmq_static_enabled()
    return solver_env_bool("JFEM_SOL105_STATIC_FLAT_PCOMP_DKMQ", false)
end

@inline function q4_sol105_flat_pcomp_plate_auto_enabled()
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_PCOMP_PLATE_AUTO", false)
end

@inline function q4_sol105_flat_pcomp_plate_auto_d16_ratio_max()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_PLATE_AUTO_D16_RATIO_MAX", 0.10), 0.0)
end

@inline function q4_sol105_flat_pcomp_plate_auto_shear_ratio_max()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_PLATE_AUTO_SHEAR_RATIO_MAX", 0.35), 0.0)
end

@inline function q4_sol105_flat_pcomp_plate_like_kg_enabled()
    return solver_env_bool("JFEM_SOL105_KG_FLAT_PCOMP_PLATE_LIKE", false)
end

@inline function q4_sol105_nonflat_pcomp_normal_only_kg_enabled()
    return solver_env_bool("JFEM_SOL105_KG_NONFLAT_PCOMP_NORMAL_ONLY", true)
end

@inline function q4_flat_pcomp_auto_g12_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_Q4_FLAT_PCOMP_AUTO_G12", false)
end

@inline function q4_flat_pcomp_auto_g12_kappa_l_max()
    return max(solver_env_float("JFEM_Q4_FLAT_PCOMP_AUTO_G12_KAPPA_L_MAX", 0.005), 0.0)
end

@inline function q4_flat_pcomp_auto_g12_cyl_ratio_max()
    return clamp(solver_env_float("JFEM_Q4_FLAT_PCOMP_AUTO_G12_CYL_RATIO_MAX", 0.05), 0.0, 1.0)
end

@inline function q4_curved_iso_eig_auto_membrane_incomp_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_CURVED_ISO_MEMBRANE_INCOMP", false)
end

@inline function q4_curved_iso_eig_auto_membrane_incomp_kappa_l_min()
    return max(solver_env_float("JFEM_SOL105_EIG_CURVED_ISO_MEMBRANE_INCOMP_KAPPA_L_MIN", 0.005), 0.0)
end

@inline function q4_curved_iso_eig_auto_membrane_incomp_cyl_ratio_max()
    return clamp(solver_env_float("JFEM_SOL105_EIG_CURVED_ISO_MEMBRANE_INCOMP_CYL_RATIO_MAX", 0.2), 0.0, 1.0)
end

@inline function q4_curved_iso_warp_membrane_incomp_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_CURVED_ISO_WARP_MEMBRANE_INCOMP", false)
end

@inline function q4_curved_iso_warp_membrane_incomp_ratio_min()
    return max(solver_env_float("JFEM_SOL105_EIG_CURVED_ISO_WARP_MEMBRANE_INCOMP_RATIO_MIN", 0.03), 0.0)
end

@inline function q4_curved_iso_warp_membrane_incomp_kappa_l_max()
    default = q4_curved_iso_eig_auto_membrane_incomp_kappa_l_min()
    return max(solver_env_float("JFEM_SOL105_EIG_CURVED_ISO_WARP_MEMBRANE_INCOMP_KAPPA_L_MAX", default), 0.0)
end

@inline function q4_curved_iso_elongated_membrane_incomp_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_CURVED_ISO_ELONGATED_MEMBRANE_INCOMP", false)
end

@inline function q4_curved_iso_elongated_membrane_incomp_aspect_ratio_min()
    return max(solver_env_float("JFEM_SOL105_EIG_CURVED_ISO_ELONGATED_MEMBRANE_INCOMP_ASPECT_RATIO_MIN", 1.5), 1.0)
end

@inline function q4_curved_iso_square_fullshear_blend_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_CURVED_ISO_SQUARE_FULLSHEAR_BLEND_ENABLED", false)
end

@inline function q4_curved_iso_square_fullshear_blend_value()
    # Default 1.0 neutral (2026-04-21 cleanup): was 0.7 eyeballed blend.
    return clamp(solver_env_float("JFEM_SOL105_EIG_CURVED_ISO_SQUARE_FULLSHEAR_BLEND", 1.0), 0.0, 1.0)
end

@inline function q4_curved_iso_square_fullshear_blend_aspect_ratio_max()
    return clamp(solver_env_float("JFEM_SOL105_EIG_CURVED_ISO_SQUARE_FULLSHEAR_BLEND_ASPECT_RATIO_MAX", 1.05), 1.0, 10.0)
end

@inline function q4_flat_iso_fullshear_selective_mode()
    raw = lowercase(strip(get(ENV, "JFEM_SOL105_EIG_FLAT_ISO_FULLSHEAR_SELECTIVE_MODE", "sy_only")))
    if raw in ("sx", "sx_only", "x", "xiz")
        return :sx_only
    elseif raw in ("sy", "sy_only", "y", "etaz", "eta")
        return :sy_only
    elseif raw in ("all", "full_selective")
        return :all
    end
    return :none
end

@inline function q4_flat_pcomp_eig_membrane_assumed_mode()
    raw = lowercase(strip(get(ENV, "JFEM_SOL105_EIG_FLAT_PCOMP_MEMBRANE_ASSUMED_MODE", "mitc4plus")))
    if raw in ("mitc4plus_all", "mitc4+_all", "ans_all", "all")
        return :mitc4plus_all
    elseif raw in ("mitc4plus", "mitc4+", "ans")
        return :mitc4plus
    end
    return :none
end

@inline function q4_flat_pcomp_taper_membrane_none_enabled()
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_PCOMP_TAPER_MEMBRANE_NONE", false)
end

@inline function q4_flat_pcomp_taper_membrane_none_ratio_max()
    return clamp(solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_TAPER_MEMBRANE_NONE_RATIO_MAX", 0.35), 0.0, 1.0)
end

@inline function q4_flat_pcomp_taper_membrane_none_aspect_min()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_TAPER_MEMBRANE_NONE_ASPECT_MIN", 2.0), 1.0)
end

@inline function q4_nonflat_pcomp_eig_membrane_assumed_mode()
    # Ko-Lee-Bathe 2016 §3 MITC4+ for warped/curved shell quads: standard MITC4
    # membrane exhibits membrane locking on curved geometries (HTP_launch 14% bias).
    # Default "none" preserves legacy behavior; "mitc4plus" enables the ANS fix on
    # non-flat curved PCOMP, mirroring what we already do for flat PCOMP.
    raw = lowercase(strip(get(ENV, "JFEM_SOL105_EIG_NONFLAT_PCOMP_MEMBRANE_ASSUMED_MODE", "none")))
    if raw in ("mitc4plus_all", "mitc4+_all", "ans_all", "all")
        return :mitc4plus_all
    elseif raw in ("mitc4plus", "mitc4+", "ans")
        return :mitc4plus
    end
    return :none
end

@inline function q4_sol105_flat_pcomp_shear_scale()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_SHEAR_SCALE", 1.0), 0.0)
end

@inline function q4_sol105_flat_pcomp_auto_shear_scale_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_PCOMP_AUTO_SHEAR_SCALE", false)
end

@inline function q4_sol105_flat_pcomp_auto_shear_scale_gain()
    return clamp(solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_AUTO_SHEAR_SCALE_GAIN", 0.5), 0.0, 2.0)
end

@inline function q4_sol105_flat_pcomp_auto_shear_scale_max()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_AUTO_SHEAR_SCALE_MAX", 1.5), 1.0)
end

@inline function q4_sol105_flat_pcomp_exact_side_shear()
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_PCOMP_EXACT_SIDE_SHEAR", false)
end

@inline function q4_sol105_flat_curved_pcomp_exact_side_shear()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_CURVED_PCOMP_EXACT_SIDE_SHEAR", false)
end

@inline function q4_sol105_flat_pcomp_exact_side_rotcorr()
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_PCOMP_EXACT_SIDE_ROTCORR", false)
end

@inline function q4_sol105_flat_pcomp_exact_membrane()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_PCOMP_EXACT_MEMBRANE", false)
end

@inline function q4_sol105_flat_iso_exact_side_shear()
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_ISO_EXACT_SIDE_SHEAR", false)
end

@inline function q4_flat_curved_iso_coarse_exact_side_shear_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_CURVED_ISO_COARSE_EXACT_SIDE_SHEAR", false)
end

@inline function q4_flat_curved_iso_coarse_exact_side_shear_aspect_ratio_max()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_CURVED_ISO_COARSE_EXACT_SIDE_SHEAR_ASPECT_RATIO_MAX", 1.1), 1.0)
end

@inline function q4_flat_curved_iso_coarse_exact_side_shear_valence_sum_min()
    raw = strip(get(ENV, "JFEM_SOL105_EIG_FLAT_CURVED_ISO_COARSE_EXACT_SIDE_SHEAR_VALENCE_SUM_MIN", "8"))
    return max(something(tryparse(Int, raw), 8), 0)
end

@inline function q4_flat_curved_iso_coarse_exact_side_shear_valence_sum_max()
    raw = strip(get(ENV, "JFEM_SOL105_EIG_FLAT_CURVED_ISO_COARSE_EXACT_SIDE_SHEAR_VALENCE_SUM_MAX", "10"))
    return max(something(tryparse(Int, raw), 10), 0)
end

@inline function q4_sol105_flat_iso_exact_side_rotcorr()
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_ISO_EXACT_SIDE_ROTCORR", false)
end

@inline function q4_flat_curved_iso_eig_center_only_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_CURVED_ISO_CENTER_ONLY", false)
end

@inline function q4_flat_curved_iso_eig_center_only_kappa_l_min()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_CURVED_ISO_CENTER_ONLY_KAPPA_L_MIN", 0.02), 0.0)
end

@inline function q4_flat_curved_iso_eig_center_only_cyl_ratio_max()
    return clamp(solver_env_float("JFEM_SOL105_EIG_FLAT_CURVED_ISO_CENTER_ONLY_CYL_RATIO_MAX", 0.05), 0.0, 1.0)
end

@inline function q4_flat_curved_iso_exact_membrane_aspect_ratio_max()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_CURVED_ISO_EXACT_MEMBRANE_ASPECT_RATIO_MAX", 1.0e12), 1.0)
end

@inline function q4_flat_curved_pcomp_fullshear_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_CURVED_PCOMP_FULLSHEAR", false)
end

@inline function q4_flat_curved_pcomp_fullshear_kappa_l_min()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_CURVED_PCOMP_FULLSHEAR_KAPPA_L_MIN", 0.005), 0.0)
end

@inline function q4_flat_curved_pcomp_fullshear_cyl_ratio_max()
    return clamp(solver_env_float("JFEM_SOL105_EIG_FLAT_CURVED_PCOMP_FULLSHEAR_CYL_RATIO_MAX", 0.8), 0.0, 1.0)
end

@inline function q4_flat_curved_pcomp_geomnormal_frame_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_CURVED_PCOMP_GEOMNORMAL_FRAME", false)
end

@inline function q4_sol105_flat_pcomp_center_only_h_over_l_max()
    # Reduced shear is only beneficial in the thin-laminate regime. A larger
    # cutoff pushes moderately thick flat laminates onto an overly soft path.
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_PCOMP_CENTER_ONLY_H_OVER_L_MAX", 0.10), 0.0)
end

@inline function q4_sol105_pcomp_auto_membrane_incomp_enabled()
    # Conservative by default. Wilson membrane modes are a legitimate
    # formulation option, but automatic PCOMP activation needs a derivation
    # stronger than ply-count/thickness thresholds.
    return solver_env_bool("JFEM_SOL105_EIG_PCOMP_AUTO_MEMBRANE_INCOMP", false)
end

@inline function q4_sol105_pcomp_auto_membrane_incomp_ply_count_min()
    raw = strip(get(ENV, "JFEM_SOL105_EIG_PCOMP_AUTO_MEMBRANE_INCOMP_PLY_COUNT_MIN", "13"))
    return max(something(tryparse(Int, raw), 13), 1)
end

@inline function q4_sol105_pcomp_auto_membrane_incomp_thickness_min()
    return max(solver_env_float("JFEM_SOL105_EIG_PCOMP_AUTO_MEMBRANE_INCOMP_THICKNESS_MIN", 2.5), 0.0)
end

@inline function q4_sol105_pcomp_auto_membrane_incomp_candidate(prop)
    q4_sol105_pcomp_auto_membrane_incomp_enabled() || return false
    get(prop, "TYPE", "") == "PCOMP_CLT" || return false
    get(prop, "IS_ISOTROPIC", false) && return false
    ply_count = length(get(prop, "PLY_DATA", Any[]))
    total_thickness = Float64(get(prop, "T", 0.0))
    return ply_count >= q4_sol105_pcomp_auto_membrane_incomp_ply_count_min() &&
           total_thickness >= q4_sol105_pcomp_auto_membrane_incomp_thickness_min()
end

@inline function q4_sol105_pcomp_auto_element_axis_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_PCOMP_AUTO_ELEMENT_AXIS", false)
end

@inline function q4_sol105_pcomp_auto_element_axis_kappa_l_min()
    return max(solver_env_float("JFEM_SOL105_EIG_PCOMP_AUTO_ELEMENT_AXIS_KAPPA_L_MIN", 0.005), 0.0)
end

@inline function q4_sol105_pcomp_auto_element_axis_cyl_ratio_min()
    return clamp(solver_env_float("JFEM_SOL105_EIG_PCOMP_AUTO_ELEMENT_AXIS_CYL_RATIO_MIN", 0.0), 0.0, 1.0)
end

@inline function q4_sol105_pcomp_auto_element_axis_candidate(prop, kappa_l::Float64, cyl_ratio::Float64)
    q4_sol105_pcomp_auto_element_axis_enabled() || return false
    q4_sol105_pcomp_auto_membrane_incomp_candidate(prop) || return false
    return kappa_l >= q4_sol105_pcomp_auto_element_axis_kappa_l_min() &&
           cyl_ratio >= q4_sol105_pcomp_auto_element_axis_cyl_ratio_min()
end

@inline function q4_sol105_pcomp_auto_element_axis_candidate(prop_candidate::Bool, kappa_l::Float64, cyl_ratio::Float64)
    q4_sol105_pcomp_auto_element_axis_enabled() || return false
    prop_candidate || return false
    return kappa_l >= q4_sol105_pcomp_auto_element_axis_kappa_l_min() &&
           cyl_ratio >= q4_sol105_pcomp_auto_element_axis_cyl_ratio_min()
end

@inline function q4_sol105_flat_cyl_iso_bend_scale()
    # Default 1.0 neutral (2026-04-21 cleanup): was 0.6 eyeballed bend softening.
    return clamp(solver_env_float("JFEM_SOL105_EIG_FLAT_CYL_ISO_BEND_SCALE", 1.0), 0.0, 1.0)
end

@inline function q4_sol105_flat_cyl_iso_bend_kappa_l_min()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_CYL_ISO_BEND_KAPPA_L_MIN", 0.06), 0.0)
end

@inline function q4_sol105_flat_cyl_iso_bend_kappa_l_full()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_CYL_ISO_BEND_KAPPA_L_FULL", 0.10), 0.0)
end

@inline function q4_sol105_flat_cyl_iso_bend_cyl_ratio_max()
    return clamp(solver_env_float("JFEM_SOL105_EIG_FLAT_CYL_ISO_BEND_CYL_RATIO_MAX", 0.02), 0.0, 1.0)
end

@inline function q4_sol105_flat_cyl_iso_bend_effective_scale(kappa_l::Float64, cyl_ratio::Float64)
    base_scale = q4_sol105_flat_cyl_iso_bend_scale()
    base_scale >= 0.999999 && return 1.0
    cyl_ratio > q4_sol105_flat_cyl_iso_bend_cyl_ratio_max() && return 1.0

    kappa_min = q4_sol105_flat_cyl_iso_bend_kappa_l_min()
    kappa_full = q4_sol105_flat_cyl_iso_bend_kappa_l_full()
    weight = if kappa_full <= kappa_min
        kappa_l >= kappa_min ? 1.0 : 0.0
    else
        clamp((kappa_l - kappa_min) / max(kappa_full - kappa_min, 1e-12), 0.0, 1.0)
    end
    return 1.0 - (1.0 - base_scale) * weight
end

@inline function shell_sol105_iso_eig_k6rot_override()
    value = solver_env_optional_float("JFEM_SOL105_EIG_ISO_K6ROT")
    isnothing(value) || return value
    return solver_env_optional_float("JFEM_SOL105_EIG_FLAT_ISO_K6ROT")
end

@inline function shell_sol101_iso_static_k6rot_override()
    value = solver_env_optional_float("JFEM_SOL101_ISO_K6ROT")
    isnothing(value) || return value
    return solver_env_optional_float("JFEM_STATIC_ISO_K6ROT")
end

@inline function shell_sol105_iso_eig_k6rot_cyl_ratio_min()
    if haskey(ENV, "JFEM_SOL105_EIG_ISO_K6ROT_CYL_RATIO_MIN")
        return clamp(solver_env_float("JFEM_SOL105_EIG_ISO_K6ROT_CYL_RATIO_MIN", 0.20), 0.0, 1.0)
    end
    return clamp(solver_env_float("JFEM_SOL105_EIG_FLAT_ISO_K6ROT_CYL_RATIO_MIN", 0.20), 0.0, 1.0)
end

@inline function shell_sol105_iso_eig_drill_scale_override()
    value = solver_env_optional_float("JFEM_SOL105_EIG_ISO_DRILL_SCALE")
    if !isnothing(value)
        return clamp(value, 0.0, 1.0)
    end
    value = solver_env_optional_float("JFEM_SOL105_EIG_FLAT_ISO_DRILL_SCALE")
    return isnothing(value) ? nothing : clamp(value, 0.0, 1.0)
end

@inline function shell_sol101_iso_static_drill_scale_override()
    value = solver_env_optional_float("JFEM_SOL101_ISO_DRILL_SCALE")
    if !isnothing(value)
        return clamp(value, 0.0, 1.0)
    end
    value = solver_env_optional_float("JFEM_STATIC_ISO_DRILL_SCALE")
    return isnothing(value) ? nothing : clamp(value, 0.0, 1.0)
end

@inline function shell_sol105_iso_eig_drill_scale_cyl_ratio_min()
    if haskey(ENV, "JFEM_SOL105_EIG_ISO_DRILL_SCALE_CYL_RATIO_MIN")
        return clamp(solver_env_float("JFEM_SOL105_EIG_ISO_DRILL_SCALE_CYL_RATIO_MIN", 0.20), 0.0, 1.0)
    end
    return clamp(solver_env_float("JFEM_SOL105_EIG_FLAT_ISO_DRILL_SCALE_CYL_RATIO_MIN", 0.20), 0.0, 1.0)
end

@inline function q4_flat_curved_iso_geomnormal_frame_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_CURVED_ISO_GEOMNORMAL_FRAME", false)
end

@inline function q4_flat_curved_iso_geomnormal_frame_aspect_ratio_min()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_CURVED_ISO_GEOMNORMAL_FRAME_ASPECT_RATIO_MIN", 1.5), 1.0)
end

@inline function q4_flat_curved_iso_geomnormal_frame_kappa_l_min()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_CURVED_ISO_GEOMNORMAL_FRAME_KAPPA_L_MIN", 0.02), 0.0)
end

@inline function q4_flat_curved_iso_geomnormal_frame_kappa_l_max()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_CURVED_ISO_GEOMNORMAL_FRAME_KAPPA_L_MAX", 0.30), 0.0)
end

@inline function q4_flat_curved_iso_geomnormal_frame_cyl_ratio_max()
    return clamp(solver_env_float("JFEM_SOL105_EIG_FLAT_CURVED_ISO_GEOMNORMAL_FRAME_CYL_RATIO_MAX", 0.05), 0.0, 1.0)
end

@inline function q4_flat_curved_iso_nodal_geomnormal_transform_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_EIG_FLAT_CURVED_ISO_NODAL_GEOMNORMAL_TRANSFORM", false)
end

@inline function q4_flat_curved_iso_nodal_geomnormal_transform_aspect_ratio_min()
    return max(solver_env_float("JFEM_SOL105_EIG_FLAT_CURVED_ISO_NODAL_GEOMNORMAL_TRANSFORM_ASPECT_RATIO_MIN", 1.8), 1.0)
end

@inline function q4_flat_curved_iso_nodal_geomnormal_transform_valence_sum_max()
    raw = strip(get(ENV, "JFEM_SOL105_EIG_FLAT_CURVED_ISO_NODAL_GEOMNORMAL_TRANSFORM_VALENCE_SUM_MAX", "6"))
    return max(something(tryparse(Int, raw), 6), 0)
end

@inline function q4_curved_iso_geomnormal_frame_enabled()
    # Default off (2026-04-21 cleanup).
    return solver_env_bool("JFEM_SOL105_CURVED_ISO_GEOMNORMAL_FRAME", false)
end

@inline function q4_curved_iso_geomnormal_frame_aspect_ratio_min()
    return max(solver_env_float("JFEM_SOL105_CURVED_ISO_GEOMNORMAL_FRAME_ASPECT_RATIO_MIN", 1.5), 1.0)
end

@inline function q4_curved_iso_geomnormal_frame_kappa_l_min()
    return max(solver_env_float("JFEM_SOL105_CURVED_ISO_GEOMNORMAL_FRAME_KAPPA_L_MIN", 0.005), 0.0)
end

@inline function q4_curved_iso_geomnormal_frame_kappa_l_max()
    return max(solver_env_float("JFEM_SOL105_CURVED_ISO_GEOMNORMAL_FRAME_KAPPA_L_MAX", 0.30), 0.0)
end

@inline function q4_curved_iso_geomnormal_frame_cyl_ratio_max()
    return clamp(solver_env_float("JFEM_SOL105_CURVED_ISO_GEOMNORMAL_FRAME_CYL_RATIO_MAX", 0.2), 0.0, 1.0)
end

@inline function shell_element_frame_quad4_with_normal(
    p1::SVector{3,Float64},
    p2::SVector{3,Float64},
    p3::SVector{3,Float64},
    p4::SVector{3,Float64},
    v3::SVector{3,Float64},
    mode::Symbol=:diag,
)
    gxi = 0.25 * (-p1 + p2 + p3 - p4)
    geta = 0.25 * (-p1 - p2 + p3 + p4)

    if mode == :parametric
        x_raw = gxi
        if dot(x_raw, x_raw) <= 1e-24
            x_raw = p2 - p1
        end
    elseif mode == :edge
        x_raw = p2 - p1
        if dot(x_raw, x_raw) <= 1e-24
            x_raw = gxi
        end
    else
        d13 = p3 - p1
        d24 = p4 - p2
        d13n = normalize(d13)
        d24n = normalize(d24)
        x_raw = d13n - d24n
        if dot(x_raw, x_raw) < 1e-20
            x_raw = d13n + d24n
        end
    end

    x_proj = x_raw - dot(x_raw, v3) * v3
    if dot(x_proj, x_proj) <= 1e-24
        x_alt = geta - dot(geta, v3) * v3
        if dot(x_alt, x_alt) <= 1e-24
            x_alt = (p2 - p1) - dot(p2 - p1, v3) * v3
        end
        x_proj = x_alt
    end

    v1 = normalize(x_proj)
    v2 = cross(v3, v1)
    return v1, v2, v3
end

@inline function shell_element_frame_quad4(p1::SVector{3,Float64}, p2::SVector{3,Float64},
                                           p3::SVector{3,Float64}, p4::SVector{3,Float64},
                                           mode::Symbol=:diag)
    gxi  = 0.25 * (-p1 + p2 + p3 - p4)
    geta = 0.25 * (-p1 - p2 + p3 + p4)
    v3_raw = cross(gxi, geta)
    if dot(v3_raw, v3_raw) <= 1e-24
        d13 = p3 - p1
        d24 = p4 - p2
        v3_raw = cross(d13, d24)
    end
    v3 = normalize(v3_raw)
    return shell_element_frame_quad4_with_normal(p1, p2, p3, p4, v3, mode)
end

function shell_element_frame_fast(p1::SVector{3,Float64}, p2::SVector{3,Float64}, p3::SVector{3,Float64}, p4::SVector{3,Float64}, n::Int)
    if n == 4
        return shell_element_frame_quad4(p1, p2, p3, p4, q4_frame_mode_from_env("JFEM_Q4_FRAME_MODE"))
    else
        v1 = normalize(p2 - p1)
        v3 = normalize(cross(v1, p3 - p1))
        v2 = cross(v3, v1)
        return v1, v2, v3
    end
end

@inline function estimate_quad4_curvature_membrane(lc::AbstractMatrix,
                                                   n1::SVector{3,Float64}, n2::SVector{3,Float64},
                                                   n3::SVector{3,Float64}, n4::SVector{3,Float64},
                                                   v1::SVector{3,Float64}, v2::SVector{3,Float64},
                                                   v3::SVector{3,Float64})
    normals = (n1, n2, n3, n4)
    a11 = 0.0; a12 = 0.0; a22 = 0.0
    bx1 = 0.0; bx2 = 0.0; bx3 = 0.0
    by1 = 0.0; by2 = 0.0; by3 = 0.0

    @inbounds for k in 1:4
        x = lc[k,1]
        y = lc[k,2]
        dn = normals[k] - dot(normals[k], v3) * v3
        a11 += x * x
        a12 += x * y
        a22 += y * y
        bx1 += x * dn[1]; bx2 += x * dn[2]; bx3 += x * dn[3]
        by1 += y * dn[1]; by2 += y * dn[2]; by3 += y * dn[3]
    end

    det = a11 * a22 - a12 * a12
    if abs(det) <= 1e-24
        return SVector(0.0, 0.0, 0.0)
    end
    invdet = 1.0 / det

    dn_dx = SVector(
        invdet * ( a22 * bx1 - a12 * by1),
        invdet * ( a22 * bx2 - a12 * by2),
        invdet * ( a22 * bx3 - a12 * by3),
    )
    dn_dy = SVector(
        invdet * (-a12 * bx1 + a11 * by1),
        invdet * (-a12 * bx2 + a11 * by2),
        invdet * (-a12 * bx3 + a11 * by3),
    )

    k11 = -dot(dn_dx, v1)
    k22 = -dot(dn_dy, v2)
    k12 = -0.5 * (dot(dn_dx, v2) + dot(dn_dy, v1))
    return SVector(k11, k22, k12)
end

@inline function q4_curvature_gaussian(curvature::SVector{3,Float64})
    return curvature[1] * curvature[2] - curvature[3] * curvature[3]
end

# Marguerre shallow-shell coupling helpers (2026-04-21)
# ----------------------------------------------------
# Ibrahimbegović 1994 Part I Eq. 6.14 couples the in-plane rotation DOFs with
# the reference-midsurface slopes f_{,α} (first derivatives of the shell
# mid-surface z-coordinate wrt element-local coordinates):
#
#     ε_αβ^lin = ½(u_{α,β} + u_{β,α}) + ½(θ̃_α f_{,β} + θ̃_β f_{,α})
#
# This is DIFFERENT from the Koiter-Donnell -κ·w coupling on the w-DOF that
# JFEM's legacy `curvature_membrane` used (now neutralized at default scale=0).
# For a planar element all 4 SNORM-averaged normals point along v3 and the
# slope is zero; for an element on a curved shell patch the SNORM normals
# tilt away from v3 and their tangential components give the slopes.

@inline function q4_marguerre_coupling_enabled()
    return solver_env_bool("JFEM_Q4_MARGUERRE_COUPLING", false)
end

@inline function q4_marguerre_coupling_scale()
    # Neutral default 1.0 — the physical coupling is derived from geometry,
    # no fudge factor. Env exposed for ablation only.
    return solver_env_float("JFEM_Q4_MARGUERRE_COUPLING_SCALE", 1.0)
end

@inline function q4_nonflat_pcomp_exact_cs_enabled()
    # Experimental (2026-04-21): use exact ply-integrated transverse shear matrix
    # (q4_Cs_raw_flat) on non-flat curved PCOMP elements, matching the existing
    # flat-PCOMP treatment. Investigating HTP_launch +3.8% mode-1 residual.
    # Hypothesis: MSC Nastran uses ply-integrated Cs on non-flat PCOMP too; JFEM's
    # default generic Cs surrogate over-stiffens these elements.
    return solver_env_bool("JFEM_SOL105_EIG_NONFLAT_PCOMP_EXACT_CS", false)
end

@inline function q4_marguerre_coupling_convention()
    # Which rotation DOF carries the ε_αα coupling.
    # :jfem_kl (default) — derived from JFEM Bb sign convention (θ_y=∂w/∂x, θ_x=∂w/∂y):
    #    ε_xx on θ_y (idx+5), ε_yy on θ_x (idx+4)
    # :handover — per HANDOVER_SOL105_CURVED_PCOMP.md §5 (Ibrahimbegović literal indexing):
    #    ε_xx on θ_x (idx+4), ε_yy on θ_y (idx+5)
    raw = lowercase(strip(get(ENV, "JFEM_Q4_MARGUERRE_CONVENTION", "jfem_kl")))
    if raw in ("handover", "ibrahimbegovic", "literal")
        return :handover
    end
    return :jfem_kl
end

# Curved-surface Jacobian helpers (2026-04-21 PM)
# ------------------------------------------------
# Foundation for SOL105 buckling fix path #2 (per handover
# HANDOVER_SOL105_SESSION_2026_04_21_PM.md §5.4 and memory record
# project_htp_jacobian_gap_diagnostic_2026_04_21.md).
#
# JFEM's current CQUAD4 stiffness assembly projects 3D corner coords onto a
# single element-center tangent plane (the `lc` 4×2 matrix at
# assembly.jl:1058-1063) before calling stiffness_quad4_matrices. Proper
# Naghdi-Reissner-Mindlin (Katili-Batoz-Ibrahimbegović 2018, "DKMQ24 for
# composite structures") integrates over the true curved mid-surface with
# a DIFFERENT local tangent basis at each Gauss point. Diagnostic on
# HTP_launch bad elements showed GP-local vs flat-projection Jacobian
# differs by 0.8%/1.55% (mean/max) — accounts for 1-3pt of the 5-9% HTP
# eigenvalue gap.
#
# `q4_curved_jacobian_enabled` env-gates the curved-integration path.  The
# formulation-level switch JFEM_Q4_CURVED_JACOBIAN applies to both the static
# preload stiffness and the SOL105 eigen stiffness, which keeps a linear
# buckling run on one B-operator family.  The older
# JFEM_SOL105_EIG_CURVED_JACOBIAN key is kept as an eigen-only ablation switch
# for reproducing previous split-K studies.
#
# Note: the full NRM kinematic coupling (Koiter -b·w in ε, curvature in χ,
# -b·u in γ) must be added together with the Jacobian change; partial NRM
# breaks variational consistency (see
# project_htp_curvature_coupling_overshoot_2026_04_21.md). The
# `q4_eig_curved_jacobian_enabled` flag gates all NRM terms in lockstep.

@inline function q4_eig_curved_jacobian_enabled()
    return solver_env_bool("JFEM_SOL105_EIG_CURVED_JACOBIAN", false)
end

@inline function q4_curved_jacobian_enabled(shear_center_only::Bool)
    if haskey(ENV, "JFEM_Q4_CURVED_JACOBIAN")
        return solver_env_bool("JFEM_Q4_CURVED_JACOBIAN", false)
    end
    if shear_center_only
        return q4_eig_curved_jacobian_enabled()
    end
    return solver_env_bool("JFEM_Q4_STATIC_CURVED_JACOBIAN", false)
end

# Compute Gauss-point-local tangent-plane data from 3D nodal coords for a
# bilinear quad at isoparametric point (xi, eta). Returns:
#   a1 — 3D tangent vector along ξ at (xi, eta)   (= Σ_k dN_k/dξ · p_k)
#   a2 — 3D tangent vector along η at (xi, eta)
#   n  — surface unit normal at (xi, eta) = (a1 × a2) / |a1 × a2|
#   area_elem — |a1 × a2|, the surface area element (dA = area_elem dξ dη)
#   J  — 2×2 matrix [a1·t1 a1·t2; a2·t1 a2·t2] where (t1, t2, n) is a
#        GP-local orthonormal frame with t1 = a1/|a1|, t2 = n × t1.
# On a FLAT quad, J matches the 2×2 Jacobian computed from the flat-projected
# lc exactly — verified in test/test_quad4_curved_jacobian_equivalence.jl.
@inline function quad4_gp_local_frame(coords_3d::AbstractMatrix,
                                       xi::Float64, eta::Float64)
    # Shape function derivatives at (ξ, η)
    dNr1 = -0.25 * (1 - eta)
    dNr2 =  0.25 * (1 - eta)
    dNr3 =  0.25 * (1 + eta)
    dNr4 = -0.25 * (1 + eta)
    dNs1 = -0.25 * (1 - xi)
    dNs2 = -0.25 * (1 + xi)
    dNs3 =  0.25 * (1 + xi)
    dNs4 =  0.25 * (1 - xi)

    # 3D tangent vectors a1 = ∂x/∂ξ, a2 = ∂x/∂η
    a1 = SVector(
        dNr1 * coords_3d[1,1] + dNr2 * coords_3d[2,1] + dNr3 * coords_3d[3,1] + dNr4 * coords_3d[4,1],
        dNr1 * coords_3d[1,2] + dNr2 * coords_3d[2,2] + dNr3 * coords_3d[3,2] + dNr4 * coords_3d[4,2],
        dNr1 * coords_3d[1,3] + dNr2 * coords_3d[2,3] + dNr3 * coords_3d[3,3] + dNr4 * coords_3d[4,3],
    )
    a2 = SVector(
        dNs1 * coords_3d[1,1] + dNs2 * coords_3d[2,1] + dNs3 * coords_3d[3,1] + dNs4 * coords_3d[4,1],
        dNs1 * coords_3d[1,2] + dNs2 * coords_3d[2,2] + dNs3 * coords_3d[3,2] + dNs4 * coords_3d[4,2],
        dNs1 * coords_3d[1,3] + dNs2 * coords_3d[2,3] + dNs3 * coords_3d[3,3] + dNs4 * coords_3d[4,3],
    )

    # Surface normal and area at the Gauss point
    cross_a = cross(a1, a2)
    area_elem = norm(cross_a)
    n_gp = area_elem > 1e-30 ? cross_a / area_elem : SVector(0.0, 0.0, 1.0)

    # GP-local orthonormal tangent basis.
    a1_len = norm(a1)
    t1 = a1_len > 1e-30 ? a1 / a1_len : SVector(1.0, 0.0, 0.0)
    t2 = cross(n_gp, t1)

    # 2×2 Jacobian in GP-local basis. By construction, J11 = |a1| and J12 = 0.
    J11 = a1_len
    J12 = 0.0
    J21 = dot(a2, t1)
    J22 = dot(a2, t2)

    return a1, a2, n_gp, area_elem, t1, t2, J11, J12, J21, J22
end

@inline function estimate_quad4_slope_membrane(n1::SVector{3,Float64},
                                                n2::SVector{3,Float64},
                                                n3::SVector{3,Float64},
                                                n4::SVector{3,Float64},
                                                v1::SVector{3,Float64},
                                                v2::SVector{3,Float64},
                                                v3::SVector{3,Float64})
    # Returns 8-tuple (s1x, s1y, s2x, s2y, s3x, s3y, s4x, s4y) — per-corner
    # (f_{,x}, f_{,y}) in element-local frame.
    # Small-angle approximation: for a node with averaged normal n,
    # the local shell slope is s_α = −(n · v_α) / (n · v3).
    # When n == v3 (planar element), slope is 0.
    function node_slopes(nk)
        n_v3 = dot(nk, v3)
        if abs(n_v3) < 1e-6
            return (0.0, 0.0)
        end
        inv = 1.0 / n_v3
        return (-dot(nk, v1) * inv, -dot(nk, v2) * inv)
    end
    s1x, s1y = node_slopes(n1)
    s2x, s2y = node_slopes(n2)
    s3x, s3y = node_slopes(n3)
    s4x, s4y = node_slopes(n4)
    return SVector{8,Float64}(s1x, s1y, s2x, s2y, s3x, s3y, s4x, s4y)
end

@inline function quad4_corner_normals(
    p1::SVector{3,Float64},
    p2::SVector{3,Float64},
    p3::SVector{3,Float64},
    p4::SVector{3,Float64},
)
    pts = (p1, p2, p3, p4)
    ref_raw = cross(p3 - p1, p4 - p2)
    ref_len = norm(ref_raw)
    if ref_len <= 1e-30
        ref_raw = cross(p2 - p1, p4 - p1)
        ref_len = norm(ref_raw)
    end
    ref_nrm = ref_len <= 1e-30 ? SVector(0.0, 0.0, 1.0) : SVector(ref_raw / ref_len)

    return ntuple(4) do k
        prev = k == 1 ? 4 : k - 1
        next = k == 4 ? 1 : k + 1
        pk = pts[k]
        pprev = pts[prev]
        pnext = pts[next]
        raw = cross(pnext - pk, pprev - pk)
        len = norm(raw)
        if len <= 1e-30
            ref_nrm
        else
            nrm = SVector(raw / len)
            dot(nrm, ref_nrm) < 0.0 ? -nrm : nrm
        end
    end
end

@inline function estimate_quad4_corner_curvature_membrane(
    lc::AbstractMatrix,
    p1::SVector{3,Float64},
    p2::SVector{3,Float64},
    p3::SVector{3,Float64},
    p4::SVector{3,Float64},
    v1::SVector{3,Float64},
    v2::SVector{3,Float64},
    v3::SVector{3,Float64},
)
    n1, n2, n3, n4 = quad4_corner_normals(p1, p2, p3, p4)
    return estimate_quad4_curvature_membrane(lc, n1, n2, n3, n4, v1, v2, v3)
end

@inline function shell_material_rotation_from_g12(v1::SVector{3,Float64}, v2::SVector{3,Float64}, v3::SVector{3,Float64},
                                                  p1::SVector{3,Float64}, p2::SVector{3,Float64}, theta_rad::Float64)
    edge12 = p2 - p1
    edge12p = edge12 - dot(edge12, v3) * v3
    len12p2 = dot(edge12p, edge12p)
    if len12p2 <= 1e-24
        return theta_rad
    end
    x_mat = edge12p / sqrt(len12p2)
    return theta_rad + atan(dot(x_mat, v2), dot(x_mat, v1))
end

@inline function shell_material_rotation_from_global_x(v1::SVector{3,Float64}, v2::SVector{3,Float64}, v3::SVector{3,Float64},
                                                       theta_rad::Float64)
    gx = SVector(1.0, 0.0, 0.0)
    x_proj = gx - dot(gx, v3) * v3
    lenxp2 = dot(x_proj, x_proj)
    if lenxp2 <= 1e-24
        return theta_rad
    end
    x_mat = x_proj / sqrt(lenxp2)
    return theta_rad + atan(dot(x_mat, v2), dot(x_mat, v1))
end

@inline function _coord_axis_svector(coord::AbstractDict, key::String, default::SVector{3,Float64})
    raw = get(coord, key, nothing)
    raw === nothing && return default
    return SVector{3,Float64}(Float64(raw[1]), Float64(raw[2]), Float64(raw[3]))
end

@inline function shell_material_rotation_from_coord(v1::SVector{3,Float64}, v2::SVector{3,Float64}, v3::SVector{3,Float64},
                                                    coord::AbstractDict, theta_rad::Float64=0.0)
    x_ref = _coord_axis_svector(coord, "U", SVector(1.0, 0.0, 0.0))
    y_ref = _coord_axis_svector(coord, "V", SVector(0.0, 1.0, 0.0))
    x_proj = x_ref - dot(x_ref, v3) * v3
    lenxp2 = dot(x_proj, x_proj)
    if lenxp2 <= 1e-24
        x_proj = y_ref - dot(y_ref, v3) * v3
        lenxp2 = dot(x_proj, x_proj)
    end
    if lenxp2 <= 1e-24
        return theta_rad
    end
    x_mat = x_proj / sqrt(lenxp2)
    return theta_rad + atan(dot(x_mat, v2), dot(x_mat, v1))
end

@inline function shell_pcomp_kg_rotation(mode::Symbol,
                                         v1::SVector{3,Float64}, v2::SVector{3,Float64}, v3::SVector{3,Float64},
                                         p1::SVector{3,Float64}, p2::SVector{3,Float64},
                                         p3::SVector{3,Float64}, p4::SVector{3,Float64},
                                         theta_rad::Float64)
    if mode === :element
        return theta_rad
    elseif mode === :g12
        return shell_material_rotation_from_g12(v1, v2, v3, p1, p2, theta_rad)
    elseif mode === :global_x
        return shell_material_rotation_from_global_x(v1, v2, v3, theta_rad)
    end

    d13_geom = p3 - p1
    d24_geom = p4 - p2
    v3_geom_raw = cross(d13_geom, d24_geom)
    v3_geom_len = norm(v3_geom_raw)
    if v3_geom_len <= 1e-12
        return theta_rad
    end

    c_warp = (p1 + p2 + p3 + p4) / 4.0
    v3g = v3_geom_raw / v3_geom_len
    max_dev = max(abs(dot(p1-c_warp, v3g)), abs(dot(p2-c_warp, v3g)),
                  abs(dot(p3-c_warp, v3g)), abs(dot(p4-c_warp, v3g)))
    L_diag = max(norm(d13_geom), norm(d24_geom))
    if max_dev / max(L_diag, 1e-12) > q4_pcomp_kg_warp_ratio_threshold()
        return shell_material_rotation_from_global_x(v1, v2, v3, theta_rad)
    end
    return theta_rad
end

@inline function shell_pcomp_kg_rotation(mode::Symbol,
                                         v1::SVector{3,Float64}, v2::SVector{3,Float64}, v3::SVector{3,Float64},
                                         p1::SVector{3,Float64}, p2::SVector{3,Float64},
                                         p3::SVector{3,Float64}, p4::SVector{3,Float64},
                                         theta_rad::Float64,
                                         mcid::Integer,
                                         cords)
    if mcid > 0 && cords !== nothing && haskey(cords, string(mcid))
        return shell_material_rotation_from_coord(v1, v2, v3, cords[string(mcid)], 0.0)
    end
    return shell_pcomp_kg_rotation(mode, v1, v2, v3, p1, p2, p3, p4, theta_rad)
end

@inline function shell_pcomp_kg_rotation(v1::SVector{3,Float64}, v2::SVector{3,Float64}, v3::SVector{3,Float64},
                                         p1::SVector{3,Float64}, p2::SVector{3,Float64},
                                         p3::SVector{3,Float64}, p4::SVector{3,Float64},
                                         theta_rad::Float64)
    return shell_pcomp_kg_rotation(q4_pcomp_kg_axis_mode(), v1, v2, v3, p1, p2, p3, p4, theta_rad)
end

@inline function shell_pcomp_material_rotation(mode::Symbol,
                                               v1::SVector{3,Float64}, v2::SVector{3,Float64}, v3::SVector{3,Float64},
                                               p1::SVector{3,Float64}, p2::SVector{3,Float64},
                                               theta_rad::Float64)
    # Triangle / 2-corner path: no warp check is possible without p3,p4,
    # so :warp_switch collapses to :element and any unknown mode falls through
    # to the element (theta_rad) frame. This keeps TRIA3 callers consistent
    # with the CQUAD4 4-corner path when the quad is planar.
    if mode === :g12
        return shell_material_rotation_from_g12(v1, v2, v3, p1, p2, theta_rad)
    elseif mode === :global_x
        return shell_material_rotation_from_global_x(v1, v2, v3, theta_rad)
    end
    return theta_rad
end

@inline function shell_pcomp_material_rotation(mode::Symbol,
                                               v1::SVector{3,Float64}, v2::SVector{3,Float64}, v3::SVector{3,Float64},
                                               p1::SVector{3,Float64}, p2::SVector{3,Float64},
                                               theta_rad::Float64,
                                               mcid::Integer,
                                               cords)
    if mcid > 0 && cords !== nothing && haskey(cords, string(mcid))
        return shell_material_rotation_from_coord(v1, v2, v3, cords[string(mcid)], 0.0)
    end
    return shell_pcomp_material_rotation(mode, v1, v2, v3, p1, p2, theta_rad)
end

# 4-corner overloads delegate to shell_pcomp_kg_rotation so that K_eig (CQUAD4)
# and Kg (CQUAD4) compute the laminate-axis rotation through a single
# implementation. This eliminates the prior asymmetry where only the Kg path
# handled :warp_switch (physical-warp → :global_x) while K_eig fell through to
# theta_rad for the same mode, producing rotation mismatches that biased the
# buckling eigenvalue problem on warped PCOMP meshes.
@inline function shell_pcomp_material_rotation(mode::Symbol,
                                               v1::SVector{3,Float64}, v2::SVector{3,Float64}, v3::SVector{3,Float64},
                                               p1::SVector{3,Float64}, p2::SVector{3,Float64},
                                               p3::SVector{3,Float64}, p4::SVector{3,Float64},
                                               theta_rad::Float64)
    return shell_pcomp_kg_rotation(mode, v1, v2, v3, p1, p2, p3, p4, theta_rad)
end

@inline function shell_pcomp_material_rotation(mode::Symbol,
                                               v1::SVector{3,Float64}, v2::SVector{3,Float64}, v3::SVector{3,Float64},
                                               p1::SVector{3,Float64}, p2::SVector{3,Float64},
                                               p3::SVector{3,Float64}, p4::SVector{3,Float64},
                                               theta_rad::Float64,
                                               mcid::Integer,
                                               cords)
    return shell_pcomp_kg_rotation(mode, v1, v2, v3, p1, p2, p3, p4, theta_rad, mcid, cords)
end

function rotation_from_normal(n_avg::Vector{Float64})
    z = normalize(n_avg)
    ref = abs(z[3]) < 0.9 ? SVector(0.0, 0.0, 1.0) : SVector(1.0, 0.0, 0.0)
    x = normalize(cross(ref, SVector(z...)))
    y = cross(SVector(z...), x)
    return hcat(x, y, z)
end

# Apply CBAR pin flags via static condensation
# PA/PB are integers whose digits specify released local DOFs (e.g. PA=45 releases DOFs 4,5 at end A)
function apply_bar_pin_flags!(Ke::Matrix{Float64}, pa::Int, pb::Int)
    if pa == 0 && pb == 0; return; end
    released = Int[]
    if pa > 0
        for ch in string(pa)
            d = parse(Int, ch)
            if 1 <= d <= 6; push!(released, d); end
        end
    end
    if pb > 0
        for ch in string(pb)
            d = parse(Int, ch)
            if 1 <= d <= 6; push!(released, d + 6); end
        end
    end
    isempty(released) && return
    retained = setdiff(1:12, released)
    Kff = Ke[released, released]
    Kfr = Ke[released, retained]
    Krf = Ke[retained, released]
    Krr = Ke[retained, retained]
    # Static condensation: K_cond = Krr - Krf * inv(Kff) * Kfr
    # Use pinv to handle singular Kff (e.g. rod-like bars with zero rotational stiffness)
    Kff_inv = pinv(Kff)
    Kcond = Krr - Krf * Kff_inv * Kfr
    # Zero entire matrix, then fill retained DOFs
    fill!(Ke, 0.0)
    for (ic, c) in enumerate(retained), (ir, r) in enumerate(retained)
        Ke[r, c] = Kcond[ir, ic]
    end
end

function get_coord_transform(model, cid, vec)
    if cid == 0; return vec; end
    if !haskey(model["CORDs"], string(cid)); return vec; end
    cord = model["CORDs"][string(cid)]
    R = hcat(cord["U"], cord["V"], cord["W"])
    return R * vec
end
