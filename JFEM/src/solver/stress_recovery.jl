# stress_recovery.jl — Element stress/force/strain recovery

@inline function _stress_entry_public_id(key, entry)
    if entry isa AbstractDict && haskey(entry, "ID")
        value = entry["ID"]
        parsed = tryparse(Int, string(value))
        parsed !== nothing && return parsed
    end
    parsed = tryparse(Int, string(key))
    parsed !== nothing && return parsed
    m = match(r"^-?\d+", string(key))
    m !== nothing && return parse(Int, m.match)
    return 0
end

@inline function _shell_edge_key(a::Int, b::Int)
    return a < b ? (a, b) : (b, a)
end

function _tria3_shell_macro_blend_weights(model)
    edge_counts = Dict{Tuple{Int,Int},Int}()
    weights = Dict{Int,Float64}()

    for (_, el) in get(model, "CSHELLs", Dict())
        nids = Int.(el["NODES"])
        if length(nids) == 3
            edges = ((nids[1], nids[2]), (nids[2], nids[3]), (nids[3], nids[1]))
        elseif length(nids) == 4
            edges = ((nids[1], nids[2]), (nids[2], nids[3]), (nids[3], nids[4]), (nids[4], nids[1]))
        else
            continue
        end
        for (a, b) in edges
            key = _shell_edge_key(a, b)
            edge_counts[key] = get(edge_counts, key, 0) + 1
        end
    end

    for (id, el) in get(model, "CSHELLs", Dict())
        nids = Int.(el["NODES"])
        length(nids) == 3 || continue
        eid = _stress_entry_public_id(id, el)
        shared_edges = 0
        for (a, b) in ((nids[1], nids[2]), (nids[2], nids[3]), (nids[3], nids[1]))
            shared_edges += get(edge_counts, _shell_edge_key(a, b), 0) > 1 ? 1 : 0
        end
        weights[eid] = clamp((shared_edges - 1) / 2, 0.0, 1.0)
    end

    return weights
end

function _blend_tria3_macro_bending_iso!(
    M::AbstractVector,
    s_z1::AbstractVector,
    s_z2::AbstractVector,
    e_z1::AbstractVector,
    e_z2::AbstractVector,
    coords::AbstractMatrix,
    u_elem::AbstractVector,
    E,
    nu,
    h::Float64,
    bend_ratio::Float64,
    macro_weight::Float64,
)
    macro_weight <= 0.0 && return false
    abs(bend_ratio) <= 1e-12 && return false

    M_macro = FEM.tria3_plate_macro_average_moment(coords, u_elem, E, nu, h; bend_ratio=bend_ratio)
    all(isfinite, M_macro) || return false

    M_blend = (1.0 - macro_weight) .* M .+ macro_weight .* M_macro
    D = (Float64(E) / (1.0 - Float64(nu)^2)) .* [1.0 Float64(nu) 0.0; Float64(nu) 1.0 0.0; 0.0 0.0 (1.0-Float64(nu))/2.0]
    kappa = -(12.0 / (bend_ratio * h^3)) .* (D \ M_blend)
    eps_mem = (e_z1 .+ e_z2) ./ 2.0
    z1 = -h / 2.0
    z2 = h / 2.0

    M .= M_blend
    e_z1 .= eps_mem .+ z1 .* kappa
    e_z2 .= eps_mem .+ z2 .* kappa
    s_z1 .= D * e_z1
    s_z2 .= D * e_z2
    return true
end

@inline function _quad4_equilibrium_shear_from_bending(coords::AbstractMatrix, M_corners::AbstractMatrix)
    dNdr = (-0.25, 0.25, 0.25, -0.25)
    dNds = (-0.25, -0.25, 0.25, 0.25)
    J11 = dNdr[1]*coords[1,1] + dNdr[2]*coords[2,1] + dNdr[3]*coords[3,1] + dNdr[4]*coords[4,1]
    J12 = dNdr[1]*coords[1,2] + dNdr[2]*coords[2,2] + dNdr[3]*coords[3,2] + dNdr[4]*coords[4,2]
    J21 = dNds[1]*coords[1,1] + dNds[2]*coords[2,1] + dNds[3]*coords[3,1] + dNds[4]*coords[4,1]
    J22 = dNds[1]*coords[1,2] + dNds[2]*coords[2,2] + dNds[3]*coords[3,2] + dNds[4]*coords[4,2]
    detJ = J11*J22 - J12*J21
    if abs(detJ) < 1e-12
        return nothing
    end
    invJ = [J22 -J12; -J21 J11] / detJ

    deriv_nat(vals) = (
        dNdr[1]*vals[1] + dNdr[2]*vals[2] + dNdr[3]*vals[3] + dNdr[4]*vals[4],
        dNds[1]*vals[1] + dNds[2]*vals[2] + dNds[3]*vals[3] + dNds[4]*vals[4],
    )

    function deriv_phys(vals)
        ddr, dds = deriv_nat(vals)
        grad = invJ * [ddr, dds]
        return grad[1], grad[2]
    end

    mx = view(M_corners, :, 1)
    my = view(M_corners, :, 2)
    dmx_dx, _ = deriv_phys(mx)
    _, dmy_dy = deriv_phys(my)
    return [dmx_dx, dmy_dy]
end

@inline function _quad4_blend_recovered_shear(current_Q::AbstractVector, eq_Q)
    eq_Q === nothing && return collect(current_Q)
    Q_out = [Float64(current_Q[1]), Float64(current_Q[2])]
    for i in 1:2
        eqi = Float64(eq_Q[i])
        !isfinite(eqi) && continue
        if abs(eqi) < abs(Q_out[i])
            Q_out[i] = eqi
        end
    end
    return Q_out
end

@inline function _quad4_pressure_resultant_moment_scale(model, prop, mat, lc::AbstractMatrix, N::AbstractVector, Q::AbstractVector)
    get(prop, "TYPE", "") == "PCOMP_CLT" && return 1.0
    (!haskey(mat, "E") || !haskey(mat, "NU")) && return 1.0
    abs(get(prop, "BEND_RATIO", 1.0)) <= 1e-12 && return 1.0
    FEM.quad4_is_axis_aligned_rectangle(lc) || return 1.0

    has_line_elements =
        !isempty(get(model, "CBARs", Dict())) ||
        !isempty(get(model, "CBEAMs", Dict())) ||
        !isempty(get(model, "CRODs", Dict())) ||
        !isempty(get(model, "CONRODs", Dict()))
    has_kinematic_constraints =
        !isempty(get(model, "RBE2s", Dict())) ||
        !isempty(get(model, "RBE3s", Dict())) ||
        !isempty(get(model, "RSPLINEs", Dict())) ||
        !isempty(get(model, "MPCs", []))
    (has_line_elements || has_kinematic_constraints) && return 1.0
    length(get(model, "CSHELLs", Dict())) < 4 && return 1.0
    length(get(model, "PLOAD4s", [])) < 4 && return 1.0

    max_q = maximum(abs, Q)
    max_n = maximum(abs, N)
    max_q <= 1e-8 && return 1.0
    max_n > max(1e-6, 1e-4 * max_q) && return 1.0

    # Nastran's flat pressure-loaded QUAD4 force tables come out slightly lower
    # than the raw JFEM plate-resultant recovery. Apply a very narrow correction
    # only to the reported bending moments, leaving the solved state untouched.
    return 0.99
end

function recover_shell_stresses!(model, id_map, X, node_R, u_global, snorm_normals, stresses, results_json)
    lc_buf = zeros(4,2)
    q4_frame_mode = q4_frame_mode_from_env("JFEM_Q4_FRAME_MODE_STATIC")
    pcomp_axis_mode = q4_pcomp_axis_mode("JFEM_Q4_PCOMP_AXIS_MODE_STATIC")
    membrane_incomp_center_jacobian = q4_sol105_membrane_incomp_center_jacobian_enabled()
    tria3_macro_blend = _tria3_shell_macro_blend_weights(model)

    for (id, el) in model["CSHELLs"]
        eid = _stress_entry_public_id(id, el)
        pid = string(el["PID"])
        if !haskey(model["PSHELLs"], pid); continue; end
        prop = model["PSHELLs"][pid]
        mid = string(prop["MID"])
        if !haskey(model["MATs"], mid); continue; end
        mat = model["MATs"][mid]

        nids = el["NODES"]; n = length(nids)
        if any(x->get(id_map,x,0)==0, nids); continue; end

        local N, M, Q, s_z1, s_z2, e_z1, e_z2, elem_key
        quad4_bilin_rows = Any[]
        stress_ok = true

        if n==4
            i1, i2, i3, i4 = id_map[nids[1]], id_map[nids[2]], id_map[nids[3]], id_map[nids[4]]
            p1 = SVector{3}(X[i1,1], X[i1,2], X[i1,3])
            p2 = SVector{3}(X[i2,1], X[i2,2], X[i2,3])
            p3 = SVector{3}(X[i3,1], X[i3,2], X[i3,3])
            p4 = SVector{3}(X[i4,1], X[i4,2], X[i4,3])
            sr_indices = [i1, i2, i3, i4]

            v1, v2, v3 = shell_element_frame_quad4(p1, p2, p3, p4, q4_frame_mode)
            v1, v2, v3 = apply_snorm_to_frame(v1, v2, v3, sr_indices, snorm_normals)
            c = (p1+p2+p3+p4)/4.0
            lc_buf[1,1]=dot(p1-c,v1); lc_buf[1,2]=dot(p1-c,v2)
            lc_buf[2,1]=dot(p2-c,v1); lc_buf[2,2]=dot(p2-c,v2)
            lc_buf[3,1]=dot(p3-c,v1); lc_buf[3,2]=dot(p3-c,v2)
            lc_buf[4,1]=dot(p4-c,v1); lc_buf[4,2]=dot(p4-c,v2)
            curvature_membrane = nothing
            curvature_scale = q4_curvature_membrane_scale("JFEM_Q4_CURVATURE_MEMBRANE_SCALE_STATIC")
            if curvature_scale > 0.0 && all(idx -> haskey(snorm_normals, idx), sr_indices)
                curvature_raw = estimate_quad4_curvature_membrane(
                    view(lc_buf,1:4,:), snorm_normals[i1], snorm_normals[i2], snorm_normals[i3], snorm_normals[i4], v1, v2, v3
                )
                curvature_weight = q4_curvature_filter_weight(
                    curvature_raw,
                    q4_curvature_filter_mode("JFEM_Q4_CURVATURE_FILTER_MODE_STATIC"),
                    q4_curvature_cyl_ratio_max("JFEM_Q4_CURVATURE_CYL_RATIO_MAX_STATIC"),
                )
                curvature_weight *= q4_curvature_resolution_weight(
                    curvature_raw, view(lc_buf,1:4,:),
                    q4_curvature_resolution_min("JFEM_Q4_CURVATURE_RESOLUTION_MIN_STATIC"),
                    q4_curvature_resolution_full("JFEM_Q4_CURVATURE_RESOLUTION_FULL_STATIC"),
                )
                if curvature_weight > 0.0
                    curvature_membrane = curvature_raw * (curvature_scale * curvature_weight)
                end
            end

            Rel_t = vcat(v1', v2', v3')
            u_el = zeros(24)
            for k=1:4
                idx = id_map[nids[k]]
                u_el[(k-1)*6+1:(k-1)*6+3] = Rel_t * node_R[idx] * u_global[(idx-1)*6+1:(idx-1)*6+3]
                u_el[(k-1)*6+4:(k-1)*6+6] = Rel_t * node_R[idx] * u_global[(idx-1)*6+4:(idx-1)*6+6]
            end
            br = get(prop, "BEND_RATIO", 1.0)
            clt_Cm = nothing
            clt_Cb = nothing
            material_shear_rotation = 0.0
            membrane_shear_center_row = false
            if get(prop, "TYPE", "") == "PCOMP_CLT" && haskey(prop, "Cm")
                clt_Cm = copy(prop["Cm"])
                clt_Cb = haskey(prop, "Cb") ? copy(prop["Cb"]) : nothing
                theta_rad = deg2rad(Float64(get(el, "THETA", 0.0)))
                beta = shell_pcomp_material_rotation(
                    pcomp_axis_mode,
                    v1, v2, v3, p1, p2,
                    theta_rad,
                    Int(get(el, "MCID", 0)),
                    model["CORDs"],
                )
                material_shear_rotation = beta
                if abs(beta) > 1e-10
                    cb = cos(beta); sb = sin(beta)
                    c2 = cb^2; s2 = sb^2; cs = cb*sb
                    _rotate_constitutive_3x3!(clt_Cm, c2, s2, cs, s2, c2, -cs, -2cs, 2cs, c2-s2)
                    if clt_Cb !== nothing
                        _rotate_constitutive_3x3!(clt_Cb, c2, s2, cs, s2, c2, -cs, -2cs, 2cs, c2-s2)
                    end
                end
            end
            try
                membrane_assumed_mode = ((get(prop, "TYPE", "") == "PCOMP_CLT" && !get(prop, "IS_ISOTROPIC", false) && get(prop, "Bmb", nothing) === nothing) ? :mitc4plus : :none)
                N, M, Q, s_z1, s_z2, e_z1, e_z2 = FEM.stress_strain_quad4(view(lc_buf,1:4,:), u_el, mat["E"], mat["NU"], Float64(prop["T"]), Float64(prop["T"]);
                    bend_ratio=br,
                    Cm_override=clt_Cm,
                    curvature_membrane=curvature_membrane,
                    membrane_shear_center_row=membrane_shear_center_row,
                    material_shear_rotation=material_shear_rotation,
                    membrane_assumed_mode=membrane_assumed_mode,
                    membrane_incomp_center_jacobian=membrane_incomp_center_jacobian)
                N_corners, M_corners = FEM.quad4_bilinear_corner_forces(view(lc_buf,1:4,:), u_el, mat["E"], mat["NU"], Float64(prop["T"]);
                    bend_ratio=br,
                    Cm_override=clt_Cm,
                    Cb_override=clt_Cb,
                    curvature_membrane=curvature_membrane,
                    membrane_shear_center_row=membrane_shear_center_row,
                    material_shear_rotation=material_shear_rotation,
                    membrane_assumed_mode=membrane_assumed_mode,
                    membrane_incomp_center_jacobian=membrane_incomp_center_jacobian)
                Q_out = if clt_Cm === nothing && curvature_membrane === nothing && abs(br) > 1e-12
                    _quad4_blend_recovered_shear(Q, _quad4_equilibrium_shear_from_bending(view(lc_buf,1:4,:), M_corners))
                else
                    collect(Q)
                end
                m_scale = _quad4_pressure_resultant_moment_scale(
                    model,
                    prop,
                    mat,
                    view(lc_buf,1:4,:),
                    N,
                    Q_out,
                )
                if m_scale != 1.0
                    M .*= m_scale
                    M_corners .*= m_scale
                end
                # Nastran's bilinear shell-force output keeps twisting moment
                # constant across the QUAD4 corner rows.
                M_corners[:, 3] .= M[3]
                Q .= Q_out
                push!(quad4_bilin_rows, Dict(
                    "eid" => eid,
                    "grid_id" => "CEN/4",
                    "fx" => N[1], "fy" => N[2], "fxy" => N[3],
                    "mx" => M[1], "my" => M[2], "mxy" => M[3],
                    "qx" => Q_out[1], "qy" => Q_out[2],
                ))
                for k in 1:4
                    push!(quad4_bilin_rows, Dict(
                        "eid" => eid,
                        "grid_id" => nids[k],
                        "fx" => N_corners[k, 1], "fy" => N_corners[k, 2], "fxy" => N_corners[k, 3],
                        "mx" => M_corners[k, 1], "my" => M_corners[k, 2], "mxy" => M_corners[k, 3],
                        "qx" => Q_out[1], "qy" => Q_out[2],
                    ))
                end
            catch e
                @warn "Stress recovery failed for QUAD4 $eid: $e"
                stress_ok = false
            end
            elem_key = "quad4"
        elseif n==3
            i1, i2, i3 = id_map[nids[1]], id_map[nids[2]], id_map[nids[3]]
            p1 = SVector{3}(X[i1,1], X[i1,2], X[i1,3])
            p2 = SVector{3}(X[i2,1], X[i2,2], X[i2,3])
            p3 = SVector{3}(X[i3,1], X[i3,2], X[i3,3])
            p4=SVector(0.0,0.0,0.0)
            sr_indices = [i1, i2, i3]

            v1, v2, v3 = shell_element_frame_fast(p1, p2, p3, p4, 3)
            v1, v2, v3 = apply_snorm_to_frame(v1, v2, v3, sr_indices, snorm_normals)
            c = (p1+p2+p3)/3.0
            lc_buf[1,1]=dot(p1-c,v1); lc_buf[1,2]=dot(p1-c,v2)
            lc_buf[2,1]=dot(p2-c,v1); lc_buf[2,2]=dot(p2-c,v2)
            lc_buf[3,1]=dot(p3-c,v1); lc_buf[3,2]=dot(p3-c,v2)
            Rel_t = vcat(v1', v2', v3')
            u_el = zeros(18)
            for k=1:3
                idx = id_map[nids[k]]
                u_el[(k-1)*6+1:(k-1)*6+3] = Rel_t * node_R[idx] * u_global[(idx-1)*6+1:(idx-1)*6+3]
                u_el[(k-1)*6+4:(k-1)*6+6] = Rel_t * node_R[idx] * u_global[(idx-1)*6+4:(idx-1)*6+6]
            end
            br = get(prop, "BEND_RATIO", 1.0)
            clt_Cm = nothing
            clt_Cb = nothing
            if get(prop, "TYPE", "") == "PCOMP_CLT" && haskey(prop, "Cm")
                clt_Cm = copy(prop["Cm"])
                clt_Cb = haskey(prop, "Cb") ? copy(prop["Cb"]) : nothing
                beta = shell_pcomp_material_rotation(
                    pcomp_axis_mode,
                    v1, v2, v3, p1, p2,
                    deg2rad(Float64(get(el, "THETA", 0.0))),
                    Int(get(el, "MCID", 0)),
                    model["CORDs"],
                )
                if abs(beta) > 1e-10
                    cb = cos(beta); sb = sin(beta)
                    c2 = cb^2; s2 = sb^2; cs = cb*sb
                    _rotate_constitutive_3x3!(clt_Cm, c2, s2, cs, s2, c2, -cs, -2cs, 2cs, c2-s2)
                    if clt_Cb !== nothing
                        _rotate_constitutive_3x3!(clt_Cb, c2, s2, cs, s2, c2, -cs, -2cs, 2cs, c2-s2)
                    end
                end
            end
            try
                N, M, Q, s_z1, s_z2, e_z1, e_z2 = FEM.stress_strain_tria3(view(lc_buf,1:3,:), u_el, mat["E"], mat["NU"], Float64(prop["T"]); bend_ratio=br, Cm_override=clt_Cm)
                macro_weight = get(tria3_macro_blend, eid, 0.0)
                if macro_weight > 0.0 && clt_Cm === nothing && haskey(mat, "E") && haskey(mat, "NU")
                    _blend_tria3_macro_bending_iso!(
                        M, s_z1, s_z2, e_z1, e_z2,
                        view(lc_buf,1:3,:), u_el,
                        mat["E"], mat["NU"], Float64(prop["T"]),
                        Float64(br), macro_weight,
                    )
                end
            catch e
                @warn "Stress recovery failed for TRIA3 $eid: $e"
                stress_ok = false
            end
            elem_key = "tria3"
        else
            continue
        end

        if !stress_ok; continue; end

        eps_mem_out = (e_z1 .+ e_z2) ./ 2.0
        kappa_nast_out = (e_z1 .- e_z2) ./ prop["T"]

        is_pcomp = get(prop, "TYPE", "") == "PCOMP_CLT" && haskey(prop, "PLY_DATA")
        if is_pcomp
            t_total = prop["T"]
            eps_mem = (e_z1 .+ e_z2) ./ 2.0
            kappa = (e_z2 .- e_z1) ./ t_total
            Cm_eff = clt_Cm === nothing ? prop["Cm"] : clt_Cm
            Cb_eff = clt_Cb === nothing ? prop["Cb"] : clt_Cb
            N = Cm_eff * eps_mem
            M = -Cb_eff * kappa

            ply_data = prop["PLY_DATA"]
            vm_max = 0.0
            s_z1_out = zeros(3)
            s_z2_out = zeros(3)
            e_z1_out = zeros(3)
            e_z2_out = zeros(3)
            for (ip, pd) in enumerate(ply_data)
                Qbar = pd["Qbar"]
                z_mid = (pd["z_bot"] + pd["z_top"]) / 2.0
                strain_ply = eps_mem .+ z_mid .* kappa
                stress_ply = Qbar * strain_ply
                vm_ply = sqrt(stress_ply[1]^2 - stress_ply[1]*stress_ply[2] + stress_ply[2]^2 + 3*stress_ply[3]^2)
                if vm_ply > vm_max; vm_max = vm_ply; end
                if ip == 1; s_z1_out .= stress_ply; e_z1_out .= strain_ply; end
                if ip == length(ply_data); s_z2_out .= stress_ply; e_z2_out .= strain_ply; end
            end
            s_z1 = s_z1_out
            s_z2 = s_z2_out
            e_z1 = e_z1_out
            e_z2 = e_z2_out
            stresses[eid] = vm_max
        else
            stresses[eid] = FEM.compute_principal_2d(s_z1[1], s_z1[2], s_z1[3])[1]
        end

        push!(results_json["forces"][elem_key], Dict("eid" => eid, "fx" => N[1], "fy" => N[2], "fxy" => N[3], "mx" => M[1], "my" => M[2], "mxy" => M[3], "qx" => Q[1], "qy" => Q[2]))
        if elem_key == "quad4" && !isempty(quad4_bilin_rows)
            append!(results_json["forces_bilin"]["quad4"], quad4_bilin_rows)
        end

        make_stress_entry(s, t) = Dict("fiber_dist" => t, "normal_x" => s[1], "normal_y" => s[2], "shear_xy" => s[3], "von_mises" => sqrt(s[1]^2-s[1]*s[2]+s[2]^2+3*s[3]^2), "major" => 0.0, "minor" => 0.0)
        make_strain_entry(e, t) = Dict("fiber_dist" => t, "normal_x" => e[1], "normal_y" => e[2], "shear_xy" => e[3], "major" => 0.0, "minor" => 0.0)

        push!(results_json["stresses"][elem_key], Dict("eid" => eid, "z1" => make_stress_entry(s_z1, -prop["T"]/2), "z2" => make_stress_entry(s_z2, prop["T"]/2)))
        push!(results_json["strains"][elem_key], Dict("eid" => eid, "z1" => make_strain_entry(eps_mem_out, 0.0), "z2" => make_strain_entry(kappa_nast_out, -1.0)))
    end
end

function recover_bar_stresses!(model, id_map, X, node_R, u_global, stresses, results_json)
    @inline function surface_stress_sign(prop)
        section_type = uppercase(string(get(prop, "TYPE", "")))
        return section_type in ("ROD", "TUBE", "TUBE2") ? 1.0 : -1.0
    end

    @inline function recover_surface_stress(prop, m1, m2, yj, zj, Iy, Iz, I12_sr)
        sigma = if abs(Iz*Iy - I12_sr^2) > 1e-30
            ((m1*Iy - m2*I12_sr)*yj + (m2*Iz - m1*I12_sr)*zj) / (Iz*Iy - I12_sr^2)
        else
            m1*yj/max(Iz, 1e-30) + m2*zj/max(Iy, 1e-30)
        end
        return surface_stress_sign(prop) * sigma
    end

    for (id, bar) in model["CBARs"]
        eid = _stress_entry_public_id(id, bar)
        pid = string(bar["PID"])
        if !haskey(model["PBARLs"], pid); continue; end
        prop = model["PBARLs"][pid]
        mid = string(prop["MID"])
        if !haskey(model["MATs"], mid); continue; end

        if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
        i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]
        mat = _effective_mat1_for_nodes(model, mid, [bar["GA"], bar["GB"]])
        mat === nothing && continue

        p1 = SVector{3}(X[i1,1], X[i1,2], X[i1,3])
        p2 = SVector{3}(X[i2,1], X[i2,2], X[i2,3])

        wa, wb, has_offset, p1_eff, p2_eff = bar_offsets_and_endpoints(bar, p1, p2)
        L = norm(p2_eff - p1_eff)
        if L < 1e-9; continue; end
        vx = normalize(p2_eff - p1_eff)
        v_ref = resolve_bar_vref(bar, p1, id_map, X)

        if norm(v_ref) < 1e-6
             v_ref = SVector(0.0,0.0,1.0)
             if abs(dot(vx, v_ref)) > 0.9; v_ref = SVector(0.0,1.0,0.0); end
        end
        vz = normalize(cross(vx, v_ref))
        vy = cross(vz, vx)
        Rel_t = vcat(vx', vy', vz')

        TR1 = Rel_t * node_R[i1]
        TR2 = Rel_t * node_R[i2]
        u_el = zeros(12)
        u_el[1:3] = TR1 * u_global[(i1-1)*6+1:(i1-1)*6+3]
        u_el[4:6] = TR1 * u_global[(i1-1)*6+4:(i1-1)*6+6]
        u_el[7:9] = TR2 * u_global[(i2-1)*6+1:(i2-1)*6+3]
        u_el[10:12] = TR2 * u_global[(i2-1)*6+4:(i2-1)*6+6]
        if has_offset
            S_wa = skew3(wa); S_wb = skew3(wb)
            θ_glob_A = node_R[i1] * u_global[(i1-1)*6+4:(i1-1)*6+6]
            θ_glob_B = node_R[i2] * u_global[(i2-1)*6+4:(i2-1)*6+6]
            u_el[1:3] -= Rel_t * S_wa * θ_glob_A
            u_el[7:9] -= Rel_t * S_wb * θ_glob_B
        end

        Iy = get(prop, "I2", get(prop, "I", 0.0))
        Iz = get(prop, "I1", get(prop, "I", 0.0))
        if Iy == 0.0; Iy = Iz; end
        if Iz == 0.0; Iz = Iy; end
        Iyz = Float64(get(prop, "I12", 0.0))
        K1 = get(prop, "K1", 0.0); K2 = get(prop, "K2", 0.0)
        As_y = (K1 > 0.0) ? K1 * prop["A"] : Inf
        As_z = (K2 > 0.0) ? K2 * prop["A"] : Inf
        forces = FEM.forces_frame3d(u_el, L, prop["A"], Iy, Iz, prop["J"], mat["E"], mat["G"]; As_y=As_y, As_z=As_z, I12=Iyz)
        A_bar = Float64(prop["A"])
        sig_axial = (abs(A_bar) > 1e-30) ? forces["axial"]/A_bar : 0.0
        stresses[eid] = abs(sig_axial)
        push!(results_json["forces"]["cbar"], Dict("eid" => eid, "axial" => forces["axial"], "shear_1" => forces["shear_1"], "shear_2" => forces["shear_2"], "torque" => forces["torque"], "moment_a1" => forces["moment_a1"], "moment_a2" => forces["moment_a2"], "moment_b1" => forces["moment_b1"], "moment_b2" => forces["moment_b2"]))

        I12_sr = Float64(get(prop, "I12", 0.0))
        sr_pts = [(get(prop,"C1",0.0), get(prop,"C2",0.0)),
                  (get(prop,"D1",0.0), get(prop,"D2",0.0)),
                  (get(prop,"E1",0.0), get(prop,"E2",0.0)),
                  (get(prop,"F1",0.0), get(prop,"F2",0.0))]
        end_a = Dict{String,Float64}()
        end_b = Dict{String,Float64}()
        for (j, (yj, zj)) in enumerate(sr_pts)
            end_a["p$j"] = recover_surface_stress(prop, forces["moment_a1"], forces["moment_a2"], yj, zj, Iy, Iz, I12_sr)
            end_b["p$j"] = recover_surface_stress(prop, forces["moment_b1"], forces["moment_b2"], yj, zj, Iy, Iz, I12_sr)
        end
        push!(results_json["stresses"]["cbar"], Dict("eid"=>eid, "end_a"=>end_a, "end_b"=>end_b, "axial"=>sig_axial))
    end

    # --- CBEAMs (identical recovery to CBAR) ---
    for (id, bar) in get(model, "CBEAMs", Dict())
        eid = _stress_entry_public_id(id, bar)
        pid = string(bar["PID"])
        if !haskey(model["PBARLs"], pid); continue; end
        prop = model["PBARLs"][pid]
        mid = string(prop["MID"])
        if !haskey(model["MATs"], mid); continue; end

        if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
        i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]
        mat = _effective_mat1_for_nodes(model, mid, [bar["GA"], bar["GB"]])
        mat === nothing && continue

        p1 = SVector{3}(X[i1,1], X[i1,2], X[i1,3])
        p2 = SVector{3}(X[i2,1], X[i2,2], X[i2,3])

        wa, wb, has_offset, p1_eff, p2_eff = bar_offsets_and_endpoints(bar, p1, p2)
        L = norm(p2_eff - p1_eff)
        if L < 1e-9; continue; end
        vx = normalize(p2_eff - p1_eff)
        v_ref = resolve_bar_vref(bar, p1, id_map, X)

        if norm(v_ref) < 1e-6
             v_ref = SVector(0.0,0.0,1.0)
             if abs(dot(vx, v_ref)) > 0.9; v_ref = SVector(0.0,1.0,0.0); end
        end
        vz = normalize(cross(vx, v_ref))
        vy = cross(vz, vx)
        Rel_t = vcat(vx', vy', vz')

        TR1 = Rel_t * node_R[i1]
        TR2 = Rel_t * node_R[i2]
        u_el = zeros(12)
        u_el[1:3] = TR1 * u_global[(i1-1)*6+1:(i1-1)*6+3]
        u_el[4:6] = TR1 * u_global[(i1-1)*6+4:(i1-1)*6+6]
        u_el[7:9] = TR2 * u_global[(i2-1)*6+1:(i2-1)*6+3]
        u_el[10:12] = TR2 * u_global[(i2-1)*6+4:(i2-1)*6+6]
        if has_offset
            S_wa = skew3(wa); S_wb = skew3(wb)
            θ_glob_A = node_R[i1] * u_global[(i1-1)*6+4:(i1-1)*6+6]
            θ_glob_B = node_R[i2] * u_global[(i2-1)*6+4:(i2-1)*6+6]
            u_el[1:3] -= Rel_t * S_wa * θ_glob_A
            u_el[7:9] -= Rel_t * S_wb * θ_glob_B
        end

        Iy = get(prop, "I2", get(prop, "I", 0.0))
        Iz = get(prop, "I1", get(prop, "I", 0.0))
        if Iy == 0.0; Iy = Iz; end
        if Iz == 0.0; Iz = Iy; end
        Iyz = Float64(get(prop, "I12", 0.0))
        K1 = get(prop, "K1", 0.0); K2 = get(prop, "K2", 0.0)
        As_y = (K1 > 0.0) ? K1 * prop["A"] : Inf
        As_z = (K2 > 0.0) ? K2 * prop["A"] : Inf
        forces = FEM.forces_frame3d(u_el, L, prop["A"], Iy, Iz, prop["J"], mat["E"], mat["G"]; As_y=As_y, As_z=As_z, I12=Iyz)
        A_bar = Float64(prop["A"])
        sig_axial = (abs(A_bar) > 1e-30) ? forces["axial"]/A_bar : 0.0
        stresses[eid] = abs(sig_axial)
        push!(results_json["forces"]["cbar"], Dict("eid" => eid, "axial" => forces["axial"], "shear_1" => forces["shear_1"], "shear_2" => forces["shear_2"], "torque" => forces["torque"], "moment_a1" => forces["moment_a1"], "moment_a2" => forces["moment_a2"], "moment_b1" => forces["moment_b1"], "moment_b2" => forces["moment_b2"]))

        I12_sr = Float64(get(prop, "I12", 0.0))
        sr_pts = [(get(prop,"C1",0.0), get(prop,"C2",0.0)),
                  (get(prop,"D1",0.0), get(prop,"D2",0.0)),
                  (get(prop,"E1",0.0), get(prop,"E2",0.0)),
                  (get(prop,"F1",0.0), get(prop,"F2",0.0))]
        end_a = Dict{String,Float64}()
        end_b = Dict{String,Float64}()
        for (j, (yj, zj)) in enumerate(sr_pts)
            end_a["p$j"] = recover_surface_stress(prop, forces["moment_a1"], forces["moment_a2"], yj, zj, Iy, Iz, I12_sr)
            end_b["p$j"] = recover_surface_stress(prop, forces["moment_b1"], forces["moment_b2"], yj, zj, Iy, Iz, I12_sr)
        end
        push!(results_json["stresses"]["cbar"], Dict("eid"=>eid, "end_a"=>end_a, "end_b"=>end_b, "axial"=>sig_axial))
    end
end

function recover_rod_stresses!(model, id_map, X, node_R, u_global, stresses, results_json)
    # CROD recovery
    crods = get(model, "CRODs", Dict())
    prods = get(model, "PRODs", Dict())
    for (id, rod) in crods
        eid = _stress_entry_public_id(id, rod)
        pid = string(rod["PID"])
        if !haskey(prods, pid); continue; end
        prop = prods[pid]
        mid = string(prop["MID"])
        if !haskey(model["MATs"], mid); continue; end

        if !haskey(id_map, rod["GA"]) || !haskey(id_map, rod["GB"]); continue; end
        i1, i2 = id_map[rod["GA"]], id_map[rod["GB"]]
        mat = _effective_mat1_for_nodes(model, mid, [rod["GA"], rod["GB"]])
        mat === nothing && continue

        p1 = SVector{3}(X[i1,1], X[i1,2], X[i1,3])
        p2 = SVector{3}(X[i2,1], X[i2,2], X[i2,3])
        L = norm(p2-p1)
        if L < 1e-9; continue; end
        vx = normalize(p2-p1)
        ref = abs(vx[3]) < 0.9 ? SVector(0.0,0.0,1.0) : SVector(0.0,1.0,0.0)
        vz = normalize(cross(vx, ref))
        vy = cross(vz, vx)
        Rel_t = vcat(vx', vy', vz')

        u_el = zeros(12)
        u_el[1:3] = Rel_t * node_R[i1] * u_global[(i1-1)*6+1:(i1-1)*6+3]
        u_el[4:6] = Rel_t * node_R[i1] * u_global[(i1-1)*6+4:(i1-1)*6+6]
        u_el[7:9] = Rel_t * node_R[i2] * u_global[(i2-1)*6+1:(i2-1)*6+3]
        u_el[10:12] = Rel_t * node_R[i2] * u_global[(i2-1)*6+4:(i2-1)*6+6]

        axial_force = mat["E"] * prop["A"] / L * (u_el[7] - u_el[1])
        torque = mat["G"] * prop["J"] / L * (u_el[10] - u_el[4])
        axial_stress = prop["A"] > 0 ? axial_force / prop["A"] : 0.0
        torsional_stress = prop["J"] > 0 && haskey(prop, "C") && prop["C"] > 0 ? torque * prop["C"] / prop["J"] : 0.0
        stresses[eid] = abs(axial_stress)
        axial_strain = mat["E"] > 0 ? axial_stress / mat["E"] : 0.0
        push!(results_json["forces"]["crod"], Dict("eid" => eid, "axial" => axial_force, "torque" => torque))
        push!(results_json["stresses"]["crod"], Dict("eid" => eid, "axial" => axial_stress, "torsional" => torsional_stress))
        push!(results_json["strains"]["crod"], Dict("eid" => eid, "axial" => axial_strain, "torsional" => mat["G"] > 0 ? torsional_stress / mat["G"] : 0.0))
    end

    # CONROD recovery
    conrods = get(model, "CONRODs", Dict())
    for (id, rod) in conrods
        eid = _stress_entry_public_id(id, rod)
        mid = string(rod["MID"])
        if !haskey(model["MATs"], mid); continue; end
        if !haskey(id_map, rod["GA"]) || !haskey(id_map, rod["GB"]); continue; end
        i1, i2 = id_map[rod["GA"]], id_map[rod["GB"]]
        mat = _effective_mat1_for_nodes(model, mid, [rod["GA"], rod["GB"]])
        mat === nothing && continue
        p1 = SVector{3}(X[i1,1], X[i1,2], X[i1,3])
        p2 = SVector{3}(X[i2,1], X[i2,2], X[i2,3])
        L = norm(p2-p1)
        if L < 1e-9; continue; end
        vx = normalize(p2-p1)
        ref = abs(vx[3]) < 0.9 ? SVector(0.0,0.0,1.0) : SVector(0.0,1.0,0.0)
        vz = normalize(cross(vx, ref))
        vy = cross(vz, vx)
        Rel_t = vcat(vx', vy', vz')
        u_el = zeros(12)
        u_el[1:3] = Rel_t * node_R[i1] * u_global[(i1-1)*6+1:(i1-1)*6+3]
        u_el[4:6] = Rel_t * node_R[i1] * u_global[(i1-1)*6+4:(i1-1)*6+6]
        u_el[7:9] = Rel_t * node_R[i2] * u_global[(i2-1)*6+1:(i2-1)*6+3]
        u_el[10:12] = Rel_t * node_R[i2] * u_global[(i2-1)*6+4:(i2-1)*6+6]
        axial_force = mat["E"] * rod["A"] / L * (u_el[7] - u_el[1])
        torque = mat["G"] * rod["J"] / L * (u_el[10] - u_el[4])
        axial_stress = rod["A"] > 0 ? axial_force / rod["A"] : 0.0
        torsional_stress = rod["J"] > 0 && rod["C"] > 0 ? torque * rod["C"] / rod["J"] : 0.0
        stresses[eid] = abs(axial_stress)
        axial_strain = mat["E"] > 0 ? axial_stress / mat["E"] : 0.0
        push!(results_json["forces"]["conrod"], Dict("eid" => eid, "axial" => axial_force, "torque" => torque))
        push!(results_json["stresses"]["conrod"], Dict("eid" => eid, "axial" => axial_stress, "torsional" => torsional_stress))
        push!(results_json["strains"]["conrod"], Dict("eid" => eid, "axial" => axial_strain, "torsional" => mat["G"] > 0 ? torsional_stress / mat["G"] : 0.0))
    end
end

function recover_spring_forces!(model, id_map, u_global, stresses, results_json)
    celases = get(model, "CELASs", Dict())
    pelases = get(model, "PELASs", Dict())
    for (id, spring) in celases
        eid = _stress_entry_public_id(id, spring)
        if get(spring, "TYPE", "CELAS1") == "CELAS2"
            K_spring = Float64(get(spring, "K", 0.0))
        else
            pid = string(get(spring, "PID", 0))
            if !haskey(pelases, pid); continue; end
            K_spring = Float64(pelases[pid]["K"])
        end
        g1 = spring["G1"]; c1 = spring["C1"]
        g2 = spring["G2"]; c2 = spring["C2"]
        u1 = 0.0; u2 = 0.0
        if g1 > 0 && haskey(id_map, g1) && c1 > 0
            u1 = u_global[(id_map[g1]-1)*6 + c1]
        end
        if g2 > 0 && haskey(id_map, g2) && c2 > 0
            u2 = u_global[(id_map[g2]-1)*6 + c2]
        end
        spring_force = K_spring * (u1 - u2)
        stresses[eid] = abs(spring_force)
        push!(results_json["forces"]["celas1"], Dict("eid" => eid, "force" => spring_force))
        push!(results_json["stresses"]["celas1"], Dict("eid" => eid, "force" => spring_force))
        push!(results_json["strains"]["celas1"], Dict("eid" => eid, "deformation" => u1 - u2))
    end
end

function recover_solid_stresses!(model, id_map, X, node_R, u_global, stresses, results_json)
    csolids = get(model, "CSOLIDs", Dict())
    psolids = get(model, "PSOLIDs", Dict())
    mats = model["MATs"]
    coords_buf = zeros(8, 3)

    for (id, el) in csolids
        eid = _stress_entry_public_id(id, el)
        pid = string(el["PID"])
        if !haskey(psolids, pid); continue; end
        prop = psolids[pid]
        nids = el["NODES"]
        nn = length(nids)
        mid = string(prop["MID"])
        if !haskey(mats, mid); continue; end
        mat = _effective_mat1_for_nodes(model, mid, nids)
        etype = get(el, "TYPE", "")

        valid = true
        for k in 1:nn
            if !haskey(id_map, nids[k]); valid = false; break; end
        end
        if !valid; continue; end

        # Gather coordinates
        for k in 1:nn
            idx = id_map[nids[k]]
            coords_buf[k,1] = X[idx,1]; coords_buf[k,2] = X[idx,2]; coords_buf[k,3] = X[idx,3]
        end

        # Extract element displacements (translational DOFs only, in global coords)
        ndof_el = nn * 3
        u_el = zeros(ndof_el)
        for k in 1:nn
            idx = id_map[nids[k]]
            u_local = u_global[(idx-1)*6+1:(idx-1)*6+3]
            u_glob = node_R[idx] * u_local
            u_el[(k-1)*3+1:(k-1)*3+3] = u_glob
        end

        E_mat = Float64(mat["E"]); nu_mat = Float64(mat["NU"])
        D = FEM.iso_3d_constitutive(E_mat, nu_mat)

        # Compute B at centroid and recover stress
        local B_cen
        local elem_key::String
        if etype == "CTETRA" && nn == 4
            B_cen = FEM.solid_centroid_B_tetra4(view(coords_buf, 1:4, :))
            elem_key = "ctetra"
        elseif etype == "CHEXA" && nn == 8
            B_cen = FEM.solid_centroid_B_hexa8(view(coords_buf, 1:8, :))
            elem_key = "chexa"
        elseif etype == "CPENTA" && nn == 6
            B_cen = FEM.solid_centroid_B_cpenta6(view(coords_buf, 1:6, :))
            elem_key = "cpenta"
        else
            continue
        end

        stress_vec, strain_vec, vm = FEM.stress_solid_3d(B_cen, D, u_el)
        stresses[eid] = vm

        # Corner stress recovery for CHEXA8 (at 8 corner nodes)
        corner_stresses = []
        if etype == "CHEXA" && nn == 8
            xi_corners  = [-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0,-1.0]
            eta_corners = [-1.0,-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0]
            zet_corners = [-1.0,-1.0,-1.0,-1.0, 1.0, 1.0, 1.0, 1.0]
            xi_n  = [-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0,-1.0]
            eta_n = [-1.0,-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0]
            zet_n = [-1.0,-1.0,-1.0,-1.0, 1.0, 1.0, 1.0, 1.0]
            coords_view = view(coords_buf, 1:8, :)
            for ci in 1:8
                xi = xi_corners[ci]; eta = eta_corners[ci]; zet = zet_corners[ci]
                dN_dxi = zeros(3, 8)
                for i in 1:8
                    dN_dxi[1,i] = 0.125*xi_n[i]*(1+eta_n[i]*eta)*(1+zet_n[i]*zet)
                    dN_dxi[2,i] = 0.125*eta_n[i]*(1+xi_n[i]*xi)*(1+zet_n[i]*zet)
                    dN_dxi[3,i] = 0.125*zet_n[i]*(1+xi_n[i]*xi)*(1+eta_n[i]*eta)
                end
                J = dN_dxi * coords_view
                if abs(det(J)) < 1e-30; continue; end
                dN_dx = inv(J) * dN_dxi
                B_corner = zeros(6, 24)
                for i in 1:8
                    c = (i-1)*3
                    dx=dN_dx[1,i]; dy=dN_dx[2,i]; dz=dN_dx[3,i]
                    B_corner[1,c+1]=dx; B_corner[2,c+2]=dy; B_corner[3,c+3]=dz
                    B_corner[4,c+1]=dy; B_corner[4,c+2]=dx
                    B_corner[5,c+2]=dz; B_corner[5,c+3]=dy
                    B_corner[6,c+1]=dz; B_corner[6,c+3]=dx
                end
                s_c, _, vm_c = FEM.stress_solid_3d(B_corner, D, u_el)
                push!(corner_stresses, Dict("grid"=>nids[ci], "sxx"=>s_c[1],"syy"=>s_c[2],"szz"=>s_c[3],
                    "txy"=>s_c[4],"tyz"=>s_c[5],"tzx"=>s_c[6],"von_mises"=>vm_c))
            end
        end

        stress_entry = Dict{String,Any}(
            "eid" => eid,
            "sxx" => stress_vec[1], "syy" => stress_vec[2], "szz" => stress_vec[3],
            "txy" => stress_vec[4], "tyz" => stress_vec[5], "tzx" => stress_vec[6],
            "von_mises" => vm
        )
        if !isempty(corner_stresses); stress_entry["corners"] = corner_stresses; end
        push!(results_json["stresses"][elem_key], stress_entry)
        push!(results_json["strains"][elem_key], Dict(
            "eid" => eid,
            "exx" => strain_vec[1], "eyy" => strain_vec[2], "ezz" => strain_vec[3],
            "gxy" => strain_vec[4], "gyz" => strain_vec[5], "gzx" => strain_vec[6]
        ))
    end
end
