# constraints.jl — RBE2, RBE3, MPC constraint assembly and DOF elimination

# Extracted from assemble_stiffness: processes all constraint elements and
# redistributes stiffness triplets for dependent DOFs.
# Returns: (rbe3_map, I_idx, J_idx, V_val)  — rbe3_map is the merged constraint map,
# and triplet arrays may be replaced if constraints exist.
@inline function _rigid_offset_matrix(dx::Float64, dy::Float64, dz::Float64)
    return [
        0.0   dz   -dy;
       -dz   0.0   dx;
        dy   -dx   0.0;
    ]
end

@inline function _rigid_component_row(node_R::AbstractMatrix, q_R::AbstractMatrix,
                                      dx::Float64, dy::Float64, dz::Float64,
                                      comp_dof::Int)
    comp_map = node_R' * q_R
    if comp_dof <= 3
        offset_map = node_R' * _rigid_offset_matrix(dx, dy, dz) * q_R
        return vcat(vec(comp_map[comp_dof, :]), vec(offset_map[comp_dof, :]))
    end
    return vcat(zeros(3), vec(comp_map[comp_dof - 3, :]))
end

@inline function _push_rigid_pairs!(pairs::Vector{Tuple{Int,Float64}}, base_dof::Int,
                                    coeff_row::AbstractVector{<:Real})
    for j in 1:length(coeff_row)
        coeff = Float64(coeff_row[j])
        if abs(coeff) > 1e-15
            push!(pairs, (base_dof + j, coeff))
        end
    end
    return pairs
end

function assemble_constraints(model, id_map, node_coords, node_R, I_idx, J_idx, V_val)
    rbe2s = get(model, "RBE2s", Dict())

    # --- RBE2 (rigid body element - MPC constraint via DOF elimination) ---
    rbe2_map = Dict{Int, Vector{Tuple{Int, Float64}}}()
    n_rbe2 = 0
    n_rbe2_dep = 0
    for (id, rbe) in rbe2s
        gn = rbe["GN"]   # master node
        if !haskey(id_map, gn); continue; end
        i_master = id_map[gn]
        cm_digits = [parse(Int, string(ch)) for ch in string(rbe["CM"]) if isdigit(ch)]
        p_m = SVector{3}(node_coords[i_master,1], node_coords[i_master,2], node_coords[i_master,3])
        R_master = node_R[i_master]

        for gs in rbe["GM"]  # slave nodes
            if !haskey(id_map, gs); continue; end
            i_slave = id_map[gs]
            p_s = SVector{3}(node_coords[i_slave,1], node_coords[i_slave,2], node_coords[i_slave,3])
            dx, dy, dz = p_s[1]-p_m[1], p_s[2]-p_m[2], p_s[3]-p_m[3]
            R_slave = node_R[i_slave]

            for c in cm_digits
                slave_dof = (i_slave-1)*6 + c
                pairs = Tuple{Int,Float64}[]
                coeff_row = _rigid_component_row(R_slave, R_master, dx, dy, dz, c)
                _push_rigid_pairs!(pairs, (i_master-1)*6, coeff_row)
                if !isempty(pairs)
                    rbe2_map[slave_dof] = pairs
                    n_rbe2_dep += 1
                end
            end
            n_rbe2 += 1
        end
    end
    if n_rbe2 > 0
        log_msg("[SOLVER] RBE2: $(length(rbe2s)) elements, $n_rbe2 master-slave pairs, $n_rbe2_dep dependent DOFs (MPC elimination)")
    end

    # --- RBE1 (general rigid element - distributed independent DOFs) ---
    rbe1s = get(model, "RBE1s", Dict())
    rbe1_map = Dict{Int, Vector{Tuple{Int, Float64}}}()
    n_rbe1 = 0
    global_R = Matrix(1.0I, 3, 3)
    for (id, rbe1) in rbe1s
        indep = rbe1["INDEP"]   # [(grid, dof_digit), ...]
        dep = rbe1["DEP"]       # [(grid, dof_digit), ...]
        n_indep = length(indep)
        if n_indep < 1 || n_indep > 6; continue; end

        # Collect independent DOFs and build rigid body transformation matrix A
        # A × q = u_indep, where q = [ux_ref, uy_ref, uz_ref, wx, wy, wz] in global axes.
        # Use centroid of independent grids as reference point
        indep_grids_unique = unique([g for (g,d) in indep])
        x_ref = zeros(3)
        n_valid = 0
        for g in indep_grids_unique
            if !haskey(id_map, g); continue; end
            gi = id_map[g]
            x_ref .+= [node_coords[gi,1], node_coords[gi,2], node_coords[gi,3]]
            n_valid += 1
        end
        if n_valid == 0; continue; end
        x_ref ./= n_valid

        # Build A matrix (n_indep × 6) and collect global DOF indices
        A_mat = zeros(n_indep, 6)
        indep_global = Int[]
        valid = true
        for (k, (g, dof)) in enumerate(indep)
            if !haskey(id_map, g); valid = false; break; end
            gi = id_map[g]
            dx = node_coords[gi,1] - x_ref[1]
            dy = node_coords[gi,2] - x_ref[2]
            dz = node_coords[gi,3] - x_ref[3]
            push!(indep_global, (gi-1)*6 + dof)
            A_mat[k,:] = _rigid_component_row(node_R[gi], global_R, dx, dy, dz, dof)
        end
        if !valid; continue; end

        # q = A_inv × u_indep (use pseudoinverse for robustness)
        A_inv = pinv(A_mat, rtol=1e-10)  # 6 × n_indep

        # For each dependent DOF: u_dep = b' × q = b' × A_inv × u_indep
        for (g, dof) in dep
            if !haskey(id_map, g); continue; end
            gi = id_map[g]
            dx = node_coords[gi,1] - x_ref[1]
            dy = node_coords[gi,2] - x_ref[2]
            dz = node_coords[gi,3] - x_ref[3]
            gdof = (gi-1)*6 + dof
            b = _rigid_component_row(node_R[gi], global_R, dx, dy, dz, dof)
            c_vec = A_inv' * b  # n_indep coefficients
            pairs = Tuple{Int,Float64}[]
            for (j, coeff) in enumerate(c_vec)
                if abs(coeff) > 1e-15
                    push!(pairs, (indep_global[j], coeff))
                end
            end
            if !isempty(pairs)
                rbe1_map[gdof] = pairs
            end
        end
        n_rbe1 += 1
    end
    if n_rbe1 > 0
        log_msg("[SOLVER] RBE1: $n_rbe1 elements, $(length(rbe1_map)) dependent DOFs")
    end

    # --- RSPLINE (spline interpolation constraint via linear interpolation) ---
    rsplines = get(model, "RSPLINEs", Dict())
    rspline_map = Dict{Int, Vector{Tuple{Int, Float64}}}()
    n_rspline = 0
    for (id, rsp) in rsplines
        indep_grids = rsp["INDEP_GRIDS"]
        dep_list = rsp["DEP"]  # [(grid, dof_digit), ...]

        # Get coordinates of independent grids (spline control points)
        indep_coords = Tuple{Int,Float64,Float64,Float64}[]
        for g in indep_grids
            if !haskey(id_map, g); continue; end
            gi = id_map[g]
            push!(indep_coords, (gi, node_coords[gi,1], node_coords[gi,2], node_coords[gi,3]))
        end
        if length(indep_coords) < 2; continue; end

        # Compute cumulative arc length along the spline control points
        arc_len = Float64[0.0]
        for i in 2:length(indep_coords)
            dx = indep_coords[i][2] - indep_coords[i-1][2]
            dy = indep_coords[i][3] - indep_coords[i-1][3]
            dz = indep_coords[i][4] - indep_coords[i-1][4]
            push!(arc_len, arc_len[end] + sqrt(dx^2 + dy^2 + dz^2))
        end
        total_len = arc_len[end]
        if total_len < 1e-30; continue; end

        for (g, dof) in dep_list
            if !haskey(id_map, g); continue; end
            gi = id_map[g]
            px = node_coords[gi,1]; py = node_coords[gi,2]; pz = node_coords[gi,3]

            # Project dependent point onto spline: find closest segment
            best_seg = 1; best_t = 0.0; best_dist = Inf
            for seg in 1:(length(indep_coords)-1)
                ax = indep_coords[seg][2]; ay = indep_coords[seg][3]; az = indep_coords[seg][4]
                bx = indep_coords[seg+1][2]; by = indep_coords[seg+1][3]; bz = indep_coords[seg+1][4]
                dx = bx-ax; dy = by-ay; dz = bz-az
                seg_len2 = dx^2 + dy^2 + dz^2
                if seg_len2 < 1e-30; continue; end
                t = clamp(((px-ax)*dx + (py-ay)*dy + (pz-az)*dz) / seg_len2, 0.0, 1.0)
                cx = ax + t*dx; cy = ay + t*dy; cz = az + t*dz
                dist = sqrt((px-cx)^2 + (py-cy)^2 + (pz-cz)^2)
                if dist < best_dist
                    best_dist = dist; best_seg = seg; best_t = t
                end
            end

            # Linear interpolation between the two bounding independent nodes
            gi_left = indep_coords[best_seg][1]
            gi_right = indep_coords[best_seg+1][1]
            gdof = (gi-1)*6 + dof
            left_dof = (gi_left-1)*6 + dof
            right_dof = (gi_right-1)*6 + dof
            w_left = 1.0 - best_t
            w_right = best_t
            pairs = Tuple{Int,Float64}[]
            if abs(w_left) > 1e-15; push!(pairs, (left_dof, w_left)); end
            if abs(w_right) > 1e-15; push!(pairs, (right_dof, w_right)); end
            if !isempty(pairs)
                rspline_map[gdof] = pairs
            end
        end
        n_rspline += 1
    end
    if n_rspline > 0
        log_msg("[SOLVER] RSPLINE: $n_rspline elements, $(length(rspline_map)) dependent DOFs")
    end

    # --- RBE3 (weighted interpolation constraint via DOF elimination) ---
    # Uses full 6-DOF rigid body formulation: fit all 6 RB modes at reference,
    # then extract only the REFC components. This properly handles offset reference nodes.
    rbe3s = get(model, "RBE3s", Dict())
    rbe3_map = Dict{Int, Vector{Tuple{Int, Float64}}}()
    n_rbe3 = 0
    for (id, rbe) in rbe3s
        ref_gid = rbe["REFGRID"]
        ref_idx = get(id_map, ref_gid, 0)
        if ref_idx == 0; continue; end

        refc_digits = sort!([parse(Int, string(ch)) for ch in string(Int(rbe["REFC"])) if isdigit(ch)])
        if isempty(refc_digits); continue; end

        p_ref = SVector{3}(node_coords[ref_idx,1], node_coords[ref_idx,2], node_coords[ref_idx,3])
        R_ref = node_R[ref_idx]

        # Build weighted Gram matrix G'WG (6×6) and per-grid G_i matrices
        # G_i maps independent DOFs at node i to full 6 RB DOFs at reference
        A6 = zeros(6, 6)
        grid_Gi = Tuple{Int, Matrix{Float64}, Float64, Vector{Int}}[]

        wt_groups = get(rbe, "WT_GROUPS", [])
        for group in wt_groups
            # Support both NamedTuple (.wt) and Dict (["wt"]) access for JSON compatibility
            wt = Float64(group isa AbstractDict ? group["wt"] : group.wt)
            comps_raw = group isa AbstractDict ? group["comps"] : group.comps
            comps_digits = sort!([parse(Int, string(ch)) for ch in string(Int(comps_raw)) if isdigit(ch)])
            if isempty(comps_digits); continue; end
            grids_raw = group isa AbstractDict ? group["grids"] : group.grids
            for dg in grids_raw
                di = get(id_map, dg, 0)
                if di == 0; continue; end
                p_i = SVector{3}(node_coords[di,1], node_coords[di,2], node_coords[di,3])
                dx = p_i[1] - p_ref[1]; dy = p_i[2] - p_ref[2]; dz = p_i[3] - p_ref[3]
                R_i = node_R[di]

                n_comp = length(comps_digits)
                G_i = zeros(n_comp, 6)  # maps comp DOFs → 6 RB DOFs at ref
                for (jj, cdof) in enumerate(comps_digits)
                    G_i[jj, :] = _rigid_component_row(R_i, R_ref, dx, dy, dz, cdof)
                end

                A6 .+= wt .* (G_i' * G_i)
                push!(grid_Gi, (di, G_i, wt, comps_digits))
            end
        end
        if isempty(grid_Gi); continue; end

        A6_inv = pinv(A6, rtol=1e-10)

        # Build a mapping from REFC digits to rows in A6
        # refc_digits maps to indices in the 6-DOF vector
        for (di, G_i, wt, comps_digits) in grid_Gi
            # C_i = A6_inv * (wt * G_i') → 6 × n_comp
            C_i = A6_inv * (wt .* G_i')

            for rdof in refc_digits
                ref_dof = (ref_idx - 1) * 6 + rdof
                if !haskey(rbe3_map, ref_dof)
                    rbe3_map[ref_dof] = Tuple{Int,Float64}[]
                end
                for (jj, cdof) in enumerate(comps_digits)
                    coeff = C_i[rdof, jj]  # row=rdof in 6-DOF space
                    if abs(coeff) > 1e-15
                        ind_dof = (di - 1) * 6 + cdof
                        push!(rbe3_map[ref_dof], (ind_dof, coeff))
                    end
                end
            end
        end
        n_rbe3 += 1
    end

    # --- Explicit MPC constraints ---
    mpc_cards = get(model, "MPCs", [])
    mpc_map = Dict{Int, Vector{Tuple{Int, Float64}}}()
    n_mpc_explicit = 0
    for mpc in mpc_cards
        terms = mpc["TERMS"]
        if length(terms) < 2; continue; end
        dep_g = terms[1]["G"]; dep_c = terms[1]["C"]; dep_a = terms[1]["A"]
        if !haskey(id_map, dep_g) || dep_c < 1 || dep_c > 6; continue; end
        if abs(dep_a) < 1e-30; continue; end
        dep_dof = (id_map[dep_g]-1)*6 + dep_c
        pairs = Tuple{Int,Float64}[]
        for i in 2:length(terms)
            t = terms[i]
            if !haskey(id_map, t["G"]) || t["C"] < 1 || t["C"] > 6; continue; end
            ind_dof = (id_map[t["G"]]-1)*6 + t["C"]
            coeff = -t["A"] / dep_a
            push!(pairs, (ind_dof, coeff))
        end
        if !isempty(pairs)
            mpc_map[dep_dof] = pairs
            n_mpc_explicit += 1
        end
    end
    if n_mpc_explicit > 0
        log_msg("[SOLVER] MPC: $n_mpc_explicit explicit MPC constraints")
    end

    # Merge all constraint maps: RBE2, RBE1, RSPLINE, RBE3, MPC
    merge!(rbe3_map, rbe2_map, rbe1_map, rspline_map, mpc_map)
    n_mpc_total = length(rbe3_map)
    n_rbe3_only = length(rbe3_map) - length(rbe2_map) - length(rbe1_map) - length(rspline_map) - length(mpc_map)

    if n_rbe3 > 0
        log_msg("[SOLVER] RBE3: $n_rbe3 elements, $n_rbe3_only dependent DOFs")
    end

    if n_mpc_total > 0
        log_msg("[SOLVER] MPC elimination: $n_mpc_total total dependent DOFs (RBE2: $(length(rbe2_map)), RBE1: $(length(rbe1_map)), RSPLINE: $(length(rspline_map)), RBE3: $n_rbe3_only, MPC: $(length(mpc_map)))")
        # Process triplets: redistribute entries involving dependent DOFs
        n_orig = length(I_idx)
        new_I = Int[]; new_J = Int[]; new_V = Float64[]
        sizehint!(new_I, n_orig + n_orig ÷ 10)
        sizehint!(new_J, n_orig + n_orig ÷ 10)
        sizehint!(new_V, n_orig + n_orig ÷ 10)

        for k in 1:n_orig
            i = I_idx[k]; j = J_idx[k]; v = V_val[k]
            i_dep = haskey(rbe3_map, i)
            j_dep = haskey(rbe3_map, j)

            if !i_dep && !j_dep
                push!(new_I, i); push!(new_J, j); push!(new_V, v)
            elseif i_dep && !j_dep
                for (ind_dof, coeff) in rbe3_map[i]
                    push!(new_I, ind_dof); push!(new_J, j); push!(new_V, v * coeff)
                end
            elseif !i_dep && j_dep
                for (ind_dof, coeff) in rbe3_map[j]
                    push!(new_I, i); push!(new_J, ind_dof); push!(new_V, v * coeff)
                end
            else
                for (ind_i, ci) in rbe3_map[i]
                    for (ind_j, cj) in rbe3_map[j]
                        push!(new_I, ind_i); push!(new_J, ind_j); push!(new_V, v * ci * cj)
                    end
                end
            end
        end
        I_idx = new_I; J_idx = new_J; V_val = new_V
        log_msg("[SOLVER] MPC: Triplets redistributed: $n_orig → $(length(I_idx))")
    end

    return rbe3_map, I_idx, J_idx, V_val
end
