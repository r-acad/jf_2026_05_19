# adjoint.jl — Adjoint sensitivity solver for JFEM
#
# Phase 1: displacement responses, shell_thickness and material_E design variables.
# Phase 2: von Mises stress and shell force responses, explicit derivatives.
#
# Usage:
#   adjoint_results = solve_adjoint(results, adjoint_config_path)
#
# where `results` is the Dict returned by solve_model() for SOL 101.

"""
    parse_adjoint_config(json_path::String) -> Dict

Parse an adjoint_config.json file specifying responses and design variables.
"""
function parse_adjoint_config(json_path::String)
    if !isfile(json_path)
        error("[ADJOINT] Config file not found: $json_path")
    end
    config = JSON.parsefile(json_path)

    if !haskey(config, "responses") || isempty(config["responses"])
        error("[ADJOINT] Config must contain at least one response")
    end
    if !haskey(config, "design_variables") || isempty(config["design_variables"])
        error("[ADJOINT] Config must contain at least one design variable")
    end

    for resp in config["responses"]
        haskey(resp, "id") || error("[ADJOINT] Each response must have an 'id'")
        haskey(resp, "type") || error("[ADJOINT] Response '$(resp["id"])' must have a 'type'")
    end
    for dv in config["design_variables"]
        haskey(dv, "id") || error("[ADJOINT] Each design variable must have an 'id'")
        haskey(dv, "type") || error("[ADJOINT] Design variable '$(dv["id"])' must have a 'type'")
    end

    return config
end

# ============================================================================
# Shell element B-matrix and stress helpers
# ============================================================================

"""
Build Bm (3×ndof_elem) and Bb (3×ndof_elem) at element centroid for QUAD4 or TRIA3.
Returns (Bm, Bb, D) where D = E/(1-nu²) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2].
"""
function _shell_centroid_B_matrices(n_nodes::Int, lc, E::Float64, nu::Float64)
    D = (E / (1 - nu^2)) .* [1.0 nu 0.0; nu 1.0 0.0; 0.0 0.0 (1-nu)/2]
    ndof_elem = n_nodes * 6

    if n_nodes == 4
        # QUAD4: shape derivative at centroid (xi=eta=0)
        dNr = SVector{4}(-0.25, 0.25, 0.25, -0.25)
        dNs = SVector{4}(-0.25, -0.25, 0.25, 0.25)
        J = [dNr'; dNs'] * lc  # 2x2 Jacobian
        invJ = inv(J)
        dN_dxy = invJ * [dNr'; dNs']  # 2x4

        Bm = zeros(3, ndof_elem)
        Bb = zeros(3, ndof_elem)
        for k in 1:4
            idx = (k-1)*6
            Bm[1, idx+1] = dN_dxy[1,k]
            Bm[2, idx+2] = dN_dxy[2,k]
            Bm[3, idx+1] = dN_dxy[2,k]
            Bm[3, idx+2] = dN_dxy[1,k]
            Bb[1, idx+5] = dN_dxy[1,k]
            Bb[2, idx+4] = -dN_dxy[2,k]
            Bb[3, idx+5] = dN_dxy[2,k]
            Bb[3, idx+4] = -dN_dxy[1,k]
        end
    else
        # TRIA3: constant strain, B is constant over element
        x, y = lc[:,1], lc[:,2]
        A = 0.5 * abs(x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2]))
        if A < 1e-12; A = 1e-12; end
        b = [y[2]-y[3], y[3]-y[1], y[1]-y[2]] ./ (2*A)
        c = [x[3]-x[2], x[1]-x[3], x[2]-x[1]] ./ (2*A)

        Bm = zeros(3, ndof_elem)
        Bb = zeros(3, ndof_elem)
        for k in 1:3
            idx = (k-1)*6
            # Membrane: u, v DOFs
            Bm[1, idx+1] = b[k]
            Bm[2, idx+2] = c[k]
            Bm[3, idx+1] = c[k]
            Bm[3, idx+2] = b[k]
            # Bending: rx, ry DOFs
            Bb[1, idx+5] = b[k]
            Bb[2, idx+4] = -c[k]
            Bb[3, idx+5] = c[k]
            Bb[3, idx+4] = -b[k]
        end
    end

    return Bm, Bb, D
end

"""
Compute von Mises stress and its derivative for 2D plane stress.
sigma = [sxx, syy, sxy]
Returns (VM, dVM_dsigma) where dVM_dsigma is a 3-vector.
"""
function _von_mises_plane_stress(sigma)
    s1, s2, s12 = sigma[1], sigma[2], sigma[3]
    VM_sq = s1^2 + s2^2 - s1*s2 + 3*s12^2
    VM = sqrt(max(VM_sq, 1e-30))  # avoid divide-by-zero
    dVM = [2*s1 - s2, 2*s2 - s1, 6*s12] ./ (2*VM)
    return VM, dVM
end

"""
Look up the shell element data (coords, E, nu, h, T matrix, DOF indices)
needed for stress response evaluation at element `eid`.
Returns a NamedTuple or nothing if element not found.
"""
function _get_shell_element_data(eid::Int, model, id_map, node_coords, node_R)
    eid_str = string(eid)
    if !haskey(model["CSHELLs"], eid_str); return nothing; end
    el = model["CSHELLs"][eid_str]
    pid_str = string(el["PID"])
    pshells = model["PSHELLs"]
    if !haskey(pshells, pid_str); return nothing; end
    prop = pshells[pid_str]
    mid_str = string(prop["MID"])
    mats = model["MATs"]
    if !haskey(mats, mid_str); return nothing; end
    mat = mats[mid_str]

    nids = el["NODES"]
    n_nodes = length(nids)
    if !all(n -> haskey(id_map, n), nids); return nothing; end
    idxs = [id_map[n] for n in nids]

    h = Float64(prop["T"])
    E = Float64(mat["E"])
    nu = Float64(mat["NU"])

    # Build coordinates and local frame
    ps = [SVector{3}(node_coords[idx,1], node_coords[idx,2], node_coords[idx,3]) for idx in idxs]
    if n_nodes == 4
        v1, v2, v3 = shell_element_frame_fast(ps[1], ps[2], ps[3], ps[4], 4)
    else
        v1, v2, v3 = shell_element_frame_fast(ps[1], ps[2], ps[3], SVector{3}(0.0,0.0,0.0), 3)
    end

    c_center = sum(ps) / n_nodes
    lc = zeros(n_nodes, 2)
    for k in 1:n_nodes
        dp = ps[k] - c_center
        lc[k,1] = dot(dp, v1); lc[k,2] = dot(dp, v2)
    end

    # Rotation matrix
    ndof_elem = n_nodes * 6
    T_mat = zeros(ndof_elem, ndof_elem)
    Rel = [v1[1] v1[2] v1[3]; v2[1] v2[2] v2[3]; v3[1] v3[2] v3[3]]
    for k in 1:n_nodes
        TR = Rel * node_R[idxs[k]]
        base = (k-1)*6
        T_mat[base+1:base+3, base+1:base+3] = TR
        T_mat[base+4:base+6, base+4:base+6] = TR
    end

    # Global DOF indices
    dofs = Vector{Int}(undef, ndof_elem)
    for k in 1:n_nodes
        base_g = (idxs[k]-1)*6; base_e = (k-1)*6
        for d in 1:6; dofs[base_e+d] = base_g+d; end
    end

    return (n_nodes=n_nodes, lc=lc, E=E, nu=nu, h=h, T_mat=T_mat, dofs=dofs,
            idxs=idxs, pid_str=pid_str, mid_str=mid_str, ndof_elem=ndof_elem)
end

# ============================================================================
# Response evaluation
# ============================================================================

"""
    evaluate_response(resp, u_global, model, id_map, ndof, node_coords, node_R) -> Float64

Evaluate the scalar response function value.
Supports: displacement, von_mises, shell_force_nx/ny/nxy, shell_moment_mx/my.
"""
function evaluate_response(resp, u_global, model, id_map, ndof, node_coords=nothing, node_R=nothing)
    rtype = resp["type"]

    if rtype == "displacement"
        grid = Int(resp["grid"]); dof = Int(resp["dof"])
        idx = id_map[grid]
        return u_global[(idx-1)*6 + dof]

    elseif rtype == "von_mises"
        eid = Int(resp["eid"])
        surface = get(resp, "surface", "top")  # "top" or "bottom"
        ed = _get_shell_element_data(eid, model, id_map, node_coords, node_R)
        isnothing(ed) && error("[ADJOINT] Element $eid not found for von_mises response")

        Bm, Bb, D = _shell_centroid_B_matrices(ed.n_nodes, ed.lc, ed.E, ed.nu)
        u_elem_global = [u_global[ed.dofs[i]] for i in 1:ed.ndof_elem]
        u_local = ed.T_mat * u_elem_global

        eps_mem = Bm * u_local
        kappa = Bb * u_local
        z = surface == "bottom" ? -ed.h/2 : ed.h/2
        sigma = D * (eps_mem .+ z .* kappa)
        VM, _ = _von_mises_plane_stress(sigma)
        return VM

    elseif startswith(rtype, "shell_force_") || startswith(rtype, "shell_moment_")
        eid = Int(resp["eid"])
        ed = _get_shell_element_data(eid, model, id_map, node_coords, node_R)
        isnothing(ed) && error("[ADJOINT] Element $eid not found for $rtype response")

        Bm, Bb, D = _shell_centroid_B_matrices(ed.n_nodes, ed.lc, ed.E, ed.nu)
        u_elem_global = [u_global[ed.dofs[i]] for i in 1:ed.ndof_elem]
        u_local = ed.T_mat * u_elem_global

        if startswith(rtype, "shell_force_")
            Cm = D * ed.h
            eps_mem = Bm * u_local
            N = Cm * eps_mem
            comp = rtype == "shell_force_nx" ? 1 : rtype == "shell_force_ny" ? 2 : 3
            return N[comp]
        else  # shell_moment_
            Cb = D * (ed.h^3 / 12.0)
            kappa = Bb * u_local
            M = -Cb * kappa
            comp = rtype == "shell_moment_mx" ? 1 : 2
            return M[comp]
        end
    else
        error("[ADJOINT] Unsupported response type: $rtype")
    end
end

# ============================================================================
# dr/du computation (adjoint RHS)
# ============================================================================

"""
    compute_dr_du(resp, u_global, model, id_map, ndof, node_coords, node_R) -> Vector{Float64}

Compute dr/du — the adjoint RHS vector.
"""
function compute_dr_du(resp, u_global, model, id_map, ndof, node_coords=nothing, node_R=nothing)
    rtype = resp["type"]

    if rtype == "displacement"
        grid = Int(resp["grid"]); dof = Int(resp["dof"])
        idx = id_map[grid]
        dr_du = zeros(ndof)
        dr_du[(idx-1)*6 + dof] = 1.0
        return dr_du

    elseif rtype == "von_mises"
        eid = Int(resp["eid"])
        surface = get(resp, "surface", "top")
        ed = _get_shell_element_data(eid, model, id_map, node_coords, node_R)
        isnothing(ed) && error("[ADJOINT] Element $eid not found")

        Bm, Bb, D = _shell_centroid_B_matrices(ed.n_nodes, ed.lc, ed.E, ed.nu)
        u_elem_global = [u_global[ed.dofs[i]] for i in 1:ed.ndof_elem]
        u_local = ed.T_mat * u_elem_global

        eps_mem = Bm * u_local
        kappa = Bb * u_local
        z = surface == "bottom" ? -ed.h/2 : ed.h/2
        sigma = D * (eps_mem .+ z .* kappa)
        _, dVM_dsigma = _von_mises_plane_stress(sigma)

        # dVM/du_local = dVM/dsigma' * D * (Bm + z*Bb)
        B_combined = Bm .+ z .* Bb
        dr_du_local = (dVM_dsigma' * D * B_combined)'  # ndof_elem vector

        # Transform to global: dr/du_global_elem = T' * dr/du_local
        # But dr/du_global needs T^{-T} because u_local = T * u_global_elem
        # dr/du_global = (dr/du_local)' * T  →  dr/du_global_elem = T' * dr/du_local
        dr_du_elem = ed.T_mat' * dr_du_local

        dr_du = zeros(ndof)
        for i in 1:ed.ndof_elem
            dr_du[ed.dofs[i]] += dr_du_elem[i]
        end
        return dr_du

    elseif startswith(rtype, "shell_force_")
        eid = Int(resp["eid"])
        ed = _get_shell_element_data(eid, model, id_map, node_coords, node_R)
        isnothing(ed) && error("[ADJOINT] Element $eid not found")

        Bm, _, D = _shell_centroid_B_matrices(ed.n_nodes, ed.lc, ed.E, ed.nu)
        Cm = D * ed.h
        comp = rtype == "shell_force_nx" ? 1 : rtype == "shell_force_ny" ? 2 : 3

        # dN_comp/du_local = Cm[comp,:] * Bm
        dr_du_local = (Cm[comp,:]' * Bm)'  # ndof_elem vector
        dr_du_elem = ed.T_mat' * dr_du_local

        dr_du = zeros(ndof)
        for i in 1:ed.ndof_elem
            dr_du[ed.dofs[i]] += dr_du_elem[i]
        end
        return dr_du

    elseif startswith(rtype, "shell_moment_")
        eid = Int(resp["eid"])
        ed = _get_shell_element_data(eid, model, id_map, node_coords, node_R)
        isnothing(ed) && error("[ADJOINT] Element $eid not found")

        _, Bb, D = _shell_centroid_B_matrices(ed.n_nodes, ed.lc, ed.E, ed.nu)
        Cb = D * (ed.h^3 / 12.0)
        comp = rtype == "shell_moment_mx" ? 1 : 2

        # M = -Cb * kappa = -Cb * Bb * u_local
        # dM_comp/du_local = -Cb[comp,:] * Bb
        dr_du_local = -(Cb[comp,:]' * Bb)'
        dr_du_elem = ed.T_mat' * dr_du_local

        dr_du = zeros(ndof)
        for i in 1:ed.ndof_elem
            dr_du[ed.dofs[i]] += dr_du_elem[i]
        end
        return dr_du

    else
        error("[ADJOINT] Unsupported response type for dr/du: $rtype")
    end
end

# ============================================================================
# dr/dx|explicit — stress depends directly on design variables
# ============================================================================

"""
    compute_dr_dx_explicit(resp, dv, model, id_map, node_coords, node_R, u_global, ndof) -> Dict{String, Float64}

Compute dr/dx|explicit for stress/force responses.
For displacement responses, this is zero.
Returns per-group dict (same structure as dKdx_u_per_group).
"""
function compute_dr_dx_explicit(resp, dv, model, id_map, node_coords, node_R, u_global, ndof)
    rtype = resp["type"]
    dv_type = dv["type"]

    # Displacement responses have no explicit derivative
    if rtype == "displacement"
        return _zero_explicit_groups(dv)
    end

    # Only stress/force responses on shells have explicit thickness derivatives
    if !(startswith(rtype, "von_mises") || startswith(rtype, "shell_force_") || startswith(rtype, "shell_moment_"))
        return _zero_explicit_groups(dv)
    end

    eid = Int(resp["eid"])
    ed = _get_shell_element_data(eid, model, id_map, node_coords, node_R)
    if isnothing(ed); return _zero_explicit_groups(dv); end

    Bm, Bb, D = _shell_centroid_B_matrices(ed.n_nodes, ed.lc, ed.E, ed.nu)
    u_elem_global = [u_global[ed.dofs[i]] for i in 1:ed.ndof_elem]
    u_local = ed.T_mat * u_elem_global
    eps_mem = Bm * u_local
    kappa = Bb * u_local

    result = Dict{String, Float64}()

    if dv_type == "shell_thickness"
        pids = Set(string.(dv["pids"]))
        # Element must belong to one of the DVs PIDs
        if !(ed.pid_str in pids)
            return _zero_explicit_groups(dv)
        end

        h = ed.h
        if rtype == "von_mises"
            surface = get(resp, "surface", "top")
            z = surface == "bottom" ? -h/2 : h/2
            sigma = D * (eps_mem .+ z .* kappa)
            _, dVM_dsigma = _von_mises_plane_stress(sigma)
            # dsigma/dh|explicit = D * (dz/dh * kappa) = D * (±1/2 * kappa)
            dz_dh = surface == "bottom" ? -0.5 : 0.5
            dsigma_dh = D * (dz_dh .* kappa)
            dr_dh = dot(dVM_dsigma, dsigma_dh)

        elseif startswith(rtype, "shell_force_")
            comp = rtype == "shell_force_nx" ? 1 : rtype == "shell_force_ny" ? 2 : 3
            # N = D*h * eps_mem → dN/dh = D * eps_mem
            dN_dh = D * eps_mem
            dr_dh = dN_dh[comp]

        elseif startswith(rtype, "shell_moment_")
            comp = rtype == "shell_moment_mx" ? 1 : 2
            # M = -D * (h³/12) * kappa → dM/dh = -D * (3h²/12) * kappa = -D * h²/4 * kappa
            dM_dh = -D * (h^2 / 4.0) .* kappa
            dr_dh = dM_dh[comp]
        else
            dr_dh = 0.0
        end

        result["PID_$(ed.pid_str)"] = dr_dh
        # Fill zeros for other PIDs in this DV
        for pid in dv["pids"]
            key = "PID_$(string(Int(pid)))"
            if !haskey(result, key); result[key] = 0.0; end
        end

    elseif dv_type == "material_E"
        mids = Set(string.(dv["mids"]))
        if !(ed.mid_str in mids)
            return _zero_explicit_groups(dv)
        end

        E = ed.E; h = ed.h

        if rtype == "von_mises"
            surface = get(resp, "surface", "top")
            z = surface == "bottom" ? -h/2 : h/2
            sigma = D * (eps_mem .+ z .* kappa)
            _, dVM_dsigma = _von_mises_plane_stress(sigma)
            # dsigma/dE|explicit = dD/dE * (eps_mem + z*kappa) = sigma/E
            dsigma_dE = sigma / E
            dr_dE = dot(dVM_dsigma, dsigma_dE)

        elseif startswith(rtype, "shell_force_")
            comp = rtype == "shell_force_nx" ? 1 : rtype == "shell_force_ny" ? 2 : 3
            # N = D*h * eps_mem → dN/dE = (D/E)*h * eps_mem = N/E
            N = D * h * eps_mem
            dr_dE = N[comp] / E

        elseif startswith(rtype, "shell_moment_")
            comp = rtype == "shell_moment_mx" ? 1 : 2
            M = -D * (h^3/12.0) * kappa
            dr_dE = M[comp] / E
        else
            dr_dE = 0.0
        end

        result["MID_$(ed.mid_str)"] = dr_dE
        for mid in dv["mids"]
            key = "MID_$(string(Int(mid)))"
            if !haskey(result, key); result[key] = 0.0; end
        end

    elseif dv_type == "material_NU"
        mids = Set(string.(dv["mids"]))
        if !(ed.mid_str in mids)
            return _zero_explicit_groups(dv)
        end

        E = ed.E; nu = ed.nu; h = ed.h
        # dD/dnu via FD (D matrix depends on nu in a complex way)
        delta_nu = max(abs(nu) * 1e-6, 1e-8)
        D_plus = (E / (1 - (nu+delta_nu)^2)) .* [1.0 nu+delta_nu 0.0; nu+delta_nu 1.0 0.0; 0.0 0.0 (1-(nu+delta_nu))/2]
        D_minus = (E / (1 - (nu-delta_nu)^2)) .* [1.0 nu-delta_nu 0.0; nu-delta_nu 1.0 0.0; 0.0 0.0 (1-(nu-delta_nu))/2]
        dD_dnu = (D_plus - D_minus) / (2.0 * delta_nu)

        if rtype == "von_mises"
            surface = get(resp, "surface", "top")
            z = surface == "bottom" ? -h/2 : h/2
            sigma = D * (eps_mem .+ z .* kappa)
            _, dVM_dsigma = _von_mises_plane_stress(sigma)
            dsigma_dnu = dD_dnu * (eps_mem .+ z .* kappa)
            dr_dnu = dot(dVM_dsigma, dsigma_dnu)

        elseif startswith(rtype, "shell_force_")
            comp = rtype == "shell_force_nx" ? 1 : rtype == "shell_force_ny" ? 2 : 3
            dN_dnu = dD_dnu * h * eps_mem
            dr_dnu = dN_dnu[comp]

        elseif startswith(rtype, "shell_moment_")
            comp = rtype == "shell_moment_mx" ? 1 : 2
            dM_dnu = -dD_dnu * (h^3/12.0) * kappa
            dr_dnu = dM_dnu[comp]
        else
            dr_dnu = 0.0
        end

        result["MID_$(ed.mid_str)"] = dr_dnu
        for mid in dv["mids"]
            key = "MID_$(string(Int(mid)))"
            if !haskey(result, key); result[key] = 0.0; end
        end

    elseif dv_type == "bar_area"
        return _zero_explicit_groups(dv)

    elseif dv_type == "pcomp_ply_thickness" || dv_type == "pcomp_ply_angle"
        pids = Set(string.(dv["pids"]))
        if !(ed.pid_str in pids); return _zero_explicit_groups(dv); end
        prop = get(model["PSHELLs"], ed.pid_str, nothing)
        if isnothing(prop) || get(prop, "TYPE", "") != "PCOMP_CLT"
            return _zero_explicit_groups(dv)
        end
        ply_idx = Int(dv["ply_index"])
        if !haskey(prop, "PLY_DATA") || ply_idx > length(prop["PLY_DATA"])
            return _zero_explicit_groups(dv)
        end

        perturb_field = dv_type == "pcomp_ply_thickness" ? :T : :THETA
        h = ed.h
        mats = model["MATs"]
        dv_method = get_dv_method(dv)

        local dCm, dCb
        if dv_method == :laminate_exact
            clt_deriv = _pcomp_exact_constitutive_derivative(prop, mats, ply_idx, perturb_field)
            if isnothing(clt_deriv)
                dv_method = :clt_fd
            else
                dCm = clt_deriv[1]
                dCb = clt_deriv[3]
            end
        end

        if dv_method == :clt_fd
            ply = prop["PLY_DATA"][ply_idx]
            if perturb_field == :T
                t_ply = Float64(ply["z_top"] - ply["z_bot"])
                delta = max(abs(t_ply) * 1e-6, 1e-12)
            else
                theta_ply = deg2rad(Float64(ply["theta"]))
                delta = max(abs(theta_ply) * 1e-6, 1e-6)
            end

            Cm_p, _, Cb_p, _ = _recompute_clt(prop, mats; perturb_ply=ply_idx, perturb_field=perturb_field, perturb_delta=delta)
            Cm_m, _, Cb_m, _ = _recompute_clt(prop, mats; perturb_ply=ply_idx, perturb_field=perturb_field, perturb_delta=-delta)
            dCm = (Cm_p - Cm_m) / (2.0 * delta)
            dCb = (Cb_p - Cb_m) / (2.0 * delta)
        elseif dv_method != :laminate_exact
            error("[ADJOINT] Unsupported explicit-response backend '$dv_method' for PCOMP ply design variable")
        end

        D_eff = dCm / max(h, 1e-30)

        if rtype == "von_mises"
            surface = get(resp, "surface", "top")
            z = surface == "bottom" ? -h/2 : h/2
            sigma = (prop["Cm"] / max(h, 1e-30)) * (eps_mem .+ z .* kappa)
            _, dVM_dsigma = _von_mises_plane_stress(sigma)
            dsigma_dx = D_eff * (eps_mem .+ z .* kappa)
            dr_dx = dot(dVM_dsigma, dsigma_dx)
        elseif startswith(rtype, "shell_force_")
            comp = rtype == "shell_force_nx" ? 1 : rtype == "shell_force_ny" ? 2 : 3
            dN = dCm * eps_mem
            dr_dx = dN[comp]
        elseif startswith(rtype, "shell_moment_")
            comp = rtype == "shell_moment_mx" ? 1 : 2
            dM = -dCb * kappa
            dr_dx = dM[comp]
        else
            dr_dx = 0.0
        end

        result["PID_$(ed.pid_str)"] = dr_dx
        for pid in dv["pids"]
            key = "PID_$(string(Int(pid)))"
            if !haskey(result, key); result[key] = 0.0; end
        end

    elseif dv_type == "node_coord"
        # Stress depends on geometry (B matrices) — use FD on response evaluation
        grid = Int(dv["grid"]); comp = Int(dv["comp"])
        grid_str = string(grid)
        if !haskey(model["GRIDs"], grid_str)
            return _zero_explicit_groups(dv)
        end

        coords_arr = model["GRIDs"][grid_str]["X"]
        x0 = Float64(coords_arr[comp])
        delta = max(abs(x0) * 1e-6, 1e-8)

        # Check if this node is connected to the response element
        eid = Int(resp["eid"])
        el = get(model["CSHELLs"], string(eid), nothing)
        if isnothing(el) || !(grid in el["NODES"])
            # Node not connected to response element — no explicit dependence
            return _zero_explicit_groups(dv)
        end

        # FD on response at perturbed geometry (fixed u)
        coords_arr[comp] = x0 + delta
        r_plus = evaluate_response(resp, u_global, model, id_map, ndof,
            _rebuild_node_coords(model, id_map), node_R)
        coords_arr[comp] = x0 - delta
        r_minus = evaluate_response(resp, u_global, model, id_map, ndof,
            _rebuild_node_coords(model, id_map), node_R)
        coords_arr[comp] = x0  # restore

        dr_dx = (r_plus - r_minus) / (2.0 * delta)
        result["GRID_$(grid)_$(comp)"] = dr_dx

    else
        return _zero_explicit_groups(dv)
    end

    return result
end

"""Rebuild node_coords matrix from model GRIDs (for FD on geometry)."""
function _rebuild_node_coords(model, id_map)
    n_nodes = length(id_map)
    X = zeros(n_nodes, 3)
    for (sid, g) in model["GRIDs"]
        idx = id_map[g["ID"]]
        X[idx, :] = g["X"]
    end
    return X
end

function _zero_explicit_groups(dv)
    dv_type = dv["type"]
    result = Dict{String, Float64}()
    if dv_type in ("shell_thickness", "bar_area", "pcomp_ply_thickness", "pcomp_ply_angle")
        for pid in dv["pids"]; result["PID_$(string(Int(pid)))"] = 0.0; end
    elseif dv_type in ("material_E", "material_NU")
        for mid in dv["mids"]; result["MID_$(string(Int(mid)))"] = 0.0; end
    elseif dv_type == "node_coord"
        grid = Int(dv["grid"]); comp = Int(dv["comp"])
        result["GRID_$(grid)_$(comp)"] = 0.0
    elseif dv_type == "topology_density"
        for eid in dv["eids"]; result["EID_$(string(Int(eid)))"] = 0.0; end
    end
    return result
end

# ============================================================================
# Design variable value extraction
# ============================================================================

function get_design_variable_values(dv, model)
    dv_type = dv["type"]
    values = Dict{String, Float64}()

    if dv_type == "shell_thickness"
        pshells = model["PSHELLs"]
        for pid in dv["pids"]
            pid_str = string(Int(pid))
            if haskey(pshells, pid_str)
                values["PID_$pid_str"] = pshells[pid_str]["T"]
            end
        end
    elseif dv_type == "material_E"
        mats = model["MATs"]
        for mid in dv["mids"]
            mid_str = string(Int(mid))
            if haskey(mats, mid_str)
                values["MID_$mid_str"] = mats[mid_str]["E"]
            end
        end
    elseif dv_type == "material_NU"
        mats = model["MATs"]
        for mid in dv["mids"]
            mid_str = string(Int(mid))
            if haskey(mats, mid_str)
                values["MID_$mid_str"] = mats[mid_str]["NU"]
            end
        end
    elseif dv_type == "bar_area"
        pbarls = get(model, "PBARLs", Dict())
        for pid in dv["pids"]
            pid_str = string(Int(pid))
            if haskey(pbarls, pid_str)
                values["PID_$pid_str"] = pbarls[pid_str]["A"]
            end
        end
    elseif dv_type == "pcomp_ply_thickness"
        pshells = model["PSHELLs"]
        ply_idx = Int(dv["ply_index"])
        for pid in dv["pids"]
            pid_str = string(Int(pid))
            prop = get(pshells, pid_str, nothing)
            if !isnothing(prop) && haskey(prop, "PLY_DATA") && ply_idx <= length(prop["PLY_DATA"])
                ply = prop["PLY_DATA"][ply_idx]
                values["PID_$pid_str"] = Float64(ply["z_top"] - ply["z_bot"])
            end
        end
    elseif dv_type == "pcomp_ply_angle"
        pshells = model["PSHELLs"]
        ply_idx = Int(dv["ply_index"])
        for pid in dv["pids"]
            pid_str = string(Int(pid))
            prop = get(pshells, pid_str, nothing)
            if !isnothing(prop) && haskey(prop, "PLY_DATA") && ply_idx <= length(prop["PLY_DATA"])
                values["PID_$pid_str"] = Float64(prop["PLY_DATA"][ply_idx]["theta"])
            end
        end
    elseif dv_type == "node_coord"
        grid = Int(dv["grid"]); comp = Int(dv["comp"])
        grid_str = string(grid)
        if haskey(model["GRIDs"], grid_str)
            values["GRID_$(grid)_$(comp)"] = Float64(model["GRIDs"][grid_str]["X"][comp])
        end
    elseif dv_type == "topology_density"
        densities = dv["densities"]
        for eid in dv["eids"]
            eid_str = string(Int(eid))
            values["EID_$eid_str"] = Float64(get(densities, eid_str, 1.0))
        end
    end

    return values
end

# ============================================================================
# Main adjoint solver
# ============================================================================

"""
    solve_adjoint(results::Dict, adjoint_config_path::String) -> Dict

Run adjoint sensitivity analysis on SOL 101 results.
"""
function solve_adjoint(results::Dict, adjoint_config_path::String)
    if results["sol_type"] != 101
        error("[ADJOINT] Adjoint solver requires SOL 101 results (got SOL $(results["sol_type"]))")
    end

    config = parse_adjoint_config(adjoint_config_path)
    responses = config["responses"]
    design_vars = config["design_variables"]

    model = results["model"]
    id_map = results["id_map"]
    X = results["node_coords"]
    K = results["K"]
    ndof = results["ndof"]
    node_R = results["node_R"]

    n_resp = length(responses)
    n_dv = length(design_vars)
    log_msg("[ADJOINT] Starting adjoint sensitivity analysis: $n_resp responses × $n_dv design variables")

    subcases = results["subcases"]
    all_adjoint_results = Dict{Int, Dict}()

    for sc in subcases
        sid = sc["sid"]
        u_global = sc["u_analysis"]
        fixed_dofs_sc = sc["fixed_dofs"]

        subcase_responses = filter(r -> get(r, "subcase", 1) == sid, responses)
        if isempty(subcase_responses); continue; end

        log_msg("[ADJOINT] Subcase $sid: $(length(subcase_responses)) responses")

        free_dofs = sort(collect(setdiff(1:ndof, fixed_dofs_sc)))
        K_ff = K[free_dofs, free_dofs]
        log_msg("[ADJOINT] Factorizing K_ff ($(length(free_dofs)) free DOFs)...")
        K_fact = cholesky(Symmetric(K_ff))

        sensitivities = Dict{String, Dict{String, Dict{String, Float64}}}()
        response_values = Dict{String, Float64}()

        for resp in subcase_responses
            resp_id = resp["id"]
            log_msg("[ADJOINT]   Response: $resp_id ($(resp["type"]))")

            # Evaluate response value
            r_value = evaluate_response(resp, u_global, model, id_map, ndof, X, node_R)
            response_values[resp_id] = r_value

            # Compute adjoint RHS: dr/du
            dr_du_full = compute_dr_du(resp, u_global, model, id_map, ndof, X, node_R)

            # Solve adjoint equation
            dr_du_f = dr_du_full[free_dofs]
            lambda_f = K_fact \ dr_du_f
            lambda_full = zeros(ndof)
            lambda_full[free_dofs] = lambda_f

            # Compute sensitivities for each design variable
            sensitivities[resp_id] = Dict{String, Dict{String, Float64}}()
            for dv in design_vars
                dv_id = dv["id"]

                # Implicit part: -lambda^T * dK/dx * u (per group)
                dKdx_u_groups = compute_dKdx_u_per_group(dv, model, id_map, X, node_R, u_global, ndof)

                # Explicit part: dr/dx|explicit (per group)
                dr_dx_explicit = compute_dr_dx_explicit(resp, dv, model, id_map, X, node_R, u_global, ndof)

                # Total: dr/dx = dr/dx|explicit + lambda^T * (dF/dx - dK/dx * u)
                # dF/dx = 0 for all current DV types
                group_sens = Dict{String, Float64}()
                for (group_label, dKdx_u_vec) in dKdx_u_groups
                    implicit = -dot(lambda_full, dKdx_u_vec)
                    explicit = get(dr_dx_explicit, group_label, 0.0)
                    group_sens[group_label] = explicit + implicit
                end
                sensitivities[resp_id][dv_id] = group_sens
            end
        end

        dv_values = Dict{String, Dict{String, Float64}}()
        for dv in design_vars
            dv_values[dv["id"]] = get_design_variable_values(dv, model)
        end

        all_adjoint_results[sid] = Dict(
            "sensitivities" => sensitivities,
            "response_values" => response_values,
            "design_variable_values" => dv_values,
        )
    end

    if length(all_adjoint_results) == 1
        return first(values(all_adjoint_results))
    end
    return all_adjoint_results
end

"""
    export_adjoint_json(adjoint_results::Dict, output_path::String)

Write adjoint sensitivity results to a JSON file.
"""
function export_adjoint_json(adjoint_results::Dict, output_path::String)
    open(output_path, "w") do f
        JSON.print(f, adjoint_results, 2)
    end
    log_msg("[ADJOINT] Results written to: $output_path")
end
