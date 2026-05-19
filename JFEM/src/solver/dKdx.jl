# dKdx.jl — dK/dx computation for adjoint sensitivity analysis
#
# Computes the pseudo-load vector  psi = dK/dx * u  for each design variable.
#
# Architecture: each DV type has a registered derivative method:
#   :analytical      — exact closed-form (e.g. dK/dE = K/E, SIMP dK/dρ = p·ρ^(p-1)·K₀)
#   :element_fd      — central FD on per-element stiffness (shell_thickness, material_NU, bar_area)
#   :clt_fd          — CLT recomputation + element FD fallback for laminate plies
#   :laminate_exact  — exact CLT laminate-matrix derivative for PCOMP ply thickness/angle
#   :full_model_fd   — full reassembly FD (node_coord — expensive but exact for any geometry)
#   :ad_forward      — ForwardDiff through supported element kernels
#
# The adjoint solver calls compute_dKdx_u() and gets a vector back regardless of method.
# To add a new DV type: register it in DV_REGISTRY and implement the _dKdx_u_* function.

# ============================================================================
# DV type registry: (method, group_key_field, group_prefix)
# ============================================================================
const DV_REGISTRY = Dict{String, NamedTuple{(:method, :key_field, :prefix), Tuple{Symbol, String, String}}}(
    "shell_thickness"      => (method=:ad_forward,    key_field="pids", prefix="PID"),
    "material_E"           => (method=:analytical,    key_field="mids", prefix="MID"),
    "material_NU"          => (method=:ad_forward,    key_field="mids", prefix="MID"),
    "bar_area"             => (method=:ad_forward,    key_field="pids", prefix="PID"),
    "pcomp_ply_thickness"  => (method=:laminate_exact,key_field="pids", prefix="PID"),
    "pcomp_ply_angle"      => (method=:laminate_exact,key_field="pids", prefix="PID"),
    "node_coord"           => (method=:full_model_fd, key_field="",     prefix=""),
    "topology_density"     => (method=:analytical,    key_field="eids", prefix="EID"),
)

# Dispatch table: method → implementation function
const _DKDX_DISPATCH = Dict{String, Function}(
    "shell_thickness"     => (dv, m, id, nc, nR, u, n) -> _dKdx_u_shell_thickness(dv, m, id, nc, nR, u, n),
    "material_E"          => (dv, m, id, nc, nR, u, n) -> _dKdx_u_material_E(dv, m, id, nc, nR, u, n),
    "material_NU"         => (dv, m, id, nc, nR, u, n) -> _dKdx_u_material_NU(dv, m, id, nc, nR, u, n),
    "bar_area"            => (dv, m, id, nc, nR, u, n) -> _dKdx_u_bar_area(dv, m, id, nc, nR, u, n),
    "pcomp_ply_thickness" => (dv, m, id, nc, nR, u, n) -> _dKdx_u_pcomp_ply(dv, m, id, nc, nR, u, n),
    "pcomp_ply_angle"     => (dv, m, id, nc, nR, u, n) -> _dKdx_u_pcomp_ply(dv, m, id, nc, nR, u, n),
    "node_coord"          => (dv, m, id, nc, nR, u, n) -> _dKdx_u_node_coord(dv, m, u, n),
    "topology_density"    => (dv, m, id, nc, nR, u, n) -> _dKdx_u_topology_density(dv, m, id, nc, nR, u, n),
)

"""
    compute_dKdx_u(dv, model, id_map, node_coords, node_R, u_global, ndof) -> Vector{Float64}

Compute the pseudo-load vector  dK/dx * u  for a single design variable `dv`.
Returns a global-size vector (length ndof).

Dispatches to the appropriate implementation based on `dv["type"]`
and an optional `dv["method"]` override:
- Analytical methods (material_E, topology_density): exact, O(N_elements)
- Element FD methods (NU, optional shell and bar fallback paths): 2 element Ke evals per element, O(N_elements)
- CLT FD (pcomp plies): CLT recompute + element FD, O(N_elements)
- Full-model FD (node_coord): 2 full assemble_stiffness calls, O(assembly)
"""
function compute_dKdx_u(dv, model, id_map, node_coords, node_R, u_global, ndof)
    dv_type = dv["type"]
    dv_method = get_dv_method(dv)

    if dv_type == "shell_thickness"
        if dv_method == :ad_forward
            return _dKdx_u_shell_thickness_ad_forward(dv, model, id_map, node_coords, node_R, u_global, ndof)
        elseif dv_method == :element_fd
            return _dKdx_u_shell_thickness(dv, model, id_map, node_coords, node_R, u_global, ndof)
        else
            error("[ADJOINT] Unsupported derivative backend '$dv_method' for design variable type '$dv_type'")
        end
    elseif dv_type == "material_NU"
        if dv_method == :ad_forward
            return _dKdx_u_material_NU_ad_forward(dv, model, id_map, node_coords, node_R, u_global, ndof)
        elseif dv_method == :element_fd
            return _dKdx_u_material_NU(dv, model, id_map, node_coords, node_R, u_global, ndof)
        else
            error("[ADJOINT] Unsupported derivative backend '$dv_method' for design variable type '$dv_type'")
        end
    elseif dv_type == "bar_area"
        if dv_method == :ad_forward
            return _dKdx_u_bar_area_ad_forward(dv, model, id_map, node_coords, node_R, u_global, ndof)
        elseif dv_method == :element_fd
            return _dKdx_u_bar_area(dv, model, id_map, node_coords, node_R, u_global, ndof)
        else
            error("[ADJOINT] Unsupported derivative backend '$dv_method' for design variable type '$dv_type'")
        end
    elseif dv_type == "pcomp_ply_thickness" || dv_type == "pcomp_ply_angle"
        if dv_method == :laminate_exact
            return _dKdx_u_pcomp_ply_exact(dv, model, id_map, node_coords, node_R, u_global, ndof)
        elseif dv_method == :clt_fd
            return _dKdx_u_pcomp_ply_fd(dv, model, id_map, node_coords, node_R, u_global, ndof)
        else
            error("[ADJOINT] Unsupported derivative backend '$dv_method' for design variable type '$dv_type'")
        end
    elseif dv_method != DV_REGISTRY[dv_type].method
        error("[ADJOINT] Design variable type '$dv_type' does not support backend override '$dv_method'")
    end

    if haskey(_DKDX_DISPATCH, dv_type)
        return _DKDX_DISPATCH[dv_type](dv, model, id_map, node_coords, node_R, u_global, ndof)
    else
        registered = join(sort(collect(keys(DV_REGISTRY))), ", ")
        error("[ADJOINT] Unsupported design variable type: $dv_type. Registered types: $registered")
    end
end

"""
    get_dv_method(dv) -> Symbol

Return the derivative backend for the design variable, honoring an optional
`dv["method"]` override when present.
"""
function get_dv_method(dv)
    dv_type = dv["type"]
    if !haskey(DV_REGISTRY, dv_type)
        error("[ADJOINT] Unsupported design variable type for method lookup: $dv_type")
    end
    default_method = DV_REGISTRY[dv_type].method
    raw_method = get(dv, "method", default_method)
    if raw_method isa Symbol
        return raw_method
    elseif raw_method isa AbstractString
        return Symbol(raw_method)
    else
        return Symbol(string(raw_method))
    end
end

# ============================================================================
# Shared helper: extract shell element local frame, coords, rotation, DOFs
# ============================================================================
function _shell_elem_local_data(el, model, id_map, node_coords, node_R)
    nids = el["NODES"]
    n_nodes = length(nids)
    if !all(n -> haskey(id_map, n), nids); return nothing; end

    pid_str = string(el["PID"])
    pshells = model["PSHELLs"]
    prop = get(pshells, pid_str, nothing)
    if isnothing(prop); return nothing; end
    mid_str = string(prop["MID"])
    mats = model["MATs"]
    mat = get(mats, mid_str, nothing)
    if isnothing(mat); return nothing; end

    idxs = [id_map[n] for n in nids]
    ndof_elem = n_nodes * 6

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

    T_mat = zeros(ndof_elem, ndof_elem)
    Rel = [v1[1] v1[2] v1[3]; v2[1] v2[2] v2[3]; v3[1] v3[2] v3[3]]
    for k in 1:n_nodes
        TR = Rel * node_R[idxs[k]]
        base = (k-1)*6
        T_mat[base+1:base+3, base+1:base+3] = TR
        T_mat[base+4:base+6, base+4:base+6] = TR
    end

    dofs = Vector{Int}(undef, ndof_elem)
    for k in 1:n_nodes
        base_g = (idxs[k]-1)*6; base_e = (k-1)*6
        for d in 1:6; dofs[base_e+d] = base_g+d; end
    end

    return (n_nodes=n_nodes, lc=lc, T_mat=T_mat, dofs=dofs, ndof_elem=ndof_elem,
            prop=prop, mat=mat, pid_str=pid_str, mid_str=mid_str, idxs=idxs)
end

"""Scatter element contribution T' * f_local to global pseudo-load vector."""
function _scatter_elem_contribution!(pseudo_load, T_mat, dKe_local, u_global, dofs, ndof_elem)
    u_elem = zeros(ndof_elem)
    for i in 1:ndof_elem; u_elem[i] = u_global[dofs[i]]; end
    u_local = T_mat * u_elem
    f_local = dKe_local * u_local
    f_global = T_mat' * f_local
    for i in 1:ndof_elem
        pseudo_load[dofs[i]] += f_global[i]
    end
end

# ============================================================================
# Shell thickness: dK/dh via central FD on element stiffness
# ============================================================================
function _dKdx_u_shell_thickness(dv, model, id_map, node_coords, node_R, u_global, ndof)
    pids = Set(string.(dv["pids"]))
    pseudo_load = zeros(ndof)

    for (_, el) in model["CSHELLs"]
        pid_str = string(el["PID"])
        if !(pid_str in pids); continue; end
        ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
        if isnothing(ed); continue; end

        h = ed.prop["T"]; E = ed.mat["E"]; nu = ed.mat["NU"]
        delta = max(abs(h) * 1e-6, 1e-12)

        if ed.n_nodes == 4
            Ke_plus  = FEM.stiffness_quad4(ed.lc, E, nu, h + delta)
            Ke_minus = FEM.stiffness_quad4(ed.lc, E, nu, h - delta)
        else
            Ke_plus  = FEM.stiffness_tria3(ed.lc, E, nu, h + delta)
            Ke_minus = FEM.stiffness_tria3(ed.lc, E, nu, h - delta)
        end
        dKe_local = (Ke_plus - Ke_minus) / (2.0 * delta)
        _scatter_elem_contribution!(pseudo_load, ed.T_mat, dKe_local, u_global, ed.dofs, ed.ndof_elem)
    end
    return pseudo_load
end

function _shell_thickness_dKe_forward_ad(ed)
    h0 = Float64(ed.prop["T"])
    E = Float64(ed.mat["E"])
    nu = Float64(ed.mat["NU"])
    stiffness_fun = if ed.n_nodes == 4
        x -> vec(FEM.stiffness_quad4_generic(ed.lc, E, nu, x[1]))
    else
        x -> vec(FEM.stiffness_tria3_generic(ed.lc, E, nu, x[1]))
    end
    jac = ForwardDiff.jacobian(stiffness_fun, [h0])
    return reshape(jac[:, 1], ed.ndof_elem, ed.ndof_elem)
end

function _dKdx_u_shell_thickness_ad_forward(dv, model, id_map, node_coords, node_R, u_global, ndof)
    pids = Set(string.(dv["pids"]))
    pseudo_load = zeros(ndof)

    for (_, el) in model["CSHELLs"]
        pid_str = string(el["PID"])
        if !(pid_str in pids); continue; end
        ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
        if isnothing(ed); continue; end

        dKe_local = _shell_thickness_dKe_forward_ad(ed)

        _scatter_elem_contribution!(pseudo_load, ed.T_mat, dKe_local, u_global, ed.dofs, ed.ndof_elem)
    end
    return pseudo_load
end

# ============================================================================
# Material E: dK/dE = K_elem / E (analytical — K proportional to E)
# ============================================================================
function _dKdx_u_material_E(dv, model, id_map, node_coords, node_R, u_global, ndof)
    mids = Set(string.(dv["mids"]))
    pseudo_load = zeros(ndof)

    for (_, el) in model["CSHELLs"]
        ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
        if isnothing(ed); continue; end
        if !(ed.mid_str in mids); continue; end

        h = ed.prop["T"]; E = ed.mat["E"]; nu = ed.mat["NU"]
        Ke_local = ed.n_nodes == 4 ?
            FEM.stiffness_quad4(ed.lc, E, nu, h) :
            FEM.stiffness_tria3(ed.lc, E, nu, h)
        dKe_local = Ke_local / E
        _scatter_elem_contribution!(pseudo_load, ed.T_mat, dKe_local, u_global, ed.dofs, ed.ndof_elem)
    end

    _dKdx_u_material_E_bars!(pseudo_load, mids, model, id_map, node_coords, node_R, u_global, ndof)
    _dKdx_u_material_E_rods!(pseudo_load, mids, model, id_map, node_coords, node_R, u_global)
    _dKdx_u_material_E_solids!(pseudo_load, mids, model, id_map, node_coords, node_R, u_global)
    return pseudo_load
end

# ============================================================================
# Material NU: dK/dnu via central FD on element stiffness
# ============================================================================
function _dKdx_u_material_NU(dv, model, id_map, node_coords, node_R, u_global, ndof)
    mids = Set(string.(dv["mids"]))
    pseudo_load = zeros(ndof)

    for (_, el) in model["CSHELLs"]
        ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
        if isnothing(ed); continue; end
        if !(ed.mid_str in mids); continue; end

        h = ed.prop["T"]; E = ed.mat["E"]; nu = ed.mat["NU"]
        delta = max(abs(nu) * 1e-6, 1e-8)

        if ed.n_nodes == 4
            Ke_plus  = FEM.stiffness_quad4(ed.lc, E, nu + delta, h)
            Ke_minus = FEM.stiffness_quad4(ed.lc, E, nu - delta, h)
        else
            Ke_plus  = FEM.stiffness_tria3(ed.lc, E, nu + delta, h)
            Ke_minus = FEM.stiffness_tria3(ed.lc, E, nu - delta, h)
        end
        dKe_local = (Ke_plus - Ke_minus) / (2.0 * delta)
        _scatter_elem_contribution!(pseudo_load, ed.T_mat, dKe_local, u_global, ed.dofs, ed.ndof_elem)
    end

    _dKdx_u_material_NU_solids!(pseudo_load, mids, model, id_map, node_coords, node_R, u_global)
    return pseudo_load
end

function _material_nu_dKe_forward_ad(ed)
    h = Float64(ed.prop["T"])
    E = Float64(ed.mat["E"])
    nu0 = Float64(ed.mat["NU"])
    stiffness_fun = if ed.n_nodes == 4
        x -> vec(FEM.stiffness_quad4_generic(ed.lc, E, x[1], h))
    else
        x -> vec(FEM.stiffness_tria3_generic(ed.lc, E, x[1], h))
    end
    jac = ForwardDiff.jacobian(stiffness_fun, [nu0])
    return reshape(jac[:, 1], ed.ndof_elem, ed.ndof_elem)
end

function _dKdx_u_material_NU_ad_forward(dv, model, id_map, node_coords, node_R, u_global, ndof)
    mids = Set(string.(dv["mids"]))
    pseudo_load = zeros(ndof)

    for (_, el) in model["CSHELLs"]
        ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
        if isnothing(ed); continue; end
        if !(ed.mid_str in mids); continue; end

        dKe_local = _material_nu_dKe_forward_ad(ed)
        _scatter_elem_contribution!(pseudo_load, ed.T_mat, dKe_local, u_global, ed.dofs, ed.ndof_elem)
    end

    _dKdx_u_material_NU_solids!(pseudo_load, mids, model, id_map, node_coords, node_R, u_global)
    return pseudo_load
end

# ============================================================================
# Bar area: dK/dA via central FD on bar element stiffness
# ============================================================================
function _dKdx_u_bar_area(dv, model, id_map, node_coords, node_R, u_global, ndof)
    pids = Set(string.(dv["pids"]))
    pseudo_load = zeros(ndof)
    pbarls = get(model, "PBARLs", Dict())
    mats = model["MATs"]

    for (_, bar) in get(model, "CBARs", Dict())
        pid_str = string(bar["PID"])
        if !(pid_str in pids); continue; end
        prop = get(pbarls, pid_str, nothing)
        if isnothing(prop); continue; end
        mid_str = string(prop["MID"])
        mat = get(mats, mid_str, nothing)
        if isnothing(mat); continue; end

        if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
        i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]

        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
        _, _, _, p1_eff, p2_eff = bar_offsets_and_endpoints(bar, p1, p2)
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

        T12 = zeros(12, 12)
        TR1 = Rel_t * node_R[i1]; TR2 = Rel_t * node_R[i2]
        T12[1:3,1:3] = TR1; T12[4:6,4:6] = TR1
        T12[7:9,7:9] = TR2; T12[10:12,10:12] = TR2

        E = mat["E"]; G = mat["G"]; A = prop["A"]
        Iy = get(prop, "I2", get(prop, "I", 0.0))
        Iz = get(prop, "I1", get(prop, "I", 0.0))
        if Iy == 0.0; Iy = Iz; end; if Iz == 0.0; Iz = Iy; end
        Iyz = Float64(get(prop, "I12", 0.0))
        K1 = get(prop, "K1", 0.0); K2 = get(prop, "K2", 0.0)

        delta = max(abs(A) * 1e-6, 1e-12)

        # FD on A (affects axial + shear correction terms)
        As_y_p = (K1 > 0.0) ? K1 * (A + delta) : Inf
        As_z_p = (K2 > 0.0) ? K2 * (A + delta) : Inf
        Ke_plus = FEM.stiffness_frame3d(L, A + delta, Iy, Iz, prop["J"], E, G; As_y=As_y_p, As_z=As_z_p, I12=Iyz)

        As_y_m = (K1 > 0.0) ? K1 * (A - delta) : Inf
        As_z_m = (K2 > 0.0) ? K2 * (A - delta) : Inf
        Ke_minus = FEM.stiffness_frame3d(L, A - delta, Iy, Iz, prop["J"], E, G; As_y=As_y_m, As_z=As_z_m, I12=Iyz)

        dKe_loc = (Ke_plus - Ke_minus) / (2.0 * delta)

        dofs = Vector{Int}(undef, 12)
        for d in 1:6; dofs[d] = (i1-1)*6+d; dofs[6+d] = (i2-1)*6+d; end
        _scatter_elem_contribution!(pseudo_load, T12, dKe_loc, u_global, dofs, 12)
    end
    return pseudo_load
end

function _bar_area_dKe_forward_ad(L, A, Iy, Iz, Jtorsion, E, G, K1, K2, Iyz)
    x0 = [Float64(A)]
    jac = ForwardDiff.jacobian(x -> begin
        a = x[1]
        As_y = (K1 > 0.0) ? K1 * a : Inf
        As_z = (K2 > 0.0) ? K2 * a : Inf
        vec(FEM.stiffness_frame3d_generic(L, a, Iy, Iz, Jtorsion, E, G; As_y=As_y, As_z=As_z, I12=Iyz))
    end, x0)
    return reshape(jac[:, 1], 12, 12)
end

function _dKdx_u_bar_area_ad_forward(dv, model, id_map, node_coords, node_R, u_global, ndof)
    pids = Set(string.(dv["pids"]))
    pseudo_load = zeros(ndof)
    pbarls = get(model, "PBARLs", Dict())
    mats = model["MATs"]

    for (_, bar) in get(model, "CBARs", Dict())
        pid_str = string(bar["PID"])
        if !(pid_str in pids); continue; end
        prop = get(pbarls, pid_str, nothing)
        if isnothing(prop); continue; end
        mid_str = string(prop["MID"])
        mat = get(mats, mid_str, nothing)
        if isnothing(mat); continue; end

        if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
        i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]

        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
        _, _, _, p1_eff, p2_eff = bar_offsets_and_endpoints(bar, p1, p2)
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

        T12 = zeros(12, 12)
        TR1 = Rel_t * node_R[i1]; TR2 = Rel_t * node_R[i2]
        T12[1:3,1:3] = TR1; T12[4:6,4:6] = TR1
        T12[7:9,7:9] = TR2; T12[10:12,10:12] = TR2

        E = mat["E"]; G = mat["G"]; A = prop["A"]
        Iy = get(prop, "I2", get(prop, "I", 0.0))
        Iz = get(prop, "I1", get(prop, "I", 0.0))
        if Iy == 0.0; Iy = Iz; end; if Iz == 0.0; Iz = Iy; end
        Iyz = Float64(get(prop, "I12", 0.0))
        K1 = get(prop, "K1", 0.0); K2 = get(prop, "K2", 0.0)

        dKe_loc = _bar_area_dKe_forward_ad(L, A, Iy, Iz, prop["J"], E, G, K1, K2, Iyz)

        dofs = Vector{Int}(undef, 12)
        for d in 1:6; dofs[d] = (i1-1)*6+d; dofs[6+d] = (i2-1)*6+d; end
        _scatter_elem_contribution!(pseudo_load, T12, dKe_loc, u_global, dofs, 12)
    end

    return pseudo_load
end

# ============================================================================
# Bar/beam elements for material_E (K ∝ E)
# ============================================================================
function _dKdx_u_material_E_bars!(pseudo_load, mids, model, id_map, node_coords, node_R, u_global, ndof)
    pbarls = get(model, "PBARLs", Dict())
    mats = model["MATs"]

    for (_, bar) in get(model, "CBARs", Dict())
        pid_str = string(bar["PID"])
        prop = get(pbarls, pid_str, nothing)
        if isnothing(prop); continue; end
        mid_str = string(prop["MID"])
        if !(mid_str in mids); continue; end
        mat = get(mats, mid_str, nothing)
        if isnothing(mat); continue; end

        if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
        i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]

        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
        _, _, _, p1_eff, p2_eff = bar_offsets_and_endpoints(bar, p1, p2)
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

        T12 = zeros(12, 12)
        TR1 = Rel_t * node_R[i1]; TR2 = Rel_t * node_R[i2]
        T12[1:3,1:3] = TR1; T12[4:6,4:6] = TR1
        T12[7:9,7:9] = TR2; T12[10:12,10:12] = TR2

        E = mat["E"]; G = mat["G"]; A = prop["A"]
        Iy = get(prop, "I2", get(prop, "I", 0.0))
        Iz = get(prop, "I1", get(prop, "I", 0.0))
        if Iy == 0.0; Iy = Iz; end; if Iz == 0.0; Iz = Iy; end
        Iyz = Float64(get(prop, "I12", 0.0))
        K1 = get(prop, "K1", 0.0); K2 = get(prop, "K2", 0.0)
        As_y = (K1 > 0.0) ? K1 * A : Inf
        As_z = (K2 > 0.0) ? K2 * A : Inf

        Ke_loc = FEM.stiffness_frame3d(L, A, Iy, Iz, prop["J"], E, G; As_y=As_y, As_z=As_z, I12=Iyz)
        dKe_loc = Ke_loc / E

        dofs = Vector{Int}(undef, 12)
        for d in 1:6; dofs[d] = (i1-1)*6+d; dofs[6+d] = (i2-1)*6+d; end
        _scatter_elem_contribution!(pseudo_load, T12, dKe_loc, u_global, dofs, 12)
    end
end

# ============================================================================
# Rod elements (CROD, CONROD) for material_E  (K ∝ E)
# ============================================================================

function _rod_local_frame_and_transform(i1, i2, node_coords, node_R)
    p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
    p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
    L = norm(p2 - p1)
    if L < 1e-9; return nothing; end
    vx = normalize(p2 - p1)
    ref = abs(vx[3]) < 0.9 ? SVector(0.0,0.0,1.0) : SVector(0.0,1.0,0.0)
    vz = normalize(cross(vx, ref))
    vy = cross(vz, vx)
    Rel_t = vcat(vx', vy', vz')

    T12 = zeros(12, 12)
    TR1 = Rel_t * node_R[i1]; TR2 = Rel_t * node_R[i2]
    T12[1:3,1:3] = TR1; T12[4:6,4:6] = TR1
    T12[7:9,7:9] = TR2; T12[10:12,10:12] = TR2

    dofs = Vector{Int}(undef, 12)
    for d in 1:6; dofs[d] = (i1-1)*6+d; dofs[6+d] = (i2-1)*6+d; end

    return (L=L, T12=T12, dofs=dofs)
end

function _rod_stiffness_inline(L, E, G, A, J)
    Ke = zeros(12, 12)
    EA_L = E * A / L
    Ke[1,1] = EA_L; Ke[1,7] = -EA_L; Ke[7,1] = -EA_L; Ke[7,7] = EA_L
    GJ_L = G * J / L
    Ke[4,4] = GJ_L; Ke[4,10] = -GJ_L; Ke[10,4] = -GJ_L; Ke[10,10] = GJ_L
    return Ke
end

function _dKdx_u_material_E_rods!(pseudo_load, mids, model, id_map, node_coords, node_R, u_global)
    prods = get(model, "PRODs", Dict())
    mats = model["MATs"]

    # CROD elements
    for (_, rod) in get(model, "CRODs", Dict())
        pid_str = string(rod["PID"])
        prop = get(prods, pid_str, nothing)
        if isnothing(prop); continue; end
        mid_str = string(prop["MID"])
        if !(mid_str in mids); continue; end
        mat = get(mats, mid_str, nothing)
        if isnothing(mat); continue; end
        if !haskey(id_map, rod["GA"]) || !haskey(id_map, rod["GB"]); continue; end
        i1, i2 = id_map[rod["GA"]], id_map[rod["GB"]]

        rd = _rod_local_frame_and_transform(i1, i2, node_coords, node_R)
        if isnothing(rd); continue; end

        Ke = _rod_stiffness_inline(rd.L, mat["E"], mat["G"], prop["A"], prop["J"])
        dKe = Ke / mat["E"]  # K ∝ E
        _scatter_elem_contribution!(pseudo_load, rd.T12, dKe, u_global, rd.dofs, 12)
    end

    # CONROD elements
    for (_, rod) in get(model, "CONRODs", Dict())
        mid_str = string(rod["MID"])
        if !(mid_str in mids); continue; end
        mat = get(mats, mid_str, nothing)
        if isnothing(mat); continue; end
        if !haskey(id_map, rod["GA"]) || !haskey(id_map, rod["GB"]); continue; end
        i1, i2 = id_map[rod["GA"]], id_map[rod["GB"]]

        rd = _rod_local_frame_and_transform(i1, i2, node_coords, node_R)
        if isnothing(rd); continue; end

        Ke = _rod_stiffness_inline(rd.L, mat["E"], mat["G"], rod["A"], Float64(get(rod, "J", 0.0)))
        dKe = Ke / mat["E"]
        _scatter_elem_contribution!(pseudo_load, rd.T12, dKe, u_global, rd.dofs, 12)
    end
end

# ============================================================================
# Solid elements (CHEXA, CPENTA, CTETRA) for material_E and material_NU
# ============================================================================

function _solid_elem_data(el, model, id_map, node_coords, node_R)
    nids = el["NODES"]
    nn = length(nids)
    if !all(n -> haskey(id_map, n), nids); return nothing; end

    pid_str = string(el["PID"])
    psolids = get(model, "PSOLIDs", Dict())
    prop = get(psolids, pid_str, nothing)
    if isnothing(prop); return nothing; end
    mid_str = string(prop["MID"])
    mats = model["MATs"]
    mat = get(mats, mid_str, nothing)
    if isnothing(mat); return nothing; end

    idxs = [id_map[n] for n in nids]
    ndof_elem = nn * 3  # translational DOFs only

    # Coordinates in global frame (solids use global coords directly)
    coords = zeros(nn, 3)
    for k in 1:nn
        for d in 1:3; coords[k,d] = node_coords[idxs[k], d]; end
    end

    # Transform: block-diagonal with node_R (translational only)
    T_mat = zeros(ndof_elem, ndof_elem)
    for k in 1:nn
        r = (k-1)*3
        for a in 1:3, b in 1:3
            T_mat[r+a, r+b] = node_R[idxs[k]][a, b]
        end
    end

    # DOF mapping: translational DOFs 1,2,3 of each node
    dofs = Vector{Int}(undef, ndof_elem)
    for k in 1:nn
        base = (idxs[k]-1)*6
        for d in 1:3; dofs[(k-1)*3+d] = base+d; end
    end

    etype = get(el, "TYPE", "")
    return (nn=nn, coords=coords, T_mat=T_mat, dofs=dofs, ndof_elem=ndof_elem,
            mat=mat, mid_str=mid_str, etype=etype)
end

function _solid_stiffness(etype, nn, coords, E, nu)
    if etype == "CTETRA" && nn == 4
        return FEM.stiffness_tetra4(coords, E, nu)
    elseif etype == "CHEXA" && nn == 8
        return FEM.stiffness_hexa8(coords, E, nu)
    elseif etype == "CPENTA" && nn == 6
        return FEM.stiffness_cpenta6(coords, E, nu)
    else
        return nothing
    end
end

function _dKdx_u_material_E_solids!(pseudo_load, mids, model, id_map, node_coords, node_R, u_global)
    for (_, el) in get(model, "CSOLIDs", Dict())
        sd = _solid_elem_data(el, model, id_map, node_coords, node_R)
        if isnothing(sd); continue; end
        if !(sd.mid_str in mids); continue; end

        E = sd.mat["E"]; nu = sd.mat["NU"]
        Ke = _solid_stiffness(sd.etype, sd.nn, sd.coords, E, nu)
        if isnothing(Ke); continue; end
        dKe = Ke / E  # K ∝ E for isotropic solids
        _scatter_elem_contribution!(pseudo_load, sd.T_mat, dKe, u_global, sd.dofs, sd.ndof_elem)
    end
end

function _dKdx_u_material_NU_solids!(pseudo_load, mids, model, id_map, node_coords, node_R, u_global)
    for (_, el) in get(model, "CSOLIDs", Dict())
        sd = _solid_elem_data(el, model, id_map, node_coords, node_R)
        if isnothing(sd); continue; end
        if !(sd.mid_str in mids); continue; end

        E = sd.mat["E"]; nu = sd.mat["NU"]
        delta = max(abs(nu) * 1e-6, 1e-8)

        Ke_plus = _solid_stiffness(sd.etype, sd.nn, sd.coords, E, nu + delta)
        Ke_minus = _solid_stiffness(sd.etype, sd.nn, sd.coords, E, nu - delta)
        if isnothing(Ke_plus) || isnothing(Ke_minus); continue; end

        dKe = (Ke_plus - Ke_minus) / (2.0 * delta)
        _scatter_elem_contribution!(pseudo_load, sd.T_mat, dKe, u_global, sd.dofs, sd.ndof_elem)
    end
end

# ============================================================================
# PCOMP ply: exact laminate derivatives with CLT FD fallback
# ============================================================================

function _qbar_plane_stress(E1, E2, nu12, G12, theta)
    T = promote_type(typeof(E1), typeof(E2), typeof(nu12), typeof(G12), typeof(theta))
    nu21 = nu12 * E2 / max(E1, T(1e-30))
    denom = one(T) - nu12 * nu21
    Q11 = E1 / denom
    Q22 = E2 / denom
    Q12 = nu12 * E2 / denom
    Q66 = G12
    c = cos(theta)
    s = sin(theta)
    c2 = c^2
    s2 = s^2
    cs = c * s
    Qb = zeros(T, 3, 3)
    Qb[1,1] = Q11*c2^2 + 2*(Q12 + 2*Q66)*c2*s2 + Q22*s2^2
    Qb[2,2] = Q11*s2^2 + 2*(Q12 + 2*Q66)*c2*s2 + Q22*c2^2
    Qb[1,2] = (Q11 + Q22 - 4*Q66)*c2*s2 + Q12*(c2^2 + s2^2)
    Qb[2,1] = Qb[1,2]
    Qb[1,3] = (Q11 - Q12 - 2*Q66)*cs*c2 + (Q12 - Q22 + 2*Q66)*cs*s2
    Qb[3,1] = Qb[1,3]
    Qb[2,3] = (Q11 - Q12 - 2*Q66)*cs*s2 + (Q12 - Q22 + 2*Q66)*cs*c2
    Qb[3,2] = Qb[2,3]
    Qb[3,3] = (Q11 + Q22 - 2*Q12 - 2*Q66)*c2*s2 + Q66*(c2^2 + s2^2)
    return Qb
end

function _qbar_shear(G13, G23, theta)
    T = promote_type(typeof(G13), typeof(G23), typeof(theta))
    c = cos(theta)
    s = sin(theta)
    Qs = zeros(T, 2, 2)
    Qs[1,1] = c^2 * G13 + s^2 * G23
    Qs[1,2] = c * s * (G13 - G23)
    Qs[2,1] = Qs[1,2]
    Qs[2,2] = s^2 * G13 + c^2 * G23
    return Qs
end

function _pcomp_ply_material_data(ply, mats)
    mid_raw = get(ply, "mid", get(ply, "MID", nothing))
    isnothing(mid_raw) && return nothing
    pm = get(mats, string(Int(mid_raw)), nothing)
    isnothing(pm) && return nothing

    if haskey(pm, "E1")
        E1 = Float64(pm["E1"])
        E2 = Float64(pm["E2"])
        nu12 = Float64(pm["NU12"])
        G12 = Float64(pm["G12"])
        G13 = Float64(get(pm, "G1Z", G12))
        G23 = Float64(get(pm, "G2Z", G12))
        G13 <= 0.0 && (G13 = G12)
        G23 <= 0.0 && (G23 = G12)
    else
        E1 = Float64(pm["E"])
        E2 = E1
        nu12 = Float64(pm["NU"])
        G12 = Float64(pm["G"])
        G13 = G12
        G23 = G12
    end

    return (E1=E1, E2=E2, nu12=nu12, G12=G12, G13=G13, G23=G23)
end

function _pcomp_ply_dQ_dtheta(ply, mats)
    pdata = _pcomp_ply_material_data(ply, mats)
    isnothing(pdata) && return nothing
    theta0 = deg2rad(Float64(ply["theta"]))

    qbar_jac = ForwardDiff.jacobian(
        x -> vec(_qbar_plane_stress(pdata.E1, pdata.E2, pdata.nu12, pdata.G12, x[1])),
        [theta0],
    )
    qshear_jac = ForwardDiff.jacobian(
        x -> vec(_qbar_shear(pdata.G13, pdata.G23, x[1])),
        [theta0],
    )

    dQb = reshape(qbar_jac[:, 1], 3, 3)
    dQs = reshape(qshear_jac[:, 1], 2, 2)
    return dQb, dQs
end

function _pcomp_exact_constitutive_derivative(prop, mats, ply_idx::Int, perturb_field::Symbol)
    ply_data = get(prop, "PLY_DATA", nothing)
    isnothing(ply_data) && return nothing
    if ply_idx < 1 || ply_idx > length(ply_data)
        return nothing
    end

    dA = zeros(3, 3)
    dB = zeros(3, 3)
    dD = zeros(3, 3)
    dAsh = zeros(2, 2)

    for (j, ply) in enumerate(ply_data)
        z_bot = Float64(ply["z_bot"])
        z_top = Float64(ply["z_top"])
        t = z_top - z_bot

        if perturb_field == :T
            Qb = Float64.(ply["Qbar"])
            Qs = Float64.(ply["Qshear"])
            dz_bot = j <= ply_idx ? -0.5 : 0.5
            dz_top = j < ply_idx ? -0.5 : 0.5

            if j == ply_idx
                dA .+= Qb
                dAsh .+= Qs
            end
            dB .+= Qb .* (z_top * dz_top - z_bot * dz_bot)
            dD .+= Qb .* (z_top^2 * dz_top - z_bot^2 * dz_bot)
        else
            j == ply_idx || continue
            dQ = _pcomp_ply_dQ_dtheta(ply, mats)
            isnothing(dQ) && return nothing
            dQb, dQs = dQ
            dA .+= dQb .* t
            dB .+= dQb .* ((z_top^2 - z_bot^2) / 2.0)
            dD .+= dQb .* ((z_top^3 - z_bot^3) / 3.0)
            dAsh .+= dQs .* t
        end
    end

    dCs = (5.0 / 6.0) .* dAsh
    dBmb = maximum(abs.(dB); init=0.0) > 0.0 ? dB : nothing
    return dA, dBmb, dD, dCs
end

"""
Recompute CLT matrices (A, B, D, Cs) from PLY_DATA with optional perturbation.
Uses the stored per-ply Qbar/z data in the PCOMP_CLT property dict.
`perturb_ply` = ply index to perturb (1-based), `perturb_field` = :T or :THETA,
`perturb_delta` = delta to add to the field.
"""
function _recompute_clt(prop, mats; perturb_ply::Int=0, perturb_field::Symbol=:T, perturb_delta::Float64=0.0)
    ply_data = prop["PLY_DATA"]
    n_plies = length(ply_data)

    ply_t = [Float64(ply_data[k]["z_top"] - ply_data[k]["z_bot"]) for k in 1:n_plies]
    ply_theta = [deg2rad(Float64(ply_data[k]["theta"])) for k in 1:n_plies]

    if perturb_ply > 0 && perturb_ply <= n_plies
        if perturb_field == :T
            ply_t[perturb_ply] += perturb_delta
        elseif perturb_field == :THETA
            ply_theta[perturb_ply] += perturb_delta
        end
    end

    total_t = sum(ply_t)
    z0 = -total_t / 2.0

    A = zeros(3, 3)
    B = zeros(3, 3)
    D = zeros(3, 3)
    Ash = zeros(2, 2)
    z_bot = z0

    for k in 1:n_plies
        t = ply_t[k]
        theta = ply_theta[k]
        z_top = z_bot + t

        if perturb_field == :THETA && k == perturb_ply
            pdata = _pcomp_ply_material_data(ply_data[k], mats)
            if !isnothing(pdata)
                Qb = _qbar_plane_stress(pdata.E1, pdata.E2, pdata.nu12, pdata.G12, theta)
                Qs = _qbar_shear(pdata.G13, pdata.G23, theta)
            else
                Qb = ply_data[k]["Qbar"]
                Qs = ply_data[k]["Qshear"]
            end
        else
            Qb = ply_data[k]["Qbar"]
            Qs = ply_data[k]["Qshear"]
        end

        A .+= Qb .* (z_top - z_bot)
        B .+= Qb .* (z_top^2 - z_bot^2) / 2.0
        D .+= Qb .* (z_top^3 - z_bot^3) / 3.0
        Ash .+= Qs .* (z_top - z_bot)
        z_bot = z_top
    end

    Cs = (5.0 / 6.0) .* Ash
    Bmb = maximum(abs.(B)) > 1e-10 * maximum(abs.(A); init=1.0) ? B : nothing
    return A, Bmb, D, Cs
end

function _dKdx_u_pcomp_ply_exact(dv, model, id_map, node_coords, node_R, u_global, ndof)
    dv_type = dv["type"]
    pids = Set(string.(dv["pids"]))
    ply_idx = Int(dv["ply_index"])
    perturb_field = dv_type == "pcomp_ply_thickness" ? :T : :THETA

    pshells = model["PSHELLs"]
    mats = model["MATs"]
    pseudo_load = zeros(ndof)

    for (_, el) in model["CSHELLs"]
        pid_str = string(el["PID"])
        pid_str in pids || continue
        prop = get(pshells, pid_str, nothing)
        if isnothing(prop) || get(prop, "TYPE", "") != "PCOMP_CLT"
            continue
        end

        clt_deriv = _pcomp_exact_constitutive_derivative(prop, mats, ply_idx, perturb_field)
        if isnothing(clt_deriv)
            return _dKdx_u_pcomp_ply_fd(dv, model, id_map, node_coords, node_R, u_global, ndof)
        end

        ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
        isnothing(ed) && continue

        dCm, dBmb, dCb, dCs = clt_deriv
        h = Float64(prop["T"])
        E_ref = Float64(get(prop, "E_ref", 1.0))
        dKe = ed.n_nodes == 4 ?
            FEM.stiffness_quad4_matrices(ed.lc, dCm, dCb, dCs, h, E_ref; Bmb=dBmb) :
            FEM.stiffness_tria3_matrices(ed.lc, dCm, dCb, dCs, h, E_ref; Bmb=dBmb)

        _scatter_elem_contribution!(pseudo_load, ed.T_mat, dKe, u_global, ed.dofs, ed.ndof_elem)
    end

    return pseudo_load
end

function _dKdx_u_pcomp_ply_fd(dv, model, id_map, node_coords, node_R, u_global, ndof)
    dv_type = dv["type"]
    pids = Set(string.(dv["pids"]))
    ply_idx = Int(dv["ply_index"])
    perturb_field = dv_type == "pcomp_ply_thickness" ? :T : :THETA

    pshells = model["PSHELLs"]
    mats = model["MATs"]
    pseudo_load = zeros(ndof)

    for (_, el) in model["CSHELLs"]
        pid_str = string(el["PID"])
        if !(pid_str in pids); continue; end
        prop = get(pshells, pid_str, nothing)
        if isnothing(prop); continue; end
        if get(prop, "TYPE", "") != "PCOMP_CLT"; continue; end
        if !haskey(prop, "PLY_DATA") || ply_idx > length(prop["PLY_DATA"]); continue; end

        ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
        if isnothing(ed); continue; end

        ply = prop["PLY_DATA"][ply_idx]
        if perturb_field == :T
            t_ply = Float64(ply["z_top"] - ply["z_bot"])
            delta = max(abs(t_ply) * 1e-6, 1e-12)
        else
            theta_ply = deg2rad(Float64(ply["theta"]))
            delta = max(abs(theta_ply) * 1e-6, 1e-6)
        end

        Cm_p, Bmb_p, Cb_p, Cs_p = _recompute_clt(prop, mats; perturb_ply=ply_idx, perturb_field=perturb_field, perturb_delta=delta)
        Cm_m, Bmb_m, Cb_m, Cs_m = _recompute_clt(prop, mats; perturb_ply=ply_idx, perturb_field=perturb_field, perturb_delta=-delta)

        h = Float64(prop["T"])
        E_ref = Float64(get(prop, "E_ref", 1.0))
        if ed.n_nodes == 4
            Ke_p = FEM.stiffness_quad4_matrices(ed.lc, Cm_p, Cb_p, Cs_p, h, E_ref; Bmb=Bmb_p)
            Ke_m = FEM.stiffness_quad4_matrices(ed.lc, Cm_m, Cb_m, Cs_m, h, E_ref; Bmb=Bmb_m)
        else
            Ke_p = FEM.stiffness_tria3_matrices(ed.lc, Cm_p, Cb_p, Cs_p, h, E_ref; Bmb=Bmb_p)
            Ke_m = FEM.stiffness_tria3_matrices(ed.lc, Cm_m, Cb_m, Cs_m, h, E_ref; Bmb=Bmb_m)
        end

        dKe = (Ke_p - Ke_m) / (2.0 * delta)
        _scatter_elem_contribution!(pseudo_load, ed.T_mat, dKe, u_global, ed.dofs, ed.ndof_elem)
    end

    return pseudo_load
end

function _dKdx_u_pcomp_ply(dv, model, id_map, node_coords, node_R, u_global, ndof)
    return _dKdx_u_pcomp_ply_exact(dv, model, id_map, node_coords, node_R, u_global, ndof)
end

# ============================================================================
# Node coordinate: dK/dx via full-model reassembly FD
# ============================================================================

"""
Compute dK/dx · u where x is a node coordinate (grid, component 1/2/3).
Uses full model reassembly FD: perturb the GRID coordinate, call assemble_stiffness,
compute K_perturbed · u, then central difference.
"""
function _dKdx_u_node_coord(dv, model, u_global, ndof)
    grid = Int(dv["grid"])
    comp = Int(dv["comp"])  # 1=x, 2=y, 3=z
    grid_str = string(grid)

    if !haskey(model["GRIDs"], grid_str)
        error("[ADJOINT] Grid $grid not found in model for node_coord DV")
    end

    coords = model["GRIDs"][grid_str]["X"]
    x0 = Float64(coords[comp])
    delta = max(abs(x0) * 1e-6, 1e-8)

    # Perturb +
    coords[comp] = x0 + delta
    K_plus, = assemble_stiffness(model)
    Ku_plus = K_plus * u_global

    # Perturb -
    coords[comp] = x0 - delta
    K_minus, = assemble_stiffness(model)
    Ku_minus = K_minus * u_global

    # Restore
    coords[comp] = x0

    return (Ku_plus - Ku_minus) / (2.0 * delta)
end

# ============================================================================
# Topology density: dK/dρ = p · ρ^(p-1) · K_e0  (SIMP analytical)
# ============================================================================

"""
Compute dK/dρ · u for topology density design variables (SIMP model).

DV config:
  "type": "topology_density"
  "eids": [1, 2, 3, ...]       — element IDs with density variables
  "densities": {"1": 0.5, ...} — current density per element (ρ ∈ [ρ_min, 1])
  "penalization": 3.0           — SIMP penalization exponent p (default 3)
  "rho_min": 1e-3               — minimum density to avoid singularity (default 1e-3)

SIMP model: K_e(ρ) = ρ^p · K_e0
Derivative:  dK_e/dρ = p · ρ^(p-1) · K_e0 = (p/ρ) · K_e(ρ)

Since K_e(ρ) is the current element stiffness (as assembled), we just need
dK_e/dρ = (p/ρ) · K_e_current. No need to recover K_e0 separately.
"""
function _dKdx_u_topology_density(dv, model, id_map, node_coords, node_R, u_global, ndof)
    eids = [Int(e) for e in dv["eids"]]
    densities = dv["densities"]
    p = Float64(get(dv, "penalization", 3.0))
    pseudo_load = zeros(ndof)

    pshells = model["PSHELLs"]
    mats = model["MATs"]

    for eid in eids
        eid_str = string(eid)
        rho = Float64(get(densities, eid_str, 1.0))

        # Shell elements
        if haskey(model["CSHELLs"], eid_str)
            el = model["CSHELLs"][eid_str]
            ed = _shell_elem_local_data(el, model, id_map, node_coords, node_R)
            if isnothing(ed); continue; end

            h = ed.prop["T"]; E = ed.mat["E"]; nu = ed.mat["NU"]
            is_pcomp = get(ed.prop, "TYPE", "") == "PCOMP_CLT"

            # Compute K_e0 at full density (ρ=1)
            if is_pcomp && haskey(ed.prop, "Cm")
                Ke0 = ed.n_nodes == 4 ?
                    FEM.stiffness_quad4_matrices(ed.lc, ed.prop["Cm"], ed.prop["Cb"], ed.prop["Cs"],
                        h, get(ed.prop, "E_ref", E); Bmb=get(ed.prop, "Bmb", nothing)) :
                    FEM.stiffness_tria3_matrices(ed.lc, ed.prop["Cm"], ed.prop["Cb"], ed.prop["Cs"],
                        h, get(ed.prop, "E_ref", E); Bmb=get(ed.prop, "Bmb", nothing))
            else
                Ke0 = ed.n_nodes == 4 ?
                    FEM.stiffness_quad4(ed.lc, E, nu, h) :
                    FEM.stiffness_tria3(ed.lc, E, nu, h)
            end

            # dK_e/dρ = (p/ρ) · K_e_current  (K_e_current already includes ρ^p scaling)
            dKe = (p / max(rho, 1e-30)) * Ke0
            _scatter_elem_contribution!(pseudo_load, ed.T_mat, dKe, u_global, ed.dofs, ed.ndof_elem)
            continue
        end

        # Bar elements
        for (_, bar) in get(model, "CBARs", Dict())
            if string(get(bar, "ID", -1)) != eid_str; continue; end
            pid_str = string(bar["PID"])
            prop = get(get(model, "PBARLs", Dict()), pid_str, nothing)
            if isnothing(prop); continue; end
            mat = get(mats, string(prop["MID"]), nothing)
            if isnothing(mat); continue; end
            if !haskey(id_map, bar["GA"]) || !haskey(id_map, bar["GB"]); continue; end
            i1, i2 = id_map[bar["GA"]], id_map[bar["GB"]]
            rd = _rod_local_frame_and_transform(i1, i2, node_coords, node_R)
            if isnothing(rd); continue; end

            E = mat["E"]; G = mat["G"]; A = prop["A"]
            Iy = get(prop, "I2", get(prop, "I", 0.0))
            Iz = get(prop, "I1", get(prop, "I", 0.0))
            if Iy == 0.0; Iy = Iz; end; if Iz == 0.0; Iz = Iy; end
            Iyz = Float64(get(prop, "I12", 0.0))
            K1 = get(prop, "K1", 0.0); K2 = get(prop, "K2", 0.0)
            As_y = (K1 > 0.0) ? K1 * A : Inf; As_z = (K2 > 0.0) ? K2 * A : Inf

            Ke0 = FEM.stiffness_frame3d(rd.L, A, Iy, Iz, prop["J"], E, G; As_y=As_y, As_z=As_z, I12=Iyz)
            dKe = p * rho^(p - 1) * Ke0
            dofs = Vector{Int}(undef, 12)
            for d in 1:6; dofs[d] = (i1-1)*6+d; dofs[6+d] = (i2-1)*6+d; end
            _scatter_elem_contribution!(pseudo_load, rd.T12, dKe, u_global, dofs, 12)
            break
        end

        # Rod elements
        prods = get(model, "PRODs", Dict())
        for (_, rod) in get(model, "CRODs", Dict())
            if string(get(rod, "ID", -1)) != eid_str; continue; end
            prop = get(prods, string(rod["PID"]), nothing)
            if isnothing(prop); continue; end
            mat = get(mats, string(prop["MID"]), nothing)
            if isnothing(mat); continue; end
            if !haskey(id_map, rod["GA"]) || !haskey(id_map, rod["GB"]); continue; end
            i1, i2 = id_map[rod["GA"]], id_map[rod["GB"]]
            rd = _rod_local_frame_and_transform(i1, i2, node_coords, node_R)
            if isnothing(rd); continue; end

            Ke0 = _rod_stiffness_inline(rd.L, mat["E"], mat["G"], prop["A"], prop["J"])
            dKe = p * rho^(p - 1) * Ke0
            _scatter_elem_contribution!(pseudo_load, rd.T12, dKe, u_global, rd.dofs, 12)
            break
        end

        # Solid elements
        for (_, el) in get(model, "CSOLIDs", Dict())
            if string(get(el, "ID", -1)) != eid_str; continue; end
            sd = _solid_elem_data(el, model, id_map, node_coords, node_R)
            if isnothing(sd); continue; end
            Ke0 = _solid_stiffness(sd.etype, sd.nn, sd.coords, sd.mat["E"], sd.mat["NU"])
            if isnothing(Ke0); continue; end
            dKe = p * rho^(p - 1) * Ke0
            _scatter_elem_contribution!(pseudo_load, sd.T_mat, dKe, u_global, sd.dofs, sd.ndof_elem)
            break
        end
    end

    return pseudo_load
end

# ============================================================================
# Per-group decomposition (separate sensitivity per PID or MID)
# ============================================================================

"""
    compute_dKdx_u_per_group(dv, model, id_map, node_coords, node_R, u_global, ndof)

Returns Dict{String, Vector{Float64}} with separate pseudo-load per group (PID/MID/EID/GRID).
Uses the DV_REGISTRY to determine group key field and prefix.
"""
function compute_dKdx_u_per_group(dv, model, id_map, node_coords, node_R, u_global, ndof)
    dv_type = dv["type"]
    if !haskey(DV_REGISTRY, dv_type)
        error("[ADJOINT] Unsupported DV type for per-group: $dv_type")
    end
    reg = DV_REGISTRY[dv_type]

    # Single-variable types (no group key field)
    if reg.key_field == ""
        label = _single_dv_label(dv)
        return Dict(label => compute_dKdx_u(dv, model, id_map, node_coords, node_R, u_global, ndof))
    end

    # Multi-group types: split by group key and compute each separately
    return _per_group_by_key(dv, reg.key_field, reg.prefix, dv_type, model, id_map, node_coords, node_R, u_global, ndof)
end

"""Generate label for single-variable DV types (node_coord, etc.)."""
function _single_dv_label(dv)
    dv_type = dv["type"]
    if dv_type == "node_coord"
        return "GRID_$(Int(dv["grid"]))_$(Int(dv["comp"]))"
    else
        return dv["id"]
    end
end

function _per_group_by_key(dv, key_field, prefix, dv_type, model, id_map, node_coords, node_R, u_global, ndof)
    ids = dv[key_field]
    result = Dict{String, Vector{Float64}}()
    for id_val in ids
        id_str = string(Int(id_val))
        # Copy the full DV dict, overriding type and narrowing the group key
        dv_single = copy(dv)
        dv_single["type"] = dv_type
        dv_single[key_field] = [Int(id_val)]
        result["$(prefix)_$id_str"] = compute_dKdx_u(dv_single, model, id_map, node_coords, node_R, u_global, ndof)
    end
    return result
end
