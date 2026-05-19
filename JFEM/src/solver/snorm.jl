# snorm.jl — SNORM normal averaging for shell elements

function compute_shell_nodal_normals(model, id_map, node_coords, angle_deg::Real; label::Union{Nothing,String}=nothing)
    angle = Float64(angle_deg)
    node_nrms = Dict{Int, Vector{SVector{3, Float64}}}()

    for (id, el) in model["CSHELLs"]
        if !haskey(el, "NODES"); continue; end
        nids = el["NODES"]
        n = length(nids)
        if n < 3; continue; end

        if !haskey(id_map, nids[1]) || !haskey(id_map, nids[2]) || !haskey(id_map, nids[3]); continue; end
        i1, i2, i3 = id_map[nids[1]], id_map[nids[2]], id_map[nids[3]]

        p1 = SVector{3}(node_coords[i1,1], node_coords[i1,2], node_coords[i1,3])
        p2 = SVector{3}(node_coords[i2,1], node_coords[i2,2], node_coords[i2,3])
        p3 = SVector{3}(node_coords[i3,1], node_coords[i3,2], node_coords[i3,3])

        if n == 4
            if !haskey(id_map, nids[4]); continue; end
            i4 = id_map[nids[4]]
            p4 = SVector{3}(node_coords[i4,1], node_coords[i4,2], node_coords[i4,3])
            nrm_raw = cross(p3 - p1, p4 - p2)
        else
            nrm_raw = cross(p2 - p1, p3 - p1)
        end
        nrm_len = norm(nrm_raw)
        if nrm_len < 1e-30; continue; end
        nrm = SVector{3}(nrm_raw / nrm_len)

        for nid in nids
             if !haskey(id_map, nid); continue; end
             idx = id_map[nid]
             if !haskey(node_nrms, idx); node_nrms[idx] = SVector{3,Float64}[]; end
             push!(node_nrms[idx], nrm)
        end
    end

    avg_normals = Dict{Int, SVector{3,Float64}}()
    cos_chk = cosd(angle)

    for (idx, nrms) in node_nrms
        if length(nrms) == 1
            avg_normals[idx] = nrms[1]
            continue
        end

        avg = sum(nrms)
        len = norm(avg)
        if len < 1e-12; continue; end
        avg = avg / len

        ok = true
        for n in nrms
            if dot(n, avg) < cos_chk; ok = false; break; end
        end

        if ok
            avg_normals[idx] = SVector{3}(avg...)
        end
    end
    if !isnothing(label)
        log_msg("[SOLVER] $label: $(length(avg_normals)) nodes with averaged normals")
    end
    return avg_normals
end

# Returns Dict{Int, SVector{3,Float64}} of averaged normals per node index.
# Does NOT modify node_R (DOF coordinate frames must stay as-is for constraints/RBE3).
# The averaged normals are used to adjust element frames in the assembly loop.
function compute_snorm_normals(model, id_map, node_coords)
    snorm_angle = get(model, "PARAM_SNORM", 0.0)
    if haskey(ENV, "JFEM_PARAM_SNORM_OVERRIDE")
        snorm_angle = something(tryparse(Float64, ENV["JFEM_PARAM_SNORM_OVERRIDE"]), snorm_angle)
    end
    if snorm_angle <= 0.0
        log_msg("[SOLVER] SNORM disabled (angle=$snorm_angle)")
        return Dict{Int, SVector{3,Float64}}()
    end

    return compute_shell_nodal_normals(model, id_map, node_coords, snorm_angle; label="SNORM")
end

function compute_geometric_nodal_normals(model, id_map, node_coords)
    raw = strip(get(ENV, "JFEM_KG_CURVATURE_NORMAL_ANGLE", "180.0"))
    angle_deg = something(tryparse(Float64, raw), 180.0)
    angle_deg <= 0.0 && return Dict{Int, SVector{3,Float64}}()
    return compute_shell_nodal_normals(model, id_map, node_coords, angle_deg)
end

# Apply SNORM: adjust element frame using averaged normals at element nodes.
# Returns modified (v1, v2, v3) with v3 tilted toward averaged surface normal.
function apply_snorm_to_frame(v1::SVector{3,Float64}, v2::SVector{3,Float64}, v3::SVector{3,Float64},
                               indices::Vector{Int}, snorm_normals::Dict{Int, SVector{3,Float64}})
    if isempty(snorm_normals); return v1, v2, v3; end

    n_avg = SVector(0.0, 0.0, 0.0)
    n_count = 0
    for idx in indices
        if haskey(snorm_normals, idx)
            n_avg = n_avg + snorm_normals[idx]
            n_count += 1
        end
    end
    if n_count == 0; return v1, v2, v3; end

    n_avg = n_avg / n_count
    len = norm(n_avg)
    if len < 1e-12; return v1, v2, v3; end
    v3_new = SVector{3}(n_avg / len)

    # Ensure v3_new is on same side as original v3
    if dot(v3_new, v3) < 0.0; v3_new = -v3_new; end

    # Recompute v1 by projecting original v1 onto plane perpendicular to new v3
    v1_proj = v1 - dot(v1, v3_new) * v3_new
    v1_len = norm(v1_proj)
    if v1_len > 1e-12
        v1_new = SVector{3}(v1_proj / v1_len)
    else
        # v1 was parallel to new v3, use v2 instead
        v2_proj = v2 - dot(v2, v3_new) * v3_new
        v1_new = SVector{3}(normalize(v2_proj))
    end
    v2_new = SVector{3}(cross(v3_new, v1_new))

    return v1_new, v2_new, v3_new
end
