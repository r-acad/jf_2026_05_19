# ============================================================================
# Export.jl — Export functions for JFEM
#
# This file is included at the top level (not a module) and has access to
# WriteVTK, JSON, and HDF5 packages from the parent scope.
#
# Contains:
#   export_hdf5                   - recursive HDF5 export for solution payloads
#   build_solution_hdf5_payload   - builds aggregated HDF5 payloads for SOL101/106
#   build_modal_hdf5_payload      - builds modal HDF5 payloads for SOL103
#   build_buckling_hdf5_payload   - builds buckling HDF5 payloads for SOL105
#   sanitize!(d)                  — NaN/Inf cleaner for JSON export
#   build_jfem_element_tables     — builds sorted element tables for JFEM binary
#   build_jfem_constraint_tables  — builds CELAS/RBE2/RBE3 tables for JFEM v3
#   collect_spc_data              — collects SPC constraints for a subcase
#   collect_point_loads           — collects FORCE/MOMENT cards for a subcase
#   collect_jfem_subcase_data     — collects per-subcase JFEM binary data
#   export_vtk_subcase            — exports a single subcase to VTK format
#   export_json                   — exports aggregated results to JSON
#   export_jfem_binary            — exports mesh + results to JFEM binary format (v4)
#   export_card_inventory         — exports card inventory to JSON
# ============================================================================

function sanitize!(d)
    if d isa Dict
        for (k,v) in d; d[k] = sanitize!(v); end
    elseif d isa Vector
        for i in eachindex(d); d[i] = sanitize!(d[i]); end
    elseif d isa Float64
        if isnan(d) || isinf(d); return 0.0; end
    end
    return d
end

@inline _export_base_name(filename) = replace(basename(filename), r"(?i)\.bdf$" => "")

@inline function _export_entry_public_id(key, entry)
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

@inline function _hdf5_is_scalar(value)
    return value isa Number || value isa AbstractString || value isa Bool || value isa Char
end

@inline function _hdf5_name(name::AbstractString)
    sanitized = replace(String(name), "/" => "__slash__")
    return isempty(sanitized) ? "__empty__" : sanitized
end

function _hdf5_write_leaf(parent, name::AbstractString, value)
    dataset_name = _hdf5_name(name)
    if value isa Char
        write(parent, dataset_name, string(value))
    elseif value isa AbstractString
        write(parent, dataset_name, String(value))
    else
        write(parent, dataset_name, value)
    end
end

function _hdf5_write_array(parent, name::AbstractString, value::AbstractArray)
    dataset_name = _hdf5_name(name)

    if isempty(value)
        group = create_group(parent, dataset_name)
        attrs = attributes(group)
        attrs["jfem_kind"] = "empty_array"
        attrs["ndims"] = Int32(ndims(value))
        attrs["eltype"] = string(eltype(value))
        return
    end

    if eltype(value) <: Number || eltype(value) <: Bool
        write(parent, dataset_name, value)
        return
    end

    if eltype(value) <: AbstractString
        write(parent, dataset_name, String.(value))
        return
    end

    if value isa AbstractVector && all(_hdf5_is_scalar, value)
        if all(v -> v isa Bool, value)
            write(parent, dataset_name, Bool.(value))
        elseif all(v -> v isa Integer, value)
            write(parent, dataset_name, Int64.(value))
        elseif all(v -> v isa Real, value)
            write(parent, dataset_name, Float64.(value))
        elseif all(v -> v isa AbstractString || v isa Char, value)
            write(parent, dataset_name, string.(value))
        else
            group = create_group(parent, dataset_name)
            attrs = attributes(group)
            attrs["jfem_kind"] = "scalar_vector_group"
            attrs["length"] = Int32(length(value))
            for (idx, item) in enumerate(value)
                _hdf5_write_leaf(group, @sprintf("item_%06d", idx), string(item))
            end
        end
        return
    end

    group = create_group(parent, dataset_name)
    attrs = attributes(group)
    attrs["jfem_kind"] = "array_group"
    attrs["length"] = Int32(length(value))
    attrs["ndims"] = Int32(ndims(value))
    attrs["eltype"] = string(eltype(value))

    if value isa AbstractVector
        for (idx, item) in enumerate(value)
            _hdf5_write_value(group, @sprintf("item_%06d", idx), item)
        end
    else
        for index in CartesianIndices(value)
            name_part = "item_" * join(Tuple(index), "_")
            _hdf5_write_value(group, name_part, value[index])
        end
    end
end

function _hdf5_write_value(parent, name::AbstractString, value)
    if value === nothing
        group = create_group(parent, _hdf5_name(name))
        attributes(group)["jfem_kind"] = "nothing"
    elseif value isa AbstractDict
        group = create_group(parent, _hdf5_name(name))
        attrs = attributes(group)
        attrs["jfem_kind"] = "dict"
        for key in sort(collect(keys(value)); by=x -> string(x))
            _hdf5_write_value(group, string(key), value[key])
        end
    elseif value isa NamedTuple
        _hdf5_write_value(parent, name, Dict(string(k) => getfield(value, k) for k in keys(value)))
    elseif value isa Tuple
        _hdf5_write_value(parent, name, collect(value))
    elseif value isa AbstractArray
        _hdf5_write_array(parent, name, value)
    elseif _hdf5_is_scalar(value)
        _hdf5_write_leaf(parent, name, value)
    else
        group = create_group(parent, _hdf5_name(name))
        attrs = attributes(group)
        attrs["jfem_kind"] = "string_fallback"
        write(group, "value", string(value))
    end
end

function export_hdf5(filename, output_dir, payload; suffix, label)
    h5_path = joinpath(output_dir, _export_base_name(filename) * suffix)
    println("\n>>> Exporting $label: $h5_path")
    h5open(h5_path, "w") do file
        attrs = attributes(file)
        attrs["openjfem_export_format"] = "HDF5"
        attrs["source_filename"] = basename(filename)
        if payload isa AbstractDict && haskey(payload, "metadata")
            metadata = payload["metadata"]
            if metadata isa AbstractDict
                if haskey(metadata, "analysis_type")
                    attrs["analysis_type"] = string(metadata["analysis_type"])
                end
                if haskey(metadata, "sol_type")
                    attrs["sol_type"] = Int32(metadata["sol_type"])
                end
            end
        elseif payload isa AbstractDict && haskey(payload, "analysis_type")
            attrs["analysis_type"] = string(payload["analysis_type"])
        end

        if payload isa AbstractDict
            for key in sort(collect(keys(payload)); by=x -> string(x))
                _hdf5_write_value(file, string(key), payload[key])
            end
        else
            _hdf5_write_value(file, "payload", payload)
        end
    end
    return h5_path
end

@inline function _internal_node_order(id_map)
    return [nid for (nid, _) in sort(collect(id_map), by=x -> x[2])]
end

function _export_clean_subcases(subcases)
    cleaned = Any[]
    for sc in subcases
        if sc isa AbstractDict
            sc_copy = deepcopy(sc)
            for key in collect(keys(sc_copy))
                startswith(string(key), "_") && delete!(sc_copy, key)
            end
            push!(cleaned, sc_copy)
        else
            push!(cleaned, deepcopy(sc))
        end
    end
    return cleaned
end

function _export_strip_private_keys!(value)
    if value isa AbstractDict
        for key in collect(keys(value))
            startswith(string(key), "_") && delete!(value, key)
        end
        for key in collect(keys(value))
            _export_strip_private_keys!(value[key])
        end
    elseif value isa AbstractVector
        for item in value
            _export_strip_private_keys!(item)
        end
    end
    return value
end

function build_optimization_export_payload(results)
    opt_payload = deepcopy(results["optimization"])
    if opt_payload isa AbstractDict && haskey(opt_payload, "model")
        delete!(opt_payload, "model")
    end
    _export_strip_private_keys!(opt_payload)

    return Dict(
        "analysis_type" => "SOL200_LITE_OPTIMIZATION",
        "forward_sol_type" => get(results, "forward_sol_type", nothing),
        "route_summary" => deepcopy(get(results, "route_summary", Dict{String,Any}())),
        "optimization" => opt_payload,
    )
end

function build_nonlinear_export_payload(subcases;
                                        diagnostics=nothing,
                                        analysis_type="SOL106_NONLINEAR_STATIC")
    exported_subcases = Any[]
    for sc in subcases
        push!(exported_subcases, Dict(
            "sid" => sc["sid"],
            "linear_solver_diagnostics" => deepcopy(get(sc, "solver_diagnostics", Dict{String,Any}())),
            "nonlinear_diagnostics" => deepcopy(get(sc, "nonlinear_diagnostics", Dict{String,Any}())),
        ))
    end

    payload = Dict(
        "analysis_type" => analysis_type,
        "subcases" => exported_subcases,
    )
    if diagnostics !== nothing
        payload["nonlinear_solver_summary"] = deepcopy(diagnostics)
    end
    return payload
end

@inline function _result_request_aliases(key::String)
    key_up = uppercase(strip(key))
    if key_up == "SPCFORCES"
        return ("SPCFORCES", "SPCFORCE")
    end
    return (key_up,)
end

@inline function _result_request_enabled_value(value)
    value === nothing && return false
    text = uppercase(strip(string(value)))
    return !(isempty(text) || text in ("NONE", "NO"))
end

@inline function subcase_result_request_enabled(sub_ctrl::AbstractDict, key::String; default_all_if_unspecified::Bool=true)
    for alias in _result_request_aliases(key)
        if haskey(sub_ctrl, alias)
            return _result_request_enabled_value(sub_ctrl[alias])
        end
    end

    if default_all_if_unspecified
        any_explicit = false
        for req in ("DISPLACEMENT", "FORCE", "STRESS", "STRAIN", "SPCFORCES")
            for alias in _result_request_aliases(req)
                if haskey(sub_ctrl, alias)
                    any_explicit = true
                    break
                end
            end
            any_explicit && break
        end
        return !any_explicit
    end

    return false
end

function append_requested_subcase_results!(global_results::AbstractDict, sc::AbstractDict, sub_ctrl::AbstractDict)
    if subcase_result_request_enabled(sub_ctrl, "DISPLACEMENT")
        append!(global_results["displacements"], sc["displacements"])
    end
    if subcase_result_request_enabled(sub_ctrl, "SPCFORCES")
        append!(global_results["spc_forces"], sc["spc_forces"])
    end
    if subcase_result_request_enabled(sub_ctrl, "FORCE")
        for k in keys(global_results["forces"])
            append!(global_results["forces"][k], get(sc["forces"], k, Any[]))
        end
        for k in keys(global_results["forces_bilin"])
            append!(global_results["forces_bilin"][k], get(sc["forces_bilin"], k, Any[]))
        end
    end
    if subcase_result_request_enabled(sub_ctrl, "STRESS")
        for k in keys(global_results["stresses"])
            append!(global_results["stresses"][k], get(sc["stresses"], k, Any[]))
        end
    end
    if subcase_result_request_enabled(sub_ctrl, "STRAIN")
        for k in keys(global_results["strains"])
            append!(global_results["strains"][k], get(sc["strains"], k, Any[]))
        end
    end
    return global_results
end

function filtered_subcase_result_payload(sc::AbstractDict, sub_ctrl::AbstractDict)
    force_enabled = subcase_result_request_enabled(sub_ctrl, "FORCE")
    stress_enabled = subcase_result_request_enabled(sub_ctrl, "STRESS")
    strain_enabled = subcase_result_request_enabled(sub_ctrl, "STRAIN")

    forces = Dict{String,Any}()
    for (k, v) in sc["forces"]
        forces[k] = force_enabled ? deepcopy(v) : Any[]
    end

    forces_bilin = Dict{String,Any}()
    for (k, v) in get(sc, "forces_bilin", Dict{String,Any}())
        forces_bilin[k] = force_enabled ? deepcopy(v) : Any[]
    end

    stresses = Dict{String,Any}()
    for (k, v) in sc["stresses"]
        stresses[k] = stress_enabled ? deepcopy(v) : Any[]
    end

    strains = Dict{String,Any}()
    for (k, v) in sc["strains"]
        strains[k] = strain_enabled ? deepcopy(v) : Any[]
    end

    return Dict(
        "displacements" => subcase_result_request_enabled(sub_ctrl, "DISPLACEMENT") ? deepcopy(sc["displacements"]) : Any[],
        "spc_forces" => subcase_result_request_enabled(sub_ctrl, "SPCFORCES") ? deepcopy(sc["spc_forces"]) : Any[],
        "forces" => forces,
        "forces_bilin" => forces_bilin,
        "stresses" => stresses,
        "strains" => strains,
    )
end

function build_buckling_export_payload(eigenvalues, mode_shapes, id_map;
                                       frequencies=nothing,
                                       mass_summary=nothing,
                                       modal_effective_mass=nothing,
                                       buckling_subcases=nothing,
                                       analysis_type="SOL105_BUCKLING",
                                       diagnostics=nothing)
    sorted_nodes = sort(collect(keys(id_map)))
    mode_count = size(mode_shapes, 2)
    modes = Any[]
    for i in 1:mode_count
        mode_data = Any[]
        for nid in sorted_nodes
            idx = id_map[nid]
            base = (idx - 1) * 6
            push!(mode_data, Dict(
                "grid_id" => nid,
                "t1" => mode_shapes[base + 1, i],
                "t2" => mode_shapes[base + 2, i],
                "t3" => mode_shapes[base + 3, i],
                "r1" => mode_shapes[base + 4, i],
                "r2" => mode_shapes[base + 5, i],
                "r3" => mode_shapes[base + 6, i],
            ))
        end

        mode_entry = Dict(
            "mode_number" => i,
            "mode_shape" => mode_data,
        )
        if frequencies !== nothing
            mode_entry["frequency_hz"] = frequencies[i]
        end
        if eigenvalues !== nothing
            mode_entry["eigenvalue"] = eigenvalues[i]
        end
        push!(modes, mode_entry)
    end

    payload = Dict(
        "analysis_type" => analysis_type,
        "grid_id_order" => sorted_nodes,
        "modes" => modes,
        "mode_shape_count" => mode_count,
    )
    if eigenvalues !== nothing
        payload["eigenvalues"] = collect(eigenvalues)
        payload["mode_shapes_omitted"] = mode_count < length(eigenvalues)
    end
    if frequencies !== nothing
        payload["frequencies"] = collect(frequencies)
    end
    if mass_summary !== nothing
        payload["mass_summary"] = deepcopy(mass_summary)
    end
    if modal_effective_mass !== nothing
        payload["modal_effective_mass"] = deepcopy(modal_effective_mass)
    end
    if buckling_subcases !== nothing
        payload["subcases"] = _export_clean_subcases(buckling_subcases)
    end
    if diagnostics !== nothing
        payload["solver_diagnostics"] = deepcopy(diagnostics)
    end
    return payload
end

function build_solution_hdf5_payload(results, filename, global_results)
    sol_type = results["sol_type"]
    model = results["model"]
    cc = get(model, "CASE_CONTROL", Dict{String,Any}())
    subcase_cc = get(cc, "SUBCASES", Dict{Int,Dict{String,Any}}())
    node_order = _internal_node_order(results["id_map"])

    exported_subcases = Any[]
    for sc in results["subcases"]
        sid = sc["sid"]
        sub_ctrl = get(subcase_cc, sid, Dict{String,Any}())
        filtered = filtered_subcase_result_payload(sc, sub_ctrl)
        sc_payload = Dict(
            "sid" => sid,
            "load_id" => get(sub_ctrl, "LOAD", nothing),
            "spc_id" => get(sub_ctrl, "SPC", nothing),
            "displacements" => filtered["displacements"],
            "spc_forces" => filtered["spc_forces"],
            "forces" => filtered["forces"],
            "forces_bilin" => filtered["forces_bilin"],
            "stresses" => filtered["stresses"],
            "strains" => filtered["strains"],
            "solver_diagnostics" => deepcopy(get(sc, "solver_diagnostics", Dict{String,Any}())),
            "raw_displacement" => collect(get(sc, "raw_displacement", Float64[])),
            "u_analysis" => collect(get(sc, "u_analysis", Float64[])),
            "fixed_dofs" => sort(collect(get(sc, "fixed_dofs", Set{Int}()))),
            "element_vonmises" => deepcopy(get(sc, "element_vonmises", Dict{Int,Float64}())),
        )
        if sol_type == 106
            sc_payload["nonlinear_diagnostics"] = deepcopy(get(sc, "nonlinear_diagnostics", Dict{String,Any}()))
        end
        push!(exported_subcases, sc_payload)
    end

    payload = Dict(
        "metadata" => Dict(
            "analysis_type" => global_results["analysis_type"],
            "sol_type" => sol_type,
            "source_file" => basename(filename),
        ),
        "mesh" => deepcopy(results["mesh"]),
        "vector_layout" => Dict(
            "node_id_order" => node_order,
            "dof_labels" => ["t1", "t2", "t3", "r1", "r2", "r3"],
        ),
        "subcases" => exported_subcases,
        "aggregated_results" => deepcopy(global_results),
    )

    if sol_type == 106
        payload["nonlinear_export"] = build_nonlinear_export_payload(results["subcases"];
            diagnostics=get(results, "solver_diagnostics", nothing),
            analysis_type=global_results["analysis_type"])
    end

    return payload
end

function build_modal_hdf5_payload(results, filename)
    payload = Dict(
        "metadata" => Dict(
            "analysis_type" => "SOL103_MODES",
            "sol_type" => 103,
            "source_file" => basename(filename),
        ),
        "mesh" => deepcopy(results["mesh"]),
        "vector_layout" => Dict(
            "node_id_order" => _internal_node_order(results["id_map"]),
            "dof_labels" => ["t1", "t2", "t3", "r1", "r2", "r3"],
        ),
        "modal_results" => build_buckling_export_payload(
            results["eigenvalues"],
            results["_raw_mode_shapes"],
            results["id_map"];
            frequencies=results["frequencies"],
            mass_summary=get(results, "mass_summary", nothing),
            modal_effective_mass=get(results, "modal_effective_mass", nothing),
            buckling_subcases=get(results, "subcases", nothing),
            analysis_type="SOL103_MODES",
            diagnostics=get(results, "solver_diagnostics", nothing),
        ),
        "raw_mode_shapes" => Array(results["_raw_mode_shapes"]),
    )
    if haskey(results, "subcases")
        payload["subcases"] = _export_clean_subcases(results["subcases"])
    end
    return payload
end

function build_buckling_hdf5_payload(results, filename)
    return Dict(
        "metadata" => Dict(
            "analysis_type" => "SOL105_BUCKLING",
            "sol_type" => 105,
            "source_file" => basename(filename),
        ),
        "mesh" => deepcopy(results["mesh"]),
        "vector_layout" => Dict(
            "node_id_order" => _internal_node_order(results["id_map"]),
            "dof_labels" => ["t1", "t2", "t3", "r1", "r2", "r3"],
        ),
        "buckling_results" => build_buckling_export_payload(
            results["eigenvalues"],
            results["_raw_mode_shapes"],
            results["id_map"];
            analysis_type="SOL105_BUCKLING",
            diagnostics=get(results, "solver_diagnostics", nothing),
        ),
        "raw_mode_shapes" => Array(results["_raw_mode_shapes"]),
        "u_static" => collect(get(results, "u_static", Float64[])),
        "fixed_dofs" => sort(collect(get(results, "fixed_dofs", Set{Int}()))),
    )
end

function build_jfem_element_tables(model, id_map)
    jfem_node_ids = sort(collect(keys(id_map)))
    pshells = model["PSHELLs"]
    pbarls  = model["PBARLs"]
    prods_m = model["PRODs"]

    # Helper to look up property by PID (handles both string and int keys)
    function find_prop(pdict, pid)
        p = get(pdict, string(pid), nothing)
        if p === nothing; p = get(pdict, pid, nothing); end
        return p
    end

    jfem_quads = Tuple{Int,Int,Vector{Int},Float32}[]   # eid, pid, nodes, thickness
    jfem_trias = Tuple{Int,Int,Vector{Int},Float32}[]
    for (id, el) in model["CSHELLs"]
        eid = _export_entry_public_id(id, el)
        if !haskey(el, "NODES"); continue; end
        nids = el["NODES"]
        if !all(n -> haskey(id_map, n), nids); continue; end
        pid = get(el, "PID", 0)
        prop = find_prop(pshells, pid)
        t = Float32(prop !== nothing ? get(prop, "T", 0.0) : 0.0)
        if length(nids) == 4
            push!(jfem_quads, (eid, pid, nids, t))
        elseif length(nids) == 3
            push!(jfem_trias, (eid, pid, nids, t))
        end
    end
    sort!(jfem_quads, by=x->x[1])
    sort!(jfem_trias, by=x->x[1])

    jfem_bars = Tuple{Int,Int,Int,Int,Float32}[]   # eid, pid, ga, gb, area
    for (id, bar) in model["CBARs"]
        eid = _export_entry_public_id(id, bar)
        if !haskey(bar, "GA"); continue; end
        ga, gb = bar["GA"], bar["GB"]
        if !haskey(id_map, ga) || !haskey(id_map, gb); continue; end
        pid = get(bar, "PID", 0)
        prop = find_prop(pbarls, pid)
        a = Float32(prop !== nothing ? get(prop, "A", 0.0) : 0.0)
        push!(jfem_bars, (eid, pid, ga, gb, a))
    end
    for (id, bar) in get(model, "CBEAMs", Dict())
        eid = _export_entry_public_id(id, bar)
        if !haskey(bar, "GA"); continue; end
        ga, gb = bar["GA"], bar["GB"]
        if !haskey(id_map, ga) || !haskey(id_map, gb); continue; end
        pid = get(bar, "PID", 0)
        prop = find_prop(pbarls, pid)
        a = Float32(prop !== nothing ? get(prop, "A", 0.0) : 0.0)
        push!(jfem_bars, (eid, pid, ga, gb, a))
    end
    sort!(jfem_bars, by=x->x[1])

    jfem_rods = Tuple{Int,Int,Int,Int,Float32}[]   # eid, pid, ga, gb, area
    for (id, rod) in model["CRODs"]
        eid = _export_entry_public_id(id, rod)
        if !haskey(rod, "GA"); continue; end
        ga, gb = rod["GA"], rod["GB"]
        if !haskey(id_map, ga) || !haskey(id_map, gb); continue; end
        pid = get(rod, "PID", 0)
        prop = find_prop(prods_m, pid)
        a = Float32(prop !== nothing ? get(prop, "A", 0.0) : 0.0)
        push!(jfem_rods, (eid, pid, ga, gb, a))
    end
    sort!(jfem_rods, by=x->x[1])

    # Solid elements: CTETRA, CHEXA, CPENTA
    jfem_tetras  = Tuple{Int,Int,Vector{Int}}[]   # eid, pid, nodes(4)
    jfem_hexas   = Tuple{Int,Int,Vector{Int}}[]   # eid, pid, nodes(8)
    jfem_pentas  = Tuple{Int,Int,Vector{Int}}[]   # eid, pid, nodes(6)
    for (id, el) in get(model, "CSOLIDs", Dict())
        if !haskey(el, "NODES"); continue; end
        eid = _export_entry_public_id(id, el)
        nids = el["NODES"]
        if !all(n -> haskey(id_map, n), nids); continue; end
        pid = get(el, "PID", 0)
        nn = length(nids)
        if nn == 4;     push!(jfem_tetras, (eid, pid, nids))
        elseif nn == 8; push!(jfem_hexas,  (eid, pid, nids))
        elseif nn == 6; push!(jfem_pentas, (eid, pid, nids))
        end
    end
    sort!(jfem_tetras, by=x->x[1])
    sort!(jfem_hexas,  by=x->x[1])
    sort!(jfem_pentas, by=x->x[1])

    return jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods, jfem_tetras, jfem_hexas, jfem_pentas
end

# --- v3 extension: constraint tables and per-subcase load data ---

# Coordinate transform for force/moment direction vectors (duplicated from solver/helpers.jl to avoid cross-module dependency)
function _export_coord_transform(model, cid, vec)
    if cid == 0; return vec; end
    if !haskey(model["CORDs"], string(cid)); return vec; end
    cord = model["CORDs"][string(cid)]
    R = hcat(cord["U"], cord["V"], cord["W"])
    return R * vec
end

function build_jfem_constraint_tables(model, id_map)
    # --- CELAS1 springs ---
    jfem_celas = Tuple{Int,Int,Int,Int,Int,Float32}[]   # eid, g1, c1, g2, c2, stiffness
    pelases = get(model, "PELASs", Dict())
    for (id, el) in get(model, "CELASs", Dict())
        eid = _export_entry_public_id(id, el)
        g1 = get(el, "G1", 0); c1 = get(el, "C1", 0)
        g2 = get(el, "G2", 0); c2 = get(el, "C2", 0)
        pid = get(el, "PID", 0)
        pelas = get(pelases, string(pid), nothing)
        if pelas === nothing; pelas = get(pelases, pid, nothing); end
        K_stiff = Float32(pelas !== nothing ? get(pelas, "K", 0.0) : 0.0)
        push!(jfem_celas, (eid, g1, c1, g2, c2, K_stiff))
    end
    sort!(jfem_celas, by=x->x[1])

    # --- RBE2 rigid body elements ---
    jfem_rbe2s = []   # (eid, gn, cm, slave_nids::Vector{Int})
    for (id, rbe) in get(model, "RBE2s", Dict())
        eid = _export_entry_public_id(id, rbe)
        gn = rbe["GN"]; cm = Int(rbe["CM"])
        slaves = Int.(rbe["GM"])
        push!(jfem_rbe2s, (eid=eid, gn=gn, cm=cm, slaves=slaves))
    end
    sort!(jfem_rbe2s, by=x->x.eid)

    # --- RBE3 interpolation elements ---
    jfem_rbe3s = []   # (eid, refgrid, refc, dep_grids::Vector{Int})
    for (id, rbe) in get(model, "RBE3s", Dict())
        eid = _export_entry_public_id(id, rbe)
        refgrid = rbe["REFGRID"]; refc = Int(rbe["REFC"])
        # Collect all independent grids from weight groups
        wt_groups = get(rbe, "WT_GROUPS", [])
        deps = Int[]
        for group in wt_groups
            grids_raw = group isa AbstractDict ? group["grids"] : group.grids
            append!(deps, Int.(grids_raw))
        end
        push!(jfem_rbe3s, (eid=eid, refgrid=refgrid, refc=refc, deps=deps))
    end
    sort!(jfem_rbe3s, by=x->x.eid)

    return jfem_celas, jfem_rbe2s, jfem_rbe3s
end

function collect_spc_data(model, spc_id)
    # Returns Dict{Int, Int} mapping nid → dof_mask (e.g., 123456)
    spc_nodes = Dict{Int, Set{Int}}()
    if isnothing(spc_id); return Dict{Int,Int}(); end

    # Resolve SPCADD
    sets = Set{Int}()
    sid = Int(spc_id)
    if haskey(model["SPCADDs"], sid)
        union!(sets, model["SPCADDs"][sid])
    else
        push!(sets, sid)
    end

    # Collect SPC1 entries matching the set
    for spc in model["SPC1s"]
        if Int(spc["SID"]) in sets
            for n in spc["NODES"]
                if !haskey(spc_nodes, n); spc_nodes[n] = Set{Int}(); end
                for ch in spc["C"]
                    if isdigit(ch); push!(spc_nodes[n], parse(Int, string(ch))); end
                end
            end
        end
    end

    # Convert to integer mask: Set{1,2,3} → 123
    result = Dict{Int,Int}()
    for (nid, dofs) in spc_nodes
        mask = 0
        for d in sort(collect(dofs))
            mask = mask * 10 + d
        end
        result[nid] = mask
    end
    return result
end

function _collect_point_loads_recursive(model, sid, scale, forces_acc, moments_acc)
    # Collect FORCE cards
    for frc in get(model, "FORCEs", [])
        if Int(frc["SID"]) == sid
            gid = frc["GID"]
            global_dir = _export_coord_transform(model, Int(frc["CID"]), frc["Dir"])
            fvec = global_dir .* (frc["Mag"] * scale)
            if !haskey(forces_acc, gid); forces_acc[gid] = zeros(3); end
            forces_acc[gid] .+= fvec
        end
    end

    # Collect MOMENT cards
    for mom in get(model, "MOMENTs", [])
        if Int(mom["SID"]) == sid
            gid = mom["GID"]
            global_dir = _export_coord_transform(model, Int(mom["CID"]), mom["Dir"])
            mvec = global_dir .* (mom["Mag"] * scale)
            if !haskey(moments_acc, gid); moments_acc[gid] = zeros(3); end
            moments_acc[gid] .+= mvec
        end
    end

    # Recurse through LOAD combos
    for c in get(model, "LOAD_COMBOS", [])
        if Int(c["SID"]) == sid
            for sub in c["COMPS"]
                _collect_point_loads_recursive(model, Int(sub["LID"]), scale * c["S"] * sub["S"], forces_acc, moments_acc)
            end
        end
    end
end

function collect_point_loads(model, load_id)
    forces_acc = Dict{Int, Vector{Float64}}()
    moments_acc = Dict{Int, Vector{Float64}}()
    if isnothing(load_id); return forces_acc, moments_acc; end
    _collect_point_loads_recursive(model, Int(load_id), 1.0, forces_acc, moments_acc)
    return forces_acc, moments_acc
end

function collect_jfem_subcase_data(u, sub_res, id_map, jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods, jfem_tetras, jfem_hexas, jfem_pentas; model=nothing, spc_id=nothing, load_id=nothing)
    safe_f32(x) = (v = Float32(x); isnan(v) || isinf(v) ? Float32(0) : v)
    nNodes_jfem = length(jfem_node_ids)
    nQuads_jfem = length(jfem_quads)
    nTrias_jfem = length(jfem_trias)
    nBars_jfem  = length(jfem_bars)
    nRods_jfem  = length(jfem_rods)

    # Displacements: 6 per node in jfem_node_ids order
    disp_jfem = Vector{Float32}(undef, nNodes_jfem * 6)
    for (i, nid) in enumerate(jfem_node_ids)
        idx = id_map[nid]
        for k in 1:6
            disp_jfem[(i-1)*6+k] = safe_f32(u[(idx-1)*6+k])
        end
    end

    # Shell results: 7 per shell (fx, fy, fxy, mx, my, mxy, vonmises)
    nShells = nQuads_jfem + nTrias_jfem
    shell_jfem = zeros(Float32, nShells * 7)
    shell_force_map = Dict{Int,Any}()
    for f in sub_res["forces"]["quad4"]; shell_force_map[f["eid"]] = f; end
    for f in sub_res["forces"]["tria3"]; shell_force_map[f["eid"]] = f; end
    shell_stress_map = Dict{Int,Any}()
    for s in sub_res["stresses"]["quad4"]; shell_stress_map[s["eid"]] = s; end
    for s in sub_res["stresses"]["tria3"]; shell_stress_map[s["eid"]] = s; end

    for (i, (eid, _, _, _)) in enumerate(jfem_quads)
        base = (i-1) * 7
        f = get(shell_force_map, eid, nothing)
        s = get(shell_stress_map, eid, nothing)
        if f !== nothing
            shell_jfem[base+1] = safe_f32(f["fx"]);  shell_jfem[base+2] = safe_f32(f["fy"])
            shell_jfem[base+3] = safe_f32(f["fxy"]); shell_jfem[base+4] = safe_f32(f["mx"])
            shell_jfem[base+5] = safe_f32(f["my"]);  shell_jfem[base+6] = safe_f32(f["mxy"])
        end
        if s !== nothing
            vm = max(s["z1"]["von_mises"], s["z2"]["von_mises"])
            shell_jfem[base+7] = safe_f32(vm)
        end
    end
    for (i, (eid, _, _, _)) in enumerate(jfem_trias)
        base = (nQuads_jfem + i - 1) * 7
        f = get(shell_force_map, eid, nothing)
        s = get(shell_stress_map, eid, nothing)
        if f !== nothing
            shell_jfem[base+1] = safe_f32(f["fx"]);  shell_jfem[base+2] = safe_f32(f["fy"])
            shell_jfem[base+3] = safe_f32(f["fxy"]); shell_jfem[base+4] = safe_f32(f["mx"])
            shell_jfem[base+5] = safe_f32(f["my"]);  shell_jfem[base+6] = safe_f32(f["mxy"])
        end
        if s !== nothing
            vm = max(s["z1"]["von_mises"], s["z2"]["von_mises"])
            shell_jfem[base+7] = safe_f32(vm)
        end
    end

    # Bar results: 7 per bar (axial, shear_1, shear_2, torque, moment_a1, moment_a2, bar_vonmises)
    bar_jfem = zeros(Float32, nBars_jfem * 7)
    bar_force_map = Dict{Int,Any}()
    for f in sub_res["forces"]["cbar"]; bar_force_map[f["eid"]] = f; end
    bar_stress_map = Dict{Int,Any}()
    for s in sub_res["stresses"]["cbar"]; bar_stress_map[s["eid"]] = s; end

    for (i, (eid, _, _, _, _)) in enumerate(jfem_bars)
        base = (i-1) * 7
        f = get(bar_force_map, eid, nothing)
        s = get(bar_stress_map, eid, nothing)
        if f !== nothing
            bar_jfem[base+1] = safe_f32(f["axial"]);    bar_jfem[base+2] = safe_f32(f["shear_1"])
            bar_jfem[base+3] = safe_f32(f["shear_2"]);  bar_jfem[base+4] = safe_f32(f["torque"])
            bar_jfem[base+5] = safe_f32(f["moment_a1"]); bar_jfem[base+6] = safe_f32(f["moment_a2"])
        end
        if s !== nothing
            vm = abs(s["axial"])
            for pk in ["p1","p2","p3","p4"]
                vm = max(vm, abs(get(s["end_a"], pk, 0.0)), abs(get(s["end_b"], pk, 0.0)))
            end
            bar_jfem[base+7] = safe_f32(vm)
        end
    end

    # Rod results: 2 per rod (axial, torque)
    rod_jfem = zeros(Float32, nRods_jfem * 2)
    rod_force_map = Dict{Int,Any}()
    for f in sub_res["forces"]["crod"]; rod_force_map[f["eid"]] = f; end

    for (i, (eid, _, _, _, _)) in enumerate(jfem_rods)
        base = (i-1) * 2
        f = get(rod_force_map, eid, nothing)
        if f !== nothing
            rod_jfem[base+1] = safe_f32(f["axial"]); rod_jfem[base+2] = safe_f32(f["torque"])
        end
    end

    # Solid results: 1 value per solid element (von_mises)
    nSolids = length(jfem_tetras) + length(jfem_hexas) + length(jfem_pentas)
    solid_jfem = zeros(Float32, nSolids)
    solid_stress_map = Dict{Int,Any}()
    for key in ["ctetra", "chexa", "cpenta"]
        for s in get(get(sub_res, "stresses", Dict()), key, [])
            solid_stress_map[s["eid"]] = s
        end
    end
    solid_idx = 0
    for (eid, _, _) in jfem_tetras
        solid_idx += 1
        s = get(solid_stress_map, eid, nothing)
        if s !== nothing; solid_jfem[solid_idx] = safe_f32(s["von_mises"]); end
    end
    for (eid, _, _) in jfem_hexas
        solid_idx += 1
        s = get(solid_stress_map, eid, nothing)
        if s !== nothing; solid_jfem[solid_idx] = safe_f32(s["von_mises"]); end
    end
    for (eid, _, _) in jfem_pentas
        solid_idx += 1
        s = get(solid_stress_map, eid, nothing)
        if s !== nothing; solid_jfem[solid_idx] = safe_f32(s["von_mises"]); end
    end

    # --- v3: SPC, forces, moments per subcase ---
    spc_data = Tuple{Int32, UInt32}[]
    force_data = Tuple{Int32, Float32, Float32, Float32}[]
    moment_data = Tuple{Int32, Float32, Float32, Float32}[]

    if model !== nothing
        # Collect SPC constraints
        spc_map = collect_spc_data(model, spc_id)
        for (nid, mask) in sort(collect(spc_map), by=x->x[1])
            push!(spc_data, (Int32(nid), UInt32(mask)))
        end

        # Collect point forces and moments
        forces_dict, moments_dict = collect_point_loads(model, load_id)
        for (nid, fvec) in sort(collect(forces_dict), by=x->x[1])
            if norm(fvec) > 1e-30
                push!(force_data, (Int32(nid), safe_f32(fvec[1]), safe_f32(fvec[2]), safe_f32(fvec[3])))
            end
        end
        for (nid, mvec) in sort(collect(moments_dict), by=x->x[1])
            if norm(mvec) > 1e-30
                push!(moment_data, (Int32(nid), safe_f32(mvec[1]), safe_f32(mvec[2]), safe_f32(mvec[3])))
            end
        end
    end

    return (disp=disp_jfem, shell=shell_jfem, bar=bar_jfem, rod=rod_jfem, solid=solid_jfem,
            spc=spc_data, forces=force_data, moments=moment_data)
end

function export_vtk_subcase(filename, output_dir, sid, model, id_map, X, u, stresses)
    base_name = basename(filename)
    vtk_base = replace(base_name, ".bdf" => "") * "_Subcase_$sid"
    vtk_path = joinpath(output_dir, vtk_base)
    points = zeros(3, length(id_map))
    disp = zeros(3, length(id_map))
    for (nid, idx) in id_map
         points[:, idx] = X[idx, :]
         disp[:, idx] = u[(idx-1)*6+1:(idx-1)*6+3]
    end
    cells = MeshCell[]
    data_vonmises = Float64[]
    for (id, el) in model["CSHELLs"]
        if !haskey(el, "NODES"); continue; end
        eid = _export_entry_public_id(id, el); nids = [get(id_map, n, 0) for n in el["NODES"]]; if 0 in nids; continue; end
        if length(nids) == 3
            push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, nids)); push!(data_vonmises, get(stresses, eid, 0.0))
        elseif length(nids) == 4
            push!(cells, MeshCell(VTKCellTypes.VTK_QUAD, nids)); push!(data_vonmises, get(stresses, eid, 0.0))
        end
    end
    for (id, bar) in model["CBARs"]
         if !haskey(bar, "GA"); continue; end
         eid = _export_entry_public_id(id, bar); nids = [get(id_map, bar["GA"], 0), get(id_map, bar["GB"], 0)]; if 0 in nids; continue; end
         push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids)); push!(data_vonmises, get(stresses, eid, 0.0))
    end
    for (id, bar) in get(model, "CBEAMs", Dict())
         if !haskey(bar, "GA"); continue; end
         eid = _export_entry_public_id(id, bar); nids = [get(id_map, bar["GA"], 0), get(id_map, bar["GB"], 0)]; if 0 in nids; continue; end
         push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids)); push!(data_vonmises, get(stresses, eid, 0.0))
    end
    for (id, rod) in model["CRODs"]
         if !haskey(rod, "GA"); continue; end
         eid = _export_entry_public_id(id, rod); nids = [get(id_map, rod["GA"], 0), get(id_map, rod["GB"], 0)]; if 0 in nids; continue; end
         push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids)); push!(data_vonmises, get(stresses, eid, 0.0))
    end
    # Solid elements
    for (id, el) in get(model, "CSOLIDs", Dict())
        if !haskey(el, "NODES"); continue; end
        eid = _export_entry_public_id(id, el); enids = el["NODES"]
        nids = [get(id_map, n, 0) for n in enids]; if 0 in nids; continue; end
        nn = length(nids)
        if nn == 4
            push!(cells, MeshCell(VTKCellTypes.VTK_TETRA, nids))
        elseif nn == 8
            push!(cells, MeshCell(VTKCellTypes.VTK_HEXAHEDRON, nids))
        elseif nn == 6
            push!(cells, MeshCell(VTKCellTypes.VTK_WEDGE, nids))
        else
            continue
        end
        push!(data_vonmises, get(stresses, eid, 0.0))
    end
    if !isempty(cells)
        vtk = vtk_grid(vtk_path, points, cells)
        vtk["Displacement", VTKPointData()] = disp
        vtk["VonMises_Stress", VTKCellData()] = data_vonmises
        vtk_save(vtk)
        println("  VTK saved: $vtk_path.vtu")
    end
end

function export_json(filename, output_dir, global_results)
    json_name = _export_base_name(filename) * ".JU.JSON"
    json_path = joinpath(output_dir, json_name)
    println("\n>>> Exporting AGGREGATED JSON: $json_path")
    sanitize!(global_results)
    open(json_path, "w") do f; JSON.print(f, global_results, 4); end
end

function export_optimization_json(filename, output_dir, results)
    json_path = joinpath(output_dir, _export_base_name(filename) * ".OPTIMIZATION.JSON")
    payload = build_optimization_export_payload(results)
    sanitize!(payload)
    open(json_path, "w") do f
        JSON.print(f, payload, 2)
    end
    println("  Optimization JSON saved: $json_path")
end

function export_nonlinear_json(filename, output_dir, subcases;
                               diagnostics=nothing, analysis_type="SOL106_NONLINEAR_STATIC")
    json_name = _export_base_name(filename) * ".NONLINEAR.JSON"
    json_path = joinpath(output_dir, json_name)
    results = build_nonlinear_export_payload(subcases;
        diagnostics=diagnostics, analysis_type=analysis_type)

    println("\n>>> Exporting NONLINEAR JSON: $json_path")
    sanitize!(results)
    open(json_path, "w") do f; JSON.print(f, results, 4); end
end

function export_jfem_binary(filename, output_dir, id_map, X, jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods, jfem_tetras, jfem_hexas, jfem_pentas, jfem_subcases_data; jfem_celas=[], jfem_rbe2s=[], jfem_rbe3s=[])
    base_name = basename(filename)
    jfem_name = replace(base_name, ".bdf" => "") * ".jfem"
    jfem_path = joinpath(output_dir, jfem_name)
    nNodes_jfem = length(jfem_node_ids)
    nQuads_jfem = length(jfem_quads)
    nTrias_jfem = length(jfem_trias)
    nBars_jfem  = length(jfem_bars)
    nRods_jfem  = length(jfem_rods)
    nCelas_jfem = length(jfem_celas)
    nRBE2_jfem  = length(jfem_rbe2s)
    nRBE3_jfem  = length(jfem_rbe3s)
    println("\n>>> Exporting JFEM binary (v4): $jfem_path")
    open(jfem_path, "w") do io
        # Magic: 'JFEM'
        write(io, UInt8('J')); write(io, UInt8('F')); write(io, UInt8('E')); write(io, UInt8('M'))
        # Header (v3: extended with constraint counts)
        write(io, UInt32(4))                          # version 4
        write(io, UInt32(nNodes_jfem))
        write(io, UInt32(nQuads_jfem))
        write(io, UInt32(nTrias_jfem))
        write(io, UInt32(nBars_jfem))
        write(io, UInt32(nRods_jfem))
        write(io, UInt32(length(jfem_subcases_data)))  # nSubcases
        write(io, UInt32(nCelas_jfem))                 # v3: nCelas
        write(io, UInt32(nRBE2_jfem))                  # v3: nRBE2
        write(io, UInt32(nRBE3_jfem))                  # v3: nRBE3
        write(io, UInt32(length(jfem_tetras)))         # v4: nTetras
        write(io, UInt32(length(jfem_hexas)))          # v4: nHexas
        write(io, UInt32(length(jfem_pentas)))         # v4: nPentas

        # Node table: nid(i32), x(f32), y(f32), z(f32)
        for nid in jfem_node_ids
            idx = id_map[nid]
            write(io, Int32(nid))
            write(io, Float32(X[idx, 1])); write(io, Float32(X[idx, 2])); write(io, Float32(X[idx, 3]))
        end

        # CQUAD4 table: eid(i32), pid(i32), g1-g4(i32), thickness(f32)
        for (eid, pid, nodes, t) in jfem_quads
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
            write(io, t)
        end

        # CTRIA3 table: eid(i32), pid(i32), g1-g3(i32), thickness(f32)
        for (eid, pid, nodes, t) in jfem_trias
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
            write(io, t)
        end

        # CBAR table: eid(i32), pid(i32), ga(i32), gb(i32), area(f32)
        for (eid, pid, ga, gb, a) in jfem_bars
            write(io, Int32(eid)); write(io, Int32(pid)); write(io, Int32(ga)); write(io, Int32(gb))
            write(io, a)
        end

        # CROD table: eid(i32), pid(i32), ga(i32), gb(i32), area(f32)
        for (eid, pid, ga, gb, a) in jfem_rods
            write(io, Int32(eid)); write(io, Int32(pid)); write(io, Int32(ga)); write(io, Int32(gb))
            write(io, a)
        end

        # v3: CELAS table: eid(i32), g1(i32), c1(i32), g2(i32), c2(i32), stiffness(f32), pad(f32)
        for (eid, g1, c1, g2, c2, K_stiff) in jfem_celas
            write(io, Int32(eid)); write(io, Int32(g1)); write(io, Int32(c1))
            write(io, Int32(g2)); write(io, Int32(c2)); write(io, K_stiff); write(io, Float32(0))
        end

        # v3: RBE2 table (variable-length): eid(i32), gn(i32), cm(i32), nSlaves(u32), [slave_nid(i32) × nSlaves]
        for rbe in jfem_rbe2s
            write(io, Int32(rbe.eid)); write(io, Int32(rbe.gn)); write(io, Int32(rbe.cm))
            write(io, UInt32(length(rbe.slaves)))
            for s in rbe.slaves; write(io, Int32(s)); end
        end

        # v3: RBE3 table (variable-length): eid(i32), refgrid(i32), refc(i32), nDep(u32), [dep_nid(i32) × nDep]
        for rbe in jfem_rbe3s
            write(io, Int32(rbe.eid)); write(io, Int32(rbe.refgrid)); write(io, Int32(rbe.refc))
            write(io, UInt32(length(rbe.deps)))
            for d in rbe.deps; write(io, Int32(d)); end
        end

        # v4: CTETRA table: eid(i32), pid(i32), g1-g4(i32)
        for (eid, pid, nodes) in jfem_tetras
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
        end

        # v4: CHEXA table: eid(i32), pid(i32), g1-g8(i32)
        for (eid, pid, nodes) in jfem_hexas
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
        end

        # v4: CPENTA table: eid(i32), pid(i32), g1-g6(i32)
        for (eid, pid, nodes) in jfem_pentas
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
        end

        # Per-subcase data
        for sc in jfem_subcases_data
            write(io, UInt32(sc.sid))
            write(io, sc.disp)    # nNodes * 6 Float32
            write(io, sc.shell)   # (nQuads + nTrias) * 7 Float32
            write(io, sc.bar)     # nBars * 7 Float32
            write(io, sc.rod)     # nRods * 2 Float32
            write(io, sc.solid)   # nSolids Float32 (von_mises per solid)

            # v3: SPC data
            write(io, UInt32(length(sc.spc)))
            for (nid, mask) in sc.spc
                write(io, nid); write(io, mask)
            end

            # v3: Applied forces
            write(io, UInt32(length(sc.forces)))
            for (nid, fx, fy, fz) in sc.forces
                write(io, nid); write(io, fx); write(io, fy); write(io, fz)
            end

            # v3: Applied moments
            write(io, UInt32(length(sc.moments)))
            for (nid, mx, my, mz) in sc.moments
                write(io, nid); write(io, mx); write(io, my); write(io, mz)
            end
        end
    end
    nTet = length(jfem_tetras); nHex = length(jfem_hexas); nPen = length(jfem_pentas)
    println("  JFEM v4: $(nNodes_jfem) nodes, $(nQuads_jfem)Q+$(nTrias_jfem)T shells, $(nBars_jfem) bars, $(nRods_jfem) rods, $(nTet)Tet+$(nHex)Hex+$(nPen)Pen solids, $(nCelas_jfem) springs, $(length(jfem_subcases_data)) subcases")
end

function export_card_inventory(cards, output_dir, filename)
    processed_card_types = Set([
        "GRID", "CORD2R", "CORD1R", "CORD2C", "CORD2S",
        "CTRIA3", "CTRIA6", "CQUAD4", "CQUAD8", "CSHEAR", "CBAR", "CBEAM", "CROD", "CONROD", "CELAS1", "CELAS2", "CBUSH",
        "CTETRA", "CHEXA", "CPENTA", "PSOLID",
        "RBE1", "RBE2", "RBE3", "RBAR", "RSPLINE",
        "PSHELL", "PSHEAR", "PBARL", "PBAR", "PBAR*", "PBEAM", "PBEAM*", "PBEAML", "PROD", "PCOMP", "PELAS", "PBUSH",
        "MAT1", "MAT2", "MAT8", "MATT1", "TABLEM1",
        "DESVAR", "DRESP1", "DVPREL1", "DVMREL1", "DCONSTR", "DOPTPRM",
        "FORCE", "MOMENT", "PLOAD4", "PLOAD2", "PLOAD1", "PLOAD", "GRAV", "RFORCE",
        "SPC1", "SPC", "SPCADD", "MPC", "MPCADD", "LOAD",
        "CONM2", "CONM1", "CMASS1", "CMASS2", "PMASS",
        "EIGRL",
        "TEMP", "TEMPD", "DMIG",
        "PARAM"
    ])
    card_counts = Dict{String,Int}()
    unprocessed_cards = Dict{String,Int}()
    for (cname, clist) in cards
        card_counts[cname] = length(clist)
        if !(cname in processed_card_types)
            unprocessed_cards[cname] = length(clist)
        end
    end
    inv_json = Dict(
        "card_counts" => Dict(card_counts),
        "processed_card_types" => sort(collect(processed_card_types)),
        "unprocessed_cards" => Dict(unprocessed_cards)
    )
    inv_path = joinpath(output_dir, replace(basename(filename), ".bdf" => "") * ".CARDS.JSON")
    open(inv_path, "w") do f; JSON.print(f, inv_json, 4); end
    println(">>> Card inventory exported: $inv_path")
    if !isempty(unprocessed_cards)
        println("    WARNING: $(length(unprocessed_cards)) unprocessed card type(s):")
        for (cname, cnt) in sort(collect(unprocessed_cards), by=x->x[1])
            println("      $cname: $cnt")
        end
    end
end

# =============================================================================
# SOL105 BUCKLING EXPORT FUNCTIONS
# =============================================================================

function export_buckling_vtk(filename, output_dir, model, id_map, X, eigenvalues, mode_shapes)
    if isempty(eigenvalues); return; end
    base_name = replace(basename(filename), ".bdf" => "")
    n_modes = length(eigenvalues)

    points = zeros(3, length(id_map))
    for (nid, idx) in id_map
        points[:, idx] = X[idx, :]
    end

    cells = MeshCell[]
    for (id, el) in model["CSHELLs"]
        if !haskey(el, "NODES"); continue; end
        nids = [get(id_map, n, 0) for n in el["NODES"]]; if 0 in nids; continue; end
        if length(nids) == 3
            push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, nids))
        elseif length(nids) == 4
            push!(cells, MeshCell(VTKCellTypes.VTK_QUAD, nids))
        end
    end
    for (id, bar) in model["CBARs"]
        if !haskey(bar, "GA"); continue; end
        nids = [get(id_map, bar["GA"], 0), get(id_map, bar["GB"], 0)]; if 0 in nids; continue; end
        push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids))
    end
    for (id, bar) in get(model, "CBEAMs", Dict())
        if !haskey(bar, "GA"); continue; end
        nids = [get(id_map, bar["GA"], 0), get(id_map, bar["GB"], 0)]; if 0 in nids; continue; end
        push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids))
    end
    for (id, rod) in model["CRODs"]
        if !haskey(rod, "GA"); continue; end
        nids = [get(id_map, rod["GA"], 0), get(id_map, rod["GB"], 0)]; if 0 in nids; continue; end
        push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids))
    end
    for (id, rod) in get(model, "CONRODs", Dict())
        if !haskey(rod, "GA"); continue; end
        nids = [get(id_map, rod["GA"], 0), get(id_map, rod["GB"], 0)]; if 0 in nids; continue; end
        push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids))
    end
    for (id, el) in get(model, "CSOLIDs", Dict())
        if !haskey(el, "NODES"); continue; end
        enids = el["NODES"]
        nids = [get(id_map, n, 0) for n in enids]; if 0 in nids; continue; end
        nn = length(nids)
        if nn == 4;     push!(cells, MeshCell(VTKCellTypes.VTK_TETRA, nids))
        elseif nn == 8; push!(cells, MeshCell(VTKCellTypes.VTK_HEXAHEDRON, nids))
        elseif nn == 6; push!(cells, MeshCell(VTKCellTypes.VTK_WEDGE, nids))
        end
    end

    if isempty(cells); return; end

    for m in 1:n_modes
        vtk_name = base_name * "_Buckling_Mode_$m"
        vtk_path = joinpath(output_dir, vtk_name)
        disp = zeros(3, length(id_map))
        for (nid, idx) in id_map
            base = (idx-1)*6
            disp[1, idx] = mode_shapes[base+1, m]
            disp[2, idx] = mode_shapes[base+2, m]
            disp[3, idx] = mode_shapes[base+3, m]
        end
        vtk = vtk_grid(vtk_path, points, cells)
        vtk["BucklingMode_$m", VTKPointData()] = disp
        vtk_save(vtk)
        println("  VTK buckling mode $m saved: $vtk_path.vtu (lambda=$(round(eigenvalues[m], digits=4)))")
    end
end

function export_buckling_json(filename, output_dir, eigenvalues, mode_shapes, id_map;
                              frequencies=nothing, mass_summary=nothing,
                              modal_effective_mass=nothing, buckling_subcases=nothing,
                              analysis_type="SOL105_BUCKLING",
                              diagnostics=nothing)
    if isempty(eigenvalues); return; end
    json_name = _export_base_name(filename) * ".BUCKLING.JSON"
    json_path = joinpath(output_dir, json_name)
    results = build_buckling_export_payload(eigenvalues, mode_shapes, id_map;
        frequencies=frequencies,
        mass_summary=mass_summary,
        modal_effective_mass=modal_effective_mass,
        buckling_subcases=buckling_subcases,
        analysis_type=analysis_type,
        diagnostics=diagnostics)
    sanitize!(results)
    open(json_path, "w") do f; JSON.print(f, results, 4); end
    println(">>> Buckling JSON exported: $json_path")
end

function export_jfem_buckling(filename, output_dir, id_map, X, jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods, jfem_tetras, jfem_hexas, jfem_pentas, eigenvalues, mode_shapes; jfem_celas=[], jfem_rbe2s=[], jfem_rbe3s=[], K_global=nothing, node_R=nothing)
    if isempty(eigenvalues); return; end
    base_name = replace(basename(filename), ".bdf" => "")
    jfem_name = base_name * ".jfem"
    jfem_path = joinpath(output_dir, jfem_name)

    nNodes = length(jfem_node_ids)
    nQuads = length(jfem_quads)
    nTrias = length(jfem_trias)
    nBars  = length(jfem_bars)
    nRods  = length(jfem_rods)
    nModes = length(eigenvalues)

    println("\n>>> Exporting JFEM binary (v3 buckling): $jfem_path")
    open(jfem_path, "w") do io
        write(io, UInt8('J')); write(io, UInt8('F')); write(io, UInt8('E')); write(io, UInt8('M'))
        write(io, UInt32(3))        # version
        write(io, UInt32(nNodes))
        write(io, UInt32(nQuads))
        write(io, UInt32(nTrias))
        write(io, UInt32(nBars))
        write(io, UInt32(nRods))
        write(io, UInt32(nModes))   # nSubcases = nModes
        write(io, UInt32(length(jfem_celas)))
        write(io, UInt32(length(jfem_rbe2s)))
        write(io, UInt32(length(jfem_rbe3s)))

        # Node table: nid(i32), x(f32), y(f32), z(f32)
        for nid in jfem_node_ids
            idx = id_map[nid]
            write(io, Int32(nid))
            write(io, Float32(X[idx,1])); write(io, Float32(X[idx,2])); write(io, Float32(X[idx,3]))
        end

        # Element connectivity (same format as v2 JFEM)
        for (eid, pid, nodes, t) in jfem_quads
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
            write(io, t)
        end
        for (eid, pid, nodes, t) in jfem_trias
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
            write(io, t)
        end
        for (eid, pid, ga, gb, a) in jfem_bars
            write(io, Int32(eid)); write(io, Int32(pid)); write(io, Int32(ga)); write(io, Int32(gb))
            write(io, a)
        end
        for (eid, pid, ga, gb, a) in jfem_rods
            write(io, Int32(eid)); write(io, Int32(pid)); write(io, Int32(ga)); write(io, Int32(gb))
            write(io, a)
        end

        # Constraints (same format as v3 JFEM)
        for (eid, g1, c1, g2, c2, K_stiff) in jfem_celas
            write(io, Int32(eid)); write(io, Int32(g1)); write(io, Int32(c1))
            write(io, Int32(g2)); write(io, Int32(c2)); write(io, K_stiff); write(io, Float32(0))
        end
        for rbe in jfem_rbe2s
            write(io, Int32(rbe.eid)); write(io, Int32(rbe.gn)); write(io, Int32(rbe.cm))
            write(io, UInt32(length(rbe.slaves)))
            for s in rbe.slaves; write(io, Int32(s)); end
        end
        for rbe in jfem_rbe3s
            write(io, Int32(rbe.eid)); write(io, Int32(rbe.refgrid)); write(io, Int32(rbe.refc))
            write(io, UInt32(length(rbe.deps)))
            for d in rbe.deps; write(io, Int32(d)); end
        end

        # Build nid→jfem_index map for fast lookup
        nid_to_jidx = Dict{Int,Int}()
        for (ji, nid) in enumerate(jfem_node_ids)
            nid_to_jidx[nid] = ji
        end

        # Per-mode data (stored as subcases, matching v3 static format)
        for m in 1:nModes
            # Subcase ID
            write(io, UInt32(m))

            # Displacements: nNodes × 6 Float32
            # Also build per-node displacement magnitude array for element results
            node_disp_mag = Vector{Float64}(undef, nNodes)
            phi = mode_shapes[:, m]
            for (ji, nid) in enumerate(jfem_node_ids)
                idx = id_map[nid]; base = (idx-1)*6
                tx = phi[base+1]
                ty = phi[base+2]
                tz = phi[base+3]
                node_disp_mag[ji] = sqrt(tx*tx + ty*ty + tz*tz)
                for d in 1:6; write(io, Float32(phi[base+d])); end
            end

            # Compute nodal strain energy: NSE_i = 0.5 * phi_i . (K*phi)_i.
            # Buckling mode vectors are exported in global components, while
            # K_eig is assembled in analysis/node coordinate components. When
            # node_R is available, rotate the mode back for the energy proxy.
            node_se = zeros(Float64, nNodes)
            if !isnothing(K_global)
                phi_energy = phi
                if !isnothing(node_R)
                    phi_energy = similar(phi)
                    for (nid, idx) in id_map
                        base = (idx - 1) * 6
                        R = node_R[idx]
                        phi_energy[base+1:base+3] = R' * phi[base+1:base+3]
                        phi_energy[base+4:base+6] = R' * phi[base+4:base+6]
                    end
                end
                f = K_global * phi_energy  # sparse matrix-vector product
                for (ji, nid) in enumerate(jfem_node_ids)
                    idx = id_map[nid]; base = (idx-1)*6
                    se = 0.0
                    for d in 1:6
                        se += phi_energy[base+d] * f[base+d]
                    end
                    node_se[ji] = 0.5 * abs(se)  # abs to avoid tiny negatives from numerics
                end
            end

            # Shell results: (nQuads + nTrias) × 7 Float32
            # Slot 0: strain energy, slots 1-5: zeros, slot 6: displacement magnitude
            for (eid, pid, nodes, t) in jfem_quads
                mag_sum = 0.0; se_sum = 0.0
                for n in nodes
                    ji = get(nid_to_jidx, n, 0)
                    if ji > 0; mag_sum += node_disp_mag[ji]; se_sum += node_se[ji]; end
                end
                nn = length(nodes)
                write(io, Float32(se_sum / nn))  # slot 0: strain energy
                for _ in 1:5; write(io, Float32(0.0)); end  # slots 1-5: zeros
                write(io, Float32(mag_sum / nn))  # slot 6: displacement magnitude
            end
            for (eid, pid, nodes, t) in jfem_trias
                mag_sum = 0.0; se_sum = 0.0
                for n in nodes
                    ji = get(nid_to_jidx, n, 0)
                    if ji > 0; mag_sum += node_disp_mag[ji]; se_sum += node_se[ji]; end
                end
                nn = length(nodes)
                write(io, Float32(se_sum / nn))
                for _ in 1:5; write(io, Float32(0.0)); end
                write(io, Float32(mag_sum / nn))
            end

            # Bar results: nBars × 7 Float32
            # Slot 0: strain energy, slots 1-5: zeros, slot 6: displacement magnitude
            for (eid, pid, ga, gb, a) in jfem_bars
                ja = get(nid_to_jidx, ga, 0)
                jb = get(nid_to_jidx, gb, 0)
                avg_mag = 0.0; avg_se = 0.0; cnt = 0
                if ja > 0; avg_mag += node_disp_mag[ja]; avg_se += node_se[ja]; cnt += 1; end
                if jb > 0; avg_mag += node_disp_mag[jb]; avg_se += node_se[jb]; cnt += 1; end
                if cnt > 0; avg_mag /= cnt; avg_se /= cnt; end
                write(io, Float32(avg_se))  # slot 0: strain energy
                for _ in 1:5; write(io, Float32(0.0)); end
                write(io, Float32(avg_mag))  # slot 6: displacement magnitude
            end

            # Rod results: nRods × 2 Float32 (zeros for buckling)
            for _ in 1:nRods; for _ in 1:2; write(io, Float32(0.0)); end; end

            # v3: SPC data (empty list)
            write(io, UInt32(0))
            # v3: Applied forces (empty list)
            write(io, UInt32(0))
            # v3: Applied moments (empty list)
            write(io, UInt32(0))
        end

        # Eigenvalue footer: 'EVAL' marker + nModes(u32) + eigenvalues(f64)
        write(io, UInt8('E')); write(io, UInt8('V')); write(io, UInt8('A')); write(io, UInt8('L'))
        write(io, UInt32(nModes))
        for ev in eigenvalues
            write(io, Float64(ev))
        end
    end
    println("  JFEM binary exported: $jfem_path ($nModes buckling modes)")
end
