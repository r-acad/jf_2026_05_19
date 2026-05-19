# optimize_thickness.jl - Element thickness optimization driver
#
# Supports two objectives:
#   :min_compliance - minimize C = F' * u for SOL 101
#   :max_buckling   - maximize lambda_1 for SOL 105
#
# Subject to: mass <= vol_frac * initial_mass
#
# This module now supports structured iteration histories, JSON checkpoints,
# and restartable optimization runs.

"""
    optimize_thickness(model::Dict, solve_fn::Function; kwargs...) -> Dict

Run element thickness optimization on the model.

Keyword arguments:
- `objective`: `:min_compliance` or `:max_buckling`
- `h_min`: minimum element thickness
- `h_max`: maximum element thickness (`0.0` means `10 * max(initial_h)`)
- `vol_frac`: target volume fraction relative to the starting design
- `max_iter`: maximum number of design evaluations for this call
- `tol`: convergence tolerance on relative objective change
- `move_limit`: maximum relative thickness change per iteration
- `eta`: OC damping exponent for compliance minimization
- `checkpoint_path`: optional JSON checkpoint file
- `checkpoint_every`: write a checkpoint every N evaluations
- `restart_from`: optional checkpoint file used to initialize thicknesses and history
- `capture_solver_diagnostics`: include forward-solver diagnostics in iteration records

Returns a Dict containing the objective history, structured iteration data,
final thicknesses, mass metadata, restart metadata, and the optimized model.
"""
function optimize_thickness(model::Dict, solve_fn::Function;
    objective::Symbol = :min_compliance,
    h_min::Float64 = 1e-4,
    h_max::Float64 = 0.0,
    vol_frac::Float64 = 0.5,
    max_iter::Int = 50,
    tol::Float64 = 1e-3,
    move_limit::Float64 = 0.2,
    eta::Float64 = 0.5,
    thickness_derivative_method::Union{Nothing,Symbol,String} = nothing,
    checkpoint_path::Union{Nothing,String} = nothing,
    checkpoint_every::Int = 1,
    restart_from::Union{Nothing,String} = nothing,
    capture_solver_diagnostics::Bool = true)

    elem_data = _prepare_element_optimization_data!(model)
    n_elem = length(elem_data)
    n_elem == 0 && error("[OPT] No shell elements available for thickness optimization")

    initial_h_vec = _current_thickness_vector(model, elem_data)
    areas = [ed.area for ed in elem_data]
    rho_vals = [ed.rho for ed in elem_data]
    mass_initial = sum(initial_h_vec .* areas .* rho_vals)
    mass_target = vol_frac * mass_initial

    if h_max <= 0.0
        h_max = 10.0 * maximum(initial_h_vec)
    end

    h_vec = clamp.(copy(initial_h_vec), h_min, h_max)
    history = Float64[]
    iterations = Any[]
    restart_summary = nothing
    converged = false
    termination_reason = "max_iter"

    if !isnothing(restart_from)
        restart_state = _load_optimization_checkpoint(restart_from)
        history, iterations, loaded_count = _restore_optimization_state!(
            h_vec, elem_data, restart_state, h_min, h_max
        )
        restart_summary = Dict(
            "source" => restart_from,
            "loaded_thickness_count" => loaded_count,
            "loaded_history_count" => length(history),
            "loaded_iteration_count" => length(iterations),
        )
        log_msg("[OPT] Restarted from $restart_from with $loaded_count thickness values and $(length(history)) prior objective values")
    end

    log_msg("[OPT] $n_elem shell elements, objective=$objective, vol_frac=$vol_frac")
    log_msg("[OPT] Initial mass: $(round(mass_initial, sigdigits=6)), target: $(round(mass_target, sigdigits=6))")
    log_msg("[OPT] h bounds: [$h_min, $h_max], move_limit=$move_limit, max_iter=$max_iter")

    if max_iter <= 0
        termination_reason = "no_iterations_requested"
    else
        for local_iter in 1:max_iter
            iter_number = length(history) + 1
            _apply_thicknesses!(model, elem_data, h_vec)
            results = solve_fn(model)

            obj_val, dobj_dh = if objective == :min_compliance
                _compliance_and_sensitivity(results, model, elem_data; thickness_derivative_method=thickness_derivative_method)
            elseif objective == :max_buckling
                _buckling_and_sensitivity(results, model, elem_data; thickness_derivative_method=thickness_derivative_method)
            else
                error("[OPT] Unknown objective: $objective")
            end

            current_mass = sum(h_vec .* areas .* rho_vals)
            rel_change = isempty(history) ? nothing :
                abs(obj_val - history[end]) / max(abs(history[end]), 1e-30)
            grad_norm = norm(dobj_dh)
            thickness_map = _thickness_map(elem_data, h_vec)

            iter_record = Dict{String,Any}(
                "iteration" => iter_number,
                "objective" => obj_val,
                "relative_change" => rel_change,
                "mass" => current_mass,
                "mass_ratio" => current_mass / max(mass_initial, 1e-30),
                "mass_constraint_violation" => max(current_mass - mass_target, 0.0),
                "min_thickness" => minimum(h_vec),
                "max_thickness" => maximum(h_vec),
                "gradient_norm" => grad_norm,
                "objective_type" => String(objective),
                "thickness_derivative_method" => isnothing(thickness_derivative_method) ? nothing : String(Symbol(thickness_derivative_method)),
                "thicknesses" => thickness_map,
            )

            if capture_solver_diagnostics
                iter_record["solver_diagnostics"] = _extract_optimization_solver_diagnostics(results, objective)
            end

            push!(history, obj_val)
            push!(iterations, iter_record)

            if isnothing(rel_change)
                log_msg("[OPT] Iter $iter_number: obj=$(round(obj_val, sigdigits=6)), mass=$(round(current_mass, sigdigits=6))")
            else
                log_msg("[OPT] Iter $iter_number: obj=$(round(obj_val, sigdigits=6)), delta=$(round(rel_change, sigdigits=4)), mass=$(round(current_mass, sigdigits=6))")
            end

            checkpoint_payload = _optimization_result_payload(
                objective, history, iterations, elem_data, initial_h_vec, h_vec,
                mass_initial, mass_target, converged, termination_reason,
                checkpoint_path, restart_summary, model, thickness_derivative_method
            )
            if !isnothing(checkpoint_path) && checkpoint_every > 0 && (local_iter % checkpoint_every == 0)
                _write_optimization_checkpoint(checkpoint_path, checkpoint_payload)
            end

            if !isnothing(rel_change) && rel_change < tol
                converged = true
                termination_reason = "converged"
                log_msg("[OPT] Converged at iteration $iter_number")
                break
            end

            if local_iter == max_iter
                break
            end

            h_vec = if objective == :min_compliance
                _oc_update(h_vec, dobj_dh, areas, rho_vals, mass_target, h_min, h_max, move_limit, eta)
            else
                _gradient_update(h_vec, dobj_dh, areas, rho_vals, mass_target, h_min, h_max, move_limit)
            end
        end
    end

    if !converged && termination_reason == "max_iter"
        log_msg("[OPT] Reached max_iter without satisfying convergence tolerance")
    end

    _apply_thicknesses!(model, elem_data, h_vec)

    result = _optimization_result_payload(
        objective, history, iterations, elem_data, initial_h_vec, h_vec,
        mass_initial, mass_target, converged, termination_reason,
        checkpoint_path, restart_summary, model, thickness_derivative_method
    )

    if !isnothing(checkpoint_path)
        _write_optimization_checkpoint(checkpoint_path, result)
    end

    return result
end

# ============================================================================
# Per-element PID splitting and restart-safe metadata
# ============================================================================

struct ElemOptData
    eid::Int
    pid_new::Int
    h0::Float64
    area::Float64
    rho::Float64
end

function _prepare_element_optimization_data!(model)
    opt_meta = get(model, "_optimization", nothing)
    if opt_meta isa AbstractDict && haskey(opt_meta, "thickness_pid_split")
        elem_data = _elem_data_from_metadata(opt_meta["thickness_pid_split"])
        if _optimization_metadata_valid(model, elem_data)
            return elem_data
        end
        log_msg("[OPT] Existing optimization metadata is stale; rebuilding per-element PSHELL split")
    end
    return _split_per_element_pids!(model)
end

function _split_per_element_pids!(model)
    pshells = model["PSHELLs"]
    mats = model["MATs"]
    elem_data = ElemOptData[]
    pid_base = 100000

    for (eid_str, el) in model["CSHELLs"]
        eid = parse(Int, eid_str)
        pid_orig = string(el["PID"])
        prop = get(pshells, pid_orig, nothing)
        isnothing(prop) && continue
        mid = string(prop["MID"])
        mat = get(mats, mid, nothing)
        isnothing(mat) && continue

        area = _element_area(el["NODES"], model)
        h0 = Float64(prop["T"])
        rho = Float64(get(mat, "RHO", 1.0))
        rho <= 0.0 && (rho = 1.0)

        pid_new = pid_base + eid
        new_prop = copy(prop)
        new_prop["PID"] = pid_new
        pshells[string(pid_new)] = new_prop
        el["PID"] = pid_new

        push!(elem_data, ElemOptData(eid, pid_new, h0, area, rho))
    end

    _store_optimization_metadata!(model, elem_data)
    return elem_data
end

function _store_optimization_metadata!(model, elem_data)
    opt_meta = get!(model, "_optimization", Dict{String,Any}())
    opt_meta["thickness_pid_split"] = [
        Dict(
            "eid" => ed.eid,
            "pid_new" => ed.pid_new,
            "h0" => ed.h0,
            "area" => ed.area,
            "rho" => ed.rho,
        ) for ed in elem_data
    ]
    opt_meta["thickness_pid_split_version"] = 1
    return nothing
end

function _elem_data_from_metadata(entries)
    elem_data = ElemOptData[]
    for entry in entries
        push!(elem_data, ElemOptData(
            Int(entry["eid"]),
            Int(entry["pid_new"]),
            Float64(entry["h0"]),
            Float64(entry["area"]),
            Float64(entry["rho"]),
        ))
    end
    return elem_data
end

function _optimization_metadata_valid(model, elem_data)
    pshells = get(model, "PSHELLs", Dict())
    shells = get(model, "CSHELLs", Dict())
    for ed in elem_data
        eid_key = string(ed.eid)
        pid_key = string(ed.pid_new)
        if !haskey(shells, eid_key) || !haskey(pshells, pid_key)
            return false
        end
        if Int(shells[eid_key]["PID"]) != ed.pid_new
            return false
        end
    end
    return true
end

function _current_thickness_vector(model, elem_data)
    pshells = model["PSHELLs"]
    return [Float64(pshells[string(ed.pid_new)]["T"]) for ed in elem_data]
end

function _element_area(nids, model)
    grids = model["GRIDs"]
    coords = [Float64.(grids[string(n)]["X"]) for n in nids]
    if length(nids) == 4
        d13 = coords[3] - coords[1]
        d24 = coords[4] - coords[2]
        return 0.5 * norm(cross(d13, d24))
    elseif length(nids) == 3
        d12 = coords[2] - coords[1]
        d13 = coords[3] - coords[1]
        return 0.5 * norm(cross(d12, d13))
    end
    return 0.0
end

function _apply_thicknesses!(model, elem_data, h_vec)
    pshells = model["PSHELLs"]
    for (i, ed) in enumerate(elem_data)
        pshells[string(ed.pid_new)]["T"] = h_vec[i]
    end
    return nothing
end

function _thickness_map(elem_data, h_vec)
    thicknesses = Dict{String,Float64}()
    for (i, ed) in enumerate(elem_data)
        thicknesses[string(ed.eid)] = h_vec[i]
    end
    return thicknesses
end

# ============================================================================
# Restart and checkpoint helpers
# ============================================================================

function _restore_optimization_state!(h_vec, elem_data, restart_state, h_min, h_max)
    history_raw = get(restart_state, "history", Any[])
    history = Float64[Float64(v) for v in history_raw]

    iterations_raw = get(restart_state, "iterations", Any[])
    iterations = Any[deepcopy(rec) for rec in iterations_raw]

    loaded_count = 0
    thicknesses = get(restart_state, "thicknesses", Dict{String,Any}())
    for (i, ed) in enumerate(elem_data)
        key = string(ed.eid)
        if haskey(thicknesses, key)
            h_vec[i] = clamp(Float64(thicknesses[key]), h_min, h_max)
            loaded_count += 1
        end
    end

    return history, iterations, loaded_count
end

function _load_optimization_checkpoint(path::String)
    if !isfile(path)
        error("[OPT] Restart checkpoint not found: $path")
    end
    state = JSON.parsefile(path)
    get(state, "checkpoint_version", 0) >= 1 || log_msg("[OPT] Restart file does not advertise a checkpoint_version; proceeding")
    return state
end

function _write_optimization_checkpoint(path::String, payload::Dict)
    dir = dirname(path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    payload_to_write = deepcopy(payload)
    payload_to_write["checkpoint_version"] = 1
    pop!(payload_to_write, "model", nothing)
    open(path, "w") do f
        JSON.print(f, payload_to_write, 2)
    end
    return nothing
end

function _optimization_result_payload(
    objective, history, iterations, elem_data, initial_h_vec, h_vec,
    mass_initial, mass_target, converged, termination_reason,
    checkpoint_path, restart_summary, model, thickness_derivative_method
)
    final_mass = sum(h_vec[i] * elem_data[i].area * elem_data[i].rho for i in eachindex(elem_data))
    result = Dict{String,Any}(
        "objective" => String(objective),
        "history" => copy(history),
        "iterations" => deepcopy(iterations),
        "thicknesses" => _thickness_map(elem_data, h_vec),
        "initial_thicknesses" => _thickness_map(elem_data, initial_h_vec),
        "converged" => converged,
        "n_iter" => length(iterations),
        "design_variable_count" => length(elem_data),
        "mass_initial" => mass_initial,
        "mass_target" => mass_target,
        "final_mass" => final_mass,
        "mass_ratio" => final_mass / max(mass_initial, 1e-30),
        "termination_reason" => termination_reason,
        "checkpoint_path" => checkpoint_path,
        "thickness_derivative_method" => isnothing(thickness_derivative_method) ? nothing : String(Symbol(thickness_derivative_method)),
        "restart" => restart_summary,
        "model" => model,
    )
    return result
end

function _extract_optimization_solver_diagnostics(results, objective::Symbol)
    if objective == :min_compliance
        subcases = get(results, "subcases", Any[])
        if !isempty(subcases)
            subcase = subcases[1]
            return Dict(
                "sol_type" => get(results, "sol_type", nothing),
                "subcase_id" => get(subcase, "subcase_id", 1),
                "details" => deepcopy(get(subcase, "solver_diagnostics", Dict{String,Any}())),
            )
        end
    elseif objective == :max_buckling
        return Dict(
            "sol_type" => get(results, "sol_type", nothing),
            "details" => deepcopy(get(results, "solver_diagnostics", Any[])),
        )
    end
    return nothing
end

# ============================================================================
# Compliance objective (SOL 101)
# ============================================================================

function _compliance_and_sensitivity(results, model, elem_data; thickness_derivative_method=nothing)
    sc = results["subcases"][1]
    u = sc["u_analysis"]
    K = results["K"]
    C = dot(u, K * u)

    id_map = results["id_map"]
    X = results["node_coords"]
    node_R = results["node_R"]
    ndof = results["ndof"]

    dC_dh = zeros(length(elem_data))
    for (i, ed) in enumerate(elem_data)
        dv = Dict("type" => "shell_thickness", "pids" => [ed.pid_new])
        if !isnothing(thickness_derivative_method)
            dv["method"] = String(Symbol(thickness_derivative_method))
        end
        dKdx_u = compute_dKdx_u(dv, model, id_map, X, node_R, u, ndof)
        dC_dh[i] = -dot(u, dKdx_u)
    end

    return C, dC_dh
end

# ============================================================================
# Buckling objective (SOL 105)
# ============================================================================

function _buckling_and_sensitivity(results, model, elem_data; thickness_derivative_method=nothing)
    eigenvalues = results["eigenvalues"]
    isempty(eigenvalues) && error("[OPT] No buckling eigenvalues found")
    lam1 = eigenvalues[1]

    config = Dict(
        "responses" => [Dict("id" => "dummy", "type" => "displacement", "grid" => 1, "dof" => 1, "subcase" => 1)],
        "design_variables" => [
            _optimization_shell_dv(ed, thickness_derivative_method)
            for ed in elem_data
        ],
    )

    config_path, tmp_io = mktemp()
    close(tmp_io)
    try
        open(config_path, "w") do f
            JSON.print(f, config, 2)
        end
        adj = solve_adjoint_buckling(results, config_path)

        dlam_dh = zeros(length(elem_data))
        for (i, ed) in enumerate(elem_data)
            dv_id = "h_$(ed.eid)"
            group_label = "PID_$(ed.pid_new)"
            if haskey(adj["sensitivities"], "mode_1") && haskey(adj["sensitivities"]["mode_1"], dv_id)
                dlam_dh[i] = get(adj["sensitivities"]["mode_1"][dv_id], group_label, 0.0)
            end
        end

        return lam1, dlam_dh
    finally
        rm(config_path; force=true)
    end
end

function _optimization_shell_dv(ed, thickness_derivative_method)
    dv = Dict{String,Any}("id" => "h_$(ed.eid)", "type" => "shell_thickness", "pids" => [ed.pid_new])
    if !isnothing(thickness_derivative_method)
        dv["method"] = String(Symbol(thickness_derivative_method))
    end
    return dv
end

# ============================================================================
# OC update (for compliance minimization)
# ============================================================================

function _design_limit_vector(limit, n::Int, label::String)
    if limit isa Number
        return fill(Float64(limit), n)
    elseif limit isa AbstractVector
        length(limit) == n || error("[OPT] Expected $label to have length $n")
        return Float64[Float64(v) for v in limit]
    end
    error("[OPT] Unsupported $label specification: $(typeof(limit))")
end

function _oc_update(h, dC_dh, areas, rho_vals, mass_target, h_min, h_max, move_limit, eta)
    n = length(h)
    dM_dh = areas .* rho_vals
    h_min_vec = _design_limit_vector(h_min, n, "lower bounds")
    h_max_vec = _design_limit_vector(h_max, n, "upper bounds")
    move_limit_vec = _design_limit_vector(move_limit, n, "move limits")
    lam_lo = 1e-20
    lam_hi = 1e20
    h_new = copy(h)

    for _ in 1:100
        lam_mid = 0.5 * (lam_lo + lam_hi)
        h_new .= h
        for i in 1:n
            if dC_dh[i] < 0
                Be = -dC_dh[i] / (lam_mid * dM_dh[i])
                h_new[i] = h[i] * Be^eta
            else
                h_new[i] = h[i]
            end

            h_new[i] = max(h_new[i], h[i] * (1 - move_limit_vec[i]))
            h_new[i] = min(h_new[i], h[i] * (1 + move_limit_vec[i]))
            h_new[i] = clamp(h_new[i], h_min_vec[i], h_max_vec[i])
        end

        mass_new = sum(h_new .* areas .* rho_vals)
        if mass_new > mass_target
            lam_lo = lam_mid
        else
            lam_hi = lam_mid
        end

        if abs(mass_new - mass_target) / max(mass_target, 1e-30) < 1e-6
            break
        end
    end

    return h_new
end

# ============================================================================
# Gradient projection update (for buckling maximization)
# ============================================================================

function _gradient_update(h, dlam_dh, areas, rho_vals, mass_target, h_min, h_max, move_limit)
    dM_dh = areas .* rho_vals
    grad_norm = norm(dlam_dh)
    grad_norm < 1e-30 && return copy(h)

    mass_grad_norm = norm(dM_dh)
    if mass_grad_norm < 1e-30
        h_min_vec = _design_limit_vector(h_min, length(h), "lower bounds")
        h_max_vec = _design_limit_vector(h_max, length(h), "upper bounds")
        return clamp.(copy(h), h_min_vec, h_max_vec)
    end

    h_min_vec = _design_limit_vector(h_min, length(h), "lower bounds")
    h_max_vec = _design_limit_vector(h_max, length(h), "upper bounds")
    move_limit_vec = _design_limit_vector(move_limit, length(h), "move limits")
    mass_grad = dM_dh / mass_grad_norm
    dlam_proj = dlam_dh - dot(dlam_dh, mass_grad) * mass_grad

    step_scale = maximum(abs.(dlam_proj)) > 1e-30 ?
        maximum(move_limit_vec .* h) / maximum(abs.(dlam_proj)) : 0.0

    h_new = h + step_scale .* dlam_proj
    h_new = clamp.(h_new,
        max.(h .* (1 .- move_limit_vec), h_min_vec),
        min.(h .* (1 .+ move_limit_vec), h_max_vec))
    h_new = clamp.(h_new, h_min_vec, h_max_vec)

    mass_curr = sum(h_new .* areas .* rho_vals)
    if mass_curr > 1e-30
        scale = mass_target / mass_curr
        h_new = clamp.(h_new .* scale, h_min_vec, h_max_vec)
    end

    return h_new
end

# ============================================================================
# Generalized sizing optimization (shell thickness + bar area)
# ============================================================================

struct SizingVarData
    kind::Symbol
    eid::Int
    pid_new::Int
    x0::Float64
    mass_coeff::Float64
    fixed_mass::Float64
end

"""
    optimize_sizing(model::Dict, solve_fn::Function; kwargs...) -> Dict

Run a restartable sizing optimization using one or more supported property
design-variable families:

- `:shell_thickness` for per-element PSHELL thickness variables
- `:bar_area` for per-element CBAR/CBEAM PBAR/PBARL area variables

The optimizer currently supports the same objectives as `optimize_thickness`:
`:min_compliance` on SOL 101 and `:max_buckling` on SOL 105.
"""
function optimize_sizing(model::Dict, solve_fn::Function;
    objective::Symbol = :min_compliance,
    design_variables::Vector{Symbol} = [:shell_thickness],
    x_min::Float64 = 1e-4,
    x_max::Float64 = 0.0,
    vol_frac::Float64 = 0.5,
    max_iter::Int = 50,
    tol::Float64 = 1e-3,
    move_limit::Float64 = 0.2,
    eta::Float64 = 0.5,
    thickness_derivative_method::Union{Nothing,Symbol,String} = nothing,
    bar_area_derivative_method::Union{Nothing,Symbol,String} = nothing,
    checkpoint_path::Union{Nothing,String} = nothing,
    checkpoint_every::Int = 1,
    restart_from::Union{Nothing,String} = nothing,
    capture_solver_diagnostics::Bool = true)

    kinds = _normalize_sizing_kinds(design_variables)
    vars = _prepare_sizing_variable_data!(model, kinds)
    n_var = length(vars)
    n_var == 0 && error("[OPT] No supported sizing variables available for $(join(string.(kinds), ", "))")

    initial_x_vec = _current_sizing_vector(model, vars)
    mass_coeffs = [sv.mass_coeff for sv in vars]
    fixed_mass = _sizing_fixed_mass(vars)
    variable_mass_initial = _sizing_variable_mass(vars, initial_x_vec)
    mass_initial = variable_mass_initial + fixed_mass
    mass_target = vol_frac * mass_initial
    variable_mass_target = max(mass_target - fixed_mass, 0.0)

    if x_max <= 0.0
        x_max = 10.0 * maximum(initial_x_vec)
    end

    x_vec = clamp.(copy(initial_x_vec), x_min, x_max)
    history = Float64[]
    iterations = Any[]
    restart_summary = nothing
    converged = false
    termination_reason = "max_iter"

    if !isnothing(restart_from)
        restart_state = _load_optimization_checkpoint(restart_from)
        history, iterations, loaded_count = _restore_sizing_state!(x_vec, vars, restart_state, x_min, x_max)
        restart_summary = Dict(
            "source" => restart_from,
            "loaded_design_variable_count" => loaded_count,
            "loaded_history_count" => length(history),
            "loaded_iteration_count" => length(iterations),
        )
        log_msg("[OPT] Restarted generalized sizing from $restart_from with $loaded_count design-variable values and $(length(history)) prior objective values")
    end

    log_msg("[OPT] $n_var sizing variables ($(join(string.(kinds), ", "))), objective=$objective, vol_frac=$vol_frac")
    log_msg("[OPT] Initial mass: $(round(mass_initial, sigdigits=6)), target: $(round(mass_target, sigdigits=6))")
    fixed_mass > 0.0 &&
        log_msg("[OPT] Fixed non-structural mass: $(round(fixed_mass, sigdigits=6)); variable target: $(round(variable_mass_target, sigdigits=6))")
    log_msg("[OPT] x bounds: [$x_min, $x_max], move_limit=$move_limit, max_iter=$max_iter")

    if max_iter <= 0
        termination_reason = "no_iterations_requested"
    else
        for local_iter in 1:max_iter
            iter_number = length(history) + 1
            _apply_sizing_values!(model, vars, x_vec)
            results = solve_fn(model)

            obj_val, dobj_dx = _sizing_objective_and_sensitivity(
                results, model, vars, objective;
                thickness_derivative_method=thickness_derivative_method,
                bar_area_derivative_method=bar_area_derivative_method,
            )

            current_variable_mass = _sizing_variable_mass(vars, x_vec)
            current_mass = current_variable_mass + fixed_mass
            rel_change = isempty(history) ? nothing :
                abs(obj_val - history[end]) / max(abs(history[end]), 1e-30)
            grad_norm = norm(dobj_dx)
            design_map = _sizing_value_map(vars, x_vec)

            iter_record = Dict{String,Any}(
                "iteration" => iter_number,
                "objective" => obj_val,
                "relative_change" => rel_change,
                "mass" => current_mass,
                "variable_mass" => current_variable_mass,
                "fixed_mass" => fixed_mass,
                "mass_ratio" => current_mass / max(mass_initial, 1e-30),
                "mass_constraint_violation" => max(current_mass - mass_target, 0.0),
                "min_design_value" => minimum(x_vec),
                "max_design_value" => maximum(x_vec),
                "gradient_norm" => grad_norm,
                "objective_type" => String(objective),
                "design_variable_types" => [String(kind) for kind in kinds],
                "design_variables" => design_map,
            )

            if !isnothing(thickness_derivative_method)
                iter_record["thickness_derivative_method"] = String(Symbol(thickness_derivative_method))
            end
            if !isnothing(bar_area_derivative_method)
                iter_record["bar_area_derivative_method"] = String(Symbol(bar_area_derivative_method))
            end
            if capture_solver_diagnostics
                iter_record["solver_diagnostics"] = _extract_optimization_solver_diagnostics(results, objective)
            end

            push!(history, obj_val)
            push!(iterations, iter_record)

            if isnothing(rel_change)
                log_msg("[OPT] Iter $iter_number: obj=$(round(obj_val, sigdigits=6)), mass=$(round(current_mass, sigdigits=6))")
            else
                log_msg("[OPT] Iter $iter_number: obj=$(round(obj_val, sigdigits=6)), delta=$(round(rel_change, sigdigits=4)), mass=$(round(current_mass, sigdigits=6))")
            end

            checkpoint_payload = _sizing_result_payload(
                objective, history, iterations, vars, initial_x_vec, x_vec,
                mass_initial, mass_target, converged, termination_reason,
                checkpoint_path, restart_summary, model,
                thickness_derivative_method, bar_area_derivative_method, kinds
            )
            if !isnothing(checkpoint_path) && checkpoint_every > 0 && (local_iter % checkpoint_every == 0)
                _write_optimization_checkpoint(checkpoint_path, checkpoint_payload)
            end

            if !isnothing(rel_change) && rel_change < tol
                converged = true
                termination_reason = "converged"
                log_msg("[OPT] Converged at iteration $iter_number")
                break
            end

            if local_iter == max_iter
                break
            end

            x_vec = if objective == :min_compliance
                _oc_update(x_vec, dobj_dx, mass_coeffs, ones(length(vars)), variable_mass_target, x_min, x_max, move_limit, eta)
            else
                _gradient_update(x_vec, dobj_dx, mass_coeffs, ones(length(vars)), variable_mass_target, x_min, x_max, move_limit)
            end
        end
    end

    if !converged && termination_reason == "max_iter"
        log_msg("[OPT] Reached max_iter without satisfying convergence tolerance")
    end

    _apply_sizing_values!(model, vars, x_vec)

    result = _sizing_result_payload(
        objective, history, iterations, vars, initial_x_vec, x_vec,
        mass_initial, mass_target, converged, termination_reason,
        checkpoint_path, restart_summary, model,
        thickness_derivative_method, bar_area_derivative_method, kinds
    )
    if !isnothing(checkpoint_path)
        _write_optimization_checkpoint(checkpoint_path, result)
    end
    return result
end

function _normalize_sizing_kinds(kinds_in)
    kinds = Symbol[]
    seen = Set{Symbol}()
    for raw in kinds_in
        kind = raw isa Symbol ? raw : Symbol(raw)
        kind in (:shell_thickness, :bar_area) || error("[OPT] Unsupported sizing design-variable family: $kind")
        if !(kind in seen)
            push!(seen, kind)
            push!(kinds, kind)
        end
    end
    isempty(kinds) && error("[OPT] At least one sizing design-variable family is required")
    return kinds
end

function _prepare_sizing_variable_data!(model, kinds)
    vars = SizingVarData[]
    if :shell_thickness in kinds
        append!(vars, _prepare_shell_sizing_data!(model))
    end
    if :bar_area in kinds
        append!(vars, _prepare_bar_sizing_data!(model))
    end
    return vars
end

function _prepare_shell_sizing_data!(model)
    elem_data = _prepare_element_optimization_data!(model)
    pshells = get(model, "PSHELLs", Dict())
    return [
        SizingVarData(
            :shell_thickness,
            ed.eid,
            ed.pid_new,
            ed.h0,
            ed.area * ed.rho,
            ed.area * Float64(get(get(pshells, string(ed.pid_new), Dict{String,Any}()), "NSM", 0.0)),
        ) for ed in elem_data
    ]
end

function _prepare_bar_sizing_data!(model)
    opt_meta = get(model, "_optimization", nothing)
    if opt_meta isa AbstractDict && haskey(opt_meta, "bar_area_pid_split") &&
       Int(get(opt_meta, "bar_area_pid_split_version", 0)) >= 2
        vars = _bar_sizing_data_from_metadata(opt_meta["bar_area_pid_split"])
        if _bar_optimization_metadata_valid(model, vars)
            return vars
        end
        log_msg("[OPT] Existing bar optimization metadata is stale; rebuilding per-element PBAR/PBARL split")
    end
    return _split_per_element_bar_pids!(model)
end

function _split_per_element_bar_pids!(model)
    pbarls = get(model, "PBARLs", Dict())
    mats = model["MATs"]
    vars = SizingVarData[]
    pid_base = 2000000

    for element_set in ("CBARs", "CBEAMs")
        for (eid_str, bar) in get(model, element_set, Dict())
            eid = parse(Int, eid_str)
            pid_orig = string(bar["PID"])
            prop = get(pbarls, pid_orig, nothing)
            isnothing(prop) && continue
            mat = get(mats, string(prop["MID"]), nothing)
            isnothing(mat) && continue

            A0 = Float64(get(prop, "A", 0.0))
            A0 > 0.0 || continue
            L = _bar_element_length(bar, model)
            L > 0.0 || continue

            rho = Float64(get(mat, "RHO", 1.0))
            rho <= 0.0 && (rho = 1.0)

            pid_new = pid_base + eid
            new_prop = copy(prop)
            new_prop["PID"] = pid_new
            pbarls[string(pid_new)] = new_prop
            bar["PID"] = pid_new

            push!(vars, SizingVarData(:bar_area, eid, pid_new, A0, L * rho, L * Float64(get(prop, "NSM", 0.0))))
        end
    end

    _store_bar_optimization_metadata!(model, vars)
    return vars
end

function _bar_element_length(bar, model)
    grids = get(model, "GRIDs", Dict())
    ga = string(bar["GA"])
    gb = string(bar["GB"])
    if !haskey(grids, ga) || !haskey(grids, gb)
        return 0.0
    end

    p1 = SVector{3,Float64}(Float64.(grids[ga]["X"]))
    p2 = SVector{3,Float64}(Float64.(grids[gb]["X"]))
    _, _, _, p1_eff, p2_eff = bar_offsets_and_endpoints(bar, p1, p2)
    return norm(p2_eff - p1_eff)
end

function _store_bar_optimization_metadata!(model, vars)
    opt_meta = get!(model, "_optimization", Dict{String,Any}())
    opt_meta["bar_area_pid_split"] = [
        Dict(
            "kind" => String(var.kind),
            "eid" => var.eid,
            "pid_new" => var.pid_new,
            "x0" => var.x0,
            "mass_coeff" => var.mass_coeff,
            "fixed_mass" => var.fixed_mass,
        ) for var in vars
    ]
    opt_meta["bar_area_pid_split_version"] = 2
    return nothing
end

function _bar_sizing_data_from_metadata(entries)
    vars = SizingVarData[]
    for entry in entries
        push!(vars, SizingVarData(
            Symbol(entry["kind"]),
            Int(entry["eid"]),
            Int(entry["pid_new"]),
            Float64(entry["x0"]),
            Float64(entry["mass_coeff"]),
            Float64(get(entry, "fixed_mass", 0.0)),
        ))
    end
    return vars
end

function _bar_optimization_metadata_valid(model, vars)
    pbarls = get(model, "PBARLs", Dict())
    element_sets = (get(model, "CBARs", Dict()), get(model, "CBEAMs", Dict()))
    for var in vars
        var.kind == :bar_area || return false
        pid_key = string(var.pid_new)
        haskey(pbarls, pid_key) || return false

        eid_key = string(var.eid)
        found = false
        for es in element_sets
            if haskey(es, eid_key) && Int(es[eid_key]["PID"]) == var.pid_new
                found = true
                break
            end
        end
        found || return false
    end
    return true
end

@inline _sizing_variable_mass(vars::AbstractVector{SizingVarData}, x_vec) =
    sum(x_vec[i] * vars[i].mass_coeff for i in eachindex(vars))

@inline _sizing_fixed_mass(vars::AbstractVector{SizingVarData}) =
    sum(var.fixed_mass for var in vars)

@inline _sizing_total_mass(vars::AbstractVector{SizingVarData}, x_vec) =
    _sizing_variable_mass(vars, x_vec) + _sizing_fixed_mass(vars)

function _current_sizing_vector(model, vars)
    pshells = get(model, "PSHELLs", Dict())
    pbarls = get(model, "PBARLs", Dict())
    x = zeros(length(vars))
    for (i, var) in enumerate(vars)
        if var.kind == :shell_thickness
            x[i] = Float64(pshells[string(var.pid_new)]["T"])
        else
            x[i] = Float64(pbarls[string(var.pid_new)]["A"])
        end
    end
    return x
end

function _apply_sizing_values!(model, vars, x_vec)
    pshells = get(model, "PSHELLs", Dict())
    pbarls = get(model, "PBARLs", Dict())
    for (i, var) in enumerate(vars)
        if var.kind == :shell_thickness
            pshells[string(var.pid_new)]["T"] = x_vec[i]
        else
            pbarls[string(var.pid_new)]["A"] = x_vec[i]
        end
    end
    return nothing
end

function _sizing_key(var::SizingVarData)
    prefix = var.kind == :shell_thickness ? "shell" : "bar"
    return "$(prefix)_$(var.eid)"
end

function _sizing_value_map(vars, x_vec)
    values = Dict{String,Float64}()
    for (i, var) in enumerate(vars)
        values[_sizing_key(var)] = x_vec[i]
    end
    return values
end

function _sizing_kind_map(vars, x_vec, kind::Symbol)
    values = Dict{String,Float64}()
    for (i, var) in enumerate(vars)
        var.kind == kind || continue
        values[string(var.eid)] = x_vec[i]
    end
    return values
end

function _restore_sizing_state!(x_vec, vars, restart_state, x_min, x_max)
    history_raw = get(restart_state, "history", Any[])
    history = Float64[Float64(v) for v in history_raw]

    iterations_raw = get(restart_state, "iterations", Any[])
    iterations = Any[deepcopy(rec) for rec in iterations_raw]

    loaded_count = 0
    design_values = get(restart_state, "design_variables", Dict{String,Any}())
    legacy_thicknesses = get(restart_state, "thicknesses", Dict{String,Any}())
    legacy_bar_areas = get(restart_state, "bar_areas", Dict{String,Any}())

    for (i, var) in enumerate(vars)
        key = _sizing_key(var)
        raw_val = if haskey(design_values, key)
            design_values[key]
        elseif var.kind == :shell_thickness && haskey(legacy_thicknesses, string(var.eid))
            legacy_thicknesses[string(var.eid)]
        elseif var.kind == :bar_area && haskey(legacy_bar_areas, string(var.eid))
            legacy_bar_areas[string(var.eid)]
        else
            nothing
        end

        if !isnothing(raw_val)
            x_vec[i] = clamp(Float64(raw_val), x_min, x_max)
            loaded_count += 1
        end
    end

    return history, iterations, loaded_count
end

function _sizing_dv(var::SizingVarData, thickness_derivative_method, bar_area_derivative_method; include_id::Bool=false)
    dv = Dict{String,Any}(
        "type" => String(var.kind),
        "pids" => [var.pid_new],
    )
    if include_id
        dv["id"] = var.kind == :shell_thickness ? "h_$(var.eid)" : "A_$(var.eid)"
    end

    method = if var.kind == :shell_thickness
        thickness_derivative_method
    else
        bar_area_derivative_method
    end
    if !isnothing(method)
        dv["method"] = String(Symbol(method))
    end
    return dv
end

function _sizing_objective_and_sensitivity(results, model, vars, objective;
    thickness_derivative_method=nothing,
    bar_area_derivative_method=nothing)

    if objective == :min_compliance
        sc = results["subcases"][1]
        u = sc["u_analysis"]
        K = results["K"]
        obj = dot(u, K * u)

        id_map = results["id_map"]
        X = results["node_coords"]
        node_R = results["node_R"]
        ndof = results["ndof"]

        grad = zeros(length(vars))
        for (i, var) in enumerate(vars)
            dv = _sizing_dv(var, thickness_derivative_method, bar_area_derivative_method)
            dKdx_u = compute_dKdx_u(dv, model, id_map, X, node_R, u, ndof)
            grad[i] = -dot(u, dKdx_u)
        end
        return obj, grad
    elseif objective == :max_buckling
        eigenvalues = results["eigenvalues"]
        isempty(eigenvalues) && error("[OPT] No buckling eigenvalues found")
        obj = eigenvalues[1]

        config = Dict(
            "responses" => [Dict("id" => "dummy", "type" => "displacement", "grid" => 1, "dof" => 1, "subcase" => 1)],
            "design_variables" => [
                _sizing_dv(var, thickness_derivative_method, bar_area_derivative_method; include_id=true)
                for var in vars
            ],
        )

        config_path, tmp_io = mktemp()
        close(tmp_io)
        try
            open(config_path, "w") do f
                JSON.print(f, config, 2)
            end
            adj = solve_adjoint_buckling(results, config_path)

            grad = zeros(length(vars))
            for (i, var) in enumerate(vars)
                dv_id = var.kind == :shell_thickness ? "h_$(var.eid)" : "A_$(var.eid)"
                group_label = "PID_$(var.pid_new)"
                if haskey(adj["sensitivities"], "mode_1") && haskey(adj["sensitivities"]["mode_1"], dv_id)
                    grad[i] = get(adj["sensitivities"]["mode_1"][dv_id], group_label, 0.0)
                end
            end
            return obj, grad
        finally
            rm(config_path; force=true)
        end
    end

    error("[OPT] Unknown objective: $objective")
end

function _sizing_result_payload(
    objective, history, iterations, vars, initial_x_vec, x_vec,
    mass_initial, mass_target, converged, termination_reason,
    checkpoint_path, restart_summary, model,
    thickness_derivative_method, bar_area_derivative_method, kinds
)
    fixed_mass = _sizing_fixed_mass(vars)
    variable_mass_initial = _sizing_variable_mass(vars, initial_x_vec)
    variable_mass_target = max(mass_target - fixed_mass, 0.0)
    variable_mass_final = _sizing_variable_mass(vars, x_vec)
    final_mass = variable_mass_final + fixed_mass
    result = Dict{String,Any}(
        "objective" => String(objective),
        "history" => copy(history),
        "iterations" => deepcopy(iterations),
        "design_variable_types" => [String(kind) for kind in kinds],
        "design_variables" => _sizing_value_map(vars, x_vec),
        "initial_design_variables" => _sizing_value_map(vars, initial_x_vec),
        "shell_thicknesses" => _sizing_kind_map(vars, x_vec, :shell_thickness),
        "initial_shell_thicknesses" => _sizing_kind_map(vars, initial_x_vec, :shell_thickness),
        "bar_areas" => _sizing_kind_map(vars, x_vec, :bar_area),
        "initial_bar_areas" => _sizing_kind_map(vars, initial_x_vec, :bar_area),
        "converged" => converged,
        "n_iter" => length(iterations),
        "design_variable_count" => length(vars),
        "fixed_mass" => fixed_mass,
        "variable_mass_initial" => variable_mass_initial,
        "variable_mass_target" => variable_mass_target,
        "variable_mass_final" => variable_mass_final,
        "mass_initial" => mass_initial,
        "mass_target" => mass_target,
        "final_mass" => final_mass,
        "mass_ratio" => final_mass / max(mass_initial, 1e-30),
        "termination_reason" => termination_reason,
        "checkpoint_path" => checkpoint_path,
        "thickness_derivative_method" => isnothing(thickness_derivative_method) ? nothing : String(Symbol(thickness_derivative_method)),
        "bar_area_derivative_method" => isnothing(bar_area_derivative_method) ? nothing : String(Symbol(bar_area_derivative_method)),
        "restart" => restart_summary,
        "model" => model,
    )

    # Preserve the thickness-only aliases used by the legacy API.
    if all(var.kind == :shell_thickness for var in vars)
        result["thicknesses"] = deepcopy(result["shell_thicknesses"])
        result["initial_thicknesses"] = deepcopy(result["initial_shell_thicknesses"])
    end
    return _finalize_optimization_result!(result)
end

function _constraint_history_key(entry)
    response_id = get(entry, "response_id", nothing)
    if !isnothing(response_id)
        return "response_" * string(response_id)
    end
    constraint_index = get(entry, "constraint_index", nothing)
    if !isnothing(constraint_index)
        return "constraint_" * lpad(string(Int(constraint_index)), 3, '0')
    end
    return "constraint_unknown"
end

function _finalize_optimization_result!(result::Dict{String,Any})
    iterations = get(result, "iterations", nothing)
    design_variables = get(result, "design_variables", nothing)
    initial_design_variables = get(result, "initial_design_variables", nothing)

    if iterations isa AbstractVector && design_variables isa AbstractDict
        trajectories = Dict{String,Any}()
        for label in sort!(collect(keys(design_variables)); by=x -> string(x))
            values = Any[]
            iteration_numbers = Int[]
            for iter in iterations
                iter isa AbstractDict || continue
                iter_designs = get(iter, "design_variables", nothing)
                iter_designs isa AbstractDict || continue
                haskey(iter_designs, label) || continue
                push!(iteration_numbers, Int(get(iter, "iteration", length(iteration_numbers) + 1)))
                push!(values, deepcopy(iter_designs[label]))
            end

            entry = Dict{String,Any}(
                "iterations" => iteration_numbers,
                "values" => values,
                "final_value" => deepcopy(design_variables[label]),
            )
            if initial_design_variables isa AbstractDict && haskey(initial_design_variables, label)
                entry["initial_value"] = deepcopy(initial_design_variables[label])
            end
            trajectories[string(label)] = entry
        end
        result["design_variable_trajectories"] = trajectories
    end

    if iterations isa AbstractVector
        active_constraint_history = Dict{String,Any}[]
        constraint_trajectories = Dict{String,Any}()

        for iter in iterations
            iter isa AbstractDict || continue
            iter_number = Int(get(iter, "iteration", length(active_constraint_history) + 1))

            if haskey(iter, "response_constraints")
                evaluated = get(iter, "response_constraints", nothing)
                if evaluated isa AbstractVector
                    for constraint in evaluated
                        constraint isa AbstractDict || continue
                        key = _constraint_history_key(constraint)
                        history_entry = get!(constraint_trajectories, key) do
                            Dict{String,Any}(
                                "response_id" => get(constraint, "response_id", nothing),
                                "constraint_index" => get(constraint, "constraint_index", nothing),
                                "response_family" => get(constraint, "response_family", nothing),
                                "constraint_grid" => get(constraint, "constraint_grid", nothing),
                                "constraint_dof" => get(constraint, "constraint_dof", nothing),
                                "history" => Any[],
                            )
                        end
                        push!(history_entry["history"], Dict{String,Any}(
                            "iteration" => iter_number,
                            "response_value" => deepcopy(get(constraint, "response_value", nothing)),
                            "response_upper_bound" => deepcopy(get(constraint, "response_upper_bound", nothing)),
                            "response_ratio" => deepcopy(get(constraint, "response_ratio", nothing)),
                            "response_margin" => deepcopy(get(constraint, "response_margin", nothing)),
                            "response_margin_ratio" => deepcopy(get(constraint, "response_margin_ratio", nothing)),
                            "response_constraint_violation" => deepcopy(get(constraint, "response_constraint_violation", nothing)),
                        ))
                    end
                end
            end

            active_constraint_index = get(iter, "active_constraint_index", nothing)
            if !isnothing(active_constraint_index)
                response_upper_bound = get(iter, "response_upper_bound", nothing)
                response_value = get(iter, "response_value", nothing)
                upper_scale = isnothing(response_upper_bound) ? 1.0 : max(abs(Float64(response_upper_bound)), 1e-30)
                response_ratio = isnothing(response_upper_bound) || isnothing(response_value) ? nothing :
                    Float64(response_value) / upper_scale
                response_margin = isnothing(response_upper_bound) || isnothing(response_value) ? nothing :
                    Float64(response_upper_bound) - Float64(response_value)
                response_margin_ratio = isnothing(response_margin) ? nothing : Float64(response_margin) / upper_scale

                active_entry = Dict{String,Any}(
                    "iteration" => iter_number,
                    "constraint_index" => Int(active_constraint_index),
                    "response_id" => get(iter, "active_response_id", nothing),
                    "response_family" => get(iter, "response_family", nothing),
                    "response_value" => deepcopy(response_value),
                    "response_upper_bound" => deepcopy(response_upper_bound),
                    "response_ratio" => response_ratio,
                    "response_margin" => response_margin,
                    "response_margin_ratio" => response_margin_ratio,
                    "response_constraint_violation" => deepcopy(get(iter, "response_constraint_violation", nothing)),
                )
                if haskey(iter, "response_grid") && !isnothing(iter["response_grid"])
                    active_entry["constraint_grid"] = iter["response_grid"]
                end
                if haskey(iter, "response_dof") && !isnothing(iter["response_dof"])
                    active_entry["constraint_dof"] = iter["response_dof"]
                end
                push!(active_constraint_history, active_entry)
            end
        end

        isempty(active_constraint_history) || (result["active_constraint_history"] = active_constraint_history)
        isempty(constraint_trajectories) || (result["constraint_trajectories"] = constraint_trajectories)
    end

    return result
end

# ============================================================================
# Grouped sizing route for SOL 200-lite exact DESVAR / DVPREL preservation
# ============================================================================

struct SizingGroupData
    label::String
    design_var_id::Int
    members::Vector{SizingVarData}
    x0::Float64
    relation_ids::Vector{Int}
    property_ids_by_kind::Dict{Symbol, Vector{Int}}
    mass_coeff::Float64
    fixed_mass::Float64
end

function optimize_grouped_sizing(model::Dict, solve_fn::Function, groups::Vector{SizingGroupData};
    objective::Symbol = :min_compliance,
    x_min = 1e-4,
    x_max = 0.0,
    vol_frac::Float64 = 0.5,
    max_iter::Int = 50,
    tol::Float64 = 1e-3,
    move_limit = 0.2,
    eta::Float64 = 0.5,
    thickness_derivative_method::Union{Nothing,Symbol,String} = nothing,
    bar_area_derivative_method::Union{Nothing,Symbol,String} = nothing,
    capture_solver_diagnostics::Bool = true)

    n_group = length(groups)
    n_group == 0 && error("[OPT] No grouped sizing variables were provided")

    initial_x_vec = _current_grouped_sizing_vector(model, groups)
    mass_coeffs = [group.mass_coeff for group in groups]
    fixed_mass = _grouped_fixed_mass(groups)
    variable_mass_initial = _grouped_variable_mass(groups, initial_x_vec)
    mass_initial = variable_mass_initial + fixed_mass
    mass_target = vol_frac * mass_initial
    variable_mass_target = max(mass_target - fixed_mass, 0.0)

    x_min_vec = _design_limit_vector(x_min, n_group, "group lower bounds")
    x_max_vec = _design_limit_vector(x_max, n_group, "group upper bounds")
    if any(v <= 0.0 for v in x_max_vec)
        fallback = 10.0 * maximum(initial_x_vec)
        x_max_vec = [v <= 0.0 ? fallback : v for v in x_max_vec]
    end
    move_limit_vec = _design_limit_vector(move_limit, n_group, "group move limits")

    x_vec = clamp.(copy(initial_x_vec), x_min_vec, x_max_vec)
    history = Float64[]
    iterations = Any[]
    converged = false
    termination_reason = "max_iter"

    grouped_kinds = sort!(collect(Set(var.kind for group in groups for var in group.members)))

    log_msg("[OPT] $n_group grouped sizing variables ($(join(string.(grouped_kinds), ", "))), objective=$objective, vol_frac=$vol_frac")
    log_msg("[OPT] Initial grouped mass: $(round(mass_initial, sigdigits=6)), target: $(round(mass_target, sigdigits=6))")
    fixed_mass > 0.0 &&
        log_msg("[OPT] Fixed non-structural mass: $(round(fixed_mass, sigdigits=6)); grouped variable target: $(round(variable_mass_target, sigdigits=6))")
    log_msg("[OPT] Group bounds prepared, max_iter=$max_iter")

    if max_iter <= 0
        termination_reason = "no_iterations_requested"
    else
        for local_iter in 1:max_iter
            iter_number = length(history) + 1
            _apply_grouped_sizing_values!(model, groups, x_vec)
            results = solve_fn(model)

            obj_val, dobj_dx = _grouped_sizing_objective_and_sensitivity(
                results, model, groups, objective;
                thickness_derivative_method=thickness_derivative_method,
                bar_area_derivative_method=bar_area_derivative_method,
            )

            current_variable_mass = _grouped_variable_mass(groups, x_vec)
            current_mass = current_variable_mass + fixed_mass
            rel_change = isempty(history) ? nothing :
                abs(obj_val - history[end]) / max(abs(history[end]), 1e-30)
            grad_norm = norm(dobj_dx)

            iter_record = Dict{String,Any}(
                "iteration" => iter_number,
                "objective" => obj_val,
                "relative_change" => rel_change,
                "mass" => current_mass,
                "variable_mass" => current_variable_mass,
                "fixed_mass" => fixed_mass,
                "mass_ratio" => current_mass / max(mass_initial, 1e-30),
                "mass_constraint_violation" => max(current_mass - mass_target, 0.0),
                "min_design_value" => minimum(x_vec),
                "max_design_value" => maximum(x_vec),
                "gradient_norm" => grad_norm,
                "objective_type" => String(objective),
                "design_variable_types" => [String(kind) for kind in grouped_kinds],
                "design_variables" => _grouped_design_value_map(groups, x_vec),
                "shell_thicknesses" => _grouped_kind_element_map(groups, x_vec, :shell_thickness),
                "bar_areas" => _grouped_kind_element_map(groups, x_vec, :bar_area),
            )

            if !isnothing(thickness_derivative_method)
                iter_record["thickness_derivative_method"] = String(Symbol(thickness_derivative_method))
            end
            if !isnothing(bar_area_derivative_method)
                iter_record["bar_area_derivative_method"] = String(Symbol(bar_area_derivative_method))
            end
            if capture_solver_diagnostics
                iter_record["solver_diagnostics"] = _extract_optimization_solver_diagnostics(results, objective)
            end

            push!(history, obj_val)
            push!(iterations, iter_record)

            if isnothing(rel_change)
                log_msg("[OPT] Iter $iter_number: obj=$(round(obj_val, sigdigits=6)), mass=$(round(current_mass, sigdigits=6))")
            else
                log_msg("[OPT] Iter $iter_number: obj=$(round(obj_val, sigdigits=6)), delta=$(round(rel_change, sigdigits=4)), mass=$(round(current_mass, sigdigits=6))")
            end

            if !isnothing(rel_change) && rel_change < tol
                converged = true
                termination_reason = "converged"
                log_msg("[OPT] Converged at iteration $iter_number")
                break
            end

            if local_iter == max_iter
                break
            end

            x_vec = if objective == :min_compliance
                _oc_update(x_vec, dobj_dx, mass_coeffs, ones(length(groups)), variable_mass_target, x_min_vec, x_max_vec, move_limit_vec, eta)
            else
                _gradient_update(x_vec, dobj_dx, mass_coeffs, ones(length(groups)), variable_mass_target, x_min_vec, x_max_vec, move_limit_vec)
            end
        end
    end

    if !converged && termination_reason == "max_iter"
        log_msg("[OPT] Reached max_iter without satisfying convergence tolerance")
    end

    _apply_grouped_sizing_values!(model, groups, x_vec)

    return _grouped_sizing_result_payload(
        objective, history, iterations, groups, initial_x_vec, x_vec,
        mass_initial, mass_target, converged, termination_reason,
        model, thickness_derivative_method, bar_area_derivative_method, grouped_kinds
    )
end

function _current_grouped_sizing_vector(model, groups)
    x = zeros(length(groups))
    for (i, group) in enumerate(groups)
        isempty(group.members) && error("[OPT] Group $(group.label) has no active members")
        x_ref = _current_sizing_value(model, group.members[1])
        for member in group.members[2:end]
            x_member = _current_sizing_value(model, member)
            abs(x_member - x_ref) <= 1e-10 * max(1.0, abs(x_ref), abs(x_member)) ||
                error("[OPT] Group $(group.label) is not synchronized across its member properties")
        end
        x[i] = x_ref
    end
    return x
end

function _current_sizing_value(model, var::SizingVarData)
    if var.kind == :shell_thickness
        return Float64(model["PSHELLs"][string(var.pid_new)]["T"])
    else
        return Float64(model["PBARLs"][string(var.pid_new)]["A"])
    end
end

function _set_sizing_value!(model, var::SizingVarData, value::Float64)
    if var.kind == :shell_thickness
        model["PSHELLs"][string(var.pid_new)]["T"] = value
    else
        model["PBARLs"][string(var.pid_new)]["A"] = value
    end
    return nothing
end

function _apply_grouped_sizing_values!(model, groups, x_vec)
    for (i, group) in enumerate(groups)
        for member in group.members
            _set_sizing_value!(model, member, x_vec[i])
        end
    end
    return nothing
end

function _grouped_design_value_map(groups, x_vec)
    values = Dict{String,Float64}()
    for (i, group) in enumerate(groups)
        values[group.label] = x_vec[i]
    end
    return values
end

@inline _grouped_variable_mass(groups::AbstractVector{SizingGroupData}, x_vec) =
    sum(x_vec[i] * groups[i].mass_coeff for i in eachindex(groups))

@inline _grouped_fixed_mass(groups::AbstractVector{SizingGroupData}) =
    sum(group.fixed_mass for group in groups)

@inline _grouped_total_mass(groups::AbstractVector{SizingGroupData}, x_vec) =
    _grouped_variable_mass(groups, x_vec) + _grouped_fixed_mass(groups)

@inline _grouped_variable_mass(group::SizingGroupData, x_value::Float64) =
    x_value * group.mass_coeff

@inline _grouped_total_mass(group::SizingGroupData, x_value::Float64) =
    _grouped_variable_mass(group, x_value) + group.fixed_mass

function _scale_design_to_mass_target(x_vec, mass_coeffs, mass_target, x_min_vec, x_max_vec)
    x_new = clamp.(copy(x_vec), x_min_vec, x_max_vec)
    for _ in 1:20
        mass_curr = sum(x_new .* mass_coeffs)
        mass_curr <= mass_target * (1 + 1e-8) && break

        free = [i for i in eachindex(x_new) if x_new[i] > x_min_vec[i] + 1e-12]
        isempty(free) && break

        reducible_mass = sum((x_new[i] - x_min_vec[i]) * mass_coeffs[i] for i in free)
        reducible_mass > 1e-30 || break

        needed = mass_curr - mass_target
        reduction_fraction = clamp(needed / reducible_mass, 0.0, 1.0)
        for i in free
            x_new[i] = x_min_vec[i] + (1.0 - reduction_fraction) * (x_new[i] - x_min_vec[i])
        end
        x_new = clamp.(x_new, x_min_vec, x_max_vec)
    end
    return x_new
end

function _grouped_kind_element_map(groups, x_vec, kind::Symbol)
    values = Dict{String,Float64}()
    for (i, group) in enumerate(groups)
        for member in group.members
            member.kind == kind || continue
            values[string(member.eid)] = x_vec[i]
        end
    end
    return values
end

function _grouped_metadata(groups)
    metadata = Any[]
    for group in groups
        kinds = sort!(collect(Set(var.kind for var in group.members)))
        element_ids_by_kind = Dict{String, Any}()
        pid_ids_by_kind = Dict{String, Any}()
        for kind in kinds
            member_subset = [member for member in group.members if member.kind == kind]
            element_ids_by_kind[String(kind)] = sort!([member.eid for member in member_subset])
            pid_ids_by_kind[String(kind)] = sort!([member.pid_new for member in member_subset])
        end
        push!(metadata, Dict(
            "label" => group.label,
            "design_var_id" => group.design_var_id,
            "design_variable_types" => [String(kind) for kind in kinds],
            "relation_ids" => copy(group.relation_ids),
            "property_ids_by_kind" => Dict(String(kind) => copy(ids) for (kind, ids) in group.property_ids_by_kind),
            "element_ids_by_kind" => element_ids_by_kind,
            "internal_pid_ids_by_kind" => pid_ids_by_kind,
            "member_count" => length(group.members),
            "initial_value" => group.x0,
            "mass_coefficient" => group.mass_coeff,
            "fixed_mass" => group.fixed_mass,
        ))
    end
    return metadata
end

function _grouped_sizing_objective_and_sensitivity(results, model, groups, objective;
    thickness_derivative_method=nothing,
    bar_area_derivative_method=nothing)

    if objective == :min_compliance
        sc = results["subcases"][1]
        u = sc["u_analysis"]
        K = results["K"]
        obj = dot(u, K * u)

        id_map = results["id_map"]
        X = results["node_coords"]
        node_R = results["node_R"]
        ndof = results["ndof"]

        grad = zeros(length(groups))
        for (i, group) in enumerate(groups)
            for dv in _grouped_sizing_dvs(group, thickness_derivative_method, bar_area_derivative_method)
                dKdx_u = compute_dKdx_u(dv, model, id_map, X, node_R, u, ndof)
                grad[i] += -dot(u, dKdx_u)
            end
        end
        return obj, grad
    elseif objective == :max_buckling
        eigenvalues = results["eigenvalues"]
        isempty(eigenvalues) && error("[OPT] No buckling eigenvalues found")
        obj = eigenvalues[1]

        design_vars = Any[]
        group_internal_ids = Dict{String, Vector{String}}()
        for group in groups
            ids = String[]
            for dv in _grouped_sizing_dvs(group, thickness_derivative_method, bar_area_derivative_method; include_id=true)
                push!(design_vars, dv)
                push!(ids, dv["id"])
            end
            group_internal_ids[group.label] = ids
        end

        config = Dict(
            "responses" => [Dict("id" => "dummy", "type" => "displacement", "grid" => 1, "dof" => 1, "subcase" => 1)],
            "design_variables" => design_vars,
        )

        config_path, tmp_io = mktemp()
        close(tmp_io)
        try
            open(config_path, "w") do f
                JSON.print(f, config, 2)
            end
            adj = solve_adjoint_buckling(results, config_path)

            grad = zeros(length(groups))
            for (i, group) in enumerate(groups)
                group_sens_total = 0.0
                for dv_id in get(group_internal_ids, group.label, String[])
                    if haskey(adj["sensitivities"], "mode_1") && haskey(adj["sensitivities"]["mode_1"], dv_id)
                        for val in values(adj["sensitivities"]["mode_1"][dv_id])
                            group_sens_total += Float64(val)
                        end
                    end
                end
                grad[i] = group_sens_total
            end
            return obj, grad
        finally
            rm(config_path; force=true)
        end
    end

    error("[OPT] Unknown objective: $objective")
end

function _grouped_sizing_dvs(group::SizingGroupData, thickness_derivative_method, bar_area_derivative_method; include_id::Bool=false)
    dvs = Dict{String,Any}[]
    kinds = sort!(collect(Set(var.kind for var in group.members)))
    for kind in kinds
        pids = sort!([member.pid_new for member in group.members if member.kind == kind])
        dv = Dict{String,Any}(
            "type" => String(kind),
            "pids" => pids,
        )
        if include_id
            dv["id"] = "$(group.label)__$(String(kind))"
        end

        method = if kind == :shell_thickness
            thickness_derivative_method
        else
            bar_area_derivative_method
        end
        if !isnothing(method)
            dv["method"] = String(Symbol(method))
        end
        push!(dvs, dv)
    end
    return dvs
end

function _grouped_sizing_result_payload(
    objective, history, iterations, groups, initial_x_vec, x_vec,
    mass_initial, mass_target, converged, termination_reason,
    model, thickness_derivative_method, bar_area_derivative_method, kinds
)
    fixed_mass = _grouped_fixed_mass(groups)
    variable_mass_initial = _grouped_variable_mass(groups, initial_x_vec)
    variable_mass_target = max(mass_target - fixed_mass, 0.0)
    variable_mass_final = _grouped_variable_mass(groups, x_vec)
    final_mass = variable_mass_final + fixed_mass
    return _finalize_optimization_result!(Dict{String,Any}(
        "objective" => String(objective),
        "history" => copy(history),
        "iterations" => deepcopy(iterations),
        "design_variable_types" => [String(kind) for kind in kinds],
        "design_variables" => _grouped_design_value_map(groups, x_vec),
        "initial_design_variables" => _grouped_design_value_map(groups, initial_x_vec),
        "shell_thicknesses" => _grouped_kind_element_map(groups, x_vec, :shell_thickness),
        "initial_shell_thicknesses" => _grouped_kind_element_map(groups, initial_x_vec, :shell_thickness),
        "bar_areas" => _grouped_kind_element_map(groups, x_vec, :bar_area),
        "initial_bar_areas" => _grouped_kind_element_map(groups, initial_x_vec, :bar_area),
        "group_metadata" => _grouped_metadata(groups),
        "converged" => converged,
        "n_iter" => length(iterations),
        "design_variable_count" => length(groups),
        "fixed_mass" => fixed_mass,
        "variable_mass_initial" => variable_mass_initial,
        "variable_mass_target" => variable_mass_target,
        "variable_mass_final" => variable_mass_final,
        "mass_initial" => mass_initial,
        "mass_target" => mass_target,
        "final_mass" => final_mass,
        "mass_ratio" => final_mass / max(mass_initial, 1e-30),
        "termination_reason" => termination_reason,
        "checkpoint_path" => nothing,
        "thickness_derivative_method" => isnothing(thickness_derivative_method) ? nothing : String(Symbol(thickness_derivative_method)),
        "bar_area_derivative_method" => isnothing(bar_area_derivative_method) ? nothing : String(Symbol(bar_area_derivative_method)),
        "restart" => nothing,
        "model" => model,
    ))
end

function _evaluate_grouped_static_response(results, model, response_family::Symbol, response_spec::Dict{String,Any})
    sc = results["subcases"][1]
    u = sc["u_analysis"]
    if response_family == :compliance
        return dot(u, results["K"] * u)
    elseif response_family == :displacement
        return evaluate_response(
            response_spec, u, model, results["id_map"], results["ndof"],
            results["node_coords"], results["node_R"]
        )
    end
    error("[OPT] Unsupported static grouped response family: $response_family")
end

function _grouped_static_response_and_sensitivity(results, model, groups, response_family::Symbol, response_spec::Dict{String,Any};
    thickness_derivative_method=nothing,
    bar_area_derivative_method=nothing)

    sc = results["subcases"][1]
    u = sc["u_analysis"]
    id_map = results["id_map"]
    X = results["node_coords"]
    node_R = results["node_R"]
    ndof = results["ndof"]

    response_value = _evaluate_grouped_static_response(results, model, response_family, response_spec)
    grad = zeros(length(groups))

    if response_family == :compliance
        for (i, group) in enumerate(groups)
            for dv in _grouped_sizing_dvs(group, thickness_derivative_method, bar_area_derivative_method)
                dKdx_u = compute_dKdx_u(dv, model, id_map, X, node_R, u, ndof)
                grad[i] += -dot(u, dKdx_u)
            end
        end
        return response_value, grad
    elseif response_family == :displacement
        dr_du_full = compute_dr_du(response_spec, u, model, id_map, ndof, X, node_R)
        fixed_dofs_sc = sc["fixed_dofs"]
        free_dofs = sort(collect(setdiff(1:ndof, fixed_dofs_sc)))
        K_ff = results["K"][free_dofs, free_dofs]
        K_fact = cholesky(Symmetric(K_ff))
        lambda_f = K_fact \ dr_du_full[free_dofs]
        lambda_full = zeros(ndof)
        lambda_full[free_dofs] = lambda_f

        for (i, group) in enumerate(groups)
            for dv in _grouped_sizing_dvs(group, thickness_derivative_method, bar_area_derivative_method)
                dKdx_u = compute_dKdx_u(dv, model, id_map, X, node_R, u, ndof)
                grad[i] += -dot(lambda_full, dKdx_u)
            end
        end
        return response_value, grad
    end

    error("[OPT] Unsupported grouped static response family: $response_family")
end

function optimize_grouped_static_response(
    model::Dict, solve_fn::Function, groups::Vector{SizingGroupData};
    response_family::Symbol,
    response_spec::Dict{String,Any},
    x_min = 1e-4,
    x_max = 0.0,
    mass_target::Float64,
    max_iter::Int = 50,
    tol::Float64 = 1e-3,
    move_limit = 0.2,
    eta::Float64 = 0.5,
    thickness_derivative_method::Union{Nothing,Symbol,String} = nothing,
    bar_area_derivative_method::Union{Nothing,Symbol,String} = nothing,
    capture_solver_diagnostics::Bool = true)

    n_group = length(groups)
    n_group == 0 && error("[OPT] No grouped sizing variables were provided")
    mass_target > 0.0 || error("[OPT] Grouped static-response minimization requires a positive mass target")

    initial_x_vec = _current_grouped_sizing_vector(model, groups)
    mass_coeffs = [group.mass_coeff for group in groups]
    fixed_mass = _grouped_fixed_mass(groups)
    variable_mass_initial = _grouped_variable_mass(groups, initial_x_vec)
    mass_initial = variable_mass_initial + fixed_mass
    variable_mass_target = max(mass_target - fixed_mass, 0.0)

    x_min_vec = _design_limit_vector(x_min, n_group, "group lower bounds")
    x_max_vec = _design_limit_vector(x_max, n_group, "group upper bounds")
    if any(v <= 0.0 for v in x_max_vec)
        fallback = 10.0 * maximum(initial_x_vec)
        x_max_vec = [v <= 0.0 ? fallback : v for v in x_max_vec]
    end
    move_limit_vec = _design_limit_vector(move_limit, n_group, "group move limits")

    x_vec = clamp.(copy(initial_x_vec), x_min_vec, x_max_vec)
    history = Float64[]
    iterations = Any[]
    converged = false
    termination_reason = "max_iter"
    last_results = nothing
    last_response_value = nothing

    grouped_kinds = sort!(collect(Set(var.kind for group in groups for var in group.members)))
    response_label = response_family == :displacement ?
        "displacement(grid=$(response_spec["grid"]), dof=$(response_spec["dof"]))" :
        String(response_family)

    log_msg("[OPT] $n_group grouped sizing variables ($(join(string.(grouped_kinds), ", "))), objective=min_$response_label, mass_target=$(round(mass_target, sigdigits=6))")
    fixed_mass > 0.0 &&
        log_msg("[OPT] Fixed non-structural mass: $(round(fixed_mass, sigdigits=6)); grouped variable target: $(round(variable_mass_target, sigdigits=6))")
    log_msg("[OPT] Group bounds prepared, max_iter=$max_iter")

    if max_iter <= 0
        termination_reason = "no_iterations_requested"
    else
        for local_iter in 1:max_iter
            iter_number = length(history) + 1
            _apply_grouped_sizing_values!(model, groups, x_vec)
            results = solve_fn(model)

            response_value, response_grad = _grouped_static_response_and_sensitivity(
                results, model, groups, response_family, response_spec;
                thickness_derivative_method=thickness_derivative_method,
                bar_area_derivative_method=bar_area_derivative_method,
            )
            last_results = results
            last_response_value = response_value

            current_variable_mass = _grouped_variable_mass(groups, x_vec)
            current_mass = current_variable_mass + fixed_mass
            rel_change = isempty(history) ? nothing :
                abs(response_value - history[end]) / max(abs(history[end]), 1e-30)
            grad_norm = norm(response_grad)

            iter_record = Dict{String,Any}(
                "iteration" => iter_number,
                "objective" => response_value,
                "relative_change" => rel_change,
                "response_value" => response_value,
                "response_family" => String(response_family),
                "mass" => current_mass,
                "variable_mass" => current_variable_mass,
                "fixed_mass" => fixed_mass,
                "mass_ratio" => current_mass / max(mass_initial, 1e-30),
                "mass_constraint_violation" => max(current_mass - mass_target, 0.0),
                "min_design_value" => minimum(x_vec),
                "max_design_value" => maximum(x_vec),
                "gradient_norm" => grad_norm,
                "objective_type" => "min_static_response",
                "design_variable_types" => [String(kind) for kind in grouped_kinds],
                "design_variables" => _grouped_design_value_map(groups, x_vec),
                "shell_thicknesses" => _grouped_kind_element_map(groups, x_vec, :shell_thickness),
                "bar_areas" => _grouped_kind_element_map(groups, x_vec, :bar_area),
            )
            if response_family == :displacement
                iter_record["response_grid"] = Int(response_spec["grid"])
                iter_record["response_dof"] = Int(response_spec["dof"])
            end
            if !isnothing(thickness_derivative_method)
                iter_record["thickness_derivative_method"] = String(Symbol(thickness_derivative_method))
            end
            if !isnothing(bar_area_derivative_method)
                iter_record["bar_area_derivative_method"] = String(Symbol(bar_area_derivative_method))
            end
            if capture_solver_diagnostics
                iter_record["solver_diagnostics"] = _extract_optimization_solver_diagnostics(results, :min_compliance)
            end

            push!(history, response_value)
            push!(iterations, iter_record)

            if isnothing(rel_change)
                log_msg("[OPT] Iter $iter_number: response=$(round(response_value, sigdigits=6)), mass=$(round(current_mass, sigdigits=6))")
            else
                log_msg("[OPT] Iter $iter_number: response=$(round(response_value, sigdigits=6)), delta=$(round(rel_change, sigdigits=4)), mass=$(round(current_mass, sigdigits=6))")
            end

            if !isnothing(rel_change) && rel_change < tol
                converged = true
                termination_reason = "converged"
                log_msg("[OPT] Converged at iteration $iter_number")
                break
            end

            if local_iter == max_iter
                break
            end

            x_candidate = _oc_update(x_vec, response_grad, mass_coeffs, ones(length(groups)),
                variable_mass_target, x_min_vec, x_max_vec, move_limit_vec, eta)
            x_vec = _scale_design_to_mass_target(x_candidate, mass_coeffs, variable_mass_target, x_min_vec, x_max_vec)
        end
    end

    if !converged && termination_reason == "max_iter"
        log_msg("[OPT] Reached max_iter without satisfying convergence tolerance")
    end

    _apply_grouped_sizing_values!(model, groups, x_vec)
    final_results = last_results
    if !(final_results isa AbstractDict)
        final_results = solve_fn(model)
        last_response_value = nothing
    end

    return _grouped_static_response_result_payload(
        history, iterations, groups, initial_x_vec, x_vec,
        response_family, response_spec, mass_initial, mass_target,
        converged, termination_reason, model, final_results,
        thickness_derivative_method, bar_area_derivative_method, grouped_kinds
        ; final_response_value=last_response_value
    )
end

function optimize_grouped_static_constraints(
    model::Dict, solve_fn::Function, groups::Vector{SizingGroupData};
    response_constraints,
    x_min = 1e-4,
    x_max = 0.0,
    mass_target::Float64,
    max_iter::Int = 50,
    tol::Float64 = 1e-3,
    move_limit = 0.2,
    eta::Float64 = 0.5,
    thickness_derivative_method::Union{Nothing,Symbol,String} = nothing,
    bar_area_derivative_method::Union{Nothing,Symbol,String} = nothing,
    capture_solver_diagnostics::Bool = true)

    normalized_constraints = _normalize_grouped_static_response_constraints(
        response_constraints=response_constraints,
    )
    length(normalized_constraints) == 1 && return optimize_grouped_static_response(
        model, solve_fn, groups;
        response_family=normalized_constraints[1]["family"],
        response_spec=normalized_constraints[1]["spec"],
        x_min=x_min,
        x_max=x_max,
        mass_target=mass_target,
        max_iter=max_iter,
        tol=tol,
        move_limit=move_limit,
        eta=eta,
        thickness_derivative_method=thickness_derivative_method,
        bar_area_derivative_method=bar_area_derivative_method,
        capture_solver_diagnostics=capture_solver_diagnostics,
    )

    n_group = length(groups)
    n_group == 0 && error("[OPT] No grouped sizing variables were provided")
    mass_target > 0.0 || error("[OPT] Grouped static-constraint reduction requires a positive mass target")

    initial_x_vec = _current_grouped_sizing_vector(model, groups)
    mass_coeffs = [group.mass_coeff for group in groups]
    fixed_mass = _grouped_fixed_mass(groups)
    variable_mass_initial = _grouped_variable_mass(groups, initial_x_vec)
    mass_initial = variable_mass_initial + fixed_mass
    variable_mass_target = max(mass_target - fixed_mass, 0.0)

    x_min_vec = _design_limit_vector(x_min, n_group, "group lower bounds")
    x_max_vec = _design_limit_vector(x_max, n_group, "group upper bounds")
    if any(v <= 0.0 for v in x_max_vec)
        fallback = 10.0 * maximum(initial_x_vec)
        x_max_vec = [v <= 0.0 ? fallback : v for v in x_max_vec]
    end
    move_limit_vec = _design_limit_vector(move_limit, n_group, "group move limits")

    x_vec = clamp.(copy(initial_x_vec), x_min_vec, x_max_vec)
    history = Float64[]
    iterations = Any[]
    converged = false
    termination_reason = "max_iter"
    last_results = nothing
    last_evaluated_constraints = nothing

    grouped_kinds = sort!(collect(Set(var.kind for group in groups for var in group.members)))
    log_msg("[OPT] $n_group grouped sizing variables ($(join(string.(grouped_kinds), ", "))), objective=min_static_constraints, mass_target=$(round(mass_target, sigdigits=6)), constraints=$(length(normalized_constraints))")
    fixed_mass > 0.0 &&
        log_msg("[OPT] Fixed non-structural mass: $(round(fixed_mass, sigdigits=6)); grouped variable target: $(round(variable_mass_target, sigdigits=6))")
    log_msg("[OPT] Group bounds prepared, max_iter=$max_iter")

    if max_iter <= 0
        termination_reason = "no_iterations_requested"
    else
        for local_iter in 1:max_iter
            iter_number = length(history) + 1
            _apply_grouped_sizing_values!(model, groups, x_vec)
            results = solve_fn(model)

            evaluated_constraints, constraint_merit, response_grad, constraint_ratios, constraint_weights =
                _grouped_static_constraints_and_sensitivity(
                    results, model, groups, normalized_constraints;
                    thickness_derivative_method=thickness_derivative_method,
                    bar_area_derivative_method=bar_area_derivative_method,
                )
            last_results = results
            last_evaluated_constraints = deepcopy(evaluated_constraints)
            active_constraint = _grouped_static_active_constraint(evaluated_constraints)

            current_variable_mass = _grouped_variable_mass(groups, x_vec)
            current_mass = current_variable_mass + fixed_mass
            max_violation = _grouped_static_constraints_max_violation(evaluated_constraints)
            rel_change = isempty(history) ? nothing :
                abs(constraint_merit - history[end]) / max(abs(history[end]), 1e-30)
            grad_norm = norm(response_grad)

            iter_record = Dict{String,Any}(
                "iteration" => iter_number,
                "objective" => constraint_merit,
                "relative_change" => rel_change,
                "response_value" => Float64(active_constraint["response_value"]),
                "response_family" => String(active_constraint["response_family"]),
                "response_ratio" => Float64(get(active_constraint, "response_ratio", 0.0)),
                "response_upper_bound" => Float64(active_constraint["response_upper_bound"]),
                "response_constraint_violation" => Float64(active_constraint["response_constraint_violation"]),
                "response_constraints" => deepcopy(evaluated_constraints),
                "constraint_count" => length(evaluated_constraints),
                "active_constraint_index" => Int(active_constraint["constraint_index"]),
                "active_response_id" => get(active_constraint, "response_id", nothing),
                "constraint_merit" => constraint_merit,
                "max_constraint_ratio" => maximum(constraint_ratios),
                "constraint_weights" => copy(constraint_weights),
                "max_constraint_violation" => max_violation,
                "mass" => current_mass,
                "variable_mass" => current_variable_mass,
                "fixed_mass" => fixed_mass,
                "mass_ratio" => current_mass / max(mass_initial, 1e-30),
                "mass_constraint_violation" => max(current_mass - mass_target, 0.0),
                "min_design_value" => minimum(x_vec),
                "max_design_value" => maximum(x_vec),
                "gradient_norm" => grad_norm,
                "objective_type" => "min_static_constraints",
                "design_variable_types" => [String(kind) for kind in grouped_kinds],
                "design_variables" => _grouped_design_value_map(groups, x_vec),
                "shell_thicknesses" => _grouped_kind_element_map(groups, x_vec, :shell_thickness),
                "bar_areas" => _grouped_kind_element_map(groups, x_vec, :bar_area),
            )
            if get(active_constraint, "response_family", nothing) == "displacement"
                iter_record["response_grid"] = get(active_constraint, "constraint_grid", nothing)
                iter_record["response_dof"] = get(active_constraint, "constraint_dof", nothing)
            end
            if !isnothing(thickness_derivative_method)
                iter_record["thickness_derivative_method"] = String(Symbol(thickness_derivative_method))
            end
            if !isnothing(bar_area_derivative_method)
                iter_record["bar_area_derivative_method"] = String(Symbol(bar_area_derivative_method))
            end
            if capture_solver_diagnostics
                iter_record["solver_diagnostics"] = _extract_optimization_solver_diagnostics(results, :min_compliance)
            end

            push!(history, constraint_merit)
            push!(iterations, iter_record)

            if isnothing(rel_change)
                log_msg("[OPT] Iter $iter_number: active_$(active_constraint["response_family"])=$(round(Float64(active_constraint["response_value"]), sigdigits=6)), merit=$(round(constraint_merit, sigdigits=6)), max_violation=$(round(max_violation, sigdigits=6)), mass=$(round(current_mass, sigdigits=6))")
            else
                log_msg("[OPT] Iter $iter_number: active_$(active_constraint["response_family"])=$(round(Float64(active_constraint["response_value"]), sigdigits=6)), merit=$(round(constraint_merit, sigdigits=6)), max_violation=$(round(max_violation, sigdigits=6)), delta=$(round(rel_change, sigdigits=4)), mass=$(round(current_mass, sigdigits=6))")
            end

            mass_target_satisfied = current_mass <= mass_target * (1 + 1e-6)
            if _grouped_static_constraints_feasible(evaluated_constraints) && mass_target_satisfied
                converged = true
                termination_reason = "feasible"
                log_msg("[OPT] Reached feasible routed static-constraint set at iteration $iter_number")
                break
            end

            if !isnothing(rel_change) && rel_change < tol && mass_target_satisfied
                converged = true
                termination_reason = "constraint_reduction_converged"
                log_msg("[OPT] Static-constraint reduction stagnated at iteration $iter_number")
                break
            end

            if local_iter == max_iter
                break
            end

            x_candidate = _gradient_update(x_vec, -response_grad, mass_coeffs, ones(length(groups)),
                variable_mass_target, x_min_vec, x_max_vec, move_limit_vec)
            x_candidate = _scale_design_to_mass_target(x_candidate, mass_coeffs, variable_mass_target, x_min_vec, x_max_vec)
            best_trial_x = copy(x_candidate)
            best_trial_violation = Inf
            best_trial_merit = Inf
            for step_scale in (1.0, 0.5, 0.25, 0.125, 0.0625)
                x_trial = x_vec .+ step_scale .* (x_candidate .- x_vec)
                x_trial = clamp.(x_trial, x_min_vec, x_max_vec)
                x_trial = _scale_design_to_mass_target(x_trial, mass_coeffs, variable_mass_target, x_min_vec, x_max_vec)
                _apply_grouped_sizing_values!(model, groups, x_trial)
                trial_results = solve_fn(model)
                trial_constraints = _evaluate_grouped_static_constraints(trial_results, model, normalized_constraints)
                trial_merit, _, _ = _grouped_static_constraint_merit(trial_constraints)
                trial_violation = _grouped_static_constraints_max_violation(trial_constraints)
                if trial_violation < best_trial_violation - 1e-10 ||
                    (trial_violation <= best_trial_violation + 1e-12 && trial_merit < best_trial_merit - 1e-6)
                    best_trial_x = copy(x_trial)
                    best_trial_violation = trial_violation
                    best_trial_merit = trial_merit
                end
            end
            x_vec = best_trial_x
        end
    end

    if !converged && termination_reason == "max_iter"
        log_msg("[OPT] Reached max_iter without satisfying the routed static constraints")
    end

    _apply_grouped_sizing_values!(model, groups, x_vec)
    final_results = last_results
    if !(final_results isa AbstractDict)
        final_results = solve_fn(model)
        last_evaluated_constraints = nothing
    end

    return _grouped_static_constraints_result_payload(
        history, iterations, groups, initial_x_vec, x_vec,
        normalized_constraints, mass_initial, mass_target,
        converged, termination_reason, model, final_results,
        thickness_derivative_method, bar_area_derivative_method, grouped_kinds
        ; evaluated_constraints=last_evaluated_constraints
    )
end

function _grouped_static_response_result_payload(
    history, iterations, groups, initial_x_vec, x_vec,
    response_family::Symbol, response_spec::Dict{String,Any},
    mass_initial::Float64, mass_target::Float64,
    converged::Bool, termination_reason::String, model, final_results,
    thickness_derivative_method, bar_area_derivative_method, kinds;
    final_response_value=nothing
)
    fixed_mass = _grouped_fixed_mass(groups)
    variable_mass_initial = _grouped_variable_mass(groups, initial_x_vec)
    variable_mass_target = max(mass_target - fixed_mass, 0.0)
    variable_mass_final = _grouped_variable_mass(groups, x_vec)
    final_mass = variable_mass_final + fixed_mass
    if isnothing(final_response_value)
        final_response_value = _evaluate_grouped_static_response(final_results, model, response_family, response_spec)
    end
    result = Dict{String,Any}(
        "objective" => "min_static_response",
        "history" => copy(history),
        "iterations" => deepcopy(iterations),
        "design_variable_types" => [String(kind) for kind in kinds],
        "design_variables" => _grouped_design_value_map(groups, x_vec),
        "initial_design_variables" => _grouped_design_value_map(groups, initial_x_vec),
        "shell_thicknesses" => _grouped_kind_element_map(groups, x_vec, :shell_thickness),
        "initial_shell_thicknesses" => _grouped_kind_element_map(groups, initial_x_vec, :shell_thickness),
        "bar_areas" => _grouped_kind_element_map(groups, x_vec, :bar_area),
        "initial_bar_areas" => _grouped_kind_element_map(groups, initial_x_vec, :bar_area),
        "group_metadata" => _grouped_metadata(groups),
        "response_family" => String(response_family),
        "response_value" => final_response_value,
        "converged" => converged,
        "n_iter" => length(iterations),
        "design_variable_count" => length(groups),
        "fixed_mass" => fixed_mass,
        "variable_mass_initial" => variable_mass_initial,
        "variable_mass_target" => variable_mass_target,
        "variable_mass_final" => variable_mass_final,
        "mass_initial" => mass_initial,
        "mass_target" => mass_target,
        "final_mass" => final_mass,
        "mass_ratio" => final_mass / max(mass_initial, 1e-30),
        "termination_reason" => termination_reason,
        "checkpoint_path" => nothing,
        "thickness_derivative_method" => isnothing(thickness_derivative_method) ? nothing : String(Symbol(thickness_derivative_method)),
        "bar_area_derivative_method" => isnothing(bar_area_derivative_method) ? nothing : String(Symbol(bar_area_derivative_method)),
        "restart" => nothing,
        "model" => model,
        "_final_results" => final_results,
    )
    if response_family == :displacement
        result["response_grid"] = Int(response_spec["grid"])
        result["response_dof"] = Int(response_spec["dof"])
    end
    return _finalize_optimization_result!(result)
end

function _grouped_static_constraints_result_payload(
    history, iterations, groups, initial_x_vec, x_vec,
    response_constraints::Vector{Dict{String,Any}},
    mass_initial::Float64, mass_target::Float64,
    converged::Bool, termination_reason::String, model, final_results,
    thickness_derivative_method, bar_area_derivative_method, kinds;
    evaluated_constraints=nothing
)
    fixed_mass = _grouped_fixed_mass(groups)
    variable_mass_initial = _grouped_variable_mass(groups, initial_x_vec)
    variable_mass_target = max(mass_target - fixed_mass, 0.0)
    variable_mass_final = _grouped_variable_mass(groups, x_vec)
    final_mass = variable_mass_final + fixed_mass
    if !(evaluated_constraints isa AbstractVector)
        evaluated_constraints = _evaluate_grouped_static_constraints(final_results, model, response_constraints)
    end
    active_constraint = isempty(evaluated_constraints) ? nothing :
        _grouped_static_active_constraint(evaluated_constraints)
    result = Dict{String,Any}(
        "objective" => "min_static_constraints",
        "history" => copy(history),
        "iterations" => deepcopy(iterations),
        "design_variable_types" => [String(kind) for kind in kinds],
        "design_variables" => _grouped_design_value_map(groups, x_vec),
        "initial_design_variables" => _grouped_design_value_map(groups, initial_x_vec),
        "shell_thicknesses" => _grouped_kind_element_map(groups, x_vec, :shell_thickness),
        "initial_shell_thicknesses" => _grouped_kind_element_map(groups, initial_x_vec, :shell_thickness),
        "bar_areas" => _grouped_kind_element_map(groups, x_vec, :bar_area),
        "initial_bar_areas" => _grouped_kind_element_map(groups, initial_x_vec, :bar_area),
        "group_metadata" => _grouped_metadata(groups),
        "response_family" => isnothing(active_constraint) ? nothing : String(active_constraint["response_family"]),
        "response_upper_bound" => isnothing(active_constraint) ? nothing : Float64(active_constraint["response_upper_bound"]),
        "response_value" => isnothing(active_constraint) ? nothing : Float64(active_constraint["response_value"]),
        "response_constraint_violation" => isnothing(active_constraint) ? nothing : Float64(active_constraint["response_constraint_violation"]),
        "response_constraints" => deepcopy(evaluated_constraints),
        "constraint_count" => length(response_constraints),
        "active_constraint_index" => isnothing(active_constraint) ? nothing : Int(active_constraint["constraint_index"]),
        "active_response_id" => isnothing(active_constraint) ? nothing : get(active_constraint, "response_id", nothing),
        "max_constraint_violation" => isempty(evaluated_constraints) ? nothing : _grouped_static_constraints_max_violation(evaluated_constraints),
        "converged" => converged,
        "n_iter" => length(iterations),
        "design_variable_count" => length(groups),
        "fixed_mass" => fixed_mass,
        "variable_mass_initial" => variable_mass_initial,
        "variable_mass_target" => variable_mass_target,
        "variable_mass_final" => variable_mass_final,
        "mass_initial" => mass_initial,
        "mass_target" => mass_target,
        "final_mass" => final_mass,
        "mass_ratio" => final_mass / max(mass_initial, 1e-30),
        "termination_reason" => termination_reason,
        "checkpoint_path" => nothing,
        "thickness_derivative_method" => isnothing(thickness_derivative_method) ? nothing : String(Symbol(thickness_derivative_method)),
        "bar_area_derivative_method" => isnothing(bar_area_derivative_method) ? nothing : String(Symbol(bar_area_derivative_method)),
        "restart" => nothing,
        "model" => model,
        "_final_results" => final_results,
    )
    if !isnothing(active_constraint) && get(active_constraint, "response_family", nothing) == "displacement"
        result["response_grid"] = get(active_constraint, "constraint_grid", nothing)
        result["response_dof"] = get(active_constraint, "constraint_dof", nothing)
    end
    return _finalize_optimization_result!(result)
end

function _normalize_grouped_static_response_constraints(;
    response_constraints=nothing,
    response_family::Union{Nothing,Symbol}=nothing,
    response_spec::Union{Nothing,Dict{String,Any}}=nothing,
    response_upper_bound::Union{Nothing,Float64}=nothing)

    if isnothing(response_constraints)
        isnothing(response_family) && error("[OPT] Missing static response family for single-variable mass minimization")
        response_spec isa Dict{String,Any} || error("[OPT] Missing static response specification for single-variable mass minimization")
        isnothing(response_upper_bound) && error("[OPT] Missing static response upper bound for single-variable mass minimization")
        return Dict{String,Any}[Dict(
            "response_id" => nothing,
            "family" => response_family,
            "spec" => deepcopy(response_spec),
            "upper_bound" => Float64(response_upper_bound),
        )]
    end

    normalized = Dict{String,Any}[]
    for constraint in response_constraints
        family = get(constraint, "family", nothing)
        family isa Symbol || error("[OPT] Static routed response constraint is missing a Symbol family")
        spec = get(constraint, "spec", nothing)
        spec isa Dict{String,Any} || error("[OPT] Static routed response constraint is missing its response specification")
        upper_bound = get(constraint, "upper_bound", nothing)
        upper_bound isa Number || error("[OPT] Static routed response constraint is missing a numeric upper bound")
        push!(normalized, Dict{String,Any}(
            "response_id" => get(constraint, "response_id", nothing),
            "family" => family,
            "spec" => deepcopy(spec),
            "upper_bound" => Float64(upper_bound),
        ))
    end
    isempty(normalized) && error("[OPT] Single-variable mass minimization requires at least one static routed response constraint")
    return normalized
end

function _evaluate_grouped_static_constraints(results, model, response_constraints::Vector{Dict{String,Any}})
    evaluated = Dict{String,Any}[]
    for (idx, constraint) in enumerate(response_constraints)
        family = constraint["family"]
        spec = constraint["spec"]
        upper_bound = Float64(constraint["upper_bound"])
        response_value = _evaluate_grouped_static_response(results, model, family, spec)
        upper_scale = max(abs(upper_bound), 1e-30)
        entry = Dict{String,Any}(
            "constraint_index" => idx,
            "response_id" => get(constraint, "response_id", nothing),
            "response_family" => String(family),
            "response_upper_bound" => upper_bound,
            "response_value" => response_value,
            "response_ratio" => response_value / upper_scale,
            "response_margin" => upper_bound - response_value,
            "response_margin_ratio" => (upper_bound - response_value) / upper_scale,
            "response_constraint_violation" => max(response_value - upper_bound, 0.0),
        )
        if family == :displacement
            entry["constraint_grid"] = Int(spec["grid"])
            entry["constraint_dof"] = Int(spec["dof"])
        end
        push!(evaluated, entry)
    end
    return evaluated
end

function _grouped_static_constraints_and_sensitivity(
    results, model, groups, response_constraints::Vector{Dict{String,Any}};
    thickness_derivative_method=nothing,
    bar_area_derivative_method=nothing,
    ks_rho::Float64 = 24.0)

    evaluated = Dict{String,Any}[]
    normalized_grads = Vector{Vector{Float64}}()
    ratios = Float64[]

    for (idx, constraint) in enumerate(response_constraints)
        family = constraint["family"]
        spec = constraint["spec"]
        upper_bound = Float64(constraint["upper_bound"])
        upper_scale = max(abs(upper_bound), 1e-30)
        response_value, response_grad = _grouped_static_response_and_sensitivity(
            results, model, groups, family, spec;
            thickness_derivative_method=thickness_derivative_method,
            bar_area_derivative_method=bar_area_derivative_method,
        )
        response_ratio = response_value / upper_scale
        entry = Dict{String,Any}(
            "constraint_index" => idx,
            "response_id" => get(constraint, "response_id", nothing),
            "response_family" => String(family),
            "response_upper_bound" => upper_bound,
            "response_value" => response_value,
            "response_ratio" => response_ratio,
            "response_margin" => upper_bound - response_value,
            "response_margin_ratio" => (upper_bound - response_value) / upper_scale,
            "response_constraint_violation" => max(response_value - upper_bound, 0.0),
        )
        if family == :displacement
            entry["constraint_grid"] = Int(spec["grid"])
            entry["constraint_dof"] = Int(spec["dof"])
        end
        push!(evaluated, entry)
        push!(ratios, response_ratio)
        push!(normalized_grads, response_grad ./ upper_scale)
    end

    merit, _, weights = _grouped_static_constraint_merit(evaluated; ks_rho=ks_rho)

    aggregate_grad = zeros(length(groups))
    for (weight, grad) in zip(weights, normalized_grads)
        aggregate_grad .+= weight .* grad
    end

    return evaluated, merit, aggregate_grad, ratios, weights
end

function _grouped_static_constraint_merit(constraint_values::Vector{Dict{String,Any}}; ks_rho::Float64 = 24.0)
    ratios = Float64[
        Float64(get(entry, "response_ratio",
            Float64(entry["response_value"]) / max(abs(Float64(entry["response_upper_bound"])), 1e-30)))
        for entry in constraint_values
    ]
    max_ratio = maximum(ratios)
    exp_terms = exp.(clamp.(ks_rho .* (ratios .- max_ratio), -50.0, 0.0))
    exp_sum = sum(exp_terms)
    weights = exp_sum > 0.0 ? exp_terms ./ exp_sum : fill(1.0 / length(ratios), length(ratios))
    merit = max_ratio + log(max(exp_sum, 1e-30)) / ks_rho
    return merit, ratios, weights
end

@inline function _grouped_static_constraints_feasible(constraint_values::Vector{Dict{String,Any}})
    all(Float64(entry["response_constraint_violation"]) <= 0.0 for entry in constraint_values)
end

@inline function _grouped_static_constraints_max_violation(constraint_values::Vector{Dict{String,Any}})
    maximum(Float64(entry["response_constraint_violation"]) for entry in constraint_values)
end

function _grouped_static_active_constraint(constraint_values::Vector{Dict{String,Any}})
    best = constraint_values[1]
    best_score = -Inf
    for entry in constraint_values
        upper_bound = max(abs(Float64(entry["response_upper_bound"])), 1e-30)
        violation = Float64(entry["response_constraint_violation"])
        response_value = Float64(entry["response_value"])
        gap = abs(Float64(entry["response_upper_bound"]) - response_value) / upper_bound
        score = violation > 0.0 ? 1.0 + violation / upper_bound : 1.0 - gap
        if score > best_score
            best = entry
            best_score = score
        end
    end
    return best
end

function _grouped_static_constraints_monotone(lower_constraints::Vector{Dict{String,Any}}, upper_constraints::Vector{Dict{String,Any}})
    length(lower_constraints) == length(upper_constraints) || return false
    for (lower, upper) in zip(lower_constraints, upper_constraints)
        tolerance = max(1e-10, 1e-8 * max(abs(Float64(lower["response_value"])), abs(Float64(upper["response_value"]))))
        Float64(upper["response_value"]) <= Float64(lower["response_value"]) + tolerance || return false
    end
    return true
end

function _grouped_bisection_tolerance(tol::Float64)
    # A pure relative bisection tolerance below machine-useful resolution can
    # leave an otherwise correct exact-search route stuck at max_iter.
    return max(tol, sqrt(eps(Float64)))
end

function _grouped_static_projected_mass_direction(mass_coeffs::Vector{Float64}, constraint_grad::Vector{Float64})
    direction = .-copy(mass_coeffs)
    grad_sq_norm = dot(constraint_grad, constraint_grad)
    if grad_sq_norm > 1e-30
        directional_merit = dot(constraint_grad, direction)
        if directional_merit > 0.0
            direction .-= (directional_merit / grad_sq_norm) .* constraint_grad
        end
    end
    return direction
end

function _grouped_static_mass_polish(
    base_model::Dict, solve_fn::Function, groups::Vector{SizingGroupData}, x_start::Vector{Float64},
    response_constraints::Vector{Dict{String,Any}};
    x_min = 1e-4,
    x_max = 0.0,
    move_limit = 0.2,
    max_iter::Int = 8,
    tol::Float64 = 1e-6,
    iteration_offset::Int = 0,
    thickness_derivative_method::Union{Nothing,Symbol,String} = nothing,
    bar_area_derivative_method::Union{Nothing,Symbol,String} = nothing,
    capture_solver_diagnostics::Bool = true)

    n_group = length(groups)
    n_group == 0 && error("[OPT] Grouped mass polish requires at least one grouped sizing variable")

    x_min_vec = _design_limit_vector(x_min, n_group, "group lower bounds")
    x_max_vec = _design_limit_vector(x_max, n_group, "group upper bounds")
    if any(v <= 0.0 for v in x_max_vec)
        fallback = 10.0 * maximum(x_start)
        x_max_vec = [v <= 0.0 ? fallback : v for v in x_max_vec]
    end
    move_limit_vec = _design_limit_vector(move_limit, n_group, "group move limits")

    fixed_mass = _grouped_fixed_mass(groups)
    mass_coeffs = Float64[group.mass_coeff for group in groups]
    kinds = sort!(collect(Set(var.kind for group in groups for var in group.members)))
    x_vec = clamp.(copy(x_start), x_min_vec, x_max_vec)

    function evaluate_design(x_trial::Vector{Float64})
        working_model = deepcopy(base_model)
        _apply_grouped_sizing_values!(working_model, groups, x_trial)
        results = solve_fn(working_model)
        evaluated_constraints, constraint_merit, response_grad, _, _ =
            _grouped_static_constraints_and_sensitivity(
                results, working_model, groups, response_constraints;
                thickness_derivative_method=thickness_derivative_method,
                bar_area_derivative_method=bar_area_derivative_method,
            )
        variable_mass = _grouped_variable_mass(groups, x_trial)
        total_mass = variable_mass + fixed_mass
        return working_model, results, evaluated_constraints, constraint_merit, response_grad, variable_mass, total_mass
    end

    current_model, current_results, current_constraints, current_merit, current_grad,
        current_variable_mass, current_mass = evaluate_design(x_vec)
    initial_constraints = deepcopy(current_constraints)
    initial_merit = current_merit
    initial_variable_mass = current_variable_mass
    initial_mass = current_mass
    initial_active_constraint = isempty(current_constraints) ? nothing :
        _grouped_static_active_constraint(current_constraints)

    iterations = Any[]
    improved = false
    termination_reason = max_iter <= 0 ? "no_iterations_requested" : "no_feasible_step"
    improvement_tol = max(1e-10, max(tol, 1e-5) * max(initial_mass, 1.0) * 0.05)

    if max_iter > 0 && _grouped_static_constraints_feasible(current_constraints)
        for _ in 1:max_iter
            direction = _grouped_static_projected_mass_direction(mass_coeffs, current_grad)
            for i in eachindex(direction)
                if x_vec[i] <= x_min_vec[i] + 1e-12 && direction[i] < 0.0
                    direction[i] = 0.0
                elseif x_vec[i] >= x_max_vec[i] - 1e-12 && direction[i] > 0.0
                    direction[i] = 0.0
                end
            end

            direction_scale = maximum(abs.(direction))
            if !(direction_scale > 1e-30)
                termination_reason = improved ? "tangent_stagnation" : "no_descent_direction"
                break
            end

            reference_design = max.(abs.(x_vec), x_min_vec)
            step_scale = maximum(move_limit_vec .* reference_design) / direction_scale
            if !(step_scale > 0.0)
                termination_reason = improved ? "move_limit_stagnation" : "no_descent_direction"
                break
            end

            local_lower = max.(x_vec .* (1 .- move_limit_vec), x_min_vec)
            local_upper = min.(x_vec .* (1 .+ move_limit_vec), x_max_vec)

            accepted = false
            previous_mass = current_mass
            previous_variable_mass = current_variable_mass
            previous_constraint_merit = current_merit
            for backtrack in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
                x_trial = x_vec .+ (backtrack * step_scale) .* direction
                x_trial = clamp.(x_trial, local_lower, local_upper)
                x_trial = clamp.(x_trial, x_min_vec, x_max_vec)
                trial_variable_mass = _grouped_variable_mass(groups, x_trial)
                trial_mass = trial_variable_mass + fixed_mass
                trial_mass < current_mass - improvement_tol || continue

                trial_model, trial_results, trial_constraints, trial_merit, trial_grad,
                    trial_variable_mass, trial_mass = evaluate_design(x_trial)
                _grouped_static_constraints_feasible(trial_constraints) || continue

                x_vec = x_trial
                current_model = trial_model
                current_results = trial_results
                current_constraints = trial_constraints
                current_merit = trial_merit
                current_grad = trial_grad
                current_variable_mass = trial_variable_mass
                current_mass = trial_mass

                active_constraint = _grouped_static_active_constraint(current_constraints)
                iter_number = iteration_offset + length(iterations) + 1
                iter_record = Dict{String,Any}(
                    "iteration" => iter_number,
                    "phase" => "mass_polish",
                    "objective" => current_mass,
                    "relative_change" => abs(current_mass - previous_mass) / max(abs(previous_mass), 1e-30),
                    "response_value" => Float64(active_constraint["response_value"]),
                    "response_family" => String(active_constraint["response_family"]),
                    "response_ratio" => Float64(get(active_constraint, "response_ratio", 0.0)),
                    "response_upper_bound" => Float64(active_constraint["response_upper_bound"]),
                    "response_constraint_violation" => Float64(active_constraint["response_constraint_violation"]),
                    "response_constraints" => deepcopy(current_constraints),
                    "constraint_count" => length(current_constraints),
                    "active_constraint_index" => Int(active_constraint["constraint_index"]),
                    "active_response_id" => get(active_constraint, "response_id", nothing),
                    "constraint_merit" => current_merit,
                    "max_constraint_violation" => _grouped_static_constraints_max_violation(current_constraints),
                    "mass" => current_mass,
                    "variable_mass" => current_variable_mass,
                    "fixed_mass" => fixed_mass,
                    "mass_ratio" => current_mass / max(initial_mass, 1e-30),
                    "mass_reduction" => previous_mass - current_mass,
                    "variable_mass_reduction" => previous_variable_mass - current_variable_mass,
                    "constraint_merit_reduction" => previous_constraint_merit - current_merit,
                    "min_design_value" => minimum(x_vec),
                    "max_design_value" => maximum(x_vec),
                    "gradient_norm" => norm(current_grad),
                    "objective_type" => "min_mass",
                    "design_variable_types" => [String(kind) for kind in kinds],
                    "design_variables" => _grouped_design_value_map(groups, x_vec),
                    "shell_thicknesses" => _grouped_kind_element_map(groups, x_vec, :shell_thickness),
                    "bar_areas" => _grouped_kind_element_map(groups, x_vec, :bar_area),
                    "mass_polish_step_scale" => backtrack * step_scale,
                    "mass_polish_backtrack" => backtrack,
                )
                if get(active_constraint, "response_family", nothing) == "displacement"
                    iter_record["response_grid"] = get(active_constraint, "constraint_grid", nothing)
                    iter_record["response_dof"] = get(active_constraint, "constraint_dof", nothing)
                end
                if capture_solver_diagnostics
                    iter_record["solver_diagnostics"] = _extract_optimization_solver_diagnostics(current_results, :min_compliance)
                end
                push!(iterations, iter_record)
                improved = true
                accepted = true

                if previous_mass - current_mass <= improvement_tol
                    termination_reason = "mass_reduction_converged"
                end
                break
            end

            termination_reason == "mass_reduction_converged" && break
            if !accepted
                termination_reason = improved ? "backtracking_stagnation" : "no_feasible_step"
                break
            end
        end
    elseif max_iter > 0
        termination_reason = "initial_infeasible"
    end

    if max_iter > 0 && (termination_reason == "no_feasible_step" || termination_reason == "no_iterations_requested") && improved
        termination_reason = "max_iter"
    end

    final_active_constraint = isempty(current_constraints) ? nothing :
        _grouped_static_active_constraint(current_constraints)
    summary = Dict{String,Any}(
        "attempted" => true,
        "applied" => improved,
        "n_iter" => length(iterations),
        "termination_reason" => termination_reason,
        "initial_mass" => initial_mass,
        "final_mass" => current_mass,
        "mass_reduction" => initial_mass - current_mass,
        "initial_variable_mass" => initial_variable_mass,
        "final_variable_mass" => current_variable_mass,
        "variable_mass_reduction" => initial_variable_mass - current_variable_mass,
        "initial_constraint_merit" => initial_merit,
        "final_constraint_merit" => current_merit,
        "initial_max_constraint_violation" => isempty(initial_constraints) ? nothing : _grouped_static_constraints_max_violation(initial_constraints),
        "final_max_constraint_violation" => isempty(current_constraints) ? nothing : _grouped_static_constraints_max_violation(current_constraints),
        "initial_active_constraint_index" => isnothing(initial_active_constraint) ? nothing : Int(initial_active_constraint["constraint_index"]),
        "final_active_constraint_index" => isnothing(final_active_constraint) ? nothing : Int(final_active_constraint["constraint_index"]),
        "initial_active_response_id" => isnothing(initial_active_constraint) ? nothing : get(initial_active_constraint, "response_id", nothing),
        "final_active_response_id" => isnothing(final_active_constraint) ? nothing : get(final_active_constraint, "response_id", nothing),
    )

    return x_vec, current_model, current_results, current_constraints, iterations, summary
end

function optimize_grouped_single_variable_mass_minimization(
    model::Dict, solve_fn::Function, group::SizingGroupData;
    response_constraints=nothing,
    response_family::Union{Nothing,Symbol}=nothing,
    response_spec::Union{Nothing,Dict{String,Any}}=nothing,
    response_upper_bound::Union{Nothing,Float64}=nothing,
    x_min::Float64,
    x_max::Float64,
    max_iter::Int = 50,
    tol::Float64 = 1e-6,
    capture_solver_diagnostics::Bool = true)

    x_min > 0.0 || error("[OPT] Single-variable mass minimization requires a positive lower bound")
    x_max <= 0.0 && (x_max = max(10.0 * group.x0, 10.0 * x_min))
    x_max > x_min || error("[OPT] Single-variable mass minimization requires x_max > x_min")
    max_iter >= 2 || error("[OPT] Single-variable mass minimization requires at least two iterations for bracketing")

    kinds = sort!(collect(Set(var.kind for var in group.members)))
    normalized_constraints = _normalize_grouped_static_response_constraints(
        response_constraints=response_constraints,
        response_family=response_family,
        response_spec=response_spec,
        response_upper_bound=response_upper_bound,
    )
    history = Float64[]
    iterations = Any[]
    best_results = nothing
    best_x = x_max
    converged = false
    termination_reason = "max_iter"
    effective_tol = _grouped_bisection_tolerance(Float64(tol))

    function evaluate_at!(x_value::Float64, iter_number::Int)
        _apply_grouped_sizing_values!(model, [group], [x_value])
        results = solve_fn(model)
        constraint_values = _evaluate_grouped_static_constraints(results, model, normalized_constraints)
        active_constraint = _grouped_static_active_constraint(constraint_values)
        variable_mass_value = _grouped_variable_mass(group, x_value)
        mass_value = variable_mass_value + group.fixed_mass
        iter_record = Dict{String,Any}(
            "iteration" => iter_number,
            "objective" => mass_value,
            "relative_change" => isempty(history) ? nothing :
                abs(mass_value - history[end]) / max(abs(history[end]), 1e-30),
            "response_value" => Float64(active_constraint["response_value"]),
            "response_family" => String(active_constraint["response_family"]),
            "response_upper_bound" => Float64(active_constraint["response_upper_bound"]),
            "response_constraint_violation" => Float64(active_constraint["response_constraint_violation"]),
            "response_constraints" => deepcopy(constraint_values),
            "constraint_count" => length(constraint_values),
            "active_constraint_index" => Int(active_constraint["constraint_index"]),
            "active_response_id" => get(active_constraint, "response_id", nothing),
            "max_constraint_violation" => _grouped_static_constraints_max_violation(constraint_values),
            "mass" => mass_value,
            "variable_mass" => variable_mass_value,
            "fixed_mass" => group.fixed_mass,
            "mass_ratio" => mass_value / max(_grouped_total_mass(group, group.x0), 1e-30),
            "min_design_value" => x_value,
            "max_design_value" => x_value,
            "gradient_norm" => nothing,
            "objective_type" => "min_mass",
            "design_variable_types" => [String(kind) for kind in kinds],
            "design_variables" => _grouped_design_value_map([group], [x_value]),
            "shell_thicknesses" => _grouped_kind_element_map([group], [x_value], :shell_thickness),
            "bar_areas" => _grouped_kind_element_map([group], [x_value], :bar_area),
        )
        if capture_solver_diagnostics
            iter_record["solver_diagnostics"] = _extract_optimization_solver_diagnostics(results, :min_compliance)
        end
        push!(history, mass_value)
        push!(iterations, iter_record)
        return results, constraint_values, mass_value
    end

    constraint_desc = if length(normalized_constraints) == 1
        only_constraint = normalized_constraints[1]
        "$(only_constraint["family"]) <= $(only_constraint["upper_bound"])"
    else
        "$(length(normalized_constraints)) upper-bound static constraints"
    end
    log_msg("[OPT] Single grouped sizing variable ($(join(string.(kinds), ", "))), objective=min_mass, response=$constraint_desc")
    log_msg("[OPT] Direct bracketing on x in [$x_min, $x_max]")

    lower_results, lower_constraints, lower_mass = evaluate_at!(x_min, 1)
    lower_active = _grouped_static_active_constraint(lower_constraints)
    log_msg("[OPT] Iter 1: x=$(round(x_min, sigdigits=6)), active_$(lower_active["response_family"])=$(round(Float64(lower_active["response_value"]), sigdigits=6)), mass=$(round(lower_mass, sigdigits=6))")
    if _grouped_static_constraints_feasible(lower_constraints)
        best_results = lower_results
        best_x = x_min
        converged = true
        termination_reason = "lower_bound_feasible"
        return _grouped_single_variable_result_payload(
            history, iterations, group, group.x0, best_x, normalized_constraints,
            converged, termination_reason, model, best_results
        )
    end

    upper_results, upper_constraints, upper_mass = evaluate_at!(x_max, 2)
    upper_active = _grouped_static_active_constraint(upper_constraints)
    log_msg("[OPT] Iter 2: x=$(round(x_max, sigdigits=6)), active_$(upper_active["response_family"])=$(round(Float64(upper_active["response_value"]), sigdigits=6)), mass=$(round(upper_mass, sigdigits=6))")
    _grouped_static_constraints_monotone(lower_constraints, upper_constraints) ||
        error("[OPT] Single-variable mass minimization requires every routed static constraint to improve monotonically with increasing design value")
    _grouped_static_constraints_feasible(upper_constraints) ||
        error("[OPT] Single-variable mass minimization found no feasible point within the translated design bounds")

    lo = x_min
    hi = x_max
    best_results = upper_results
    best_x = x_max

    for iter_idx in 3:max_iter
        mid = 0.5 * (lo + hi)
        mid_results, mid_constraints, mid_mass = evaluate_at!(mid, iter_idx)
        mid_active = _grouped_static_active_constraint(mid_constraints)
        log_msg("[OPT] Iter $iter_idx: x=$(round(mid, sigdigits=6)), active_$(mid_active["response_family"])=$(round(Float64(mid_active["response_value"]), sigdigits=6)), mass=$(round(mid_mass, sigdigits=6))")

        if _grouped_static_constraints_feasible(mid_constraints)
            hi = mid
            best_x = mid
            best_results = mid_results
        else
            lo = mid
        end

        interval_gap = abs(hi - lo) / max(abs(hi), 1.0)
        response_gap = if length(normalized_constraints) == 1
            abs(Float64(mid_active["response_value"]) - Float64(mid_active["response_upper_bound"])) /
                max(abs(Float64(mid_active["response_upper_bound"])), 1e-30)
        else
            Inf
        end
        if response_gap <= effective_tol || interval_gap <= effective_tol
            converged = true
            termination_reason = "constraint_bisection"
            break
        end
    end

    !converged && (termination_reason = "max_iter")
    _apply_grouped_sizing_values!(model, [group], [best_x])
    return _grouped_single_variable_result_payload(
        history, iterations, group, group.x0, best_x, normalized_constraints,
        converged, termination_reason, model, best_results
    )
end

function optimize_grouped_static_response_mass_minimization(
    base_model::Dict, solve_fn::Function, groups::Vector{SizingGroupData};
    response_constraints=nothing,
    response_family::Union{Nothing,Symbol}=nothing,
    response_spec::Union{Nothing,Dict{String,Any}}=nothing,
    response_upper_bound::Union{Nothing,Float64}=nothing,
    x_min = 1e-4,
    x_max = 0.0,
    max_iter::Int = 30,
    tol::Float64 = 1e-6,
    move_limit = 0.2,
    eta::Float64 = 0.5,
    thickness_derivative_method::Union{Nothing,Symbol,String} = nothing,
    bar_area_derivative_method::Union{Nothing,Symbol,String} = nothing,
    capture_solver_diagnostics::Bool = true)

    length(groups) >= 2 || error("[OPT] Multi-group static-response mass minimization requires at least two grouped sizing variables")
    max_iter >= 2 || error("[OPT] Multi-group static-response mass minimization requires at least two bisection iterations")

    normalized_constraints = _normalize_grouped_static_response_constraints(
        response_constraints=response_constraints,
        response_family=response_family,
        response_spec=response_spec,
        response_upper_bound=response_upper_bound,
    )
    initial_x_vec = _current_grouped_sizing_vector(base_model, groups)
    mass_coeffs = [group.mass_coeff for group in groups]
    fixed_mass = _grouped_fixed_mass(groups)
    variable_mass_initial = _grouped_variable_mass(groups, initial_x_vec)
    mass_initial = variable_mass_initial + fixed_mass

    n_group = length(groups)
    x_min_vec = _design_limit_vector(x_min, n_group, "group lower bounds")
    x_max_vec = _design_limit_vector(x_max, n_group, "group upper bounds")
    if any(v <= 0.0 for v in x_max_vec)
        fallback = 10.0 * maximum(initial_x_vec)
        x_max_vec = [v <= 0.0 ? fallback : v for v in x_max_vec]
    end

    lower_mass_target = sum(x_min_vec .* mass_coeffs) + fixed_mass
    upper_mass_target = sum(x_max_vec .* mass_coeffs) + fixed_mass
    history = Float64[]
    iterations = Any[]
    best_result = nothing
    best_results = nothing
    best_mass_target = upper_mass_target
    converged = false
    termination_reason = "max_iter"
    mass_polish = Dict{String,Any}(
        "attempted" => false,
        "applied" => false,
        "n_iter" => 0,
        "termination_reason" => "not_applicable",
        "fallback_due_to_nonmonotone_route" => false,
    )
    effective_tol = _grouped_bisection_tolerance(Float64(tol))

    inner_max_iter = max(12, min(max_iter, 25))
    inner_tol = max(tol, 1e-4)
    kinds = sort!(collect(Set(var.kind for group in groups for var in group.members)))

    function evaluate_mass_target!(mass_target::Float64, iter_number::Int)
        working_model = deepcopy(base_model)
        inner_result = if length(normalized_constraints) == 1
            optimize_grouped_static_response(
                working_model, solve_fn, groups;
                response_family=normalized_constraints[1]["family"],
                response_spec=normalized_constraints[1]["spec"],
                x_min=x_min_vec,
                x_max=x_max_vec,
                mass_target=mass_target,
                max_iter=inner_max_iter,
                tol=inner_tol,
                move_limit=move_limit,
                eta=eta,
                thickness_derivative_method=thickness_derivative_method,
                bar_area_derivative_method=bar_area_derivative_method,
                capture_solver_diagnostics=capture_solver_diagnostics,
            )
        else
            optimize_grouped_static_constraints(
                working_model, solve_fn, groups;
                response_constraints=normalized_constraints,
                x_min=x_min_vec,
                x_max=x_max_vec,
                mass_target=mass_target,
                max_iter=inner_max_iter,
                tol=inner_tol,
                move_limit=move_limit,
                eta=eta,
                thickness_derivative_method=thickness_derivative_method,
                bar_area_derivative_method=bar_area_derivative_method,
                capture_solver_diagnostics=capture_solver_diagnostics,
            )
        end

        final_results = get(inner_result, "_final_results", nothing)
        if !(final_results isa AbstractDict)
            final_results = solve_fn(inner_result["model"])
        end
        evaluated_constraints = _evaluate_grouped_static_constraints(final_results, inner_result["model"], normalized_constraints)
        active_constraint = _grouped_static_active_constraint(evaluated_constraints)
        mass_value = Float64(inner_result["final_mass"])
        variable_mass_value = Float64(get(inner_result, "variable_mass_final", mass_value - fixed_mass))
        rel_change = isempty(history) ? nothing :
            abs(mass_value - history[end]) / max(abs(history[end]), 1e-30)

        iter_record = Dict{String,Any}(
            "iteration" => iter_number,
            "objective" => mass_value,
            "relative_change" => rel_change,
            "response_value" => Float64(active_constraint["response_value"]),
            "response_family" => String(active_constraint["response_family"]),
            "response_upper_bound" => Float64(active_constraint["response_upper_bound"]),
            "response_constraint_violation" => Float64(active_constraint["response_constraint_violation"]),
            "response_constraints" => deepcopy(evaluated_constraints),
            "constraint_count" => length(evaluated_constraints),
            "active_constraint_index" => Int(active_constraint["constraint_index"]),
            "active_response_id" => get(active_constraint, "response_id", nothing),
            "max_constraint_violation" => _grouped_static_constraints_max_violation(evaluated_constraints),
            "mass" => mass_value,
            "variable_mass" => variable_mass_value,
            "fixed_mass" => fixed_mass,
            "mass_target_trial" => mass_target,
            "mass_ratio" => mass_value / max(mass_initial, 1e-30),
            "min_design_value" => minimum(values(inner_result["design_variables"])),
            "max_design_value" => maximum(values(inner_result["design_variables"])),
            "gradient_norm" => nothing,
            "objective_type" => "min_mass",
            "design_variable_types" => [String(kind) for kind in kinds],
            "design_variables" => deepcopy(inner_result["design_variables"]),
            "shell_thicknesses" => deepcopy(inner_result["shell_thicknesses"]),
            "bar_areas" => deepcopy(inner_result["bar_areas"]),
            "response_optimization_iterations" => Int(inner_result["n_iter"]),
            "response_optimization_termination_reason" => inner_result["termination_reason"],
        )
        if get(active_constraint, "response_family", nothing) == "displacement"
            iter_record["response_grid"] = get(active_constraint, "constraint_grid", nothing)
            iter_record["response_dof"] = get(active_constraint, "constraint_dof", nothing)
        end
        if capture_solver_diagnostics
            iter_record["solver_diagnostics"] = _extract_optimization_solver_diagnostics(final_results, :min_compliance)
        end

        push!(history, mass_value)
        push!(iterations, iter_record)
        return inner_result, final_results, evaluated_constraints, active_constraint
    end

    response_desc = if length(normalized_constraints) == 1
        routed_constraint = normalized_constraints[1]
        routed_constraint["family"] == :displacement ?
            "displacement(grid=$(routed_constraint["spec"]["grid"]), dof=$(routed_constraint["spec"]["dof"])) <= $(routed_constraint["upper_bound"])" :
            "$(routed_constraint["family"]) <= $(routed_constraint["upper_bound"])"
    else
        "$(length(normalized_constraints)) upper-bound static constraints"
    end
    log_msg("[OPT] $(length(groups)) grouped sizing variables ($(join(string.(kinds), ", "))), objective=min_mass, routed response=$response_desc")
    log_msg("[OPT] Outer bisection on mass target in [$(round(lower_mass_target, sigdigits=6)), $(round(upper_mass_target, sigdigits=6))]")

    lower_result, lower_results, lower_constraints, lower_active = evaluate_mass_target!(lower_mass_target, 1)
    log_msg("[OPT] Iter 1: mass_target=$(round(lower_mass_target, sigdigits=6)), active_$(lower_active["response_family"])=$(round(Float64(lower_active["response_value"]), sigdigits=6)), mass=$(round(Float64(lower_result["final_mass"]), sigdigits=6))")
    if _grouped_static_constraints_feasible(lower_constraints)
        best_result = lower_result
        best_results = lower_results
        best_mass_target = lower_mass_target
        converged = true
        termination_reason = "lower_bound_feasible"
        mass_polish = Dict{String,Any}(
            "attempted" => false,
            "applied" => false,
            "n_iter" => 0,
            "termination_reason" => "lower_bound_feasible",
            "initial_mass" => Float64(lower_result["final_mass"]),
            "final_mass" => Float64(lower_result["final_mass"]),
            "mass_reduction" => 0.0,
            "initial_variable_mass" => Float64(get(lower_result, "variable_mass_final", lower_result["final_mass"] - fixed_mass)),
            "final_variable_mass" => Float64(get(lower_result, "variable_mass_final", lower_result["final_mass"] - fixed_mass)),
        )
        return _grouped_static_mass_result_payload(
            history, iterations, groups, initial_x_vec,
            Float64[Float64(best_result["design_variables"][group.label]) for group in groups],
            normalized_constraints, converged, termination_reason, best_result["model"], best_results;
            mass_polish=mass_polish
        )
    end

    upper_result, upper_results, upper_constraints, upper_active = evaluate_mass_target!(upper_mass_target, 2)
    log_msg("[OPT] Iter 2: mass_target=$(round(upper_mass_target, sigdigits=6)), active_$(upper_active["response_family"])=$(round(Float64(upper_active["response_value"]), sigdigits=6)), mass=$(round(Float64(upper_result["final_mass"]), sigdigits=6))")
    _grouped_static_constraints_feasible(upper_constraints) ||
        error("[OPT] Multi-group mass minimization found no feasible point within the translated design bounds")

    monotone = _grouped_static_constraints_monotone(lower_constraints, upper_constraints)
    if !monotone
        log_msg("[OPT] Routed static constraints were not monotone over the initial bracket; switching to direct feasible mass-polish fallback")
        final_x_vec = Float64[Float64(upper_result["design_variables"][group.label]) for group in groups]
        final_x_vec, final_model, final_results, _, polish_iterations, mass_polish =
            _grouped_static_mass_polish(
                base_model, solve_fn, groups, final_x_vec, normalized_constraints;
                x_min=x_min_vec,
                x_max=x_max_vec,
                move_limit=move_limit,
                max_iter=max(12, min(max_iter * 2, 30)),
                tol=max(tol * 0.25, 1e-5),
                iteration_offset=length(iterations),
                thickness_derivative_method=thickness_derivative_method,
                bar_area_derivative_method=bar_area_derivative_method,
                capture_solver_diagnostics=capture_solver_diagnostics,
            )
        for iter_record in polish_iterations
            push!(history, Float64(iter_record["objective"]))
            push!(iterations, iter_record)
        end
        mass_polish["fallback_due_to_nonmonotone_route"] = true
        converged = true
        termination_reason = "nonmonotone_mass_polish_fallback"
        _apply_grouped_sizing_values!(final_model, groups, final_x_vec)
        return _grouped_static_mass_result_payload(
            history, iterations, groups, initial_x_vec, final_x_vec,
            normalized_constraints, converged, termination_reason, final_model, final_results;
            mass_polish=mass_polish
        )
    end

    lo = lower_mass_target
    hi = upper_mass_target
    best_result = upper_result
    best_results = upper_results
    best_mass_target = upper_mass_target

    for iter_idx in 3:max_iter
        mid = 0.5 * (lo + hi)
        mid_result, mid_results, mid_constraints, mid_active = evaluate_mass_target!(mid, iter_idx)
        log_msg("[OPT] Iter $iter_idx: mass_target=$(round(mid, sigdigits=6)), active_$(mid_active["response_family"])=$(round(Float64(mid_active["response_value"]), sigdigits=6)), mass=$(round(Float64(mid_result["final_mass"]), sigdigits=6))")

        if _grouped_static_constraints_feasible(mid_constraints)
            hi = mid
            best_result = mid_result
            best_results = mid_results
            best_mass_target = mid
        else
            lo = mid
        end

        response_gap = length(normalized_constraints) == 1 ?
            abs(Float64(mid_active["response_value"]) - Float64(mid_active["response_upper_bound"])) /
                max(abs(Float64(mid_active["response_upper_bound"])), 1e-30) :
            Inf
        interval_gap = abs(hi - lo) / max(abs(hi), 1.0)
        if response_gap <= effective_tol || interval_gap <= effective_tol
            converged = true
            termination_reason = "mass_target_bisection"
            break
        end
    end

    !converged && (termination_reason = "max_iter")
    final_x_vec = Float64[Float64(best_result["design_variables"][group.label]) for group in groups]
    polish_iter_offset = length(iterations)
    final_x_vec, final_model, final_results, _, polish_iterations, mass_polish =
        _grouped_static_mass_polish(
            base_model, solve_fn, groups, final_x_vec, normalized_constraints;
            x_min=x_min_vec,
            x_max=x_max_vec,
            move_limit=move_limit,
            max_iter=min(8, max_iter),
            tol=max(tol * 0.25, 1e-5),
            iteration_offset=polish_iter_offset,
            thickness_derivative_method=thickness_derivative_method,
            bar_area_derivative_method=bar_area_derivative_method,
            capture_solver_diagnostics=capture_solver_diagnostics,
        )
    for iter_record in polish_iterations
        push!(history, Float64(iter_record["objective"]))
        push!(iterations, iter_record)
    end
    _apply_grouped_sizing_values!(final_model, groups, final_x_vec)
    return _grouped_static_mass_result_payload(
        history, iterations, groups, initial_x_vec, final_x_vec,
        normalized_constraints, converged, termination_reason, final_model, final_results;
        mass_polish=mass_polish
    )
end

function _grouped_single_variable_result_payload(
    history, iterations, group::SizingGroupData, initial_x::Float64, final_x::Float64,
    response_constraints::Vector{Dict{String,Any}},
    converged::Bool, termination_reason::String, model, final_results)
    variable_mass_initial = _grouped_variable_mass(group, initial_x)
    variable_mass_final = _grouped_variable_mass(group, final_x)
    initial_mass = variable_mass_initial + group.fixed_mass
    final_mass = variable_mass_final + group.fixed_mass
    evaluated_constraints = final_results === nothing ? Dict{String,Any}[] :
        _evaluate_grouped_static_constraints(final_results, model, response_constraints)
    active_constraint = isempty(evaluated_constraints) ? nothing :
        _grouped_static_active_constraint(evaluated_constraints)
    return _finalize_optimization_result!(Dict{String,Any}(
        "objective" => "min_mass",
        "history" => copy(history),
        "iterations" => deepcopy(iterations),
        "design_variable_types" => [String(kind) for kind in sort!(collect(Set(var.kind for var in group.members)))],
        "design_variables" => _grouped_design_value_map([group], [final_x]),
        "initial_design_variables" => _grouped_design_value_map([group], [initial_x]),
        "shell_thicknesses" => _grouped_kind_element_map([group], [final_x], :shell_thickness),
        "initial_shell_thicknesses" => _grouped_kind_element_map([group], [initial_x], :shell_thickness),
        "bar_areas" => _grouped_kind_element_map([group], [final_x], :bar_area),
        "initial_bar_areas" => _grouped_kind_element_map([group], [initial_x], :bar_area),
        "group_metadata" => _grouped_metadata([group]),
        "response_family" => isnothing(active_constraint) ? nothing : String(active_constraint["response_family"]),
        "response_upper_bound" => isnothing(active_constraint) ? nothing : Float64(active_constraint["response_upper_bound"]),
        "response_value" => isnothing(active_constraint) ? nothing : Float64(active_constraint["response_value"]),
        "response_constraint_violation" => isnothing(active_constraint) ? nothing : Float64(active_constraint["response_constraint_violation"]),
        "response_constraints" => deepcopy(evaluated_constraints),
        "constraint_count" => length(response_constraints),
        "active_constraint_index" => isnothing(active_constraint) ? nothing : Int(active_constraint["constraint_index"]),
        "active_response_id" => isnothing(active_constraint) ? nothing : get(active_constraint, "response_id", nothing),
        "converged" => converged,
        "n_iter" => length(iterations),
        "design_variable_count" => 1,
        "fixed_mass" => group.fixed_mass,
        "variable_mass_initial" => variable_mass_initial,
        "variable_mass_target" => nothing,
        "variable_mass_final" => variable_mass_final,
        "mass_initial" => initial_mass,
        "mass_target" => nothing,
        "final_mass" => final_mass,
        "mass_ratio" => final_mass / max(initial_mass, 1e-30),
        "termination_reason" => termination_reason,
        "checkpoint_path" => nothing,
        "thickness_derivative_method" => nothing,
        "bar_area_derivative_method" => nothing,
        "restart" => nothing,
        "model" => model,
    ))
end

function _grouped_static_mass_result_payload(
    history, iterations, groups::Vector{SizingGroupData}, initial_x_vec, final_x_vec,
    response_constraints::Vector{Dict{String,Any}},
    converged::Bool, termination_reason::String, model, final_results;
    mass_polish=nothing)
    fixed_mass = _grouped_fixed_mass(groups)
    variable_mass_initial = _grouped_variable_mass(groups, initial_x_vec)
    variable_mass_final = _grouped_variable_mass(groups, final_x_vec)
    initial_mass = variable_mass_initial + fixed_mass
    final_mass = variable_mass_final + fixed_mass
    evaluated_constraints = final_results === nothing ? Dict{String,Any}[] :
        _evaluate_grouped_static_constraints(final_results, model, response_constraints)
    active_constraint = isempty(evaluated_constraints) ? nothing :
        _grouped_static_active_constraint(evaluated_constraints)
    result = Dict{String,Any}(
        "objective" => "min_mass",
        "history" => copy(history),
        "iterations" => deepcopy(iterations),
        "design_variable_types" => [String(kind) for kind in sort!(collect(Set(var.kind for group in groups for var in group.members)))],
        "design_variables" => _grouped_design_value_map(groups, final_x_vec),
        "initial_design_variables" => _grouped_design_value_map(groups, initial_x_vec),
        "shell_thicknesses" => _grouped_kind_element_map(groups, final_x_vec, :shell_thickness),
        "initial_shell_thicknesses" => _grouped_kind_element_map(groups, initial_x_vec, :shell_thickness),
        "bar_areas" => _grouped_kind_element_map(groups, final_x_vec, :bar_area),
        "initial_bar_areas" => _grouped_kind_element_map(groups, initial_x_vec, :bar_area),
        "group_metadata" => _grouped_metadata(groups),
        "response_family" => isnothing(active_constraint) ? nothing : String(active_constraint["response_family"]),
        "response_upper_bound" => isnothing(active_constraint) ? nothing : Float64(active_constraint["response_upper_bound"]),
        "response_value" => isnothing(active_constraint) ? nothing : Float64(active_constraint["response_value"]),
        "response_constraint_violation" => isnothing(active_constraint) ? nothing : Float64(active_constraint["response_constraint_violation"]),
        "response_constraints" => deepcopy(evaluated_constraints),
        "constraint_count" => length(response_constraints),
        "active_constraint_index" => isnothing(active_constraint) ? nothing : Int(active_constraint["constraint_index"]),
        "active_response_id" => isnothing(active_constraint) ? nothing : get(active_constraint, "response_id", nothing),
        "converged" => converged,
        "n_iter" => length(iterations),
        "design_variable_count" => length(groups),
        "fixed_mass" => fixed_mass,
        "variable_mass_initial" => variable_mass_initial,
        "variable_mass_target" => nothing,
        "variable_mass_final" => variable_mass_final,
        "mass_initial" => initial_mass,
        "mass_target" => nothing,
        "final_mass" => final_mass,
        "mass_ratio" => final_mass / max(initial_mass, 1e-30),
        "termination_reason" => termination_reason,
        "checkpoint_path" => nothing,
        "thickness_derivative_method" => nothing,
        "bar_area_derivative_method" => nothing,
        "restart" => nothing,
        "model" => model,
    )
    !isnothing(mass_polish) && (result["mass_polish"] = deepcopy(mass_polish))
    return _finalize_optimization_result!(result)
end
