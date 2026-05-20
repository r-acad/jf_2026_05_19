# JFEMSolver.jl — Clean solver API for JFEM
#
# Architecture:
#   Stage 1: bdf_to_model(filename) -> model::Dict        (BDF parsing + model building)
#   Stage 2: solve_model(model)     -> results::Dict       (pure computation, no file I/O)
#   Stage 3: caller writes output files from results Dict  (see Export.jl helpers)
#
# The model Dict IS the JSON data structure. It can be serialized with JSON.json()
# and deserialized with JSON.parse() for cross-language interop (Python, etc.).
#
# Usage from Julia:
#   model = bdf_to_model("path/to/file.bdf")
#   results = solve_model(model)
#   export_results(results, "output_name", "output/dir")
#
# Usage from Python (via JSON):
#   model_json = julia.bdf_to_model_json("path/to/file.bdf")
#   results_json = julia.solve_model_json(model_json)

using LinearAlgebra
using SparseArrays
using Printf
using Statistics
using JSON

# ============================================================================
# Stage 1: BDF → Model Dict
# ============================================================================

"""
    bdf_to_model(filename::String; export_json::Bool=false) -> Dict

Parse a Nastran BDF file and return a self-contained model Dict.
This Dict can be serialized to JSON for cross-language interop.

If `export_json=true`, writes the model to `<filename>.json`
(e.g. `test.bdf` → `test.bdf.json`).
"""
function bdf_to_model(filename::String; export_json::Bool=false)
    if !isfile(filename)
        error("File not found: $filename")
    end

    println(">>> Reading BDF file: $filename")
    lines = readlines(filename)
    lines = NastranParser.resolve_includes(lines, dirname(abspath(filename)))

    println(">>> Checking format...")
    lines = NastranParser.convert_mystran_to_nastran(lines)

    println(">>> Parsing Bulk Data...")
    cc, bulk = NastranParser.read_bulk_and_case(lines)
    cards = NastranParser.process_cards(bulk)

    # Report card inventory
    _report_card_inventory(cards)

    println(">>> Constructing Model Data...")
    model = build_model(cards, cc)
    resolve_nested_coords!(model)
    transform_geometry!(model)

    if export_json
        json_path = filename * ".json"
        println(">>> Exporting model JSON: $json_path")
        open(json_path, "w") do f
            JSON.print(f, model, 2)
        end
        println(">>> Model JSON exported: $json_path")
    end

    return model
end

"""
    bdf_to_model_json(filename::String) -> String

Parse a BDF file and return the model as a JSON string.
Convenience wrapper for cross-language interop.
"""
function bdf_to_model_json(filename::String)
    model = bdf_to_model(filename)
    return JSON.json(model)
end

"""
    json_to_model(json_path::String; export_json::Bool=false) -> Dict

Read a raw JSON model file (as produced by bdf_to_json.jl) and build a solver-ready model.
The JSON must contain only raw BDF card data — all derived quantities (CLT matrices,
section properties, material completion) are computed by this function.
"""
function json_to_model(json_path::String; export_json::Bool=false)
    if !isfile(json_path)
        error("JSON file not found: $json_path")
    end
    println(">>> Reading raw JSON model: $json_path")
    raw = JSON.parsefile(json_path)
    model = build_model_from_json(raw)

    if export_json
        out_path = json_path * ".model.json"
        println(">>> Exporting derived model JSON: $out_path")
        open(out_path, "w") do f
            JSON.print(f, model, 2)
        end
    end

    return model
end

function _report_card_inventory(cards)
    processed = Set(["GRID", "GRDSET", "CORD2R", "CORD1R", "CORD2C", "CORD2S",
        "CTRIA3", "CTRIA6", "CQUAD4", "CQUAD8", "CSHEAR", "CBAR", "CBEAM", "CROD", "CONROD", "CELAS1", "CELAS2", "CBUSH",
        "RBE1", "RBE2", "RBE3", "RBAR", "RSPLINE",
        "PSHELL", "PSHEAR", "PBARL", "PBAR", "PBAR*", "PBEAM", "PBEAM*", "PBEAML", "PROD", "PCOMP", "PELAS", "PBUSH", "PSOLID",
        "MAT1", "MAT2", "MAT8", "MATT1", "TABLEM1",
        "DESVAR", "DRESP1", "DVPREL1", "DVMREL1", "DCONSTR", "DOPTPRM",
        "FORCE", "MOMENT", "PLOAD4", "PLOAD2", "PLOAD", "PLOAD1", "GRAV", "RFORCE",
        "SPC1", "SPC", "SPCADD", "MPC", "MPCADD", "LOAD",
        "CONM2", "CONM1", "CMASS1", "CMASS2", "PMASS",
        "CTETRA", "CHEXA", "CPENTA",
        "EIGRL", "TEMP", "TEMPD", "DMIG", "PARAM"])
    unprocessed = Dict{String,Int}()
    for (cname, clist) in cards
        if !(cname in processed)
            unprocessed[cname] = length(clist)
        end
    end
    if !isempty(unprocessed)
        println("    WARNING: $(length(unprocessed)) unprocessed card type(s):")
        for (cname, cnt) in sort(collect(unprocessed), by=x->x[1])
            println("      $cname: $cnt")
        end
    end
end

@inline function _solver_entry_public_id(key, entry)
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

@inline function _canonical_sol_type(sol_type)
    return sol_type in (63, 103) ? 103 : sol_type
end

@inline function _sol200_lite_lookup_by_id(entries)
    lookup = Dict{Int, Dict{String,Any}}()
    for entry in entries
        lookup[Int(entry["id"])] = entry
    end
    return lookup
end

@inline function _model_has_temperature_dependent_mat1(model)
    return !isempty(get(model, "MATT1s", Dict())) && !isempty(get(model, "TABLEM1s", Dict()))
end

function _with_active_temperature_material(model::Dict, temp_sid, f::Function)
    had_prior = haskey(model, "_active_temp_sid")
    prior = had_prior ? model["_active_temp_sid"] : nothing
    if isnothing(temp_sid)
        had_prior && delete!(model, "_active_temp_sid")
    else
        model["_active_temp_sid"] = Int(temp_sid)
    end
    try
        return f()
    finally
        if had_prior
            model["_active_temp_sid"] = prior
        elseif haskey(model, "_active_temp_sid")
            delete!(model, "_active_temp_sid")
        end
    end
end

@inline function _sol200_lite_numeric_equal(a, b)
    af = Float64(a)
    bf = Float64(b)
    return abs(af - bf) <= 1e-10 * max(1.0, abs(af), abs(bf))
end

function _sol200_lite_uniform_value(values, label; default=nothing)
    filtered = Any[v for v in values if !isnothing(v)]
    isempty(filtered) && return default
    ref = filtered[1]
    for v in filtered[2:end]
        if ref isa Number && v isa Number
            _sol200_lite_numeric_equal(ref, v) || error("SOL 200-lite route requires uniform $label across the supported design variables")
        else
            v == ref || error("SOL 200-lite route requires uniform $label across the supported design variables")
        end
    end
    return ref
end

function _sol200_lite_property_usage(model, kind::Symbol)
    usage = Dict{Int, Int}()
    if kind == :shell_thickness
        for (_, el) in get(model, "CSHELLs", Dict())
            pid = Int(get(el, "PID", 0))
            pid > 0 || continue
            usage[pid] = get(usage, pid, 0) + 1
        end
    elseif kind == :bar_area
        for element_set in ("CBARs", "CBEAMs")
            for (_, el) in get(model, element_set, Dict())
                pid = Int(get(el, "PID", 0))
                pid > 0 || continue
                usage[pid] = get(usage, pid, 0) + 1
            end
        end
    end
    return usage
end

function _sol200_lite_property_membership(model, kind::Symbol)
    membership = Dict{Int, Vector{Int}}()
    if kind == :shell_thickness
        for (key, el) in get(model, "CSHELLs", Dict())
            pid = Int(get(el, "PID", 0))
            pid > 0 || continue
            eid = _solver_entry_public_id(key, el)
            eid > 0 || continue
            push!(get!(membership, pid, Int[]), eid)
        end
    elseif kind == :bar_area
        for element_set in ("CBARs", "CBEAMs")
            for (key, el) in get(model, element_set, Dict())
                pid = Int(get(el, "PID", 0))
                pid > 0 || continue
                eid = _solver_entry_public_id(key, el)
                eid > 0 || continue
                push!(get!(membership, pid, Int[]), eid)
            end
        end
    end

    for eids in values(membership)
        sort!(eids)
        unique!(eids)
    end
    return membership
end

function _sol200_lite_apply_group_initial_value!(model, kind::Symbol, pid::Int, value)
    if kind == :shell_thickness
        haskey(get(model, "PSHELLs", Dict()), string(pid)) ||
            error("SOL 200-lite route could not find PSHELL $pid while applying DESVAR initialization")
        model["PSHELLs"][string(pid)]["T"] = Float64(value)
    elseif kind == :bar_area
        haskey(get(model, "PBARLs", Dict()), string(pid)) ||
            error("SOL 200-lite route could not find bar property $pid while applying DESVAR initialization")
        model["PBARLs"][string(pid)]["A"] = Float64(value)
    else
        error("SOL 200-lite route does not support initial-value application for sizing family $kind")
    end
    return nothing
end

function _sol200_lite_validate_mass_scope(model, kinds)
    if !isempty(get(model, "CRODs", Dict())) || !isempty(get(model, "CONRODs", Dict())) ||
       !isempty(get(model, "CSOLIDs", Dict())) || !isempty(get(model, "CONM1s", Dict())) ||
       !isempty(get(model, "CONM2s", Dict())) || !isempty(get(model, "CMASS1s", Dict())) ||
       !isempty(get(model, "CMASS2s", Dict())) || !isempty(get(model, "PMASSs", Dict()))
        error("SOL 200-lite route currently supports shell/bar sizing decks without rods, solids, or concentrated mass contributions")
    end

    if !(:shell_thickness in kinds) && !isempty(get(model, "CSHELLs", Dict()))
        error("SOL 200-lite route requires shell elements to be included in the selected sizing families")
    end
    if !(:bar_area in kinds) && (!isempty(get(model, "CBARs", Dict())) || !isempty(get(model, "CBEAMs", Dict())))
        error("SOL 200-lite route requires CBAR/CBEAM elements to be included in the selected sizing families")
    end

    return nothing
end

function _sol200_lite_select_subcases!(model, objective::Symbol, opt::Dict{String,Any})
    cc = model["CASE_CONTROL"]
    subcases = get(cc, "SUBCASES", Dict{Int, Dict{String,Any}}())
    dessub = get(opt, "global_design_subcase_selector", nothing)

    if objective in (:min_compliance, :min_mass_static_response)
        selected_sid = if !isnothing(dessub)
            sid = Int(dessub)
            haskey(subcases, sid) || error("SOL 200-lite route could not find DESSUB subcase $sid")
            sid
        elseif length(subcases) == 1
            first(sort(collect(keys(subcases))))
        else
            error("SOL 200-lite static routing requires exactly one subcase or a global DESSUB selector")
        end

        cc["SUBCASES"] = Dict(selected_sid => deepcopy(subcases[selected_sid]))
        return selected_sid
    end

    if !isnothing(dessub)
        selected_sid = Int(dessub)
        haskey(subcases, selected_sid) || error("SOL 200-lite route could not find DESSUB subcase $selected_sid")
        sub = subcases[selected_sid]
        if haskey(sub, "STATSUB") && !isnothing(sub["STATSUB"])
            static_sid = Int(sub["STATSUB"])
            haskey(subcases, static_sid) || error("SOL 200-lite route could not find STATSUB=$static_sid for buckling design subcase $selected_sid")
            cc["SUBCASES"] = Dict(
                selected_sid => deepcopy(sub),
                static_sid => deepcopy(subcases[static_sid]),
            )
        end
        return selected_sid
    end

    return nothing
end

function _sol200_lite_static_response_spec(response::Dict{String,Any})
    family = get(response, "candidate_response_family", nothing)
    if family == "compliance"
        return :compliance, Dict{String,Any}("type" => "compliance")
    elseif family == "displacement"
        grid = get(response, "atta", nothing)
        dof = get(response, "attb", nothing)
        isnothing(grid) && error("SOL 200-lite displacement routing requires DRESP1 ATTA to identify the grid")
        isnothing(dof) && error("SOL 200-lite displacement routing requires DRESP1 ATTB to identify the component")
        return :displacement, Dict{String,Any}(
            "type" => "displacement",
            "grid" => Int(grid),
            "dof" => Int(dof),
        )
    end
    response_id = get(response, "id", nothing)
    response_label = get(response, "label", nothing)
    error("SOL 200-lite static response route only supports compliance or displacement constraints (DRESP1 $(response_id), label=$(response_label), family=$(family))")
end

function _sol200_lite_translate(model::Dict)
    opt = get(model, "OPTIMIZATION", nothing)
    opt isa Dict{String,Any} || error("SOL 200 deck is missing normalized OPTIMIZATION metadata")

    responses = _sol200_lite_lookup_by_id(get(opt, "responses", Any[]))
    design_vars = _sol200_lite_lookup_by_id(get(opt, "design_variables", Any[]))
    objective_def = get(opt, "objective", nothing)
    objective_def isa AbstractDict || error("SOL 200-lite route requires a DESOBJ objective")

    objective_response_id = Int(objective_def["response_id"])
    haskey(responses, objective_response_id) || error("SOL 200-lite objective response $objective_response_id was not found")
    objective_response = responses[objective_response_id]
    objective_family = get(objective_response, "candidate_response_family", nothing)
    objective_sense = uppercase(strip(string(get(objective_def, "sense", ""))))

    translated_objective, forward_sol_type =
        if objective_family == "compliance" && objective_sense == "MIN"
            (:min_compliance, 101)
        elseif objective_family == "mass" && objective_sense == "MIN"
            (:min_mass_static_response, 101)
        elseif objective_family == "buckling_eigenvalue" && objective_sense == "MAX"
            (:max_buckling, 105)
        else
            error("SOL 200-lite route currently supports DESOBJ(MIN)=compliance, DESOBJ(MIN)=weight/mass on a narrow static subset, or DESOBJ(MAX)=buckling only (DRESP1 $(objective_response_id), family=$(objective_family), sense=$(objective_sense))")
        end

    material_relations = get(opt, "material_relations", Any[])
    if !isempty(material_relations)
        relation_ids = sort!(Int[Int(rel["id"]) for rel in material_relations])
        error("SOL 200-lite route does not yet support DVMREL-driven material optimization (DVMREL1 IDs: $(join(relation_ids, ", ")))")
    end

    property_relations = get(opt, "property_relations", Any[])
    isempty(property_relations) && error("SOL 200-lite route requires supported DVPREL1 property relations")

    relation_ids = Int[]
    kinds_raw = Symbol[]
    property_owner = Dict{Symbol, Dict{Int, Int}}()
    group_specs = Dict{Int, Dict{String,Any}}()

    for relation in property_relations
        relation_id = Int(relation["id"])
        push!(relation_ids, relation_id)
        family = get(relation, "candidate_design_variable_family", nothing)
        family in ("shell_thickness", "bar_area") || error("SOL 200-lite route only supports shell-thickness and bar-area DVPREL1 relations")
        kind = Symbol(family)
        push!(kinds_raw, kind)
        property_id = Int(relation["property_id"])
        owner = get!(property_owner, kind, Dict{Int, Int}())
        if haskey(owner, property_id)
            error("SOL 200-lite route requires at most one supported DVPREL1 relation per active property in sizing family $kind")
        end
        owner[property_id] = relation_id

        coeffs = get(relation, "coefficients", Any[])
        length(coeffs) == 1 || error("SOL 200-lite route requires one DESVAR per supported DVPREL1 relation")
        coef = Float64(coeffs[1]["COEF"])
        _sol200_lite_numeric_equal(coef, 1.0) || error("SOL 200-lite route requires unit DVPREL1 coefficients on the supported subset")
        _sol200_lite_numeric_equal(get(relation, "offset", 0.0), 0.0) || error("SOL 200-lite route requires zero DVPREL1 offsets on the supported subset")

        desvar_id = Int(coeffs[1]["DESVAR_ID"])
        haskey(design_vars, desvar_id) || error("SOL 200-lite route could not find DESVAR $desvar_id referenced by DVPREL1 $relation_id")
        dv = design_vars[desvar_id]
        x_init = Float64(get(dv, "x_init", 0.0))
        group = get!(group_specs, desvar_id) do
            lower_candidates = Float64[]
            upper_candidates = Float64[]
            dv_lower = get(dv, "lower_bound", nothing)
            dv_upper = get(dv, "upper_bound", nothing)
            !isnothing(dv_lower) && push!(lower_candidates, Float64(dv_lower))
            !isnothing(dv_upper) && push!(upper_candidates, Float64(dv_upper))
            Dict{String,Any}(
                "design_var_id" => desvar_id,
                "label" => isempty(strip(string(get(dv, "label", "")))) ? "desvar_$desvar_id" : strip(string(get(dv, "label", ""))),
                "initial_value" => x_init,
                "move_limit" => get(dv, "move_limit", nothing),
                "relation_ids" => Int[],
                "property_ids_by_kind" => Dict{Symbol, Set{Int}}(),
                "lower_candidates" => lower_candidates,
                "upper_candidates" => upper_candidates,
            )
        end
        _sol200_lite_numeric_equal(Float64(group["initial_value"]), x_init) ||
            error("SOL 200-lite route found inconsistent DESVAR initialization for grouped design variable $desvar_id")
        push!(group["relation_ids"], relation_id)
        push!(get!(group["property_ids_by_kind"], kind, Set{Int}()), property_id)
        relation_lower = get(relation, "lower_bound", nothing)
        relation_upper = get(relation, "upper_bound", nothing)
        !isnothing(relation_lower) && push!(group["lower_candidates"], Float64(relation_lower))
        !isnothing(relation_upper) && push!(group["upper_candidates"], Float64(relation_upper))
    end

    kinds = Solver._normalize_sizing_kinds(kinds_raw)
    _sol200_lite_validate_mass_scope(model, kinds)

    for kind in kinds
        usage = _sol200_lite_property_usage(model, kind)
        isempty(usage) && error("SOL 200-lite route did not find any elements for sizing family $kind")
        selected_pids = Set{Int}()
        for group in values(group_specs)
            for pid in get(group["property_ids_by_kind"], kind, Set{Int}())
                push!(selected_pids, pid)
            end
        end
        selected_pids == Set(keys(usage)) || error("SOL 200-lite route requires DVPREL1 coverage for every active property in sizing family $kind")
    end

    constraint_summary = Dict{String,Any}()
    mass_target = nothing
    response_constraints = Dict{String,Any}[]
    if translated_objective == :min_compliance || translated_objective == :max_buckling
        mass_constraints = Dict{String,Any}[]
        for constraint in get(opt, "constraints", Any[])
            response_id = Int(constraint["response_id"])
            haskey(responses, response_id) || error("SOL 200-lite route could not find DCONSTR response $response_id")
            response = responses[response_id]
            family = get(response, "candidate_response_family", nothing)
            if family == "mass" && isnothing(get(constraint, "lower_allowable", nothing)) && !isnothing(get(constraint, "upper_allowable", nothing))
                push!(mass_constraints, constraint)
            else
                error("SOL 200-lite route currently supports exactly one upper-bound WEIGHT/MASS constraint and no other DCONSTR entries on the compliance/buckling subset")
            end
        end
        length(mass_constraints) == 1 || error("SOL 200-lite route requires exactly one upper-bound WEIGHT/MASS constraint")

        mass_constraint = mass_constraints[1]
        mass_target = Float64(mass_constraint["upper_allowable"])
        mass_target > 0.0 || error("SOL 200-lite route requires a positive WEIGHT/MASS upper bound")
        constraint_summary = Dict(
            "constraint_response_id" => Int(mass_constraint["response_id"]),
            "constraint_response_family" => "mass",
            "absolute_mass_target" => mass_target,
        )
    else
        static_constraints = Dict{String,Any}[]
        for constraint in get(opt, "constraints", Any[])
            isnothing(get(constraint, "lower_allowable", nothing)) ||
                error("SOL 200-lite mass-minimization route currently supports upper-bound static constraints only")
            !isnothing(get(constraint, "upper_allowable", nothing)) ||
                error("SOL 200-lite mass-minimization route requires an upper allowable on the routed static constraint")
            response_id = Int(constraint["response_id"])
            haskey(responses, response_id) || error("SOL 200-lite route could not find DCONSTR response $response_id")
            response = responses[response_id]
            family = get(response, "candidate_response_family", nothing)
            family in ("compliance", "displacement") ||
                error("SOL 200-lite mass-minimization route only supports upper-bound compliance or displacement constraints on the routed static subset")
            push!(static_constraints, Dict(
                "constraint" => constraint,
                "response" => response,
                "family" => family,
            ))
        end
        !isempty(static_constraints) || error("SOL 200-lite mass-minimization route requires at least one upper-bound compliance or displacement constraint")
        constraint_entries = Dict{String,Any}[]
        for selected_constraint in static_constraints
            response_upper_bound = Float64(selected_constraint["constraint"]["upper_allowable"])
            isfinite(response_upper_bound) || error("SOL 200-lite mass-minimization route requires finite static-response upper bounds")
            response_family, response_spec = _sol200_lite_static_response_spec(selected_constraint["response"])
            push!(response_constraints, Dict{String,Any}(
                "response_id" => Int(selected_constraint["constraint"]["response_id"]),
                "family" => response_family,
                "spec" => deepcopy(response_spec),
                "upper_bound" => response_upper_bound,
            ))
            entry = Dict{String,Any}(
                "constraint_response_id" => Int(selected_constraint["constraint"]["response_id"]),
                "constraint_response_family" => String(response_family),
                "constraint_response_upper_bound" => response_upper_bound,
            )
            if response_family == :displacement
                entry["constraint_grid"] = Int(response_spec["grid"])
                entry["constraint_dof"] = Int(response_spec["dof"])
            end
            push!(constraint_entries, entry)
        end
        constraint_summary = Dict(
            "constraint_count" => length(constraint_entries),
            "constraint_responses" => deepcopy(constraint_entries),
        )
        if length(constraint_entries) == 1
            merge!(constraint_summary, constraint_entries[1])
        end
    end

    optimizer_params = get(opt, "optimizer_params", Dict{String,Any}())
    max_iter = Int(round(Float64(get(optimizer_params, "DESMAX", 50))))
    tol = Float64(get(optimizer_params, "CONV1", 1e-3))
    default_move_limit_raw = get(optimizer_params, "DELX", nothing)
    default_move_limit =
        if isnothing(default_move_limit_raw)
            0.2
        else
            value = Float64(default_move_limit_raw)
            value > 0.0 || error("SOL 200-lite route requires DOPTPRM DELX to be positive when provided")
            value
        end

    grouped_design_variables = Dict{String,Any}[]
    x_min = Float64[]
    x_max = Float64[]
    move_limit = Float64[]
    sorted_group_specs = Dict{String,Any}[]
    for desvar_id in sort!(collect(keys(group_specs)))
        group = group_specs[desvar_id]
        lower_candidates = Float64[group["lower_candidates"]...]
        upper_candidates = Float64[group["upper_candidates"]...]
        group_x_min = isempty(lower_candidates) ? 1e-4 : maximum(lower_candidates)
        group_x_max = isempty(upper_candidates) ? 0.0 : minimum(upper_candidates)
        group_move_limit = isnothing(group["move_limit"]) ? default_move_limit : Float64(group["move_limit"])
        x_init = Float64(group["initial_value"])
        group_x_max > 0.0 && group_x_max + 1e-12 < group_x_min &&
            error("SOL 200-lite route found incompatible lower/upper bounds on grouped design variable $desvar_id")
        x_init + 1e-12 < group_x_min &&
            error("SOL 200-lite route requires DESVAR $desvar_id initial value to satisfy the translated lower bound")
        group_x_max > 0.0 && x_init > group_x_max + 1e-12 &&
            error("SOL 200-lite route requires DESVAR $desvar_id initial value to satisfy the translated upper bound")

        property_ids_by_kind = Dict{String,Any}()
        design_variable_types = String[]
        for kind in sort!(collect(keys(group["property_ids_by_kind"])); by=String)
            push!(design_variable_types, String(kind))
            property_ids_by_kind[String(kind)] = sort!(collect(group["property_ids_by_kind"][kind]))
        end

        group_summary = Dict{String,Any}(
            "design_var_id" => desvar_id,
            "label" => String(group["label"]),
            "design_variable_types" => design_variable_types,
            "relation_ids" => sort!(copy(group["relation_ids"])),
            "property_ids_by_kind" => property_ids_by_kind,
            "initial_value" => x_init,
            "lower_bound" => group_x_min,
            "upper_bound" => group_x_max,
            "move_limit" => group_move_limit,
        )
        push!(grouped_design_variables, deepcopy(group_summary))

        push!(sorted_group_specs, Dict(
            "design_var_id" => desvar_id,
            "label" => String(group["label"]),
            "relation_ids" => sort!(copy(group["relation_ids"])),
            "property_ids_by_kind" => Dict(kind => sort!(collect(ids)) for (kind, ids) in group["property_ids_by_kind"]),
            "initial_value" => x_init,
        ))
        push!(x_min, group_x_min)
        push!(x_max, group_x_max)
        push!(move_limit, group_move_limit)
    end

    route_summary = Dict{String,Any}(
        "translation_mode" => translated_objective == :min_mass_static_response ?
            (length(sorted_group_specs) == 1 ?
                "single_group_static_response_mass_minimization" :
                "multi_group_static_response_mass_minimization") :
            "grouped_sizing_exact_on_supported_subset",
        "translated_objective" => String(translated_objective),
        "forward_sol_type" => forward_sol_type,
        "objective_response_id" => objective_response_id,
        "objective_response_family" => objective_family,
        "design_variable_types" => [String(kind) for kind in kinds],
        "relation_ids" => relation_ids,
        "grouped_design_variable_count" => length(sorted_group_specs),
        "grouped_design_variables" => grouped_design_variables,
        "group_semantics_preserved" => true,
        "default_move_limit" => default_move_limit,
        "max_iter" => max_iter,
        "tol" => tol,
    )
    merge!(route_summary, constraint_summary)

    return translated_objective, forward_sol_type, kinds, x_min, x_max, move_limit, max_iter, tol, mass_target, response_constraints, route_summary, sorted_group_specs
end

function _solve_sol200_lite(model::Dict)
    translated_objective, forward_sol_type, kinds, x_min, x_max, move_limit, max_iter, tol, mass_target, response_constraints, route_summary, group_specs =
        _sol200_lite_translate(model)

    opt_model = deepcopy(model)
    opt_model["SOL"] = forward_sol_type
    opt_model["CASE_CONTROL"]["SOL"] = forward_sol_type
    selected_sid = _sol200_lite_select_subcases!(opt_model, translated_objective, opt_model["OPTIMIZATION"])
    if !isnothing(selected_sid)
        route_summary["selected_subcase"] = selected_sid
    end

    membership_by_kind = Dict{Symbol, Dict{Int, Vector{Int}}}()
    for kind in kinds
        membership_by_kind[kind] = _sol200_lite_property_membership(opt_model, kind)
    end

    for group in group_specs
        for (kind, pids) in group["property_ids_by_kind"]
            for pid in pids
                _sol200_lite_apply_group_initial_value!(opt_model, kind, pid, group["initial_value"])
            end
        end
    end

    vars = Solver._prepare_sizing_variable_data!(opt_model, kinds)
    isempty(vars) && error("SOL 200-lite route did not produce any active sizing variables")
    var_lookup = Dict{Tuple{Symbol, Int}, Solver.SizingVarData}()
    for var in vars
        var_lookup[(var.kind, var.eid)] = var
    end

    grouped_vars = Solver.SizingGroupData[]
    for group in group_specs
        members = Solver.SizingVarData[]
        active_element_count_by_kind = Dict{String,Any}()
        for kind in sort!(collect(keys(group["property_ids_by_kind"])); by=String)
            member_eids = Int[]
            for pid in group["property_ids_by_kind"][kind]
                append!(member_eids, get(membership_by_kind[kind], pid, Int[]))
            end
            sort!(member_eids)
            unique!(member_eids)
            active_element_count_by_kind[String(kind)] = length(member_eids)
            for eid in member_eids
                haskey(var_lookup, (kind, eid)) ||
                    error("SOL 200-lite route could not map grouped design variable $(group["design_var_id"]) onto active element $eid in sizing family $kind")
                push!(members, var_lookup[(kind, eid)])
            end
        end

        sort!(members, by=var -> (String(var.kind), var.eid, var.pid_new))
        isempty(members) && error("SOL 200-lite route found no active split properties for grouped design variable $(group["design_var_id"])")
        mass_coeff = sum(var.mass_coeff for var in members)
        fixed_mass = sum(var.fixed_mass for var in members)
        push!(grouped_vars, Solver.SizingGroupData(
            String(group["label"]),
            Int(group["design_var_id"]),
            members,
            Float64(group["initial_value"]),
            Int[group["relation_ids"]...],
            Dict(kind => Int[ids...] for (kind, ids) in group["property_ids_by_kind"]),
            mass_coeff,
            fixed_mass,
        ))
        for entry in get(route_summary, "grouped_design_variables", Any[])
            if Int(entry["design_var_id"]) == Int(group["design_var_id"])
                entry["active_element_count_by_kind"] = deepcopy(active_element_count_by_kind)
                entry["member_count"] = length(members)
            end
        end
    end

    variable_mass_initial = sum(grouped_var.x0 * grouped_var.mass_coeff for grouped_var in grouped_vars)
    fixed_mass_initial = sum(grouped_var.fixed_mass for grouped_var in grouped_vars)
    total_mass_initial = variable_mass_initial + fixed_mass_initial
    variable_mass_initial > 0.0 || error("SOL 200-lite route could not compute a positive initial sizing mass")
    route_summary["variable_mass_initial"] = variable_mass_initial
    route_summary["fixed_mass_initial"] = fixed_mass_initial
    route_summary["total_mass_initial"] = total_mass_initial

    opt_result = if translated_objective == :min_mass_static_response
        if length(grouped_vars) == 1
            route_summary["single_group_exact_search"] = true
            route_summary["grouped_design_variable_count"] = 1
            Solver.optimize_grouped_single_variable_mass_minimization(
                opt_model, solve_model, grouped_vars[1];
                response_constraints=response_constraints,
                x_min=Float64(x_min[1]),
                x_max=Float64(x_max[1]),
                max_iter=max_iter,
                tol=tol,
            )
        else
            route_summary["single_group_exact_search"] = false
            route_summary["multi_group_response_search"] = true
            route_summary["multi_group_constraint_search"] = length(response_constraints) > 1
            Solver.optimize_grouped_static_response_mass_minimization(
                opt_model, solve_model, grouped_vars;
                response_constraints=response_constraints,
                x_min=x_min,
                x_max=x_max,
                max_iter=max_iter,
                tol=tol,
                move_limit=move_limit,
            )
        end
    else
        effective_variable_mass_target = max(mass_target - fixed_mass_initial, 0.0)
        vol_frac = mass_target / total_mass_initial
        route_summary["effective_variable_mass_target"] = effective_variable_mass_target
        route_summary["vol_frac"] = vol_frac
        Solver.optimize_grouped_sizing(opt_model, solve_model, grouped_vars;
            objective=translated_objective,
            x_min=x_min,
            x_max=x_max,
            vol_frac=Float64(vol_frac),
            max_iter=max_iter,
            tol=tol,
            move_limit=move_limit)
    end

    if translated_objective == :min_mass_static_response
        mass_polish = get(opt_result, "mass_polish", nothing)
        if mass_polish isa AbstractDict
            route_summary["mass_polish_attempted"] = Bool(get(mass_polish, "attempted", false))
            route_summary["mass_polish_applied"] = Bool(get(mass_polish, "applied", false))
            route_summary["used_nonmonotone_mass_fallback"] = Bool(get(mass_polish, "fallback_due_to_nonmonotone_route", false))
            route_summary["mass_polish_iterations"] = Int(get(mass_polish, "n_iter", 0))
        end
    end

    forward_results = solve_model(opt_result["model"])
    route_summary["final_forward_sol_type"] = forward_results["sol_type"]

    return Dict(
        "sol_type" => 200,
        "analysis_type" => "SOL200_LITE_OPTIMIZATION",
        "optimization" => opt_result,
        "route_summary" => route_summary,
        "forward_sol_type" => forward_results["sol_type"],
        "forward_results" => forward_results,
        "mesh" => get(forward_results, "mesh", Dict{String,Any}()),
        "model" => forward_results["model"],
        "id_map" => forward_results["id_map"],
        "node_coords" => forward_results["node_coords"],
        "solver_diagnostics" => Dict(
            "optimization_iterations" => get(opt_result, "n_iter", 0),
            "optimization_termination_reason" => get(opt_result, "termination_reason", ""),
            "forward_sol_type" => forward_results["sol_type"],
        ),
    )
end

# ============================================================================
# Stage 2: Model Dict → Solver → Results Dict
# ============================================================================

"""
    solve_model(model::Dict) -> Dict

Run the FEM solver on the model Dict and return a results Dict.
No file I/O is performed. The model Dict must contain all required fields
(as produced by bdf_to_model or deserialized from JSON).

Returns a Dict with keys:
  - "sol_type" => Int (101, 103, 105, or experimental 106)
  - "mesh"     => Dict with node/element tables for export
  - ... solution-specific results (displacements, eigenvalues, etc.)
"""
function solve_model(model::Dict)
    t_solve_start = time_ns()
    cc = model["CASE_CONTROL"]
    raw_sol_type = get(model, "SOL", get(cc, "SOL", 101))
    sol_type = _canonical_sol_type(raw_sol_type)
    if sol_type == 200
        results = _solve_sol200_lite(model)
        results["input_sol_type"] = raw_sol_type
        results["timings"] = Dict{String,Any}(
            "solve" => (time_ns() - t_solve_start) * 1e-9,
        )
        return results
    end
    sol105_snorm_angle = sol_type == 105 ? Solver.sol105_snorm_angle_override() : nothing

    # --- Assemble global stiffness ---
    t_asm = time_ns()
    static_membrane_incomp = sol_type == 105 ? Solver.sol105_static_membrane_incomp_enabled() : true
    K, id_map, X, ndof, node_R, max_elem_stiff, rbe3_map, snorm_normals, orig_diag = Solver.assemble_stiffness(
        model; snorm_angle_override=sol105_snorm_angle,
        membrane_incomp=static_membrane_incomp
    )
    t_asm_K = (time_ns() - t_asm) * 1e-9

    K_eig = K
    t_asm_Keig = 0.0
    sol105_use_static_k = sol_type == 105 && Solver.sol105_use_static_k_enabled()
    if sol_type == 105 && !sol105_use_static_k
        t_asm2 = time_ns()
        membrane_incomp_eig = Solver.solver_env_bool("JFEM_SOL105_EIG_MEMBRANE_INCOMP", false)
        pcomp_membrane_incomp_eig = Solver.solver_env_bool("JFEM_SOL105_EIG_PCOMP_MEMBRANE_INCOMP", false)
        bending_incomp_eig = Solver.sol105_eig_bending_incomp_enabled()
        K_eig, = Solver.assemble_stiffness(
            model;
            shear_center_only=true, bending_incomp=bending_incomp_eig,
            membrane_incomp=membrane_incomp_eig,
            pcomp_membrane_incomp=pcomp_membrane_incomp_eig,
            snorm_angle_override=sol105_snorm_angle, iso_no_incomp=true,
        )
        t_asm_Keig = (time_ns() - t_asm2) * 1e-9
    end

    sorted_sids = sort(collect(keys(cc["SUBCASES"])))

    # --- Build mesh tables for output ---
    mesh = _build_mesh_output(model, id_map, X)

    # --- Dispatch to solution type ---
    t_disp = time_ns()
    results = if sol_type == 105
        _solve_sol105(model, cc, K, K_eig, id_map, X, ndof, node_R,
            max_elem_stiff, rbe3_map, snorm_normals, orig_diag,
            sorted_sids, sol105_snorm_angle, mesh)
    elseif sol_type == 103
        _solve_sol103(model, cc, K, id_map, X, ndof, node_R,
            max_elem_stiff, rbe3_map, orig_diag, sorted_sids, mesh)
    elseif sol_type == 106
        _solve_sol106(model, cc, K, id_map, X, ndof, node_R,
            max_elem_stiff, rbe3_map, snorm_normals, orig_diag,
            sorted_sids, sol105_snorm_angle, mesh)
    else
        _solve_sol101(model, cc, K, id_map, X, ndof, node_R,
            max_elem_stiff, rbe3_map, snorm_normals, orig_diag,
            sorted_sids, mesh)
    end
    t_dispatch = (time_ns() - t_disp) * 1e-9

    results["input_sol_type"] = raw_sol_type
    existing_timings = get(results, "timings", Dict{String,Any}())
    merged_timings = Dict{String,Any}()
    if existing_timings isa AbstractDict
        for (k, v) in existing_timings
            merged_timings[string(k)] = v
        end
    end
    merge!(merged_timings, Dict{String,Any}(
        "assembly_K"    => t_asm_K,
        "assembly_Keig" => t_asm_Keig,
        "solve_cases"   => t_dispatch,
        "solve"         => (time_ns() - t_solve_start) * 1e-9,
    ))
    results["timings"] = merged_timings
    return results
end

function _solve_sol106(model, cc, K, id_map, X, ndof, node_R,
                       max_elem_stiff, rbe3_map, snorm_normals, orig_diag,
                       sorted_sids, sol106_snorm_angle, mesh)
    println("\n>>> SOL 106 Experimental Geometric Nonlinear Static Analysis")

    nl_residual_raw = lowercase(strip(string(get(model, "PARAM_NLRESMODEL", "tangent_operator"))))
    nl_residual_model =
        if nl_residual_raw in ("tangent", "tangent_operator", "keff", "linear_plus_geometric")
            :tangent_operator
        else
            :secant_geometric
        end
    nl_method_raw = lowercase(strip(string(get(model, "PARAM_NLMETHOD", "auto"))))
    nl_method =
        if nl_method_raw in ("formal", "formal_shell_von_karman", "formal_von_karman", "formal_shell_vk")
            :formal_shell_von_karman
        else
            :auto
        end

    load_steps = Int(get(model, "PARAM_NLLOADSTEPS", 4))
    max_iter = Int(get(model, "PARAM_NLMAXITER", 8))
    tol = Float64(get(model, "PARAM_NLTOL", 1e-6))
    relaxation = Float64(get(model, "PARAM_NLRELAX", 1.0))
    residual_tol = Float64(get(model, "PARAM_NLRESTOL", tol))
    line_search_max_backtracks = Int(get(model, "PARAM_NLLSMAX", 6))
    line_search_reduction = Float64(get(model, "PARAM_NLLSREDUCE", 0.5))
    max_cutbacks = Int(get(model, "PARAM_NLCUTMAX", 6))
    cutback_reduction = Float64(get(model, "PARAM_NLCUTREDUCE", 0.5))
    step_growth = Float64(get(model, "PARAM_NLSTEPGROW", 1.25))

    subcases_results = []
    solver_diagnostics = Any[]
    last_Kg = spzeros(ndof, ndof)
    last_K = K

    for sid in sorted_sids
        sub = cc["SUBCASES"][sid]
        println(">>> Solving Nonlinear Subcase $sid...")
        load_id = get(sub, "LOAD", nothing)
        spc_id = get(sub, "SPC", nothing)
        temp_load_id = Solver._subcase_temp_load_sid(sub, cc)
        needs_temp_reassembly = !isnothing(temp_load_id) && _model_has_temperature_dependent_mat1(model)

        K_sub = K
        id_map_sub = id_map
        X_sub = X
        ndof_sub = ndof
        node_R_sub = node_R
        max_elem_stiff_sub = max_elem_stiff
        rbe3_map_sub = rbe3_map
        snorm_normals_sub = snorm_normals
        orig_diag_sub = orig_diag

        run_subcase = function ()
            if needs_temp_reassembly
                K_sub, id_map_sub, X_sub, ndof_sub, node_R_sub, max_elem_stiff_sub, rbe3_map_sub, snorm_normals_sub, orig_diag_sub =
                    Solver.assemble_stiffness(model)
            end
            return Solver.solve_nonlinear_static(
                K_sub, ndof_sub, model, id_map_sub, X_sub, load_id, spc_id, node_R_sub;
                load_steps=load_steps,
                max_iter=max_iter,
                tol=tol,
                relaxation=relaxation,
                residual_tol=residual_tol,
                residual_model=nl_residual_model,
                nonlinear_method=nl_method,
                line_search_max_backtracks=line_search_max_backtracks,
                line_search_reduction=line_search_reduction,
                max_cutbacks=max_cutbacks,
                cutback_reduction=cutback_reduction,
                step_growth=step_growth,
                max_elem_stiff=max_elem_stiff_sub,
                rbe3_map=rbe3_map_sub,
                snorm_normals=snorm_normals_sub,
                orig_diag=orig_diag_sub,
                temp_load_id=temp_load_id,
            )
        end

        t_sc = time_ns()
        u, stresses, sub_res, u_analysis, fixed_dofs_sc, Kg =
            needs_temp_reassembly ?
                _with_active_temperature_material(model, temp_load_id, run_subcase) :
                run_subcase()
        wall_seconds = (time_ns() - t_sc) * 1e-9
        last_K = K_sub
        last_Kg = Kg
        push!(solver_diagnostics, Dict(
            "sid" => sid,
            "details" => get(sub_res, "nonlinear_diagnostics", Dict{String,Any}()),
        ))

        push!(subcases_results, Dict(
            "sid" => sid,
            "displacements" => sub_res["displacements"],
            "spc_forces" => sub_res["spc_forces"],
            "forces" => sub_res["forces"],
            "forces_bilin" => get(sub_res, "forces_bilin", Dict("quad4" => Any[], "tria3" => Any[])),
            "stresses" => sub_res["stresses"],
            "strains" => sub_res["strains"],
            "solver_diagnostics" => get(sub_res, "solver_diagnostics", Dict{String,Any}()),
            "nonlinear_diagnostics" => get(sub_res, "nonlinear_diagnostics", Dict{String,Any}()),
            "raw_displacement" => collect(u),
            "element_vonmises" => stresses,
            "u_analysis" => u_analysis,
            "fixed_dofs" => fixed_dofs_sc,
            "wall_seconds" => wall_seconds,
        ))
    end

    return Dict(
        "sol_type" => 106,
        "subcases" => subcases_results,
        "solver_diagnostics" => solver_diagnostics,
        "mesh" => mesh,
        "model" => model,
        "id_map" => id_map,
        "node_coords" => X,
        "K" => last_K,
        "Kg" => last_Kg,
        "ndof" => ndof,
        "node_R" => node_R,
        "rbe3_map" => rbe3_map,
    )
end

"""
    solve_model_json(model_json::String) -> String

Solve from JSON input and return JSON output.
Convenience wrapper for cross-language interop.
"""
function solve_model_json(model_json::String)
    model = JSON.parse(model_json)
    results = solve_model(model)
    return JSON.json(results)
end

# ============================================================================
# SOL 101: Linear Static
# ============================================================================
function _solve_sol101(model, cc, K, id_map, X, ndof, node_R,
                       max_elem_stiff, rbe3_map, snorm_normals, orig_diag,
                       sorted_sids, mesh)
    println("\n>>> SOL 101 Linear Static Analysis")

    subcases_results = []
    last_K = K
    linear_solve_cache = ndof >= Solver.linear_solve_cache_min_ndof() ? Solver.create_linear_solve_cache() : nothing
    for sid in sorted_sids
        sub = cc["SUBCASES"][sid]
        println(">>> Solving Subcase $sid...")
        load_id = get(sub, "LOAD", nothing)
        spc_id = get(sub, "SPC", nothing)
        temp_load_id = Solver._subcase_temp_load_sid(sub, cc)
        needs_temp_reassembly = !isnothing(temp_load_id) && _model_has_temperature_dependent_mat1(model)

        K_sub = K
        id_map_sub = id_map
        X_sub = X
        ndof_sub = ndof
        node_R_sub = node_R
        max_elem_stiff_sub = max_elem_stiff
        rbe3_map_sub = rbe3_map
        snorm_normals_sub = snorm_normals
        orig_diag_sub = orig_diag

        run_subcase = function ()
            if needs_temp_reassembly
                K_sub, id_map_sub, X_sub, ndof_sub, node_R_sub, max_elem_stiff_sub, rbe3_map_sub, snorm_normals_sub, orig_diag_sub =
                    Solver.assemble_stiffness(model)
            end
            return Solver.solve_case(
                K_sub, ndof_sub, model, id_map_sub, X_sub, load_id, spc_id, node_R_sub;
                max_elem_stiff=max_elem_stiff_sub, rbe3_map=rbe3_map_sub,
                snorm_normals=snorm_normals_sub, orig_diag=orig_diag_sub,
                temp_load_id=temp_load_id, linear_cache=linear_solve_cache)
        end

        t_sc = time_ns()
        u, stresses, sub_res, u_analysis, fixed_dofs_sc =
            needs_temp_reassembly ?
                _with_active_temperature_material(model, temp_load_id, run_subcase) :
                run_subcase()
        wall_seconds = (time_ns() - t_sc) * 1e-9
        last_K = K_sub

        # Store raw displacement vector for VTK/binary export
        push!(subcases_results, Dict(
            "sid" => sid,
            "displacements" => sub_res["displacements"],
            "spc_forces" => sub_res["spc_forces"],
            "forces" => sub_res["forces"],
            "forces_bilin" => get(sub_res, "forces_bilin", Dict("quad4" => Any[], "tria3" => Any[])),
            "stresses" => sub_res["stresses"],
            "strains" => sub_res["strains"],
            "solver_diagnostics" => get(sub_res, "solver_diagnostics", Dict{String,Any}()),
            "raw_displacement" => collect(u),
            "element_vonmises" => stresses,
            "u_analysis" => u_analysis,
            "fixed_dofs" => fixed_dofs_sc,
            "wall_seconds" => wall_seconds,
        ))
    end

    return Dict(
        "sol_type" => 101,
        "subcases" => subcases_results,
        "mesh" => mesh,
        "model" => model,      # pass through for export functions that need it
        "id_map" => id_map,
        "node_coords" => X,
        "K" => last_K,
        "ndof" => ndof,
        "node_R" => node_R,
        "rbe3_map" => rbe3_map,
    )
end

# ============================================================================
# SOL 103: Normal Modes
# ============================================================================
function _solve_sol103(model, cc, K, id_map, X, ndof, node_R,
                       max_elem_stiff, rbe3_map, orig_diag, sorted_sids, mesh)
    println("\n>>> SOL 103 Normal Modes Analysis")

    modal_sids = Int[]
    global_method = get(cc, "METHOD", nothing)
    for sid in sorted_sids
        sub = cc["SUBCASES"][sid]
        method_id = get(sub, "METHOD", global_method)
        if !isnothing(method_id) || isempty(sorted_sids)
            push!(modal_sids, sid)
        end
    end
    if isempty(modal_sids) && !isempty(sorted_sids)
        push!(modal_sids, sorted_sids[1])
    end

    all_eigenvalues = Float64[]
    all_frequencies = Float64[]
    all_mode_shapes = Vector{Vector{Float64}}()
    all_modal_effective_mass = Any[]
    modal_subcases = Any[]
    modal_case_diagnostics = Any[]
    eigen_solve_cache = ndof >= Solver.eigen_solve_cache_min_ndof() ? Solver.create_eigen_solve_cache() : nothing

    id_map_modal = id_map
    X_modal = X
    first_mass_summary = Dict{String,Any}()

    for sid in modal_sids
        sub = cc["SUBCASES"][sid]
        method_id = get(sub, "METHOD", global_method)
        eigrl = nothing
        if !isnothing(method_id)
            eigrl = get(model["EIGRLs"], string(Int(method_id)), nothing)
        end

        default_modes = eigrl === nothing ? 10 : Int(get(eigrl, "ND", 10))
        num_modes = Int(get(sub, "MODES", default_modes))
        num_modes <= 0 && (num_modes = max(default_modes, 1))

        eigrl_v1 = eigrl === nothing ? 0.0 : Float64(get(eigrl, "V1", 0.0))
        eigrl_v2 = eigrl === nothing ? 0.0 : Float64(get(eigrl, "V2", 0.0))
        eigrl_norm = eigrl === nothing ? "MASS" : string(get(eigrl, "NORM", "MASS"))
        spc_id = get(sub, "SPC", get(cc, "SPC", nothing))
        temp_load_id = Solver._subcase_temp_load_sid(sub, cc)
        needs_temp_reassembly = !isnothing(temp_load_id) && _model_has_temperature_dependent_mat1(model)

        K_sub = K
        id_map_sub = id_map
        X_sub = X
        ndof_sub = ndof
        node_R_sub = node_R
        max_elem_stiff_sub = max_elem_stiff
        rbe3_map_sub = rbe3_map
        orig_diag_sub = orig_diag

        run_modal = function ()
            if needs_temp_reassembly
                K_sub, id_map_sub, X_sub, ndof_sub, node_R_sub, max_elem_stiff_sub, rbe3_map_sub, _, orig_diag_sub =
                    Solver.assemble_stiffness(model)
            end
            return Solver.assemble_mass(model, id_map_sub, X_sub, node_R_sub, ndof_sub)
        end

        M = needs_temp_reassembly ?
            _with_active_temperature_material(model, temp_load_id, run_modal) :
            run_modal()

        if length(modal_sids) == 1
            println(">>> Solving Normal Modes ($num_modes modes)")
        else
            println(">>> Solving modal subcase $sid ($num_modes mode$(num_modes == 1 ? "" : "s"), METHOD=$(something(method_id, "none")), SPC=$(something(spc_id, "none")))")
        end

        t_modes = time_ns()
        eigenvalues, frequencies, mode_shapes, solver_diagnostics = Solver.solve_modes(
            K_sub, M, ndof_sub, model, id_map_sub, X_sub, spc_id, node_R_sub, num_modes;
            rbe3_map=rbe3_map_sub, max_elem_stiff=max_elem_stiff_sub, orig_diag=orig_diag_sub,
            eigrl_v1=eigrl_v1, eigrl_v2=eigrl_v2, eigrl_norm=eigrl_norm,
            eigen_cache=eigen_solve_cache, return_diagnostics=true)
        modal_wall_seconds = (time_ns() - t_modes) * 1e-9

        total_mass = [0.0, 0.0, 0.0]
        for (_, idx) in id_map_sub
            base = (idx - 1) * 6
            for d in 1:3
                total_mass[d] += M[base + d, base + d]
            end
        end
        wtmass = Float64(get(model, "PARAM_WTMASS", 1.0))

        modal_effective_mass = []
        for i in eachindex(frequencies)
            phi = mode_shapes[:, i]
            meff = zeros(3)
            for dir in 1:3
                L_eff = 0.0
                for (_, idx_n) in id_map_sub
                    base = (idx_n - 1) * 6
                    L_eff += M[base + dir, base + dir] * phi[base + dir]
                    for d2 in 1:6
                        if d2 != dir
                            L_eff += M[base + dir, base + d2] * phi[base + d2]
                        end
                    end
                end
                gen_mass = dot(phi, M * phi)
                meff[dir] = gen_mass > 1e-30 ? L_eff^2 / gen_mass : 0.0
            end
            push!(modal_effective_mass, Dict(
                "mode" => i,
                "freq" => frequencies[i],
                "meff_x" => meff[1],
                "meff_y" => meff[2],
                "meff_z" => meff[3],
            ))
        end

        if length(modal_sids) > 1
            println(">>> SOL 103 modal subcase $sid results")
        end
        _print_sol103_results(eigenvalues, frequencies, total_mass, wtmass, modal_effective_mass)

        mass_summary = Dict(
            "total_mass_x" => total_mass[1],
            "total_mass_y" => total_mass[2],
            "total_mass_z" => total_mass[3],
            "wtmass" => wtmass,
        )
        mode_shapes_out = _mode_shapes_to_list(mode_shapes, id_map_sub)

        push!(modal_subcases, Dict(
            "sid" => sid,
            "method_id" => method_id,
            "requested_modes" => num_modes,
            "spc_id" => spc_id,
            "temp_load_id" => temp_load_id,
            "eigenvalues" => collect(eigenvalues),
            "frequencies" => collect(frequencies),
            "mode_shapes" => mode_shapes_out,
            "mass_summary" => mass_summary,
            "modal_effective_mass" => modal_effective_mass,
            "solver_diagnostics" => solver_diagnostics,
            "_raw_mode_shapes" => mode_shapes,
            "wall_seconds" => modal_wall_seconds,
        ))
        push!(modal_case_diagnostics, Dict(
            "sid" => sid,
            "details" => solver_diagnostics,
        ))

        isempty(first_mass_summary) && merge!(first_mass_summary, mass_summary)
        id_map_modal = id_map_sub
        X_modal = X_sub

        for i in eachindex(frequencies)
            push!(all_eigenvalues, eigenvalues[i])
            push!(all_frequencies, frequencies[i])
            push!(all_mode_shapes, mode_shapes[:, i])
            push!(all_modal_effective_mass, Dict(
                "mode" => length(all_modal_effective_mass) + 1,
                "sid" => sid,
                "local_mode" => i,
                "freq" => frequencies[i],
                "meff_x" => modal_effective_mass[i]["meff_x"],
                "meff_y" => modal_effective_mass[i]["meff_y"],
                "meff_z" => modal_effective_mass[i]["meff_z"],
            ))
        end
    end

    combined_mode_shapes = isempty(all_mode_shapes) ? zeros(ndof, 0) : hcat(all_mode_shapes...)
    combined_mode_shapes_out = _mode_shapes_to_list(combined_mode_shapes, id_map_modal)
    top_level_diagnostics = length(modal_subcases) == 1 ?
        modal_subcases[1]["solver_diagnostics"] :
        modal_case_diagnostics

    return Dict(
        "sol_type" => 103,
        "eigenvalues" => collect(all_eigenvalues),
        "frequencies" => collect(all_frequencies),
        "mode_shapes" => combined_mode_shapes_out,
        "mass_summary" => first_mass_summary,
        "modal_effective_mass" => all_modal_effective_mass,
        "solver_diagnostics" => top_level_diagnostics,
        "subcases" => modal_subcases,
        "mesh" => mesh,
        "model" => model,
        "id_map" => id_map_modal,
        "node_coords" => X_modal,
        "_raw_mode_shapes" => combined_mode_shapes,  # kept for binary export, not serialized to JSON
    )
end

# ============================================================================
# SOL 105: Linear Buckling
# ============================================================================
function _solve_sol105(model, cc, K, K_eig, id_map, X, ndof, node_R,
                       max_elem_stiff, rbe3_map, snorm_normals, orig_diag,
                       sorted_sids, sol105_snorm_angle, mesh)
    println("\n>>> SOL 105 Linear Buckling Analysis")

    # Collect buckling subcases
    buckling_subcases = Tuple{Int, Int}[]
    for sid in sorted_sids
        sub = cc["SUBCASES"][sid]
        if haskey(sub, "METHOD") && !isnothing(sub["METHOD"])
            statsub_ref = get(sub, "STATSUB", nothing)
            static_sid = !isnothing(statsub_ref) ? Int(statsub_ref) : nothing
            push!(buckling_subcases, (sid, isnothing(static_sid) ? 0 : static_sid))
        end
    end

    if !isempty(buckling_subcases) && all(p -> p[2] == 0, buckling_subcases)
        default_static = nothing
        for sid in sorted_sids
            sub = cc["SUBCASES"][sid]
            if haskey(sub, "LOAD") && !isnothing(sub["LOAD"]) && !any(p -> p[1] == sid, buckling_subcases)
                default_static = sid; break
            end
        end
        if isnothing(default_static); default_static = sorted_sids[1]; end
        buckling_subcases = [(p[1], default_static) for p in buckling_subcases]
    end

    if isempty(buckling_subcases)
        buckling_subcases = [(length(sorted_sids) >= 2 ? sorted_sids[2] : sorted_sids[1], sorted_sids[1])]
    end

    static_cache = Dict{Int, Any}()
    static_linear_solve_cache = ndof >= Solver.linear_solve_cache_min_ndof() ? Solver.create_linear_solve_cache() : nothing
    eigen_solve_cache = ndof >= Solver.eigen_solve_cache_min_ndof() ? Solver.create_eigen_solve_cache() : nothing
    sol105_use_static_k = Solver.sol105_use_static_k_enabled()
    sol105_static_wall_seconds = 0.0
    sol105_kg_wall_seconds = 0.0
    sol105_buckling_wall_seconds = 0.0
    sol105_static_cache_hits = 0
    sol105_static_cache_misses = 0
    sol105_eigen_seeded_from_static = 0
    sol105_static_full_recovery = Solver.solver_env_bool("JFEM_SOL105_STATIC_FULL_RECOVERY", false)
    sol105_eigenvalues_only = Solver.solver_env_bool("JFEM_SOL105_EIGENVALUES_ONLY", false)
    all_eigenvalues = Float64[]
    all_mode_shapes = Vector{Vector{Float64}}()
    buckling_case_diagnostics = Any[]
    last_Kg = spzeros(ndof, ndof)
    last_u_static = zeros(ndof)
    last_fixed_dofs = Set{Int}()
    last_K = K
    last_K_eig = K_eig

    for (buck_sid, stat_sid) in buckling_subcases
        sub_buck = cc["SUBCASES"][buck_sid]
        sub_static = haskey(cc["SUBCASES"], stat_sid) ? cc["SUBCASES"][stat_sid] : cc["SUBCASES"][sorted_sids[1]]
        load_id = get(sub_static, "LOAD", nothing)
        spc_id_static = get(sub_static, "SPC", nothing)
        temp_load_id_static = Solver._subcase_temp_load_sid(sub_static, cc)
        needs_temp_reassembly = !isnothing(temp_load_id_static) && _model_has_temperature_dependent_mat1(model)
        static_membrane_incomp_default = Solver.sol105_static_membrane_incomp_enabled()
        static_membrane_incomp_auto_load = Solver.sol105_static_membrane_incomp_auto_load_enabled()
        static_membrane_incomp_for_load = static_membrane_incomp_default
        if !static_membrane_incomp_for_load && static_membrane_incomp_auto_load
            static_membrane_incomp_for_load =
                Solver.kg_quad4_auto_avg_load_classifier(model, load_id) === true
            if static_membrane_incomp_for_load
                println(">>> SOL105 static membrane-incomp auto-load enabled for LOAD=$load_id")
            end
        end
        needs_static_reassembly =
            needs_temp_reassembly ||
            static_membrane_incomp_for_load != static_membrane_incomp_default
        static_wall_seconds = 0.0

        if !haskey(static_cache, stat_sid)
            sol105_static_cache_misses += 1
            println(">>> Solving static reference subcase $stat_sid (LOAD=$load_id, SPC=$spc_id_static)")
            K_static = K
            K_eig_static = K_eig
            id_map_static = id_map
            X_static = X
            ndof_static = ndof
            node_R_static = node_R
            max_elem_stiff_static = max_elem_stiff
            rbe3_map_static = rbe3_map
            snorm_normals_static = snorm_normals
            orig_diag_static = orig_diag

            run_static = function ()
                if needs_static_reassembly
                    K_static, id_map_static, X_static, ndof_static, node_R_static, max_elem_stiff_static, rbe3_map_static, snorm_normals_static, orig_diag_static =
                        Solver.assemble_stiffness(
                            model;
                            snorm_angle_override=sol105_snorm_angle,
                            membrane_incomp=static_membrane_incomp_for_load,
                        )
                    if sol105_use_static_k
                        K_eig_static = K_static
                    else
                        membrane_incomp_eig = Solver.solver_env_bool("JFEM_SOL105_EIG_MEMBRANE_INCOMP", false)
                        pcomp_membrane_incomp_eig = Solver.solver_env_bool("JFEM_SOL105_EIG_PCOMP_MEMBRANE_INCOMP", false)
                        bending_incomp_eig = Solver.sol105_eig_bending_incomp_enabled()
                        K_eig_static, = Solver.assemble_stiffness(
                            model;
                            shear_center_only=true, bending_incomp=bending_incomp_eig,
                            membrane_incomp=membrane_incomp_eig,
                            pcomp_membrane_incomp=pcomp_membrane_incomp_eig,
                            snorm_angle_override=sol105_snorm_angle, iso_no_incomp=true,
                        )
                    end
                end
                if sol105_static_full_recovery
                    _, _, _, u_static_analysis, fixed_dofs_static = Solver.solve_case(
                        K_static, ndof_static, model, id_map_static, X_static, load_id, spc_id_static, node_R_static;
                        max_elem_stiff=max_elem_stiff_static, rbe3_map=rbe3_map_static,
                        snorm_normals=snorm_normals_static, orig_diag=orig_diag_static,
                        temp_load_id=temp_load_id_static, linear_cache=static_linear_solve_cache)
                else
                    u_static_analysis, fixed_dofs_static, _, _ = Solver.solve_case_state(
                        K_static, ndof_static, model, id_map_static, X_static, load_id, spc_id_static, node_R_static;
                        max_elem_stiff=max_elem_stiff_static, rbe3_map=rbe3_map_static, orig_diag=orig_diag_static,
                        temp_load_id=temp_load_id_static, linear_cache=static_linear_solve_cache)
                end
                return u_static_analysis, fixed_dofs_static
            end

            t_static_ref = time_ns()
            u_static_analysis, fixed_dofs_static =
                needs_temp_reassembly ?
                    _with_active_temperature_material(model, temp_load_id_static, run_static) :
                    run_static()
            static_wall_seconds = (time_ns() - t_static_ref) * 1e-9
            sol105_static_wall_seconds += static_wall_seconds
            static_cache[stat_sid] = (
                u_static_analysis, fixed_dofs_static,
                K_static, K_eig_static,
                id_map_static, X_static, ndof_static, node_R_static,
                max_elem_stiff_static, rbe3_map_static, snorm_normals_static, orig_diag_static,
            )
        else
            sol105_static_cache_hits += 1
        end
        u_static, fixed_dofs_static,
        K_static, K_eig_static,
        id_map_static, X_static, ndof_static, node_R_static,
        max_elem_stiff_static, rbe3_map_static, snorm_normals_static, orig_diag_static = static_cache[stat_sid]
        last_u_static = u_static
        last_fixed_dofs = fixed_dofs_static
        last_K = K_static
        last_K_eig = K_eig_static

        println(">>> Assembling Geometric Stiffness for buckling subcase $buck_sid (STATSUB=$stat_sid)")
        t_kg = time_ns()
        kg_phase_timings = Dict{String,Any}()
        Kg =
            if needs_temp_reassembly
                _with_active_temperature_material(model, temp_load_id_static) do
                    Solver.assemble_geometric_stiffness(
                        model, id_map_static, X_static, node_R_static, ndof_static, u_static, snorm_normals_static, rbe3_map_static;
                        snorm_angle_override=sol105_snorm_angle,
                        buckling_subcase=buck_sid,
                        static_load_id=load_id,
                        timings=kg_phase_timings)
                end
            else
                Solver.assemble_geometric_stiffness(
                    model, id_map_static, X_static, node_R_static, ndof_static, u_static, snorm_normals_static, rbe3_map_static;
                    snorm_angle_override=sol105_snorm_angle,
                    buckling_subcase=buck_sid,
                    static_load_id=load_id,
                    timings=kg_phase_timings)
            end
        kg_wall_seconds = (time_ns() - t_kg) * 1e-9
        sol105_kg_wall_seconds += kg_wall_seconds
        last_Kg = Kg

        method_id = get(sub_buck, "METHOD", nothing)
        num_modes = 3; eigrl_v1 = 0.0; eigrl_v2 = 0.0
        if !isnothing(method_id)
            eigrl = get(model["EIGRLs"], string(Int(method_id)), nothing)
            if !isnothing(eigrl)
                num_modes = get(eigrl, "ND", 3)
                eigrl_v1 = get(eigrl, "V1", 0.0)
                eigrl_v2 = get(eigrl, "V2", 0.0)
            end
        end

        spc_id_buck = get(sub_buck, "SPC", spc_id_static)
        eigrl_has_range = (eigrl_v1 != 0.0 || eigrl_v2 != 0.0) && eigrl_v2 > eigrl_v1
        if Solver.seed_eigen_solve_cache_from_linear!(
            eigen_solve_cache, static_linear_solve_cache,
            K_eig_static, ndof_static, model, spc_id_buck, rbe3_map_static)
            sol105_eigen_seeded_from_static += 1
            println(">>> Reusing static K factorization for buckling eigen solve")
        end
        println(">>> Solving Buckling Subcase $buck_sid ($num_modes modes)")
        t_buck = time_ns()
        eigenvalues, mode_shapes, buckling_diag = Solver.solve_buckling(K_eig_static, Kg, ndof_static, model, id_map_static, X_static, spc_id_buck, node_R_static, num_modes;
            rbe3_map=rbe3_map_static, max_elem_stiff=max_elem_stiff_static, orig_diag=orig_diag_static,
            eigrl_v1=eigrl_v1, eigrl_v2=eigrl_v2,
            eigen_cache=eigen_solve_cache,
            buckling_subcase=buck_sid,
            static_subcase=stat_sid,
            return_diagnostics=true)
        buck_wall_seconds = (time_ns() - t_buck) * 1e-9
        sol105_buckling_wall_seconds += buck_wall_seconds
        push!(buckling_case_diagnostics, Dict(
            "buckling_subcase" => buck_sid,
            "static_subcase" => stat_sid,
            "num_modes_requested" => num_modes,
            "eigrl_v1" => eigrl_v1,
            "eigrl_v2" => eigrl_v2,
            "eigrl_has_range" => eigrl_has_range,
            "eigenvalues" => collect(eigenvalues),
            "wall_seconds" => buck_wall_seconds,
            "phase_timings" => Dict{String,Any}(
                "static_reference_solve" => static_wall_seconds,
                "kg_assembly" => kg_wall_seconds,
                "buckling_eigensolve" => buck_wall_seconds,
            ),
            "kg_timings" => kg_phase_timings,
            "details" => buckling_diag,
        ))

        append!(all_eigenvalues, eigenvalues)
        for i in 1:size(mode_shapes, 2)
            push!(all_mode_shapes, mode_shapes[:, i])
        end
    end

    eigenvalues = all_eigenvalues
    mode_shapes = isempty(all_mode_shapes) ? zeros(ndof, 0) : hcat(all_mode_shapes...)

    # Console output
    println("\n>>> ============================================")
    println(">>> SOL 105 BUCKLING RESULTS")
    println(">>> ============================================")
    if !isempty(eigenvalues)
        @printf("    %-6s  %-20s\n", "Mode", "Lambda (Load Factor)")
        @printf("    %-6s  %-20s\n", "----", "--------------------")
        for (i, lam) in enumerate(eigenvalues)
            @printf("    %-6d  %20.6f\n", i, lam)
        end
    else
        println("    No valid buckling modes found.")
    end
    println(">>> ============================================")

    store_public_mode_shapes = Solver.solver_env_bool("JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES", true)
    mode_shapes_out = store_public_mode_shapes ? _mode_shapes_to_list(mode_shapes, id_map) : []

    return Dict(
        "sol_type" => 105,
        "eigenvalues" => collect(eigenvalues),
        "mode_shapes" => mode_shapes_out,
        "solver_diagnostics" => buckling_case_diagnostics,
        "mesh" => mesh,
        "model" => model,
        "id_map" => id_map,
        "node_coords" => X,
        "_raw_mode_shapes" => mode_shapes,
        "K" => last_K,
        "K_eig" => last_K_eig,
        "Kg" => last_Kg,
        "ndof" => ndof,
        "node_R" => node_R,
        "rbe3_map" => rbe3_map,
        "u_static" => last_u_static,
        "fixed_dofs" => last_fixed_dofs,
        "cache_diagnostics" => Dict{String,Any}(
            "static_cache_hits" => sol105_static_cache_hits,
            "static_cache_misses" => sol105_static_cache_misses,
            "linear_solve_cache_enabled" => static_linear_solve_cache !== nothing,
            "eigen_solve_cache_enabled" => eigen_solve_cache !== nothing,
            "eigen_seeded_from_static_linear_cache" => sol105_eigen_seeded_from_static,
            "static_full_recovery" => sol105_static_full_recovery,
            "eigenvalues_only" => sol105_eigenvalues_only,
            "public_mode_shapes_stored" => store_public_mode_shapes,
        ),
        "timings" => Dict{String,Any}(
            "sol105_static_reference_solve" => sol105_static_wall_seconds,
            "sol105_kg_assembly" => sol105_kg_wall_seconds,
            "sol105_buckling_eigensolve" => sol105_buckling_wall_seconds,
        ),
    )
end

# ============================================================================
# Helper: build mesh output dict
# ============================================================================
function _build_mesh_output(model, id_map, X)
    node_ids = sort(collect(keys(id_map)))
    nodes = []
    for nid in node_ids
        idx = id_map[nid]
        push!(nodes, Dict("id" => nid, "x" => X[idx,1], "y" => X[idx,2], "z" => X[idx,3]))
    end

    quads = []; trias = []
    for (id, el) in model["CSHELLs"]
        if !haskey(el, "NODES"); continue; end
        nids = el["NODES"]
        if !all(n -> haskey(id_map, n), nids); continue; end
        d = Dict("eid" => _solver_entry_public_id(id, el), "pid" => get(el, "PID", 0), "nodes" => nids)
        length(nids) == 4 ? push!(quads, d) : push!(trias, d)
    end

    bars = []
    for (id, bar) in model["CBARs"]
        if !haskey(bar, "GA"); continue; end
        push!(bars, Dict("eid" => _solver_entry_public_id(id, bar), "pid" => get(bar, "PID", 0), "ga" => bar["GA"], "gb" => bar["GB"]))
    end
    for (id, bar) in get(model, "CBEAMs", Dict())
        if !haskey(bar, "GA"); continue; end
        push!(bars, Dict("eid" => _solver_entry_public_id(id, bar), "pid" => get(bar, "PID", 0), "ga" => bar["GA"], "gb" => bar["GB"]))
    end

    rods = []
    for (id, rod) in model["CRODs"]
        if !haskey(rod, "GA"); continue; end
        push!(rods, Dict("eid" => _solver_entry_public_id(id, rod), "pid" => get(rod, "PID", 0), "ga" => rod["GA"], "gb" => rod["GB"]))
    end
    for (id, rod) in get(model, "CONRODs", Dict())
        if !haskey(rod, "GA"); continue; end
        push!(rods, Dict("eid" => _solver_entry_public_id(id, rod), "ga" => rod["GA"], "gb" => rod["GB"]))
    end

    solids = []
    for (id, el) in get(model, "CSOLIDs", Dict())
        if !haskey(el, "NODES"); continue; end
        nids = el["NODES"]
        if !all(n -> haskey(id_map, n), nids); continue; end
        push!(solids, Dict("eid" => _solver_entry_public_id(id, el), "pid" => get(el, "PID", 0), "nodes" => nids, "type" => get(el, "TYPE", "")))
    end

    return Dict("nodes" => nodes, "quads" => quads, "trias" => trias, "bars" => bars, "rods" => rods, "solids" => solids)
end

# ============================================================================
# Helper: convert mode shapes matrix to per-node list format
# ============================================================================
function _mode_shapes_to_list(mode_shapes, id_map)
    if size(mode_shapes, 2) == 0; return []; end
    sorted_nodes = sort(collect(keys(id_map)))
    n_nodes = length(sorted_nodes)
    n_modes = size(mode_shapes, 2)
    modes = Vector{Any}(undef, n_modes)
    for m in 1:n_modes
        mode_data = Vector{Any}(undef, n_nodes)
        for (node_pos, nid) in enumerate(sorted_nodes)
            idx = id_map[nid]; base = (idx-1)*6
            mode_data[node_pos] = Dict(
                "grid_id" => nid,
                "t1" => mode_shapes[base+1, m], "t2" => mode_shapes[base+2, m], "t3" => mode_shapes[base+3, m],
                "r1" => mode_shapes[base+4, m], "r2" => mode_shapes[base+5, m], "r3" => mode_shapes[base+6, m],
            )
        end
        modes[m] = mode_data
    end
    return modes
end

# ============================================================================
# Helper: print SOL 103 results to console
# ============================================================================
function _print_sol103_results(eigenvalues, frequencies, total_mass, wtmass, modal_effective_mass)
    println("\n>>> ============================================")
    println(">>> MASS SUMMARY")
    println(">>> ============================================")
    @printf("    Total Mass (X): %20.6e\n", total_mass[1])
    @printf("    Total Mass (Y): %20.6e\n", total_mass[2])
    @printf("    Total Mass (Z): %20.6e\n", total_mass[3])
    if wtmass != 1.0
        @printf("    WTMASS applied:  %g\n", wtmass)
        @printf("    Weight (X):     %20.6e\n", total_mass[1] / wtmass)
    end
    println(">>> ============================================")

    println("\n>>> ============================================")
    println(">>> SOL 103 NORMAL MODES RESULTS")
    println(">>> ============================================")
    if !isempty(frequencies)
        @printf("    %-6s  %-20s  %-20s\n", "Mode", "Frequency (Hz)", "Eigenvalue (rad²/s²)")
        @printf("    %-6s  %-20s  %-20s\n", "----", "--------------", "--------------------")
        for (i, (f, lam)) in enumerate(zip(frequencies, eigenvalues))
            @printf("    %-6d  %20.6f  %20.6e\n", i, f, lam)
        end

        println("\n>>> MODAL EFFECTIVE MASS")
        @printf("    %-6s  %-12s  %-14s  %-14s  %-14s\n", "Mode", "Freq (Hz)", "Meff-X", "Meff-Y", "Meff-Z")
        @printf("    %-6s  %-12s  %-14s  %-14s  %-14s\n", "----", "---------", "------", "------", "------")
        sum_meff = [0.0, 0.0, 0.0]
        for m in modal_effective_mass
            sum_meff[1] += m["meff_x"]; sum_meff[2] += m["meff_y"]; sum_meff[3] += m["meff_z"]
            @printf("    %-6d  %12.4f  %14.6e  %14.6e  %14.6e\n", m["mode"], m["freq"], m["meff_x"], m["meff_y"], m["meff_z"])
        end
        @printf("    %-6s  %-12s  %14.6e  %14.6e  %14.6e\n", "SUM", "", sum_meff[1], sum_meff[2], sum_meff[3])
        pct = [total_mass[d] > 0 ? sum_meff[d]/total_mass[d]*100 : 0.0 for d in 1:3]
        @printf("    %-6s  %-12s  %13.1f%%  %13.1f%%  %13.1f%%\n", "%Tot", "", pct[1], pct[2], pct[3])
    else
        println("    No valid normal modes found.")
    end
    println(">>> ============================================")
end

function _ensure_full_solver_extensions!()
    if isdefined(@__MODULE__, :Solver) && isdefined(Solver, :load_full_solver_extensions!)
        Solver.load_full_solver_extensions!()
    end
    return nothing
end

function _ensure_export_extensions!()
    if !isdefined(@__MODULE__, :build_jfem_element_tables)
        isdefined(@__MODULE__, :WriteVTK) || (@eval using WriteVTK)
        isdefined(@__MODULE__, :HDF5) || (@eval using HDF5)
        Base.include(@__MODULE__, joinpath(@__DIR__, "Export.jl"))
    end
    if !isdefined(@__MODULE__, :export_markdown_report)
        isdefined(@__MODULE__, :Printf) || (@eval using Printf)
        Base.include(@__MODULE__, joinpath(@__DIR__, "MarkdownReport.jl"))
    end
    return nothing
end

# ============================================================================
# Adjoint Sensitivity Analysis (optional post-solve step)
# ============================================================================

"""
    solve_adjoint(results::Dict, adjoint_config_path::String) -> Dict

Run adjoint sensitivity analysis on SOL 101 results.
Reads response/design-variable definitions from `adjoint_config_path` (JSON),
solves the adjoint equation for each response, and returns a Dict of sensitivities.

Call this after solve_model() for SOL 101:

    results = solve_model(model)
    adj = solve_adjoint(results, "adjoint_config.json")
    export_adjoint_json(adj, "output/model.ADJOINT.JSON")
"""
function solve_adjoint(results::Dict, adjoint_config_path::String)
    _ensure_full_solver_extensions!()
    return Base.invokelatest(Solver.solve_adjoint, results, adjoint_config_path)
end

"""
    solve_adjoint_buckling(results::Dict, adjoint_config_path::String) -> Dict

Run adjoint sensitivity analysis on SOL 105 buckling results.
Computes dλ/dx for each eigenvalue and design variable.
"""
function solve_adjoint_buckling(results::Dict, adjoint_config_path::String)
    _ensure_full_solver_extensions!()
    return Base.invokelatest(Solver.solve_adjoint_buckling, results, adjoint_config_path)
end

"""
    optimize_thickness(model::Dict; objective=:min_compliance, vol_frac=0.5, ...) -> Dict

Run per-element thickness optimization on the model.
Supports `:min_compliance` (SOL 101) and `:max_buckling` (SOL 105).
See `Solver.optimize_thickness` for the full keyword argument list,
including JSON checkpointing, restart, structured iteration histories,
and optional thickness derivative backend selection.
"""
function optimize_thickness(model::Dict; kwargs...)
    _ensure_full_solver_extensions!()
    return Base.invokelatest(Solver.optimize_thickness, model, solve_model; kwargs...)
end

"""
    optimize_sizing(model::Dict; objective=:min_compliance, design_variables=[:shell_thickness], ...) -> Dict

Run generalized sizing optimization on the model.
Supports `:shell_thickness`, `:bar_area`, or both, with the same
checkpoint/restart/history workflow as `optimize_thickness`.
"""
function optimize_sizing(model::Dict; kwargs...)
    _ensure_full_solver_extensions!()
    return Base.invokelatest(Solver.optimize_sizing, model, solve_model; kwargs...)
end

"""
    export_adjoint_json(adjoint_results::Dict, output_path::String)

Write adjoint sensitivity results to a JSON file.
"""
function export_adjoint_json(adjoint_results::Dict, output_path::String)
    _ensure_full_solver_extensions!()
    Base.invokelatest(Solver.export_adjoint_json, adjoint_results, output_path)
end

# ============================================================================
# Stage 3: Results Dict → Files
# ============================================================================

"""
    export_results(results::Dict, filename::String, output_dir::String;
                   export_json::Bool=false,
                   export_vtk::Bool=false,
                   export_hdf5::Bool=false,
                   export_jfem_binary::Bool=true)

Export JFEM binary by default.
Set `export_json=true`, `export_vtk=true`, and/or `export_hdf5=true`
to request additional output formats.
This is the Stage 3 entry point — called by the application after solve_model().
"""
function _export_results_impl(results::Dict, filename::String, output_dir::String;
                              export_json::Bool=false,
                              export_vtk::Bool=false,
                              export_hdf5::Bool=false,
                              export_jfem_binary::Bool=true,
                              export_report::Bool=true,
                              timings=nothing)
    if !isdir(output_dir); mkpath(output_dir); end

    sol_type = results["sol_type"]
    if sol_type == 200
        if export_json
            export_optimization_json(filename, output_dir, results)
        end
        if export_hdf5
            getfield(@__MODULE__, :export_hdf5)(filename, output_dir, build_optimization_export_payload(results);
                suffix=".OPTIMIZATION.H5", label="OPTIMIZATION HDF5")
        end
        if export_report
            export_markdown_report(results, filename, output_dir; timings=timings)
        end
        if haskey(results, "forward_results")
            final_filename = occursin(".bdf", lowercase(filename)) ?
                replace(filename, r"(?i)\.bdf$" => "_OPT_FINAL.bdf") :
                filename * "_OPT_FINAL.bdf"
            export_results(results["forward_results"], final_filename, output_dir;
                export_json=export_json,
                export_vtk=export_vtk,
                export_hdf5=export_hdf5,
                export_jfem_binary=export_jfem_binary,
                export_report=export_report)
        end
        return
    end

    model = results["model"]
    id_map = results["id_map"]
    X = results["node_coords"]

    jfem_node_ids = Int[]
    jfem_quads = Any[]
    jfem_trias = Any[]
    jfem_bars = Any[]
    jfem_rods = Any[]
    jfem_tetras = Any[]
    jfem_hexas = Any[]
    jfem_pentas = Any[]
    jfem_celas = Any[]
    jfem_rbe2s = Any[]
    jfem_rbe3s = Any[]
    if export_jfem_binary
        jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods, jfem_tetras, jfem_hexas, jfem_pentas =
            build_jfem_element_tables(model, id_map)
        jfem_celas, jfem_rbe2s, jfem_rbe3s = build_jfem_constraint_tables(model, id_map)
    end

    if sol_type == 101 || sol_type == 106
        _export_sol101(results, filename, output_dir, model, id_map, X,
            jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods,
            jfem_tetras, jfem_hexas, jfem_pentas,
            jfem_celas, jfem_rbe2s, jfem_rbe3s;
            export_json=export_json,
            export_vtk=export_vtk,
            export_hdf5=export_hdf5,
            export_jfem_binary=export_jfem_binary)
    elseif sol_type == 103
        mode_shapes = results["_raw_mode_shapes"]
        frequencies = results["frequencies"]
        eigenvalues = results["eigenvalues"]
        if export_vtk
            export_buckling_vtk(filename, output_dir, model, id_map, X, frequencies, mode_shapes)
        end
        if export_json
            export_buckling_json(filename, output_dir, eigenvalues, mode_shapes, id_map;
                frequencies=frequencies,
                mass_summary=get(results, "mass_summary", nothing),
                modal_effective_mass=get(results, "modal_effective_mass", nothing),
                buckling_subcases=get(results, "subcases", nothing),
                analysis_type="SOL103_MODES", diagnostics=get(results, "solver_diagnostics", nothing))
        end
        if export_hdf5
            getfield(@__MODULE__, :export_hdf5)(filename, output_dir, build_modal_hdf5_payload(results, filename);
                suffix=".MODES.H5", label="MODAL HDF5")
        end
        if export_jfem_binary
            export_jfem_buckling(filename, output_dir, id_map, X,
                jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods,
                jfem_tetras, jfem_hexas, jfem_pentas,
                frequencies, mode_shapes;
                jfem_celas=jfem_celas, jfem_rbe2s=jfem_rbe2s, jfem_rbe3s=jfem_rbe3s, K_global=nothing)
        end
    elseif sol_type == 105
        mode_shapes = results["_raw_mode_shapes"]
        eigenvalues = results["eigenvalues"]
        mode_shapes_available = size(mode_shapes, 2) >= length(eigenvalues)
        if !mode_shapes_available && (export_vtk || export_hdf5 || export_jfem_binary)
            println(">>> Mode shapes are omitted for this SOL 105 run; skipping VTK/HDF5/.jfem mode-shape exports.")
        end
        if export_vtk && mode_shapes_available
            export_buckling_vtk(filename, output_dir, model, id_map, X, eigenvalues, mode_shapes)
        end
        if export_json
            export_buckling_json(filename, output_dir, eigenvalues, mode_shapes, id_map;
                analysis_type="SOL105_BUCKLING", diagnostics=get(results, "solver_diagnostics", nothing))
        end
        if export_hdf5 && mode_shapes_available
            getfield(@__MODULE__, :export_hdf5)(filename, output_dir, build_buckling_hdf5_payload(results, filename);
                suffix=".BUCKLING.H5", label="BUCKLING HDF5")
        end
        if export_jfem_binary && mode_shapes_available
            export_jfem_buckling(filename, output_dir, id_map, X,
                jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods,
                jfem_tetras, jfem_hexas, jfem_pentas,
                eigenvalues, mode_shapes;
                jfem_celas=jfem_celas, jfem_rbe2s=jfem_rbe2s, jfem_rbe3s=jfem_rbe3s,
                K_global=get(results, "K_eig", nothing), node_R=get(results, "node_R", nothing))
        end
    end

    if export_report
        export_markdown_report(results, filename, output_dir; timings=timings)
    end
end

function export_results(results::Dict, filename::String, output_dir::String;
                        export_json::Bool=false,
                        export_vtk::Bool=false,
                        export_hdf5::Bool=false,
                        export_jfem_binary::Bool=true,
                        export_report::Bool=true,
                        timings=nothing)
    _ensure_export_extensions!()
    return Base.invokelatest(_export_results_impl, results, filename, output_dir;
        export_json=export_json,
        export_vtk=export_vtk,
        export_hdf5=export_hdf5,
        export_jfem_binary=export_jfem_binary,
        export_report=export_report,
        timings=timings)
end

function _export_sol101(results, filename, output_dir, model, id_map, X,
                        jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods,
                        jfem_tetras, jfem_hexas, jfem_pentas,
                        jfem_celas, jfem_rbe2s, jfem_rbe3s;
                        export_json::Bool=false,
                        export_vtk::Bool=false,
                        export_hdf5::Bool=false,
                        export_jfem_binary::Bool=true)
    is_nonlinear = get(results, "sol_type", 101) == 106
    needs_aggregated_results = export_json || export_hdf5
    global_results =
        if needs_aggregated_results
            payload = Dict(
                "analysis_type" => is_nonlinear ? "SOL106_NONLINEAR_STATIC" : "SOL101_STATIC",
                "displacements" => [], "spc_forces" => [],
                "forces" => Dict("cbar"=>[], "quad4"=>[], "tria3"=>[], "crod"=>[], "conrod"=>[], "celas1"=>[]),
                "forces_bilin" => Dict("quad4"=>[], "tria3"=>[]),
                "stresses" => Dict("cbar"=>[], "quad4"=>[], "tria3"=>[], "crod"=>[], "conrod"=>[], "celas1"=>[], "ctetra"=>[], "chexa"=>[], "cpenta"=>[]),
                "strains" => Dict("cbar"=>[], "quad4"=>[], "tria3"=>[], "crod"=>[], "conrod"=>[], "celas1"=>[], "ctetra"=>[], "chexa"=>[], "cpenta"=>[]),
                "solver_diagnostics" => Any[],
            )
            if is_nonlinear
                payload["nonlinear_diagnostics"] = Any[]
                payload["nonlinear_solver_summary"] = get(results, "solver_diagnostics", Any[])
            end
            payload
        else
            nothing
        end
    jfem_subcases_data = []

    for sc in results["subcases"]
        sid = sc["sid"]
        u = sc["raw_displacement"]
        stresses = sc["element_vonmises"]
        cc = model["CASE_CONTROL"]
        sub_cc = haskey(cc["SUBCASES"], sid) ? cc["SUBCASES"][sid] : Dict()

        if needs_aggregated_results
            append_requested_subcase_results!(global_results, sc, sub_cc)
            push!(global_results["solver_diagnostics"], Dict("sid" => sid, "details" => get(sc, "solver_diagnostics", Dict{String,Any}())))
            if is_nonlinear
                push!(global_results["nonlinear_diagnostics"], Dict(
                    "sid" => sid,
                    "details" => get(sc, "nonlinear_diagnostics", Dict{String,Any}()),
                ))
            end
        end

        if export_jfem_binary
            # Build JFEM subcase data only when binary export is requested.
            spc_id = get(sub_cc, "SPC", nothing)
            load_id = get(sub_cc, "LOAD", nothing)

            sc_data = collect_jfem_subcase_data(u, sc, id_map, jfem_node_ids,
                jfem_quads, jfem_trias, jfem_bars, jfem_rods,
                jfem_tetras, jfem_hexas, jfem_pentas;
                model=model, spc_id=spc_id, load_id=load_id)
            push!(jfem_subcases_data, (sid=sid, disp=sc_data.disp, shell=sc_data.shell,
                bar=sc_data.bar, rod=sc_data.rod, solid=sc_data.solid, spc=sc_data.spc,
                forces=sc_data.forces, moments=sc_data.moments))
        end

        if export_vtk
            export_vtk_subcase(filename, output_dir, sid, model, id_map, X, u, stresses)
        end
    end

    if export_json
        getfield(@__MODULE__, :export_json)(filename, output_dir, global_results)
    end
    if export_hdf5
        hdf5_payload = build_solution_hdf5_payload(results, filename, global_results)
        getfield(@__MODULE__, :export_hdf5)(filename, output_dir, hdf5_payload;
            suffix=".JU.H5", label="AGGREGATED HDF5")
    end
    if is_nonlinear
        if export_json
            export_nonlinear_json(filename, output_dir, results["subcases"];
                diagnostics=get(results, "solver_diagnostics", nothing))
        end
        if export_hdf5
            nonlinear_payload = build_nonlinear_export_payload(results["subcases"];
                diagnostics=get(results, "solver_diagnostics", nothing))
            getfield(@__MODULE__, :export_hdf5)(filename, output_dir, nonlinear_payload;
                suffix=".NONLINEAR.H5", label="NONLINEAR HDF5")
        end
    end
    if export_jfem_binary
        getfield(@__MODULE__, :export_jfem_binary)(filename, output_dir, id_map, X,
            jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods,
            jfem_tetras, jfem_hexas, jfem_pentas,
            jfem_subcases_data;
            jfem_celas=jfem_celas, jfem_rbe2s=jfem_rbe2s, jfem_rbe3s=jfem_rbe3s)
    end
end
