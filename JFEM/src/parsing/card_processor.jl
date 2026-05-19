# card_processor.jl — Card processing, INCLUDE resolution, and case control / bulk data splitting

"""
    resolve_includes(lines, base_dir; depth=0)

Recursively resolve INCLUDE cards in a Nastran BDF file.
Replaces each INCLUDE line with the contents of the referenced file.
Supports quoted filenames, relative and absolute paths, and nested includes (max 10 levels).
"""
function resolve_includes(lines::Vector{String}, base_dir::String; depth::Int=0)
    if depth > 10
        println("[WARN] INCLUDE nesting depth > 10, stopping recursion")
        return lines
    end
    result = String[]
    for line in lines
        stripped = strip(line)
        upper_stripped = uppercase(stripped)
        if startswith(upper_stripped, "INCLUDE")
            # Extract filename: INCLUDE 'filename' or INCLUDE "filename" or INCLUDE filename
            rest = strip(stripped[8:end])  # skip "INCLUDE"
            # Remove quotes if present
            if length(rest) >= 2 && ((rest[1] == '\'' && rest[end] == '\'') || (rest[1] == '"' && rest[end] == '"'))
                inc_file = rest[2:end-1]
            else
                inc_file = rest
            end
            inc_file = strip(inc_file)
            # Resolve path: if not absolute, resolve relative to base_dir
            if !isabspath(inc_file)
                inc_file = joinpath(base_dir, inc_file)
            end
            inc_file = normpath(inc_file)
            if isfile(inc_file)
                inc_lines = readlines(inc_file)
                inc_dir = dirname(inc_file)
                resolved = resolve_includes(inc_lines, inc_dir; depth=depth+1)
                append!(result, resolved)
                println("[INFO] INCLUDE resolved: $(basename(inc_file)) ($(length(resolved)) lines, depth=$depth)")
            else
                println("[WARN] INCLUDE file not found: $inc_file")
                push!(result, line)  # keep original line
            end
        else
            push!(result, line)
        end
    end
    return result
end

function process_cards(lines)
    processed = Dict{String, Vector{Any}}()
    i = 1
    while i <= length(lines)
        line = lines[i]
        clean_line = strip(line)
        if startswith(clean_line, '$') || isempty(clean_line)
            i += 1; continue
        end

        name = get_nastran_card_name(line)
        if startswith(name, "+") || name == "*"
             i += 1; continue
        end

        # Detect large-field format (card name ends with *)
        is_large = endswith(name, "*")
        base_name = strip(is_large ? name[1:end-1] : name)

        if !isnothing(base_name) && !isempty(base_name)
            fields = Any["SMALL"; String(base_name)]
            append!(fields, get_nastran_fields_from_line(line; large_field=is_large))

            steps = 1
            while i + steps <= length(lines)
                next_line = lines[i+steps]
                next_clean = strip(next_line)
                if startswith(next_clean, '$')
                    steps += 1; continue
                end

                is_cont = false
                is_cont_large = false
                if startswith(next_clean, "*")
                    is_cont = true
                    is_cont_large = true
                elseif startswith(next_line, " ") || startswith(next_clean, "+")
                    is_cont = true
                elseif occursin(",", next_line) && startswith(next_clean, ",")
                    is_cont = true
                end

                if is_cont
                    append!(fields, get_nastran_fields_from_line(next_line; large_field=(is_large || is_cont_large)))
                    steps += 1
                else
                    break
                end
            end

            if !haskey(processed, base_name); processed[base_name] = []; end
            push!(processed[base_name], fields)
            i += steps
        else
            i += 1
        end
    end
    return processed
end

function read_bulk_and_case(lines::Vector{String})
    case_control = Dict{String, Any}("SUBCASES" => Dict{Int, Dict{String, Any}}())
    bulk_lines = String[]
    in_bulk = false
    past_cend = false
    global_load, global_spc, global_mpc = nothing, nothing, nothing
    current_sub = 0

    # Check if BEGIN BULK exists anywhere in the file
    has_begin_bulk = any(occursin("BEGIN BULK", uppercase(l)) for l in lines)

    # Default SOL type
    case_control["SOL"] = 101

    function _store_case_control_entry!(target::Dict{String, Any}, key_raw::AbstractString, value)
        key_clean = strip(key_raw)
        modifier = nothing
        modifier_match = match(r"^([^(]+)\(([^)]*)\)$", key_clean)
        if modifier_match !== nothing
            key_clean = strip(modifier_match.captures[1])
            modifier = strip(modifier_match.captures[2])
        end

        target[key_clean] = value
        if !isnothing(modifier) && !isempty(modifier)
            target["$(key_clean)_MODIFIER"] = uppercase(modifier)
        end
        return key_clean
    end

    for line in lines
        cl = uppercase(split(line, '$')[1])
        if occursin("BEGIN BULK", cl); in_bulk=true; continue; end
        # If no BEGIN BULK in file, treat everything after CEND as bulk
        if !has_begin_bulk && past_cend && !in_bulk
            in_bulk = true
        end
        if occursin("ENDDATA", cl); break; end

        if !in_bulk
            stripped_cl = strip(cl)

            # Parse SOL type (executive control)
            if startswith(stripped_cl, "SOL ")
                m = match(r"SOL\s+(\d+)", stripped_cl)
                if m !== nothing
                    case_control["SOL"] = parse(Int, m.captures[1])
                end
                continue
            end

            # Skip MYSTRAN-specific case control keywords
            if startswith(stripped_cl, "ELDATA"); continue; end
            if startswith(stripped_cl, "LABEL"); continue; end
            if startswith(stripped_cl, "ECHO"); continue; end
            if startswith(stripped_cl, "SET "); continue; end
            if startswith(stripped_cl, "GPFORCE"); continue; end
            if startswith(stripped_cl, "MPCFORCE"); continue; end
            if startswith(stripped_cl, "OLOAD"); continue; end
            if startswith(stripped_cl, "TITLE"); continue; end
            if startswith(stripped_cl, "SUBTI"); continue; end
            if startswith(stripped_cl, "CEND"); past_cend = true; continue; end

            if startswith(stripped_cl, "SUBCASE")
                parts = split(cl)
                if length(parts) >= 2
                    val = try parse(Int, parts[2]) catch; 0 end
                    if val > 0
                        current_sub = val
                        case_control["SUBCASES"][current_sub] = Dict{String, Any}("LOAD"=>global_load, "SPC"=>global_spc, "MPC"=>global_mpc)
                    end
                end
            elseif occursin("=", cl)
                # Parse key = value, handling parenthetical modifiers like DISP(PRINT,PLOT) = ALL
                eq_parts = split(cl, "="; limit=2)
                k_raw = strip(eq_parts[1])
                v_raw = strip(eq_parts[2])
                val = try parse(Int, v_raw) catch; v_raw end
                if current_sub > 0
                    _store_case_control_entry!(case_control["SUBCASES"][current_sub], k_raw, val)
                else
                    k = _store_case_control_entry!(case_control, k_raw, val)
                    if k == "LOAD"; global_load = val; end
                    if k == "SPC"; global_spc = val; end
                    if k == "MPC"; global_mpc = val; end
                end
            end
        else
            if length(strip(cl)) > 1
                push!(bulk_lines, String(rstrip(cl)))
            end
        end
    end

    # Create default subcase if none defined but global load/spc exists
    if isempty(case_control["SUBCASES"]) && (!isnothing(global_load) || !isnothing(global_spc))
        case_control["SUBCASES"][1] = Dict{String, Any}("LOAD"=>global_load, "SPC"=>global_spc, "MPC"=>global_mpc)
        println("[INFO] No SUBCASE defined. Created default SUBCASE 1 with LOAD=$global_load, SPC=$global_spc")
    end

    return case_control, bulk_lines
end
