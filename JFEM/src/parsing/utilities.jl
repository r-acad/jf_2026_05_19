# utilities.jl — Low-level parsing helpers for NASTRAN BDF format

function safe_get(arr::Vector{Any}, idx::Int, default_val=nothing)
    if idx > length(arr); return default_val; end
    return arr[idx]
end

function parse_nastran_number(field::Any, default_val=nothing)
    if isa(field, Number); return field; end
    if isnothing(field) || strip(string(field)) == ""; return default_val; end

    field_str = String(strip(string(field)))
    clean_field = replace(field_str, r"([\d.])([+-])(\d)" => s"\1e\2\3")
    clean_field = replace(clean_field, "ee" => "e")

    try
        val = parse(Float64, clean_field)
        if abs(val) >= 0.5 && abs(val - round(val)) < 1e-8; return Int(round(val)); end
        return val
    catch; return default_val; end
end

function to_id(val)
    if isa(val, Integer); return val; end
    if isa(val, AbstractFloat); return Int(round(val)); end
    return 0
end

function expand_nastran_list(raw_fields)
    normalized_fields = Any[]
    for field in raw_fields
        if isa(field, AbstractString)
            st = strip(field)
            isempty(st) && continue
            if occursin(r"\s+", st)
                append!(normalized_fields, split(st))
            else
                push!(normalized_fields, st)
            end
        else
            push!(normalized_fields, field)
        end
    end

    result = Int[]
    i = 1
    while i <= length(normalized_fields)
        val = normalized_fields[i]
        if isa(val, AbstractString) && uppercase(strip(val)) == "THRU"
            if isempty(result) || i == length(normalized_fields)
                i += 1; continue
            end
            start_id = result[end]
            end_id = to_id(parse_nastran_number(normalized_fields[i+1]))
            if end_id > start_id; append!(result, (start_id+1):end_id); end
            i += 2
        else
            parsed = to_id(parse_nastran_number(val, 0))
            if parsed > 0; push!(result, parsed); end
            i += 1
        end
    end
    return result
end

function get_nastran_card_name(line::AbstractString)
    if occursin(",", line)
        parts = split(line, ",")
        return uppercase(strip(parts[1]))
    end
    if occursin('\t', line)
        parts = split(strip(line))
        isempty(parts) || return uppercase(strip(parts[1]))
    end
    return uppercase(strip(line[1:min(8, length(line))]))
end

function get_nastran_fixed_fields_from_line(line::AbstractString; large_field::Bool=false, truncate_to_line_length::Bool=false)
    fields = Any[]
    if large_field
        # Large-field format: 4 fields of 16 chars each at cols 9-24, 25-40, 41-56, 57-72
        padded = rpad(line, 80, ' ')
        limit_len = truncate_to_line_length ? ncodeunits(line) : ncodeunits(padded)
        for i in 0:3
            s = 9 + i*16
            e = s + 15
            s > limit_len && break
            push!(fields, strip(padded[s:min(e, length(padded))]))
        end
    else
        padded = rpad(line, 80, ' ')
        limit_len = truncate_to_line_length ? ncodeunits(line) : ncodeunits(padded)
        for i in 0:7
            s = 9 + i*8
            e = s + 7
            s > limit_len && break
            push!(fields, strip(padded[s:min(e, end)]))
        end
    end
    return fields
end

function get_nastran_fields_from_line(line::AbstractString; large_field::Bool=false)
    fields = []
    if occursin(",", line)
        parts = split(line, ",")
        head = parts[1]
        head_stripped = strip(head)
        is_continuation = startswith(head_stripped, "+") || startswith(head_stripped, "*")
        if is_continuation && ncodeunits(head) > 8
            head_payload = strip(String(head[9:end]))
            !isempty(head_payload) && push!(fields, head_payload)
        end
        if length(parts) > 1
            for p in parts[2:end]; push!(fields, strip(string(p))); end
        end
        # Pad free-field lines to 8 fields to maintain NASTRAN card field alignment
        while length(fields) < 8; push!(fields, ""); end
    elseif occursin('\t', line)
        # Mixed fixed-field + tab-delimited decks are common in production inputs.
        # Preserve empty tab fields so cards like PSHELL can intentionally skip slots.
        segments = split(line, '\t'; keepempty=true)
        prefix = rstrip(segments[1])
        if ncodeunits(prefix) > 8
            append!(fields, get_nastran_fixed_fields_from_line(prefix; large_field=large_field, truncate_to_line_length=true))
        end
        for seg in segments[2:end]
            push!(fields, strip(String(seg)))
        end
        while length(fields) < 8; push!(fields, ""); end
    else
        append!(fields, get_nastran_fixed_fields_from_line(line; large_field=large_field))
    end
    return fields
end
