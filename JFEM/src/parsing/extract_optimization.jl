# extract_optimization.jl - SOL 200-lite optimization card extraction

@inline function _opt_card_token(field; make_uppercase::Bool=false)
    if isnothing(field)
        return nothing
    end
    raw = strip(string(field))
    isempty(raw) && return nothing

    parsed = parse_nastran_number(raw, nothing)
    if !isnothing(parsed)
        return parsed
    end

    return make_uppercase ? uppercase(raw) : raw
end

@inline function _opt_float_or_nothing(field)
    parsed = parse_nastran_number(field, nothing)
    return isnothing(parsed) ? nothing : Float64(parsed)
end

function extract_desvar(cards)
    d = Dict{String, Any}()
    for c in cards
        id = to_id(parse_nastran_number(safe_get(c, 3), 0))
        id <= 0 && continue

        label = strip(string(safe_get(c, 4, "")))
        xinit = Float64(parse_nastran_number(safe_get(c, 5), 0.0))
        xlb = _opt_float_or_nothing(safe_get(c, 6))
        xub = _opt_float_or_nothing(safe_get(c, 7))
        delx = _opt_float_or_nothing(safe_get(c, 8))
        ddval = parse_nastran_number(safe_get(c, 9), nothing)

        d[string(id)] = Dict(
            "ID" => id,
            "LABEL" => label,
            "XINIT" => xinit,
            "XLB" => xlb,
            "XUB" => xub,
            "DELX" => delx,
            "DDVAL" => isnothing(ddval) ? nothing : to_id(ddval),
        )
    end
    return d
end

function extract_dresp1(cards)
    d = Dict{String, Any}()
    for c in cards
        id = to_id(parse_nastran_number(safe_get(c, 3), 0))
        id <= 0 && continue

        label = strip(string(safe_get(c, 4, "")))
        rtype = uppercase(strip(string(safe_get(c, 5, ""))))
        ptype = _opt_card_token(safe_get(c, 6); make_uppercase=true)
        region = _opt_card_token(safe_get(c, 7))
        atta = _opt_card_token(safe_get(c, 8); make_uppercase=true)
        attb = _opt_card_token(safe_get(c, 9); make_uppercase=true)
        atti = Any[]
        for k in 10:length(c)
            token = _opt_card_token(safe_get(c, k); make_uppercase=true)
            isnothing(token) || push!(atti, token)
        end

        d[string(id)] = Dict(
            "ID" => id,
            "LABEL" => label,
            "RTYPE" => rtype,
            "PTYPE" => ptype,
            "REGION" => region,
            "ATTA" => atta,
            "ATTB" => attb,
            "ATTI" => atti,
        )
    end
    return d
end

function _extract_relation1(cards, target_key::String)
    d = Dict{String, Any}()
    for c in cards
        id = to_id(parse_nastran_number(safe_get(c, 3), 0))
        id <= 0 && continue

        rel_type = uppercase(strip(string(safe_get(c, 4, ""))))
        target_id = to_id(parse_nastran_number(safe_get(c, 5), 0))
        target_id <= 0 && continue

        field_ref = _opt_card_token(safe_get(c, 6); make_uppercase=true)
        lower_bound = _opt_float_or_nothing(safe_get(c, 7))
        upper_bound = _opt_float_or_nothing(safe_get(c, 8))
        c0 = Float64(parse_nastran_number(safe_get(c, 9), 0.0))

        coeffs = Any[]
        k = 10
        while k <= length(c)
            desvar_id = to_id(parse_nastran_number(safe_get(c, k), 0))
            coef = _opt_float_or_nothing(safe_get(c, k + 1))
            if desvar_id > 0
                push!(coeffs, Dict(
                    "DESVAR_ID" => desvar_id,
                    "COEF" => isnothing(coef) ? 1.0 : coef,
                ))
            end
            k += 2
        end

        entry = Dict{String, Any}(
            "ID" => id,
            "TYPE" => rel_type,
            target_key => target_id,
            "PNAME_FID" => field_ref,
            "LOWER_BOUND" => lower_bound,
            "UPPER_BOUND" => upper_bound,
            "C0" => c0,
            "COEFFICIENTS" => coeffs,
        )
        d[string(id)] = entry
    end
    return d
end

function extract_dvprel1(cards)
    return _extract_relation1(cards, "PID")
end

function extract_dvmrel1(cards)
    return _extract_relation1(cards, "MID")
end

function extract_dconstr(cards)
    d = Dict{String, Any}()
    for c in cards
        id = to_id(parse_nastran_number(safe_get(c, 3), 0))
        id <= 0 && continue

        response_id = to_id(parse_nastran_number(safe_get(c, 4), 0))
        response_id <= 0 && continue

        lower_allowable = _opt_float_or_nothing(safe_get(c, 5))
        upper_allowable = _opt_float_or_nothing(safe_get(c, 6))
        lowfq = _opt_float_or_nothing(safe_get(c, 7))
        highfq = _opt_float_or_nothing(safe_get(c, 8))

        d[string(id)] = Dict(
            "ID" => id,
            "RID" => response_id,
            "LOWER_ALLOWABLE" => lower_allowable,
            "UPPER_ALLOWABLE" => upper_allowable,
            "LOWFQ" => lowfq,
            "HIGHFQ" => highfq,
        )
    end
    return d
end

function extract_doptprm(cards)
    params = Dict{String, Any}()
    for c in cards
        k = 3
        while k <= length(c)
            name_raw = strip(string(safe_get(c, k, "")))
            if !isempty(name_raw)
                name = uppercase(name_raw)
                value = _opt_card_token(safe_get(c, k + 1); make_uppercase=true)
                params[name] = value
            end
            k += 2
        end
    end
    return params
end
