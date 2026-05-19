# extract_materials.jl — MAT1, MAT2, MAT8, MATT1, TABLEM1

function extract_mats(cards)
    d = Dict()
    for c in cards
        mid = to_id(parse_nastran_number(safe_get(c, 3)))
        E_raw = parse_nastran_number(safe_get(c, 4), nothing)
        G_raw = parse_nastran_number(safe_get(c, 5), nothing)
        nu_raw = parse_nastran_number(safe_get(c, 6), nothing)
        E = isnothing(E_raw) ? 0.0 : Float64(E_raw)
        G = isnothing(G_raw) ? 0.0 : Float64(G_raw)
        nu = isnothing(nu_raw) ? -1.0 : Float64(nu_raw)
        # Compute missing property from the other two (MAT1: at least 2 of E, G, NU must be given)
        if E > 0 && G > 0 && nu < 0
            nu = E / (2*G) - 1.0
        elseif E > 0 && nu >= 0 && G <= 0
            G = E / (2*(1+nu))
        elseif G > 0 && nu >= 0 && E <= 0
            E = 2*G*(1+nu)
        end
        if nu < 0; nu = 0.3; end
        RHO = Float64(parse_nastran_number(safe_get(c, 7), 0.0))
        ALPHA = Float64(parse_nastran_number(safe_get(c, 8), 0.0))
        TREF = Float64(parse_nastran_number(safe_get(c, 9), 0.0))
        d[string(mid)] = Dict("MID"=>mid, "E"=>E, "G"=>G, "NU"=>nu, "RHO"=>RHO, "ALPHA"=>ALPHA, "TREF"=>TREF)
    end
    return d
end

function extract_mat2(cards)
    d = Dict()
    for c in cards
        mid = to_id(parse_nastran_number(safe_get(c, 3)))
        G11 = Float64(parse_nastran_number(safe_get(c, 4), 0.0))
        G12 = Float64(parse_nastran_number(safe_get(c, 5), 0.0))
        G13 = Float64(parse_nastran_number(safe_get(c, 6), 0.0))
        G22 = Float64(parse_nastran_number(safe_get(c, 7), 0.0))
        G23 = Float64(parse_nastran_number(safe_get(c, 8), 0.0))
        G33 = Float64(parse_nastran_number(safe_get(c, 9), 0.0))
        RHO = Float64(parse_nastran_number(safe_get(c, 10), 0.0))
        if mid > 0
            # Estimate equivalent E, G, NU for compatibility
            nu_eq = G11 > 0 ? clamp(G12 / G11, 0.0, 0.49) : 0.3
            E_eq  = G11 > 0 ? G11 * (1 - nu_eq^2) : 0.0
            G_eq  = G33
            d[string(mid)] = Dict("MID"=>mid,
                "G11"=>G11, "G12"=>G12, "G13"=>G13,
                "G22"=>G22, "G23"=>G23, "G33"=>G33,
                "RHO"=>RHO,
                "E"=>E_eq, "G"=>G_eq, "NU"=>nu_eq, "TYPE"=>"MAT2")
        end
    end
    return d
end

function extract_mat8(cards)
    d = Dict()
    for c in cards
        mid  = to_id(parse_nastran_number(safe_get(c, 3)))
        E1   = Float64(parse_nastran_number(safe_get(c, 4), 0.0))
        E2   = Float64(parse_nastran_number(safe_get(c, 5), 0.0))
        NU12 = Float64(parse_nastran_number(safe_get(c, 6), 0.0))
        G12  = Float64(parse_nastran_number(safe_get(c, 7), 0.0))
        G1Z_raw = strip(String(safe_get(c, 8, "")))
        G2Z_raw = strip(String(safe_get(c, 9, "")))
        G1Z  = Float64(parse_nastran_number(G1Z_raw, 0.0))
        G2Z  = Float64(parse_nastran_number(G2Z_raw, 0.0))
        RHO  = Float64(parse_nastran_number(safe_get(c, 10), 0.0))
        if mid > 0
            d[string(mid)] = Dict("MID"=>mid, "E1"=>E1, "E2"=>E2, "NU12"=>NU12,
                "G12"=>G12, "G1Z"=>G1Z, "G2Z"=>G2Z, "RHO"=>RHO,
                "G1Z_BLANK"=>isempty(G1Z_raw), "G2Z_BLANK"=>isempty(G2Z_raw),
                # For compatibility with MAT1 interface
                "E"=>E1, "G"=>G12, "NU"=>NU12, "TYPE"=>"MAT8")
        end
    end
    return d
end

function extract_matt1(cards)
    d = Dict{String,Dict{String,Any}}()
    for c in cards
        mid = to_id(parse_nastran_number(safe_get(c, 3), 0))
        mid <= 0 && continue

        entry = Dict{String,Any}("MID" => mid)
        e_tid = to_id(parse_nastran_number(safe_get(c, 4), 0))
        g_tid = to_id(parse_nastran_number(safe_get(c, 5), 0))
        nu_tid = to_id(parse_nastran_number(safe_get(c, 6), 0))
        rho_tid = to_id(parse_nastran_number(safe_get(c, 7), 0))
        alpha_tid = to_id(parse_nastran_number(safe_get(c, 8), 0))

        e_tid > 0 && (entry["E_TABLE"] = e_tid)
        g_tid > 0 && (entry["G_TABLE"] = g_tid)
        nu_tid > 0 && (entry["NU_TABLE"] = nu_tid)
        rho_tid > 0 && (entry["RHO_TABLE"] = rho_tid)
        alpha_tid > 0 && (entry["ALPHA_TABLE"] = alpha_tid)

        d[string(mid)] = entry
    end
    return d
end

function extract_tablem1(cards)
    d = Dict{String,Dict{String,Any}}()
    for c in cards
        tid = to_id(parse_nastran_number(safe_get(c, 3), 0))
        tid <= 0 && continue

        points = Dict{String,Float64}[]
        numeric_tokens = Float64[]
        for k in 4:length(c)
            token = uppercase(strip(string(safe_get(c, k, ""))))
            token == "ENDT" && break
            value = parse_nastran_number(safe_get(c, k), nothing)
            value isa Number && push!(numeric_tokens, Float64(value))
        end
        for k in 1:2:length(numeric_tokens)-1
            push!(points, Dict("X" => numeric_tokens[k], "Y" => numeric_tokens[k + 1]))
        end

        d[string(tid)] = Dict{String,Any}(
            "TID" => tid,
            "POINTS" => points,
        )
    end
    return d
end
