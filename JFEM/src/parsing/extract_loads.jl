# extract_loads.jl — FORCE, MOMENT, PLOAD4, PLOAD2, PLOAD1, GRAV, LOAD combos

function extract_loads(cards)
    f = []
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        gid = to_id(parse_nastran_number(safe_get(c, 4)))
        cid = to_id(parse_nastran_number(safe_get(c, 5), 0))
        mag = parse_nastran_number(safe_get(c, 6), 0.0)
        dir = [parse_nastran_number(safe_get(c, 7),0.0), parse_nastran_number(safe_get(c, 8),0.0), parse_nastran_number(safe_get(c, 9),0.0)]
        push!(f, Dict("TYPE"=>"FORCE", "SID"=>sid, "GID"=>gid, "CID"=>cid, "Mag"=>mag, "Dir"=>dir))
    end
    return f
end

function extract_moments(cards)
    m = []
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        gid = to_id(parse_nastran_number(safe_get(c, 4)))
        cid = to_id(parse_nastran_number(safe_get(c, 5), 0))
        mag = parse_nastran_number(safe_get(c, 6), 0.0)
        dir = [parse_nastran_number(safe_get(c, 7),0.0), parse_nastran_number(safe_get(c, 8),0.0), parse_nastran_number(safe_get(c, 9),0.0)]
        push!(m, Dict("TYPE"=>"MOMENT", "SID"=>sid, "GID"=>gid, "CID"=>cid, "Mag"=>mag, "Dir"=>dir))
    end
    return m
end

function extract_pload4(cards)
    p = []
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        eid = to_id(parse_nastran_number(safe_get(c, 4)))
        press = parse_nastran_number(safe_get(c, 5), 0.0)
        # G1, G3 for solid face identification (fields 8, 9)
        g1 = to_id(parse_nastran_number(safe_get(c, 8), 0))
        g3 = to_id(parse_nastran_number(safe_get(c, 9), 0))
        # Continuation line: CID(field 10→c[11]), N1(c[12]), N2(c[13]), N3(c[14])
        cid = to_id(parse_nastran_number(safe_get(c, 11), 0))
        n1 = parse_nastran_number(safe_get(c, 12), nothing)
        n2 = parse_nastran_number(safe_get(c, 13), nothing)
        n3 = parse_nastran_number(safe_get(c, 14), nothing)
        d = Dict{String,Any}("TYPE"=>"PLOAD4", "SID"=>sid, "EID"=>eid, "P"=>press)
        if g1 > 0; d["G1"] = g1; end
        if g3 > 0; d["G3"] = g3; end
        if n1 !== nothing && n2 !== nothing && n3 !== nothing
            d["CID"] = cid
            d["N"] = [Float64(n1), Float64(n2), Float64(n3)]
        end
        push!(p, d)
    end
    return p
end

function extract_pload2(cards)
    p = []
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        press = parse_nastran_number(safe_get(c, 4), 0.0)
        for k in 5:length(c)
            eid = to_id(parse_nastran_number(safe_get(c, k), 0))
            if eid > 0
                push!(p, Dict("TYPE"=>"PLOAD4", "SID"=>sid, "EID"=>eid, "P"=>press))
            end
        end
    end
    return p
end

function extract_pload(cards)
    p = []
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        press = parse_nastran_number(safe_get(c, 4), 0.0)
        g1 = to_id(parse_nastran_number(safe_get(c, 5), 0))
        g2 = to_id(parse_nastran_number(safe_get(c, 6), 0))
        g3 = to_id(parse_nastran_number(safe_get(c, 7), 0))
        g4 = to_id(parse_nastran_number(safe_get(c, 8), 0))
        nodes = filter(x -> x > 0, [g1, g2, g3, g4])
        if length(nodes) >= 3
            push!(p, Dict("TYPE"=>"PLOAD", "SID"=>sid, "P"=>press, "NODES"=>nodes))
        end
    end
    return p
end

function extract_pload1(cards)
    p = []
    for c in cards
        sid   = to_id(parse_nastran_number(safe_get(c, 3)))
        eid   = to_id(parse_nastran_number(safe_get(c, 4)))
        # TYPE field: can be integer (1-6) or string ("FX","FY","FZ","MX","MY","MZ")
        ltype_raw = strip(string(safe_get(c, 5, "0")))
        ltype_map = Dict("FX"=>1,"FY"=>2,"FZ"=>3,"MX"=>4,"MY"=>5,"MZ"=>6)
        ltype = get(ltype_map, uppercase(ltype_raw), to_id(parse_nastran_number(ltype_raw, 0)))
        scale_str = strip(string(safe_get(c, 6, "")))
        x1    = Float64(parse_nastran_number(safe_get(c, 7), 0.0))
        p1    = Float64(parse_nastran_number(safe_get(c, 8), 0.0))
        x2    = Float64(parse_nastran_number(safe_get(c, 9), 1.0))
        p2    = Float64(parse_nastran_number(safe_get(c, 10), 0.0))
        if sid > 0 && eid > 0
            push!(p, Dict("TYPE"=>"PLOAD1", "SID"=>sid, "EID"=>eid,
                          "LOAD_TYPE"=>ltype, "SCALE"=>scale_str,
                          "X1"=>x1, "P1"=>p1, "X2"=>x2, "P2"=>p2))
        end
    end
    return p
end

function extract_grav(cards)
    g = []
    for c in cards
        sid   = to_id(parse_nastran_number(safe_get(c, 3)))
        cid   = to_id(parse_nastran_number(safe_get(c, 4), 0))
        accel = Float64(parse_nastran_number(safe_get(c, 5), 0.0))
        n1    = Float64(parse_nastran_number(safe_get(c, 6), 0.0))
        n2    = Float64(parse_nastran_number(safe_get(c, 7), 0.0))
        n3    = Float64(parse_nastran_number(safe_get(c, 8), 0.0))
        if sid > 0
            push!(g, Dict("TYPE"=>"GRAV", "SID"=>sid, "CID"=>cid,
                           "A"=>accel, "N"=>[n1, n2, n3]))
        end
    end
    return g
end

function extract_rforce(cards)
    r = []
    for c in cards
        sid    = to_id(parse_nastran_number(safe_get(c, 3)))
        g_node = to_id(parse_nastran_number(safe_get(c, 4), 0))  # rotation center grid (0=origin)
        cid    = to_id(parse_nastran_number(safe_get(c, 5), 0))
        A_val  = Float64(parse_nastran_number(safe_get(c, 6), 0.0))  # angular velocity scale
        r1     = Float64(parse_nastran_number(safe_get(c, 7), 0.0))
        r2     = Float64(parse_nastran_number(safe_get(c, 8), 0.0))
        r3     = Float64(parse_nastran_number(safe_get(c, 9), 0.0))
        method = to_id(parse_nastran_number(safe_get(c, 10), 2))   # 1 or 2
        if sid > 0
            push!(r, Dict("TYPE"=>"RFORCE", "SID"=>sid, "G"=>g_node, "CID"=>cid,
                           "A"=>A_val, "R"=>[r1, r2, r3], "METHOD"=>method))
        end
    end
    return r
end

function extract_load_combos(cards)
    combos = []
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        s = parse_nastran_number(safe_get(c, 4), 1.0)
        comps = []
        for i in 5:2:length(c)-1
            s_i = parse_nastran_number(safe_get(c, i), nothing)
            l_i = to_id(parse_nastran_number(safe_get(c, i+1), nothing))
            if !isnothing(s_i) && l_i > 0
                push!(comps, Dict("S"=>s_i, "LID"=>l_i))
            end
        end
        push!(combos, Dict("SID"=>sid, "S"=>s, "COMPS"=>comps))
    end
    return combos
end

function extract_temp(cards)
    temps = Dict{Int, Dict{Int,Float64}}()  # SID => {GID => T}
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        if sid <= 0; continue; end
        if !haskey(temps, sid); temps[sid] = Dict{Int,Float64}(); end
        # Read every grid-temperature pair carried by the flattened card,
        # including continuation fields beyond the first three pairs.
        for i in 4:2:length(c)-1
            gid = to_id(parse_nastran_number(safe_get(c, i), 0))
            t_val = parse_nastran_number(safe_get(c, i + 1), nothing)
            if gid > 0 && t_val !== nothing
                temps[sid][gid] = Float64(t_val)
            end
        end
    end
    return temps
end

function extract_tempd(cards)
    tempd = Dict{Int, Float64}()  # SID => default temperature
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        t_val = parse_nastran_number(safe_get(c, 4), 0.0)
        if sid > 0
            tempd[sid] = Float64(t_val)
        end
    end
    return tempd
end

"""
Parse DMIG cards into sparse matrix entries.
Returns Dict{String, Dict}: name => {"type"=>symmetric/square, "entries"=>[(row_grid, row_dof, col_grid, col_dof, value), ...]}
"""
function extract_dmig(cards)
    matrices = Dict{String, Dict{String,Any}}()

    for c in cards
        name = strip(string(safe_get(c, 3, "")))
        if isempty(name); continue; end

        # Detect header vs data: header has IFO in field 4 (small integer 0-9)
        # and TIN in field 5. Data has GJ (grid ID) in field 4 and CJ (1-6) in field 5.
        f4 = parse_nastran_number(safe_get(c, 4), nothing)
        f5 = parse_nastran_number(safe_get(c, 5), nothing)
        f6 = parse_nastran_number(safe_get(c, 6), nothing)

        if f4 !== nothing && f5 !== nothing && f6 !== nothing &&
           Float64(f4) >= 0.0 && Float64(f4) <= 9.0 &&
           Float64(f5) >= 1.0 && Float64(f5) <= 9.0 &&
           !haskey(matrices, name)
            # Header card: DMIG NAME IFO TIN TOUT
            ifo = to_id(f4); tin = to_id(f5); tout = to_id(f6)
            is_sym = (ifo == 6 || tout == 2)
            matrices[name] = Dict{String,Any}("type" => is_sym ? "symmetric" : "square",
                                               "entries" => Tuple{Int,Int,Int,Int,Float64}[])
        elseif haskey(matrices, name) && f4 !== nothing && f5 !== nothing
            # Data card: DMIG NAME GJ CJ [blank] G1 C1 A1 [G2 C2 A2 ...]
            gj = to_id(f4)  # column grid
            cj = to_id(f5)  # column DOF (1-6)
            if gj <= 0 || cj <= 0 || cj > 6; continue; end

            entries = matrices[name]["entries"]
            # Parse row entries starting at field 7 (skipping blank field 6)
            k = 7
            while k + 2 <= length(c)
                # Skip continuation markers
                val_str = strip(string(safe_get(c, k, "")))
                if !isempty(val_str) && (startswith(val_str, "+") || startswith(val_str, "*"))
                    k += 1; continue
                end
                gi = to_id(parse_nastran_number(safe_get(c, k), 0))
                ci = to_id(parse_nastran_number(safe_get(c, k+1), 0))
                ai = parse_nastran_number(safe_get(c, k+2), nothing)
                if gi > 0 && ci >= 1 && ci <= 6 && ai !== nothing
                    push!(entries, (gi, ci, gj, cj, Float64(ai)))
                end
                k += 3
            end
        elseif !haskey(matrices, name) && f4 !== nothing
            # Data card appearing before header — create a default entry
            matrices[name] = Dict{String,Any}("type" => "square",
                                               "entries" => Tuple{Int,Int,Int,Int,Float64}[])
            # Re-parse this card as data
            gj = to_id(f4); cj = f5 !== nothing ? to_id(f5) : 0
            if gj > 0 && cj >= 1 && cj <= 6
                entries = matrices[name]["entries"]
                k = 7
                while k + 2 <= length(c)
                    val_str = strip(string(safe_get(c, k, "")))
                    if !isempty(val_str) && (startswith(val_str, "+") || startswith(val_str, "*"))
                        k += 1; continue
                    end
                    gi = to_id(parse_nastran_number(safe_get(c, k), 0))
                    ci = to_id(parse_nastran_number(safe_get(c, k+1), 0))
                    ai = parse_nastran_number(safe_get(c, k+2), nothing)
                    if gi > 0 && ci >= 1 && ci <= 6 && ai !== nothing
                        push!(entries, (gi, ci, gj, cj, Float64(ai)))
                    end
                    k += 3
                end
            end
        end
    end
    return matrices
end
