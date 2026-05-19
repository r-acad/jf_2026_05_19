# extract_elements.jl — CROD, CELAS1, CONROD, CONM2, CTETRA, CHEXA, CPENTA

function extract_crod(cards)
    d = Dict()
    for c in cards
        id  = to_id(parse_nastran_number(safe_get(c, 3)))
        pid = to_id(parse_nastran_number(safe_get(c, 4)))
        ga  = to_id(parse_nastran_number(safe_get(c, 5)))
        gb  = to_id(parse_nastran_number(safe_get(c, 6)))
        if id > 0
            d[string(id)] = Dict("ID"=>id, "PID"=>pid, "GA"=>ga, "GB"=>gb, "TYPE"=>"CROD")
        end
    end
    return d
end

function extract_celas1(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        pid = to_id(parse_nastran_number(safe_get(c, 4)))
        g1  = to_id(parse_nastran_number(safe_get(c, 5), 0))
        c1  = to_id(parse_nastran_number(safe_get(c, 6), 0))
        g2  = to_id(parse_nastran_number(safe_get(c, 7), 0))
        c2  = to_id(parse_nastran_number(safe_get(c, 8), 0))
        if eid > 0
            d[string(eid)] = Dict("ID"=>eid, "PID"=>pid,
                                  "G1"=>g1, "C1"=>c1, "G2"=>g2, "C2"=>c2, "TYPE"=>"CELAS1")
        end
    end
    return d
end

function extract_celas2(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        K   = Float64(parse_nastran_number(safe_get(c, 4), 0.0))
        g1  = to_id(parse_nastran_number(safe_get(c, 5), 0))
        c1  = to_id(parse_nastran_number(safe_get(c, 6), 0))
        g2  = to_id(parse_nastran_number(safe_get(c, 7), 0))
        c2  = to_id(parse_nastran_number(safe_get(c, 8), 0))
        if eid > 0
            d[string(eid)] = Dict("ID"=>eid, "K"=>K,
                                  "G1"=>g1, "C1"=>c1, "G2"=>g2, "C2"=>c2, "TYPE"=>"CELAS2")
        end
    end
    return d
end

function extract_cbush(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        pid = to_id(parse_nastran_number(safe_get(c, 4)))
        ga  = to_id(parse_nastran_number(safe_get(c, 5), 0))
        gb  = to_id(parse_nastran_number(safe_get(c, 6), 0))
        # Fields 7-9: GO or X1,X2,X3 (orientation); field 10: CID
        cid = to_id(parse_nastran_number(safe_get(c, 10), 0))
        if eid > 0 && ga > 0
            d[string(eid)] = Dict("ID"=>eid, "PID"=>pid,
                "GA"=>ga, "GB"=>gb, "CID"=>cid, "TYPE"=>"CBUSH")
        end
    end
    return d
end

function extract_conrod(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        g1  = to_id(parse_nastran_number(safe_get(c, 4)))
        g2  = to_id(parse_nastran_number(safe_get(c, 5)))
        mid = to_id(parse_nastran_number(safe_get(c, 6)))
        A   = Float64(parse_nastran_number(safe_get(c, 7), 0.0))
        J   = Float64(parse_nastran_number(safe_get(c, 8), 0.0))
        C_  = Float64(parse_nastran_number(safe_get(c, 9), 0.0))
        NSM = Float64(parse_nastran_number(safe_get(c, 10), 0.0))
        if eid > 0
            d[string(eid)] = Dict("ID"=>eid, "GA"=>g1, "GB"=>g2,
                "MID"=>mid, "A"=>A, "J"=>J, "C"=>C_, "NSM"=>NSM, "TYPE"=>"CONROD")
        end
    end
    return d
end

function extract_conm2(cards)
    d = Dict()
    for c in cards
        eid  = to_id(parse_nastran_number(safe_get(c, 3)))
        gid  = to_id(parse_nastran_number(safe_get(c, 4)))
        cid  = to_id(parse_nastran_number(safe_get(c, 5), 0))
        mass = Float64(parse_nastran_number(safe_get(c, 6), 0.0))
        x1   = Float64(parse_nastran_number(safe_get(c, 7), 0.0))
        x2   = Float64(parse_nastran_number(safe_get(c, 8), 0.0))
        x3   = Float64(parse_nastran_number(safe_get(c, 9), 0.0))
        I11  = Float64(parse_nastran_number(safe_get(c, 11), 0.0))
        I21  = Float64(parse_nastran_number(safe_get(c, 12), 0.0))
        I22  = Float64(parse_nastran_number(safe_get(c, 13), 0.0))
        I31  = Float64(parse_nastran_number(safe_get(c, 14), 0.0))
        I32  = Float64(parse_nastran_number(safe_get(c, 15), 0.0))
        I33  = Float64(parse_nastran_number(safe_get(c, 16), 0.0))
        if eid > 0
            d[string(eid)] = Dict("ID"=>eid, "GID"=>gid, "CID"=>cid,
                "M"=>mass, "X"=>[x1, x2, x3],
                "I"=>[I11, I21, I22, I31, I32, I33], "TYPE"=>"CONM2")
        end
    end
    return d
end

function extract_conm1(cards)
    d = Dict()
    for c in cards
        eid  = to_id(parse_nastran_number(safe_get(c, 3)))
        gid  = to_id(parse_nastran_number(safe_get(c, 4)))
        cid  = to_id(parse_nastran_number(safe_get(c, 5), 0))
        # 6x6 symmetric mass matrix (21 upper-triangle terms)
        m11 = Float64(parse_nastran_number(safe_get(c, 6), 0.0))
        m21 = Float64(parse_nastran_number(safe_get(c, 7), 0.0))
        m22 = Float64(parse_nastran_number(safe_get(c, 8), 0.0))
        m31 = Float64(parse_nastran_number(safe_get(c, 9), 0.0))
        m32 = Float64(parse_nastran_number(safe_get(c, 10), 0.0))
        m33 = Float64(parse_nastran_number(safe_get(c, 11), 0.0))
        m41 = Float64(parse_nastran_number(safe_get(c, 12), 0.0))
        m42 = Float64(parse_nastran_number(safe_get(c, 13), 0.0))
        m43 = Float64(parse_nastran_number(safe_get(c, 14), 0.0))
        m44 = Float64(parse_nastran_number(safe_get(c, 15), 0.0))
        m51 = Float64(parse_nastran_number(safe_get(c, 16), 0.0))
        m52 = Float64(parse_nastran_number(safe_get(c, 17), 0.0))
        m53 = Float64(parse_nastran_number(safe_get(c, 18), 0.0))
        m54 = Float64(parse_nastran_number(safe_get(c, 19), 0.0))
        m55 = Float64(parse_nastran_number(safe_get(c, 20), 0.0))
        m61 = Float64(parse_nastran_number(safe_get(c, 21), 0.0))
        m62 = Float64(parse_nastran_number(safe_get(c, 22), 0.0))
        m63 = Float64(parse_nastran_number(safe_get(c, 23), 0.0))
        m64 = Float64(parse_nastran_number(safe_get(c, 24), 0.0))
        m65 = Float64(parse_nastran_number(safe_get(c, 25), 0.0))
        m66 = Float64(parse_nastran_number(safe_get(c, 26), 0.0))
        if eid > 0
            d[string(eid)] = Dict("ID"=>eid, "GID"=>gid, "CID"=>cid,
                "M"=>m11, "TYPE"=>"CONM1",
                "M_DIAG"=>[m11, m22, m33, m44, m55, m66])
        end
    end
    return d
end

function extract_cmass1(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        pid = to_id(parse_nastran_number(safe_get(c, 4)))
        g1  = to_id(parse_nastran_number(safe_get(c, 5), 0))
        c1  = to_id(parse_nastran_number(safe_get(c, 6), 0))
        g2  = to_id(parse_nastran_number(safe_get(c, 7), 0))
        c2  = to_id(parse_nastran_number(safe_get(c, 8), 0))
        if eid > 0
            d[string(eid)] = Dict("ID"=>eid, "PID"=>pid,
                "G1"=>g1, "C1"=>c1, "G2"=>g2, "C2"=>c2, "TYPE"=>"CMASS1")
        end
    end
    return d
end

function extract_cmass2(cards)
    d = Dict()
    for c in cards
        eid  = to_id(parse_nastran_number(safe_get(c, 3)))
        mass = Float64(parse_nastran_number(safe_get(c, 4), 0.0))
        g1   = to_id(parse_nastran_number(safe_get(c, 5), 0))
        c1   = to_id(parse_nastran_number(safe_get(c, 6), 0))
        g2   = to_id(parse_nastran_number(safe_get(c, 7), 0))
        c2   = to_id(parse_nastran_number(safe_get(c, 8), 0))
        if eid > 0
            d[string(eid)] = Dict("ID"=>eid, "M"=>mass,
                "G1"=>g1, "C1"=>c1, "G2"=>g2, "C2"=>c2, "TYPE"=>"CMASS2")
        end
    end
    return d
end

# --- Solid elements ---

# Helper: collect N node IDs from card starting at field `start`, skipping continuation markers
function _collect_node_ids(c, start::Int, count::Int)
    nodes = Int[]
    k = start
    while length(nodes) < count && k <= length(c)
        val = strip(string(safe_get(c, k, "")))
        if !isempty(val) && !startswith(val, "+") && !startswith(val, "*")
            nid = to_id(parse_nastran_number(safe_get(c, k)))
            push!(nodes, nid)
        end
        k += 1
    end
    return nodes
end

function extract_ctetra(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        pid = to_id(parse_nastran_number(safe_get(c, 4)))
        nodes = _collect_node_ids(c, 5, 4)
        if eid > 0 && length(nodes) == 4 && all(n -> n > 0, nodes)
            d[string(eid)] = Dict("ID"=>eid, "PID"=>pid,
                "NODES"=>nodes, "TYPE"=>"CTETRA")
        end
    end
    return d
end

function extract_chexa(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        pid = to_id(parse_nastran_number(safe_get(c, 4)))
        nodes = _collect_node_ids(c, 5, 8)
        if eid > 0 && length(nodes) == 8 && all(n -> n > 0, nodes)
            d[string(eid)] = Dict("ID"=>eid, "PID"=>pid,
                "NODES"=>nodes, "TYPE"=>"CHEXA")
        end
    end
    return d
end

function extract_cpenta(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        pid = to_id(parse_nastran_number(safe_get(c, 4)))
        nodes = _collect_node_ids(c, 5, 6)
        if eid > 0 && length(nodes) == 6 && all(n -> n > 0, nodes)
            d[string(eid)] = Dict("ID"=>eid, "PID"=>pid,
                "NODES"=>nodes, "TYPE"=>"CPENTA")
        end
    end
    return d
end
