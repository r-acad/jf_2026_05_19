# extract_constraints.jl — SPC1, SPC, SPCADD, RBE2, RBE3, MPC, MPCADD

function extract_spc1(cards)
    spcs = []
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        comp = string(parse_nastran_number(safe_get(c, 4), ""))
        raw_nodes = c[5:end]
        nodes = expand_nastran_list(raw_nodes)
        push!(spcs, Dict("SID"=>sid, "C"=>comp, "NODES"=>nodes))
    end
    return spcs
end

function extract_spcadd(cards)
    d = Dict()
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        raw_sets = c[4:end]
        sets = expand_nastran_list(raw_sets)
        d[sid] = sets
    end
    return d
end

function extract_spc(cards)
    spcs = []
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        gid = to_id(parse_nastran_number(safe_get(c, 4)))
        comp = string(parse_nastran_number(safe_get(c, 5), ""))
        d1 = Float64(parse_nastran_number(safe_get(c, 6), 0.0))
        if gid > 0
            push!(spcs, Dict("SID"=>sid, "C"=>comp, "NODES"=>[gid], "D"=>d1))
        end
        gid2 = to_id(parse_nastran_number(safe_get(c, 7), 0))
        if gid2 > 0
            comp2 = string(parse_nastran_number(safe_get(c, 8), ""))
            d2 = Float64(parse_nastran_number(safe_get(c, 9), 0.0))
            push!(spcs, Dict("SID"=>sid, "C"=>comp2, "NODES"=>[gid2], "D"=>d2))
        end
    end
    return spcs
end

function extract_rbe2(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        gn  = to_id(parse_nastran_number(safe_get(c, 4)))  # master node
        cm  = to_id(parse_nastran_number(safe_get(c, 5)))   # constrained components (e.g. 123456)
        # Remaining fields are slave nodes
        slave_grids = Int[]
        for k in 6:length(c)
            g = to_id(parse_nastran_number(safe_get(c, k), 0))
            if g > 0; push!(slave_grids, g); end
        end
        if eid > 0 && gn > 0
            d[string(eid)] = Dict("ID"=>eid, "GN"=>gn, "CM"=>cm, "GM"=>slave_grids)
        end
    end
    return d
end

function _is_continuation_marker(s::AbstractString)
    st = strip(s)
    isempty(st) && return false
    return st[1] == '+' || st[1] == '*'
end

function _is_nastran_real(s::AbstractString)
    # Detect if a field is a real number (weight) vs an integer (grid ID).
    # Nastran reals have '.', 'E', 'e', 'D', 'd', or Nastran-style +/- exponent.
    st = strip(s)
    isempty(st) && return false
    _is_continuation_marker(st) && return false
    return occursin('.', st) || occursin(r"[EeDd]", st) ||
           (length(st) > 1 && occursin(r"[\+\-]", st[2:end]))
end

function extract_rbe3(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        refgrid = to_id(parse_nastran_number(safe_get(c, 5)))
        refc = to_id(parse_nastran_number(safe_get(c, 6)))

        # Parse multiple weight groups: WT1,C1,G1...,Gn, WT2,C2,G1...,Gm, ...
        # Continuation markers (+XX, *XX) are skipped as they are parser artifacts.
        wt_groups = []
        pos = 7
        while pos <= length(c)
            wt_str = strip(safe_get(c, pos))
            if isempty(wt_str) || _is_continuation_marker(wt_str); pos += 1; continue; end
            wt_val = parse_nastran_number(wt_str, NaN)
            if isnan(wt_val); break; end
            pos += 1
            # Skip blanks/continuations before comps field
            while pos <= length(c)
                cs = strip(safe_get(c, pos))
                if !isempty(cs) && !_is_continuation_marker(cs); break; end
                pos += 1
            end
            if pos > length(c); break; end
            comps_val = to_id(parse_nastran_number(safe_get(c, pos)))
            pos += 1
            grids = Int[]
            while pos <= length(c)
                g_str = strip(safe_get(c, pos))
                if isempty(g_str); pos += 1; continue; end
                if _is_continuation_marker(g_str); pos += 1; continue; end
                # New weight group starts if field looks like a real number
                if _is_nastran_real(g_str); break; end
                gid = parse_nastran_number(g_str, NaN)
                if isnan(gid); break; end
                push!(grids, Int(gid))
                pos += 1
            end
            if !isempty(grids)
                push!(wt_groups, (wt=wt_val, comps=comps_val, grids=grids))
            end
        end

        d[string(eid)] = Dict("ID"=>eid, "REFGRID"=>refgrid, "REFC"=>refc,
                              "WT_GROUPS"=>wt_groups)
    end
    return d
end

function extract_rbar(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        ga  = to_id(parse_nastran_number(safe_get(c, 4)))
        gb  = to_id(parse_nastran_number(safe_get(c, 5)))
        cna = to_id(parse_nastran_number(safe_get(c, 6), 0))
        cnb = to_id(parse_nastran_number(safe_get(c, 7), 0))
        cma = to_id(parse_nastran_number(safe_get(c, 8), 0))
        cmb = to_id(parse_nastran_number(safe_get(c, 9), 0))
        if eid > 0 && ga > 0 && gb > 0
            d[string(eid)] = Dict("ID"=>eid, "GA"=>ga, "GB"=>gb, "CNA"=>cna, "CNB"=>cnb, "CMA"=>cma, "CMB"=>cmb)
        end
    end
    return d
end

function extract_rbe1(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        # Find "UM" marker — separates independent from dependent DOFs
        um_idx = 0
        for k in 3:length(c)
            s = strip(uppercase(string(safe_get(c, k))))
            if s == "UM"
                um_idx = k
                break
            end
        end
        if eid <= 0 || um_idx == 0; continue; end

        # Parse independent DOFs: pairs of (grid, component) before UM
        indep = Tuple{Int,Int}[]
        k = 4
        while k + 1 <= length(c) && k < um_idx
            g = to_id(parse_nastran_number(safe_get(c, k), 0))
            cn = to_id(parse_nastran_number(safe_get(c, k+1), 0))
            if g > 0 && cn > 0
                # Expand component digits (e.g. 123 → [1,2,3])
                for ch in string(cn)
                    if isdigit(ch)
                        push!(indep, (g, parse(Int, string(ch))))
                    end
                end
            end
            k += 2
        end

        # Parse dependent DOFs: pairs of (grid, component) after UM
        dep = Tuple{Int,Int}[]
        k = um_idx + 1
        while k + 1 <= length(c)
            g = to_id(parse_nastran_number(safe_get(c, k), 0))
            cn = to_id(parse_nastran_number(safe_get(c, k+1), 0))
            if g > 0 && cn > 0
                for ch in string(cn)
                    if isdigit(ch)
                        push!(dep, (g, parse(Int, string(ch))))
                    end
                end
            end
            k += 2
        end

        if !isempty(indep) && !isempty(dep)
            d[string(eid)] = Dict("ID"=>eid, "INDEP"=>indep, "DEP"=>dep)
        end
    end
    return d
end

function extract_rspline(cards)
    d = Dict()
    for c in cards
        eid = to_id(parse_nastran_number(safe_get(c, 3)))
        # Field 4: D/L ratio (not used for stiffness, only for constraint weights)
        dl = Float64(parse_nastran_number(safe_get(c, 4), 0.1))
        # Field 5: first independent grid
        g1 = to_id(parse_nastran_number(safe_get(c, 5), 0))
        if eid <= 0 || g1 <= 0; continue; end

        # Parse remaining fields: ordered sequence of grids
        # Independent grids have no component after them; dependent grids do
        indep_grids = [g1]
        dep_list = Tuple{Int,Int}[]  # (grid, component)
        k = 6
        while k <= length(c)
            val = parse_nastran_number(safe_get(c, k), 0)
            g = to_id(val)
            if g <= 0; k += 1; continue; end

            # Check next field: if it's a small number (1-6 or combo like 123456), this grid is dependent
            if k + 1 <= length(c)
                next_val = parse_nastran_number(safe_get(c, k+1), -1)
                next_int = to_id(next_val)
                # Component numbers: single digits 1-6, or combos like 12, 123, 123456
                # Grid IDs: typically > 1000
                if next_int > 0 && next_int <= 123456 && all(ch -> ch in "0123456", string(next_int))
                    # This is a dependent grid with component
                    for ch in string(next_int)
                        if isdigit(ch) && ch != '0'
                            push!(dep_list, (g, parse(Int, string(ch))))
                        end
                    end
                    k += 2
                    continue
                end
            end
            # No component follows → this is an independent grid
            push!(indep_grids, g)
            k += 1
        end

        if !isempty(dep_list) && length(indep_grids) >= 2
            d[string(eid)] = Dict("ID"=>eid, "DL"=>dl, "INDEP_GRIDS"=>indep_grids, "DEP"=>dep_list)
        end
    end
    return d
end

function extract_mpc(cards)
    mpcs = []
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        terms = []
        k = 4
        while k + 2 <= length(c)
            gid   = to_id(parse_nastran_number(safe_get(c, k), 0))
            comp  = to_id(parse_nastran_number(safe_get(c, k+1), 0))
            coeff = Float64(parse_nastran_number(safe_get(c, k+2), 0.0))
            if gid > 0 && comp > 0
                push!(terms, Dict("G"=>gid, "C"=>comp, "A"=>coeff))
            end
            k += 3
        end
        if length(terms) >= 2
            push!(mpcs, Dict("SID"=>sid, "TERMS"=>terms))
        end
    end
    return mpcs
end

function extract_mpcadd(cards)
    d = Dict()
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        subs = Int[]
        for k in 4:length(c)
            s = to_id(parse_nastran_number(safe_get(c, k), 0))
            if s > 0; push!(subs, s); end
        end
        if sid > 0 && !isempty(subs)
            d[sid] = subs
        end
    end
    return d
end
