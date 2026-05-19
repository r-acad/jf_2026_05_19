# compare_mode_mac.jl
#
# Compare JFEM buckling mode shapes against MSC Nastran eigenvectors using
# Modal Assurance Criterion (MAC). JFEM modes are read from the binary .jfem
# export; subcase grouping is recovered from the companion .REPORT.md file.
#
# Usage:
#   julia --project=. tools/testing/compare_mode_mac.jl <case.jfem> <nastran.pch|nastran.f06> [--dofs=123] [--subcase=511002] [--subspace] [--window=0.05] [--csv=mode_mac.csv]
#
# OP2-only SOL105 vector output can be converted first with:
#   python tools/testing/op2_eigenvectors_to_pch.py <nastran.op2> <nastran_vectors.pch>
#
# Notes:
# - Default MAC uses translational DOFs 1,2,3. Rotations are often less
#   portable across element frames and output-coordinate conventions.
# - Many production .f06 files contain only eigenvalues and strain energy.
#   Feed this tool an .f06 with printed buckling eigenvectors, a
#   .pch containing $EIGENVECTOR blocks, or a converted OP2 eigenvector file.
#   Static $DISPLACEMENTS punch blocks are ignored.
# - --subspace reports how well each Nastran vector projects into the JFEM
#   modal subspace. This is useful for clustered or repeated roots where a
#   physically matching subspace can have poor one-to-one MAC.

using LinearAlgebra
using Printf

const DEFAULT_DOFS = [1, 2, 3]

function ascii_upper(s::AbstractString)
    bytes = UInt8[]
    sizehint!(bytes, ncodeunits(s))
    for b in codeunits(s)
        push!(bytes, b < 0x80 ? b : UInt8(' '))
    end
    return uppercase(String(bytes))
end

struct JFEMModes
    node_ids::Vector{Int}
    eigenvalues::Vector{Float64}
    modes::Matrix{Float64}  # nnode*6 by nmode, node-major order
end

mutable struct NastranMode
    subcase::String
    mode::Int
    eigenvalue::Union{Nothing,Float64}
    values::Dict{Int,NTuple{6,Float64}}
end

function parse_float_token(tok::AbstractString)
    s = replace(strip(tok), 'D' => 'E', 'd' => 'E')
    val = tryparse(Float64, s)
    val !== nothing && return val
    m = match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)$", s)
    m === nothing && return nothing
    return tryparse(Float64, string(m.captures[1], "E", m.captures[2]))
end

function read_jfem_buckling(path::AbstractString)
    open(path, "r") do io
        magic = String(read(io, 4))
        magic == "JFEM" || error("Not a JFEM binary: $path")
        version = read(io, UInt32)
        version == 3 || error("Expected buckling .jfem version 3, got version $version")
        n_nodes = Int(read(io, UInt32))
        n_quads = Int(read(io, UInt32))
        n_trias = Int(read(io, UInt32))
        n_bars = Int(read(io, UInt32))
        n_rods = Int(read(io, UInt32))
        n_modes = Int(read(io, UInt32))
        n_celas = Int(read(io, UInt32))
        n_rbe2 = Int(read(io, UInt32))
        n_rbe3 = Int(read(io, UInt32))

        node_ids = Vector{Int}(undef, n_nodes)
        for i in 1:n_nodes
            node_ids[i] = Int(read(io, Int32))
            read(io, Float32); read(io, Float32); read(io, Float32)
        end

        for _ in 1:n_quads
            for _ in 1:6; read(io, Int32); end
            read(io, Float32)
        end
        for _ in 1:n_trias
            for _ in 1:5; read(io, Int32); end
            read(io, Float32)
        end
        for _ in 1:n_bars
            for _ in 1:4; read(io, Int32); end
            read(io, Float32)
        end
        for _ in 1:n_rods
            for _ in 1:4; read(io, Int32); end
            read(io, Float32)
        end
        for _ in 1:n_celas
            for _ in 1:5; read(io, Int32); end
            read(io, Float32); read(io, Float32)
        end
        for _ in 1:n_rbe2
            read(io, Int32); read(io, Int32); read(io, Int32)
            n = Int(read(io, UInt32))
            for _ in 1:n; read(io, Int32); end
        end
        for _ in 1:n_rbe3
            read(io, Int32); read(io, Int32); read(io, Int32)
            n = Int(read(io, UInt32))
            for _ in 1:n; read(io, Int32); end
        end

        modes = zeros(Float64, n_nodes * 6, n_modes)
        for m in 1:n_modes
            read(io, UInt32)  # stored as sequential mode/subcase id in v3 buckling export
            for i in 1:n_nodes, d in 1:6
                modes[(i - 1) * 6 + d, m] = Float64(read(io, Float32))
            end
            for _ in 1:(n_quads + n_trias), __ in 1:7; read(io, Float32); end
            for _ in 1:n_bars, __ in 1:7; read(io, Float32); end
            for _ in 1:n_rods, __ in 1:2; read(io, Float32); end
            read(io, UInt32); read(io, UInt32); read(io, UInt32)
        end

        marker = String(read(io, 4))
        marker == "EVAL" || error("Missing EVAL footer in $path")
        n_eval = Int(read(io, UInt32))
        eigenvalues = [read(io, Float64) for _ in 1:n_eval]
        n_eval == n_modes || error("EVAL count $n_eval does not match mode count $n_modes")
        return JFEMModes(node_ids, eigenvalues, modes)
    end
end

function companion_report_path(jfem_path::AbstractString)
    base = replace(jfem_path, r"\.jfem$"i => "")
    return base * ".REPORT.md"
end

function parse_report_eigenvalues(md_path::AbstractString)
    per_subcase = Pair{String,Vector{Float64}}[]
    isfile(md_path) || return per_subcase
    current = ""
    vals = Float64[]
    in_table = false
    function flush!()
        if !isempty(current)
            push!(per_subcase, current => copy(vals))
        end
    end
    for line in eachline(md_path)
        m = match(r"### Buckling subcase\s+(\S+)", line)
        if m !== nothing
            flush!()
            current = String(m.captures[1])
            empty!(vals)
            in_table = false
            continue
        end
        if startswith(line, "|---") && !isempty(current)
            in_table = true
            continue
        end
        if in_table && startswith(line, "|")
            parts = [strip(p) for p in split(line, "|") if !isempty(strip(p))]
            if length(parts) >= 2
                val = parse_float_token(parts[2])
                val !== nothing && push!(vals, val)
            end
        elseif in_table && isempty(strip(line))
            in_table = false
        end
    end
    flush!()
    return per_subcase
end

function split_jfem_by_report(jfem::JFEMModes, report_path::AbstractString)
    groups = Dict{String,Tuple{Vector{Float64},Matrix{Float64}}}()
    report = parse_report_eigenvalues(report_path)
    if isempty(report)
        groups["ALL"] = (jfem.eigenvalues, jfem.modes)
        return groups
    end
    col0 = 1
    for (subcase, vals) in report
        n = length(vals)
        if col0 + n - 1 > size(jfem.modes, 2)
            @warn "Report has more modes than .jfem; truncating subcase" subcase
            n = max(size(jfem.modes, 2) - col0 + 1, 0)
        end
        n <= 0 && continue
        groups[subcase] = (jfem.eigenvalues[col0:col0+n-1], jfem.modes[:, col0:col0+n-1])
        col0 += n
    end
    return groups
end

function ensure_mode!(modes::Vector{NastranMode}, current_sc::String,
                      current_mode::Int, current_eval::Union{Nothing,Float64})
    idx = findfirst(m -> m.subcase == current_sc && m.mode == current_mode, modes)
    if idx === nothing
        push!(modes, NastranMode(current_sc, current_mode, current_eval, Dict{Int,NTuple{6,Float64}}()))
        return modes[end]
    end
    if modes[idx].eigenvalue === nothing && current_eval !== nothing
        modes[idx].eigenvalue = current_eval
    end
    return modes[idx]
end

function parse_nastran_punch_or_f06(path::AbstractString)
    modes = NastranMode[]
    current_sc = "ALL"
    current_mode = 0
    current_eval::Union{Nothing,Float64} = nothing
    active_mode::Union{Nothing,NastranMode} = nothing
    current_point_id::Union{Nothing,Int} = nothing
    pending_pch::Union{Nothing,Tuple{NastranMode,Int,Vector{Float64}}} = nothing
    in_f06_disp_table = false
    in_punch_vector = false

    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue

        if startswith(line, "\$")
            upper = ascii_upper(line)
            pending_pch = nothing
            if occursin("EIGENVECTOR", upper)
                in_punch_vector = true
            elseif occursin("DISPLACEMENT", upper) || occursin("ELEMENT", upper) ||
                   occursin("STRAIN", upper) || occursin("FORCE", upper)
                in_punch_vector = false
            end
        end

        m_sc = match(r"SUBCASE(?:\s+ID)?\s*=?\s*(\d+)", line)
        if m_sc !== nothing
            next_sc = String(m_sc.captures[1])
            if next_sc != current_sc
                current_sc = next_sc
                current_mode = 0
                current_eval = nothing
                active_mode = nothing
                pending_pch = nothing
            end
        end
        m_eval = match(r"EIGENVALUE\s*=?\s*([+-]?\S+)", line)
        if m_eval !== nothing
            current_eval = parse_float_token(m_eval.captures[1])
            active_mode = nothing
            pending_pch = nothing
        end
        m_mode = match(r"(?:EIGENVECTOR|MODE)\s*=?\s*(\d+)", line)
        if m_mode !== nothing && !occursin("EIGENVALUES", line)
            current_mode = something(tryparse(Int, m_mode.captures[1]), current_mode)
            active_mode = ensure_mode!(modes, current_sc, current_mode, current_eval)
        end
        upper_line = ascii_upper(line)
        compact = replace(upper_line, r"\s+" => "")
        if occursin("REALEIGENVECTORNO.", compact)
            m_no = match(r"REALEIGENVECTORNO\.(\d+)", compact)
            if m_no !== nothing
                current_mode = something(tryparse(Int, m_no.captures[1]), current_mode)
                active_mode = ensure_mode!(modes, current_sc, current_mode, current_eval)
            end
        end
        m_point = match(r"POINT\s+ID\s*=?\s*(\d+)", line)
        if m_point !== nothing
            current_point_id = tryparse(Int, m_point.captures[1])
        end

        if occursin("POINT ID", line) && occursin("TYPE", line)
            in_f06_disp_table = true
            current_mode == 0 && (current_mode = length(modes) + 1)
            active_mode = ensure_mode!(modes, current_sc, current_mode, current_eval)
            continue
        end
        if startswith(line, "\$") || startswith(line, "***")
            startswith(line, "***") && (in_f06_disp_table = false)
            continue
        end
        (in_punch_vector || in_f06_disp_table) || continue

        if startswith(ascii_upper(line), "-CONT-")
            pending_pch === nothing && continue
            mode, nid, first_vals = pending_pch
            toks = split(replace(line, ',' => ' '))
            cont_vals = Float64[]
            for tok in toks[2:end]
                v = parse_float_token(tok)
                v === nothing && continue
                push!(cont_vals, v)
                length(cont_vals) == 3 && break
            end
            length(cont_vals) == 3 || continue
            vals = vcat(first_vals, cont_vals)
            mode.values[nid] = (vals[1], vals[2], vals[3], vals[4], vals[5], vals[6])
            pending_pch = nothing
            continue
        end

        toks = split(replace(line, ',' => ' '))
        length(toks) >= 5 || continue
        nid = tryparse(Int, toks[1])
        nid === nothing && continue
        has_grid_type = tryparse(Float64, toks[2]) === nothing
        if in_f06_disp_table && !has_grid_type
            continue
        end

        if current_point_id !== nothing && tryparse(Float64, toks[2]) === nothing && in_punch_vector
            # SORT2 punch stores the grid in "$POINT ID" and the row key in column 1.
            nid = current_point_id
        end

        offset = has_grid_type ? 3 : 2
        length(toks) >= offset + 2 || continue
        vals = Float64[]
        ok = true
        for k in offset:length(toks)
            v = parse_float_token(toks[k])
            if v === nothing
                continue
            end
            push!(vals, v)
            length(vals) == 6 && break
        end
        ok || continue
        if active_mode === nothing
            if !in_f06_disp_table && current_mode == 0
                current_mode = 1
            end
            active_mode = ensure_mode!(modes, current_sc, current_mode, current_eval)
        end
        if length(vals) >= 6
            active_mode.values[Int(nid)] = (vals[1], vals[2], vals[3], vals[4], vals[5], vals[6])
        elseif in_punch_vector && length(vals) >= 3
            pending_pch = (active_mode, Int(nid), vals[1:3])
        end
    end

    return [m for m in modes if !isempty(m.values) && m.eigenvalue !== nothing]
end

function jfem_vector(modes::Matrix{Float64}, node_index::Dict{Int,Int},
                     nids::Vector{Int}, mode_col::Int, dofs::Vector{Int})
    v = Vector{Float64}(undef, length(nids) * length(dofs))
    k = 1
    for nid in nids
        idx = node_index[nid]
        base = (idx - 1) * 6
        for d in dofs
            v[k] = modes[base + d, mode_col]
            k += 1
        end
    end
    return v
end

function nastran_vector(mode::NastranMode, nids::Vector{Int}, dofs::Vector{Int})
    v = Vector{Float64}(undef, length(nids) * length(dofs))
    k = 1
    for nid in nids
        vals = mode.values[nid]
        for d in dofs
            v[k] = vals[d]
            k += 1
        end
    end
    return v
end

function mac(a::AbstractVector, b::AbstractVector)
    aa = dot(a, a)
    bb = dot(b, b)
    (aa <= 0.0 || bb <= 0.0) && return 0.0
    return (dot(a, b)^2) / (aa * bb)
end

function orthonormal_basis(cols::Matrix{Float64}; tol::Float64=1e-10)
    qs = Vector{Vector{Float64}}()
    for j in 1:size(cols, 2)
        v = copy(view(cols, :, j))
        # Modified Gram-Schmidt with one reorthogonalization pass is enough for
        # these small modal blocks and avoids depending on dense full-size QR.
        for _ in 1:2
            for q in qs
                v .-= dot(q, v) .* q
            end
        end
        nrm = norm(v)
        if nrm > tol
            push!(qs, v ./ nrm)
        end
    end
    return isempty(qs) ? zeros(Float64, size(cols, 1), 0) : reduce(hcat, qs)
end

function projection_mac(v::AbstractVector, Q::Matrix{Float64})
    vv = dot(v, v)
    (vv <= 0.0 || size(Q, 2) == 0) && return NaN
    vu = v ./ sqrt(vv)
    return sum(abs2, transpose(Q) * vu)
end

function subspace_overlap(Qa::Matrix{Float64}, Qb::Matrix{Float64})
    (size(Qa, 2) == 0 || size(Qb, 2) == 0) && return Float64[]
    return svdvals(transpose(Qa) * Qb).^2
end

function jfem_matrix(modes::Matrix{Float64}, node_index::Dict{Int,Int},
                     nids::Vector{Int}, mode_cols::Vector{Int}, dofs::Vector{Int})
    mat = Matrix{Float64}(undef, length(nids) * length(dofs), length(mode_cols))
    for (jout, jmode) in enumerate(mode_cols)
        mat[:, jout] = jfem_vector(modes, node_index, nids, jmode, dofs)
    end
    return mat
end

function nastran_matrix(n_modes::Vector{NastranMode}, nids::Vector{Int}, dofs::Vector{Int})
    sorted = sort(n_modes; by=m -> m.mode)
    mat = Matrix{Float64}(undef, length(nids) * length(dofs), length(sorted))
    for (j, mode) in enumerate(sorted)
        mat[:, j] = nastran_vector(mode, nids, dofs)
    end
    return mat, sorted
end

function rel_lambda(a::Float64, b::Float64)
    abs(a) < 1e-30 && return Inf
    return abs(b - a) / abs(a)
end

function candidate_modes_by_window(jfem_vals::Vector{Float64}, lam_n::Union{Nothing,Float64},
                                   window::Float64)
    lam_n === nothing && return Int[]
    return [j for j in eachindex(jfem_vals) if rel_lambda(lam_n, jfem_vals[j]) <= window]
end

function format_mode_list(js::Vector{Int})
    isempty(js) && return "-"
    if length(js) <= 8
        return join(js, ",")
    end
    return string(join(js[1:4], ","), ",...,", join(js[end-2:end], ","))
end

function format_projection(x::Float64)
    isnan(x) ? "NA" : @sprintf("%.4f", x)
end

function csv_cell(x)
    x === nothing && return ""
    if x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.12g", x) : ""
    end
    s = string(x)
    if any(ch -> ch == ',' || ch == '"' || ch == '\r' || ch == '\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_mac_csv(path::AbstractString, rows::Vector{Dict{String,Any}})
    headers = [
        "subcase",
        "nastran_mode",
        "nastran_lambda",
        "best_jfem_mode",
        "best_jfem_lambda",
        "best_rel_lambda",
        "best_mac",
        "projection_all_jfem",
        "window_rel_tol",
        "window_modes",
        "projection_window",
        "common_grids",
    ]
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(headers, ","))
        for row in rows
            println(io, join((csv_cell(get(row, h, "")) for h in headers), ","))
        end
    end
end

function print_subspace_diagnostics(jfem_vals::Vector{Float64}, jfem_modes::Matrix{Float64},
                                    node_index::Dict{Int,Int}, common::Vector{Int},
                                    n_modes::Vector{NastranMode}, dofs::Vector{Int},
                                    window::Float64)
    n_mat, n_sorted = nastran_matrix(n_modes, common, dofs)
    j_cols = collect(1:size(jfem_modes, 2))
    j_mat = jfem_matrix(jfem_modes, node_index, common, j_cols, dofs)

    Qn = orthonormal_basis(n_mat)
    Qj_all = orthonormal_basis(j_mat)
    all_proj = [projection_mac(view(n_mat, :, k), Qj_all) for k in 1:size(n_mat, 2)]
    overlap = subspace_overlap(Qn, Qj_all)

    @printf("\nSubspace diagnostics:\n")
    @printf("JFEM basis modes: %d (rank %d)   Nastran modes: %d (rank %d)\n",
            size(jfem_modes, 2), size(Qj_all, 2), length(n_sorted), size(Qn, 2))
    if !isempty(overlap)
        @printf("Nastran subspace vs all JFEM modes: singular-MAC min=%.4f mean=%.4f max=%.4f, mean N-vector projection=%.4f\n",
                minimum(overlap), sum(overlap) / length(overlap), maximum(overlap),
                sum(all_proj) / length(all_proj))
    else
        @printf("Nastran subspace vs all JFEM modes: NA\n")
    end

    @printf("\n| N mode | lambda N | proj into all J | J modes within %.1f%% | proj into window |\n", 100.0 * window)
    @printf("|---:|---:|---:|:---|---:|\n")
    for (k, nm) in enumerate(n_sorted)
        lam_n = nm.eigenvalue
        cand = candidate_modes_by_window(jfem_vals, lam_n, window)
        win_proj = NaN
        if !isempty(cand)
            Qj_win = orthonormal_basis(jfem_matrix(jfem_modes, node_index, common, cand, dofs))
            win_proj = projection_mac(view(n_mat, :, k), Qj_win)
        end
        @printf("| %d | %s | %s | %s | %s |\n",
                nm.mode,
                lam_n === nothing ? "NA" : @sprintf("%.6e", lam_n),
                format_projection(all_proj[k]),
                format_mode_list(cand),
                format_projection(win_proj))
    end
end

function compare_subcase(subcase::String, jfem::JFEMModes,
                         jfem_vals::Vector{Float64}, jfem_modes::Matrix{Float64},
                         n_modes::Vector{NastranMode}, dofs::Vector{Int},
                         subspace::Bool, window::Float64; csv_rows=nothing)
    isempty(n_modes) && return
    node_index = Dict(nid => i for (i, nid) in enumerate(jfem.node_ids))
    common_set = Set(keys(n_modes[1].values))
    for nm in n_modes[2:end]
        intersect!(common_set, keys(nm.values))
    end
    common = [nid for nid in sort!(collect(common_set)) if haskey(node_index, nid)]
    if isempty(common)
        @printf("\n## Subcase %s\nNo common grid IDs between JFEM and Nastran vectors.\n", subcase)
        return
    end

    @printf("\n## Subcase %s\n", subcase)
    @printf("Common grids: %d   DOFs: %s\n", length(common), join(dofs, ","))
    @printf("| N mode | lambda N | best J mode | lambda J | rel lambda | MAC |\n")
    @printf("|---:|---:|---:|---:|---:|---:|\n")

    Qj_all = nothing
    if csv_rows !== nothing
        j_cols = collect(1:size(jfem_modes, 2))
        j_mat = jfem_matrix(jfem_modes, node_index, common, j_cols, dofs)
        Qj_all = orthonormal_basis(j_mat)
    end

    mac_values = Float64[]
    for nm in sort(n_modes; by=m -> m.mode)
        vn = nastran_vector(nm, common, dofs)
        best_j = 0
        best_mac = -1.0
        for j in 1:size(jfem_modes, 2)
            vj = jfem_vector(jfem_modes, node_index, common, j, dofs)
            mval = mac(vj, vn)
            if mval > best_mac
                best_mac = mval
                best_j = j
            end
        end
        push!(mac_values, best_mac)
        lam_n = nm.eigenvalue
        lam_j = best_j > 0 && best_j <= length(jfem_vals) ? jfem_vals[best_j] : NaN
        rel = lam_n === nothing || abs(lam_n) < 1e-30 ? NaN : abs(lam_j - lam_n) / abs(lam_n)
        @printf("| %d | %s | %d | %.6e | %s | %.4f |\n",
                nm.mode,
                lam_n === nothing ? "NA" : @sprintf("%.6e", lam_n),
                best_j,
                lam_j,
                isnan(rel) ? "NA" : @sprintf("%.4f", rel),
                best_mac)

        if csv_rows !== nothing
            cand = candidate_modes_by_window(jfem_vals, lam_n, window)
            win_proj = NaN
            if !isempty(cand)
                Qj_win = orthonormal_basis(jfem_matrix(jfem_modes, node_index, common, cand, dofs))
                win_proj = projection_mac(vn, Qj_win)
            end
            push!(csv_rows, Dict{String,Any}(
                "subcase" => subcase,
                "nastran_mode" => nm.mode,
                "nastran_lambda" => lam_n,
                "best_jfem_mode" => best_j,
                "best_jfem_lambda" => lam_j,
                "best_rel_lambda" => rel,
                "best_mac" => best_mac,
                "projection_all_jfem" => projection_mac(vn, Qj_all),
                "window_rel_tol" => window,
                "window_modes" => format_mode_list(cand),
                "projection_window" => win_proj,
                "common_grids" => length(common),
            ))
        end
    end
    @printf("\nMAC summary: min=%.4f mean=%.4f max=%.4f\n",
            minimum(mac_values), sum(mac_values) / length(mac_values), maximum(mac_values))

    if subspace
        print_subspace_diagnostics(jfem_vals, jfem_modes, node_index, common, n_modes, dofs, window)
    end
end

function parse_args(args)
    jfem_path = ""
    nas_path = ""
    dofs = copy(DEFAULT_DOFS)
    subcase_filter = nothing
    subspace = false
    window = 0.05
    csv_path = nothing
    for arg in args
        if startswith(arg, "--dofs=")
            dofs = [parse(Int, c) for c in collect(replace(arg[8:end], "," => ""))]
        elseif startswith(arg, "--subcase=")
            subcase_filter = arg[11:end]
        elseif arg == "--subspace"
            subspace = true
        elseif startswith(arg, "--window=")
            window = parse(Float64, arg[10:end])
        elseif startswith(arg, "--csv=")
            csv_path = arg[7:end]
        elseif isempty(jfem_path)
            jfem_path = arg
        elseif isempty(nas_path)
            nas_path = arg
        else
            error("Unexpected argument: $arg")
        end
    end
    isempty(jfem_path) && error("Missing <case.jfem>")
    isempty(nas_path) && error("Missing <nastran.pch|nastran.f06>")
    all(1 .<= dofs .<= 6) || error("--dofs must contain digits 1..6")
    window >= 0.0 || error("--window must be non-negative")
    if csv_path !== nothing && isempty(strip(csv_path))
        error("--csv requires a non-empty output path")
    end
    return jfem_path, nas_path, dofs, subcase_filter, subspace, window, csv_path
end

function main()
    jfem_path, nas_path, dofs, subcase_filter, subspace, window, csv_path = parse_args(ARGS)
    isfile(jfem_path) || error("JFEM file not found: $jfem_path")
    isfile(nas_path) || error("Nastran vector file not found: $nas_path")

    jfem = read_jfem_buckling(jfem_path)
    report_path = companion_report_path(jfem_path)
    jfem_groups = split_jfem_by_report(jfem, report_path)
    nas_modes = parse_nastran_punch_or_f06(nas_path)

    @printf("JFEM: %s\n", jfem_path)
    @printf("  nodes=%d modes=%d report=%s\n", length(jfem.node_ids), size(jfem.modes, 2),
            isfile(report_path) ? report_path : "<none>")
    @printf("Nastran vectors: %s\n", nas_path)
    @printf("  parsed modes=%d\n", length(nas_modes))
    if isempty(nas_modes)
        println("\nNo Nastran eigenvectors were parsed.")
        println("Use an .f06 with printed buckling eigenvectors or a .pch containing \$EIGENVECTOR blocks; static \$DISPLACEMENTS punch blocks are ignored.")
        return
    end

    by_subcase = Dict{String,Vector{NastranMode}}()
    for m in nas_modes
        push!(get!(by_subcase, m.subcase, NastranMode[]), m)
    end

    csv_rows = csv_path === nothing ? nothing : Vector{Dict{String,Any}}()
    for subcase in sort(collect(keys(by_subcase)))
        subcase_filter !== nothing && subcase != subcase_filter && continue
        if !haskey(jfem_groups, subcase)
            if length(jfem_groups) == 1 && haskey(jfem_groups, "ALL")
                vals, modes = jfem_groups["ALL"]
                compare_subcase(subcase, jfem, vals, modes, by_subcase[subcase], dofs, subspace, window;
                                csv_rows=csv_rows)
            else
                @printf("\n## Subcase %s\nNo matching JFEM subcase in report.\n", subcase)
            end
            continue
        end
        vals, modes = jfem_groups[subcase]
        compare_subcase(subcase, jfem, vals, modes, by_subcase[subcase], dofs, subspace, window;
                        csv_rows=csv_rows)
    end
    if csv_path !== nothing
        write_mac_csv(csv_path, csv_rows)
        @printf("\nWrote MAC CSV: %s (%d rows)\n", csv_path, length(csv_rows))
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
