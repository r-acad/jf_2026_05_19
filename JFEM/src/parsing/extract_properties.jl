# extract_properties.jl — PSHELL, PBARL, PBAR, PCOMP, PROD, PELAS, EIGRL, PSOLID

function extract_props_shell(cards)
    d = Dict()
    for c in cards
        pid = to_id(parse_nastran_number(safe_get(c, 3)))
        mid = to_id(parse_nastran_number(safe_get(c, 4)))
        t = parse_nastran_number(safe_get(c, 5), 0.0)
        # MID2 = field 5 (c[6]): blank → membrane-only (Nastran V70.5: blank MID2 = no bending stiffness)
        # 12I/T^3 = field 6 (c[7]), MID3 = field 7 (c[8]), TS/T = field 8 (c[9]), NSM = field 9 (c[10])
        mid2_val = parse_nastran_number(safe_get(c, 6), nothing)
        mid2_blank = isnothing(mid2_val) || mid2_val == 0
        mid2 = mid2_blank ? 0 : to_id(mid2_val)
        bend_ratio = mid2_blank ? 0.0 : parse_nastran_number(safe_get(c, 7), 1.0)  # 12I/T^3, default=1.0
        mid3_val = parse_nastran_number(safe_get(c, 8), nothing)
        mid3 = (isnothing(mid3_val) || mid3_val == 0) ? 0 : to_id(mid3_val)
        ts_t = parse_nastran_number(safe_get(c, 9), 5.0/6.0)    # TS/T, default=5/6
        nsm = parse_nastran_number(safe_get(c, 10), 0.0)        # NSM, non-structural mass per area
        d[string(pid)] = Dict(
            "PID"=>pid,
            "MID"=>mid,
            "MID2"=>mid2,
            "MID3"=>mid3,
            "T"=>t,
            "BEND_RATIO"=>bend_ratio,
            "TS_T"=>ts_t,
            "NSM"=>Float64(nsm),
        )
    end
    return d
end

function extract_pbarl(cards)
    d = Dict()
    for c in cards
        pid = to_id(parse_nastran_number(safe_get(c, 3)))
        mid = to_id(parse_nastran_number(safe_get(c, 4)))
        type = "ROD"
        dim_start_idx = 7
        for k in 5:min(12, length(c))
            val = strip(string(safe_get(c, k, "")))
            if !isempty(val) && occursin(r"^[A-Za-z]", val)
                type = uppercase(val); dim_start_idx = k + 1; break
            end
        end
        numeric_values = Float64[]
        for k in dim_start_idx:length(c)
            val = parse_nastran_number(safe_get(c, k), nothing)
            if isa(val, Number)
                push!(numeric_values, Float64(val))
            end
        end
        expected_dim_count = if type == "ROD"
            1
        elseif type == "TUBE" || type == "TUBE2"
            2
        elseif type == "BAR"
            2
        elseif type == "BOX"
            4
        elseif type == "I"
            6
        elseif type == "CHAN"
            4
        elseif type == "HAT"
            4
        elseif type == "T" || type == "T1" || type == "T2"
            4
        elseif type == "Z"
            4
        elseif type == "L"
            4
        else
            length(numeric_values)
        end
        dims = numeric_values[1:min(expected_dim_count, length(numeric_values))]
        nsm = length(numeric_values) > expected_dim_count ? numeric_values[expected_dim_count + 1] : 0.0

        A, I1, I2, J = 1.0, 1.0, 1.0, 1.0
        K1, K2 = 0.0, 0.0
        if type == "ROD" && length(dims) >= 1
            R = dims[1]; A = pi*R^2; I1 = pi*R^4/4; I2 = I1; J = pi*R^4/2
        elseif (type == "TUBE" || type == "TUBE2") && length(dims) >= 2
            R_out = dims[1]; R_in = dims[2]
            if R_in < 0; R_in = 0.0; end
            A = pi*(R_out^2 - R_in^2)
            I1 = pi*(R_out^4 - R_in^4)/4
            I2 = I1
            J = pi*(R_out^4 - R_in^4)/2
        elseif type == "BAR" && length(dims) >= 2
            w = dims[1]; h = dims[2]
            A = w * h
            # NASTRAN BAR: DIM1=z-extent, DIM2=y-extent
            # I1 = Iz (Plane 1) = DIM1*DIM2^3/12;  I2 = Iy (Plane 2) = DIM2*DIM1^3/12
            I1 = w * h^3 / 12
            I2 = h * w^3 / 12
            if w >= h && w > 0
                J = w * h^3 / 3 * (1 - 0.63 * h / w)
            elseif h > 0
                J = h * w^3 / 3 * (1 - 0.63 * w / h)
            else
                J = I1 + I2
            end
        elseif type == "BOX" && length(dims) >= 4
            w = dims[1]; h = dims[2]; tw = dims[3]; th = dims[4]
            w_in = w - 2*tw; h_in = h - 2*th
            if w_in < 0; w_in = 0; end
            if h_in < 0; h_in = 0; end
            A = w*h - w_in*h_in
            I1 = (w*h^3 - w_in*h_in^3)/12
            I2 = (h*w^3 - h_in*w_in^3)/12
            J = 2*tw*th*(w-tw)^2*(h-th)^2 / (tw*(w-tw) + th*(h-th))
        elseif type == "I" && length(dims) >= 6
            H = dims[1]; Bb = dims[2]; Bt = dims[3]; tw = dims[4]; tfb = dims[5]; tft = dims[6]
            hw = H - tfb - tft
            A = Bb*tfb + hw*tw + Bt*tft
            yb = (Bb*tfb*tfb/2 + hw*tw*(tfb+hw/2) + Bt*tft*(H-tft/2)) / A
            I1 = Bb*tfb^3/12 + Bb*tfb*(yb-tfb/2)^2 + tw*hw^3/12 + tw*hw*(yb-tfb-hw/2)^2 + Bt*tft^3/12 + Bt*tft*(yb-(H-tft/2))^2
            I2 = tfb*Bb^3/12 + hw*tw^3/12 + tft*Bt^3/12
            J = (Bb*tfb^3 + hw*tw^3 + Bt*tft^3) / 3
        elseif type == "CHAN" && length(dims) >= 4
            bf = dims[1]; h = dims[2]; tw = dims[3]; tf = dims[4]
            hw = h - 2*tf
            A = 2*bf*tf + hw*tw
            yc = (2*bf*tf*(bf/2) + hw*tw*0) / A
            I1 = (tw*hw^3)/12 + 2*(bf*tf^3/12 + bf*tf*((h-tf)/2)^2)
            I2 = (hw*tw^3)/12 + 2*(tf*bf^3/12 + bf*tf*(bf/2 - yc)^2) + hw*tw*yc^2
            J = (2*bf*tf^3 + hw*tw^3) / 3
        elseif type == "HAT" && length(dims) >= 4
            w_top = dims[1]; t = dims[2]; w_bot = dims[3]; h_hat = dims[4]
            h_web = h_hat - t
            A = w_top*t + 2*h_web*t + w_bot*t
            yc = (w_bot*t*t/2 + 2*h_web*t*(t + h_web/2) + w_top*t*(h_hat - t/2)) / A
            I1 = w_bot*t^3/12 + w_bot*t*(yc - t/2)^2 + 2*(t*h_web^3/12 + t*h_web*(yc - t - h_web/2)^2) + w_top*t^3/12 + w_top*t*(yc - h_hat + t/2)^2
            I2 = t*w_bot^3/12 + 2*h_web*t^3/12 + t*w_top^3/12
            J = (w_bot*t^3 + 2*h_web*t^3 + w_top*t^3) / 3
        elseif (type == "T" || type == "T1" || type == "T2") && length(dims) >= 4
            bf = dims[1]; h_t = dims[2]; tf = dims[3]; tw = dims[4]
            hw = h_t - tf
            A = bf*tf + hw*tw
            yc = (hw*tw*hw/2 + bf*tf*(hw + tf/2)) / A
            I1 = tw*hw^3/12 + tw*hw*(yc - hw/2)^2 + bf*tf^3/12 + bf*tf*(hw + tf/2 - yc)^2
            I2 = hw*tw^3/12 + tf*bf^3/12
            J = (hw*tw^3 + bf*tf^3) / 3
        elseif type == "Z" && length(dims) >= 4
            bf = dims[1]; tf = dims[2]; hw = dims[3]; h_z = dims[4]
            tw = tf
            A = 2*bf*tf + hw*tw
            I1 = tw*hw^3/12 + 2*(bf*tf^3/12 + bf*tf*(hw/2 + tf/2)^2)
            I2 = 2*(tf*bf^3/12) + hw*tw^3/12
            J = (2*bf*tf^3 + hw*tw^3) / 3
        elseif type == "L" && length(dims) >= 4
            bf = dims[1]; h = dims[2]; tf = dims[3]; tw = dims[4]
            hw = h - tf  # vertical leg height above horizontal flange
            A = bf*tf + hw*tw
            yc = (bf*tf*(tf/2) + hw*tw*(tf + hw/2)) / A  # centroid from bottom
            xc = (bf*tf*(bf/2) + hw*tw*(tw/2)) / A       # centroid from left
            I1 = bf*tf^3/12 + bf*tf*(yc-tf/2)^2 + tw*hw^3/12 + tw*hw*(yc-tf-hw/2)^2
            I2 = tf*bf^3/12 + bf*tf*(xc-bf/2)^2 + hw*tw^3/12 + hw*tw*(xc-tw/2)^2
            J  = (bf*tf^3 + hw*tw^3) / 3
        else
            if !isempty(dims)
                R = dims[1]; A = pi*R^2; I1 = pi*R^4/4; I2 = I1; J = pi*R^4/2
            end
            println("[WARNING] PBARL type '$type' not fully supported. Using approximate properties.")
        end
        # Stress recovery coefficients (y,z coordinates of 4 corner points)
        C1, C2, D1, D2, E1, E2, F1, F2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if type == "ROD" && length(dims) >= 1
            R = dims[1]
            C1=-R; C2=0.0; D1=0.0; D2=-R; E1=R; E2=0.0; F1=0.0; F2=R
        elseif (type == "TUBE" || type == "TUBE2") && length(dims) >= 1
            R_o = dims[1]
            C1=-R_o; C2=0.0; D1=0.0; D2=-R_o; E1=R_o; E2=0.0; F1=0.0; F2=R_o
        elseif type == "BAR" && length(dims) >= 2
            w2 = dims[1]/2; h2 = dims[2]/2
            C1=h2; C2=w2; D1=-h2; D2=w2; E1=-h2; E2=-w2; F1=h2; F2=-w2
        elseif type == "BOX" && length(dims) >= 2
            w2 = dims[1]/2; h2 = dims[2]/2
            C1=h2; C2=w2; D1=-h2; D2=w2; E1=-h2; E2=-w2; F1=h2; F2=-w2
        end
        # Auto-compute shear factors if not provided (PBARL doesn't have K1/K2 fields)
        if K1 == 0.0 && K2 == 0.0 && !isempty(dims)
            K1, K2 = compute_pbarl_shear_factors(type, dims, 0.3)  # use nu=0.3 default
        end
        d[string(pid)] = Dict("PID"=>pid, "MID"=>mid, "A"=>A, "I"=>I1, "I1"=>I1, "I2"=>I2, "J"=>J, "TYPE"=>type, "K1"=>K1, "K2"=>K2,
            "C1"=>C1, "C2"=>C2, "D1"=>D1, "D2"=>D2, "E1"=>E1, "E2"=>E2, "F1"=>F1, "F2"=>F2, "DIMS"=>copy(dims), "NSM"=>Float64(nsm))
    end
    return d
end

function extract_pbar(cards)
    d = Dict()
    for c in cards
        pid = to_id(parse_nastran_number(safe_get(c, 3)))
        mid = to_id(parse_nastran_number(safe_get(c, 4)))
        A   = parse_nastran_number(safe_get(c, 5), 0.0)
        I1  = parse_nastran_number(safe_get(c, 6), 0.0)
        I2  = parse_nastran_number(safe_get(c, 7), 0.0)
        J   = parse_nastran_number(safe_get(c, 8), 0.0)
        NSM = parse_nastran_number(safe_get(c, 9), 0.0)
        # Stress recovery points on continuation line 1 (indices 11-18)
        C1 = parse_nastran_number(safe_get(c, 11), 0.0)
        C2 = parse_nastran_number(safe_get(c, 12), 0.0)
        D1 = parse_nastran_number(safe_get(c, 13), 0.0)
        D2 = parse_nastran_number(safe_get(c, 14), 0.0)
        E1 = parse_nastran_number(safe_get(c, 15), 0.0)
        E2 = parse_nastran_number(safe_get(c, 16), 0.0)
        F1 = parse_nastran_number(safe_get(c, 17), 0.0)
        F2 = parse_nastran_number(safe_get(c, 18), 0.0)
        # Second continuation: K1, K2, I12
        K1 = parse_nastran_number(safe_get(c, 19), 0.0)
        K2 = parse_nastran_number(safe_get(c, 20), 0.0)
        I12 = parse_nastran_number(safe_get(c, 21), 0.0)
        if pid > 0
            d[string(pid)] = Dict(
                "PID"=>pid, "MID"=>mid, "A"=>Float64(A),
                "I1"=>Float64(I1), "I2"=>Float64(I2), "I12"=>Float64(I12), "J"=>Float64(J),
                "I"=>Float64(I1), "NSM"=>Float64(NSM),
                "TYPE"=>"PBAR", "K1"=>Float64(K1), "K2"=>Float64(K2),
                "C1"=>Float64(C1), "C2"=>Float64(C2), "D1"=>Float64(D1), "D2"=>Float64(D2),
                "E1"=>Float64(E1), "E2"=>Float64(E2), "F1"=>Float64(F1), "F2"=>Float64(F2)
            )
        end
    end
    return d
end

function extract_pbeam(cards)
    d = Dict()
    for c in cards
        pid = to_id(parse_nastran_number(safe_get(c, 3)))
        mid = to_id(parse_nastran_number(safe_get(c, 4)))

        # Station A properties (fields 5-10)
        A_a   = Float64(parse_nastran_number(safe_get(c, 5), 0.0))
        I1_a  = Float64(parse_nastran_number(safe_get(c, 6), 0.0))
        I2_a  = Float64(parse_nastran_number(safe_get(c, 7), 0.0))
        I12_a = Float64(parse_nastran_number(safe_get(c, 8), 0.0))
        J_a   = Float64(parse_nastran_number(safe_get(c, 9), 0.0))

        # Station A stress recovery (fields 11-18)
        C1_a = Float64(parse_nastran_number(safe_get(c, 11), 0.0))
        C2_a = Float64(parse_nastran_number(safe_get(c, 12), 0.0))
        D1_a = Float64(parse_nastran_number(safe_get(c, 13), 0.0))
        D2_a = Float64(parse_nastran_number(safe_get(c, 14), 0.0))
        E1_a = Float64(parse_nastran_number(safe_get(c, 15), 0.0))
        E2_a = Float64(parse_nastran_number(safe_get(c, 16), 0.0))
        F1_a = Float64(parse_nastran_number(safe_get(c, 17), 0.0))
        F2_a = Float64(parse_nastran_number(safe_get(c, 18), 0.0))

        # Default station B = station A
        A_b = A_a; I1_b = I1_a; I2_b = I2_a; I12_b = I12_a; J_b = J_a
        C1_b = C1_a; C2_b = C2_a; D1_b = D1_a; D2_b = D2_a
        E1_b = E1_a; E2_b = E2_a; F1_b = F1_a; F2_b = F2_a

        # Scan for intermediate/end stations (16-field blocks starting at field 19)
        # Format: SO X/XB A I1 I2 I12 J NSM | C1 C2 D1 D2 E1 E2 F1 F2
        k = 19
        while k + 7 <= length(c)
            so_raw = uppercase(strip(string(safe_get(c, k, ""))))
            if so_raw == "YES" || so_raw == "YESA" || so_raw == "NO"
                A_b   = Float64(parse_nastran_number(safe_get(c, k+2), A_a))
                I1_b  = Float64(parse_nastran_number(safe_get(c, k+3), I1_a))
                I2_b  = Float64(parse_nastran_number(safe_get(c, k+4), I2_a))
                I12_b = Float64(parse_nastran_number(safe_get(c, k+5), I12_a))
                J_b   = Float64(parse_nastran_number(safe_get(c, k+6), J_a))
                # Stress recovery for this station (8 more fields)
                if k + 15 <= length(c)
                    C1_b = Float64(parse_nastran_number(safe_get(c, k+8),  0.0))
                    C2_b = Float64(parse_nastran_number(safe_get(c, k+9),  0.0))
                    D1_b = Float64(parse_nastran_number(safe_get(c, k+10), 0.0))
                    D2_b = Float64(parse_nastran_number(safe_get(c, k+11), 0.0))
                    E1_b = Float64(parse_nastran_number(safe_get(c, k+12), 0.0))
                    E2_b = Float64(parse_nastran_number(safe_get(c, k+13), 0.0))
                    F1_b = Float64(parse_nastran_number(safe_get(c, k+14), 0.0))
                    F2_b = Float64(parse_nastran_number(safe_get(c, k+15), 0.0))
                end
                k += 16
            else
                break  # Reached K1/K2 or end of card
            end
        end

        # K1, K2 follow after all stations
        K1 = Float64(parse_nastran_number(safe_get(c, k),   0.0))
        K2 = Float64(parse_nastran_number(safe_get(c, k+1), 0.0))

        # Average properties between station A and station B
        A_avg   = (A_a + A_b) / 2.0
        I1_avg  = (I1_a + I1_b) / 2.0
        I2_avg  = (I2_a + I2_b) / 2.0
        I12_avg = (I12_a + I12_b) / 2.0
        J_avg   = (J_a + J_b) / 2.0

        if pid > 0
            d[string(pid)] = Dict(
                "PID"=>pid, "MID"=>mid, "A"=>A_avg,
                "I1"=>I1_avg, "I2"=>I2_avg, "I12"=>I12_avg, "J"=>J_avg,
                "I"=>I1_avg,
                "TYPE"=>"PBEAM", "K1"=>K1, "K2"=>K2,
                "C1"=>(C1_a+C1_b)/2, "C2"=>(C2_a+C2_b)/2,
                "D1"=>(D1_a+D1_b)/2, "D2"=>(D2_a+D2_b)/2,
                "E1"=>(E1_a+E1_b)/2, "E2"=>(E2_a+E2_b)/2,
                "F1"=>(F1_a+F1_b)/2, "F2"=>(F2_a+F2_b)/2
            )
        end
    end
    return d
end

function extract_pbeaml(cards)
    # PBEAML uses same section-type calculation as PBARL
    # For NAPA_101: only constant-section BAR and ROD types
    return extract_pbarl(cards)
end

function extract_pcomp(cards)
    d = Dict()
    for c in cards
        pid  = to_id(parse_nastran_number(safe_get(c, 3)))
        z0   = parse_nastran_number(safe_get(c, 4), nothing)
        nsm  = parse_nastran_number(safe_get(c, 5), 0.0)
        # Field 10 (c[10]) is LAM (SYM, MEM, BEND, SMEAR, SMCORE)
        lam_field = strip(string(safe_get(c, 10, "")))
        is_sym = uppercase(lam_field) == "SYM"

        # Parse plies starting from field 11 (continuation)
        plies = []
        k = 11
        while k + 3 <= length(c)
            ply_mid   = to_id(parse_nastran_number(safe_get(c, k), 0))
            ply_t     = parse_nastran_number(safe_get(c, k+1), 0.0)
            ply_theta = parse_nastran_number(safe_get(c, k+2), 0.0)
            ply_sout  = strip(string(safe_get(c, k+3, "")))
            if ply_mid > 0 && ply_t > 0
                push!(plies, Dict("MID"=>ply_mid, "T"=>Float64(ply_t),
                                  "THETA"=>Float64(ply_theta), "SOUT"=>ply_sout))
            end
            k += 4
        end

        # Handle SYM: mirror plies
        if is_sym && !isempty(plies)
            full_plies = copy(plies)
            for i in length(plies):-1:1
                push!(full_plies, plies[i])
            end
            plies = full_plies
        end

        # Compute total thickness
        total_t = sum(p["T"] for p in plies; init=0.0)

        if pid > 0
            d[string(pid)] = Dict("PID"=>pid, "Z0"=>isnothing(z0) ? -total_t/2 : Float64(z0),
                "NSM"=>Float64(nsm), "LAM"=>lam_field, "PLIES"=>plies,
                "T"=>Float64(total_t), "TYPE"=>"PCOMP")
        end
    end
    return d
end

function extract_prod(cards)
    d = Dict()
    for c in cards
        pid = to_id(parse_nastran_number(safe_get(c, 3)))
        mid = to_id(parse_nastran_number(safe_get(c, 4)))
        A   = parse_nastran_number(safe_get(c, 5), 0.0)
        J   = parse_nastran_number(safe_get(c, 6), 0.0)
        C   = parse_nastran_number(safe_get(c, 7), 0.0)
        NSM = parse_nastran_number(safe_get(c, 8), 0.0)
        if pid > 0
            d[string(pid)] = Dict("PID"=>pid, "MID"=>mid, "A"=>Float64(A), "J"=>Float64(J),
                                  "C"=>Float64(C), "NSM"=>Float64(NSM), "TYPE"=>"PROD")
        end
    end
    return d
end

function extract_pshear(cards)
    d = Dict()
    for c in cards
        pid = to_id(parse_nastran_number(safe_get(c, 3)))
        mid = to_id(parse_nastran_number(safe_get(c, 4)))
        t   = Float64(parse_nastran_number(safe_get(c, 5), 0.0))
        nsm = Float64(parse_nastran_number(safe_get(c, 6), 0.0))
        if pid > 0
            d[string(pid)] = Dict("PID"=>pid, "MID"=>mid, "T"=>t, "NSM"=>nsm, "TYPE"=>"PSHEAR")
        end
    end
    return d
end

function extract_pelas(cards)
    d = Dict()
    for c in cards
        pid = to_id(parse_nastran_number(safe_get(c, 3)))
        K   = Float64(parse_nastran_number(safe_get(c, 4), 0.0))
        GE  = Float64(parse_nastran_number(safe_get(c, 5), 0.0))
        if pid > 0
            d[string(pid)] = Dict("PID"=>pid, "K"=>K, "GE"=>GE)
        end
    end
    return d
end

function extract_pmass(cards)
    d = Dict()
    for c in cards
        pid  = to_id(parse_nastran_number(safe_get(c, 3)))
        mass = Float64(parse_nastran_number(safe_get(c, 4), 0.0))
        if pid > 0
            d[string(pid)] = Dict("PID"=>pid, "M"=>mass, "TYPE"=>"PMASS")
        end
    end
    return d
end

# EIGRL — Real eigenvalue extraction (Lanczos method parameters for SOL103/SOL105)
# EIGRL  SID  V1  V2  ND  MSGLVL  MAXSET  SHFSCL
function _eigrl_canonical_option_name(name::AbstractString)
    key = uppercase(strip(name))
    key == "MSGL" && return "MSGLVL"
    key == "SHFS" && return "SHFSCL"
    return key
end

function _eigrl_parse_option_value(raw)
    parsed = parse_nastran_number(raw, nothing)
    if isnothing(parsed)
        return uppercase(strip(string(raw)))
    end
    return isa(parsed, Integer) ? parsed : Float64(parsed)
end

function _eigrl_parse_options(card)
    opts = Dict{String,Any}()
    for k in 11:length(card)
        token = strip(string(safe_get(card, k, "")))
        isempty(token) && continue
        for chunk in split(token, r"[,\s]+"; keepempty=false)
            occursin("=", chunk) || continue
            parts = split(chunk, "="; limit=2)
            length(parts) == 2 || continue
            key = _eigrl_canonical_option_name(parts[1])
            isempty(key) && continue
            opts[key] = _eigrl_parse_option_value(parts[2])
        end
    end
    return opts
end

function extract_eigrl(cards)
    d = Dict()
    for c in cards
        sid = to_id(parse_nastran_number(safe_get(c, 3)))
        sid <= 0 && continue

        opts = _eigrl_parse_options(c)

        v1_raw = haskey(opts, "V1") ? opts["V1"] : parse_nastran_number(safe_get(c, 4), nothing)
        v2_raw = haskey(opts, "V2") ? opts["V2"] : parse_nastran_number(safe_get(c, 5), nothing)
        nd_raw = haskey(opts, "ND") ? opts["ND"] : parse_nastran_number(safe_get(c, 6), 0)
        msglvl_raw = haskey(opts, "MSGLVL") ? opts["MSGLVL"] : parse_nastran_number(safe_get(c, 7), nothing)
        maxset_raw = haskey(opts, "MAXSET") ? opts["MAXSET"] : parse_nastran_number(safe_get(c, 8), nothing)
        shfscl_raw = haskey(opts, "SHFSCL") ? opts["SHFSCL"] : parse_nastran_number(safe_get(c, 9), nothing)

        norm_field = uppercase(strip(string(safe_get(c, 10, ""))))
        norm_raw = haskey(opts, "NORM") ? uppercase(string(opts["NORM"])) : norm_field

        v1 = Float64(isnothing(v1_raw) ? 0.0 : parse_nastran_number(v1_raw, 0.0))
        v2 = Float64(isnothing(v2_raw) ? 0.0 : parse_nastran_number(v2_raw, 0.0))
        nd = to_id(parse_nastran_number(nd_raw, 0))
        if nd == 0; nd = 3; end  # default: 3 eigenvalues

        entry = Dict{String,Any}("SID"=>sid, "V1"=>v1, "V2"=>v2, "ND"=>nd)
        !isnothing(msglvl_raw) && (entry["MSGLVL"] = to_id(parse_nastran_number(msglvl_raw, 0)))
        !isnothing(maxset_raw) && (entry["MAXSET"] = to_id(parse_nastran_number(maxset_raw, 0)))
        !isnothing(shfscl_raw) && (entry["SHFSCL"] = Float64(parse_nastran_number(shfscl_raw, 0.0)))
        !isempty(norm_raw) && (entry["NORM"] = norm_raw)
        !isempty(opts) && (entry["OPTIONS"] = opts)
        d[string(sid)] = entry
    end
    return d
end

function extract_psolid(cards)
    d = Dict()
    for c in cards
        pid   = to_id(parse_nastran_number(safe_get(c, 3)))
        mid   = to_id(parse_nastran_number(safe_get(c, 4)))
        cordm = to_id(parse_nastran_number(safe_get(c, 5), 0))
        in_scheme  = to_id(parse_nastran_number(safe_get(c, 6), 0))
        stress_loc = to_id(parse_nastran_number(safe_get(c, 7), 0))
        isop  = to_id(parse_nastran_number(safe_get(c, 8), 0))
        if pid > 0 && mid > 0
            d[string(pid)] = Dict("PID"=>pid, "MID"=>mid, "CORDM"=>cordm,
                "IN"=>in_scheme, "STRESS"=>stress_loc, "ISOP"=>isop, "TYPE"=>"PSOLID")
        end
    end
    return d
end

function extract_pbush(cards)
    d = Dict()
    for c in cards
        pid = to_id(parse_nastran_number(safe_get(c, 3)))
        # PBUSH format: PBUSH PID "K" K1 K2 K3 K4 K5 K6
        # Field 4 is the "K" keyword, fields 5-10 are stiffness values
        # But field 4 might be the keyword "K" (string) or the first stiffness value
        f4 = strip(string(safe_get(c, 4, "")))
        local K_vals::Vector{Float64}
        if uppercase(f4) == "K"
            # "K" keyword present — stiffness starts at field 5
            K_vals = [Float64(parse_nastran_number(safe_get(c, 5+k), 0.0)) for k in 0:5]
        else
            # No keyword — stiffness values start at field 4
            K_vals = [Float64(parse_nastran_number(safe_get(c, 4+k), 0.0)) for k in 0:5]
        end
        if pid > 0
            d[string(pid)] = Dict("PID"=>pid, "K"=>K_vals, "TYPE"=>"PBUSH")
        end
    end
    return d
end

"""
    compute_pbarl_shear_factors(type, dims, nu) -> (K1, K2)

Compute Timoshenko shear correction factors K1, K2 for PBARL sections.
K1 = shear factor for Plane 1 (maps to As_y = K1*A in assembly).
K2 = shear factor for Plane 2 (maps to As_z = K2*A in assembly).
Returns (0.0, 0.0) if section type is unknown.
"""
function compute_pbarl_shear_factors(type::String, dims::Vector{Float64}, nu::Float64)
    if type == "ROD" && length(dims) >= 1
        # Cowper (1966) formula for solid circle
        K = 6.0*(1.0+nu) / (7.0+6.0*nu)
        return (K, K)

    elseif (type == "TUBE" || type == "TUBE2") && length(dims) >= 2
        R_out = dims[1]; R_in = max(dims[2], 0.0)
        m = R_in / R_out
        if m < 1e-10
            K = 6.0*(1.0+nu) / (7.0+6.0*nu)  # solid circle
        else
            # Cowper formula for hollow circle
            m2 = m*m
            K = 6.0*(1.0+nu)*(1.0+m2)^2 / ((7.0+6.0*nu)*(1.0+m2)^2 + (20.0+12.0*nu)*m2)
        end
        return (K, K)

    elseif type == "BAR" && length(dims) >= 2
        # Solid rectangle: Cowper formula
        K = 10.0*(1.0+nu) / (12.0+11.0*nu)
        return (K, K)

    elseif type == "BOX" && length(dims) >= 4
        w = dims[1]; h = dims[2]; tw = dims[3]; th = dims[4]
        A = w*h - max(w-2*tw,0)*max(h-2*th,0)
        if A < 1e-30; return (0.0, 0.0); end
        # Approximate: shear area = web area for vertical shear, flange area for horizontal
        A_webs = 2*tw*max(h-2*th,0)  # vertical shear
        A_flanges = 2*th*w            # horizontal shear
        K1 = min(A_webs / A, 1.0)
        K2 = min(A_flanges / A, 1.0)
        return (K1, K2)

    elseif (type == "T" || type == "T1" || type == "T2") && length(dims) >= 4
        bf = dims[1]; h_t = dims[2]; tf = dims[3]; tw = dims[4]
        hw = max(h_t - tf, 0.0)
        A = bf*tf + hw*tw
        if A < 1e-30; return (0.0, 0.0); end
        K1 = min(hw*tw / A, 1.0)   # strong-axis shear through web
        K2 = min(bf*tf / A, 1.0)   # weak-axis shear through flange
        return (K1, K2)

    elseif type == "I" && length(dims) >= 6
        H = dims[1]; Bb = dims[2]; Bt = dims[3]; tw = dims[4]; tfb = dims[5]; tft = dims[6]
        hw = max(H - tfb - tft, 0.0)
        A = Bb*tfb + hw*tw + Bt*tft
        if A < 1e-30; return (0.0, 0.0); end
        K1 = min(hw*tw / A, 1.0)              # strong-axis shear through web
        K2 = min((Bb*tfb + Bt*tft) / A, 1.0)  # weak-axis shear through flanges
        return (K1, K2)

    elseif type == "L" && length(dims) >= 4
        # L-section: approximate using web/flange area ratios
        bf = dims[1]; h = dims[2]; tf = dims[3]; tw = dims[4]
        hw = h - tf
        A = bf*tf + hw*tw
        if A < 1e-30; return (0.0, 0.0); end
        K1 = min(hw*tw / A, 1.0)  # vertical shear through web
        K2 = min(bf*tf / A, 1.0)  # horizontal shear through flange
        return (K1, K2)

    elseif type == "CHAN" && length(dims) >= 4
        bf = dims[1]; h = dims[2]; tw = dims[3]; tf = dims[4]
        hw = h - 2*tf
        A = 2*bf*tf + hw*tw
        if A < 1e-30; return (0.0, 0.0); end
        K1 = min(hw*tw / A, 1.0)   # vertical shear through web
        K2 = min(2*bf*tf / A, 1.0) # horizontal shear through flanges
        return (K1, K2)
    end
    return (0.0, 0.0)
end

