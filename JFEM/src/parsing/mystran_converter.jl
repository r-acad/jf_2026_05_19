# mystran_converter.jl — Convert MYSTRAN-format BDF files to NASTRAN format

function convert_mystran_to_nastran(lines::Vector{String})
    is_mystran = false
    for line in lines
        cl = strip(uppercase(line))
        if startswith(cl, "SOL ")
            m = match(r"SOL\s+(\d+)", cl)
            if m !== nothing
                sol_num = parse(Int, m.captures[1])
                if sol_num < 100; is_mystran = true; end
            end
            break
        end
    end

    if is_mystran
        println("[INFO] MYSTRAN format detected. Converting to NASTRAN format...")
    end

    # Permanent GRID/GRDSET constraints are now handled natively by the parser
    # and boundary-condition application. The converter only rewrites
    # MYSTRAN-specific syntax and control cards.
    !is_mystran && return lines

    # Phase 2: Transform lines
    result = String[]
    in_bulk = false

    for line in lines
        cl = strip(uppercase(line))

        if is_mystran && startswith(cl, "ID ") && !in_bulk; continue; end

        if is_mystran && startswith(cl, "SOL ") && !in_bulk
            m = match(r"SOL\s+(\d+)", cl)
            if m !== nothing
                sol_num = parse(Int, m.captures[1])
                if sol_num < 100
                    push!(result, "SOL $(sol_num + 100)")
                    println("[INFO]   SOL $sol_num -> SOL $(sol_num + 100)")
                    continue
                end
            end
        end

        if occursin("BEGIN BULK", cl)
            in_bulk = true
            push!(result, line)
            continue
        end

        if in_bulk
            card_name = ""
            if occursin(",", cl)
                card_name = uppercase(strip(split(cl, ",")[1]))
            elseif length(cl) >= 4
                card_name = uppercase(strip(cl[1:min(8, length(cl))]))
            end

            if is_mystran && card_name == "DEBUG"; continue; end
            if is_mystran && card_name == "PARAM" && occursin("SOLLIB", cl); continue; end
        else
            if is_mystran && startswith(cl, "ELDATA"); continue; end
        end

        push!(result, line)
    end

    return result
end
