# main.jl — JFEM entry point
#
# Three-stage architecture:
#   Stage 1: bdf_to_model(filename)           -> model Dict (JSON-serializable)
#   Stage 2: solve_model(model)               -> results Dict
#   Stage 3: export_results(results, ...)     -> writes JFEM binary by default
#                                                with JSON/VTK/HDF5 available on request
#
# The old monolithic main(filename) is preserved as a convenience wrapper
# that runs all three stages sequentially.

using JSON

_jfem_modules_preloaded =
    isdefined(@__MODULE__, :FEM) &&
    isdefined(@__MODULE__, :NastranParser) &&
    isdefined(@__MODULE__, :Solver) &&
    isdefined(@__MODULE__, :solve_model)

println(_jfem_modules_preloaded ? ">>> Reusing Loaded Modules..." : ">>> Loading Modules...")

include("jfem_bootstrap.jl")

# ============================================================================
# Convenience entry point: BDF file → solve → export (backward compatible)
# ============================================================================

function main(filename::String;
              output_dir::Union{String,Nothing}=nothing,
              export_model_json::Bool=false,
              export_card_inventory::Bool=false,
              export_json::Bool=false,
              export_vtk::Bool=false,
              export_hdf5::Bool=false,
              export_jfem_binary::Bool=true,
              export_report::Bool=true)
    if !isfile(filename)
        println("ERROR: File not found at: $filename")
        return
    end

    script_dir = @__DIR__
    if isnothing(output_dir)
        output_dir = joinpath(script_dir, "..", "output")
    end
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    # --- Stage 1: BDF → Model Dict ---
    # Parse BDF once, build model, and export card inventory
    t_parse_start = time_ns()
    println(">>> Reading BDF file: $filename")
    lines = readlines(filename)
    lines = NastranParser.resolve_includes(lines, dirname(abspath(filename)))
    println(">>> Checking format...")
    lines = NastranParser.convert_mystran_to_nastran(lines)
    println(">>> Parsing Bulk Data...")
    cc, bulk = NastranParser.read_bulk_and_case(lines)
    cards = NastranParser.process_cards(bulk)
    if export_card_inventory
        _ensure_export_extensions!()
        Base.invokelatest(export_card_inventory, cards, output_dir, filename)
    end
    t_parse = (time_ns() - t_parse_start) * 1e-9

    t_build_start = time_ns()
    println(">>> Constructing Model Data...")
    model = build_model(cards, cc)
    resolve_nested_coords!(model)
    transform_geometry!(model)
    t_build = (time_ns() - t_build_start) * 1e-9

    # Optionally write the model dict as JSON (e.g. test.bdf → test.bdf.json)
    if export_model_json
        json_path = filename * ".json"
        println(">>> Exporting model JSON: $json_path")
        open(json_path, "w") do f
            JSON.print(f, model, 2)
        end
        println(">>> Model JSON exported: $json_path")
    end

    # --- Stage 2: Solve ---
    t_solve_start = time_ns()
    results = solve_model(model)
    t_solve = (time_ns() - t_solve_start) * 1e-9

    # --- Stage 3: Export ---
    # Run the non-report exporters first so we can measure their wall time
    # and include it in the markdown report, then write the report itself.
    t_export_start = time_ns()
    export_results(results, filename, output_dir;
        export_json=export_json,
        export_vtk=export_vtk,
        export_hdf5=export_hdf5,
        export_jfem_binary=export_jfem_binary,
        export_report=false)
    t_export = (time_ns() - t_export_start) * 1e-9

    pipeline_timings = Dict{String,Any}(
        "parse"       => t_parse,
        "build_model" => t_build,
        "solve"       => t_solve,
        "export"      => t_export,
    )
    if export_report
        Base.invokelatest(export_markdown_report, results, filename, output_dir; timings=pipeline_timings)
    end

    # --- Adjoint sensitivity (optional) ---
    adjoint_config = joinpath(dirname(abspath(filename)), "adjoint_config.json")
    if isfile(adjoint_config)
        sol = get(results, "sol_type", 0)
        base_name = splitext(basename(filename))[1]
        adj_output = joinpath(output_dir, base_name * ".ADJOINT.JSON")
        if sol == 101
            println(">>> Found adjoint_config.json — running adjoint sensitivity analysis (SOL 101)")
            adj_results = solve_adjoint(results, adjoint_config)
            export_adjoint_json(adj_results, adj_output)
            println(">>> Adjoint results written to: $adj_output")
        elseif sol == 105
            println(">>> Found adjoint_config.json — running buckling sensitivity analysis (SOL 105)")
            adj_results = solve_adjoint_buckling(results, adjoint_config)
            export_adjoint_json(adj_results, adj_output)
            println(">>> Buckling adjoint results written to: $adj_output")
        end
    end
    return results
end

# ============================================================================
# CLI entry point
# ============================================================================

if normpath(abspath(PROGRAM_FILE)) == normpath(@__FILE__)
    if !isempty(ARGS)
        # Parse CLI arguments: <bdf_file> [output_dir] [--export-model-json]
        positional = filter(a -> !startswith(a, "--"), ARGS)
        flags = filter(a -> startswith(a, "--"), ARGS)
        do_export_json = "--export-model-json" in flags

        target_file = positional[1]
        if !isabspath(target_file)
            target_file = joinpath(@__DIR__, "..", target_file)
        end
        target_file = normpath(target_file)
        out_dir = length(positional) >= 2 ? normpath(positional[2]) : nothing
        main(target_file; output_dir=out_dir, export_model_json=do_export_json)
    else
        target_file = joinpath(@__DIR__, "..", "models", "OpenJFEM.bdf")
        target_file = normpath(target_file)
        main(target_file)
    end
end
