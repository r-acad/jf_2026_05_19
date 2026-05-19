# Run a JSON batch manifest in one Julia process.
#
# Usage:
#   julia --threads=auto --startup-file=no --project=JFEM JFEM/tools/run_batch_manifest.jl cases.json [--quiet] [--stop-on-error]

include(joinpath(@__DIR__, "testing", "run_manifest.jl"))
include(joinpath(@__DIR__, "manifest_batch_core.jl"))

function _usage()
    return """
    usage:
      julia --threads=auto --startup-file=no --project=JFEM JFEM/tools/run_batch_manifest.jl <cases.json> [--quiet] [--stop-on-error]

    The manifest defines output_root, defaults, and cases. Use --quiet to keep
    per-case solver output out of stdout; solver output is then written to each
    case's jfem_case_stdout.log.
    """
end

if any(arg -> arg in ("-h", "--help", "help"), ARGS)
    print(_usage())
    exit(0)
end

isempty(ARGS) && (print(_usage()); exit(1))

manifest_path = ARGS[1]
isfile(manifest_path) || error("manifest not found: $manifest_path")
quiet = "--quiet" in ARGS[2:end]
stop_on_error = "--stop-on-error" in ARGS[2:end]
unknown = [arg for arg in ARGS[2:end] if !(arg in ("--quiet", "--stop-on-error"))]
isempty(unknown) || error("unknown option(s): $(join(unknown, ", "))")

manifest, manifest_abs = load_batch_manifest(manifest_path)
default_flags = manifest_default_flags(manifest)
manifest_apply_flags!(default_flags)

using OpenJFEM

summary = run_batch_manifest!(manifest;
    manifest_path=manifest_abs,
    repo_root=normpath(joinpath(@__DIR__, "..", "..")),
    script_path=@__FILE__,
    args=ARGS,
    quiet=quiet,
    stop_on_error_override=stop_on_error ? true : nothing)

println(">>> batch_summary_json=$(summary["summary_json"])")
println(">>> batch_summary_csv=$(summary["summary_csv"])")
println(">>> completed=$(summary["completed"]) failed=$(summary["failed"]) total_wall=$(round(summary["total_wall_s"]; digits=3)) s")
