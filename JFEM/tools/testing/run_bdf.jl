# Usage:
#   julia --project=JFEM JFEM/tools/testing/run_bdf.jl model.bdf output_dir [FLAG=val;FLAG2=val2]

include(joinpath(@__DIR__, "run_manifest.jl"))
using OpenJFEM

length(ARGS) < 2 && error("usage: run_bdf.jl <model.bdf> <output_dir> [FLAG=val;FLAG2=val2]")

const BDF_PATH = ARGS[1]
const OUT_DIR = ARGS[2]
const FLAGS_RAW = length(ARGS) >= 3 ? ARGS[3] : ""
const APPLIED_FLAGS = apply_jfem_flags!(FLAGS_RAW)
const EXPORT_JFEM_BINARY = lowercase(strip(get(ENV, "JFEM_EXPORT_BINARY", "true"))) in
                           ("1", "true", "yes", "on")

mkpath(OUT_DIR)
write_run_manifest(OUT_DIR;
    repo_root=normpath(joinpath(@__DIR__, "..", "..", "..")),
    bdf_path=BDF_PATH,
    script_path=@__FILE__,
    args=ARGS,
    applied_flags=APPLIED_FLAGS,
    extra=Dict{String,Any}(
        "flags_raw" => FLAGS_RAW,
        "export_jfem_binary" => EXPORT_JFEM_BINARY,
    ))
OpenJFEM.main(BDF_PATH; output_dir=OUT_DIR, export_jfem_binary=EXPORT_JFEM_BINARY)
