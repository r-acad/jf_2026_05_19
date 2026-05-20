#!/usr/bin/env bash
# Fast deployment/precompile helper for OpenJFEM on Linux/macOS.
#
# This script is a user-friendly wrapper around tools/deploy_fast.jl. It:
#   1. Finds the JFEM Julia project directory automatically.
#   2. Instantiates Julia dependencies.
#   3. Runs package precompilation with representative decks.
#   4. Optionally builds a PackageCompiler sysimage.
#
# Typical use:
#   ./tools/precompile_jfem.sh
#
# With a representative production deck:
#   ./tools/precompile_jfem.sh --deck /path/to/representative_sol105.bdf
#
# With a batch manifest:
#   ./tools/precompile_jfem.sh --manifest /path/to/cases.json
#
# With a sysimage:
#   ./tools/precompile_jfem.sh --sysimage ./build/OpenJFEM_sysimage.so --install-packagecompiler

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  precompile_jfem.sh [options]

Options:
  --julia <exe>                  Julia executable to use. Default: julia
  --threads <n|auto>             Julia threads. Default: auto
  --deck <path>                  Representative BDF/DAT/NAS deck to warm.
  --manifest <path>              JSON batch manifest; every case input is used.
  --flags <FLAG=val,...>         Precompile-time JFEM flags.
  --sysimage <path>              Build a PackageCompiler sysimage at this path.
  --install-packagecompiler      Allow installing PackageCompiler if missing.
  -h, --help                     Show this help.

With no --deck or --manifest, OpenJFEM uses bundled tiny precompile decks from:
  JFEM/examples/precompile

Recommended production pattern:
  1. Run this script once after installation or after code/package changes.
  2. Use JFEM/tools/run_batch_manifest.jl for command-line batches.
  3. Use JFEM/tools/jfem_worker_jsonl.jl for heavy Python optimization loops.

If a sysimage is built, pass it to future Julia launches with:
  julia --sysimage /path/to/OpenJFEM_sysimage.so ...
EOF
}

log() {
    printf '[precompile_jfem] %s\n' "$*"
}

die() {
    printf '[precompile_jfem] ERROR: %s\n' "$*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
jfem_project_dir="$(cd -- "$script_dir/.." && pwd -P)"

if [ ! -f "$jfem_project_dir/Project.toml" ]; then
    die "Could not find Project.toml in '$jfem_project_dir'. Run this script from the JFEM/tools directory shipped with OpenJFEM."
fi

julia_exe="julia"
threads="auto"
deploy_args=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --julia)
            [ "$#" -ge 2 ] || die "--julia requires a value"
            julia_exe="$2"
            shift 2
            ;;
        --julia=*)
            julia_exe="${1#*=}"
            shift
            ;;
        --threads)
            [ "$#" -ge 2 ] || die "--threads requires a value"
            threads="$2"
            shift 2
            ;;
        --threads=*)
            threads="${1#*=}"
            shift
            ;;
        --deck)
            [ "$#" -ge 2 ] || die "--deck requires a path"
            [ -f "$2" ] || die "deck not found: $2"
            deploy_args+=(--deck "$2")
            shift 2
            ;;
        --deck=*)
            deck="${1#*=}"
            [ -f "$deck" ] || die "deck not found: $deck"
            deploy_args+=(--deck "$deck")
            shift
            ;;
        --manifest)
            [ "$#" -ge 2 ] || die "--manifest requires a path"
            [ -f "$2" ] || die "manifest not found: $2"
            deploy_args+=(--manifest "$2")
            shift 2
            ;;
        --manifest=*)
            manifest="${1#*=}"
            [ -f "$manifest" ] || die "manifest not found: $manifest"
            deploy_args+=(--manifest "$manifest")
            shift
            ;;
        --flags)
            [ "$#" -ge 2 ] || die "--flags requires a value"
            deploy_args+=(--flags "$2")
            shift 2
            ;;
        --flags=*)
            deploy_args+=(--flags "${1#*=}")
            shift
            ;;
        --sysimage)
            [ "$#" -ge 2 ] || die "--sysimage requires a path"
            deploy_args+=(--sysimage="$2")
            shift 2
            ;;
        --sysimage=*)
            deploy_args+=("$1")
            shift
            ;;
        --install-packagecompiler)
            deploy_args+=(--install-packagecompiler)
            shift
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

command -v "$julia_exe" >/dev/null 2>&1 || die "Julia executable not found: $julia_exe"

log "JFEM project: $jfem_project_dir"
log "Julia:       $("$julia_exe" --version)"
log "Threads:     $threads"
if [ "${#deploy_args[@]}" -eq 0 ]; then
    log "Workload:    bundled precompile decks"
else
    log "Options:     ${deploy_args[*]}"
fi

export JFEM_SUPPRESS_THREAD_HINT=1

"$julia_exe" \
    --threads="$threads" \
    --startup-file=no \
    --project="$jfem_project_dir" \
    "$jfem_project_dir/tools/deploy_fast.jl" \
    "${deploy_args[@]}"

log "Precompile/deploy step complete."
log "For fastest repeated Python optimization, keep one JSONL worker alive:"
log "  $julia_exe --threads=$threads --startup-file=no --project=\"$jfem_project_dir\" \"$jfem_project_dir/tools/jfem_worker_jsonl.jl\""
