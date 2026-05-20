"""Command-line helper for the manifest-based OpenJFEM production workflow.

This script is intentionally small and stdlib-only. It supports the first
production automation contract:

1. Create a JSON manifest that names input decks and output folders.
2. Run that manifest in one Julia process, or submit it to a JSONL worker.

For heavy optimization loops, import ``JFEMWorker`` from ``jfem_client.py`` and
keep the worker open across many iterations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from JFEM.python.jfem_client import JFEMWorker, run_batch_once, write_batch_manifest
except ModuleNotFoundError:
    from jfem_client import JFEMWorker, run_batch_once, write_batch_manifest


DEFAULT_PATTERNS = ("*.bdf", "*.dat", "*.nas")


def _as_path(value: str) -> Path:
    return Path(value).expanduser()


def _case_id(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _collect_input_decks(inputs: Iterable[str], input_dir: Optional[str], patterns: Iterable[str]) -> List[Path]:
    decks: List[Path] = []
    for raw in inputs:
        deck = _as_path(raw)
        if not deck.is_file():
            raise FileNotFoundError(f"input deck not found: {deck}")
        decks.append(deck)

    if input_dir:
        root = _as_path(input_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"input directory not found: {root}")
        for pattern in patterns:
            decks.extend(path for path in root.glob(pattern) if path.is_file())

    unique: Dict[str, Path] = {}
    for deck in decks:
        resolved = deck.resolve()
        unique[str(resolved)] = resolved
    return [unique[key] for key in sorted(unique)]


def _cases_from_decks(decks: Iterable[Path], output_root: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    used: Dict[str, int] = {}
    for deck in decks:
        base = _case_id(deck)
        count = used.get(base, 0) + 1
        used[base] = count
        case_id = base if count == 1 else f"{base}_{count}"
        cases.append(
            {
                "case_id": case_id,
                "input": str(deck),
                "output_dir": str(output_root / case_id),
            }
        )
    return cases


def _cmd_make(args: argparse.Namespace) -> int:
    patterns = args.pattern or list(DEFAULT_PATTERNS)
    decks = _collect_input_decks(args.input, args.input_dir, patterns)
    if not decks:
        raise ValueError("no input decks found")

    output_root = _as_path(args.output_root).resolve()
    output_options = {
        "binary": bool(args.export_binary),
        "json": bool(args.export_json),
        "model_json": bool(args.export_model_json),
        "card_inventory": bool(args.export_card_inventory),
        "vtk": bool(args.export_vtk),
        "hdf5": bool(args.export_hdf5),
        "eigenvalues_only": bool(args.eigenvalues_only),
        "eigenvectors": bool(args.export_eigenvectors),
        "report": not args.no_report,
    }
    manifest = write_batch_manifest(
        args.manifest,
        _cases_from_decks(decks, output_root),
        output_root,
        batch_id=args.batch_id,
        output_options=output_options,
        gc_between=not args.no_gc_between,
        stop_on_error=args.stop_on_error,
    )
    print(manifest)
    return 0


def _cmd_run_once(args: argparse.Namespace) -> int:
    result = run_batch_once(
        args.manifest,
        repo_root=args.repo_root,
        julia=args.julia,
        threads=args.threads,
        quiet=not args.verbose,
        stop_on_error=args.stop_on_error,
        sysimage=args.sysimage,
    )
    return result.returncode


def _cmd_run_worker(args: argparse.Namespace) -> int:
    with JFEMWorker(
        repo_root=args.repo_root,
        julia=args.julia,
        threads=args.threads,
        sysimage=args.sysimage,
    ) as worker:
        response = worker.run_batch(args.manifest)
    print(json.dumps(response, indent=2))
    return 0 if response.get("status") == "ok" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenJFEM manifest-based production workflow helper."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make", help="create a JSON batch manifest from input decks")
    make.add_argument("--manifest", required=True, help="JSON manifest file to create")
    make.add_argument("--output-root", required=True, help="batch output root")
    make.add_argument("--input", action="append", default=[], help="input deck; may be repeated")
    make.add_argument("--input-dir", help="directory containing input decks")
    make.add_argument("--pattern", action="append", help="glob pattern for --input-dir; may be repeated")
    make.add_argument("--batch-id", help="optional batch identifier")
    make.add_argument("--stop-on-error", action="store_true", help="stop the batch at the first failed case")
    make.add_argument("--no-gc-between", action="store_true", help="disable garbage collection between cases")
    make.add_argument("--export-binary", action="store_true", help="write .jfem binary files")
    make.add_argument("--export-json", dest="export_json", action="store_true", default=True, help="write .JU.JSON result files")
    make.add_argument("--no-export-json", dest="export_json", action="store_false", help="skip .JU.JSON result files")
    make.add_argument("--export-model-json", action="store_true", help="write parsed model JSON")
    make.add_argument("--export-card-inventory", action="store_true", help="write parsed card inventory")
    make.add_argument("--export-vtk", action="store_true", help="write VTK output")
    make.add_argument("--export-hdf5", action="store_true", help="write HDF5 output")
    make.add_argument(
        "--eigenvalues-only",
        action="store_true",
        help="SOL 105 optimization mode: write buckling factors but skip full mode-shape expansion and mode-dependent exports",
    )
    make.add_argument(
        "--export-eigenvectors",
        action="store_true",
        help="SOL 105 optimization mode: write buckling eigenvectors/mode shapes to .BUCKLING.JSON",
    )
    make.add_argument("--no-report", action="store_true", help="skip Markdown .REPORT.md files")
    make.set_defaults(func=_cmd_make)

    run_once = sub.add_parser("run-once", help="run one manifest by launching one Julia process")
    run_once.add_argument("manifest", help="JSON manifest to run")
    run_once.add_argument("--repo-root", default=".", help="OpenJFEM repository root")
    run_once.add_argument("--julia", default="julia", help="Julia executable")
    run_once.add_argument("--threads", default="auto", help="Julia thread count")
    run_once.add_argument("--sysimage", help="optional Julia sysimage")
    run_once.add_argument("--stop-on-error", action="store_true", help="override manifest and stop at first failure")
    run_once.add_argument("--verbose", action="store_true", help="send per-case solver output to the console")
    run_once.set_defaults(func=_cmd_run_once)

    run_worker = sub.add_parser("run-worker", help="run one manifest through a JSONL worker")
    run_worker.add_argument("manifest", help="JSON manifest to run")
    run_worker.add_argument("--repo-root", default=".", help="OpenJFEM repository root")
    run_worker.add_argument("--julia", default="julia", help="Julia executable")
    run_worker.add_argument("--threads", default="auto", help="Julia thread count")
    run_worker.add_argument("--sysimage", help="optional Julia sysimage")
    run_worker.set_defaults(func=_cmd_run_worker)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
