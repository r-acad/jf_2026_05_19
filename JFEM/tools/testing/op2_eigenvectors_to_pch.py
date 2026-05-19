#!/usr/bin/env python
"""Export MSC Nastran OP2 eigenvectors to a simple PUNCH-like text file.

The companion Julia MAC tool intentionally accepts a small subset of PUNCH/F06
syntax. MSC Nastran can write SOL105 buckling vectors into OP2 even when OFP
refuses to print OUGV2, so this utility bridges that result back into the same
text parser.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
    from cpylog import get_logger
    from pyNastran.op2.op2 import OP2
except ImportError as exc:  # pragma: no cover - exercised by users without pyNastran
    raise SystemExit(
        "pyNastran is required. Install it in the active Python environment with "
        "`python -m pip install pyNastran`."
    ) from exc


GRID_TYPE_NAMES = {
    1: "G",
    2: "S",
    3: "E",
    4: "M",
    5: "P",
    7: "L",
    10: "H",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OP2 eigenvectors to the PUNCH-like format used by compare_mode_mac.jl."
    )
    parser.add_argument("op2", type=Path, help="Input OP2 file")
    parser.add_argument("out", type=Path, help="Output text/PUNCH file")
    parser.add_argument(
        "--subcase",
        action="append",
        type=int,
        default=None,
        help="Only export this subcase. May be supplied more than once.",
    )
    parser.add_argument(
        "--max-modes",
        type=int,
        default=None,
        help="Export at most this many modes per subcase.",
    )
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{float(value): .9E}"


def mode_eigenvalues(obj: object, count: int) -> list[float | None]:
    for name in ("eigrs", "eigns", "eigenvalues"):
        vals = getattr(obj, name, None)
        if vals is not None:
            out = [float(v) for v in vals]
            if len(out) >= count:
                return out[:count]
    return [None] * count


def iter_sorted_eigenvectors(eigenvectors: dict) -> Iterable[tuple[tuple, object]]:
    def subcase_of(item: tuple[tuple, object]) -> int:
        key, obj = item
        isubcase = getattr(obj, "isubcase", None)
        if isubcase is not None:
            return int(isubcase)
        if isinstance(key, tuple) and key:
            return int(key[0])
        return int(key)

    return sorted(eigenvectors.items(), key=subcase_of)


def export_op2_eigenvectors(op2_path: Path, out_path: Path, subcases: set[int] | None, max_modes: int | None) -> int:
    # MSC OP2s from SOL105 can carry duplicate static/eigenvector result keys.
    # pyNastran logs those collisions at error level even though the OUGV1
    # eigenvector blocks are still populated correctly, so keep the converter
    # quiet unless pyNastran raises a real exception.
    log = get_logger(log=None, level="critical")
    model = OP2(debug=False, log=log)
    model.read_op2(str(op2_path), build_dataframe=False)

    if not model.eigenvectors:
        raise SystemExit(f"No eigenvectors were found in {op2_path}")

    exported = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("$EIGENVECTOR\n")
        out.write("$REAL OUTPUT\n")
        out.write(f"$SOURCE OP2 = {op2_path}\n")

        for key, obj in iter_sorted_eigenvectors(model.eigenvectors):
            isubcase = int(getattr(obj, "isubcase", key[0] if isinstance(key, tuple) else key))
            if subcases is not None and isubcase not in subcases:
                continue

            data = getattr(obj, "data", None)
            node_gridtype = getattr(obj, "node_gridtype", None)
            if data is None or node_gridtype is None:
                continue

            nmodes = int(data.shape[0])
            if max_modes is not None:
                nmodes = min(nmodes, max_modes)
            evals = mode_eigenvalues(obj, nmodes)

            for imode in range(nmodes):
                mode_id = imode + 1
                eig = evals[imode]
                out.write("$EIGENVECTOR\n")
                out.write(f"$SUBCASE ID = {isubcase}\n")
                if eig is not None:
                    out.write(f"$EIGENVALUE = {fmt(eig).strip()}\n")
                out.write(f"$EIGENVECTOR = {mode_id}\n")

                mode_data = data[imode]
                for inode, (nid, grid_type) in enumerate(node_gridtype):
                    label = GRID_TYPE_NAMES.get(int(grid_type), str(int(grid_type)))
                    values = mode_data[inode]
                    out.write(
                        f"{int(nid):8d} {label:<2s} "
                        f"{fmt(values[0])} {fmt(values[1])} {fmt(values[2])}\n"
                    )
                    out.write(
                        f"-CONT-    "
                        f"{fmt(values[3])} {fmt(values[4])} {fmt(values[5])}\n"
                    )
                exported += 1

    return exported


def main() -> None:
    args = parse_args()
    if not args.op2.is_file():
        raise SystemExit(f"OP2 file not found: {args.op2}")
    if args.max_modes is not None and args.max_modes < 1:
        raise SystemExit("--max-modes must be positive")

    count = export_op2_eigenvectors(
        args.op2,
        args.out,
        set(args.subcase) if args.subcase else None,
        args.max_modes,
    )
    print(f"exported {count} eigenvector blocks to {args.out}")


if __name__ == "__main__":
    main()
