"""Small stdlib-only Python client for OpenJFEM batch execution.

The fastest optimization workflow is:

1. Start one JSONL worker process.
2. Generate one JSON manifest per optimization iteration.
3. Send ``{"command": "run_batch", "manifest": "..."}`` to the worker.
4. Read ``batch_summary.json`` or ``batch_summary.csv``.

This module intentionally has no third-party Python dependencies.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union


DEFAULT_FAST_FLAGS: Dict[str, str] = {
    "JFEM_EXPORT_BINARY": "false",
    "JFEM_SUPPRESS_THREAD_HINT": "1",
}


def write_batch_manifest(
    path: Union[str, Path],
    cases: Iterable[Mapping[str, Any]],
    output_root: Union[str, Path],
    *,
    batch_id: Optional[str] = None,
    flags: Optional[Mapping[str, Any]] = None,
    output_options: Optional[Mapping[str, Any]] = None,
    gc_between: bool = True,
    stop_on_error: bool = False,
) -> Path:
    """Write an OpenJFEM JSON batch manifest and return its path.

    Each case must contain at least ``input``. ``case_id`` and ``output_dir`` are
    optional; JFEM will derive stable output folders when they are omitted.
    """

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    merged_flags = dict(DEFAULT_FAST_FLAGS)
    if flags:
        merged_flags.update({str(k): str(v) for k, v in flags.items()})
    manifest: Dict[str, Any] = {
        "output_root": str(output_root),
        "defaults": {
            "flags": merged_flags,
            "output_options": dict(output_options or {"binary": False, "json": True}),
            "gc_between": gc_between,
            "stop_on_error": stop_on_error,
        },
        "cases": [dict(case) for case in cases],
    }
    if batch_id is not None:
        manifest["batch_id"] = str(batch_id)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest_path


def run_batch_once(
    manifest: Union[str, Path],
    *,
    repo_root: Union[str, Path] = ".",
    julia: str = "julia",
    threads: str = "auto",
    quiet: bool = False,
    stop_on_error: bool = False,
    sysimage: Optional[Union[str, Path]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a manifest by launching one Julia process for the full batch."""

    repo = Path(repo_root)
    cmd: List[str] = [
        julia,
        f"--threads={threads}",
        "--startup-file=no",
    ]
    if sysimage is not None:
        cmd.append(f"--sysimage={sysimage}")
    cmd.extend(
        [
            "--project=JFEM",
            "JFEM/tools/run_batch_manifest.jl",
            str(manifest),
        ]
    )
    if quiet:
        cmd.append("--quiet")
    if stop_on_error:
        cmd.append("--stop-on-error")
    return subprocess.run(cmd, cwd=repo, text=True, check=True)


class JFEMWorker:
    """Persistent JSONL worker client."""

    def __init__(
        self,
        *,
        repo_root: Union[str, Path] = ".",
        julia: str = "julia",
        threads: str = "auto",
        flags: Optional[Mapping[str, Any]] = None,
        sysimage: Optional[Union[str, Path]] = None,
        stderr: Any = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        merged_flags = dict(DEFAULT_FAST_FLAGS)
        if flags:
            merged_flags.update({str(k): str(v) for k, v in flags.items()})
        flags_raw = ",".join(f"{k}={v}" for k, v in sorted(merged_flags.items()))
        cmd: List[str] = [
            julia,
            f"--threads={threads}",
            "--startup-file=no",
        ]
        if sysimage is not None:
            cmd.append(f"--sysimage={sysimage}")
        cmd.extend(
            [
                "--project=JFEM",
                "JFEM/tools/jfem_worker_jsonl.jl",
                flags_raw,
            ]
        )
        self.process = subprocess.Popen(
            cmd,
            cwd=self.repo_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr if stderr is not None else sys.stderr,
            text=True,
            bufsize=1,
        )
        ready = self._read_response()
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"OpenJFEM worker did not start cleanly: {ready}")

    def _read_response(self) -> Dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("worker stdout is not available")
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(f"OpenJFEM worker closed unexpectedly, returncode={code}")
        return json.loads(line)

    def request(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if self.process.stdin is None:
            raise RuntimeError("worker stdin is not available")
        self.process.stdin.write(json.dumps(dict(payload)) + "\n")
        self.process.stdin.flush()
        response = self._read_response()
        if response.get("status") == "error":
            raise RuntimeError(response.get("error", "OpenJFEM worker command failed"))
        return response

    def status(self) -> Dict[str, Any]:
        return self.request({"command": "status"})

    def gc(self) -> Dict[str, Any]:
        return self.request({"command": "gc"})

    def run_batch(self, manifest: Union[str, Path], *, request_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"command": "run_batch", "manifest": str(manifest)}
        if request_id is not None:
            payload["id"] = request_id
        return self.request(payload)

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"command": "quit"}) + "\n")
                self.process.stdin.flush()
                self._read_response()
        finally:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)

    def __enter__(self) -> "JFEMWorker":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def load_summary(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "DEFAULT_FAST_FLAGS",
    "JFEMWorker",
    "load_summary",
    "run_batch_once",
    "write_batch_manifest",
]
