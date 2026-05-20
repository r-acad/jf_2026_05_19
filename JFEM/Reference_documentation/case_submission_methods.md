# JFEM Case Submission Methods

This document describes the supported ways to submit one case or many cases to
JFEM. It is intended as a practical reference for users, Python automation, and
solver development.

The recommended method depends on the workflow:

| Workflow | Recommended method |
|---|---|
| Heavy Python optimization loop | Persistent JSONL worker |
| Command-line production batch | JSON manifest batch runner |
| Command-line production single case | JSON manifest batch runner with one case |
| Quick manual smoke test | Single-case runner |
| Solver development and debugging | Direct Julia API |
| Legacy directory/list batch | Legacy simple batch runner |

All commands below assume they are run from the repository root:

```powershell
cd C:\path\to\jf_2026_05_19
```

On Linux, use the same commands with `/` paths and `--project=./JFEM`.

## Common Concepts

### Input Deck

The input file is normally a `.bdf`, `.dat`, or `.nas` bulk data deck.

Example:

```text
C:\models\panel_001.bdf
```

### Output Directory

Every case should write to a dedicated output directory.

Example:

```text
D:\jfem_runs\panel_001
```

For SOL 105 buckling runs with JSON output enabled, the main result file is:

```text
<input-file-stem>.BUCKLING.JSON
```

For example:

```text
panel_001.BUCKLING.JSON
```

### Useful SOL 105 Output Options

For buckling optimization, the most important output choices are:

| Option | Meaning |
|---|---|
| `json=true` | Write structured result JSON. |
| `binary=false` | Skip `.jfem` binary output. This is faster when post-processing is not needed. |
| `report=false` | Skip Markdown report generation. This is faster for automated loops. |
| `eigenvalues_only=true` | Fastest SOL 105 path when only buckling factors are needed. Omits mode shapes. |
| `eigenvectors=true` | Writes buckling eigenvectors/mode shapes to `.BUCKLING.JSON`. |

Use `eigenvalues_only=true` when the optimizer only needs buckling load
factors.

Use `eigenvectors=true` when the optimizer also needs mode shapes/eigenvectors.
When `eigenvectors=true` is requested, it takes priority over
`eigenvalues_only=true` because eigenvectors cannot be exported if mode-shape
recovery is disabled.

### Common Fast Flags

The runners and Python helper use these fast defaults where appropriate:

```text
JFEM_EXPORT_BINARY=false
JFEM_MATRIX_ASYMMETRY_CHECK=false
JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES=false
JFEM_SUPPRESS_THREAD_HINT=1
```

For eigenvector export, do not suppress public mode shapes unless the manifest
explicitly requests `eigenvectors=true`, because that option handles the needed
mode-shape output path.

## Method 1: Direct Julia API

The lowest-level way to run JFEM is to call the Julia API directly.

```julia
using OpenJFEM

OpenJFEM.main(
    "C:/models/panel_001.bdf";
    output_dir="D:/jfem_runs/panel_001",
    export_json=true,
    export_jfem_binary=false,
    export_report=false,
)
```

### Characteristics

| Characteristic | Value |
|---|---|
| Best for | Development, debugging, custom Julia scripts |
| Startup cost | Paid each time unless inside a persistent Julia session |
| Batch support | Manual, by writing a Julia loop |
| Output control | Full |
| Python integration | Indirect, normally through subprocess or Julia/Python bridge packages |
| Recommended for production automation | No, unless wrapped carefully |

### Notes

The direct API gives full control over the solver call, but it is not the most
convenient production interface. Long Julia `-e` commands are fragile on
Windows PowerShell because quote escaping is easy to get wrong. For command-line
production use, prefer the JSON manifest runner.

## Method 2: Single-Case Command Runner

Script:

```text
JFEM/tools/testing/run_bdf.jl
```

Basic Windows example:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\testing\run_bdf.jl C:\models\panel_001.bdf D:\jfem_runs\panel_001 "JFEM_EXPORT_BINARY=false,JFEM_MATRIX_ASYMMETRY_CHECK=false,JFEM_SUPPRESS_THREAD_HINT=1"
```

Basic Linux example:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/testing/run_bdf.jl /home/user/models/panel_001.bdf /home/user/jfem_runs/panel_001 "JFEM_EXPORT_BINARY=false,JFEM_MATRIX_ASYMMETRY_CHECK=false,JFEM_SUPPRESS_THREAD_HINT=1"
```

Argument order:

```text
run_bdf.jl <input-deck> <output-directory> [FLAG=value,FLAG=value]
```

### Characteristics

| Characteristic | Value |
|---|---|
| Best for | Quick manual single-case runs and smoke tests |
| Startup cost | One Julia startup per case |
| Batch support | No |
| Output control | Environment flags plus default exporter behavior |
| Summary output | Per-case `run_manifest.json` |
| Recommended for production automation | Only for simple single-case workflows |

### Output

The runner creates the requested output directory and writes a
`run_manifest.json` file recording:

- input deck path,
- input deck hash,
- Julia version,
- thread count,
- git information,
- applied `JFEM_*` flags,
- output directory.

### When To Use

Use this when you want to check that one deck runs correctly from the command
line. For SOL 105 optimization work, the JSON manifest runner is usually better
because it gives more explicit control over `eigenvalues_only`, `eigenvectors`,
JSON output, reports, and per-case summaries.

## Method 3: JSON Manifest Batch Runner

Script:

```text
JFEM/tools/run_batch_manifest.jl
```

This is the preferred command-line interface for both one case and many cases.

Run command:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\run_batch_manifest.jl C:\models\cases.json --quiet
```

Linux:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/run_batch_manifest.jl /home/user/models/cases.json --quiet
```

### Manifest Structure

Minimal one-case manifest:

```json
{
  "batch_id": "panel_001_run",
  "output_root": "D:/jfem_runs/panel_001_run",
  "defaults": {
    "flags": {
      "JFEM_EXPORT_BINARY": "false",
      "JFEM_MATRIX_ASYMMETRY_CHECK": "false",
      "JFEM_SUPPRESS_THREAD_HINT": "1"
    },
    "output_options": {
      "binary": false,
      "json": true,
      "report": false
    },
    "gc_between": true,
    "stop_on_error": false
  },
  "cases": [
    {
      "case_id": "panel_001",
      "input": "C:/models/panel_001.bdf",
      "output_dir": "D:/jfem_runs/panel_001_run/panel_001"
    }
  ]
}
```

Multiple cases:

```json
{
  "batch_id": "panel_family_001",
  "output_root": "D:/jfem_runs/panel_family_001",
  "defaults": {
    "flags": {
      "JFEM_EXPORT_BINARY": "false",
      "JFEM_MATRIX_ASYMMETRY_CHECK": "false",
      "JFEM_SUPPRESS_THREAD_HINT": "1"
    },
    "output_options": {
      "binary": false,
      "json": true,
      "report": false
    },
    "gc_between": true,
    "stop_on_error": false
  },
  "cases": [
    {
      "case_id": "panel_001",
      "input": "C:/models/panel_001.bdf"
    },
    {
      "case_id": "panel_002",
      "input": "C:/models/panel_002.bdf"
    },
    {
      "case_id": "panel_003",
      "input": "C:/models/panel_003.bdf"
    }
  ]
}
```

If `output_dir` is omitted for a case, JFEM derives one under `output_root`.

### SOL 105: Fast Buckling Factors Only

Use this when only eigenvalues/buckling load factors are needed:

```json
"output_options": {
  "binary": false,
  "json": true,
  "eigenvalues_only": true,
  "report": false
}
```

This is the fastest SOL 105 option because it skips public mode-shape recovery
and mode-dependent exports.

### SOL 105: Buckling Factors And Eigenvectors

Use this when the optimizer needs eigenvectors/mode shapes:

```json
"output_options": {
  "binary": false,
  "json": true,
  "eigenvectors": true,
  "report": false
}
```

The batch summary will include:

```json
"result_json": "D:/jfem_runs/panel_family_001/panel_001/panel_001.BUCKLING.JSON",
"mode_shape_count": 3,
"mode_shapes_available": true
```

The eigenvectors are written in the per-case `.BUCKLING.JSON` file under:

```text
modes[].mode_shape
```

### Runner Options

| Option | Meaning |
|---|---|
| `--quiet` | Redirect per-case solver output to `jfem_case_stdout.log` instead of stdout. Recommended for automation. |
| `--stop-on-error` | Abort the batch after the first failed case. |

### Batch Summary

The manifest runner writes:

```text
batch_summary.json
batch_summary.csv
```

For modal and buckling cases, these include:

- `sol_type`,
- `eigenvalue_count`,
- `first_eigenvalue`,
- `eigenvalues`,
- `mode_shape_count`,
- `mode_shapes_available`,
- `result_json`,
- `wall_s`,
- `status`,
- `log`,
- `error`.

### Characteristics

| Characteristic | Value |
|---|---|
| Best for | Command-line production runs |
| Startup cost | One Julia startup for the full batch |
| Batch support | Yes |
| Output control | Explicit and reproducible |
| Summary output | Batch JSON and CSV |
| Python integration | Good, because Python can generate manifests |
| Recommended for production automation | Yes |

## Method 4: Python Generates A Manifest And Launches One Batch

Python helper:

```text
JFEM/python/jfem_client.py
```

The helper has no third-party Python dependencies.

Example:

```python
from pathlib import Path

from jfem_client import load_summary, run_batch_once, write_batch_manifest

repo = Path(r"C:\path\to\jf_2026_05_19")

manifest = write_batch_manifest(
    r"D:\jfem_runs\iteration_001\cases.json",
    cases=[
        {
            "case_id": "panel_001",
            "input": r"C:\models\panel_001.bdf",
        },
        {
            "case_id": "panel_002",
            "input": r"C:\models\panel_002.bdf",
        },
    ],
    output_root=r"D:\jfem_runs\iteration_001",
    output_options={
        "binary": False,
        "json": True,
        "eigenvectors": True,
        "report": False,
    },
)

run_batch_once(
    manifest,
    repo_root=repo,
    threads="auto",
    quiet=True,
)

summary = load_summary(r"D:\jfem_runs\iteration_001\batch_summary.json")
first_case = summary["cases"][0]
print(first_case["first_eigenvalue"])
print(first_case["result_json"])
```

### Characteristics

| Characteristic | Value |
|---|---|
| Best for | Python workflows that submit batches occasionally |
| Startup cost | One Julia startup per batch |
| Batch support | Yes |
| Output control | Manifest-based |
| Summary output | Batch JSON and CSV |
| Recommended for heavy optimization loops | Acceptable, but persistent worker is faster |

### When To Use

Use this when Python generates a set of decks, launches JFEM once for the set,
then reads the batch summary. If Python repeats this many times in a tight
optimization loop, use the persistent worker instead.

## Method 5: Persistent JSONL Worker

Script:

```text
JFEM/tools/jfem_worker_jsonl.jl
```

Python helper class:

```text
JFEM/python/jfem_client.py
```

The worker starts Julia once and then accepts JSON commands on stdin. It writes
one JSON response per line on stdout. Solver output goes to per-case log files,
which keeps the protocol safe for Python parsing.

Python example:

```python
from pathlib import Path

from jfem_client import JFEMWorker, load_summary, write_batch_manifest

repo = Path(r"C:\path\to\jf_2026_05_19")

with JFEMWorker(repo_root=repo, threads="auto") as worker:
    for iteration in range(100):
        output_root = Path(r"D:\jfem_runs") / f"iteration_{iteration:04d}"
        manifest = write_batch_manifest(
            output_root / "cases.json",
            cases=[
                {
                    "case_id": "design_001",
                    "input": rf"D:\optimizer_models\iteration_{iteration:04d}\design_001.bdf",
                }
            ],
            output_root=output_root,
            output_options={
                "binary": False,
                "json": True,
                "eigenvectors": True,
                "report": False,
            },
        )

        response = worker.run_batch(manifest, request_id=f"iter-{iteration:04d}")
        summary = load_summary(response["summary_json"])

        case = summary["cases"][0]
        eigenvalues = case["eigenvalues"]
        result_json = case["result_json"]

        # Use eigenvalues and result_json to update design variables.
```

### Worker Commands

The JSONL protocol supports:

```json
{"command": "status"}
{"command": "run_batch", "manifest": "C:/path/cases.json"}
{"command": "gc"}
{"command": "quit"}
```

The worker also accepts manifest data directly:

```json
{
  "command": "run_batch",
  "manifest_data": {
    "output_root": "D:/jfem_runs/iteration_001",
    "cases": [
      {
        "case_id": "panel_001",
        "input": "C:/models/panel_001.bdf"
      }
    ]
  }
}
```

Writing a manifest file is usually easier to debug and leaves a useful audit
trail, so the file-based manifest path is preferred for production.

### Characteristics

| Characteristic | Value |
|---|---|
| Best for | Heavy Python optimization loops |
| Startup cost | Paid once for the whole worker session |
| Batch support | Yes |
| Output control | Manifest-based |
| Summary output | Batch JSON and CSV |
| Python integration | Best available production path |
| Recommended for heavy optimization loops | Yes |

### Why This Is Fast

The persistent worker avoids repeating:

- Julia process startup,
- package loading,
- method compilation,
- solver warmup,
- repeated environment initialization.

For optimization loops that generate, run, read, modify, and rerun many decks,
this is the preferred production architecture.

## Method 6: Legacy Simple Batch Runner

Script:

```text
JFEM/tools/testing/run_bdf_batch.jl
```

This runner accepts:

- one deck,
- a directory containing decks,
- a text or CSV file with one deck path per line or in the first column.

Directory example:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\testing\run_bdf_batch.jl C:\models D:\jfem_runs "JFEM_EXPORT_BINARY=false,JFEM_SUPPRESS_THREAD_HINT=1" --recursive
```

Simple list example:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\testing\run_bdf_batch.jl C:\models\deck_list.txt D:\jfem_runs "JFEM_EXPORT_BINARY=false,JFEM_SUPPRESS_THREAD_HINT=1"
```

Options:

| Option | Meaning |
|---|---|
| `--recursive` | Recurse through subdirectories when input is a directory. |
| `--stop-on-error` | Stop after the first failed case. |
| `--dry-run` | List cases and output directories without solving. |
| `--ext=.bdf,.dat,.nas` | Select deck extensions. |
| `--no-gc-between` | Skip explicit garbage collection between cases. |

### Characteristics

| Characteristic | Value |
|---|---|
| Best for | Existing simple scripts and quick directory sweeps |
| Startup cost | One Julia startup for the batch |
| Batch support | Yes |
| Output control | Less explicit than JSON manifests |
| Summary output | Simple batch summary |
| Recommended for new automation | No, prefer JSON manifests |

### When To Use

Use this if you already have a directory of decks and want a quick sweep. For
new production workflows, use the JSON manifest runner because it records output
options more clearly and is easier to reproduce from Python.

## Method 7: Sysimage Or Precompiled Deployment

This is not a separate submission interface. It is a speed layer that can be
combined with:

- the JSON manifest batch runner,
- the persistent JSONL worker,
- Python `run_batch_once`,
- Python `JFEMWorker`.

The purpose is to reduce Julia compilation and startup overhead.

### Characteristics

| Characteristic | Value |
|---|---|
| Best for | Repeated command-line runs and production workers |
| Startup cost | Reduced |
| Batch support | Depends on the runner used |
| Output control | Depends on the runner used |
| Recommended for production | Yes, when deployment time is acceptable |

### Notes

The largest speed benefit for heavy optimization usually comes from the
persistent worker. A sysimage can further improve startup and first-run latency,
but it does not replace the worker. The fastest production setup is typically:

```text
Python optimizer -> persistent JFEM worker -> manifest batch execution
```

with a precompiled package cache or sysimage when available.

## Choosing The Right Method

### For One Manual Case

Use the JSON manifest runner if you want fully controlled output.

Use `run_bdf.jl` if you want the shortest quick test command.

### For A Command-Line Batch

Use:

```text
JFEM/tools/run_batch_manifest.jl
```

This gives one Julia process for the whole batch and writes batch summaries.

### For Python Optimization

Use:

```text
JFEM/python/jfem_client.py
JFEM/tools/jfem_worker_jsonl.jl
```

The optimizer should:

1. Generate input decks.
2. Write a manifest for the current iteration.
3. Send the manifest to the persistent JFEM worker.
4. Read `batch_summary.json`.
5. Read per-case `.BUCKLING.JSON` files only when eigenvectors or detailed
   mode data are required.
6. Update design variables.
7. Repeat without restarting Julia.

### For Buckling Factors Only

Use:

```json
"output_options": {
  "binary": false,
  "json": true,
  "eigenvalues_only": true,
  "report": false
}
```

Read:

```text
batch_summary.json -> cases[].first_eigenvalue
batch_summary.json -> cases[].eigenvalues
```

### For Buckling Factors And Eigenvectors

Use:

```json
"output_options": {
  "binary": false,
  "json": true,
  "eigenvectors": true,
  "report": false
}
```

Read:

```text
batch_summary.json -> cases[].result_json
<case>.BUCKLING.JSON -> modes[].mode_shape
```

## Practical Recommendations

1. Use JSON manifests for every production run, even for one case.
2. Use the persistent JSONL worker for Python-driven optimization.
3. Use `--quiet` for automated batches so stdout remains readable.
4. Keep one output directory per case.
5. Keep `batch_summary.json` and `run_manifest.json`; they are the audit trail.
6. Skip reports, binary export, VTK, and HDF5 unless they are needed.
7. Request `eigenvectors=true` only when the optimizer really needs mode shapes.
8. Request `eigenvalues_only=true` for the fastest SOL 105 factor-only loop.
9. Use Julia threads with `--threads=auto`.
10. Add a sysimage/precompiled deployment once the workflow is stable.

## Summary Table

| Method | One case | Batch | Python-friendly | Fast for repeated optimization | Best use |
|---|---:|---:|---:|---:|---|
| Direct Julia API | Yes | Manual | Medium | Medium inside persistent Julia | Development |
| `run_bdf.jl` | Yes | No | Low | Low | Quick smoke test |
| `run_batch_manifest.jl` | Yes | Yes | High | Medium | Production CLI |
| Python `run_batch_once` | Yes | Yes | High | Medium | Python batch launcher |
| Persistent JSONL worker | Yes | Yes | Very high | Very high | Heavy optimization |
| `run_bdf_batch.jl` | Yes | Yes | Medium | Medium | Legacy/simple sweeps |
| Sysimage/precompile | N/A | N/A | N/A | Improves all compatible runners | Deployment speed layer |
