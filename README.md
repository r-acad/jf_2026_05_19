# OpenJFEM

OpenJFEM is a Julia finite-element solver focused on fast linear buckling
analysis for bulk-data structural models. It reads an input deck, builds the
finite-element model, solves the requested case, and writes a run manifest plus
human-readable reports and optional visualization files.

The recommended workflow is intentionally small:

1. Install Julia dependencies once.
2. Deploy the fast runtime once with bundled or representative decks.
3. Run single cases or batches with Julia threads enabled.
4. For Python-driven optimization loops, keep one JSONL worker open and submit
   batch manifests repeatedly so Julia startup and compilation are not paid per
   iteration.

## Requirements

- Julia 1.12.x
- Git
- One or more bulk-data input decks, usually with `.bdf`, `.dat`, or `.nas`
  extension

Run all commands from the repository root. The repository tree below uses `/`
as a visual convention. Command examples are written separately for Windows
PowerShell and Linux/macOS Bash, and each command is a single line with no
line-continuation character.

## Repository Layout

```text
.
|-- JFEM/
|   |-- Project.toml
|   |-- Manifest.toml
|   |-- examples/
|   |   |-- manifests/
|   |   `-- precompile/
|   |-- python/
|   |   |-- jfem_client.py
|   |   `-- jfem_manifest_cli.py
|   |-- src/
|   |-- POST/
|   |   |-- postv11.html
|   |   `-- POST_GUIDE.html
|   `-- tools/
|       |-- deploy_fast.jl
|       |-- jfem_worker_jsonl.jl
|       |-- manifest_batch_core.jl
|       |-- precompile_sol105.jl
|       |-- run_batch_manifest.jl
|       |-- sol105_worker.jl
|       `-- testing/
|           |-- run_bdf.jl
|           |-- run_bdf_batch.jl
|           `-- run_manifest.jl
|-- general_description.md
`-- README.md
```

- `JFEM/src`: solver source code.
- `JFEM/tools/deploy_fast.jl`: preferred one-command fast deployment and broad
  precompile setup.
- `JFEM/examples/precompile`: small bundled decks used by `deploy_fast.jl` when
  the user does not provide representative cases.
- `JFEM/examples/manifests`: runnable JSON manifest examples.
- `JFEM/tools/run_batch_manifest.jl`: JSON manifest batch runner for explicit
  input/output mapping.
- `JFEM/tools/jfem_worker_jsonl.jl`: persistent JSONL worker for Python-driven
  optimization loops.
- `JFEM/python/jfem_client.py`: Python 3.8+ stdlib-only helper for writing
  manifests and talking to the JSONL worker.
- `JFEM/python/jfem_manifest_cli.py`: Python command-line helper for creating
  and running manifests from external workflows.
- `JFEM/tools/precompile_sol105.jl`: older focused SOL 105 precompile helper.
- `JFEM/tools/testing/run_bdf.jl`: preferred single-case runner.
- `JFEM/tools/testing/run_bdf_batch.jl`: simple text-list batch runner retained
  for existing scripts. New automation should use `run_batch_manifest.jl`.
- `JFEM/tools/sol105_worker.jl`: human-oriented persistent prompt retained for
  interactive local studies. Python automation should use `jfem_worker_jsonl.jl`.
- `JFEM/POST/postv11.html`: browser viewer for `.jfem` result files.
- `general_description.md`: broader description of solver capabilities.

## Installation And Fast Deployment

Clone the repository and enter it:

Windows PowerShell:

```powershell
git clone <repository-url> OpenJFEM
cd OpenJFEM
```

Linux/macOS Bash:

```bash
git clone <repository-url> OpenJFEM
cd OpenJFEM
```

Run the fast deployment step once after installation or after updating the
solver. With no user-supplied deck, OpenJFEM uses bundled tiny SOL 101, SOL 103,
and SOL 105 decks from `JFEM/examples/precompile` to warm common parser,
assembly, solve, and report paths:

Windows PowerShell:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\deploy_fast.jl
```

Linux/macOS Bash:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/deploy_fast.jl
```

For best performance on a specific model family, add one or more representative
decks:

Windows PowerShell:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\deploy_fast.jl --deck C:\models\representative_sol105.bdf
```

Linux/macOS Bash:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/deploy_fast.jl --deck /home/user/models/representative_sol105.bdf
```

You can also precompile from a JSON batch manifest:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\deploy_fast.jl --manifest C:\models\cases.json
```

What this does:

- Julia compiles functions when it first sees the specific data types and code
  paths used by a run.
- The bundled decks exercise common SOL 101, SOL 103, and SOL 105 paths.
- Representative user decks exercise the exact element, material, property,
  load, constraint, and output paths expected in production.
- The step is a speed optimization only. It does not change the model, solver
  equations, load factors, or numerical results.

Optional sysimage build:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\deploy_fast.jl --sysimage=JFEM\build\OpenJFEM_sysimage.dll --install-packagecompiler
```

The sysimage step uses PackageCompiler if available. It can reduce startup and
package-load time, but it is platform-specific and should be rebuilt after
solver or dependency updates.

## Fast Settings

The commands below assume the precompile step above has already been run. They
use the default fast operating profile:

- `--threads=auto`: lets Julia use available CPU threads for assembly.
- `--startup-file=no`: avoids user startup files changing the run.
- `JFEM_EXPORT_BINARY=false`: skips `.jfem` binary export for maximum solve
  throughput.
- `JFEM_MATRIX_ASYMMETRY_CHECK=false`: skips an expensive buckling diagnostic
  matrix difference in production runs. The stiffness pair is still
  symmetrized before the eigenproblem.
- `JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES=false`: skips the duplicate in-memory
  `mode_shapes` list in the returned Julia dictionary. Report and JSON/VTK/HDF5
  exports still use the raw mode matrix.
- `JFEM_SUPPRESS_THREAD_HINT=1`: keeps batch logs compact.

Use this flag string in direct single-case and text-batch runs:

```text
JFEM_EXPORT_BINARY=false,JFEM_MATRIX_ASYMMETRY_CHECK=false,JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES=false,JFEM_SUPPRESS_THREAD_HINT=1
```

## Run One SOL 105 Case

Use this command for a single buckling deck:

Windows PowerShell:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\testing\run_bdf.jl C:\models\panel_001.bdf JFEM\output\panel_001 "JFEM_EXPORT_BINARY=false,JFEM_MATRIX_ASYMMETRY_CHECK=false,JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES=false,JFEM_SUPPRESS_THREAD_HINT=1"
```

Linux/macOS Bash:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/testing/run_bdf.jl /home/user/models/panel_001.bdf JFEM/output/panel_001 "JFEM_EXPORT_BINARY=false,JFEM_MATRIX_ASYMMETRY_CHECK=false,JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES=false,JFEM_SUPPRESS_THREAD_HINT=1"
```

Read the command from left to right:

- `julia`: starts Julia.
- `--threads=auto`: lets Julia use available CPU threads.
- `--startup-file=no`: ignores any local Julia startup script.
- `--project=.\JFEM` or `--project=./JFEM`: selects the OpenJFEM Julia
  environment.
- `.\JFEM\tools\testing\run_bdf.jl` or
  `./JFEM/tools/testing/run_bdf.jl`: runs one input deck.
- `C:\models\panel_001.bdf` or `/home/user/models/panel_001.bdf`: this is the
  input file to solve. Replace it with your SOL 105 deck.
- `JFEM\output\panel_001` or `JFEM/output/panel_001`: this is the output folder
  created by OpenJFEM for this run.
- the quoted `JFEM_...` string: speed-oriented run flags. Keep the quotes.

The output folder is the second path after the input deck. To write results to
a custom folder, change that argument:

Windows PowerShell:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\testing\run_bdf.jl C:\models\panel_001.bdf D:\jfem_runs\panel_001 "JFEM_EXPORT_BINARY=false,JFEM_MATRIX_ASYMMETRY_CHECK=false,JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES=false,JFEM_SUPPRESS_THREAD_HINT=1"
```

Linux/macOS Bash:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/testing/run_bdf.jl /home/user/models/panel_001.bdf /home/user/jfem_runs/panel_001 "JFEM_EXPORT_BINARY=false,JFEM_MATRIX_ASYMMETRY_CHECK=false,JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES=false,JFEM_SUPPRESS_THREAD_HINT=1"
```

Outputs are written under the output directory:

Windows:

```text
JFEM\output\panel_001\run_manifest.json
JFEM\output\panel_001\panel_001.REPORT.md
```

Linux/macOS:

```text
JFEM/output/panel_001/run_manifest.json
JFEM/output/panel_001/panel_001.REPORT.md
```

For the example above, `panel_001.REPORT.md` is the main result file to open.
It is named from the input file stem: `panel_001.bdf` becomes
`panel_001.REPORT.md`.

The report contains the buckling load factors, model counts, active flags, and
solver timing.

## First Production Interface: JSON Manifests

The first production automation interface is manifest-based. A manifest is a
small JSON file that describes the run before Julia starts solving anything.
It is robust because the external workflow does not depend on shell quoting,
current working directory assumptions, or positional command arguments for
every case.

The same manifest can be used by:

- a command-line batch run;
- a Python optimization loop;
- a job scheduler;
- a future faster worker or sysimage-based deployment.

Required manifest fields:

| Field | Meaning |
|---|---|
| `output_root` | batch-level output folder where summaries are written |
| `cases` | list of cases to solve |
| `cases[].input` | input `.bdf`, `.dat`, or `.nas` deck |

Recommended manifest fields:

| Field | Meaning |
|---|---|
| `batch_id` | readable name for the batch |
| `defaults.flags` | default `JFEM_*` run flags for all cases |
| `defaults.output_options` | result formats to write; use `eigenvalues_only` and `report` for optimization loops |
| `defaults.gc_between` | run garbage collection between cases |
| `defaults.stop_on_error` | stop the batch at the first failed case |
| `cases[].case_id` | stable case name used in summaries |
| `cases[].output_dir` | exact output folder for that case |

You can write the JSON file yourself, or generate it from existing decks with
the included Python helper.

Create a manifest from one directory of decks:

Windows PowerShell:

```powershell
python .\JFEM\python\jfem_manifest_cli.py make --input-dir C:\models --manifest C:\models\cases.json --output-root D:\jfem_runs\batch_001 --batch-id batch_001
```

Linux/macOS Bash:

```bash
python ./JFEM/python/jfem_manifest_cli.py make --input-dir /home/user/models --manifest /home/user/models/cases.json --output-root /home/user/jfem_runs/batch_001 --batch-id batch_001
```

Create a manifest from specific decks:

Windows PowerShell:

```powershell
python .\JFEM\python\jfem_manifest_cli.py make --input C:\models\panel_001.bdf --input C:\models\panel_002.bdf --manifest C:\models\cases.json --output-root D:\jfem_runs\batch_001 --batch-id batch_001
```

Linux/macOS Bash:

```bash
python ./JFEM/python/jfem_manifest_cli.py make --input /home/user/models/panel_001.bdf --input /home/user/models/panel_002.bdf --manifest /home/user/models/cases.json --output-root /home/user/jfem_runs/batch_001 --batch-id batch_001
```

## Run a Batch of SOL 105 Cases

For more than one case, use a JSON batch manifest. This is the preferred
command-line batch interface because every input deck, output folder, run flag,
and output option is explicit.

In JSON files, using `/` as the path separator is valid on Windows, Linux, and
macOS. That keeps the examples easier to read and avoids escaping every
backslash.

Example `cases.json`:

```json
{
  "batch_id": "batch_001",
  "output_root": "D:/jfem_runs/batch_001",
  "defaults": {
    "flags": {
      "JFEM_EXPORT_BINARY": "false",
      "JFEM_MATRIX_ASYMMETRY_CHECK": "false",
      "JFEM_SOL105_STORE_PUBLIC_MODE_SHAPES": "false",
      "JFEM_SUPPRESS_THREAD_HINT": "1"
    },
    "output_options": {
      "binary": false,
      "json": true,
      "report": true
    },
    "gc_between": true,
    "stop_on_error": false
  },
  "cases": [
    {
      "case_id": "panel_001",
      "input": "C:/models/panel_001.bdf",
      "output_dir": "D:/jfem_runs/batch_001/panel_001"
    },
    {
      "case_id": "panel_002",
      "input": "C:/models/panel_002.bdf",
      "output_dir": "D:/jfem_runs/batch_001/panel_002"
    }
  ]
}
```

Run it:

Windows PowerShell:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\run_batch_manifest.jl C:\models\cases.json --quiet
```

Linux/macOS Bash:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/run_batch_manifest.jl /home/user/models/cases.json --quiet
```

Run the included manifest example:

Windows PowerShell:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\run_batch_manifest.jl .\JFEM\examples\manifests\sol105_batch_manifest.json --quiet
```

Linux/macOS Bash:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/run_batch_manifest.jl ./JFEM/examples/manifests/sol105_batch_manifest.json --quiet
```

The batch writes:

```text
<output_root>/batch_summary.csv
<output_root>/batch_summary.json
<output_root>/<case_id>/run_manifest.json
<output_root>/<case_id>/<input_stem>.REPORT.md
<output_root>/<case_id>/<input_stem>.JU.JSON
<output_root>/<case_id>/jfem_case_stdout.log
```

When `output_options.report` is `false`, the `.REPORT.md` file is skipped and
the `report` column in `batch_summary.csv` is left blank for that case.

Use JSON manifests when another program needs exact control of input paths,
output paths, and output options. The `.JU.JSON` file is written when
`"json": true` is present in `output_options`.

For SOL 105 optimization loops that only need buckling load factors, add:

```json
"eigenvalues_only": true
```

inside `output_options`. This skips full mode-shape expansion and disables
mode-dependent exports such as VTK, HDF5, and `.jfem` for that run. Add
`"report": false` when the optimizer reads `.BUCKLING.JSON` or
`batch_summary.csv` directly and does not need `.REPORT.md` files.

## Python Optimization Loop

For heavy optimization, Python should not launch Julia for every case or every
iteration. Run Python from the repository root, or put the repository root on
`PYTHONPATH`, then start one OpenJFEM JSONL worker, keep it warm, and send
batch manifests repeatedly.

Python example:

```python
from pathlib import Path
from JFEM.python.jfem_client import JFEMWorker, write_batch_manifest, load_summary

repo = Path(r"C:\path\to\OpenJFEM")

with JFEMWorker(repo_root=repo, threads="auto") as worker:
    for iteration in range(100):
        output_root = repo / "JFEM" / "output" / f"opt_{iteration:04d}"
        cases = [
            {
                "case_id": f"design_{iteration:04d}",
                "input": rf"C:\opt\decks\design_{iteration:04d}.bdf",
                "output_dir": str(output_root / f"design_{iteration:04d}"),
            }
        ]
        manifest = write_batch_manifest(
            output_root / "cases.json",
            cases,
            output_root,
            batch_id=f"opt_{iteration:04d}",
            output_options={
                "binary": False,
                "json": True,
                "eigenvalues_only": True,
                "report": False,
            },
        )
        response = worker.run_batch(manifest)
        summary = load_summary(response["summary_json"])
        # Read summary/results, update design variables, generate next decks.
```

The JSONL worker keeps stdout as protocol-only JSON. Solver output is written
to each case's `jfem_case_stdout.log`, which makes the protocol safe for Python
parsing.

The example uses `eigenvalues_only=True` and `report=False`, which is the
preferred setting when the optimizer reads buckling factors from JSON/CSV and
does not inspect mode shapes or Markdown reports.

This is the fastest production automation path currently exposed by OpenJFEM:

```text
deploy once -> start JSONL worker once -> Python sends many manifests -> Python reads summaries/results
```

For a simple one-manifest Python-triggered production run, use:

Windows PowerShell:

```powershell
python .\JFEM\python\jfem_manifest_cli.py run-worker C:\models\cases.json --repo-root .
```

Linux/macOS Bash:

```bash
python ./JFEM/python/jfem_manifest_cli.py run-worker /home/user/models/cases.json --repo-root .
```

For heavy optimization, prefer the `JFEMWorker` example above so the same Julia
worker remains open across many design iterations.

To create an eigenvalues-only manifest from the command line:

```powershell
python .\JFEM\python\jfem_manifest_cli.py make --input-dir C:\models --manifest C:\models\cases.json --output-root D:\jfem_runs\batch_001 --eigenvalues-only --no-report
```

## Post-Processing

The fastest commands above skip `.jfem` export. When you need interactive
visual inspection, enable binary export for that run and open:

```text
JFEM/POST/postv11.html
```

The viewer loads `.jfem` result files directly in the browser. See
`JFEM/POST/POST_GUIDE.html` for its controls.

## Reading SOL 105 Results

For buckling cases, inspect the generated report first:

- Positive buckling load factors by subcase.
- Static preload status.
- Geometric-stiffness diagnostics.
- Solver timing and model counts.

The batch summary files provide a compact view of success/failure status and
runtime across all cases.

## Troubleshooting

Package cannot be found:

```text
ERROR: ArgumentError: Package OpenJFEM not found
```

Run the command from the repository root with `--project=.\JFEM` on Windows or
`--project=./JFEM` on Linux/macOS.

First run is slower than later runs:

Julia compiles methods on first use. Run the SOL 105 precompile step with a
representative deck, then use the batch runner for production so any remaining
compilation is paid once for the full set of cases.

No `.jfem` file is written:

The fast commands use `JFEM_EXPORT_BINARY=false`. Set it to `true` only for
runs that need browser visualization output.

Large batches use too much memory:

Keep the default batch behavior, which performs garbage collection between
cases. Avoid changing memory behavior unless you have measured the effect on
your model family.

## Source-Control Policy

The repository tracks source, configuration, documentation, runner scripts, and
the HTML post-processing viewer. Generated solver products are ignored:

- `JFEM/output/`
- solver run products such as `.f04`, `.f06`, `.log`, `.op2`, `.pch`, and
  `.xdb`
- OpenJFEM reports and result exports such as `.REPORT.md`, `.JU.JSON`,
  `.jfem`, `.h5`, and `.vtk`
- Julia caches and local temporary files
