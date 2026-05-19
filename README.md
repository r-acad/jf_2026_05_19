# OpenJFEM

OpenJFEM is a Julia finite-element solver focused on fast linear buckling
analysis for bulk-data structural models. It reads an input deck, builds the
finite-element model, solves the requested case, and writes a run manifest plus
human-readable reports and optional visualization files.

The recommended workflow is intentionally small:

1. Install Julia dependencies once.
2. Precompile with a representative SOL 105 deck.
3. Run SOL 105 cases with Julia threads enabled.
4. Use batch mode for production work so compilation is paid once.

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
|   |-- src/
|   |-- POST/
|   |   |-- postv11.html
|   |   `-- POST_GUIDE.html
|   `-- tools/
|       |-- precompile_sol105.jl
|       `-- testing/
|           |-- run_bdf.jl
|           |-- run_bdf_batch.jl
|           `-- run_manifest.jl
|-- general_description.md
`-- README.md
```

- `JFEM/src`: solver source code.
- `JFEM/tools/precompile_sol105.jl`: one-command SOL 105 precompile setup.
- `JFEM/tools/testing/run_bdf.jl`: preferred single-case runner.
- `JFEM/tools/testing/run_bdf_batch.jl`: preferred batch runner.
- `JFEM/POST/postv11.html`: browser viewer for `.jfem` result files.
- `general_description.md`: broader description of solver capabilities.

## Installation And SOL 105 Precompile

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

Pick one representative SOL 105 deck from the same model family you expect to
run. This one-time step precompiles the solver using that deck and the same
fast flags used later for production runs:

What this does:

- Julia compiles functions when it first sees the specific data types and code
  paths used by a run.
- A SOL 105 buckling case exercises many solver paths: parsing, model building,
  shell assembly, static preload, geometric stiffness assembly, eigenvalue
  setup, and report generation.
- The helper tells OpenJFEM to run one typical SOL 105 deck during package
  precompilation so those hot methods are compiled before the production run.
- The deck should be representative. A tiny or very different deck will not
  exercise the same element, material, property, load, constraint, and buckling
  paths as the cases you run later.
- The step is a speed optimization only. It does not change the model, solver
  equations, buckling load factors, or numerical results.

This step is not required for correctness, but it is the fastest setup for
repeated SOL 105 work because later single-case and batch runs start closer to
a warmed-up Julia session.

The helper below sets the required precompile environment internally and uses
the same fast flags as the run commands.

Windows PowerShell:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\precompile_sol105.jl C:\models\representative_sol105.bdf
```

Linux/macOS Bash:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/precompile_sol105.jl /home/user/models/representative_sol105.bdf
```

## Fast SOL 105 Settings

The commands below assume the precompile step above has already been run. They
use the default fast operating profile:

- `--threads=auto`: lets Julia use available CPU threads for assembly.
- `--startup-file=no`: avoids user startup files changing the run.
- `JFEM_EXPORT_BINARY=false`: skips `.jfem` binary export for maximum solve
  throughput.
- `JFEM_SUPPRESS_THREAD_HINT=1`: keeps batch logs compact.

Use this flag string in both single-case and batch runs:

```text
JFEM_EXPORT_BINARY=false,JFEM_SUPPRESS_THREAD_HINT=1
```

## Run One SOL 105 Case

Use this command for a single buckling deck:

Windows PowerShell:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\testing\run_bdf.jl C:\models\panel_001.bdf JFEM\output\panel_001 "JFEM_EXPORT_BINARY=false,JFEM_SUPPRESS_THREAD_HINT=1"
```

Linux/macOS Bash:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/testing/run_bdf.jl /home/user/models/panel_001.bdf JFEM/output/panel_001 "JFEM_EXPORT_BINARY=false,JFEM_SUPPRESS_THREAD_HINT=1"
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

The report contains the buckling load factors, model counts, active flags, and
solver timing.

## Run a Batch of SOL 105 Cases

For more than one case, use one warm Julia session with a manifest file. This
is the preferred production path.

Create `cases.txt`:

Windows example:

```text
C:\models\panel_001.bdf
C:\models\panel_002.bdf
C:\models\panel_003.bdf
```

Linux/macOS example:

```text
/home/user/models/panel_001.bdf
/home/user/models/panel_002.bdf
/home/user/models/panel_003.bdf
```

Run the batch:

Windows PowerShell:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\testing\run_bdf_batch.jl cases.txt JFEM\output\batch_sol105 "JFEM_EXPORT_BINARY=false,JFEM_SUPPRESS_THREAD_HINT=1" --stop-on-error
```

Linux/macOS Bash:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/testing/run_bdf_batch.jl cases.txt JFEM/output/batch_sol105 "JFEM_EXPORT_BINARY=false,JFEM_SUPPRESS_THREAD_HINT=1" --stop-on-error
```

Batch outputs:

Windows:

```text
JFEM\output\batch_sol105\batch_summary.csv
JFEM\output\batch_sol105\batch_summary.json
JFEM\output\batch_sol105\<case_slug>\run_manifest.json
JFEM\output\batch_sol105\<case_slug>\<case>.REPORT.md
```

Linux/macOS:

```text
JFEM/output/batch_sol105/batch_summary.csv
JFEM/output/batch_sol105/batch_summary.json
JFEM/output/batch_sol105/<case_slug>/run_manifest.json
JFEM/output/batch_sol105/<case_slug>/<case>.REPORT.md
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
- OpenJFEM binary or visualization outputs such as `.jfem`, `.h5`, and `.vtk`
- Julia caches and local temporary files
