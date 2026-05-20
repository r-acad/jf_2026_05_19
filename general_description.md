# OpenJFEM General Description

OpenJFEM is a Julia-based finite-element analysis code for structural models
defined with bulk-data-style input decks. The code is organized as a complete
analysis pipeline: it reads the input deck, builds a solver-ready finite-element
model, executes the selected solution sequence, and exports reports and
machine-readable results.

The current public workflow emphasizes fast SOL 105 linear buckling analysis,
but the codebase contains broader structural-analysis capabilities, including
linear statics, normal modes, nonlinear statics, sizing optimization, adjoint
sensitivity analysis, and post-processing support.

## High-Level Architecture

The solver follows a staged architecture:

1. **Parsing**: read the input deck, resolve includes, parse case-control and
   bulk-data cards, and build typed card collections.
2. **Model building**: convert parsed cards into a unified model dictionary
   with resolved coordinates, transformed geometry, materials, properties,
   elements, constraints, loads, design variables, and subcase metadata.
3. **Solution**: dispatch to the appropriate solver path based on the requested
   solution type.
4. **Export**: write a markdown report and optional JSON, HDF5, VTK, and
   compact `.jfem` binary result files.
5. **Post-processing**: inspect `.jfem` files in the included browser-based
   viewer.

The main public API is exposed by `OpenJFEM.jl` and includes:

- `main(filename; ...)`: parse, solve, and export in one call.
- `bdf_to_model(filename)`: parse an input deck into a model dictionary.
- `solve_model(model)`: solve an already-built model.
- `export_results(results, filename, output_dir; ...)`: write result files.
- `optimize_thickness` and `optimize_sizing`: run sizing workflows.
- `solve_adjoint` and `solve_adjoint_buckling`: run sensitivity analyses.

## Primary Analysis Capability

The main production focus is **SOL 105 linear buckling**:

- Static preload solution.
- Geometric stiffness assembly.
- Buckling eigenvalue extraction.
- Positive buckling load-factor reporting.
- Mode-shape data export when binary output is enabled.
- EIGRL-style request handling, including mode counts and bounded ranges.
- Range-completeness controls for returning all roots in a requested interval.
- Solver timing and diagnostics in the generated report.

The SOL 105 implementation includes specialized shell-element paths and tuning
flags for fast shell assembly, geometric stiffness assembly, eigen stiffness
assembly, and repeated buckling studies.

## Additional Solution Sequences

OpenJFEM also includes solver paths for:

| Solution | Capability |
|---|---|
| `SOL 101` | Linear static displacement, force, stress, and reaction recovery |
| `SOL 103` / `SOL 63` | Normal modes and frequency extraction |
| `SOL 105` | Linear buckling with static preload and geometric stiffness |
| `SOL 106` | Geometric nonlinear static response path |
| `SOL 200-lite` | Practical sizing optimization subset routed through static or buckling analyses |

The public README focuses on SOL 105 to keep the launch workflow simple, but
the source tree keeps these additional paths available through the Julia API.

## Input Model Coverage

The parser supports common bulk-data formatting styles, including fixed-field,
large-field, free-field, and tab-delimited cards. It resolves nested `INCLUDE`
statements, parses case-control data, and reports unknown card types so users
can see input coverage clearly.

Common input families supported by the parser and model builder include:

| Family | Representative cards |
|---|---|
| Geometry and coordinates | `GRID`, `GRDSET`, `CORD1R`, `CORD2R`, `CORD2C`, `CORD2S` |
| Shell elements | `CQUAD4`, `CQUAD8`, `CTRIA3`, `CTRIA6`, `CSHEAR` |
| Bars, beams, and rods | `CBAR`, `CBEAM`, `CROD`, `CONROD` |
| Springs and bushings | `CELAS1`, `CELAS2`, `CBUSH` |
| Solid elements | `CTETRA`, `CHEXA`, `CPENTA` |
| Mass elements | `CONM1`, `CONM2`, `CMASS1`, `CMASS2`, `PMASS` |
| Rigid and multipoint constraints | `RBE1`, `RBE2`, `RBE3`, `RBAR`, `RSPLINE`, `MPC`, `MPCADD` |
| Properties | `PSHELL`, `PCOMP`, `PSHEAR`, `PBAR`, `PBARL`, `PBEAM`, `PBEAML`, `PROD`, `PELAS`, `PBUSH`, `PSOLID` |
| Materials | `MAT1`, `MAT2`, `MAT8`, `MATT1`, `TABLEM1` |
| Loads | `FORCE`, `MOMENT`, `PLOAD`, `PLOAD1`, `PLOAD2`, `PLOAD4`, `GRAV`, `RFORCE`, `LOAD` |
| Constraints | `SPC`, `SPC1`, `SPCADD` |
| Eigen requests | `EIGRL` |
| Temperature and matrices | `TEMP`, `TEMPD`, `DMIG` |
| Optimization | `DESVAR`, `DRESP1`, `DVPREL1`, `DVMREL1`, `DCONSTR`, `DOPTPRM` |

## Shell And Composite Modeling

The shell implementation is one of the most developed parts of the codebase.
It includes:

- Quadrilateral and triangular shell element support.
- Isotropic shell properties through `PSHELL`.
- Composite laminate properties through `PCOMP`.
- Classical laminate theory processing for composite ABD stiffness terms.
- Transverse-shear treatment for laminate models.
- Material-axis handling for shell and composite elements.
- Curvature, warping, and local-frame utilities for non-flat shell geometry.
- Drilling stiffness, bending, membrane, and shear behavior controls.
- Specialized SOL 105 shell paths for eigen stiffness and geometric stiffness.

These capabilities are concentrated in `FEMKernels.jl`, `ModelBuilder.jl`, and
the solver assembly modules.

## Loads And Boundary Conditions

The solver can assemble common structural load and constraint definitions:

- Concentrated forces and moments.
- Pressure loads on shell and solid regions.
- Gravity and rotating-force style loads.
- Load combinations through `LOAD`.
- Single-point constraints through `SPC`, `SPC1`, and `SPCADD`.
- Multipoint and rigid constraint support through MPC and rigid-element
  families.
- SPC reaction recovery for static-style solutions.

The case-control data selects the active load, constraint, and eigen extraction
definitions by subcase.

## Solver Implementation

The solver stack is organized under `JFEM/src/solver/`:

- `assembly.jl`: stiffness, mass, and geometric-stiffness assembly.
- `solve_case.jl`: solution-sequence dispatch and eigenproblem handling.
- `loads.jl`: load vector construction.
- `constraints.jl` and `boundary_conditions.jl`: constraint application.
- `stress_recovery.jl`: element stress and force recovery.
- `adjoint.jl` and `adjoint_buckling.jl`: sensitivity-analysis paths.
- `optimize_thickness.jl`: sizing and design-variable workflow.
- `SolverCore.jl` and `Solver.jl`: module organization and extension loading.

Large shell assembly is parallelized with Julia threads. The public run
commands use `--threads=auto` so the runtime can use available CPU cores.

## Performance Features

OpenJFEM includes several mechanisms intended for repeated SOL 105 runs:

- Threaded shell stiffness and geometric-stiffness assembly.
- A lean solver bootstrap path for solver-only workflows.
- A full package precompile path using `PrecompileTools`.
- One-command fast deployment with `JFEM/tools/deploy_fast.jl`, driven by
  bundled precompile decks, user representative decks, or JSON batch manifests.
- Batch execution in one Julia process to avoid repeated startup and method
  compilation.
- JSON manifest batch execution with explicit per-case input and output paths.
- A JSONL persistent worker for Python-driven optimization loops.
- A Python manifest CLI for creating production manifests from generated or
  existing decks.
- A SOL 105 preload path that solves only for the static displacement state
  needed by geometric stiffness, avoiding unused static result recovery.
- Production fast flags that skip development-only buckling matrix-asymmetry
  diagnostics and the duplicate public `mode_shapes` list while keeping raw
  mode data available for reports and exports.
- An opt-in SOL 105 eigenvalues-only mode for optimization loops that need
  buckling load factors but not full mode-shape recovery or mode-dependent
  exports.
- Manifest-level report suppression for JSON/CSV-driven optimization loops that
  do not consume Markdown reports.
- Batch summaries that carry `first_eigenvalue` and full `eigenvalues` vectors,
  allowing Python optimization loops to read buckling factors from a single
  summary file.
- Allocation-reduced node force transforms and modal post-processing loops for
  repeated buckling and modal runs.
- Optional PackageCompiler sysimage creation from the deployment helper when
  PackageCompiler is available.
- Optional suppression of `.jfem` binary export for timing-sensitive runs.
- Run manifests that record active `JFEM_*` flags and command arguments.

For production SOL 105 work, the recommended path is:

1. Run `JFEM/tools/deploy_fast.jl` once with bundled or representative decks.
2. Use JSON manifests as the first production automation contract.
3. Run command-line single cases with `run_bdf.jl` and command-line batches
   with `run_batch_manifest.jl`.
4. For optimization loops, start `JFEM/tools/jfem_worker_jsonl.jl` once and
   submit JSON manifests from Python using `JFEM/python/jfem_client.py`.
5. Disable `.jfem` export unless interactive visualization is required.
6. Set `output_options.eigenvalues_only=true` when the outer optimizer only
   needs SOL 105 buckling factors.
7. Set `output_options.report=false` when the outer optimizer reads JSON/CSV
   outputs directly.
8. Read `summary["cases"][i]["first_eigenvalue"]` or `["eigenvalues"]` from
   `batch_summary.json` to avoid reopening every per-case result file.

## Optimization And Sensitivity Analysis

The code includes a practical sizing workflow and adjoint tools:

- Shell-thickness and bar-area design-variable support.
- Design-variable and response extraction from bulk-data-style optimization
  cards.
- Sizing through `DESVAR`, `DVPREL1`, `DRESP1`, `DCONSTR`, and `DOPTPRM`.
- Static compliance sensitivity support.
- Linear-buckling sensitivity support.
- JSON export of adjoint sensitivity results when an `adjoint_config.json`
  file is present next to the input deck.

This capability is intended for lightweight design loops and solver-integrated
optimization studies.

## Exported Results

OpenJFEM writes a markdown report for normal runs. Depending on export flags,
it can also write:

| Format | Use |
|---|---|
| Markdown report | Human-readable analysis summary and diagnostics |
| `.jfem` binary | Compact structural result file for the browser viewer |
| JSON | Scripted result inspection |
| HDF5 | Structured numerical result storage |
| VTK | Visualization in VTK-compatible tools |
| Model JSON | Parsed model inspection |
| Card inventory JSON | Input-coverage auditing |

The markdown report includes model statistics, environment and threading
information, timing breakdowns, and solution-specific result tables.

## Browser Post-Processor

The repository includes an HTML post-processing utility in `JFEM/POST`:

- `postv11.html`: interactive viewer.
- `POST_GUIDE.html`: user guide for the viewer.

The viewer loads `.jfem` files directly in a browser and supports structural
mesh display, subcase selection, result coloring, deformed-shape display, and
side-by-side model inspection.

## Public Runner Scripts

The public command-line runner scripts are:

- `deploy_fast.jl`: preferred fast deployment and broad precompile workflow.
- `run_batch_manifest.jl`: preferred explicit JSON-manifest batch runner.
- `jfem_worker_jsonl.jl`: persistent JSONL worker for Python-driven
  optimization loops.
- `JFEM/python/jfem_client.py`: Python 3.8+ stdlib-only helper for writing
  batch manifests and controlling the JSONL worker.
- `JFEM/python/jfem_manifest_cli.py`: Python command-line helper for creating
  manifests and launching manifest-based runs from external workflows.
- `precompile_sol105.jl`: focused legacy helper for representative SOL 105
  precompile setup.
- `run_bdf.jl`: preferred runner for one case.
- `run_bdf_batch.jl`: simple text-list batch runner retained for existing
  scripts.
- `run_manifest.jl`: shared manifest and environment-flag helper.

Each run writes `run_manifest.json`, recording input path, script path, command
arguments, active flags, Git metadata when available, and output metadata.

## Intended Use

OpenJFEM is best suited for:

- Fast repeated SOL 105 buckling studies.
- Shell and stiffened-panel model families.
- Batch screening of many related structural decks.
- Solver development and inspection through Julia APIs.
- Lightweight sizing and sensitivity workflows tied to structural analyses.
- Browser-based inspection of exported `.jfem` results.

The codebase is intentionally source-transparent: core algorithms, element
kernels, solver assembly, exports, and post-processing are all included in the
repository.
