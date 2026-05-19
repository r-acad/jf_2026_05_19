# Example Manifests

This folder contains small JSON manifests that demonstrate the first production
OpenJFEM execution contract.

The manifest is the boundary between external automation and the solver. A
Python optimizer, shell script, job scheduler, or human command line can all
produce the same manifest format and run it with the same OpenJFEM tools.

Run the bundled SOL 105 example from the repository root:

Windows PowerShell:

```powershell
julia --threads=auto --startup-file=no --project=.\JFEM .\JFEM\tools\run_batch_manifest.jl .\JFEM\examples\manifests\sol105_batch_manifest.json --quiet
```

Linux/macOS Bash:

```bash
julia --threads=auto --startup-file=no --project=./JFEM ./JFEM/tools/run_batch_manifest.jl ./JFEM/examples/manifests/sol105_batch_manifest.json --quiet
```

The example writes results under:

```text
JFEM/output/example_sol105_manifest
```
