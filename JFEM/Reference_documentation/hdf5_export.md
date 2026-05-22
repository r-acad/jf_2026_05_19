# HDF5 Export

JFEM can export a compact HDF5 results file with an MSC/Nastran-like
`/NASTRAN` layout. This is intended for post-processing workflows that expect a
Nastran-style HDF5 hierarchy, especially SOL 101, SOL 103, and SOL 105 result
files.

## Scope

The compact HDF5 exporter currently covers:

- SOL 101 static displacement results.
- SOL 103 eigenvalues and eigenvectors.
- SOL 105 static reference displacements, buckling eigenvalues, and buckling
  eigenvectors.
- Input model tables for grids, shell/bar/beam/rod/solid/spring/mass elements,
  rigid elements, materials, properties, constraints, loads, EIGRL, parameters,
  and case-control subcase metadata.
- Result tables under `/NASTRAN/RESULT`, including nodal and elemental table
  paths expected by MSC/Nastran-style readers for the supported probe models.

The old recursive internal-object dump has been removed. HDF5 output is now a
targeted compatibility export rather than a raw dump of JFEM's Julia objects.

## How To Enable It

From Julia:

```julia
using OpenJFEM

OpenJFEM.main(
    "model.bdf";
    output_dir="run_output",
    export_hdf5=true,
    export_json=false,
    export_vtk=false,
    export_jfem_binary=false,
)
```

From a manifest, enable the HDF5 flag in `output_options`:

```json
{
  "cases": [
    {
      "input": "model.bdf",
      "output_dir": "run_output",
      "output_options": {
        "hdf5": true,
        "json": false,
        "vtk": false,
        "binary": false,
        "report": false
      }
    }
  ]
}
```

## Compatibility Status

The HDF5 schema probe deck set compares JFEM HDF5 files against MSC/Nastran
2021 HDF5 files for SOL 101, SOL 103, and SOL 105. The current comparison checks:

- Dataset paths.
- Dataset shapes.
- Compound field names.
- Compound field scalar types.
- Field array dimensions.
- HDF5 attribute names and values.

Current result on the probe set:

```text
SOL 101: 117/117 dataset paths matched, 0 field/type/shape mismatches.
SOL 103:  49/49  dataset paths matched, 0 field/type/shape mismatches.
SOL 105: 122/122 dataset paths matched, 0 field/type/shape mismatches.
```

All table-level `version` attributes now match the MSC/Nastran probe files. The
only remaining attribute differences are the provenance attributes on the
`/NASTRAN` group, such as timestamp, host name, architecture, and solver version.
JFEM intentionally writes truthful OpenJFEM provenance there instead of spoofing
MSC/Nastran metadata.

## Important Limitations

Schema compatibility is not the same as numerical result parity.

For nodal displacement and eigenvector quantities, JFEM exports the quantities
computed by the solver. Some elemental stress, strain, and force result tables
are currently emitted with MSC/Nastran-compatible table structure but placeholder
values where full physical recovery has not yet been implemented for every
element family. These tables are present so readers can open the HDF5 file and
so the schema remains stable while result recovery is expanded.

Before relying on a specific elemental quantity in production, validate both
the table schema and the numerical values against a trusted reference case.

