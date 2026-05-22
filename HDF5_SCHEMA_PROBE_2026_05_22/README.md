# HDF5 Schema Probe Decks

These three decks use the same compact finite-element model with different
solution sequences:

- `sol101_hdf5_schema_probe.bdf`
- `sol103_hdf5_schema_probe.bdf`
- `sol105_hdf5_schema_probe.bdf`

The purpose is to generate small Nastran HDF5 result files that expose the
result-tree structure used by the Nastran version available to the user.  These
files are not intended to be high-fidelity validation cases.  They are schema
probes: small enough to inspect, but rich enough to include the result and model
families needed by a future compact JFEM HDF5 exporter.

The model includes:

- GRID points
- CQUAD4 and CTRIA3 shell elements
- PSHELL isotropic shell property
- PCOMP/MAT8 laminated shell property
- CBAR, CBEAM, CROD, and CONROD line elements
- CELAS1/CELAS2 spring elements
- CMASS1/CMASS2 and CONM2 mass elements
- RBE2 and RBE3 interpolation/rigid elements
- CHEXA, CPENTA, and CTETRA solid elements
- MAT1, MAT8, PBAR, PBEAM, PROD, PELAS, PMASS, and PSOLID properties
- SPC constraints, FORCE, MOMENT, and PLOAD4 loads

These decks are configured for MSC/Nastran 2021 HDF5 output using:

```text
MDLPRM,HDF5,1
```

This entry belongs in the Bulk Data section and requests creation of the
MSC/Nastran `.h5` result database.  The result requests include `PLOT` so the
requested data is written to the post-processing database that HyperView reads.

After running Nastran, keep the generated HDF5 files together with these BDF
files.  The next useful step is to inspect the HDF5 trees and compare:

- group names
- dataset names
- dataset shapes
- numeric types
- attributes
- how subcases, modes, eigenvalues, grids, components, and elements are indexed

That information is the input needed to design a compact Nastran-like HDF5
exporter in JFEM.
