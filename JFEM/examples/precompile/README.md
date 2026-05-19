# Bundled Precompile Decks

This folder contains tiny redistributable bulk-data decks used by
`JFEM/tools/deploy_fast.jl` when the user does not provide representative
production cases.

The decks are not validation benchmarks. Their purpose is to exercise common
parser, model-building, assembly, solve, and reporting paths during Julia
precompilation so later user runs start from a warmer runtime.

Included decks:

| File | Purpose |
|---|---|
| `sol101_quad_static.bdf` | linear static shell solve path |
| `sol103_quad_modes.bdf` | normal modes shell solve path |
| `sol105_quad_buckling.bdf` | linear buckling shell solve path |

For best performance on a specific production model family, run
`deploy_fast.jl` with one or more representative user decks. The bundled decks
are the safe default when no representative deck is available yet.
