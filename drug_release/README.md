# Drug-release workspace note

The historical `release-foundation` codebase is currently drug-release
first. Most mature release code still lives at the repository root and in
`scripts/`.

This folder exists as a boundary marker:

- future drug-release-only notes, adapters, and migration plans can live here;
- battery trajectory work must not be added to legacy release scripts unless
  the script is explicitly a cross-domain benchmark;
- `D:\battery` material-screening files must not be copied here.

Current drug-release assets include:

- PLGA ODE / `PLGABiphasic`,
- Active Observer scripts,
- CASP / conformal calibration,
- PLGA 321 and LAI/Bannigan datasets,
- liposome IVR intake work,
- chitosan prospective validation.

Before moving root modules into this folder, update `DECISIONS.md` and
respect the architecture contract in `ARCHITECTURE.md`.
