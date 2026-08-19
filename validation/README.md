# Canonical validation suite moved

The maintained validation suite now lives in the standalone public repository:

- repository: <https://github.com/jpezzulli/pennyroyal-validation>
- canonical consolidation commit:
  [`2950b2eec7e95c69174fb4950f78886064f08203`](https://github.com/jpezzulli/pennyroyal-validation/commit/2950b2eec7e95c69174fb4950f78886064f08203)
- public baseline release:
  [`public-baseline-2026-08-18`](https://github.com/jpezzulli/pennyroyal-validation/releases/tag/public-baseline-2026-08-18)

This repository no longer maintains a duplicate working copy. Historical
validation source remains available in Git history. The suite first entered
this repository through PR #3 at `93810cd`; the final maintained snapshot here
was PR #17 merge
`0710574f21dc555653a87ee530f4e8ce1d87afdb`, which the standalone
consolidation records as provenance.

Runtime implementation, launch recipes, kernels, and the historical
DeepSeek-V4-Flash result discussion remain in this repository.
