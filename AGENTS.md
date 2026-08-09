# AGENTS.md — repository operating contract

This repository publishes vLLM-MoET runtime patches, serving recipes,
documentation, validation suites, kernels, and measured results. Keep changes
reviewable and proportional to what actually changed.

## Runtime and generated-patch contract

- The owned `jpezzulli/vllm` `moet-v0.24.0` branch is the source of truth for
  every vLLM runtime change.
- Never hand-edit `patch/vllm-moet-v0.24.0.patch`, `patch/FILES.txt`, or
  `patch/SOURCE.txt`.
- Regenerate that trio only with:

  ```bash
  VLLM_MOET_FORK=/path/to/vllm-fork \
    python tools/check_patch_files.py --update
  ```

- Commit runtime source first. The publication commit must name the exact
  runtime SHA, and the runtime/publication commits must be pushed together.
- Review every regenerated patch diff. No previously published patch file may
  disappear unless that removal is intentional and documented.
- Preserve unrelated worktree changes and stage explicit paths only. Never
  force-push, rewrite shared history, use `git add -A` in a mixed worktree, or
  edit generated patch artifacts into agreement by hand.

## Proportional validation

Run each unique test set once. Do not rerun overlapping subsets merely to
accumulate pass counts. Record the wall-clock duration and result of every
validation command.

- Runtime code: run focused unit tests for the changed behavior, relevant
  Python compilation/lint checks, and `tools/check_patch_files.py` after the
  runtime commit is regenerated here.
- Patch-only publication: run `tools/check_patch_files.py` with the generating
  runtime checkout available so byte identity is verified.
- Bench recipes, models, matrices, results, or generated README inputs: run
  the corresponding bench lint, tests, and render check. Do not run those
  suites for unrelated documentation or runtime-only changes.
- Launcher or serving configuration: validate shell syntax and the rendered
  command. Perform hardware startup/smoke validation only when the requested
  change requires it.
- Validation harness changes: run its dry-run/replay/integrity tests. Do not
  contact a model unless the task explicitly calls for live collection.

If a required check cannot run, record the exact reason and keep the claim
boundary narrow.

## Documentation and evidence

- Update `README.md` whenever public runtime behavior, configuration, startup,
  measured performance, validation status, or a known limitation changes.
- Preserve historical benchmark and qualification claims unless their exact
  suites are rerun. New smoke tests do not replace old qualification evidence.
- Report configured admission, runtime-reported capacity, and exercised
  context as different facts.
- Do not commit model files, expert packs, credentials, ad-hoc logs, or local
  artifact paths. Put reproducible public recipes and concise evidence in the
  repository; retain bulky raw artifacts outside Git.

## Repository map

- `patch/`: sanctioned generated runtime patch and source fingerprints.
- `scripts/` and `bench/recipes/`: reproducible launch/build configuration.
- `validation/`: public cases, fixtures, graders, and replay-safe checks.
- `bench/`: benchmark definitions and published result rendering.
- `docs/`, `README.md`, `BUILD-AND-RUN.md`, `VALIDATION.md`: public behavior,
  architecture, operation, provenance, and claim boundaries.
- `kernels/`: MoET kernel sources, generated cubins, and manifests.

Commit messages should state what changed, why, the exact runtime SHA for a
patch regeneration, and the focused validation evidence.
