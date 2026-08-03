# Source and evidence provenance

> **Historical provenance:** this records the earlier three-layer publication
> lineage. See [PROVENANCE.md](../PROVENANCE.md) for the final DSpark-4/1M
> runtime, launcher, model, and evidence identities.

## Repository ownership

The runtime source of truth is the owned `jpezzulli/vllm` fork, whose parent
is `kacper-daftcode/vllm`. The mapped-host feature branch starts at the clean
Runner V2/DSpark source commit `4cb967b451f4a6ae1b284e3d6e3a8ca601c6027e`.
Its validated mapped-host tip is
`98cef19a50765148aba29084dc88da5d16f31700`.
That lineage ultimately applies over official vLLM tag `v0.24.0`
(`ee0da84ab9e04ac7610e28580af62c365e898389`).

The publication repository is the owned `jpezzulli/vLLM-Moet` fork, whose
parent is `kacper-daftcode/vLLM-Moet`. Its mapped-host branch starts after
publication commit `6ba8f224`, which generated the Runner V2/DSpark patch from
vLLM commit `4cb967b451f4a6ae1b284e3d6e3a8ca601c6027e`. Publication commit
`42c8b60f61640fb3cbb17968950914aa9534cf3a` carries the byte-exact patch from
the validated vLLM tip.

vLLM owns:

- `vllm/model_executor/layers/quantization/utils/moe_w2_mapped_host.py`
- the mapped construction, dispatch, and shutdown hooks in
  `moe_w2_cubit.py`
- `tests/model_executor/layers/test_moe_w2_mapped_host.py`

vLLM-MoET owns the generated patch and fingerprints, the validated recipe,
the standalone diagnostic, and the human documentation. The patch is generated
only by `tools/check_patch_files.py --update` from the pinned vLLM source tip.

## Experimental source audit

The successful sealed proof ran from detached vLLM commit
`7878ad5cfce923d0e62f8233ec7a6173c8e0de90`. Its retained worktree had exactly
these source changes after the run:

- modified `moe_w2_cubit.py`
- added `moe_w2_mapped_host.py`
- untracked runtime dependency symlinks under `vllm/third_party/` for
  `deep_gemm`, `fmha_sm100`, and `triton_kernels`

The reviewable implementation was rebuilt on the later clean DSpark tip rather
than committing that detached worktree. The runtime symlinks are local
installation aids and are not part of either branch.

The stopped eager/post-build experiment, its duplicate W2 representation, the
initial hook in the wrong checkpoint-metadata FP8 builder, and the failed 7 GiB
FP4 attempt are not present in the durable code. No host replay, GPU staging
cache, allocator interception, automatic overflow, or host-backed DSpark path
was carried forward.

## Sealed evidence

The immutable proof artifacts remain under
`/opt/ai-artifacts/logs/moet-mapped-w2-step2-20260801-222626`. The standalone
CUDA mechanism proof remains under
`/opt/ai-artifacts/logs/cuda-mapped-host-step1-20260801-220331`; its reusable
source was copied byte-for-byte into `tools/cuda_mapped_host_probe/`.

Local artifact paths are evidence locations on the validation host, not
repository runtime dependencies. See `docs/validation.md` for hashes and the
precise boundary between tested and untested claims.

The clean native reproduction is under
`/opt/ai-artifacts/logs/moet-mapped-w2-clean-validation-20260802-005954`.
Its `SHA256SUMS` hashes to
`9f00c763eb7736852c249b261aa1d4609f032d30185008c4a792c27c5982d901`.
