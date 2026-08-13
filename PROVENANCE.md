# Provenance and ownership

## Attribution

This work is built on
[`kacper-daftcode/vLLM-Moet`](https://github.com/kacper-daftcode/vLLM-Moet)
and its companion
[`kacper-daftcode/vllm`](https://github.com/kacper-daftcode/vllm) lineage.

Kacper created the SM120 runtime, hand-written W2 execution kernels, 2-bit
expert representation, FP4 correction/recovery system, Runner V2 and DSpark
W2 integration, persistence and replay machinery, and the broader model,
multi-GPU, host/NVMe tier, and benchmarking foundation.

This fork adds the narrow complete-layer mapped-host W2 path: canonical
construction in NUMA-local pinned memory, UVA and page-placement validation,
stable graph-lifetime ownership, accounting, rollback/cleanup tests, the
single-RTX-PRO serving recipe and scripts, and the validation documented here.
It also integrates MoET's existing planes and delta formats into a bounded
direct startup loader.

## Exact source identities

| Component | Identity | Role |
|---|---|---|
| Official vLLM base | `vllm-project/vllm@ee0da84ab9e04ac7610e28580af62c365e898389` (`v0.24.0`) | original release lineage |
| Kacper runtime base for final mapped-host commits | `kacper-daftcode/vllm@960abee41` | current Runner V2/DSpark/W2 foundation used by the final branch |
| Mapped-host implementation | `jpezzulli/vllm@f80b8ec884a3f455a900ba6c57ad4633b4ded0b2` | canonical mapped-host W2 source |
| Hardware-validated direct loader | `jpezzulli/vllm@752637ef1a78787e84bb33efe835b4c0b06e4918` | measured cache generation, direct startup, geometry, and smoke requests |
| Final runtime tip | `jpezzulli/vllm@9b8460737e77c2f826dc5b5d9d918fe91a240feb` | reconciled runtime plus layer-scoped FP4 correction exclusion |
| Current publication branch | `jpezzulli/vLLM-Moet:rtx-pro6000` | generated patch and serving recipes for runtime `9b846073` |
| Maintained MoET publication ancestry | `kacper-daftcode/vLLM-Moet@c03f5d303597fbd23b3659ae3b532bfe6634a082` | merged as a real parent before Penny publication changes |
| Historical OCI packaging | `jpezzulli/vLLM-Moet@0544e69e63dce5a9cf597797df3db140391ba832` | sealed v4 image built from runtime 95ef4a88 |
| DeepGEMM | `deepseek-ai/DeepGEMM@a6b593d2826719dcf4892609af7b84ee23aaf32a` | validated DeepGEMM build |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` | official 0731 model |

The published vLLM branch is `jpezzulli/vllm:moet-v0.24.0` at
`9b8460737e77c2f826dc5b5d9d918fe91a240feb`.

## Runtime versus generated-patch lineage

The final DSpark-4/1M direct-cache startup was measured at `752637ef1`. The
later runtime line adds the tested builder-scope guard, upstream
reconciliation, and the layer-scoped FP4 correction exclusion; the current
`rtx-pro6000` publication patch is generated from `9b846073`. The
published v4 OCI image is a separate historical artifact: its sealed source
used runtime `95ef4a88c63c9ed88f2384977e05d788897af6c3` and publication
packaging `0544e69e63dce5a9cf597797df3db140391ba832`. The current patch must not
be presented as the source of that already-published image.

The generated patch trio is updated only through `tools/check_patch_files.py`
and records runtime source `9b846073`; it is not edited by hand.

## Production launcher identity

The repository production launcher SHA-256 is:

```text
6280dc3040c2419cc20655169337800c1cf90951712ebc6171cc9f6b4b2aab14
```

The promoted host launcher has SHA-256
`a4f8447f140c9724e25b278b315134778f7a7e702cf4a3421864b437735a432b`.
It resolves the same serving geometry but uses the host's established absolute
environment paths rather than the repository script's path variables.

It selects mapped target layer 42 plus mapped DSpark layers 43–45, DSpark-4,
a target-only 6 GiB FP4 correction tier, FP8 MLA KV, four sequences, 2,048
maximum batched tokens, full and piecewise graphs, DeepGEMM,
`gpu_memory_utilization=0.98446`, and a 524,288-token admission limit.

The repository launcher exposes model path, served name, port, cache, pack,
and audit locations as environment variables while keeping that validated
geometry fixed.

## August 9, 2026 maintenance lineage

Upstream-derived adaptations in the runtime are:

- `6dc88e63e`, the packed KV-block zeroing repair from vLLM `d6af803` /
  PR #50276;
- `41cb52466`, the merged structured-output/speculative-decoding behavior from
  PRs #44297 and #44993;
- `47bb963ea`, the explicit strict-tool override adapted from PR #49885.

MoET-specific work is `fdc9c1a39`, preserving nested DeepSeek DSML objects,
and `752637ef1`, loading completed runtime planes directly while using the
matching delta pack. Commit `e89479ec2` limits that path to the covered native
MXFP4 builder. These are runtime integration changes, not Hermes-specific
behavior.

## Container lineage

The earlier reproducible OCI path was published by commits `911fa67` and
`0544e69`. Its immutable image is
`ghcr.io/jpezzulli/vllm-moet:ds4flash-0731-sm120-v4` at digest
`sha256:d9dfc7f74ed95c4dc9b4170dff627cd08ec7f1ca4c4f26463d4497b4b4c6e99d`.
It validates the three-mapped-layer DSpark-3/393,216-token geometry from the
95ef patch lineage, not the final native configuration.

## Evidence roots

- DSpark-4 frozen suites:
  `/opt/ai-artifacts/logs/pennyroyal-dspark4-performance-suites-20260802-205032`
- Million-token retrieval:
  `/opt/ai-artifacts/logs/pennyroyal-dspark4-1m-needle-20260802-220952`
- Earlier native three-layer reproduction:
  `/opt/ai-artifacts/logs/moet-mapped-w2-clean-validation-20260802-005954`

Local evidence paths document origin and custody; they are not public runtime
inputs. See [VALIDATION.md](VALIDATION.md) for hashes and claim boundaries.
