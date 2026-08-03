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

## Exact source identities

| Component | Identity | Role |
|---|---|---|
| Official vLLM base | `vllm-project/vllm@ee0da84ab9e04ac7610e28580af62c365e898389` (`v0.24.0`) | original release lineage |
| Kacper runtime base for final mapped-host commits | `kacper-daftcode/vllm@960abee41` | current Runner V2/DSpark/W2 foundation used by the final branch |
| Mapped-host implementation | `jpezzulli/vllm@f80b8ec884a3f455a900ba6c57ad4633b4ded0b2` | canonical mapped-host W2 source |
| Validated runtime tip | `jpezzulli/vllm@a2131dd7a944353e9323566107c72f4a17441024` | implementation plus focused lifecycle tests |
| Current publication base | `jpezzulli/vLLM-Moet@28254cd8d8d58d5a331e6027f8b48a867c03bbda` | generated a2131 patch and mapped-host publication |
| Historical OCI packaging | `jpezzulli/vLLM-Moet@0544e69e63dce5a9cf597797df3db140391ba832` | sealed v4 image built from runtime 95ef4a88 |
| DeepGEMM | `deepseek-ai/DeepGEMM@a6b593d2826719dcf4892609af7b84ee23aaf32a` | validated DeepGEMM build |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` | official 0731 model |

The validated vLLM branch is
`jpezzulli/vllm:production/rtx-pro6000-ds4flash0731-mapped-w2`; its remote tip
is `a2131dd7a944353e9323566107c72f4a17441024`.

## Runtime versus generated-patch lineage

The final DSpark-4/1M native service runs the directly pinned a2131 runtime.
The current `rtx-pro6000` publication patch is also generated from a2131. The
published v4 OCI image is a separate historical artifact: its sealed source
used runtime `95ef4a88c63c9ed88f2384977e05d788897af6c3` and publication
packaging `0544e69e63dce5a9cf597797df3db140391ba832`. The current patch must not
be presented as the source of that already-published image.

No runtime or generated-patch change is part of this documentation branch.
`patch/vllm-moet-v0.24.0.patch`, `patch/FILES.txt`, and `patch/SOURCE.txt`
remain byte-for-byte unchanged from publication base `28254cd8`.

## Production launcher identity

The final hardware-validated launcher SHA-256 is:

```text
7d1b1a29f257ea1efd8cb991e6d5576f02b0ec7a5773fceb3ecdbab46f9d383d
```

It selects mapped target layers 38–42, GPU-resident DSpark layers 43–45,
DSpark-4, a 6 GiB FP4 correction tier, FP8 MLA KV, four sequences, 2,048
maximum batched tokens, full and piecewise graphs, DeepGEMM,
`gpu_memory_utilization=0.974`, and a 1,000,000-token admission limit.

The repository launcher exposes model path, served name, port, cache, pack,
and audit locations as environment variables while keeping that validated
geometry fixed.

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
