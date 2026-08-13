# DeepSeek V4 Flash on one RTX PRO 6000 Blackwell

> **Attribution:** This work is built on
> [`kacper-daftcode/vLLM-Moet`](https://github.com/kacper-daftcode/vLLM-Moet).
> Kacper created the underlying SM120 runtime, W2 execution kernels, 2-bit
> expert and FP4 recovery system, DSpark/Runner V2 integration, and broader
> vLLM-MoET foundation. This fork adds targeted complete-layer mapped-host W2
> placement, NUMA-aware allocation and auditing, the single-RTX-PRO recipe,
> native reproduction path, and the validation reported here.

This fork serves the official **DeepSeek-V4-Flash-0731** checkpoint on one
**RTX PRO 6000 Blackwell Workstation Edition (96 GB)**. Selected complete W2
layers live in NUMA-local, CUDA-mapped host memory and are read directly by
the existing SM120 kernels over PCIe. Two current DSpark-4 profiles preserve
the full target-only 6 GiB FP4 correction tier: a faster **300,000-token**
profile and a larger **524,288-token** profile.

| Current profile | 300K performance | 512K capacity |
|---|---:|---:|
| Mapped W2 layers | DSpark 43–45 | target 42 + DSpark 43–45 |
| Mapped host bytes | 5,435,817,984 | 7,247,757,312 |
| Configured admission | 300,000 tokens | 524,288 tokens |
| Runtime-reported KV capacity | 378,490 tokens | 901,924 tokens |
| Exact uncached prefill exercised | 250,000 tokens | 500,000 tokens |
| **Server-reported prompt throughput** | **24,998.6 tok/s**, 0% cache hit | **49,974.3 tok/s**, 0% cache hit |
| Supplemental client TTFT / prompt÷TTFT | 80.499 s / 3,105.64 tok/s | 274.690 s / 1,820.23 tok/s |
| Exact 1×1,024 decode after first token | 89.82 tok/s | 68.23 tok/s |
| Exact 4×1,024 aggregate generation | 152.31 tok/s | 126.83 tok/s |

The bold prefill values are the literal numbers emitted by vLLM's standard
service logger. For these unusually long prefills, the logger emitted no
periodic throughput line while the GPU was occupied, then attributed the
completed prompt-token count to one nominal reporting interval. Consequently,
`24,998.6` and `49,974.3` are preserved as server-reported counters but must
not be interpreted as sustained physical prefill rates. The client-derived
TTFT values require a separate timing harness and provide the corresponding
request-wall measurements. Both requests reported 0% prefix-cache hit.
Decode rate varies with generated content: an earlier
300K exact 1,024-token run measured 73.63 tok/s, while the deliberately
captured 1×/4× baseline measured 89.82 tok/s. The 512K profile's 500K request
decoded at 85.56 tok/s after prefill. These are bounded measurements, not a
universal throughput guarantee.

The frozen quality and near-million-token results below are preserved from
the earlier five-target-layer profile; they were not rerun for either current
profile.

| Historical validated result | Outcome |
|---|---:|
| [Frozen reasoning quality](validation/README.md#frozen-reasoning-suite) | **97.07/100** |
| [Tool/agent suite](validation/README.md#tool-and-agent-suite) | **30/30** exact tool selections and arguments |
| Combined suite-wall throughput | **53.44 tok/s** including harness overhead |
| Model-generation throughput | approximately **56.49 tok/s**, excluding harness overhead |
| [Long-context retrieval](validation/README.md#opt-in-near-million-token-needle) | **994,987 input tokens**, correct needle |
| [Million-token prefill](validation/README.md#opt-in-near-million-token-needle) | **971.495 s at 1,024.18 tok/s** |
| [Time to first token](validation/README.md#opt-in-near-million-token-needle) | **975.550 s**, approximately 16 minutes |
| [Post-prefill decode](validation/README.md#opt-in-near-million-token-needle) | **64.12 tok/s** |
| [Immediate follow-up](validation/README.md#opt-in-near-million-token-needle) | Correct: `37 + 58` returned `95` |
| Lowest directly sampled physical VRAM free | **119 MiB** during the long-context run |

## What changed

The mapped allocation path now supports a layer-scoped FP4 correction
exclusion. This lets DSpark W2 layers 43–45 use mapped host memory while the
shared 6 GiB correction pool remains target-only. The 300K profile keeps all
target W2 layers in VRAM and maps DSpark layers 43–45. The 512K profile also
maps complete target layer 42; target layer 42 remains eligible for FP4
correction, while DSpark layers 43–45 remain excluded.

Each selected complete W2 layer is constructed directly in pinned, GPU-local
host memory:

- 1,811,939,328 bytes per mapped layer;
- one canonical allocation per layer, with no redundant complete GPU W2 copy;
- automatic selected-GPU PCI/NUMA resolution with fail-closed locality checks;
- stable UVA pointers retained through normal full and piecewise CUDA graphs;
- direct kernel reads over PCIe, without a host-replay or GPU staging cache.

The recovered VRAM is used for **512 × 12 MiB FP4 correction slots (6 GiB)**
and FP8 MLA KV. The preferred capacity launcher uses DSpark-4, four sequences,
2,048 maximum batched tokens, DeepGEMM,
`--gpu-memory-utilization 0.98446`, and `--max-model-len 524288`.

## Verified cache-backed startup

The launcher uses MoET's existing persistent formats as one matched cache
set: the per-layer base/planes cache contains the runtime-ready 2-bit planes
and scales, while the pack store contains the FP4 correction rows. On a
compatible hit, the checkpoint loader skips the corresponding native MXFP4
expert payloads and four workers load at most 12 completed plane layers per
batch directly into their final GPU or canonical mapped-host destinations.
It performs no FP4-to-2-bit projection and no W2 reconstruction at startup.

Eligibility is deliberately cheap and fail closed: checkpoint/cache key,
quantizer ABI and zero mode, TP rank, exact geometry, layer coverage, sidecar,
and exact file sizes. Payload packs are not hashed or scanned. A missing,
partial, stale, or incompatible layer follows the normal checkpoint-source
construction path and is atomically republished into the existing formats.

The August 9, 2026 hardware run measured:

| Startup phase | Result |
|---|---:|
| Cache eligibility | 46 layers in **0.045 s** |
| Base/planes reads | **77.625 GiB** in five batches, **55.898 s** |
| Remaining target checkpoint loading | **9.20 s** |
| DSpark checkpoint loading | **3.84 s** |
| Total model loading | **89.011 s** |
| CUDA/DSpark graph capture | **3 s** |
| Engine profile, KV creation, and warm-up | **19.02 s** |
| Service start to ready | **130 s** |

The prior comparable model-load baseline was **309.302 s**, so direct cache
loading reduced that phase by **220.291 s (71.2%)**. The correction pack is
opened after its sidecar/geometry check and supplies rows lazily; it is not
bulk-read into a second host representation during startup.

To regenerate a clean cache, stop all writers, empty only the configured MoET
cache directory, then start this same launcher once. That source-construction
run creates one cache-keyed planes directory and one matching delta pack.
Subsequent starts select the direct path automatically.

## Runtime maintenance

The August 9, 2026 runtime tip also incorporates narrowly adapted upstream
vLLM repairs:

- packed DeepSeek-V4 KV-block zeroing from vLLM commit `d6af803` / PR #50276;
- structured output with speculative decoding from merged PRs #44297 and
  #44993;
- the merged parser-engine migration from PR #45877 and special-token repair
  from PR #48748.

The DS4 parser line preserves `string="true|false"` and guarded wrapper
handling from PR #41801 at merge
`95582868efd4db0b120e3640bbc61dcfce20d59f`; incremental DSML argument
streaming from PR #42879 at merge
`b372ad3e9018f032478619adbc7f7fdcc9318212`; the shared parser-engine migration
from PR #45877 at merge `fb5291b35`; and declared-tool-only orphan recovery
adapted from open PR #49117. MoET additionally decodes recursive DSML parameter
trees emitted for nested open objects, rather than flattening nested members
into the outer call.

The launcher does not force strict tool calling. Structural grammar remains
request controlled through each function's `strict` field. A strict outer
bridge can require its declared `name` and `arguments` members, but an open
nested `arguments` object cannot constrain fields from a deferred schema that
was never included in the request. Clients must validate or materialize that
schema; the runtime does not invent missing nested arguments.

Focused maintenance validation completed without leaked DSML, `R0TURN`, JSON
errors, or parser-finalization exceptions. The historical full quality suites
were not rerun for this parser maintenance change.

## Why it matters

Moving selected complete W2 layers to CUDA-mapped, NUMA-local host memory
frees VRAM for the KV cache while keeping the target model's full 6 GiB FP4
correction tier. The existing GPU kernels read those canonical weights
directly over PCIe, without redundant complete GPU copies, host replay, or a
GPU staging cache.

## Build and run

Native source build is the preferred path:

- [Native build script](scripts/build-native.sh)
- [Production launcher](scripts/serve-pro6000-ds4flash.sh)
- [Lightweight API validation](scripts/validate-api.sh)

```bash
./scripts/build-native.sh
MODEL_PATH=/path/to/DeepSeek-V4-Flash-0731 \
SERVED_MODEL_NAME=pennyroyal \
./scripts/serve-pro6000-ds4flash.sh
./scripts/validate-api.sh
```

Read [BUILD-AND-RUN.md](BUILD-AND-RUN.md) before starting. It contains the
hardware and host-memory contract, exact pinned sources, build details,
startup checks, and expected audit geometry.

## Current serving profiles

| Component | 300K performance | 512K capacity |
|---|---|---|
| GPU / checkpoint | 1× RTX PRO 6000 96 GB / DS4-Flash-0731 | same |
| Runtime | Runner V2, DeepGEMM, full + piecewise graphs | same |
| Mapped target W2 | none | layer 42 |
| GPU-resident target W2 | layers 0–42 | layers 0–41 |
| Mapped DSpark W2 | layers 43–45 | layers 43–45 |
| Speculation | DSpark-4, greedy draft | same |
| Target FP4 correction | 512 × 12 MiB = 6 GiB | same |
| FP8 MLA KV capacity | 378,490 tokens reported | 901,924 tokens reported |
| GPU utilization budget | 0.974 | 0.98446 |
| Scheduler | 4 sequences, 2,048 batched tokens | same |
| Configured admission | 300,000 tokens | 524,288 tokens |

The older five-target-layer 1M profile remains valuable historical evidence:
it correctly retrieved a needle from 994,987 input tokens, but took roughly
16 minutes to first token. It is not the current serving recipe. Configured
admission, runtime-reported KV capacity, and exercised input length are kept
separate throughout this repository.

## Documentation

- [BUILD-AND-RUN.md](BUILD-AND-RUN.md) — pinned native build, launch, and smoke validation
- [HOW-IT-WORKS.md](HOW-IT-WORKS.md) — allocation, NUMA/UVA, PCIe, graphs, and memory geometry
- [VALIDATION.md](VALIDATION.md) — quality, tools, throughput, 1M retrieval, power, and limits
- [Public validation suites](validation/README.md) — runnable smoke, reasoning, tools, and opt-in needle tests
- [PROVENANCE.md](PROVENANCE.md) — ownership, exact commits, model pin, and evidence roots

The large OCI image remains available as an earlier, fully reproducible
DSpark-3/393,216-token artifact; it is not the final production shape. See
[its container record](docs/container-ds4flash-0731-sm120.md).

> For GLM, Kimi, RTX 5090, multi-GPU, NVMe expert-tier, kernel, and broader
> vLLM-MoET development, see
> [Kacper's upstream project](https://github.com/kacper-daftcode/vLLM-Moet).

## Claim boundary

Direct validation covers this checkpoint, this single 96 GB GPU, full and
piecewise CUDA graphs, the bounded profile measurements above, the historical
frozen reasoning/tool suites, and one historical 994,987-token request. The
frozen suites were not rerun for the new 300K/512K profiles. Validation does
not establish 512K concurrency, tensor parallelism, sustained soak or reload,
another GPU or checkpoint, or DSpark-4 support inside the historical v4 OCI
container.

## EXL3 M=8 readiness

The maintained runtime now includes Kacper's EXL3 base+delta serving line from
publication `66ec4e7f5b098827f5f759779f874737715bb841` (runtime source
`b02ae23bcb5ba9c1e3d74acba0d11586e547b747`), including the M=1 wave path and
M=2 through M=8 eager and capture-safe CUDA-graph routes. The matching SM120
M8/m8g canons, cubins, SASS, and map-builder tests are published here.

EXL3 is **not enabled** in the Penny recipe. It requires model-specific EXL3
base and residual-delta packs with the exact pack-v3 geometry expected by the
runtime, plus the matching exllamav3 extension entry points. Kacper's public
repository currently publishes the serving implementation and kernels but not
the referenced base/delta pack builders or compatible DeepSeek-V4-Flash pack
artifacts. The existing 2-bit planes and 12 MiB FP4 correction packs are not
interchangeable with those EXL3 inputs.

The reconciliation was therefore validated with the existing production
geometry unchanged. It completed all graph captures, reported 1,058,042 KV
tokens, placed layers 38–42 in 9,059,696,640 bytes of NUMA-local mapped host
memory with zero redundant complete GPU W2 bytes, and passed bounded arithmetic
and exact tool-call smoke requests. This maintenance smoke does not replace
the frozen quality or long-context results above. EXL3 activation requires a
separate candidate and proportional qualification after the pack tooling is
available.

## Repository map

- `scripts/` — pinned native build, final production launcher, and API validation
- `bench/recipes/` — machine-readable serving recipes and historical benchmark matrix
- `patch/` — sanctioned generated patch for the runtime in `patch/SOURCE.txt`
- `kernels/` — Kacper's SM120 SASS, generated cubins, and manifests
- `container/` and `Containerfile.ds4flash-0731-sm120` — historical OCI v4
  material, reproduced exactly from tag `history/oci-v4-20260802`
- `docs/` — detailed historical implementation, benchmark, and container records
- `validation/` — frozen public cases, mock tools, graders, replay fixtures, and runners

Never hand-edit `patch/vllm-moet-v0.24.0.patch`, `patch/FILES.txt`, or
`patch/SOURCE.txt`; see [AGENTS.md](AGENTS.md).
