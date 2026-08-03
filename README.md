# DeepSeek V4 Flash on one RTX PRO 6000 Blackwell

This fork serves the official **DeepSeek-V4-Flash-0731** checkpoint on one
**RTX PRO 6000 Blackwell Workstation Edition (96 GB)**. Five complete target
W2 layers live in NUMA-local, CUDA-mapped host memory and are read directly by
the existing SM120 kernels over PCIe. That releases enough VRAM for a 6 GiB
FP4 correction tier, DSpark-4, and a practical **1,000,000-token admission
limit**.

| Validated result | Outcome |
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

> **Attribution:** This work is built on
> [`kacper-daftcode/vLLM-Moet`](https://github.com/kacper-daftcode/vLLM-Moet).
> Kacper created the underlying SM120 runtime, W2 execution kernels, 2-bit
> expert and FP4 recovery system, DSpark/Runner V2 integration, and broader
> vLLM-MoET foundation. This fork adds targeted complete-layer mapped-host W2
> placement, NUMA-aware allocation and auditing, the single-RTX-PRO recipe,
> native reproduction path, and the validation reported here.

## What changed

The production shape keeps dense weights, target W2 layers 0–37, and all
three DSpark W2 layers 43–45 on the GPU. Complete target W2 layers **38–42**
are constructed directly in pinned, GPU-local host memory:

- 1,811,939,328 bytes per mapped layer; 9,059,696,640 bytes total;
- one canonical allocation per layer, with no redundant complete GPU W2 copy;
- automatic selected-GPU PCI/NUMA resolution with fail-closed locality checks;
- stable UVA pointers retained through normal full and piecewise CUDA graphs;
- direct kernel reads over PCIe, without a host-replay or GPU staging cache.

The recovered VRAM is used for **512 × 12 MiB FP4 correction slots (6 GiB)**
and FP8 MLA KV. The final launcher uses DSpark-4, four sequences, 2,048 maximum
batched tokens, DeepGEMM, `--gpu-memory-utilization 0.974`, and
`--max-model-len 1000000`.

## Why it matters

Moving five complete W2 layers to CUDA-mapped, NUMA-local host memory frees
VRAM for the KV cache and FP4 correction tier. The existing GPU kernels read
those canonical weights directly over PCIe, without redundant complete GPU
copies, host replay, or a GPU staging cache.

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

## Validated production shape

| Component | Setting |
|---|---|
| GPU | 1× RTX PRO 6000 Blackwell Workstation Edition, 96 GB |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Runtime | Model Runner V2, DeepGEMM, full and piecewise CUDA graphs |
| Mapped target W2 | layers 38–42, GPU-local NUMA host pages |
| GPU-resident target W2 | layers 0–37 |
| GPU-resident DSpark W2 | layers 43–45 |
| Speculation | DSpark-4, greedy draft |
| FP4 correction | 512 slots × 12 MiB = 6 GiB |
| KV | FP8 MLA, 1,058,256 tokens reported in the final 1M shape |
| Scheduler | 4 sequences, 2,048 maximum batched tokens |
| Admission | 1,000,000 tokens |

The 994,987-token test is an exceptional/batch workload, not an interactive
context claim: TTFT was approximately **16 minutes**. The exact 1,048,576-token
setting is not supported by the validated memory geometry and is deliberately
not advertised.

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
piecewise CUDA graphs, the frozen reasoning/tool suites, and one 994,987-token
request. It does not establish the exact 1,048,576-token setting,
one-million-token concurrency, tensor parallelism, sustained soak or reload,
another GPU or checkpoint, or DSpark-4 support inside the historical v4 OCI
container.

## Repository map

- `scripts/` — pinned native build, final production launcher, and API validation
- `bench/recipes/` — machine-readable serving recipes and historical benchmark matrix
- `patch/` — sanctioned generated patch for current runtime `a2131dd7`
- `kernels/` — Kacper's SM120 SASS, generated cubins, and manifests
- `container/` and `Containerfile.ds4flash-0731-sm120` — historical OCI v4
  material, reproduced exactly from tag `history/oci-v4-20260802`
- `docs/` — detailed historical implementation, benchmark, and container records
- `validation/` — frozen public cases, mock tools, graders, replay fixtures, and runners

Never hand-edit `patch/vllm-moet-v0.24.0.patch`, `patch/FILES.txt`, or
`patch/SOURCE.txt`; see [AGENTS.md](AGENTS.md).
