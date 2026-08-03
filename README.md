# DeepSeek V4 Flash on one RTX PRO 6000 Blackwell

This fork serves the official **DeepSeek-V4-Flash-0731** checkpoint on one
**RTX PRO 6000 Blackwell Workstation Edition (96 GB)**. Five complete target
W2 layers live in NUMA-local, CUDA-mapped host memory and are read directly by
the existing SM120 kernels over PCIe. That releases enough VRAM for a 6 GiB
FP4 correction tier, DSpark-4, and a practical **1,000,000-token admission
limit**.

| Validated result | Outcome |
|---|---:|
| Frozen reasoning quality | **97.07/100** |
| Tool/agent suite | **30/30** exact tool selections and arguments |
| Combined suite throughput | **53.44 tok/s** including harness overhead |
| Representative model generation | **56.49 tok/s** excluding harness overhead |
| Long-context retrieval | **994,987 input tokens**, correct needle |
| Million-token prefill | **1,024.18 tok/s**, 975.550 s TTFT |
| Post-prefill decode | **64.12 tok/s** |

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

## Build and run

Native source build is the preferred path:

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
| Runtime | Model Runner V2, DeepGEMM, normal CUDA graphs |
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
- [PROVENANCE.md](PROVENANCE.md) — ownership, exact commits, model pin, and evidence roots

The large OCI image remains available as an earlier, fully reproducible
DSpark-3/393,216-token artifact; it is not the final production shape. See
[its container record](docs/container-ds4flash-0731-sm120.md). Kacper's
[upstream repository](https://github.com/kacper-daftcode/vLLM-Moet) remains
the authoritative place for the broader GLM, Kimi, multi-GPU, RTX 5090, NVMe
expert-tier, and general vLLM-MoET work.

## Claim boundary

Direct validation covers this checkpoint, this single 96 GB GPU, normal CUDA
graphs, the frozen reasoning/tool suites, and one 994,987-token request.
It does not establish tensor parallelism, another GPU or checkpoint,
multi-request concurrency at 1M, long soak/reload stability, or the final
DSpark-4 shape inside the older v4 container.

<details>
<summary>Historical benchmark tables inherited from the broader upstream project</summary>

These generated tables are retained for repository integrity and historical
reference. They are not evidence for the final single-card production shape
described above.

<!-- bench:table:begin (generated by bench/runner/render.py - do not edit) -->

Release **`baseline-2026-07-10`** — one row per supported recipe (`bench/recipes/`), measured by `bench/runner/bench.py`; full report: [`docs/benchmarks/baseline-2026-07-10.md`](docs/benchmarks/baseline-2026-07-10.md). Single-stream decode and prefill are medians; batch is aggregate tok/s at the noted concurrency.

| model | hardware | config | ctx | decode tok/s | batch | prefill 8K | needle | notes |
|---|---|---|---:|---:|---:|---:|---|---|
| deepseek-v4-flash | 1x RTX 5090 (32 GB) | host-resident 2-bit base, GPU as expert cache | 8K | **38** | — | — | — | acc 2.83 † |
| deepseek-v4-flash | 4x RTX 5090 TP4 | consumer-card throughput | 16K | **214.4** | 1 560 @32 | 6 101 | — | acc 2.6 † |
| deepseek-v4-flash | 2x RTX PRO 6000 TP2 | throughput | 24K | **209.6** | 380 @3 | 5 791 | — | acc 2.6 † |
| glm-5.2-nvfp4 | 2x RTX PRO 6000 TP2 | host-resident base, 44 GiB/rank expert cache | 32K | **33** | — | — | PASS ≤27K tok | acc 3 † |
| glm-5.2-nvfp4 | 4x RTX PRO 6000 TP4 | 2-bit base + MTP k=2, 128K window | 128K | **105** | — | 2 500 | PASS ≤276K tok | † |
| glm-5.2-nvfp4 | 4x RTX PRO 6000 TP4 | + FP4 delta (auto) + confidence gate tau=0.60 | 128K | **84** | — | — | — | † |
| kimi-k2.7-code-nvfp4 | 2x RTX PRO 6000 TP2 | host-resident base, 52 GiB/rank cache (~39% coverage) | 16K | **14.4** | — | — | PASS ≤8K tok | † |
| kimi-k2.7-code-nvfp4 | 4x RTX PRO 6000 TP4 | GPU-resident 2-bit + FP4 delta, 256K window | 256K | **51** | 222 @8 | 2 448 | PASS ≤248K tok | † |
| kimi-k2.7-code-nvfp4 | 4x RTX PRO 6000 TP4 | + Eagle3 drafter (k=3, drafter TP4) | 256K | **57** (±44%) | — | — | — | acc 3.5 † |

† imported from pre-harness measurements (README/docs history) — re-measured on the next release.

<!-- bench:table:end -->

<!-- bench:quality:begin (generated by bench/runner/render.py - do not edit) -->

Quality release **`v2026.07.30-quality`** — dataset evals vs the committed **native baselines** (`bench/baselines/`, same tool/checkpoint/hardware, stock serving path). Cells: accuracy (Δpp vs native, completion-token inflation vs native). Full report: [`docs/benchmarks/v2026.07.30-quality.md`](docs/benchmarks/v2026.07.30-quality.md); process: [`bench/README.md`](bench/README.md).

| model | hardware | config | GSM8K | GPQA | GPQA think | needle | notes |
|---|---|---|---|---|---|---|---|
| deepseek-v4-flash | 2x RTX PRO 6000 TP2 | max-quality candidate (revalidation required) | **95.5%** (-1.5pp, tok +1.3%) | **73.2%** (-1.0pp, tok +10.6%) | — | — | † |

† imported from the measurement campaign logs — re-measured by the harness on the next quality release.

<!-- bench:quality:end -->

</details>

## Repository map

- `scripts/` — pinned native build, final production launcher, and API validation
- `bench/recipes/` — machine-readable serving recipes and historical benchmark matrix
- `patch/` — sanctioned generated patch for current runtime `a2131dd7`
- `kernels/` — Kacper's SM120 SASS, generated cubins, and manifests
- `container/` and `Containerfile.ds4flash-0731-sm120` — historical OCI v4
  material, reproduced exactly from tag `history/oci-v4-20260802`
- `docs/` — detailed historical implementation, benchmark, and container records

Never hand-edit `patch/vllm-moet-v0.24.0.patch`, `patch/FILES.txt`, or
`patch/SOURCE.txt`; see [AGENTS.md](AGENTS.md).
