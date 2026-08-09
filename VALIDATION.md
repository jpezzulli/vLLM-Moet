# Validation

All results below are from one RTX PRO 6000 Blackwell Workstation Edition
using DeepSeek-V4-Flash-0731. Historical qualification used the earlier
mapped-host runtime; the focused August 9 maintenance validation used runtime
`752637ef1a78787e84bb33efe835b4c0b06e4918` without rerunning those suites.

## Direct-cache startup maintenance validation

One clean cache generation and one subsequent direct-cache start were run on
the production DSpark-4/1M geometry. The final directory contained one delta
pack generation and 46 layers with exactly four base-plane parts each; no
temporary files or duplicate FP4 plane files remained.

| Check | Result |
|---|---:|
| Eligibility probes | 46 layers, 0.045 s |
| Direct base/planes loading | 77.625 GiB, five batches, 55.898 s |
| Remaining target checkpoint load | 9.20 s |
| DSpark checkpoint load | 3.84 s |
| Total model load | 89.011 s |
| Previous model-load baseline | 309.302 s |
| Graph capture | 3 s |
| Start to API ready | 130 s |
| Runtime KV capacity | 1,058,256 tokens |

Logs explicitly reported no CPU projection or reconstruction for every direct
batch. Mapped layers 38–42 contained 9,059,696,640 bytes on NUMA node 0 with
zero redundant complete GPU W2 bytes. `/health`, `/v1/models`, arithmetic
(`37 + 58 = 95`), and one strict `lookup_order` tool call all passed. No OOM,
crash, or parser-finalization error occurred.

This was proportional maintenance validation, not a rerun of the frozen
reasoning, 30-tool, or long-context suites. An xgrammar stop-boundary warning
previously observed with strict structured output plus DSpark remains an
unresolved qualification warning; the focused tool smoke completed normally.

## Runnable public validation

The exact public harnesses are documented in
[validation/README.md](validation/README.md):

1. lightweight API and mapped-W2 smoke validation;
2. the frozen nine-request reasoning suite and blinded rubric;
3. the exact 30-invocation tool/agent suite with local mock tools;
4. the explicitly acknowledged 994,987-token needle test.

The scores below are historical validated results. A rerun creates a new
dated result and does not replace this evidence. Reasoning quality includes
blinded reviewer judgment; tool scoring is automatic and compares emitted
calls with local schemas and expected arguments without performing external
side effects. The near-million-token test is opt-in and historically required
about 16 minutes to reach its first token. Exact 1,048,576-token support was
not demonstrated.

## DSpark-4 quality and tool qualification

The final performance qualification used mapped target W2 layers 38–42,
GPU-resident DSpark layers 43–45, a 6 GiB correction tier, FP8 MLA KV,
DeepGEMM, full and piecewise graphs, `reasoning_effort=max`, and `top_p=0.95`.
This qualification used the earlier 393,216-token admission and
`gpu_memory_utilization=0.988`; the only change from its DSpark-7 comparison
was speculative depth 4.

| Suite | Quality gate | Server generation | Suite-wall effective |
|---|---:|---:|---:|
| [Frozen reasoning, 9 requests](validation/README.md#frozen-reasoning-suite) | **97.07/100** | **56.50 tok/s** | 55.89 tok/s |
| [Tool/agent, 30 invocations](validation/README.md#tool-and-agent-suite) | **30/30** | **56.44 tok/s** | 40.36 tok/s |
| Combined workload-weighted | all gates passed | **56.49 tok/s** | — |

The model-generation value excludes harness and grading overhead. It must not
be confused with the combined **53.44 tok/s** suite-wall result, which includes
harness overhead. Against the matched DSpark-7 windows, DSpark-4 improved
combined model-generation throughput by 16.00% while preserving quality and
30/30 exact tool selections and arguments.

Reasoning C5 retained minor gaps: no explicit integer-cents requirement, no
request-ID/fingerprint claim before account work, and one unsupported missing
source-row assertion. No fatal reasoning finding occurred. All 30 tool calls
had the exact selection and arguments; no parser failure, tool failure, loop,
OOM, or runtime error occurred.

## Million-token validation

The final production launcher changed only admission and vLLM utilization
from the qualified DSpark-4 launcher:

- `--max-model-len`: 393,216 → 1,000,000
- `--gpu-memory-utilization`: 0.988 → 0.974

All other serving geometry remained fixed.

| Measurement | Result |
|---|---:|
| Actual server-rendered input | **994,987 tokens** |
| Needle position | zero-based token **154** |
| Retrieval | exact `GRID-NEEDLE-7B91E2C4A6F0D835` recovered |
| Prefill | **971.495 s**, **1,024.18 tok/s** |
| TTFT | **975.550 s** |
| Decode | **64.12 tok/s** over 116 completion tokens |
| Client request wall | **977.482 s** |
| Runtime-reported KV | 4.96 GiB, **1,058,256 tokens** |
| Follow-up | `37 + 58` returned `95` in 1.189 s |

The DeepSeek reasoning parser emitted the retrieved answer in the reasoning
field, including an explicit `Final answer` line. The original generic
harness checked a different field; the raw stream and corrected interpretation
are the authoritative retrieval evidence.

One million tokens is an exceptional or batch workload, not an interactive
window: TTFT was about 16 minutes.

## Rejected boundary

- 1,048,576 admission at utilization 0.988 failed during FlashInfer sparse-MLA
  autotuning when a 1.02 GiB allocation was attempted with 174.19 MiB free.
  Cleanup was followed by one Xid 31; the GPU recovered.
- 1,048,576 at utilization 0.974 was rejected before warm-up because vLLM
  estimated a maximum supported length of 1,004,288.
- 1,000,000 at utilization 0.974 started, captured every graph, completed the
  994,987-token request, and remained healthy.

The failed 1,048,576 shape is not recommended or advertised.

## Power, temperature, and memory

| Metric | Reasoning suite | Tool suite | 994,987-token request |
|---|---:|---:|---:|
| Host power mean / peak | 507.03 / 659 W | 475.43 / 603 W | 479.48 / 608 W |
| GPU power mean / peak | 155.40 / 167.06 W | 141.50 / 205.92 W | 202.78 / 377.53 W |
| Combined mean / peak | 662.42 / 812.77 W | 616.93 / 756.05 W | 682.26 / 918.57 W |
| CPU package peak | 71 C | 63 C | 69 C |
| GPU peak | 43 C | 45 C | 56 C |
| PCIe RX peak | 29,983 MiB/s | 30,043 MiB/s | 30,821 MiB/s |
| Minimum physical VRAM free | 7 MiB sampled across qualification | 7 MiB sampled across qualification | 119 MiB |

The host power meter excludes the separately powered GPU; combined values add
synchronized host and GPU samples. No thermal stop occurred.

## Evidence identities

| Evidence | Local root | Final-report SHA-256 |
|---|---|---|
| DSpark-4 qualification | `/opt/ai-artifacts/logs/pennyroyal-dspark4-performance-suites-20260802-205032` | `e239eb376351fcaabdf4ac360d7144785de918478067b6f037be3f8e517728d2` |
| Million-token retrieval | `/opt/ai-artifacts/logs/pennyroyal-dspark4-1m-needle-20260802-220952` | `a4b64f5a0b459d577b13ea3376088ec564ff0ce10a4405abe1ba32f809f82687` |

These paths identify sealed evidence on the validation host; they are not
runtime dependencies or bundled public downloads.

## Claim limits

Validated: this checkpoint, one 96 GB RTX PRO 6000, mapped layers 38–42,
DSpark-4, normal CUDA graphs, the frozen suites, and one 994,987-token request.

Not validated: another checkpoint or GPU, tensor parallelism, the final shape
inside the older container, one-million-token concurrency, sustained soak,
repeated in-process teardown/reload, or the exact 1,048,576-token setting.
