# Validation status

## Evidence boundary

The sealed candidate evidence is the 2026-08-01 artifact directory
`moet-mapped-w2-step2-20260801-222626`. Its `SHA256SUMS` file hashes to
`cbb86e42a7d77268ce46aa20edec1f78d6b361b2e08a5d6dec238dc01f6b73c5`.
The final report hashes to
`69f625f7c673e1de2f51113a6b9ddaddf153cdb30d12d8777796688f97261cf3`.
The original overlay was experimental; the reviewable implementation replaces
it. The independent clean-checkout reproduction is preserved under
`moet-mapped-w2-clean-validation-20260802-005954`; its `SHA256SUMS` file hashes
to `9f00c763eb7736852c249b261aa1d4609f032d30185008c4a792c27c5982d901`, and its
report hashes to
`44396bcf4384da3cf4f900b106bf99d4029dbac43c919e185a4bb84986470bb5`.

| Area | Status | Direct evidence | What is not proved |
|---|---|---|---|
| Standalone mapped CUDA/NUMA mechanism | Passed | Exactly 3.375 GiB; all 884,736 pages on NUMA node 0; zero correctness errors; 24.089 GiB/s and 304.53 simulated steps/s; VRAM delta 6 MiB | End-to-end MoET behavior |
| Startup | Passed on sealed overlay and clean checkout | Runner V2, DSpark-3, 512 FP4 slots/6 GiB, layers 40–42 mapped, all graph captures complete; clean KV capacity also 625,757 | Other GPUs, checkpoints, layer selections, or TP ranks |
| Allocation accounting | Passed on sealed overlay and clean checkout | 1,811,939,328 bytes per layer; 5,435,817,984 bytes total; zero complete duplicate GPU W2 bytes; both runs had 1,057 MiB physical free after capture | Long-duration fragmentation or repeated reloads |
| Arithmetic correctness | Passed on sealed overlay and clean checkout | Both returned `37 - 9 = 28`, `28 - 6 = 22`, total 50, HTTP 200, normal stop | Broad model quality |
| Bounded decode | Passed on sealed overlay and clean checkout | Same 37-input/1,024-output request: sealed 55.016 tok/s, clean 53.829 tok/s after first token (-2.16%); clean TTFT 0.4520 s and wall 19.4567 s | Other prompts, concurrency, or sampling regimes |
| DSpark acceptance | Measured on both bounded runs | Sealed 608/1,251 = 48.60%; clean 574/1,350 = 42.52%. Clean positions: 63.11%, 41.11%, 23.33% | Acceptance stability across workloads |
| Context | Capacity only | Configured limit 393,216; runtime-reported KV capacity 625,757 tokens | A 393,216-token request or retrieval correctness at that length |
| Frozen quality suites | Passed twice | Baseline: reasoning 92.66/100 and tools 30/30 reviewed at `reasoning_effort=high`. Production: reasoning 96.86/100 and tools 30/30 automatic/reviewed at `reasoning_effort=max`, `top_p=0.95`; see [mapped-W2 quality validation](mapped-w2-quality-validation.md) | Setting-level causality, arbitrary-workload quality, or multi-pass variance |
| Soak/reload | Not run | None | Long-running stability and repeated in-process teardown |
| Container | Not validated | None | Image build or container serving |

The 1,024-token summary file hashes to
`db8003696ece30ac454c90a6874c9c36791fa07c6954f0d361cc109951398297`;
the mapped audit hashes to
`b7b6dda19438c13e90f0fd07b3faff346513ac85aaf5d5931e383627771f3d21`.

The clean checkout used a Python 3.13 native build because bundled
`vllm-flash-attn` does not build on Python 3.14. Its stable-ABI extensions were
loaded by the Python 3.14 runtime. The precompiled nightly path was rejected
after it reproduced an `_moe_C` operator-signature mismatch. Focused clean
tests passed 57 with 5 hardware skips, and focused Ruff checks passed.

## Automated checks

The source branch includes focused tests for parsing and compatibility guards,
the exact DeepSeek layer byte geometry, selection, duplicate allocation,
mapped pointer validation, NUMA allocation failure, construction rollback,
accounting, synchronization, and `cudaFreeHost` cleanup. These tests use a
fake CUDA Runtime and do not substitute for the hardware-dependent startup
check.

Run:

```bash
python -m pytest -q \
  tests/model_executor/layers/test_moe_w2_mapped_host.py \
  tests/model_executor/layers/test_moe_w2_persistence.py
```

Run the standalone hardware diagnostic separately from
`tools/cuda_mapped_host_probe/`. A production hardware integration check must
also inspect the JSON audit, server graph-capture logs, physical VRAM, and a
request result.
