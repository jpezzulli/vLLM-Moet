# Mapped-W2 quality validation

> **Historical quality baseline:** these DSpark-3 runs remain useful evidence,
> but they are not the final production qualification. See
> [VALIDATION.md](../VALIDATION.md) for DSpark-4 at 97.07/100 and 30/30 tools.

## Scope and source identity

Two frozen-suite passes were run on the same clean native mapped-W2 candidate.
The runtime used the vLLM-MoET publication source commit
`42c8b60f61640fb3cbb17968950914aa9534cf3a`, generated from vLLM source
commit `98cef19a50765148aba29084dc88da5d16f31700` over official vLLM v0.24.0
commit `ee0da84ab9e04ac7610e28580af62c365e898389`. DeepGEMM was pinned at
`a6b593d2826719dcf4892609af7b84ee23aaf32a`.

Both passes used Runner V2, DSpark-3, canonical W2 layers 40–42 mapped to
NUMA node 0, 512 FP4 correction slots/6 GiB, FP8 MLA KV, normal CUDA graphs,
`gpu_memory_utilization=0.988`, configured context 393,216 tokens, and the
runtime-reported KV capacity of 625,757 tokens. Neither token count is an
exercised-context result.

## Frozen suite revision

| Component | SHA-256 |
|---|---|
| Reasoning suite | `d1397529eedf72b0f80d5c452c378ed15fb10f1122e4fb9b50f69e1074c0e756` |
| Reasoning protocol | `56fb8d57dda656495c5b18d5fc6ea35f533bae351d9696cba087fa2913bde928` |
| Reasoning rubric | `ea85f50f9c3cce2f9d9b6b63a3611ffd2e14d9cb90dc4a3d7b5bf7dc0f9c1c65` |
| Reasoning runner | `926db09bef5f3a8a1bde9e24edf506e42687433e233411acb0bc329092707b5b` |
| Tool suite | `2be4040e75d8f0f70bf472d5a0f686cd92ceb58bc6304db52ca6758797f4e294` |

The production pass used a one-line tool-harness execution overlay because
the frozen harness hardcoded request-level `reasoning_effort=high`, which
would override the production launcher. Its SHA-256 is
`77d5624692e928003500eb1c729bb61ec7aafad6dc0359631d6dca4b25df4299`;
the preserved diff changes only `high` to `max`. Prompts, tool schemas,
expected outputs, evaluator, thresholds, and order were unchanged.

## Run 1: frozen baseline

Settings were `reasoning_effort=high`, checkpoint-default `top_p=1.0`, and
the already active DeepGEMM FP8/MXFP4 kernels.

| Suite | Result | Token usage | Throughput |
|---|---:|---|---|
| Reasoning, 9 measured requests | 92.66/100; no fatal findings | 5,310 server prompt; 28,928 server generation; 28,568 API completion; 28,326 locally counted generation | 56.14 server decode tok/s; 53.47 effective local tok/s |
| Tool/agent, 30 invocations | 30/30 reviewed; 29/30 automatic | 66,009 server prompt; 4,938 server/API completion | 61.07 server decode tok/s; 41.27 effective completion tok/s |

The one automatic tool miss was an equivalent human-readable date instead of
ISO formatting; exact tool selection and arguments were correct and review
accepted it. Reasoning C5 omitted distinct-account and integer-amount
validation and incorrectly described the effect of a negative transfer. C8
corrected its first-turn containment error; C4 was correct but repeatedly
revisited branches. No OOM, crash, loop, parser failure, length termination,
heat stop, or runtime instability occurred.

Sealed local evidence:

- `/opt/ai-artifacts/logs/clean-native-mapped-w2-frozen-suites-20260802-060854`
- `SHA256SUMS` SHA-256: `0463709a2d0cd198fdb386763f092042aa5763ea76e60f9ba7b4d944c8eadd1c`
- Final report SHA-256: `8f4ee5ade58d6bb919622078e13f6a888b380908670986e7465565e7dafd3d28`

## Run 2: production settings

Settings were launcher-level `reasoning_effort=max`, launcher-level
`top_p=0.95`, and explicit `VLLM_USE_DEEP_GEMM=1`. Startup confirmed effective
sampling `temperature=1.0`, `top_p=0.95`,
`DeepGemmFp8BlockScaledMMKernel`, and `DEEPGEMM_MXFP4`. Because the frozen
baseline had already selected the same DeepGEMM kernels, DeepGEMM was
controlled rather than a newly introduced performance variable.

The host production launcher is preserved at
`/opt/ai-artifacts/pennyroyal-moet-DS4flash.sh`; its SHA-256 is
`e039e22705e20211da0f6121e6cd070aaca1466793b39172ca4748130ae9d995`.

| Suite | Result | Token usage | Throughput |
|---|---:|---|---|
| Reasoning, 9 measured requests | 96.86/100; no fatal findings | 6,081 server prompt; 35,864 server generation; 35,689 API completion; 35,408 locally counted generation | 58.44 server decode tok/s; 57.17 effective local tok/s |
| Tool/agent, 30 invocations | 30/30 automatic and reviewed | 70,207 server prompt; 5,591 server/API completion | 60.24 server decode tok/s; 42.46 effective completion tok/s |

C5 materially improved: it rejected self-transfers, handled missing accounts,
persisted the insufficient-funds outcome, compared request fingerprints, used
canonical dual-account locks, and relied on rollback after a uniqueness
conflict. Remaining C5 gaps were failure to persist every invalid outcome by
request ID, checking positivity without explicitly requiring integer cents,
and one missing-row behavior assumption not supplied by the case. C8 was fully
correct on both its initial and correction turns. No OOM, crash, loop, retry,
parser failure, length termination, heat stop, or runtime instability occurred.

Sealed local evidence:

- `/opt/ai-artifacts/logs/clean-native-mapped-w2-max-top095-deepgemm-20260802-064459`
- Manifest SHA-256: `8c4a1d036bea36daf90f95dec5d460b8c6f599330fd0eb43cf12c3860fd1f84f`
- Final report SHA-256: `b9a84ef244786906a50b19e3115074fe276f814ba2722b07debca45e502f0cdc`

## Production conclusion and limits

The production settings improve the single-pass frozen reasoning score by
4.20 points while preserving the 30/30 reviewed tool result. Server decode
changed from 56.14 to 58.44 tok/s on reasoning and from 61.07 to 60.24 tok/s
on tools; the latter -1.4% is treated as run noise rather than a material
regression. The two settings changed together in one stochastic pass, so the
individual contribution of `reasoning_effort` versus `top_p` is not isolated.

The tool suite again reached a sampled minimum of 3 MiB physical free VRAM.
The passes establish frozen-suite behavior, not arbitrary-workload headroom,
soak stability, a 393,216-token request, or container validation.
