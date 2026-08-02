# Mapped-host configuration

## Mapped W2 options

| Option | Default | Meaning and constraints |
|---|---|---|
| `VLLM_MOE_W2_MAPPED_LAYERS` | empty | Comma-separated non-negative MoET layer keys. Empty disables mapped W2. Duplicate, negative, or empty entries fail. The validated recipe uses `40,41,42`. |
| `VLLM_MOE_W2_MAPPED_NUMA_NODE` | auto | Optional non-negative node guard. The active GPU's node is always derived from sysfs; a mismatch fails. The validated host resolved node 0. |
| `VLLM_MOE_W2_MAPPED_PCI` | auto | Optional active-GPU BDF guard, such as `0000:31:00.0`. A mismatch fails. It does not select a GPU. |
| `VLLM_MOE_W2_MAPPED_AUDIT_PATH` | unset | Optional JSON audit output. Its parent is created and the file is atomically replaced as state changes. |

Mapped layers require `VLLM_USE_V2_MODEL_RUNNER=1`, the MXFP4 W2 builder, a
Linux x86_64 host, CUDA mapped-host support, UVA, and a GPU with resolvable
sysfs NUMA locality. They cannot be combined with
`VLLM_MOE_W2_BASE_CACHE_GB`. Only complete canonical layers are supported;
there is no partial-tensor selector.

## Validated serving shape

| Setting | Value | Tradeoff or constraint |
|---|---:|---|
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | Required for the mapped path and DSpark lifecycle. |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `0` | Preserves the validated normal graph path. |
| `VLLM_USE_DEEP_GEMM` | `1` | Explicit production setting. Startup selected the same DeepGEMM FP8 and MXFP4 kernels in both frozen quality runs; this is not a newly measured speed optimization. |
| `VLLM_MOE_W2` | `1` | Enables MoET W2. |
| `VLLM_MOE_W2_FORCE_RESIDENT` | `1` | User consent to continue past the existing conservative resident estimate; a real shortfall can still OOM. |
| `VLLM_MOE_W2_DELTA_GB` | `6` | Exactly 512 FP4 correction slots at 12 MiB/slot for this model. This tier is outside vLLM's utilization budget. |
| `VLLM_MOE_W2_DELTA_RESERVE_GB` | `0` | Prevents the automatic 3 GiB post-KV reserve from reducing the fixed tier. |
| `VLLM_MOE_W2_MTP_LAYERS` | `3` | Keeps the model-declared DSpark-3 layer set. |
| `VLLM_MOE_W2_DELTA_POLICY` | `freq` | Existing FP4 slot policy. |
| `VLLM_MOE_W2_PREFILL_FP4` | `1` | Existing FP4 prefill behavior used by the sealed run. |
| `VLLM_MOE_W2_GATE` | `0` | Confidence-gate replay disabled in the validated candidate. |
| `VLLM_MOE_W2_AFRAG` | `1` | Existing fragment-major prefill path enabled. |
| `VLLM_MOE_W2_FUSED_UNPERMUTE` | `0` | Keeps the validated legacy reduction path. |
| `--gpu-memory-utilization` | `0.988` | vLLM model/KV budget. Mapped W2 and MoET FP4 allocations are external to it. |
| `--max-model-len` | `393216` | Configured admission limit only; 393,216 tokens were not exercised. |
| `--kv-cache-dtype` | `fp8` | Validated MLA KV type. |
| `--block-size` | `256` | Validated KV block shape. |
| `--max-num-batched-tokens` | `2048` | Validated profile/prefill chunk shape. |
| `--max-num-seqs` | `4` | Configured scheduler concurrency; the bounded decode result was single-request. |
| `--default-chat-template-kwargs` | thinking enabled, `reasoning_effort=max` | Production quality setting. The separately preserved frozen baseline used `high`. |
| `--override-generation-config` | `{"top_p":0.95}` | Production quality setting. The separately preserved frozen baseline used the checkpoint default `top_p=1.0`. |
| `--speculative-config` | DSpark, 3 tokens, greedy draft | Validated speculative depth and draft method. |
| `--compilation-config` | full and piecewise graphs, all custom ops | Eager mode is not the supported candidate. |

The fixed recipe is
`bench/recipes/deepseek-v4-flash/pro6000x1-mapped-w2-dspark3.yaml`. Do not
infer a safe slot count, context, or utilization value for another checkpoint
or GPU from this one result.

## Memory interpretation

The validated three-layer mapping moved 5.0625 GiB of complete W2 base storage
to host RAM. The runtime still needed the dense model, 40 target W2 layers,
three GPU-resident DSpark layers, graph workspaces, KV, and the 6 GiB FP4 tier
on the GPU. After graph capture, `nvidia-smi` reported 1,057 MiB physically
free. vLLM reported capacity for 625,757 KV tokens, but that is allocation
capacity rather than an exercised request length.
