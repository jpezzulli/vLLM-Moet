# How mapped-host W2 works

## Memory placement

DeepSeek-V4-Flash-0731 has target-model MoE W2 layers followed by three W2
layers used by its DSpark draft model. The validated layout is:

| Layer keys | Placement | Role |
|---|---|---|
| 0–37 | GPU VRAM | target-model W2 |
| 38–42 | NUMA-local pinned host RAM | target-model W2 read over PCIe |
| 43–45 | GPU VRAM | DSpark W2 |

Each selected target layer occupies exactly 1,811,939,328 bytes (1.6875 GiB),
so five mapped layers place 9,059,696,640 bytes (8.4375 GiB) outside VRAM.
The DSpark layers remain resident because speculative decoding repeatedly
uses them and because the validated recipe does not implement host-backed
draft W2.

## Canonical construction, not offload replay

Mapped W2 is selected before the affected canonical tensors are constructed.
For each selected complete layer, the runtime:

1. allocates page-aligned host virtual memory;
2. first-touches and binds the pages on the selected GPU's local NUMA node;
3. registers the allocation as mapped, pinned CUDA host memory;
4. obtains and validates the UVA device pointer;
5. constructs the layer's complete W2 planes and scales directly in that
   allocation; and
6. installs those stable pointers into the normal Runner V2 W2 path.

There is no second complete W2 allocation in VRAM. There is also no host
replay, automatic `cudaMalloc` overflow, per-step upload, or GPU staging cache.
The existing SM120 kernels dereference the mapped device pointers and read the
selected expert planes directly across PCIe.

## NUMA locality and auditing

The runtime identifies the actual visible GPU, resolves its PCI bus address,
reads the corresponding sysfs NUMA node, and verifies that node is usable by
the process. A valid explicit NUMA or PCI value acts as a guard. On a
multi-node system, unresolved locality is an actionable startup failure rather
than an implicit choice of node 0.

After allocation, page placement is verified through Linux page-location
interfaces. The JSON audit records configured layers, byte geometry, device
and host pointers, page-node counts, UVA equality, kernel dispatch state, and
redundant GPU W2 bytes. The validated five-layer audit reports zero redundant
complete GPU bytes.

## CUDA graph lifetime

Mapped allocations are owned for the full model-runner lifetime. Their device
pointers remain stable through eager setup, DeepGEMM warm-up, full graph
capture, piecewise graph capture, DSpark capture, and serving. Synchronization
precedes rollback and shutdown cleanup; partial construction failures release
only allocations whose ownership was successfully established.

This path is deliberately narrow. Unsupported builders, malformed layer
selectors, missing UVA support, NUMA mismatch, registration failure, pointer
mismatch, or duplicate complete allocation fail closed.

## What uses the recovered VRAM

The final serving shape retains a fixed FP4 correction tier of 512 slots at
12 MiB per slot: 6 GiB total. Frequently routed experts can use the correction
tier while the compact W2 base remains the default execution representation.
FP8 MLA KV receives the remaining vLLM budget.

At `gpu_memory_utilization=0.974`, the final startup reported 4.96 GiB of KV
allocation and capacity for 1,058,256 tokens. The admission limit is
1,000,000. Allocation capacity and configured admission are distinct from the
994,987-token request that was actually exercised.

## Tradeoff

Direct PCIe reads reduce VRAM residency at the cost of decode throughput and
substantial PCIe traffic. On the representative DSpark-4 suites, PCIe RX
peaked near 30 GiB/s and model generation averaged about 56.49 tok/s. This is
an intentional capacity-for-throughput trade: the goal is a coherent,
agent-capable 159B model with exceptional single-request context on one 96 GB
GPU, not maximum short-context tokens per second.
