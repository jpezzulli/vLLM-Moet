# How mapped-host W2 works

## Memory placement

DeepSeek-V4-Flash-0731 has target-model MoE W2 layers followed by three W2
layers used by its DSpark draft model. The current capacity layout is:

| Layer keys | Placement | Role |
|---|---|---|
| 0–41 | GPU VRAM | target-model W2 |
| 42 | NUMA-local pinned host RAM | target-model W2 read over PCIe |
| 43–45 | NUMA-local pinned host RAM | DSpark W2 read over PCIe |

Each selected complete layer occupies exactly 1,811,939,328 bytes
(1.6875 GiB), so four mapped layers place 7,247,757,312 bytes (6.75 GiB)
outside VRAM. The performance sibling maps only DSpark layers 43–45, placing
5,435,817,984 bytes (5.0625 GiB) outside VRAM while keeping every target W2
layer resident.

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
redundant GPU W2 bytes. The exercised three- and four-layer audits report zero
redundant complete GPU bytes.

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

The 300K performance profile uses `gpu_memory_utilization=0.974` and reported
capacity for 378,490 KV tokens. The 512K capacity profile uses
`gpu_memory_utilization=0.98446` and reported capacity for 901,924 KV tokens.
Their configured admission limits are 300,000 and 524,288 respectively.
Allocation capacity and configured admission are distinct from the exact
250,000- and 500,000-token inputs actually exercised.

## Tradeoff

Direct PCIe reads reduce VRAM residency at the cost of decode throughput and
substantial PCIe traffic. The bounded 300K profile reached 89.82 tok/s for one
exact 1,024-token decode and 152.31 tok/s aggregate for four; the 512K profile
measured 68.23 and 126.83 tok/s respectively. This is an intentional
capacity-for-throughput trade, not a claim that every prompt has those rates.
