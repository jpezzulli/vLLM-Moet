# Mapped-host W2 architecture

> **Historical implementation detail:** this document describes the earlier
> three-layer DSpark-3 candidate. The current five-layer DSpark-4 production
> architecture is [HOW-IT-WORKS.md](../HOW-IT-WORKS.md).

## Scope

This feature places an explicit set of complete Runner V2 MXFP4-derived W2
layers in CUDA-mapped, page-locked host memory. It is not a general CUDA
allocator, an automatic `cudaMalloc` overflow path, or a host replay cache.
The validated shape maps target layer keys 40–42 and leaves the three DSpark
layers 43–45 resident on the GPU.

## Serving path

DeepSeek-V4-Flash's routed experts enter MoET through the MXFP4 builder. For a
selected layer, the builder calculates the exact sizes of `planes13`, `sc13`,
`planes2`, and `sc2` before allocating any complete output tensor. One
`cudaHostAllocMapped | cudaHostAllocPortable` allocation is made for the
layer, and the four tensors are non-overlapping views of that allocation.
Quantization writes directly into these canonical views.

The allocation owner obtains the CUDA-visible pointer with
`cudaHostGetDevicePointer`. Under UVA on the validated platform, the host and
device pointer values are equal. The normal Runner V2 descriptor path retains
those pointers and the existing W2 cubins dereference them over PCIe. No
complete GPU W2 output is constructed first, copied aside, replayed, promoted,
or kept as a fallback.

The FP4 correction tier is separate. In the validated configuration it holds
512 slots of 12 MiB each, exactly 6 GiB. It remains GPU-resident and follows
the existing frequency policy. Mapping the 2-bit W2 base does not change the
FP4 tier's slot format or allocation mechanism.

DSpark uses three model-declared MTP layers. Runner V2 validates their layer
metadata and leaves them GPU-resident. The target W2 mapping selector contains
only 40–42; selecting an unsupported FP8 or modelopt-NVFP4 W2 builder fails
before that builder can silently create a complete GPU W2 representation.

## NUMA and CUDA lifetime

The active CUDA device's PCI address is derived from PyTorch. Sysfs supplies
its NUMA node and local CPU list. During each allocation, the process's current
CPU affinity is narrowed to allowed CPUs local to that GPU and Linux memory
policy is bound to the local node. Both are restored immediately afterward.
Every page is then queried with `move_pages`; startup fails if any page is not
on the resolved node. Optional PCI and NUMA settings act as guards against a
wrong device or host topology, not as topology discovery.

The four tensor views and their owning allocation remain in process-global
model state for the full model and CUDA-graph lifetime. Quantization completes
with an explicit CUDA synchronization before the descriptors are installed.
Normal full, piecewise, and DSpark graph capture then sees stable addresses.
Worker shutdown destroys model graphs first, drops the tensor views, performs
one device synchronization, and releases each allocation with `cudaFreeHost`.
Partial construction failures free their allocation before propagating an
actionable error.

## Accounting and duplicate protection

Startup logs report each mapped layer's bytes, UVA pointer, PCI function, NUMA
node, and page counts. The optional JSON audit records tensor offsets,
pointers, devices, allocation totals, construction completion, the first
production-cubin dispatch, and `redundant_gpu_w2_bytes`. Construction is
rejected if a canonical tensor is not a CPU view wholly contained in its
mapped allocation.

For DeepSeek-V4-Flash TP1, one layer is 1,811,939,328 bytes (1.6875 GiB) and
three layers are 5,435,817,984 bytes (5.0625 GiB). These bytes consume host RAM
and pinned CUDA resources, while GPU reads traverse PCIe. They are outside
vLLM's `gpu_memory_utilization` budget. The separate 6 GiB FP4 tier is also
allocated by MoET outside that budget; the validated 0.988 setting is an
observed working configuration, not an accounting formula that applies to
other shapes.

Mapping is disabled by default, so existing Runner V1 and tensor-parallel
behavior is unchanged. With mapping enabled, each worker resolves its own
active CUDA device and local NUMA topology. Multi-rank configurations have not
been hardware-validated; explicit topology guards must match each worker or
startup fails.
