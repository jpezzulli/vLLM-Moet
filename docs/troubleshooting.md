# Mapped-host troubleshooting

> This page retains implementation-level troubleshooting from the earlier
> candidate. Apply the final geometry and claim boundaries from
> [BUILD-AND-RUN.md](../BUILD-AND-RUN.md) and [VALIDATION.md](../VALIDATION.md).

## CUDA allocation or mapping failure

`cudaHostAllocMapped` consumes page-locked system RAM, not swap-backed virtual
capacity. Check available host RAM, per-process limits, driver health, and
whether another process holds large pinned allocations. The startup error
includes the layer key, requested GiB, and NUMA node. Do not respond by
intercepting `cudaMalloc`, faking VRAM, or silently changing the selected
layer set.

The CUDA Runtime releases these allocations with `cudaFreeHost`, not
`cudaFree`. Linux `VmLck` and `/proc/meminfo` `Mlocked` did not reflect the
3.375 GiB standalone CUDA allocation on the validated driver, so zero in those
counters does not prove the pages are unpinned. The CUDA API result, mapped
device pointer, direct kernel access, and page-placement audit are the useful
checks.

## NUMA mismatch

Inspect the active GPU rather than assuming its node:

```bash
nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv
nvidia-smi topo -m
cat /sys/bus/pci/devices/0000:31:00.0/numa_node
cat /sys/bus/pci/devices/0000:31:00.0/local_cpulist
numactl --hardware
```

The validated RTX PRO 6000 was at `0000:31:00.0` on NUMA node 0; the second
GPU on that host was on node 1. If an explicit PCI or NUMA guard disagrees
with the active CUDA device, fix the launch selection or remove the guard and
allow automatic discovery. Do not weaken page verification.

## Unsupported W2 builder or base cache

Mapped W2 currently supports the MXFP4-derived Runner V2 path. Selecting a
layer built by the FP8 or modelopt-NVFP4 W2 builder fails before a complete GPU
copy is allocated. `VLLM_MOE_W2_BASE_CACHE_GB` is also incompatible because it
is a separate host-store/GPU-cache architecture. Use one mechanism, not both.

## CUDA graph capture

Eager mode is not the candidate. The mapped allocation owner must remain live
through full, piecewise, and DSpark capture and through inference. A graph
capture failure should be investigated at the reported CUDA operation and
shape. Do not build a duplicate eager-only W2 set. Confirm the first production
cubin dispatch and stable pointers in the audit file.

## Native operator signature mismatch

An error such as `_moe_C::topk_softplus_sqrt() is missing value for argument
is_padding` means the Python source and loaded native extension were built from
different vLLM revisions. A clean-checkout validation reproduced this when
`VLLM_USE_PRECOMPILED=1` downloaded a moving development extension for the
pinned v0.24.0 source. It is a binary/source ABI mismatch, not a mapped-W2 or
model-quality failure. Remove the mismatched extension and follow
`docs/native-build-and-run.md` to build the stable-ABI extensions from the
pinned checkout; do not change the Python call to match an unrelated binary.

## OOM and accounting

`--gpu-memory-utilization` controls vLLM's model/KV budget. The MoET FP4 tier
and mapped host allocations are external. In the validated shape, the mapped
5.0625 GiB is host memory, while the 6 GiB FP4 tier is still GPU memory. The
0.988 value worked with exactly three mapped layers and 512 FP4 slots; it is
not a universal safe percentage.

Compare both allocator and physical readings:

```bash
nvidia-smi --query-gpu=memory.used,memory.free,power.draw,temperature.gpu \
  --format=csv -l 1
curl -s http://127.0.0.1:8001/metrics
```

The sealed run had 1,057 MiB physical free after capture. If a clean run is
materially lower, inspect unexpected GPU processes, resolved FP4 slots, DSpark
depth, graph shapes, and whether a selected W2 layer was actually mapped.
Do not retry the failed 7 GiB shape as a troubleshooting step.

## PCIe behavior

Mapped W2 reads cross PCIe by design. Use GPU PCIe RX/TX telemetry where
available and compare request-matched workloads. The sealed 1,024-token decode
averaged 8,564 MB/s PCIe RX and peaked at 12,393 MB/s. High traffic is not
itself an error; a collapse in throughput accompanied by remote-NUMA pages is.

## Capacity versus exercised context

The engine reported KV capacity for 625,757 tokens and accepted a configured
limit of 393,216 tokens. Neither number demonstrates that a 393,216-token
request completed or retrieved correctly. Documentation and issue reports must
state capacity, configured limit, and exercised prompt length separately.
