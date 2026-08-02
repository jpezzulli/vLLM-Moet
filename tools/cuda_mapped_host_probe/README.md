# CUDA mapped-host diagnostic

This standalone diagnostic proves the CUDA/NUMA mechanism without loading
vLLM or a model. It allocates exactly 3,623,878,656 bytes (3.375 GiB) with
`cudaHostAllocMapped`, verifies every backing page's NUMA placement, checks
GPU reads and writes, and measures sequential, random-plane, and simulated
two-layer decode traffic.

Build on SM120 with the installed CUDA toolkit:

```bash
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 \
  -ccbin /usr/bin/g++-15 -lineinfo -Xcompiler=-Wall,-Wextra \
  tools/cuda_mapped_host_probe/mapped_host_probe.cu \
  -o /tmp/mapped_host_probe
```

Run with an output directory, the GPU-local NUMA node, and an allowed CPU on
that node:

```bash
mkdir -p /tmp/mapped-host-results
/tmp/mapped_host_probe /tmp/mapped-host-results 0 0
```

Confirm locality first with `nvidia-smi topo -m`, the GPU's
`/sys/bus/pci/devices/<BDF>/numa_node`, and `numactl --hardware`. The probe is
destructive only to its own process-local CUDA resources and output files.

The preserved 2026-08-01 run used source SHA-256
`f16313e8da07301b1fe908a159e005cfa3d61e6779b1f28e22fdda60cb35ed64`.
It passed correctness and measured 24.089 GiB/s for the simulated 81 MiB
two-layer step, or 304.53 steps/s. That result proves the standalone access
mechanism, not end-to-end MoET performance.
