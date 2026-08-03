# DS4Flash mapped-W2 container files

- `entrypoint.sh` seals the validated serving geometry.
- `topology-preflight.py` resolves and validates GPU-local NUMA placement.
- `seccomp-mapped-w2.json` allows only the additional `move_pages` syscall.
- `requirements-runtime.lock.txt` pins the Python runtime environment.
- `cuda-fedora44.repo` declares the signed CUDA 13.3 RPM source.
- `probe-image.sh` verifies imports, native linkage, and `Python.h` without
  exposing a GPU.
- `SOURCE-PROVENANCE.md` records the complete source and integrity map.

Build and run instructions are in
[`docs/container-ds4flash-0731-sm120.md`](../../docs/container-ds4flash-0731-sm120.md).
