# DeepSeek-V4-Flash-0731 mapped-W2 OCI image

This recipe builds the complete Pennyroyal serving image from declared source
inputs. It does not copy a virtual environment, extension, wheel, or generated
Python package from a live installation.

The image is intentionally narrow:

- one NVIDIA RTX PRO 6000 (SM120), selected through CDI by physical UUID;
- vLLM base `ee0da84ab9e04ac7610e28580af62c365e898389` plus the generated
  vLLM-MoET patch sourced from runtime commit
  `95ef4a88c63c9ed88f2384977e05d788897af6c3`;
- Runner V2 with full, piecewise, and DSpark CUDA graphs;
- main-model W2 layers 0–39 GPU-resident and complete layers 40–42 mapped
  from GPU-local pinned host memory;
- DSpark W2 layers 43–45 GPU-resident, speculative depth 3;
- 512 FP4 correction slots at 12 MiB each (6 GiB total);
- FP8 MLA KV, explicit DeepGEMM, `gpu_memory_utilization=0.988`; and
- configured admission limit 393,216 tokens.

The exact v4 image completed clean startup and smoke validation on `thegrid`
on 2026-08-02. The runtime reported 625,415 tokens of KV capacity. That value
and the 393,216-token configured limit are capacity/admission facts, not
exercised-context results. Concurrency, soak, other GPUs, other checkpoints,
tensor parallelism, and orchestration remain unvalidated.

## Why v3 failed

Runtime commit `95ef4a88…` contains only the FlashAttention package shell and
interface; it does not check in the `cute` subtree. The pinned vLLM CMake
component `_vllm_fa4_cutedsl_C` generates that subtree by copying the pinned
FlashAttention `flash_attn/cute/*.py` files and rewriting their package imports
to `vllm.vllm_flash_attn.cute`. The validated native installation contained
that generated Python source.

The v3 recipe installed vLLM with `VLLM_TARGET_DEVICE=empty` and staged only
selected native extensions. That path correctly skipped native compilation,
but it also skipped the CMake copy step and did not stage the generated Python
source. Package discovery, `.dockerignore`, and the sanctioned runtime patch
were not the cause. The correction is packaging-only: v4 performs the full
pinned source build and copies its declared outputs into a clean final stage.

## Reproducible sources

The Containerfile pins the Fedora base by digest, reconstructs vLLM from its
base commit and sanctioned generated patch, and compiles the stable-ABI vLLM
extensions. The pinned vLLM build fetches its declared FlashAttention source
at `803020a8fa15407871341d41eba4919ade2ee1ee`; that build generates and
installs `vllm.vllm_flash_attn.cute` along with the FA2 and FA3 extensions.
DeepGEMM is built from `a6b593d2826719dcf4892609af7b84ee23aaf32a`.

The exact Python lock, RPM inventory, pip install report, CUDA compiler
identity, source commits, and hashes of source-built artifacts are stored in
`/opt/provenance` in the image. See
[`SOURCE-PROVENANCE.md`](../container/ds4flash-0731-sm120/SOURCE-PROVENANCE.md)
for the complete input map.

## Build

From the repository root:

```bash
podman build --format oci --pull=never --layers \
  -f Containerfile.ds4flash-0731-sm120 \
  -t localhost/jpezzulli/vllm-moet:ds4flash-0731-sm120-v4-candidate .
```

The equivalent Docker BuildKit command is:

```bash
DOCKER_BUILDKIT=1 docker build --pull=false \
  -f Containerfile.ds4flash-0731-sm120 \
  -t localhost/jpezzulli/vllm-moet:ds4flash-0731-sm120-v4-candidate .
```

The immutable Fedora image must already be available when pulls are disabled.
Source repositories and pinned Python distributions are fetched during the
build. No model, MoET pack, cache, secret, host extension, or validation log is
included in the build context.

Before using a GPU, run the reusable import, linkage, and `Python.h` probe:

```bash
container/ds4flash-0731-sm120/probe-image.sh \
  localhost/jpezzulli/vllm-moet:ds4flash-0731-sm120-v4-candidate
```

The probe mounts only the host driver libraries read-only. It does not expose
GPU device nodes or load the model.

## Host contract

Podman or Docker must expose exactly one physical GPU through NVIDIA CDI, by
UUID. The entrypoint resolves the visible UUID and PCI BDF, reads the GPU's
local NUMA node from PCI sysfs, and checks that this node is permitted by the
effective CPU and memory-node constraints. It logs the resolved topology and
affinity before starting vLLM.

If sysfs reports `-1`, node fallback is permitted only when the container has
one effective memory node. Multi-node hosts with unresolved locality must set
the existing `VLLM_MOE_W2_MAPPED_NUMA_NODE=<node>` override. The optional
`VLLM_MOE_W2_MAPPED_PCI=<BDF>` guard rejects a mismatched GPU. Public examples
do not hard-code node 0.

The host user needs read access to the model and read/write access to the pack,
cache, and audit directories. The existing checkpoint and pack are mounted;
they are not copied. `--ipc=host` is required by vLLM. The narrow repository
seccomp profile is Podman's default profile with only `move_pages` allowed, so
the runtime can verify mapped-page NUMA placement. Do not replace it with
`--privileged` or an unconfined seccomp policy.

Create the small writable directories once:

```bash
mkdir -p /srv/cache/vllm-moet-container /opt/vllm-moet/container-validation-output
```

## Podman run

```bash
podman run --rm --name pennyroyal-moet \
  --device nvidia.com/gpu=GPU-<UUID> \
  --security-opt label=disable \
  --security-opt seccomp=$(pwd)/container/ds4flash-0731-sm120/seccomp-mapped-w2.json \
  --ipc=host \
  -p 127.0.0.1:8001:8001 \
  -v /srv/models/hf/ds4flash0731:/models/ds4flash0731:ro \
  -v /srv/models/moet-packs/DeepSeek-V4-Flash:/packs/DeepSeek-V4-Flash:rw \
  -v /srv/cache/vllm-moet-container:/cache:rw \
  -v /opt/vllm-moet/container-validation-output:/run/vllm-moet:rw \
  ghcr.io/jpezzulli/vllm-moet:ds4flash-0731-sm120-v4
```

Use a different host-side port, such as `18001:8001`, if port 8001 is already
occupied.

## Docker run

Docker Engine installations with NVIDIA CDI enabled use the same topology-
relative selection:

```bash
docker run --rm --name pennyroyal-moet \
  --device nvidia.com/gpu=GPU-<UUID> \
  --security-opt seccomp=$(pwd)/container/ds4flash-0731-sm120/seccomp-mapped-w2.json \
  --ipc=host \
  -p 127.0.0.1:8001:8001 \
  -v /srv/models/hf/ds4flash0731:/models/ds4flash0731:ro \
  -v /srv/models/moet-packs/DeepSeek-V4-Flash:/packs/DeepSeek-V4-Flash:rw \
  -v /srv/cache/vllm-moet-container:/cache:rw \
  -v /opt/vllm-moet/container-validation-output:/run/vllm-moet:rw \
  ghcr.io/jpezzulli/vllm-moet:ds4flash-0731-sm120-v4
```

The image has a sealed serving command. Only mount-path variables documented
in `entrypoint.sh` and the two topology overrides above are accepted.

Follow startup with `podman logs --follow pennyroyal-moet` and stop cleanly
with `podman stop --time 120 pennyroyal-moet`.

## Validated startup geometry

The clean v4 run resolved GPU
`GPU-02eb6916-662d-631d-e541-46b3c977ed78`, PCI `0000:31:00.0`, and NUMA node
0 automatically. It verified 442,368 pages on node 0 for each mapped layer,
5,435,817,984 mapped bytes in total, and no redundant complete GPU W2 copy.
Model load used 87.92 GiB. All 5 piecewise, 3 full, and 2 DSpark graph-capture
shapes completed. Physical free VRAM was 1,077 MiB immediately after capture
and 831 MiB after the smoke requests.

`/health` and `/v1/models` returned HTTP 200. A reasoning request correctly
returned 323 for 17 × 19. The exact required tool request returned
`finish_reason=tool_calls`, selected `get_weather`, and emitted
`{"city":"Boston","unit":"celsius"}`. These are smoke checks, not the frozen
quality suites.

The local evidence directory for that run was
`/opt/vllm-moet/container-validation-output/v4-clean-20260802`. Local paths are
recorded here only to identify the validation source; they are not image build
inputs or runtime requirements.
