# Native build and run

This is the preferred reproduction path for the validated single-card
DeepSeek-V4-Flash-0731 configuration. It builds the directly pinned runtime
source used by the final hardware tests; it does not apply or regenerate the
repository's generated publication patch.

## Supported target

- Linux x86_64
- one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120), 96 GB
- GPU-local NUMA topology visible through PCI sysfs
- CUDA mapped-host memory and UVA
- at least 192 GiB host RAM and a memlock limit sufficient for 9,059,696,640
  mapped W2 bytes plus normal runtime allocations
- CUDA toolkit 13.3, GCC/G++ 15, CMake 3.26.1 or newer, Ninja, `uv`, Git,
  `curl`, and the Hugging Face `hf` CLI when downloading the checkpoint

The validated software pins are listed in [PROVENANCE.md](PROVENANCE.md).
The important runtime source is
`jpezzulli/vllm@a2131dd7a944353e9323566107c72f4a17441024`.

The validated core environment used Python 3.14, PyTorch 2.11.0+cu130,
TorchVision 0.26.0+cu130, TorchAudio 2.11.0+cu130, FlashInfer 0.6.14,
`flashinfer-jit-cache` 0.6.14+cu130, Transformers 5.14.1, Tokenizers 0.22.2,
Triton 3.6.0, and DeepGEMM 2.5.0 at the commit above. The build script pins
the compatibility-sensitive packages and derives the remaining requirements
from the pinned vLLM source tree.

## 1. Get this repository and the checkpoint

```bash
git clone https://github.com/jpezzulli/vLLM-Moet.git
cd vLLM-Moet
git checkout rtx-pro6000

hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  --local-dir /srv/models/hf/ds4flash0731
```

The model path is a default, not a requirement. Set `MODEL_PATH` when serving
from another location. The MoET pack directory is a persistent quantization
cache; it avoids rebuilding existing FP4 data but is not the backing store for
mapped W2 layers.

## 2. Build the native environment

```bash
MAX_JOBS=16 ./scripts/build-native.sh
```

The script uses strict mode, checks its prerequisites, and installs beneath
`.native/` by default. Override `NATIVE_ROOT`, `CUDA_HOME`, `CC`, `CXX`, or
`MAX_JOBS` explicitly when required.

The build deliberately uses two Python environments:

- Python 3.13 builds the stable-ABI vLLM extensions because the pinned
  FlashAttention build does not support Python 3.14;
- Python 3.14 is the serving environment used in validation.

Dependencies are installed in the same staged order as the clean native
reproduction: PyTorch 2.11.0+cu130 first, vLLM requirements next, then
FlashInfer 0.6.14 without allowing it to replace PyTorch. DeepGEMM is built
from its pinned commit. `VLLM_USE_PRECOMPILED=1` is intentionally unsupported
because it previously produced an operator-signature mismatch.

The script itself is a reviewable consolidation of the recorded build; this
exact convenience script has not been used for a fresh full build in this PR.
The underlying staged method and native extensions were validated on the
target host.

## 3. Start the production shape

```bash
MODEL_PATH=/srv/models/hf/ds4flash0731 \
SERVED_MODEL_NAME=pennyroyal \
MOET_STORE_DIR=/srv/models/moet-packs/DeepSeek-V4-Flash \
./scripts/serve-pro6000-ds4flash.sh
```

Useful path overrides are:

| Variable | Default |
|---|---|
| `NATIVE_ROOT` | repository `.native/` directory |
| `MODEL_PATH` | `/srv/models/hf/ds4flash0731` |
| `SERVED_MODEL_NAME` | `pennyroyal` |
| `PORT` | `8001` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| `MOET_STORE_DIR` | `/srv/models/moet-packs/DeepSeek-V4-Flash` |
| `CACHE_ROOT` | `/srv/cache/vllm-moet` |
| `W2_AUDIT_PATH` | `/tmp/pennyroyal-mapped-w2-audit.json` |

The serving geometry is intentionally fixed in the launcher: mapped layers
38–42, GPU DSpark layers 43–45, DSpark-4, 6 GiB FP4 correction, FP8 MLA KV,
four sequences, 2,048 batched tokens, `gpu_memory_utilization=0.974`, and a
1,000,000-token admission limit.

The runtime automatically resolves the selected visible GPU's PCI BDF and
local NUMA node. Use `VLLM_MOE_W2_MAPPED_NUMA_NODE` only as an explicit guard
or override on unusual systems. Multi-node systems fail closed if locality
cannot be determined.

## 4. Verify startup and the API

Wait for model loading, KV initialization, DeepGEMM warm-up, and every full,
piecewise, and DSpark graph capture to finish. Startup must report:

- Runner V2 and DSpark-4;
- target W2 layers 38–42 mapped and layers 43–45 GPU-resident;
- 512 correction slots × 12 MiB = 6 GiB;
- FP8 MLA KV and a 1,000,000-token admission limit;
- 9,059,696,640 mapped bytes and zero redundant complete GPU W2 bytes.

Then run:

```bash
BASE_URL=http://127.0.0.1:8001 \
SERVED_MODEL_NAME=pennyroyal \
W2_AUDIT_PATH=/tmp/pennyroyal-mapped-w2-audit.json \
./scripts/validate-api.sh
```

This lightweight check verifies the import when the local environment is
available, `/health`, `/v1/models`, the mapped audit, and one small arithmetic
response. It does not rerun the frozen quality or million-token tests.

## Container alternative

The published v4 OCI image is an immutable reproduction of the earlier
three-mapped-layer DSpark-3/393,216-token candidate. It used runtime
`95ef4a88c63c9ed88f2384977e05d788897af6c3` and was sealed by publication
commit `0544e69e63dce5a9cf597797df3db140391ba832`. It is useful for checking
container packaging and topology handling, but it is not the final a2131
DSpark-4 one-million-token configuration. See
[the container record](docs/container-ds4flash-0731-sm120.md).
