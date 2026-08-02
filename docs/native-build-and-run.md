# Native build and run

This procedure builds a native editable vLLM checkout from the generated
vLLM-MoET patch. It does not build or validate a container.

## Clean checkout

```bash
git clone https://github.com/jpezzulli/vLLM-Moet.git /opt/vllm-moet-clean
git -C /opt/vllm-moet-clean checkout agent/mapped-host-w2-public

git clone --branch v0.24.0 https://github.com/vllm-project/vllm.git \
  /opt/vllm-v0.24.0-clean
git -C /opt/vllm-v0.24.0-clean apply --check \
  /opt/vllm-moet-clean/patch/vllm-moet-v0.24.0.patch
git -C /opt/vllm-v0.24.0-clean apply \
  /opt/vllm-moet-clean/patch/vllm-moet-v0.24.0.patch
```

Verify the source pin before installing:

```bash
python3 /opt/vllm-moet-clean/tools/check_patch_files.py
cat /opt/vllm-moet-clean/patch/SOURCE.txt
```

## Python environment

The validated native environment used Python 3.14, CUDA 13.1, GCC/G++ 15,
PyTorch `2.11.0+cu130`, FlashInfer `0.6.14`,
`flashinfer-jit-cache==0.6.14+cu130`, and DeepGEMM commit
`a6b593d2826719dcf4892609af7b84ee23aaf32a`.

```bash
python3.14 -m venv /opt/vllm-moet-clean/.venv
source /opt/vllm-moet-clean/.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

export CUDA_HOME=/usr/local/cuda
export CC=/usr/bin/gcc-15
export CXX=/usr/bin/g++-15
export NVCC_CCBIN=/usr/bin/g++-15

cd /opt/vllm-v0.24.0-clean
VLLM_USE_PRECOMPILED=1 python -m pip install -e . \
  --no-build-isolation
python -m pip install --upgrade \
  flashinfer-python==0.6.14 \
  'flashinfer-jit-cache==0.6.14+cu130'

git clone https://github.com/deepseek-ai/DeepGEMM.git \
  /opt/vllm-moet-clean/DeepGEMM
git -C /opt/vllm-moet-clean/DeepGEMM checkout \
  a6b593d2826719dcf4892609af7b84ee23aaf32a
cd /opt/vllm-moet-clean/DeepGEMM
python setup.py bdist_wheel
python -m pip install --force-reinstall dist/deep_gemm-*.whl
```

The validated shape uses FP8 MLA KV, so the optional NVFP4 MLA write extension
is not required. The MoET W2 cubins are prebuilt in
`kernels/cubins-sm120`; do not regenerate them for this Python-only change.

## Tests

```bash
cd /opt/vllm-v0.24.0-clean
python -m pip install pytest tblib ruff
python -m pytest -q \
  tests/model_executor/layers/test_moe_w2_mapped_host.py \
  tests/model_executor/layers/test_moe_w2_persistence.py
python -m ruff check \
  vllm/model_executor/layers/quantization/utils/moe_w2_mapped_host.py \
  tests/model_executor/layers/test_moe_w2_mapped_host.py
```

## Validated host launch

The following is the exact serving shape. The pack directory only avoids
rebuilding previously generated FP4 data; it is not the mapped W2 backing.
Remove `VLLM_MOE_W2_STORE_DIR` only when deliberately accepting a cold
rebuild. Run from any directory after activating the clean environment.

```bash
source /opt/vllm-moet-clean/.venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export VLLM_MOE_W2_CUBIT_DIR=/opt/vllm-moet-clean/kernels/cubins-sm120
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_MOE_W2=1
export VLLM_MOE_W2_FORCE_RESIDENT=1
export VLLM_MOE_W2_DELTA_GB=6
export VLLM_MOE_W2_DELTA_RESERVE_GB=0
export VLLM_MOE_W2_MTP_LAYERS=3
export VLLM_MOE_W2_DELTA_POLICY=freq
export VLLM_MOE_W2_PREFILL_FP4=1
export VLLM_MOE_W2_GATE=0
export VLLM_MOE_W2_AFRAG=1
export VLLM_MOE_W2_FUSED_UNPERMUTE=0
export VLLM_MOE_W2_STORE_DIR=/srv/models/moet-packs/DeepSeek-V4-Flash
export VLLM_MOE_W2_MAPPED_LAYERS=40,41,42
export VLLM_MOE_W2_MAPPED_NUMA_NODE=0
export VLLM_MOE_W2_MAPPED_PCI=0000:31:00.0
export VLLM_MOE_W2_MAPPED_AUDIT_PATH=/tmp/pennyroyal-mapped-w2-audit.json

/opt/vllm-moet-clean/.venv/bin/python -m vllm.entrypoints.cli.main serve \
  /srv/models/hf/ds4flash0731 \
  --served-model-name pennyroyal --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 1 --max-model-len 393216 --max-num-seqs 4 \
  --kv-cache-dtype fp8 --block-size 256 --gpu-memory-utilization 0.988 \
  --max-num-batched-tokens 2048 --enable-prefix-caching \
  --trust-remote-code --load-format safetensors \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs \
    '{"enable_thinking":true,"reasoning_effort":"high"}' \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --speculative-config \
    '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"greedy"}' \
  --compilation-config \
    '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
```

Before sending a request, require all full, piecewise, and DSpark graph
captures to complete. Verify the audit lists exactly layers 40–42, total
allocation 5,435,817,984 bytes, NUMA node 0, and zero redundant GPU W2 bytes.
Record runtime-reported KV capacity and physical free VRAM separately from any
request-length result.
