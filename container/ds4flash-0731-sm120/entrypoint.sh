#!/usr/bin/env bash
set -euo pipefail

model_dir=${PENNYROYAL_MODEL_DIR:-/models/ds4flash0731}
pack_dir=${PENNYROYAL_PACK_DIR:-/packs/DeepSeek-V4-Flash}
cache_dir=${PENNYROYAL_CACHE_DIR:-/cache}
audit_path=${PENNYROYAL_AUDIT_PATH:-/run/vllm-moet/mapped-w2-audit.json}

for required_dir in "${model_dir}" "${pack_dir}" "${cache_dir}" "$(dirname "${audit_path}")"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "required mounted directory is missing: ${required_dir}" >&2
    exit 2
  fi
done
if [[ ! -r "${model_dir}/config.json" ]]; then
  echo "model mount does not contain a readable config.json: ${model_dir}" >&2
  exit 2
fi
if [[ ! -w "${pack_dir}" || ! -w "${cache_dir}" || ! -w "$(dirname "${audit_path}")" ]]; then
  echo "pack, cache, and audit mounts must be writable by uid $(id -u)" >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CC=/usr/bin/gcc-15
export CXX=/usr/bin/g++-15
export NVCC_CCBIN=/usr/bin/g++-15
unset NVCC_PREPEND_FLAGS

export XDG_CACHE_HOME="${cache_dir}/root-cache"
export FLASHINFER_WORKSPACE_BASE="${cache_dir}/root-cache/flashinfer"
export TORCH_EXTENSIONS_DIR="${cache_dir}/root-cache/torch_extensions"
export TRITON_CACHE_DIR="${cache_dir}/triton"
export CUDA_CACHE_PATH="${cache_dir}/nvidia/ComputeCache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_DEEP_GEMM=1
export VLLM_MOE_W2_CUBIT_DIR=/opt/vllm-moet/kernels/cubins-sm120
export VLLM_MOE_W2=1
unset VLLM_MOE_W2_BASE_CACHE_GB
unset VLLM_MOE_W2_FORCE_POOL
export VLLM_MOE_W2_FORCE_RESIDENT=1
export VLLM_MOE_W2_DELTA_GB=6
export VLLM_MOE_W2_DELTA_RESERVE_GB=0
export VLLM_MOE_W2_MTP_LAYERS=3
export VLLM_MOE_W2_DELTA_POLICY=freq
export VLLM_MOE_W2_PREFILL_FP4=1
export VLLM_MOE_W2_GATE=0
export VLLM_MOE_W2_AFRAG=1
export VLLM_MOE_W2_FUSED_UNPERMUTE=0
export VLLM_MOE_W2_STORE_DIR="${pack_dir}"
export VLLM_MOE_W2_MAPPED_LAYERS=40,41,42
export VLLM_MOE_W2_MAPPED_AUDIT_PATH="${audit_path}"

mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}"

/opt/venv/bin/python /opt/vllm-moet/topology-preflight.py

if (($#)); then
  echo "this candidate image has a sealed serving command; command overrides are not accepted" >&2
  exit 2
fi

exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve \
  "${model_dir}" \
  --served-model-name pennyroyal \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 393216 \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization 0.988 \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --trust-remote-code \
  --load-format safetensors \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"max"}' \
  --override-generation-config '{"top_p":0.95}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"greedy"}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
