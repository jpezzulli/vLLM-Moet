#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

NATIVE_ROOT="${NATIVE_ROOT:-${REPO_ROOT}/.native}"
VLLM_PYTHON="${VLLM_PYTHON:-${NATIVE_ROOT}/runtime-venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/srv/models/hf/ds4flash0731}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-pennyroyal}"
PORT="${PORT:-8001}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MOET_STORE_DIR="${MOET_STORE_DIR:-/srv/models/moet-packs/DeepSeek-V4-Flash}"
MOET_PLANES_CACHE_DIR="${MOET_PLANES_CACHE_DIR:-${MOET_STORE_DIR}}"
CACHE_ROOT="${CACHE_ROOT:-/srv/cache/vllm-moet}"
W2_AUDIT_PATH="${W2_AUDIT_PATH:-/tmp/pennyroyal-mapped-w2-audit.json}"

fail() {
  echo "error: $*" >&2
  exit 1
}

[[ -x "${VLLM_PYTHON}" ]] || fail "vLLM Python is not executable: ${VLLM_PYTHON}"
[[ -d "${MODEL_PATH}" ]] || fail "model directory does not exist: ${MODEL_PATH}"
[[ -d "${REPO_ROOT}/kernels/cubins-sm120" ]] || fail "SM120 cubins are missing"
[[ -n "${SERVED_MODEL_NAME}" ]] || fail "SERVED_MODEL_NAME cannot be empty"
[[ "${PORT}" =~ ^[0-9]+$ ]] || fail "PORT must be numeric"

locked_kib="$(ulimit -l)"
if [[ "${locked_kib}" != "unlimited" ]] && \
   (( locked_kib < 9000000 )); then
  fail "memlock limit ${locked_kib} KiB is too small for five mapped W2 layers"
fi

mkdir -p "${MOET_STORE_DIR}" "${MOET_PLANES_CACHE_DIR}" \
  "${CACHE_ROOT}/root-cache/flashinfer" \
  "${CACHE_ROOT}/root-cache/torch_extensions" "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/nvidia/ComputeCache"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CC="${CC:-/usr/bin/gcc-15}"
export CXX="${CXX:-/usr/bin/g++-15}"
export NVCC_CCBIN="${NVCC_CCBIN:-${CXX}}"
unset NVCC_PREPEND_FLAGS

export XDG_CACHE_HOME="${CACHE_ROOT}/root-cache"
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}/root-cache/flashinfer"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/root-cache/torch_extensions"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/nvidia/ComputeCache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_DEEP_GEMM=1
export VLLM_MOE_W2_CUBIT_DIR="${REPO_ROOT}/kernels/cubins-sm120"
export VLLM_MOE_W2=1
unset VLLM_MOE_W2_BASE_CACHE_GB
unset VLLM_MOE_W2_FORCE_POOL
unset VLLM_MOE_W2_MTP_LAYERS
export VLLM_MOE_W2_FORCE_RESIDENT=1
export VLLM_MOE_W2_DELTA_GB=6
export VLLM_MOE_W2_DELTA_RESERVE_GB=0
export VLLM_MOE_W2_DRAFT=1
export VLLM_MOE_W2_NUM_LAYERS=46
export VLLM_MOE_W2_DELTA_POLICY=freq
export VLLM_MOE_W2_PREFILL_FP4=1
export VLLM_MOE_W2_GATE=0
export VLLM_MOE_W2_AFRAG=1
export VLLM_MOE_W2_FUSED_UNPERMUTE=0
export VLLM_MOE_W2_STORE_DIR="${MOET_STORE_DIR}"
export VLLM_MOE_W2_PLANES_CACHE="${MOET_PLANES_CACHE_DIR}"
export VLLM_MOE_W2_FAST_LOAD=1
export VLLM_MOE_W2_FAST_LOAD_WORKERS=4
export VLLM_MOE_W2_FAST_LOAD_BATCH_LAYERS=12
export VLLM_MOE_W2_MAPPED_LAYERS=38,39,40,41,42
export VLLM_MOE_W2_MAPPED_AUDIT_PATH="${W2_AUDIT_PATH}"

exec "${VLLM_PYTHON}" -m vllm.entrypoints.cli.main serve \
  "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 1000000 \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization 0.974 \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --trust-remote-code \
  --load-format safetensors \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"high"}' \
  --override-generation-config '{"top_p":0.95}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --speculative-config \
    '{"method":"dspark","num_speculative_tokens":4,"draft_sample_method":"greedy"}' \
  --compilation-config \
    '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
