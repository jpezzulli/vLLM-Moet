#!/usr/bin/env bash
set -euo pipefail

readonly VLLM_COMMIT="a2131dd7a944353e9323566107c72f4a17441024"
readonly DEEPGEMM_COMMIT="a6b593d2826719dcf4892609af7b84ee23aaf32a"
readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

NATIVE_ROOT="${NATIVE_ROOT:-${REPO_ROOT}/.native}"
VLLM_SRC="${VLLM_SRC:-${NATIVE_ROOT}/vllm}"
RUNTIME_VENV="${RUNTIME_VENV:-${NATIVE_ROOT}/runtime-venv}"
BUILD_VENV="${BUILD_VENV:-${NATIVE_ROOT}/build-venv-py313}"
DEEPGEMM_SRC="${DEEPGEMM_SRC:-${NATIVE_ROOT}/DeepGEMM}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CC="${CC:-/usr/bin/gcc-15}"
CXX="${CXX:-/usr/bin/g++-15}"
MAX_JOBS="${MAX_JOBS:-16}"
NVCC_THREADS="${NVCC_THREADS:-2}"

fail() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

for command_name in git uv cmake ninja; do
  require_command "${command_name}"
done
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || fail "nvcc not found under CUDA_HOME=${CUDA_HOME}"
[[ -x "${CC}" ]] || fail "C compiler is not executable: ${CC}"
[[ -x "${CXX}" ]] || fail "C++ compiler is not executable: ${CXX}"
[[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_JOBS must be a positive integer"
[[ "${NVCC_THREADS}" =~ ^[1-9][0-9]*$ ]] || fail "NVCC_THREADS must be a positive integer"

mkdir -p "${NATIVE_ROOT}"

if [[ ! -d "${VLLM_SRC}/.git" ]]; then
  git clone https://github.com/jpezzulli/vllm.git "${VLLM_SRC}"
fi
[[ -z "$(git -C "${VLLM_SRC}" status --porcelain)" ]] || \
  fail "vLLM checkout is dirty: ${VLLM_SRC}"
git -C "${VLLM_SRC}" fetch origin "${VLLM_COMMIT}"
git -C "${VLLM_SRC}" checkout --detach "${VLLM_COMMIT}"
[[ "$(git -C "${VLLM_SRC}" rev-parse HEAD)" == "${VLLM_COMMIT}" ]] || \
  fail "vLLM source did not resolve to ${VLLM_COMMIT}"

uv python install 3.13
uv python install 3.14
[[ -x "${RUNTIME_VENV}/bin/python" ]] || \
  uv venv --python 3.14 --seed "${RUNTIME_VENV}"
[[ -x "${BUILD_VENV}/bin/python" ]] || \
  uv venv --python 3.13 --seed "${BUILD_VENV}"

export CUDA_HOME CUDA_PATH="${CUDA_HOME}" CC CXX NVCC_CCBIN="${CXX}"

uv pip install --python "${RUNTIME_VENV}/bin/python" --torch-backend=cu130 \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
"${RUNTIME_VENV}/bin/python" -m pip install \
  'cmake>=3.26.1' ninja 'packaging>=24.2' \
  'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8' \
  'setuptools-rust>=1.9.0' wheel 'jinja2>=3.1.6' regex build
"${RUNTIME_VENV}/bin/python" -m pip install \
  -r "${VLLM_SRC}/requirements/common.txt"
"${RUNTIME_VENV}/bin/python" -m pip install \
  'numba==0.65.0' 'apache-tvm-ffi==0.1.9' 'tilelang==0.1.9' \
  'nvidia-cudnn-frontend>=1.19.1' 'fastsafetensors>=0.3.2' \
  'nvidia-cutlass-dsl[cu13]==4.5.2' 'quack-kernels>=0.3.3' \
  'tokenspeed-mla==0.1.2' 'humming-kernels[cu13]==0.1.6'
"${RUNTIME_VENV}/bin/python" -m pip uninstall -y flashinfer-cubin || true
"${RUNTIME_VENV}/bin/python" -m pip install --no-deps --upgrade \
  flashinfer-python==0.6.14
"${RUNTIME_VENV}/bin/python" -m pip install --no-deps \
  --index-url https://flashinfer.ai/whl/cu130 \
  'flashinfer-jit-cache==0.6.14+cu130'

if [[ ! -d "${DEEPGEMM_SRC}/.git" ]]; then
  git clone https://github.com/deepseek-ai/DeepGEMM.git "${DEEPGEMM_SRC}"
fi
[[ -z "$(git -C "${DEEPGEMM_SRC}" status --porcelain)" ]] || \
  fail "DeepGEMM checkout is dirty: ${DEEPGEMM_SRC}"
git -C "${DEEPGEMM_SRC}" fetch origin "${DEEPGEMM_COMMIT}"
git -C "${DEEPGEMM_SRC}" checkout --detach "${DEEPGEMM_COMMIT}"
git -C "${DEEPGEMM_SRC}" submodule update --init --recursive
(
  cd "${DEEPGEMM_SRC}"
  "${RUNTIME_VENV}/bin/python" setup.py bdist_wheel
  "${RUNTIME_VENV}/bin/python" -m pip install --no-deps --force-reinstall \
    dist/deep_gemm-*.whl
)

uv pip install --python "${BUILD_VENV}/bin/python" --torch-backend=cu130 \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
"${BUILD_VENV}/bin/python" -m pip install \
  'cmake>=3.26.1' ninja 'packaging>=24.2' \
  'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8' \
  'setuptools-rust>=1.9.0' wheel 'jinja2>=3.1.6' regex build

(
  cd "${VLLM_SRC}"
  VLLM_USE_PRECOMPILED=0 \
  MAX_JOBS="${MAX_JOBS}" NVCC_THREADS="${NVCC_THREADS}" \
  CMAKE_BUILD_TYPE=Release TORCH_CUDA_ARCH_LIST=12.0a \
    "${BUILD_VENV}/bin/python" -m pip install \
      --no-deps --no-build-isolation -e .

  VLLM_TARGET_DEVICE=empty \
    "${RUNTIME_VENV}/bin/python" -m pip install \
      --no-deps --no-build-isolation -e .
)

"${RUNTIME_VENV}/bin/python" -c \
  'import torch, vllm; print("vllm", vllm.__version__); print("torch", torch.__version__)'
echo "native build complete: ${RUNTIME_VENV}"
