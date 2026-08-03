#!/usr/bin/env bash
set -euo pipefail

image=${1:-localhost/jpezzulli/vllm-moet:ds4flash-0731-sm120-v4-candidate}
container_runtime=${CONTAINER_RUNTIME:-podman}

lookup_library() {
  ldconfig -p | awk -v target="$1" \
    '$1 == target && !found {path = $NF; found = 1} END {if (found) print path}'
}

libcuda=${LIBCUDA_PATH:-$(lookup_library libcuda.so.1)}
libptxjit=${LIBPTXJIT_PATH:-$(lookup_library libnvidia-ptxjitcompiler.so.1)}

if [[ -z "${libcuda}" || ! -r "${libcuda}" ]]; then
  echo "set LIBCUDA_PATH to a readable host libcuda.so.1" >&2
  exit 2
fi
if [[ -z "${libptxjit}" || ! -r "${libptxjit}" ]]; then
  echo "set LIBPTXJIT_PATH to a readable host libnvidia-ptxjitcompiler.so.1" >&2
  exit 2
fi

"${container_runtime}" run --rm -i \
  --entrypoint /bin/bash \
  --security-opt label=disable \
  -e LD_LIBRARY_PATH=/opt/driver \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${libcuda}:/opt/driver/libcuda.so.1:ro" \
  -v "${libptxjit}:/opt/driver/libnvidia-ptxjitcompiler.so.1:ro" \
  "${image}" -s <<'CONTAINER_PROBE'
set -euo pipefail

if find /dev -maxdepth 1 -type c -name 'nvidia*' -print -quit | grep -q .; then
  echo "unexpected NVIDIA device node exposed to offline probe" >&2
  exit 1
fi
echo "no GPU device nodes exposed"

/opt/venv/bin/python - <<'PY'
import importlib
import importlib.util

modules = (
    "vllm.vllm_flash_attn.cute",
    "vllm.vllm_flash_attn.cute.utils",
    "vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl",
    "vllm.vllm_flash_attn._vllm_fa2_C",
    "vllm.vllm_flash_attn._vllm_fa3_C",
)
for name in modules:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise SystemExit(f"missing module: {name}")
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    if origin is not None and not origin.startswith("/opt/"):
        raise SystemExit(f"non-image module origin for {name}: {origin}")
    print(f"{name}: {origin or spec.origin}")
PY

torch_lib=$(/opt/venv/bin/python -c \
  'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="/opt/driver:${torch_lib}:/usr/local/cuda/lib64"

for extension in \
  /opt/vllm/vllm/_C_stable_libtorch.abi3.so \
  /opt/vllm/vllm/_moe_C_stable_libtorch.abi3.so \
  /opt/vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
  /opt/vllm/vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so; do
  if ldd "${extension}" | grep -q 'not found'; then
    ldd "${extension}" >&2
    exit 1
  fi
  echo "linkage OK: ${extension}"
done

tmpdir=$(mktemp -d)
trap 'rm -rf "${tmpdir}"' EXIT
cat >"${tmpdir}/probe.c" <<'C'
#include <Python.h>
static PyObject *answer(PyObject *self, PyObject *args) {
    return PyLong_FromLong(314);
}
static PyMethodDef methods[] = {
    {"answer", answer, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "python_h_probe", NULL, -1, methods,
};
PyMODINIT_FUNC PyInit_python_h_probe(void) { return PyModule_Create(&module); }
C
suffix=$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
gcc -shared -fPIC $(/usr/bin/python3-config --includes) \
  "${tmpdir}/probe.c" -o "${tmpdir}/python_h_probe${suffix}" \
  $(/usr/bin/python3-config --ldflags)
PYTHONPATH="${tmpdir}" /opt/venv/bin/python -c \
  'import python_h_probe; assert python_h_probe.answer() == 314; print("Python.h probe: 314")'
CONTAINER_PROBE
