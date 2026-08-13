#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-pennyroyal}"
W2_AUDIT_PATH="${W2_AUDIT_PATH:-/tmp/pennyroyal-mapped-w2-audit.json}"
VLLM_PYTHON="${VLLM_PYTHON:-${REPO_ROOT}/.native/runtime-venv/bin/python}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-30}"

fail() {
  echo "error: $*" >&2
  exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
[[ "${STARTUP_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]] || fail "STARTUP_TIMEOUT_S must be positive"

if [[ -x "${VLLM_PYTHON}" ]]; then
  "${VLLM_PYTHON}" -c 'import torch, vllm; print("import:", vllm.__version__, torch.__version__)'
else
  echo "note: skipping local import check; VLLM_PYTHON is not executable: ${VLLM_PYTHON}"
fi

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl --fail --silent --show-error "${BASE_URL}/health" >/dev/null; do
  (( SECONDS < deadline )) || fail "health endpoint did not become ready: ${BASE_URL}/health"
  sleep 1
done
echo "health: ok"

tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "${tmp_dir}"' EXIT

curl --fail --silent --show-error "${BASE_URL}/v1/models" > "${tmp_dir}/models.json"
python3 - "${tmp_dir}/models.json" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected = sys.argv[2]
models = payload.get("data") or []
match = next((model for model in models if model.get("id") == expected), None)
if match is None:
    raise SystemExit(f"served model {expected!r} not found")
limit = match.get("max_model_len")
if limit != 524_288:
    raise SystemExit(f"expected max_model_len 524288, got {limit!r}")
print(f"models: {expected}, max_model_len={limit}")
PY

[[ -f "${W2_AUDIT_PATH}" ]] || fail "mapped-W2 audit not found: ${W2_AUDIT_PATH}"
python3 - "${W2_AUDIT_PATH}" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
expected_layers = [42, 43, 44, 45]
if audit.get("configured_layers") != expected_layers:
    raise SystemExit(f"unexpected mapped layers: {audit.get('configured_layers')!r}")
if audit.get("total_allocation_bytes") != 7_247_757_312:
    raise SystemExit(f"unexpected mapped bytes: {audit.get('total_allocation_bytes')!r}")
if audit.get("redundant_gpu_w2_bytes") != 0:
    raise SystemExit("audit reports redundant complete GPU W2 bytes")
for layer in audit.get("layers") or []:
    if not layer.get("construction_complete") or not layer.get("uva_pointer_equal"):
        raise SystemExit(f"incomplete/UVA-invalid mapped layer: {layer.get('layer_key')}")
    if layer.get("numa_node") is None or not layer.get("page_nodes"):
        raise SystemExit(f"missing NUMA placement evidence: {layer.get('layer_key')}")
print("mapped audit: layers 42-45, 7247757312 bytes, zero redundant GPU W2")
PY

python3 - "${SERVED_MODEL_NAME}" > "${tmp_dir}/request.json" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Return only the integer result of 37 - 9."}],
    "temperature": 0,
    "max_tokens": 128,
    "stream": False,
}))
PY
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  --data-binary "@${tmp_dir}/request.json" \
  "${BASE_URL}/v1/chat/completions" > "${tmp_dir}/response.json"
python3 - "${tmp_dir}/response.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
choice = (payload.get("choices") or [{}])[0]
message = choice.get("message") or {}
text = "\n".join(str(message.get(key) or "") for key in ("reasoning", "reasoning_content", "content"))
if "28" not in text or choice.get("finish_reason") != "stop":
    raise SystemExit(f"arithmetic smoke failed: finish={choice.get('finish_reason')!r}, text={text!r}")
print("inference: 37 - 9 = 28, finish_reason=stop")
PY

echo "validation: passed"
