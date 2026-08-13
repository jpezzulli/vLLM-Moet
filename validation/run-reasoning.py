#!/usr/bin/env python3
"""Direct OpenAI-compatible streaming collector for the frozen suite."""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import base_url, read_jsonl, served_model_name, write_json  # noqa: E402


def load_suite(path):
    spec = importlib.util.spec_from_file_location("frozen_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def append_jsonl(path, item):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def post_json(base, path, payload, timeout=120):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def exact_block_loop(tokens):
    max_block = min(1024, len(tokens) // 4)
    for size in range(64, max_block + 1):
        block = tokens[-size:]
        if (
            tokens[-2 * size:-size] == block
            and tokens[-3 * size:-2 * size] == block
            and tokens[-4 * size:-3 * size] == block
        ):
            return {"type": "exact_block_x4", "block_tokens": size}
    return None


def sentence_loop(text):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    normalized = []
    for sentence in sentences:
        value = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if value:
            normalized.append(value)
    if len(normalized) < 10:
        return None
    tail = normalized[-10:]
    if len(tail[0].split()) >= 8 and len(set(tail)) == 1:
        return {"type": "normalized_sentence_x10", "sentence": tail[0]}
    return None


def ngrams(tokens, n=5):
    return {tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}


def semantic_cycle_flag(tokens):
    if len(tokens) < 768:
        return None
    windows = [tokens[i:i + 256] for i in range(0, len(tokens) - 255, 256)]
    if len(windows) < 3:
        return None
    sets = [ngrams(window) for window in windows]
    for i in range(len(sets) - 2):
        matches = [i]
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            score = len(sets[i] & sets[j]) / len(union) if union else 0.0
            if score >= 0.85:
                matches.append(j)
        if len(matches) >= 3:
            return {"type": "semantic_cycle_candidate", "windows": matches[:3]}
    return None


def stream_chat(base, payload, raw_dir, request_id, tokenizer, heat_sentinel):
    request_path = raw_dir / f"{request_id}.request.json"
    chunks_path = raw_dir / f"{request_id}.sse.jsonl"
    result_path = raw_dir / f"{request_id}.result.json"
    request_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    started_at = now_iso()
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started_epoch = time.time()
    started_mono = time.monotonic()
    reasoning_parts = []
    content_parts = []
    finish_reason = None
    usage = None
    response_headers = None
    status = None
    error = None
    termination = None
    loop_events = []
    first_reasoning_s = None
    first_content_s = None
    last_loop_check_chars = 0
    last_loop_check_tokens = 0
    token_ids = []

    try:
        with urllib.request.urlopen(request, timeout=7200) as response:
            status = response.status
            response_headers = dict(response.headers.items())
            while True:
                raw = response.readline()
                if not raw:
                    break
                elapsed = time.monotonic() - started_mono
                decoded = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                append_jsonl(chunks_path, {
                    "elapsed_s": elapsed,
                    "raw_line": decoded,
                })
                if heat_sentinel.exists():
                    termination = "heat_guard"
                    break
                if not decoded.startswith("data:"):
                    continue
                data = decoded[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    content = delta.get("content") or ""
                    if reasoning:
                        if first_reasoning_s is None:
                            first_reasoning_s = elapsed
                        reasoning_parts.append(reasoning)
                    if content:
                        if first_content_s is None:
                            first_content_s = elapsed
                        content_parts.append(content)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]

                combined = "".join(reasoning_parts) + "\n" + "".join(content_parts)
                if len(combined) - last_loop_check_chars >= 256:
                    last_loop_check_chars = len(combined)
                    token_ids = tokenize(tokenizer, combined)
                    if len(token_ids) - last_loop_check_tokens >= 32:
                        last_loop_check_tokens = len(token_ids)
                        event = exact_block_loop(token_ids) or sentence_loop(combined)
                        if event:
                            event["generated_tokens_at_detection"] = len(token_ids)
                            event["elapsed_s"] = elapsed
                            loop_events.append(event)
                            termination = "hard_loop"
                            break
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ended_epoch = time.time()
    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)
    if not token_ids:
        token_ids = tokenize(tokenizer, reasoning + "\n" + content)
    semantic = semantic_cycle_flag(token_ids)
    if semantic:
        loop_events.append(semantic)
    result = {
        "request_id": request_id,
        "started_at": started_at,
        "started_epoch": started_epoch,
        "ended_epoch": ended_epoch,
        "wall_seconds": ended_epoch - started_epoch,
        "http_status": status,
        "response_headers": response_headers,
        "reasoning_content": reasoning,
        "content": content,
        "usage": usage,
        "local_generated_tokens": len(token_ids),
        "finish_reason": finish_reason,
        "time_to_first_reasoning_s": first_reasoning_s,
        "time_to_first_content_s": first_content_s,
        "error": error,
        "termination": termination,
        "loop_events": loop_events,
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def wait_ready(base, timeout=1200):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=10) as response:
                if response.status == 200:
                    return json.loads(response.read())
        except Exception as exc:
            last = exc
        time.sleep(5)
    raise TimeoutError(f"server did not become ready: {last}")


def messages(system, prompt):
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def measured_payload(model, request_messages):
    """Build a measured request without a harness-imposed output-token cap."""
    return {
        "model": model,
        "messages": request_messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def measured_plan(suite):
    plan = []
    for case in suite.CASES:
        plan.append({"request_id": f"{case['id'].lower()}-r1", "case_id": case["id"], "turn": 1})
        if case["id"] == "C8":
            plan.append({"request_id": "c8-r1-correction", "case_id": "C8", "turn": 2})
    return plan


def replay_manifest(path, suite):
    rows = read_jsonl(path)
    measured = [row for row in rows if row.get("kind") == "measured"]
    expected = [(item["case_id"], item["turn"]) for item in measured_plan(suite)]
    actual = [(item.get("case_id"), item.get("turn")) for item in measured]
    errors = [item.get("error") for item in measured if item.get("error")]
    manifest = {
        "mode": "replay",
        "source": str(path),
        "expected_measured_requests": 9,
        "measured_requests": len(measured),
        "request_order_matches": actual == expected,
        "errors": errors,
        "gate_passed": len(measured) == 9 and actual == expected and not errors,
        "grading": (
            "Responses require blinded qualitative grading with cases/reasoning-rubric.json; "
            "use score-reasoning.py after scores are locked."
        ),
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Run or inspect the frozen nine-request reasoning suite."
    )
    parser.add_argument("--suite", type=Path, default=ROOT / "cases/reasoning.py")
    parser.add_argument("--output", "--output-dir", dest="output", type=Path)
    parser.add_argument("--runtime", default="candidate")
    parser.add_argument("--base", "--base-url", dest="base", default=base_url())
    parser.add_argument("--model", "--served-model-name", dest="model", default=served_model_name())
    parser.add_argument("--heat-sentinel", type=Path)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--list", action="store_true", help="list deterministic request IDs")
    parser.add_argument("--dry-run", action="store_true", help="print the request plan without contacting a server")
    parser.add_argument("--replay", type=Path, help="validate an existing results.jsonl without contacting a server")
    args = parser.parse_args()

    suite = load_suite(args.suite)
    plan = measured_plan(suite)
    if args.list:
        for item in plan:
            print(f"{item['request_id']}\t{item['case_id']}\tturn {item['turn']}")
        return 0
    if args.dry_run:
        print(json.dumps({
            "mode": "dry-run",
            "base_url": args.base,
            "served_model_name": args.model,
            "warmups": 2,
            "measured_requests": plan,
            "request_parameters": {
                "max_tokens": None,
                "max_tokens_field": "omitted",
                "stream": True,
                "stream_options": {"include_usage": True},
                "sampling_overrides": [],
            },
        }, indent=2))
        return 0
    if args.replay:
        manifest = replay_manifest(args.replay, suite)
        print(json.dumps(manifest, indent=2))
        return 0 if manifest["gate_passed"] else 1
    if not args.tokenizer_path:
        parser.error("--tokenizer-path is required for a live run")

    out = args.output or Path(
        "validation-results", time.strftime("reasoning-%Y%m%d-%H%M%S")
    )
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    heat_sentinel = args.heat_sentinel or out / "HEAT_STOP"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    ready = wait_ready(args.base)
    (out / "models-ready.json").write_text(json.dumps(ready, indent=2) + "\n")
    events_path = out / "events.jsonl"
    results_path = out / "results.jsonl"
    append_jsonl(events_path, {"event": "ready", "epoch": time.time()})

    warmups = [
        (
            "warmup-prefill",
            "Read the following repeated calibration labels and return only the final label: "
            + "alpha beta gamma delta " * 450
            + "FINAL-LABEL-731",
            32,
        ),
        (
            "warmup-reasoning",
            "A sealed box contains 4 red and 6 blue tokens. Without replacement, what is "
            "the probability that two draws have different colors? Give the reduced fraction.",
            512,
        ),
    ]
    for request_id, prompt, cap in warmups:
        payload = {
            "model": args.model,
            "messages": messages(suite.SYSTEM, prompt),
            "max_tokens": cap,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        print(f"START {request_id}", flush=True)
        result = stream_chat(
            args.base, payload, raw_dir, request_id, tokenizer, heat_sentinel
        )
        result["kind"] = "warmup"
        append_jsonl(results_path, result)
        print(
            f"END {request_id} wall={result['wall_seconds']:.2f}s "
            f"tokens={result['local_generated_tokens']} finish={result['finish_reason']} "
            f"termination={result['termination']} error={result['error']}",
            flush=True,
        )
        if result["termination"] == "heat_guard":
            return 75
        if result["error"]:
            return 2

    for case in suite.CASES:
        request_id = f"{case['id'].lower()}-r1"
        payload = measured_payload(
            args.model, messages(suite.SYSTEM, case["prompt"])
        )
        print(f"START {request_id} {case['title']}", flush=True)
        result = stream_chat(
            args.base, payload, raw_dir, request_id, tokenizer, heat_sentinel
        )
        result.update({"kind": "measured", "case_id": case["id"], "turn": 1})
        append_jsonl(results_path, result)
        print(
            f"END {request_id} wall={result['wall_seconds']:.2f}s "
            f"tokens={result['local_generated_tokens']} finish={result['finish_reason']} "
            f"termination={result['termination']} error={result['error']}",
            flush=True,
        )
        if result["termination"] == "heat_guard":
            return 75
        if result["error"]:
            return 2
        if case["id"] != "C8":
            continue

        assistant = {"role": "assistant", "content": result["content"]}
        if result["reasoning_content"]:
            assistant["reasoning_content"] = result["reasoning_content"]
        correction_id = "c8-r1-correction"
        correction_messages = messages(suite.SYSTEM, case["prompt"])
        correction_messages.extend([
            assistant,
            {"role": "user", "content": case["correction"]},
        ])
        correction_payload = measured_payload(args.model, correction_messages)
        print(f"START {correction_id}", flush=True)
        corrected = stream_chat(
            args.base, correction_payload, raw_dir, correction_id,
            tokenizer, heat_sentinel
        )
        corrected.update({"kind": "measured", "case_id": "C8", "turn": 2})
        append_jsonl(results_path, corrected)
        print(
            f"END {correction_id} wall={corrected['wall_seconds']:.2f}s "
            f"tokens={corrected['local_generated_tokens']} "
            f"finish={corrected['finish_reason']} "
            f"termination={corrected['termination']} error={corrected['error']}",
            flush=True,
        )
        if corrected["termination"] == "heat_guard":
            return 75
        if corrected["error"]:
            return 2

    append_jsonl(events_path, {"event": "suite_complete", "epoch": time.time()})
    write_json(out / "manifest.json", {
        "runtime": args.runtime,
        "base_url": args.base,
        "served_model_name": args.model,
        "suite_source": str(args.suite),
        "warmups": 2,
        "measured_requests": 9,
        "measured_max_tokens": None,
        "measured_max_tokens_field": "omitted",
        "request_plan": plan,
        "results": "results.jsonl",
        "grading": "blinded qualitative review required; see cases/reasoning-rubric.json",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
