#!/usr/bin/env python3
"""Construct and stream one tokenizer-exact long-context needle request."""

import argparse
import hashlib
import json
import os
import sys
import time
from array import array
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import base_url, response_fragments, served_model_name, write_json  # noqa: E402


NEEDLE = "GRID-NEEDLE-7B91E2C4A6F0D835"
SYSTEM = (
    "This is a closed-context retrieval task. Use only the supplied archive. "
    "At the end, return only the exact requested needle value, with no explanation."
)
PREFIX = (
    "Read this archive and retain the unique retrieval value. Most of the archive "
    "is harmless filler and contains no competing value.\n\n"
    "The unique retrieval needle is: " + NEEDLE + "\n"
)
FILLER = (
    "\nArchive filler record: meadow cedar quartz cloud. "
    "This harmless record contains no retrieval value."
)
SUFFIX = (
    "\n\nEnd of archive. What is the exact unique retrieval needle stated near "
    "the beginning? Return only that value."
)


def locate(haystack, needle):
    width = len(needle)
    for index in range(len(haystack) - width + 1):
        if haystack[index : index + width] == needle:
            return index
    return None


def locate_needle(tokenizer, prompt_ids):
    for candidate in (NEEDLE, " " + NEEDLE, ": " + NEEDLE, "' " + NEEDLE):
        position = locate(prompt_ids, tokenizer.encode(candidate, add_special_tokens=False))
        if position is not None:
            return position
    return None


def fixture_smoke():
    events = [
        {"choices": [{"delta": {"reasoning": "Found " + NEEDLE}}]},
        {"choices": [{"delta": {"reasoning_content": " and retained it"}}]},
        {"choices": [{"delta": {"content": NEEDLE}}]},
    ]
    fields = [response_fragments(event) for event in events]
    reasoning = "".join(pair[0] for pair in fields)
    content = "".join(pair[1] for pair in fields)
    return {
        "mode": "fixture-smoke",
        "reasoning": reasoning,
        "content": content,
        "reasoning_field_supported": NEEDLE in reasoning,
        "content_field_supported": content == NEEDLE,
        "passed": NEEDLE in reasoning and content == NEEDLE,
    }


def metrics_snapshot(base):
    wanted = {
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_decode_time_seconds_sum",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:e2e_request_latency_seconds_sum",
    }
    values = {name: 0.0 for name in wanted}
    try:
        with httpx.Client(timeout=10) as client:
            lines = client.get(base + "/metrics").text.splitlines()
    except Exception:
        return {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        metric, _, raw = line.rpartition(" ")
        name = metric.split("{", 1)[0]
        if name in wanted:
            values[name] += float(raw)
    return values


def metric_delta(before, after, name):
    if name not in before or name not in after:
        return None
    return after[name] - before[name]


def main():
    parser = argparse.ArgumentParser(
        description="Run the opt-in tokenizer-exact near-million-token needle test."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-input-tokens", type=int, default=994987)
    parser.add_argument("--base", "--base-url", dest="base", default=base_url())
    parser.add_argument("--model", "--served-model-name", dest="model", default=served_model_name())
    parser.add_argument("--tokenizer", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="exercise response extraction with local fixtures only")
    parser.add_argument("--i-understand-this-may-take-20-minutes", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("needle-994987\t994987 server-rendered input tokens\tfollow-up 37 + 58")
        return 0
    if args.smoke:
        result = fixture_smoke()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    if args.dry_run:
        print(json.dumps({
            "mode": "dry-run",
            "base_url": args.base,
            "served_model_name": args.model,
            "target_input_tokens": args.target_input_tokens,
            "needle": NEEDLE,
            "expected_needle_start_zero_based_token": 154,
            "stream": True,
            "follow_up": "37 + 58 must return 95",
            "requires_acknowledgement": True,
        }, indent=2))
        return 0
    if not args.i_understand_this_may_take_20_minutes:
        parser.error(
            "the live near-million-token run requires "
            "--i-understand-this-may-take-20-minutes"
        )
    if not args.tokenizer:
        parser.error("--tokenizer or MODEL_PATH is required for a live run")

    output = args.output_dir or Path(
        "validation-results", time.strftime("needle-%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer_started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def messages(repeats):
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PREFIX + FILLER * repeats + SUFFIX},
        ]

    def body_for(repeats):
        return {
            "model": args.model,
            "messages": messages(repeats),
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    # This checkpoint intentionally has no tokenizer_config chat template.  Ask
    # the running vLLM endpoint to apply its actual DeepSeek-V4 renderer so the
    # measured input count exactly matches what generation will admit.
    base_raw = len(tokenizer.encode(SYSTEM + PREFIX + SUFFIX, add_special_tokens=False))
    one_raw = len(
        tokenizer.encode(SYSTEM + PREFIX + FILLER + SUFFIX, add_special_tokens=False)
    )
    per_repeat = one_raw - base_raw
    repeats = max(0, (args.target_input_tokens - base_raw - 256) // per_repeat)
    render_started = time.time()
    with httpx.Client(timeout=None) as client:
        for _ in range(8):
            rendered = client.post(
                args.base + "/v1/chat/completions/render", json=body_for(repeats)
            )
            rendered.raise_for_status()
            rendered = rendered.json()
            prompt_ids = rendered["token_ids"]
            delta = args.target_input_tokens - len(prompt_ids)
            if delta == 0:
                break
            if delta > 0:
                repeats += max(1, delta // per_repeat)
            else:
                repeats -= max(1, (-delta + per_repeat - 1) // per_repeat)
        else:
            raise RuntimeError("server-rendered token target did not converge exactly")

    body = body_for(repeats)

    if len(prompt_ids) != args.target_input_tokens:
        raise RuntimeError(
            f"server rendered {len(prompt_ids)} tokens, expected exactly "
            f"{args.target_input_tokens}"
        )
    needle_position = locate_needle(tokenizer, prompt_ids)
    token_bytes = array("I", prompt_ids).tobytes()
    actual_input_tokens = len(prompt_ids)
    tokenization = {
        "tokenizer_path": args.tokenizer,
        "tokenizer_class": tokenizer.__class__.__name__,
        "target_input_tokens": args.target_input_tokens,
        "actual_input_tokens": actual_input_tokens,
        "filler_repeats": repeats,
        "filler_tokens_per_repeat": per_repeat,
        "token_count_authority": "POST /v1/chat/completions/render",
        "renderer_wall_seconds": time.time() - render_started,
        "needle": NEEDLE,
        "needle_token_position_zero_based": needle_position,
        "tokens_after_needle_start": (
            actual_input_tokens - needle_position if needle_position is not None else None
        ),
        "prompt_token_ids_sha256_uint32_native": hashlib.sha256(token_bytes).hexdigest(),
        "tokenizer_wall_seconds": time.time() - tokenizer_started,
    }
    (output / "tokenization.json").write_text(json.dumps(tokenization, indent=2) + "\n")
    rendered_summary = {key: value for key, value in rendered.items() if key != "token_ids"}
    rendered_summary["token_ids_count"] = actual_input_tokens
    rendered_summary["token_ids_sha256_uint32_native"] = tokenization[
        "prompt_token_ids_sha256_uint32_native"
    ]
    (output / "render-summary.json").write_text(
        json.dumps(rendered_summary, indent=2) + "\n"
    )
    (output / "request.json").write_text(json.dumps(body, ensure_ascii=False))

    metrics_before = metrics_snapshot(args.base)
    started = time.time()
    first_reasoning = None
    first_content = None
    reasoning_parts = []
    content_parts = []
    usage = None
    finish_reason = None
    error = None
    http_status = None
    with (output / "stream.jsonl").open("w", buffering=1) as stream_log:
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream(
                    "POST", args.base + "/v1/chat/completions", json=body
                ) as response:
                    http_status = response.status_code
                    response.raise_for_status()
                    for line in response.iter_lines():
                        now = time.time()
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            stream_log.write(json.dumps({"epoch": now, "done": True}) + "\n")
                            continue
                        event = json.loads(payload)
                        stream_log.write(json.dumps({"epoch": now, "event": event}) + "\n")
                        if event.get("usage"):
                            usage = event["usage"]
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        reasoning, content = response_fragments(event)
                        if reasoning and first_reasoning is None:
                            first_reasoning = now
                        if content and first_content is None:
                            first_content = now
                        reasoning_parts.append(reasoning)
                        content_parts.append(content)
        except Exception as exc:
            error = repr(exc)

    ended = time.time()
    metrics_after = metrics_snapshot(args.base)
    reasoning_text = "".join(reasoning_parts)
    content_text = "".join(content_parts)
    combined_text = "\n".join(part for part in (reasoning_text, content_text) if part)
    prompt_tokens = metric_delta(
        metrics_before, metrics_after, "vllm:prompt_tokens_total"
    )
    generation_tokens = metric_delta(
        metrics_before, metrics_after, "vllm:generation_tokens_total"
    )
    prefill_seconds = metric_delta(
        metrics_before, metrics_after, "vllm:request_prefill_time_seconds_sum"
    )
    decode_seconds = metric_delta(
        metrics_before, metrics_after, "vllm:request_decode_time_seconds_sum"
    )
    server_metrics = {
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": (
            prompt_tokens / prefill_seconds
            if prompt_tokens is not None and prefill_seconds
            else None
        ),
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (
            generation_tokens / decode_seconds
            if generation_tokens is not None and decode_seconds
            else None
        ),
        "time_to_first_token_seconds": metric_delta(
            metrics_before, metrics_after, "vllm:time_to_first_token_seconds_sum"
        ),
        "end_to_end_seconds": metric_delta(
            metrics_before, metrics_after, "vllm:e2e_request_latency_seconds_sum"
        ),
    }
    result = {
        "started_epoch": started,
        "ended_epoch": ended,
        "wall_seconds": ended - started,
        "http_status": http_status,
        "time_to_first_reasoning_seconds": (
            first_reasoning - started if first_reasoning is not None else None
        ),
        "time_to_first_content_seconds": (
            first_content - started if first_content is not None else None
        ),
        "time_to_first_token_seconds": (
            min(value for value in (first_reasoning, first_content) if value is not None) - started
            if first_reasoning is not None or first_content is not None
            else None
        ),
        "finish_reason": finish_reason,
        "usage": usage,
        "reasoning_content": reasoning_text,
        "content": content_text,
        "needle": NEEDLE,
        "needle_present_in_reasoning": NEEDLE in reasoning_text,
        "needle_present_in_visible_content": NEEDLE in content_text,
        "needle_present_in_any_response_field": NEEDLE in combined_text,
        "visible_content_exact_after_strip": content_text.strip() == NEEDLE,
        "error": error,
        "server_metrics": server_metrics,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    follow_started = time.time()
    follow_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "What is 37 + 58? Return only the number."}],
        "temperature": 0,
        "max_tokens": 64,
    }
    follow_error = None
    follow_response = None
    try:
        with httpx.Client(timeout=None) as client:
            response = client.post(args.base + "/v1/chat/completions", json=follow_payload)
            response.raise_for_status()
            follow_response = response.json()
    except Exception as exc:
        follow_error = repr(exc)
    follow_reasoning, follow_content = response_fragments(follow_response or {})
    follow_result = {
        "request": follow_payload,
        "response": follow_response,
        "wall_seconds": time.time() - follow_started,
        "reasoning": follow_reasoning,
        "content": follow_content,
        "passed": follow_content.strip() == "95",
        "error": follow_error,
    }
    write_json(output / "follow-up.json", follow_result)
    manifest = {
        "target_input_tokens": args.target_input_tokens,
        "actual_input_tokens": actual_input_tokens,
        "needle": NEEDLE,
        "needle_start_zero_based_token": needle_position,
        "needle_retrieved": result["needle_present_in_any_response_field"],
        "follow_up_passed": follow_result["passed"],
        "usage": usage,
        "finish_reason": finish_reason,
        "wall_seconds": result["wall_seconds"],
        "time_to_first_token_seconds": result["time_to_first_token_seconds"],
        "server_metrics": server_metrics,
        "error": error or follow_error,
    }
    manifest["gate_passed"] = (
        manifest["actual_input_tokens"] == manifest["target_input_tokens"]
        and manifest["needle_start_zero_based_token"] == 154
        and manifest["needle_retrieved"]
        and manifest["follow_up_passed"]
        and manifest["error"] is None
    )
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"tokenization": tokenization, "result": result, "follow_up": follow_result}, indent=2))
    return 0 if manifest["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
