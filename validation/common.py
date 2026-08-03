"""Shared helpers for the public Pennyroyal validation runners."""

from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_MODEL = "pennyroyal"


def base_url() -> str:
    return os.environ.get("BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def served_model_name() -> str:
    return os.environ.get("SERVED_MODEL_NAME", DEFAULT_MODEL)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain a JSON array or JSON Lines")
        return value
    return [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]


def response_fragments(payload: dict) -> tuple[str, str]:
    """Return reasoning and visible text from chat messages or stream events.

    DeepSeek-compatible servers have emitted hidden reasoning under both
    ``reasoning`` and ``reasoning_content``.  Keep both distinct from ordinary
    content while accepting either spelling.
    """

    choices = payload.get("choices") or []
    if not choices:
        return "", ""
    choice = choices[0]
    message = choice.get("delta") or choice.get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content") or ""
    return str(reasoning), str(content)


def combined_response_text(payload: dict) -> str:
    reasoning, content = response_fragments(payload)
    return "\n".join(part for part in (reasoning, content) if part)
