#!/usr/bin/env python3
"""Build text-only grading packets, excluding all operational metadata."""

import argparse
import importlib.util
import json
from pathlib import Path


def load_suite(path):
    spec = importlib.util.spec_from_file_location("frozen_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-id", default="candidate-blinded")
    parser.add_argument("--run-prefix", default="screen")
    args = parser.parse_args()
    suite = load_suite(Path(args.suite))
    cases = {case["id"]: case for case in suite.CASES}
    results = [
        json.loads(line) for line in Path(args.results).read_text().splitlines()
        if line.strip()
    ]
    measured = [item for item in results if item.get("kind") == "measured"]
    by_case = {}
    for item in measured:
        by_case.setdefault(item["case_id"], {})[item["turn"]] = item
    packets = []
    for index, case_id in enumerate(suite.CASE_ORDER, start=1):
        case = cases[case_id]
        turns = by_case.get(case_id, {})
        first = turns.get(1, {})
        transcript = [
            {"role": "system", "content": suite.SYSTEM},
            {"role": "user", "content": case["prompt"]},
            {
                "role": "assistant",
                "reasoning_content": first.get("reasoning_content", ""),
                "content": first.get("content", ""),
            },
        ]
        if case_id == "C8":
            second = turns.get(2, {})
            transcript.extend([
                {"role": "user", "content": case["correction"]},
                {
                    "role": "assistant",
                    "reasoning_content": second.get("reasoning_content", ""),
                    "content": second.get("content", ""),
                },
            ])
        packets.append({
            "candidate_id": args.candidate_id,
            "run_id": f"{args.run_prefix}-{index:02d}",
            "case_id": case_id,
            "transcript": transcript,
        })
    Path(args.output).write_text(
        json.dumps(packets, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
