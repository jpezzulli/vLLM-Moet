#!/usr/bin/env python3
"""Validate and aggregate a blinded frozen-suite grading file."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_python(path: Path):
    spec = importlib.util.spec_from_file_location("reasoning_cases", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def score_grade(grade: dict, cases: list[dict], rubric: dict) -> dict:
    case_map = {case["id"]: case for case in cases}
    scores = grade.get("cases") or {}
    if set(scores) != set(case_map):
        raise ValueError("grade must contain exactly C1-C8")

    dimension_weights = rubric["dimension_weights"]
    dimension_averages = {}
    case_scores = {}
    for case_id, case_grade in scores.items():
        applicable = {
            name: value
            for name, value in case_grade.items()
            if name in dimension_weights
        }
        if not applicable:
            raise ValueError(f"{case_id} has no scored dimensions")
        for name, value in applicable.items():
            if not isinstance(value, (int, float)) or not 0 <= value <= 4:
                raise ValueError(f"{case_id}.{name} must be between 0 and 4")
        total_weight = sum(dimension_weights[name] for name in applicable)
        case_scores[case_id] = 100 * sum(
            value * dimension_weights[name] for name, value in applicable.items()
        ) / (4 * total_weight)

    for dimension in dimension_weights:
        participating = [
            (case_map[case_id]["weight"], case_grade[dimension])
            for case_id, case_grade in scores.items()
            if dimension in case_grade
        ]
        if not participating:
            raise ValueError(f"dimension {dimension} has no scores")
        dimension_averages[dimension] = sum(
            weight * value for weight, value in participating
        ) / sum(weight for weight, _ in participating)

    qualitative = sum(
        dimension_averages[name] * weight
        for name, weight in dimension_weights.items()
    ) / (4 * sum(dimension_weights.values())) * 100

    fatal_findings = grade.get("fatal_findings") or []
    capped = qualitative
    for finding in fatal_findings:
        cap_name = finding.get("cap")
        if cap_name not in rubric["fatal_caps"]:
            raise ValueError(f"unknown fatal cap: {cap_name}")
        capped = min(capped, rubric["fatal_caps"][cap_name])

    return {
        "case_scores": {key: round(value, 4) for key, value in case_scores.items()},
        "dimension_averages": {
            key: round(value, 4) for key, value in dimension_averages.items()
        },
        "qualitative_score": round(qualitative, 4),
        "fatal_findings": fatal_findings,
        "final_score": round(capped, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate a completed blinded grade for the frozen reasoning suite."
    )
    parser.add_argument("--grade", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "cases/reasoning.py")
    parser.add_argument(
        "--rubric", type=Path, default=ROOT / "cases/reasoning-rubric.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = load_python(args.cases).CASES
    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    grade = json.loads(args.grade.read_text(encoding="utf-8"))
    result = score_grade(grade, cases, rubric)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
