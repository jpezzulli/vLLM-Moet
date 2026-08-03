import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


common = load("validation_common", ROOT / "common.py")
reasoning_cases = load("reasoning_cases", ROOT / "cases/reasoning.py")
reasoning_runner = load("reasoning_runner", ROOT / "run-reasoning.py")
reasoning_scorer = load("reasoning_scorer", ROOT / "score-reasoning.py")
tool_runner = load("tool_runner", ROOT / "run-tools.py")
needle_runner = load("needle_runner", ROOT / "run-needle.py")


class ResponseExtractionTests(unittest.TestCase):
    def test_all_deepseek_response_fields(self):
        for field in ("reasoning", "reasoning_content"):
            reasoning, content = common.response_fragments(
                {"choices": [{"delta": {field: "hidden", "content": "visible"}}]}
            )
            self.assertEqual(reasoning, "hidden")
            self.assertEqual(content, "visible")

    def test_needle_fixture_smoke(self):
        self.assertTrue(needle_runner.fixture_smoke()["passed"])


class ReasoningSuiteTests(unittest.TestCase):
    def test_frozen_request_count(self):
        self.assertEqual(len(reasoning_cases.CASES), 8)
        self.assertEqual(len(reasoning_runner.measured_plan(reasoning_cases)), 9)

    def test_historical_grade_reproduces_published_score(self):
        grade = json.loads(
            (ROOT / "fixtures/reasoning-dspark4-grade.json").read_text()
        )
        rubric = json.loads(
            (ROOT / "cases/reasoning-rubric.json").read_text()
        )
        result = reasoning_scorer.score_grade(
            grade, reasoning_cases.CASES, rubric
        )
        self.assertEqual(result["final_score"], 97.07)
        self.assertLess(result["case_scores"]["C5"], 100)


class ToolSuiteTests(unittest.TestCase):
    def test_frozen_invocation_count_and_public_definition(self):
        self.assertEqual(len(tool_runner.invocation_plan()), 30)
        definition = json.loads((ROOT / "cases/tools.json").read_text())
        self.assertEqual(len(definition["ordinary_cases"]), 14)
        self.assertEqual(len(definition["concurrent_cases"]), 3)
        self.assertEqual(definition["system_message"], tool_runner.SYSTEM_MESSAGE)
        self.assertEqual(definition["tool_schemas"], tool_runner.TOOLS)
        self.assertEqual(definition["ordinary_cases"], tool_runner.CASES)
        self.assertEqual(definition["concurrent_cases"], tool_runner.CONCURRENT)

    def test_exact_argument_comparison(self):
        good = {
            "case_id": "02_obvious_weather",
            "calls": [{
                "name": "get_weather",
                "arguments": {"location": "Boston", "unit": "celsius"},
            }],
        }
        self.assertTrue(tool_runner.strict_calls_match(good))
        good["calls"][0]["arguments"]["unit"] = "fahrenheit"
        self.assertFalse(tool_runner.strict_calls_match(good))

    def test_safe_replay_scores_30_of_30(self):
        rows = json.loads((ROOT / "fixtures/tools-dspark4-replay.json").read_text())
        result = tool_runner.score_rows(rows)
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["exact_tool_selection_and_arguments"], 30)
        self.assertFalse(result["external_side_effects_executed"])


class NeedleConstructionTests(unittest.TestCase):
    def test_locate_including_boundaries(self):
        self.assertEqual(needle_runner.locate([1, 2, 3, 4], [1, 2]), 0)
        self.assertEqual(needle_runner.locate([1, 2, 3, 4], [3, 4]), 2)
        self.assertIsNone(needle_runner.locate([1, 2, 3], [2, 4]))


class PublicationTests(unittest.TestCase):
    def test_public_markdown_local_links_resolve(self):
        repository = ROOT.parent
        checked = 0
        for markdown in repository.rglob("*.md"):
            text = markdown.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                local = target.split("#", 1)[0]
                if not local:
                    continue
                checked += 1
                self.assertTrue(
                    (markdown.parent / local).exists(),
                    f"broken link in {markdown}: {target}",
                )
        self.assertGreater(checked, 0)

    def test_validation_tree_has_no_host_evidence_paths_or_credentials(self):
        forbidden = (
            "/home/" + "mrkaos",
            "/opt/" + "ai-artifacts",
            "/srv/" + "models",
            "gh" + "o_",
            "192." + "168.",
        )
        for path in ROOT.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, text, f"private value in {path}")


if __name__ == "__main__":
    unittest.main()
