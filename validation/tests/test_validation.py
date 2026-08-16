import copy
import importlib.util
import json
import re
import subprocess
import sys
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
    def setUp(self):
        self.grade = json.loads(
            (ROOT / "fixtures/reasoning-dspark4-grade.json").read_text()
        )
        self.rubric = json.loads(
            (ROOT / "cases/reasoning-rubric.json").read_text()
        )

    def score(self, grade=None):
        return reasoning_scorer.score_grade(
            grade or self.grade, reasoning_cases.CASES, self.rubric
        )

    def test_frozen_request_count(self):
        self.assertEqual(len(reasoning_cases.CASES), 8)
        self.assertEqual(len(reasoning_runner.measured_plan(reasoning_cases)), 9)

    def test_measured_requests_have_no_output_token_cap(self):
        payload = reasoning_runner.measured_payload("pennyroyal", [])
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("max_completion_tokens", payload)

    def test_historical_grade_reproduces_published_score(self):
        result = self.score()
        self.assertEqual(result["final_score"], 97.07)
        self.assertLess(result["case_scores"]["C5"], 100)

    def test_missing_c5_correctness_is_rejected(self):
        grade = copy.deepcopy(self.grade)
        del grade["cases"]["C5"]["correctness"]
        with self.assertRaisesRegex(ValueError, r"C5.*missing=.*correctness"):
            self.score(grade)

    def test_only_c5_correctness_is_rejected(self):
        grade = copy.deepcopy(self.grade)
        grade["cases"]["C5"] = {"correctness": 4}
        with self.assertRaisesRegex(ValueError, r"C5.*missing="):
            self.score(grade)

    def test_unauthorized_dimension_is_rejected(self):
        grade = copy.deepcopy(self.grade)
        grade["cases"]["C5"]["presentation_polish"] = 4
        with self.assertRaisesRegex(
            ValueError, r"C5.*unexpected=.*presentation_polish"
        ):
            self.score(grade)

    def test_revision_quality_is_rejected_outside_c8(self):
        grade = copy.deepcopy(self.grade)
        grade["cases"]["C1"]["revision_quality"] = 4
        with self.assertRaisesRegex(
            ValueError, r"C1.*unexpected=.*revision_quality"
        ):
            self.score(grade)

    def test_complete_c8_dimension_set_is_accepted(self):
        expected = reasoning_scorer.expected_case_dimensions("C8", self.rubric)
        supplied = set(self.grade["cases"]["C8"]) - {"note"}
        self.assertEqual(supplied, expected)
        self.assertEqual(self.score()["final_score"], 97.07)


class ToolSuiteTests(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads(
            (ROOT / "fixtures/tools-dspark4-replay.json").read_text()
        )

    def assert_plan_rejected(self, rows):
        result = tool_runner.score_rows(rows)
        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["schedule_integrity"]["passed"])
        self.assertTrue(result["schedule_integrity"]["mismatches"])
        return result

    def test_frozen_invocation_count_and_public_definition(self):
        self.assertEqual(len(tool_runner.invocation_plan()), 30)
        definition = json.loads((ROOT / "cases/tools.json").read_text())
        self.assertEqual(len(definition["ordinary_cases"]), 14)
        self.assertEqual(len(definition["concurrent_cases"]), 3)
        self.assertEqual(definition["system_message"], tool_runner.SYSTEM_MESSAGE)
        self.assertEqual(definition["tool_schemas"], tool_runner.TOOLS)
        self.assertEqual(definition["ordinary_cases"], tool_runner.CASES)
        self.assertEqual(definition["concurrent_cases"], tool_runner.CONCURRENT)

    def test_sealed_controls_are_opt_in_and_versioned(self):
        self.assertEqual(len(tool_runner.invocation_plan()), 30)
        self.assertEqual(
            set(tool_runner.SEALED_CONTROLS),
            {"agentic", "natural-decode"},
        )
        self.assertEqual(
            tool_runner.SEALED_CONTROLS["agentic"]["id"],
            "sealed_agentic_release_note_v2",
        )
        self.assertEqual(
            tool_runner.SEALED_CONTROLS["natural-decode"]["id"],
            "sealed_natural_decode_v2",
        )
        decode = tool_runner.SEALED_CONTROLS["natural-decode"]
        self.assertEqual(decode["max_tokens"], 3072)
        self.assertEqual(decode["reasoning_effort"], "low")
        self.assertEqual(decode["temperature"], 0.0)
        self.assertTrue(decode["return_token_ids"])
        self.assertTrue(
            set(tool_runner.SEALED_TOOLS).isdisjoint(tool_runner.TOOLS)
        )

    def test_generated_stream_digests_cover_direct_token_ids_and_output(self):
        turns = [
            {
                "response": {
                    "choices": [
                        {
                            "token_ids": [17, 23, 41],
                            "message": {
                                "reasoning_content": "brief",
                                "content": "answer",
                            },
                        }
                    ]
                }
            }
        ]
        first = tool_runner.generated_stream_digests(turns)
        second = tool_runner.generated_stream_digests(copy.deepcopy(turns))
        self.assertEqual(first, second)
        self.assertEqual(first["token_id_count"], 3)
        changed = copy.deepcopy(turns)
        changed[0]["response"]["choices"][0]["token_ids"][-1] = 42
        self.assertNotEqual(
            first["token_ids_sha256"],
            tool_runner.generated_stream_digests(changed)["token_ids_sha256"],
        )

    def test_journal_analysis_preserves_execution_path_metrics(self):
        lines = [
            (
                "Engine: Avg generation throughput: 81.5 tokens/s, "
                "Prefix cache hit rate: 0.0%"
            ),
            (
                "SpecDecoding metrics: Accepted: 120 tokens, "
                "Drafted: 200 tokens"
            ),
            "torch.compile graph capture occurred",
        ]
        result = tool_runner.journal_analysis(lines)
        self.assertEqual(result["engine_generation_throughput_samples"], [81.5])
        self.assertEqual(result["journal_accepted_tokens_total"], 120)
        self.assertEqual(result["journal_drafted_tokens_total"], 200)
        self.assertEqual(result["journal_acceptance_rate"], 0.6)
        self.assertEqual(result["prefix_cache_hit_rate_samples"], [0.0])
        self.assertEqual(
            result["compile_jit_cuda_graph_activity"],
            ["torch.compile graph capture occurred"],
        )

    def test_sealed_agentic_artifact_requires_real_revision(self):
        state = {}
        created = tool_runner.execute_tool(
            "create_release_note",
            {
                "title": "Atlas release readiness",
                "markdown": (
                    "# Summary\n12 of 20 nodes validated; 8 nodes remain.\n"
                    "# Schedule\n2026-09-02 22:00 UTC\n"
                    "# Owner\nRiley Chen; rollback: Morgan Lee\n"
                    "# Risk\n8 nodes remain unvalidated.\n"
                    "# Next Action\nValidate by 2026-09-01 18:00 UTC"
                ),
            },
            state,
        )
        first = tool_runner.execute_tool(
            "inspect_release_note",
            {"artifact_id": created["artifact_id"]},
            state,
        )
        self.assertEqual(first["status"], "needs_revision")
        revised = tool_runner.execute_tool(
            "revise_release_note",
            {
                "artifact_id": created["artifact_id"],
                "title": state["release_note"]["title"],
                "markdown": state["release_note"]["markdown"],
                "review_acknowledgement": "8 nodes remain unvalidated",
            },
            state,
        )
        final = tool_runner.execute_tool(
            "inspect_release_note",
            {"artifact_id": revised["artifact_id"]},
            state,
        )
        self.assertEqual(revised["version"], 2)
        self.assertEqual(final["status"], "passed")

    def test_agentic_gate_uses_artifact_state_not_redundant_version_prose(self):
        artifact = {
            "artifact_id": "NOTE-ATLAS-17",
            "version": 2,
            "title": "Atlas release readiness",
            "markdown": (
                "# Summary\n12 of 20 nodes validated; 8 nodes remain.\n"
                "# Schedule\n2026-09-02 22:00 UTC\n"
                "# Owner\nRiley Chen; rollback: Morgan Lee\n"
                "# Risk\n8 nodes remain unvalidated.\n"
                "# Next Action\nValidate by 2026-09-01 18:00 UTC"
            ),
            "review_acknowledgement": "8 nodes remain unvalidated",
        }
        names = [
            "inspect_release_brief",
            "create_release_note",
            "inspect_release_note",
            "revise_release_note",
            "inspect_release_note",
        ]
        calls = [{"name": name, "arguments": {}, "result": {}} for name in names]
        calls[2]["result"] = {"status": "needs_revision"}
        calls[4]["result"] = {
            "artifact_id": "NOTE-ATLAS-17",
            "version": 2,
            "status": "passed",
            "issues": [],
        }
        result = {
            "case_id": "sealed_agentic_release_note_v2",
            "calls": calls,
            "final_tool_state": {"release_note": artifact},
            "final": "Artifact NOTE-ATLAS-17 passed its final inspection.",
            "finish_reason": "stop",
            "model_turn_count": 6,
            "tool_call_count": 5,
            "error": None,
        }
        scored = tool_runner.evaluate_control(result)
        self.assertTrue(scored["score"]["passed"])
        self.assertFalse(
            scored["score"]["behavioral_observations"][
                "final_explicitly_identifies_version_2"
            ]
        )
        broken = copy.deepcopy(result)
        broken["final_tool_state"]["release_note"]["version"] = 1
        self.assertFalse(tool_runner.evaluate_control(broken)["score"]["passed"])

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
        result = tool_runner.score_rows(copy.deepcopy(self.rows))
        self.assertTrue(result["gate_passed"])
        self.assertTrue(result["schedule_integrity"]["passed"])
        self.assertEqual(result["exact_tool_selection_and_arguments"], 30)
        self.assertFalse(result["external_side_effects_executed"])

    def test_thirty_copies_of_one_passing_row_are_rejected(self):
        rows = [copy.deepcopy(self.rows[0]) for _ in range(30)]
        result = self.assert_plan_rejected(rows)
        self.assertTrue(result["schedule_integrity"]["missing_identities"])
        self.assertTrue(result["schedule_integrity"]["unexpected_identities"])

    def test_replacing_required_row_with_duplicate_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[5] = copy.deepcopy(rows[0])
        self.assert_plan_rejected(rows)

    def test_omitted_invocation_is_rejected(self):
        self.assert_plan_rejected(copy.deepcopy(self.rows[:-1]))

    def test_extra_invocation_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows.append(copy.deepcopy(rows[-1]))
        self.assert_plan_rejected(rows)

    def test_wrong_phase_or_repeat_is_rejected(self):
        for field, value in (("phase", "smoke"), ("repeat", 99)):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[4][field] = value
                self.assert_plan_rejected(rows)

    def test_boolean_repeats_are_rejected(self):
        for index, value in ((2, False), (1, True)):
            with self.subTest(value=value):
                rows = copy.deepcopy(self.rows)
                rows[index]["repeat"] = value
                result = self.assert_plan_rejected(rows)
                mismatch = result["schedule_integrity"]["mismatches"][0]
                self.assertEqual(mismatch["actual_repeat_type"], "bool")

    def test_integral_float_repeats_are_rejected(self):
        for index, value in ((2, 0.0), (1, 1.0)):
            with self.subTest(value=value):
                rows = copy.deepcopy(self.rows)
                rows[index]["repeat"] = value
                result = self.assert_plan_rejected(rows)
                mismatch = result["schedule_integrity"]["mismatches"][0]
                self.assertEqual(mismatch["actual_repeat_type"], "float")

    def test_none_is_allowed_only_at_canonical_none_positions(self):
        for index, value in ((2, None), (0, 0)):
            with self.subTest(index=index, value=value):
                rows = copy.deepcopy(self.rows)
                rows[index]["repeat"] = value
                self.assert_plan_rejected(rows)

    def test_reordered_invocations_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[2], rows[3] = rows[3], rows[2]
        self.assert_plan_rejected(rows)

    def test_malformed_replay_cli_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            replay = Path(temporary) / "incomplete.json"
            replay.write_text(json.dumps(self.rows[:-1]), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "run-tools.py"), "--replay", str(replay)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        manifest = json.loads(completed.stdout)
        self.assertFalse(manifest["gate_passed"])
        self.assertFalse(manifest["schedule_integrity"]["passed"])

    def test_malformed_repeat_replay_cli_returns_nonzero(self):
        rows = copy.deepcopy(self.rows)
        rows[2]["repeat"] = False
        with tempfile.TemporaryDirectory() as temporary:
            replay = Path(temporary) / "boolean-repeat.json"
            replay.write_text(json.dumps(rows), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run-tools.py"),
                    "--replay",
                    str(replay),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        manifest = json.loads(completed.stdout)
        self.assertFalse(manifest["gate_passed"])
        self.assertFalse(manifest["schedule_integrity"]["passed"])
        mismatch = manifest["schedule_integrity"]["mismatches"][0]
        self.assertEqual(mismatch["actual_repeat_type"], "bool")


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
