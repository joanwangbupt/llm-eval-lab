from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from llm_eval_lab.deterministic import load_dataset
from llm_eval_lab.semantic_eval import build_test_case, main, measure_semantic


ROOT = Path(__file__).resolve().parents[1]


class SemanticEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = load_dataset(ROOT / "datasets/refund_case_v1.jsonl")[0]
        # No dependence on ignored local experiment files.
        self.raw = json.dumps({"status": "answered", "answer":
                               "Yes: cancellation was 8 days after renewal with no service usage."})

    def test_builds_test_case_with_extracted_answer_and_context(self):
        test_case = build_test_case(self.case, self.raw)
        self.assertEqual(test_case.actual_output, json.loads(self.raw)["answer"])
        self.assertEqual(test_case.expected_output, self.case["expected_answer"])
        self.assertEqual(test_case.input,
                         f"Context:\n{self.case['context']}\n\nQuestion:\n{self.case['question']}")

    def test_invalid_contract_never_repaired(self):
        for raw in ("not JSON", '```json\n' + self.raw + '\n```',
                    '{"status":"answered","answer":1}',
                    '{"status":"answered","answer":"yes","extra":true}'):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "unavailable"):
                build_test_case(self.case, raw)

    def test_abstention_has_no_answer_to_judge(self):
        with self.assertRaisesRegex(ValueError, "answer strings"):
            build_test_case(self.case, '{"status":"insufficient_information","answer":null}')

    def test_uses_native_judge_and_returns_metric_score_unchanged(self):
        test_case = build_test_case(self.case, self.raw)
        with patch("llm_eval_lab.semantic_eval.OllamaModel") as model, \
             patch("llm_eval_lab.semantic_eval.GEval") as metric:
            metric.return_value.score = 0.4
            metric.return_value.reason = "Mock Judge disagrees"
            result = measure_semantic(test_case, judge_model="local-demo", api_base="http://localhost:11434")
            self.assertEqual(result, (0.4, "Mock Judge disagrees"))
            metric.return_value.measure.assert_called_once_with(test_case)
            self.assertIs(metric.call_args.kwargs["model"], model.return_value)
            self.assertEqual(metric.call_args.kwargs["name"], "Semantic Correctness")

    def run_cli(self, raw, failure=None):
        with TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            predictions.write_text(json.dumps({"id": self.case["id"], "raw_output": raw}) + "\n")
            with patch("llm_eval_lab.semantic_eval.measure_semantic", return_value=(0.4, "Mock reason"),
                       side_effect=failure) as judge, redirect_stdout(StringIO()) as stdout:
                code = main(["--dataset", str(ROOT / "datasets/refund_case_v1.jsonl"),
                             "--predictions", str(predictions), "--judge-model", "mock-model",
                             "--api-base", "http://localhost:11434"])
            return code, stdout.getvalue(), judge.call_count

    def test_cli_keeps_results_independent_even_with_low_semantic_score(self):
        code, stdout, calls = self.run_cli(self.raw)
        self.assertEqual(code, 0)
        self.assertEqual(calls, 1)
        for text in ("exact_match_mismatch", "DeepEval Score (0–1): 0.4", "Mock reason"):
            self.assertIn(text, stdout)

    def test_cli_format_failure_skips_judge(self):
        code, stdout, calls = self.run_cli("bad JSON")
        self.assertEqual(code, 2)
        self.assertEqual(calls, 0)
        self.assertIn("format_failure", stdout)
        self.assertNotIn("DeepEval Score", stdout)

    def test_cli_judge_error_is_not_quality_failure(self):
        code, stdout, calls = self.run_cli(self.raw, RuntimeError("do not expose provider detail"))
        self.assertEqual(code, 3)
        self.assertEqual(calls, 1)
        self.assertIn("Judge execution failed (RuntimeError)", stdout)
        self.assertNotIn("DeepEval Score", stdout)
        self.assertNotIn("do not expose provider detail", stdout)


if __name__ == "__main__":
    unittest.main()
