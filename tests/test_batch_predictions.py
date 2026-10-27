from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from llm_eval_lab.batch_predictions import render_prompt, run_batch
from llm_eval_lab.deterministic import evaluate, load_dataset, load_predictions


ROOT = Path(__file__).resolve().parents[1]


class BatchPredictionTests(unittest.TestCase):
    def setUp(self):
        self.cases = load_dataset(ROOT / "datasets/mini_eval.jsonl")
        self.template = (ROOT / "prompts/context_qa.txt").read_text(encoding="utf-8")
        self.config = {"model": "ollama_chat/mock", "temperature": 0}
        self.client = Mock()
        self.client.generate.return_value = self.result(' {"status":"answered","answer":"2018"}\n')

    def result(self, output):
        return SimpleNamespace(output=output, model="ollama_chat/mock", latency_seconds=1.2,
                               prompt_tokens=20, completion_tokens=10, total_tokens=30)

    def run_in(self, directory, cases=None):
        with redirect_stdout(StringIO()) as stdout:
            results, errors = run_batch(
                self.cases if cases is None else cases, self.template, client=self.client,
                model_config=self.config, api_base="http://localhost:11434",
                output_dir=Path(directory) / "run",
            )
        return results, errors, stdout.getvalue()

    def test_prompt_mapping_and_no_ground_truth_leak(self):
        case = {**self.cases[0], "expected_answer": "PRIVATE_GROUND_TRUTH_SENTINEL"}
        with TemporaryDirectory() as directory:
            self.run_in(directory, [case])
        kwargs = self.client.generate.call_args.kwargs
        prompt = kwargs["prompt"]
        self.assertIn("Context:\n" + case["context"], prompt)
        self.assertIn("Question:\n" + case["question"], prompt)
        for private in ("PRIVATE_GROUND_TRUTH_SENTINEL", "expected_answer", "requires_abstention"):
            self.assertNotIn(private, prompt)
        # Changing either evaluation-only field cannot affect the model input.
        changed = {**case, "expected_answer": None, "requires_abstention": True}
        with TemporaryDirectory() as directory:
            self.run_in(directory, [changed])
        self.assertEqual(self.client.generate.call_args.kwargs, kwargs)
        self.assertEqual(set(kwargs), {"model", "prompt", "api_base", "temperature"})
        self.assertEqual(kwargs["model"], self.config["model"])

    def test_render_preserves_json_examples_and_placeholder_like_input(self):
        prompt = render_prompt(self.template, context="Keep {question} literally", question="Who?")
        self.assertIn('For an answer: {"status": "answered", "answer": "2018"}', prompt)
        self.assertIn("Context:\nKeep {question} literally", prompt)
        self.assertIn("Question:\nWho?", prompt)

    def test_missing_placeholder_rejected(self):
        with self.assertRaises(ValueError):
            render_prompt("{context}", context="x", question="y")

    def test_raw_outputs_preserved_and_replayed(self):
        raw = '```json\n{"status":"answered","answer":"2018"}\n```\n'
        self.client.generate.return_value = self.result(raw)
        with TemporaryDirectory() as directory:
            results, errors, _ = self.run_in(directory, self.cases[:1])
            predictions = load_predictions(Path(directory) / "run/predictions.jsonl")
            self.assertEqual(predictions, {"qa-001": raw})
            self.assertEqual(evaluate(self.cases[:1], predictions), results)
            self.assertEqual(results[0].category, "format_failure")
            self.assertEqual(errors, [])

    def test_all_ten_cases_and_full_replay(self):
        outputs = [json.dumps({"status": "insufficient_information" if case["requires_abstention"]
                               else "answered", "answer": case["expected_answer"]}) for case in self.cases]
        self.client.generate.side_effect = [self.result(output) for output in outputs]
        with TemporaryDirectory() as directory:
            results, errors, stdout = self.run_in(directory)
            self.assertEqual(self.client.generate.call_count, 10)
            self.assertTrue(all(result.passed for result in results))
            self.assertEqual(errors, [])
            self.assertIn("Pass Rate: 100.00%", stdout)
            path = Path(directory) / "run"
            self.assertEqual(evaluate(self.cases, load_predictions(path / "predictions.jsonl")), results)
            metadata = [json.loads(line) for line in (path / "metadata.jsonl").read_text().splitlines()]
            self.assertEqual(len(metadata), 10)
            self.assertEqual(metadata[0]["latency_seconds"], 1.2)

    def test_call_error_separate_from_scoring_failure_and_continues(self):
        self.client.generate.side_effect = [RuntimeError("SECRET must not be logged"), self.result("bad JSON")]
        with TemporaryDirectory() as directory:
            results, errors, stdout = self.run_in(directory, self.cases[:2])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].category, "format_failure")
            self.assertEqual(errors, [{"id": "qa-001", "error_type": "RuntimeError"}])
            self.assertIn("Call Failures: 1", stdout)
            self.assertIn("Failed: 1 (scoring failures)", stdout)
            self.assertIn("Incomplete batch", stdout)
            for path in (Path(directory) / "run").iterdir():
                self.assertNotIn("SECRET", path.read_text())
            self.assertNotIn("SECRET", stdout)

    def test_all_calls_failed_has_no_quality_failures(self):
        self.client.generate.side_effect = RuntimeError("offline")
        with TemporaryDirectory() as directory:
            results, errors, stdout = self.run_in(directory, self.cases[:1])
            self.assertEqual(results, [])
            self.assertEqual(len(errors), 1)
            self.assertIn("Failed: 0 (scoring failures)", stdout)

    def test_existing_run_is_not_overwritten(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "run").mkdir()
            with self.assertRaises(FileExistsError):
                self.run_in(directory)
            self.client.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
