from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from llm_eval_lab.batch_predictions import main
from llm_eval_lab.deterministic import evaluate, load_dataset, load_predictions


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/refund_case_v1.jsonl"
PROMPT = ROOT / "prompts/context_qa_explanation.txt"


class RefundCaseTests(unittest.TestCase):
    def test_dataset_contains_fixed_policy_record_and_ground_truth(self):
        cases = load_dataset(DATASET)
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertFalse(case["requires_abstention"])
        self.assertEqual(case["expected_answer"],
                         "Yes. The customer is eligible for a refund because cancellation was "
                         "requested within 14 days of renewal and the service was not used during that period.")
        for fact in ("within 14 days", "has not been used", "end of the current billing term",
                     "Subscription: Annual", "1 October 2026", "9 October 2026",
                     "No activity since renewal"):
            self.assertIn(fact, case["context"])

    def test_cli_accepts_single_case_and_custom_prompt_with_offline_replay(self):
        # A controlled paraphrase tests plumbing; it is not a real model result.
        raw = json.dumps({"status": "answered", "answer":
                          "Yes: the request was made 8 days after renewal, and there was no usage."})
        client = Mock()
        client.generate.return_value = SimpleNamespace(
            output=raw, model="ollama_chat/mock", latency_seconds=1.0,
            prompt_tokens=10, completion_tokens=20, total_tokens=30,
        )
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            with TemporaryDirectory() as directory, \
                 patch("llm_eval_lab.llm_client.LLMClient", return_value=client), \
                 redirect_stdout(StringIO()):
                output = Path(directory) / "run"
                code = main(["--dataset", str(DATASET), "--prompt", str(PROMPT),
                             "--api-base", "http://localhost:11434", "--output-dir", str(output)])
                self.assertEqual(code, 1)
                client.generate.assert_called_once()
                prompt = client.generate.call_args.kwargs["prompt"]
                case = load_dataset(DATASET)[0]
                self.assertIn(case["context"], prompt)
                self.assertIn(case["question"], prompt)
                self.assertIn("in your own words", prompt)
                self.assertNotIn(case["expected_answer"], prompt)
                predictions = load_predictions(output / "predictions.jsonl")
                self.assertEqual(predictions[case["id"]], raw)
                result = evaluate([case], predictions)[0]
                self.assertEqual(result.category, "incorrect_answer")
                self.assertEqual(result.reason, "exact_match_mismatch")


if __name__ == "__main__":
    unittest.main()
