from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from llm_eval_lab.deterministic import (
    evaluate, load_dataset, load_predictions, main, normalize, score, validate_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "mini_eval.jsonl"
PREDICTIONS = ROOT / "tests" / "fixtures" / "mini_predictions.jsonl"


def answered(value):
    return json.dumps({"status": "answered", "answer": value})


ABSTAIN = '{"status":"insufficient_information","answer":null}'


class DatasetTests(unittest.TestCase):
    def test_fixed_dataset_schema_count_and_ground_truth(self):
        cases = load_dataset(DATASET)
        self.assertEqual(len(cases), 10)
        self.assertEqual([case["id"] for case in cases], [f"qa-{i:03}" for i in range(1, 11)])
        self.assertEqual(sum(case["requires_abstention"] for case in cases), 3)
        for case in cases:
            if case["requires_abstention"]:
                self.assertIsNone(case["expected_answer"])
            else:
                self.assertIn(case["expected_answer"], case["context"])

    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            validate_dataset([])

    def test_invalid_schema_and_types_rejected(self):
        base = load_dataset(DATASET)[0]
        invalid = [None, {}, {**base, "extra": 1}]
        for field, value in (
            ("id", " "), ("context", None), ("question", 1),
            ("requires_abstention", 0), ("expected_answer", None),
            ("expected_answer", " "), ("expected_answer", 2018),
            ("requires_abstention", True),
        ):
            invalid.append({**base, field: value})
        for case in invalid:
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_dataset([case])

    def test_duplicate_dataset_ids_rejected(self):
        case = load_dataset(DATASET)[0]
        with self.assertRaisesRegex(ValueError, "Duplicate dataset ID"):
            validate_dataset([case, case])

    def test_invalid_jsonl_has_line_number(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{}\nnot json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bad.jsonl:2"):
                load_dataset(path)


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.cases = load_dataset(DATASET)
        self.case = self.cases[0]

    def test_correct_answer_and_structured_result(self):
        result = score(self.case, answered("2018"))
        self.assertTrue(result.passed)
        self.assertEqual(result.category, "pass")
        self.assertEqual(result.to_dict()["case_id"], "qa-001")
        self.assertEqual(result.normalized_actual, "2018")

    def test_wrong_answer(self):
        result = score(self.case, answered("2019"))
        self.assertFalse(result.passed)
        self.assertEqual((result.category, result.reason),
                         ("incorrect_answer", "exact_match_mismatch"))

    def test_correct_abstention(self):
        result = score(self.cases[7], ABSTAIN)
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "correct_abstention")

    def test_both_incorrect_abstention_directions(self):
        for case, raw, reason in (
            (self.case, ABSTAIN, "answerable_case_but_abstained"),
            (self.cases[7], answered("12"), "required_abstention_but_answered"),
        ):
            with self.subTest(reason=reason):
                result = score(case, raw)
                self.assertFalse(result.passed)
                self.assertEqual((result.category, result.reason),
                                 ("incorrect_abstention", reason))

    def test_invalid_json_and_non_json_wrapping(self):
        for raw in ("", "not JSON", "```json\n" + answered("2018") + "\n```",
                    answered("2018") + " explanation", '{"status":NaN,"answer":null}',
                    '{"status":"answered","answer":"wrong","answer":"2018"}'):
            with self.subTest(raw=raw):
                result = score(self.case, raw)
                self.assertEqual((result.category, result.reason),
                                 ("format_failure", "invalid_json"))

    def test_top_level_must_be_object(self):
        for raw in ("[]", '"2018"', "null", "42", "true"):
            with self.subTest(raw=raw):
                self.assertEqual(score(self.case, raw).reason, "top_level_must_be_object")

    def test_missing_and_extra_fields(self):
        for output in ({}, {"status": "answered"}, {"answer": "2018"},
                       {"status": "answered", "answer": "2018", "extra": 1}):
            with self.subTest(output=output):
                result = score(self.case, json.dumps(output))
                self.assertEqual(result.category, "format_failure")
                self.assertEqual(result.reason, "expected_only_status_and_answer")

    def test_status_type_and_enum(self):
        for status in (None, 1, [], {}, "unknown", "ANSWERED"):
            with self.subTest(status=status):
                result = score(self.case, json.dumps({"status": status, "answer": "2018"}))
                self.assertEqual(result.category, "format_failure")
                self.assertEqual(result.reason,
                                 "invalid_status" if isinstance(status, str) else "status_must_be_string")

    def test_answer_type_and_status_consistency(self):
        for answer in (None, 2018, True, [], {}, "", "  "):
            with self.subTest(answer=answer):
                self.assertEqual(score(self.case, answered(answer)).reason,
                                 "answered_requires_non_empty_string")
        for answer in ("2018", "", 0, False, []):
            with self.subTest(abstention_answer=answer):
                result = score(self.cases[7], json.dumps({
                    "status": "insufficient_information", "answer": answer,
                }))
                self.assertEqual(result.category, "format_failure")
                self.assertEqual(result.reason, "abstention_requires_null")

    def test_conservative_normalization(self):
        self.assertEqual(normalize("  TeAL\n"), "teal")
        self.assertTrue(score(self.cases[6], answered("  TEAL\n")).passed)
        self.assertFalse(score(self.cases[6], answered("teal.")).passed)
        self.assertFalse(score(self.cases[3], answered("Mira  Solen")).passed)

    def test_semantically_correct_sentence_fails_exact_match(self):
        result = score(self.case, answered("The company was founded in 2018."))
        self.assertFalse(result.passed)
        self.assertEqual(result.category, "incorrect_answer")
        self.assertEqual(result.normalized_expected, "2018")

    def test_repeated_scoring_is_identical_and_does_not_mutate_case(self):
        before = deepcopy(self.case)
        self.assertEqual(score(self.case, answered("2018")), score(self.case, answered("2018")))
        self.assertEqual(self.case, before)


class OfflineEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = load_dataset(DATASET)
        self.predictions = load_predictions(PREDICTIONS)

    def test_fixture_results(self):
        results = evaluate(self.cases, self.predictions)
        self.assertEqual(len(results), 10)
        self.assertEqual(sum(result.passed for result in results), 4)
        self.assertEqual({result.category for result in results}, {
            "pass", "format_failure", "incorrect_answer", "incorrect_abstention",
        })

    def test_missing_and_unknown_ids_rejected(self):
        for predictions in (
            {key: value for key, value in self.predictions.items() if key != "qa-001"},
            {**self.predictions, "unknown": ABSTAIN},
            {"unknown": ABSTAIN},
        ):
            with self.subTest(ids=list(predictions)), self.assertRaisesRegex(ValueError, "IDs mismatch"):
                evaluate(self.cases, predictions)

    def test_duplicate_and_invalid_prediction_records_rejected(self):
        record = {"id": "qa-001", "raw_output": answered("2018")}
        for records in (
            [record, record], [{"id": "qa-001"}],
            [{"id": "qa-001", "raw_output": {}}],
            [{"id": "", "raw_output": ""}],
        ):
            with self.subTest(records=records), TemporaryDirectory() as directory:
                path = Path(directory) / "predictions.jsonl"
                path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_predictions(path)

    def test_different_batch_and_order_supported(self):
        predictions = {
            case["id"]: ABSTAIN if case["requires_abstention"] else answered(case["expected_answer"])
            for case in reversed(self.cases)
        }
        results = evaluate(self.cases, predictions)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual([result.case_id for result in results], [case["id"] for case in self.cases])

    def test_cli_summary_and_failure_exit_code(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            code = main(["--dataset", str(DATASET), "--predictions", str(PREDICTIONS)])
        self.assertEqual(code, 1)
        for text in ("Total: 10", "Passed: 4", "Failed: 6", "Pass Rate: 40.00%",
                     "qa-006: incorrect_answer / exact_match_mismatch"):
            self.assertIn(text, stdout.getvalue())
        for case_id in ("qa-002", "qa-003", "qa-004", "qa-005", "qa-006", "qa-008"):
            self.assertIn(case_id, stdout.getvalue())

    def test_cli_validation(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            code = main(["--dataset", str(DATASET), "--validate-only"])
        self.assertEqual(code, 0)
        self.assertIn("Total=10, Answerable=7, Abstention=3", stdout.getvalue())

    def test_cli_input_error(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                main(["--dataset", str(DATASET), "--predictions", str(path)])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("IDs mismatch", stderr.getvalue())

    def test_scorer_imports_without_site_packages(self):
        # -S removes third-party site packages; this path must not import LiteLLM.
        result = subprocess.run(
            [sys.executable, "-S", "-m", "llm_eval_lab.deterministic",
             "--dataset", str(DATASET), "--validate-only"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
