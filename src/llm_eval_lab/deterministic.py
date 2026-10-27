"""Offline contract validation and exact-match scoring; standard library only.

Prediction batches must contain exactly one raw output per dataset ID.
CLI exit codes: 0 = all passed/validation succeeded, 1 = scoring failures,
2 = invalid dataset, prediction batch, or file input.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-JSON constant: {value}")


def _parse(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_reject_constant)


def _read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(_parse(line))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def validate_dataset(cases):
    if not cases:
        raise ValueError("Dataset must not be empty")
    ids = set()
    fields = {"id", "context", "question", "expected_answer", "requires_abstention"}
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or set(case) != fields:
            raise ValueError(f"Case {index}: invalid dataset fields")
        for field in ("id", "context", "question"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(f"Case {index}: {field} must be a non-empty string")
        if case["id"] in ids:
            raise ValueError(f"Duplicate dataset ID: {case['id']}")
        ids.add(case["id"])
        if type(case["requires_abstention"]) is not bool:
            raise ValueError(f"Case {index}: requires_abstention must be boolean")
        expected = case["expected_answer"]
        if case["requires_abstention"]:
            if expected is not None:
                raise ValueError(f"Case {index}: abstention requires null expected_answer")
        elif not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"Case {index}: answered case requires non-empty expected_answer")


def load_dataset(path):
    cases = _read_jsonl(path)
    validate_dataset(cases)
    return cases


def load_predictions(path):
    predictions = {}
    for index, record in enumerate(_read_jsonl(path), 1):
        if not isinstance(record, dict) or set(record) != {"id", "raw_output"}:
            raise ValueError(f"Prediction {index}: expected id and raw_output fields")
        case_id = record["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Prediction {index}: id must be a non-empty string")
        if not isinstance(record["raw_output"], str):
            raise ValueError(f"Prediction {index}: raw_output must be a string")
        if case_id in predictions:
            raise ValueError(f"Duplicate prediction ID: {case_id}")
        predictions[case_id] = record["raw_output"]
    return predictions


def normalize(answer: str) -> str:
    """Only trim surrounding whitespace and ignore case; retain punctuation."""
    return answer.strip().lower()


@dataclass(frozen=True)
class ScoreResult:
    case_id: str
    passed: bool
    category: str
    reason: str
    normalized_actual: str | None = None
    normalized_expected: str | None = None

    def to_dict(self):
        return asdict(self)


def score(case, raw_output: str) -> ScoreResult:
    """Score a validated dataset case without loading generation prompts."""
    def fail(category, reason):
        return ScoreResult(case["id"], False, category, reason)

    try:
        output = _parse(raw_output)
    except (ValueError, TypeError):
        return fail("format_failure", "invalid_json")
    if not isinstance(output, dict):
        return fail("format_failure", "top_level_must_be_object")
    if set(output) != {"status", "answer"}:
        return fail("format_failure", "expected_only_status_and_answer")
    status, answer = output["status"], output["answer"]
    if not isinstance(status, str):
        return fail("format_failure", "status_must_be_string")
    if status not in ("answered", "insufficient_information"):
        return fail("format_failure", "invalid_status")
    if status == "answered" and (not isinstance(answer, str) or not answer.strip()):
        return fail("format_failure", "answered_requires_non_empty_string")
    if status == "insufficient_information" and answer is not None:
        return fail("format_failure", "abstention_requires_null")

    if case["requires_abstention"]:
        if status != "insufficient_information":
            return fail("incorrect_abstention", "required_abstention_but_answered")
    else:
        if status != "answered":
            return fail("incorrect_abstention", "answerable_case_but_abstained")
        actual, expected = normalize(answer), normalize(case["expected_answer"])
        matched = actual == expected
        return ScoreResult(
            case["id"], matched, "pass" if matched else "incorrect_answer",
            "exact_match" if matched else "exact_match_mismatch", actual, expected,
        )
    return ScoreResult(case["id"], True, "pass", "correct_abstention")


def evaluate(cases, predictions):
    validate_dataset(cases)
    ids = {case["id"] for case in cases}
    missing, unknown = ids - predictions.keys(), predictions.keys() - ids
    if missing or unknown:
        raise ValueError(
            f"Prediction IDs mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return [score(case, predictions[case["id"]]) for case in cases]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path)
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        cases = load_dataset(args.dataset)
        if args.validate_only:
            abstentions = sum(case["requires_abstention"] for case in cases)
            print(f"Dataset valid: Total={len(cases)}, Answerable={len(cases) - abstentions}, Abstention={abstentions}")
            return 0
        results = evaluate(cases, load_predictions(args.predictions))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Input error: {error}\n")
    passed = sum(result.passed for result in results)
    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")
    print(f"Pass Rate: {passed / len(results):.2%}")
    for result in results:
        if not result.passed:
            print(f"{result.case_id}: {result.category} / {result.reason}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
