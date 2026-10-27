"""Compare exact match with one local DeepEval semantic metric on saved output.

Exit codes: 0 = semantic score obtained (not a quality pass),
2 = invalid input/format or unavailable answer, 3 = Judge execution failure.
"""

import argparse
import json
import os
from pathlib import Path

# This local demo must not load secret files, send telemetry, or persist DeepEval
# settings. Set these before importing DeepEval, including in isolated tests.
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_FILE_SYSTEM"] = "READ_ONLY"
os.environ["DEEPEVAL_UPDATE_WARNING_OPT_IN"] = "0"

from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from llm_eval_lab.deterministic import evaluate, load_dataset, load_predictions, score


ROOT = Path(__file__).resolve().parents[2]
CRITERIA = (
    "Assess whether the actual output correctly answers the refund eligibility "
    "question using the policy and customer record in the input and the expected "
    "output as a reference. Check the eligibility conclusion and the application "
    "of both conditions: cancellation within 14 days of renewal and no service "
    "usage during that period. Penalize an incorrect business conclusion, omission "
    "of either key condition, or explanations contradicting the context. Accept "
    "equivalent wording and any order of explanation. Do not penalize lack of "
    "verbatim agreement with the expected output."
)


def build_test_case(case, raw_output):
    """Use the existing contract validator before extracting answer, never repair."""
    deterministic = score(case, raw_output)
    if deterministic.category == "format_failure":
        raise ValueError(f"Semantic evaluation unavailable: {deterministic.reason}")
    answer = json.loads(raw_output)["answer"]
    if answer is None or case["expected_answer"] is None:
        raise ValueError("Semantic evaluation requires actual and expected answer strings")
    return LLMTestCase(
        input=f"Context:\n{case['context']}\n\nQuestion:\n{case['question']}",
        actual_output=answer,
        expected_output=case["expected_answer"],
    )


def measure_semantic(test_case, *, judge_model, api_base):
    judge = OllamaModel(model=judge_model, base_url=api_base, temperature=0, timeout=180)
    metric = GEval(
        name="Semantic Correctness",
        criteria=CRITERIA,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=None,
        async_mode=False,
    )
    metric.measure(test_case)
    return metric.score, metric.reason


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/refund_case_v1.jsonl")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--case-id", default="refund-001")
    parser.add_argument("--judge-model", default=os.getenv("OLLAMA_JUDGE_MODEL"))
    parser.add_argument("--api-base", default=os.getenv("OLLAMA_API_BASE"))
    args = parser.parse_args(argv)
    if not args.judge_model or not args.api_base:
        parser.error("Set --judge-model / OLLAMA_JUDGE_MODEL and --api-base / OLLAMA_API_BASE")
    try:
        cases = load_dataset(args.dataset)
        predictions = load_predictions(args.predictions)
        results = evaluate(cases, predictions)
        case = next((case for case in cases if case["id"] == args.case_id), None)
        if case is None:
            raise ValueError("Requested case ID not found in dataset")
        deterministic = next(result for result in results if result.case_id == args.case_id)
        print(f"Case ID: {args.case_id}")
        print(f"Deterministic Result: {'Pass' if deterministic.passed else 'Fail'} / "
              f"{deterministic.category} / {deterministic.reason}")
        test_case = build_test_case(case, predictions[args.case_id])
    except (OSError, ValueError) as error:
        # Only validation errors, not raw provider responses, are reported here.
        print(f"Semantic evaluation unavailable ({type(error).__name__}): {error}")
        return 2
    print("DeepEval Metric: Semantic Correctness")
    print(f"Judge Model: {args.judge_model}")
    print("Limit: LLM Judge may be biased; scores are not objective ground truth.", flush=True)
    try:
        semantic_score, reason = measure_semantic(
            test_case, judge_model=args.judge_model, api_base=args.api_base,
        )
    except Exception as error:
        print(f"Judge execution failed ({type(error).__name__}); no semantic score obtained.")
        print("Check Ollama connectivity and native structured-output compatibility.")
        return 3
    print(f"DeepEval Score (0–1): {semantic_score}")
    print(f"Judge Reason: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
