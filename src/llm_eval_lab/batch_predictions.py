"""Generate raw predictions and score them. No automatic JSON repair.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from llm_eval_lab.deterministic import evaluate, load_dataset, load_predictions, validate_dataset


ROOT = Path(__file__).resolve().parents[2]


def render_prompt(template, *, context, question):
    """Substitute only input fields, preserving JSON examples and input braces."""
    for field in ("context", "question"):
        if "{" + field + "}" not in template:
            raise ValueError(f"Prompt missing {{{field}}} placeholder")
    values = {"context": context, "question": question}
    return re.sub(r"\{(context|question)\}", lambda match: values[match[1]], template)


def run_batch(cases, template, *, client, model_config, api_base, output_dir):
    validate_dataset(cases)
    prompts = [render_prompt(template, context=case["context"], question=case["question"])
               for case in cases]
    model = model_config["model"]
    temperature = model_config.get("temperature", 0)
    output_dir = Path(output_dir)
    # Refuse to overwrite an earlier experiment, including partially completed runs.
    output_dir.mkdir(parents=True, exist_ok=False)
    errors = []
    successful = []
    with (output_dir / "predictions.jsonl").open("x", encoding="utf-8") as predictions, \
         (output_dir / "call_errors.jsonl").open("x", encoding="utf-8") as error_file, \
         (output_dir / "metadata.jsonl").open("x", encoding="utf-8") as metadata:
        for case, prompt in zip(cases, prompts):
            print(f"Generating {case['id']}...", flush=True)
            try:
                result = client.generate(
                    model=model, prompt=prompt, api_base=api_base, temperature=temperature,
                )
            except Exception as error:
                # Provider messages may contain connection credentials; save type only.
                record = {"id": case["id"], "error_type": type(error).__name__}
                error_file.write(json.dumps(record) + "\n")
                error_file.flush()
                errors.append(record)
                continue
            predictions.write(json.dumps({"id": case["id"], "raw_output": result.output},
                                         ensure_ascii=False) + "\n")
            predictions.flush()
            metadata.write(json.dumps({
                "id": case["id"], "model": result.model, "temperature": temperature,
                "latency_seconds": result.latency_seconds,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            }) + "\n")
            metadata.flush()
            successful.append(case)
    # Read the persisted outputs through the existing strict prediction loader.
    results = evaluate(successful, load_predictions(output_dir / "predictions.jsonl")) if successful else []
    passed = sum(result.passed for result in results)
    print(f"Predictions: {output_dir / 'predictions.jsonl'}")
    print(f"Total: {len(cases)}")
    print(f"Generated: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed} (scoring failures)")
    print(f"Call Failures: {len(errors)}")
    print(f"Pass Rate: {passed / len(cases):.2%} (passed / total cases)")
    for result in results:
        if not result.passed:
            print(f"{result.case_id}: {result.category} / {result.reason}")
    for error in errors:
        print(f"{error['id']}: call_failure / {error['error_type']}")
    if errors:
        print("Incomplete batch: full-dataset offline replay requires outputs for failed calls.")
    return results, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/mini_eval.jsonl")
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompts/context_qa.txt")
    parser.add_argument("--config", type=Path, default=ROOT / "config/models.yaml")
    parser.add_argument("--model-key", default="gemma4_12b")
    parser.add_argument("--api-base", default=os.getenv("OLLAMA_API_BASE"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    # Import the model client only for real generation, never for offline scoring.
    import yaml
    from llm_eval_lab.llm_client import LLMClient

    try:
        cases = load_dataset(args.dataset)
        template = args.prompt.read_text(encoding="utf-8")
        with args.config.open(encoding="utf-8") as stream:
            model_config = yaml.safe_load(stream)["models"][args.model_key]
        output_dir = args.output_dir or ROOT / "results/runs" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        )
        results, errors = run_batch(
            cases, template, client=LLMClient(), model_config=model_config,
            api_base=args.api_base, output_dir=output_dir,
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.exit(2, f"Input/output error ({type(error).__name__}); check paths and configuration.\n")
    return 3 if errors else (0 if all(result.passed for result in results) else 1)


if __name__ == "__main__":
    raise SystemExit(main())
