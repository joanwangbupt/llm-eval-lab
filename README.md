# LLM Eval Lab

**From Exact Match to LLM-as-a-Judge: A Practical Exploration of LLM Evaluation**

A lightweight, reproducible experiment exploring how to evaluate LLM outputs, understand the limitations of rule-based scoring, and compare different models using a consistent evaluation approach.

Built with Python, LiteLLM, Ollama, and DeepEval.

## 1. Why I Built This

While working on AI workflows and retrieval-augmented applications, I repeatedly encountered the same engineering questions:

- How do we know whether a prompt change improves the quality of an LLM's output?
- How can we detect regressions when changing prompts, models, or application logic?
- How do we compare models consistently, rather than relying on a few subjective examples?
- What does it actually mean for an LLM response to be *correct*?

These questions become especially important as AI applications evolve. A change that improves one response may degrade another, and a perfectly valid answer may look very different from a reference answer.

I wanted to explore the evaluation methodology separately from any particular application.

Rather than building a full evaluation platform, I created a small experiment to understand the progression from deterministic scoring to semantic evaluation using LLM-as-a-Judge.

The focus is on the reasoning behind each evaluation approach, not on the number of metrics or frameworks integrated.

## 2. Experiment Overview

The experiment follows a simple evaluation workflow:

```text
 Dataset Inputs (Context + Question)
               |
               v
       Generation Prompt
               |
               v
       LiteLLM + Ollama
               |
               v
       Raw Model Output
               |
        +------+------+
        |             |
        v             v
  Deterministic    DeepEval GEval
     Scorer        LLM-as-a-Judge
        |             |
        +------+------+
               |
               v
      Compare and Analyse
```

The dataset, expected answers, generation prompt, and scoring rules are kept separate so that the same evaluation approach can be reused across models. Only context and question are inserted into the generation prompt. Expected answers and abstention labels are used for evaluation, never sent to the model being tested.

## 3. How the Experiment Evolved

### Phase 1 — Establishing a Reliable Model Interface

I started with LiteLLM and a local Ollama model to establish a consistent model invocation interface.

The initial focus was straightforward:

- Send a prompt and retrieve the model response.
- Capture latency and token usage where available.
- Handle missing usage information and API errors.
- Validate the client behaviour with isolated unit tests.

This provided the foundation for running repeatable evaluation experiments.

### Phase 2 — Deterministic Evaluation

Next, I created a fixed dataset containing 10 context-based QA cases, with manually verifiable expected answers.

The deterministic scorer checks:

- JSON validity and required fields.
- Response status and answer consistency.
- Normalised Exact Match.
- Correct abstention when information is insufficient.

The scorer runs entirely offline and does not require an LLM.

An intentionally mixed prediction fixture produced **4 passes and 6 failures**, validating the scorer's ability to distinguish correct answers, incorrect answers, formatting errors, and incorrect abstentions.

### Phase 3 — Evaluating Real Model Outputs

I connected the dataset to the Ollama generation pipeline and ran the 10 cases using Gemma.

**Result: 10/10 passed deterministic evaluation.**

This confirmed that the generation, prediction storage, and scoring pipeline worked end to end.

However, these questions were relatively simple and had short, well-defined answers. They did not fully expose the limitations of Exact Match.

### Phase 4 — A More Realistic Business Question

To explore those limitations, I introduced a single SaaS subscription refund eligibility case.

**Policy**

Annual subscriptions are eligible for a refund when cancellation is requested within 14 days of renewal and the service has not been used during that period.

**Customer record**

- Annual subscription.
- Renewed on 1 October 2026.
- Cancellation requested on 9 October 2026.
- No service usage since renewal.

**Question**

*Is this customer eligible for a refund under the stated policy? Explain your reasoning.*

The expected conclusion was that the customer qualifies for a refund because the cancellation request was made within the required period and the service had not been used.

Gemma generated a correct explanation, including the relevant dates and policy conditions.

The output was valid JSON, but the answer did not match the reference text exactly.

**Deterministic result: FAIL — `exact_match_mismatch`**

Manual review confirmed that the business conclusion and supporting reasoning were correct.

This was a false negative when Exact Match was used as a measure of semantic correctness.

The scorer was behaving as implemented. Exact Match was too restrictive to measure semantic correctness for this explanatory answer.

### Phase 5 — Introducing DeepEval

This provided a practical reason to introduce DeepEval.

Using DeepEval's `GEval` metric, I evaluated the previously saved model response without regenerating it.

The evaluation assessed semantic correctness, including whether the response:

- Reached the correct refund eligibility conclusion.
- Applied the relevant policy conditions.
- Used the customer record accurately.
- Was semantically consistent with the expected answer.

**DeepEval result: 1.0 / 1.0**

The Judge explicitly recognised that the answer correctly applied the cancellation deadline and the no-usage requirement.

The same response therefore produced two different results:

| Evaluation method | Result |
|---|---|
| Exact Match | Fail |
| DeepEval GEval | 1.0 / 1.0 |
| Manual review | Correct |

This demonstrated the benefit of semantic evaluation for natural-language answers.

It also highlighted an important distinction: deterministic validation and semantic evaluation measure different properties and should complement rather than automatically replace each other.

### Phase 6 — Switching Models

Finally, I repeated the refund experiment using a second locally hosted model, Qwen, while keeping the dataset, generation prompt, and evaluation criteria unchanged.

Gemma remained the Judge for both model evaluations.

Both models produced correct answers and received a DeepEval semantic correctness score of 1.0.

The experiment demonstrated that the same evaluation workflow could be reused across models without changing the test case or scoring criteria.

## 4. Results

### Evaluation Progression

| Stage | Evaluation Approach | Observed Result | Key Finding |
|---|---|---|---|
| Fixed QA fixture | Deterministic scoring | 4 Pass / 6 Fail (intentional) | Validated the scoring rules and failure classifications |
| Real basic QA | Gemma + Exact Match | 10/10 Pass | Suitable for constrained, short factual answers |
| Refund eligibility | Gemma + Exact Match | Fail | Correct natural-language answers can fail strict matching |
| Semantic evaluation | Gemma + DeepEval GEval | 1.0 | Judge recognised the correct conclusion and reasoning |
| Model switching | Qwen + DeepEval GEval | 1.0 | Reused the same evaluation approach with another model |

### Model Comparison — Refund Eligibility Case

| Metric | Gemma 4 (26B A4B) | Qwen 3.8 (27B) |
|---|---|---|
| JSON format | Pass | Pass |
| Exact Match | Fail | Fail |
| DeepEval GEval | 1.0 | 1.0 |
| Manual conclusion | Correct | Correct |
| Generation latency | 14.149s | 33.97s |
| Prompt tokens | 263 | 259 |
| Completion tokens | 654 | 295 |
| Total tokens | 917 | 554 |

**Judge:** `gemma4:26b-a4b-it-qat`, fixed across both evaluations.

The Judge recognised that both answers correctly applied the refund eligibility conditions.

The latency and token differences are observations from individual local runs, not a controlled performance benchmark. Model loading, caching, hardware utilisation, and thinking-token accounting may influence these values. Token usage is provider-reported, not an API cost estimate; separate thinking-token counts were not recorded. Qwen's returned thinking text is stored separately from its final answer.

These results do not establish which model is generally better.

## 5. Running the Experiment

### Requirements

- Python 3.12 (the version used for these experiments; package requirement: >=3.12)
- `uv`
- Ollama with the required local models
- Project dependencies defined in `pyproject.toml` and `uv.lock`

Run the following Bash commands from the repository root (for example, in a WSL terminal). Install the locked dependencies:

```bash
uv sync --frozen
```

Run all automated tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The completed experiment passed **51 automated tests**. These tests use mocks for model calls and do not require Ollama or external services.

### Run Deterministic Evaluation Offline

```bash
PYTHONPATH=src .venv/bin/python -m llm_eval_lab.deterministic \
  --dataset datasets/mini_eval.jsonl \
  --predictions tests/fixtures/mini_predictions.jsonl
```

The fixture intentionally contains failures, so this command reports a 40% pass rate and exits with code 1.

### Generate a Real Model Response

Start Ollama and make its API accessible. The model tags in `config/models.yaml` must already be available in Ollama. Set `OLLAMA_API_BASE` to the address reachable from the Python environment:

```bash
export OLLAMA_API_BASE=http://localhost:11434
```

`localhost` applies when the service is reachable there. If Python runs in WSL and Ollama runs on Windows, use the reachable Windows host address instead. The batch and semantic entry points use CLI arguments or process environment variables; they do not load `.env`. Do not commit a machine-specific address.

Generate the refund answer with the default Gemma model:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONPATH=src \
.venv/bin/python -m llm_eval_lab.batch_predictions \
  --dataset datasets/refund_case_v1.jsonl \
  --prompt prompts/context_qa_explanation.txt \
  --api-base "$OLLAMA_API_BASE"
```

To run Qwen instead, add:

```text
--model-key qwen3_8_27b
```

The runner writes predictions and available metadata under `results/runs/`.

Without `--output-dir`, each run gets a new directory. An explicit output directory must not already exist. `predictions.jsonl` contains only `id` and `raw_output`; `metadata.jsonl` contains available model, latency, token, and thinking information; `call_errors.jsonl` records call failures separately. Local run files are ignored by Git and are not included in a fresh clone.

The runner exits with code 0 when all cases pass, 1 for scoring failures, 2 for input/output errors, and 3 for model call failures. An Exact Match failure on an explanatory answer is an expected experimental outcome, not necessarily a generation error. Incomplete batches cannot be replayed against the full dataset until all case outputs are present.

To replay a completed refund run offline:

```bash
PYTHONPATH=src .venv/bin/python -m llm_eval_lab.deterministic \
  --dataset datasets/refund_case_v1.jsonl \
  --predictions "results/runs/<run-id>/predictions.jsonl"
```

Replace `<run-id>` with the directory printed by the runner. To generate the original 10-case benchmark, use `datasets/mini_eval.jsonl` with `prompts/context_qa.txt` instead.

### Run DeepEval Semantic Evaluation

Using the predictions from a completed generation run:

```bash
PYTHONPATH=src .venv/bin/python -m llm_eval_lab.semantic_eval \
  --dataset datasets/refund_case_v1.jsonl \
  --predictions "results/runs/<run-id>/predictions.jsonl" \
  --judge-model gemma4:26b-a4b-it-qat \
  --api-base "$OLLAMA_API_BASE"
```

Replace `<run-id>` with the generated run directory.

The semantic evaluation reuses the stored answer and does not regenerate it. It extracts `answer` only after the existing JSON contract checks pass. Invalid format or unavailable answer text prevents semantic scoring; Judge execution failures are reported separately from quality scores.

The Judge stays Gemma even when generation uses Qwen. DeepEval runs in score-only mode: exit code 0 means a score was obtained, not that the answer passed a quality threshold. Codes 2 and 3 indicate invalid/unavailable input and Judge execution failure respectively. The entry point disables dotenv autoload, telemetry, update checks, and DeepEval file writes; it prints the score and reason to the terminal.

## 6. Key Takeaways

**1. Evaluation starts with defining correctness.**

Before choosing an evaluation tool, we need to establish the task, expected behaviour, reference information, and acceptance criteria.

**2. Deterministic evaluation is valuable, but task-dependent.**

Exact Match and schema validation remain useful for structured outputs and explicit rules. They are less suitable as the sole measure of natural-language correctness.

**3. LLM-as-a-Judge adds semantic understanding, not objective truth.**

DeepEval made it possible to evaluate whether a response expressed the correct business conclusion and reasoning despite different wording.

However, Judge scores depend on the model, rubric, and evaluation context. They can be biased or inconsistent and should be validated against human judgement when used for important decisions.

**4. Keep evaluation criteria consistent when comparing models.**

A meaningful model comparison requires a shared dataset, prompt, reference answers, and evaluation criteria.

**5. Preserve raw outputs and make experiments reproducible.**

Saving model responses separately from evaluation results allows scoring methods to evolve without repeatedly calling the generation model.

## 7. Scope and Limitations

This repository is an exploration of evaluation methodology, not a production evaluation platform.

The semantic experiment used one realistic business case, a single evaluation run per model, and a Gemma Judge. Because Gemma also evaluated its own generated answer, model-related Judge bias is possible.

The project does not establish statistical model rankings, Judge reliability across diverse tasks, or production-ready quality thresholds.

Reproducibility here means fixed inputs, locked Python dependencies, and offline replay of saved outputs. New generation or Judge calls may produce different results even at temperature 0; Ollama model tags and local runtime versions are not pinned by `uv.lock`.

Prompt regression, RAG-specific metrics, broader evaluation datasets, and CI quality gates are potential applications of the methodology but are not fully implemented in this demo.

## 8. Final Reflection

**LLM Evaluation is not just about scoring model outputs. It is about defining what correctness means, choosing appropriate evaluation methods, and understanding their limitations.**

The most useful outcome of this experiment was not a perfect score. It was observing a genuine limitation of Exact Match, understanding why it occurred, and using a more suitable evaluation method to examine the same answer.

The resulting approach provides a practical foundation for evaluating prompt changes, model replacements, and regressions in larger AI applications.

Licensed under the [MIT License](LICENSE).
