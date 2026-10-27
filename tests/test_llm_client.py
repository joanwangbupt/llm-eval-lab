from contextlib import redirect_stdout
from io import StringIO
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# LiteLLM must use its bundled pricing data even when this test command is
# executed without environment flags. Do not import model_comparison here:
# importing it loads .env and is unnecessary for these client unit tests.
with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
    from llm_eval_lab.llm_client import LLMClient
    from litellm.exceptions import APIConnectionError


class LLMClientTests(unittest.TestCase):
    def setUp(self):
        self.completion_patch = patch("llm_eval_lab.llm_client.completion")
        self.completion = self.completion_patch.start()
        self.addCleanup(self.completion_patch.stop)
        self.clock_patch = patch(
            "llm_eval_lab.llm_client.perf_counter", side_effect=[10.0, 10.1236]
        )
        self.clock = self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.completion.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Canberra"))],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=2, total_tokens=12
            ),
        )

    def test_forwards_request_and_serializes_result_without_printing(self):
        events = []

        def tick():
            events.append("clock")
            return 10.0 if len(events) == 1 else 10.1236

        def complete(**kwargs):
            events.append("completion")
            return self.completion.return_value

        self.clock.side_effect = tick
        self.completion.side_effect = complete
        stdout = StringIO()
        with redirect_stdout(stdout):
            result = LLMClient().generate(
                model="ollama_chat/test-model",
                prompt="Name Australia's capital.",
                api_base="http://localhost:11434",
                temperature=0.3,
            )

        self.completion.assert_called_once_with(
            model="ollama_chat/test-model",
            messages=[{"role": "user", "content": "Name Australia's capital."}],
            api_base="http://localhost:11434",
            temperature=0.3,
        )
        self.assertEqual(events, ["clock", "completion", "clock"])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(result.to_dict(), {
            "model": "ollama_chat/test-model",
            "output": "Canberra",
            "latency_seconds": 0.124,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        })

    def test_missing_usage_and_empty_content(self):
        del self.completion.return_value.usage
        self.completion.return_value.choices[0].message.content = None

        result = LLMClient().generate(model="test-model", prompt="Question")

        self.completion.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Question"}],
            api_base=None,
            temperature=0,
        )
        self.assertEqual(result.output, "")
        self.assertIsNone(result.prompt_tokens)
        self.assertIsNone(result.completion_tokens)
        self.assertIsNone(result.total_tokens)

    def test_partial_usage_preserves_missing_values(self):
        self.completion.return_value.usage = SimpleNamespace(prompt_tokens=0)

        result = LLMClient().generate(model="test-model", prompt="Question")

        self.assertEqual(result.prompt_tokens, 0)
        self.assertIsNone(result.completion_tokens)
        self.assertIsNone(result.total_tokens)

    def test_none_usage_preserves_missing_values(self):
        self.completion.return_value.usage = None

        result = LLMClient().generate(model="test-model", prompt="Question")

        self.assertIsNone(result.prompt_tokens)
        self.assertIsNone(result.completion_tokens)
        self.assertIsNone(result.total_tokens)

    def test_provider_error_propagates(self):
        error = APIConnectionError(
            message="Provider unavailable",
            llm_provider="ollama_chat",
            model="test-model",
        )
        self.completion.side_effect = error

        with self.assertRaises(APIConnectionError) as caught:
            LLMClient().generate(model="test-model", prompt="Question")

        self.assertIs(caught.exception, error)

    def test_latency_uses_elapsed_clock_time(self):
        self.clock.side_effect = [100.0, 108.11]

        result = LLMClient().generate(model="test-model", prompt="Question")

        self.assertEqual(result.latency_seconds, 8.11)
        self.assertEqual(self.clock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
