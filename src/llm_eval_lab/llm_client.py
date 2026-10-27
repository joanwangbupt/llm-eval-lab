from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from litellm import completion

@dataclass
class LLMResult:
    model: str
    output: str
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    reasoning_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.reasoning_content is None:
            result.pop("reasoning_content")
        return result


class LLMClient:
    def generate(
            self,
            *,
            model: str,
            prompt: str,
            api_base: str | None = None,
            temperature: float = 0,
    ) -> LLMResult:
        start_time = perf_counter()
        response = completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            api_base=api_base,
            temperature=temperature,
        )

        latency = perf_counter() - start_time

        usage = getattr(response, "usage", None)

        return LLMResult(
            model=model,
            output=response.choices[0].message.content or "",
            latency_seconds=round(latency, 3),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            reasoning_content=getattr(response.choices[0].message, "reasoning_content", None),
        )
