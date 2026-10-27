import os
from pathlib import Path
from unittest import result
from dotenv import load_dotenv
import yaml
from llm_eval_lab.llm_client import LLMClient

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = ROOT / "config" / "models.yaml"

def load_models() -> dict:
    with MODEL_CONFIG.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["models"]

def main() -> None:
    models = load_models()
    model_config = models["gemma4_12b"]

    client = LLMClient()
    result = client.generate(
        model=model_config["model"],
        api_base=os.getenv("OLLAMA_API_BASE"),
        temperature=model_config.get("temperature", 0),
        prompt="What is the capital of Australia? Answer in one short sentence.",
    )

    print("\n=== LLM Result ===")
    print(f"Model:              {result.model}")
    print(f"Output:             {result.output}")
    print(f"Latency:            {result.latency_seconds}s")
    print(f"Prompt tokens:      {result.prompt_tokens}")
    print(f"Completion tokens:  {result.completion_tokens}")
    print(f"Total tokens:       {result.total_tokens}")

if __name__ == "__main__":
    main()