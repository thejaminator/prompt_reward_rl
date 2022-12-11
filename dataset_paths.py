from pathlib import Path

anthropic_harmless_path: Path = Path("datasets/anthropic/harmless-base-train.jsonl")
anthropic_helpful_path: Path = Path("datasets/anthropic/harmless-base-train.jsonl")
processed_completions = Path("datasets/processed/completions.json")
moderated_completions = Path("datasets/processed/completions_moderated.json")
