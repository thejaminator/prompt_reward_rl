from pathlib import Path

anthropic_harmless_path: Path = Path("datasets/anthropic/harmless-base-train.jsonl")
anthropic_helpful_path: Path = Path("datasets/anthropic/helpful-base-train.jsonl")
moderated_harmless_rejected = Path("datasets/processed/moderated-helpful-base-train.jsonl")
