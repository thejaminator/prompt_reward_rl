from pathlib import Path

anthropic_harmless_train_path: Path = Path("datasets/anthropic/harmless-base-train.jsonl")
anthropic_harmless_test_path: Path = Path("datasets/anthropic/harmless-base-test.jsonl")
anthropic_helpful__train_path: Path = Path("datasets/anthropic/helpful-base-train.jsonl")
anthropic_helpful_test_path: Path = Path("datasets/anthropic/helpful-base-test.jsonl")
moderated_harmless_rejected = Path(
    "datasets/processed/moderated-helpful-base-train.jsonl"
)
