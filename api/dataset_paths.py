from pathlib import Path

anthropic_harmless_train_path: Path = Path(
    "datasets/anthropic/harmless-base-train.jsonl"
)
anthropic_harmless_test_path: Path = Path("datasets/anthropic/harmless-base-test.jsonl")
anthropic_helpful_train_path: Path = Path("datasets/anthropic/helpful-base-train.jsonl")
anthropic_helpful_test_path: Path = Path("datasets/anthropic/helpful-base-test.jsonl")
anthropic_online_train_path: Path = Path(
    "datasets/anthropic/helpful-online-train.jsonl"
)
anthropic_rejection_sampled_train_path: Path = Path(
    "datasets/anthropic/helpful-rejection-sampled-train.jsonl"
)
moderated_harmless_rejected = Path(
    "datasets/processed/moderated-helpful-base-train.jsonl"
)
