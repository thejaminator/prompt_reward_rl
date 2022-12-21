from typing import Optional

from slist import Slist

from api.dataset_paths import anthropic_harmless_test_path, anthropic_helpful_test_path
from calculate_reward import AnthropicRawFormat, get_raw_anthropic


def get_harmless_helpful_test() -> Slist[AnthropicRawFormat]:
    harmless_test: Slist[AnthropicRawFormat] = get_raw_anthropic(
        anthropic_harmless_test_path
    )
    print(f"Loaded {len(harmless_test)} harmless test examples")
    helpful_test: Slist[AnthropicRawFormat] = get_raw_anthropic(
        anthropic_helpful_test_path
    )
    print(f"Loaded {len(helpful_test)} helpful test examples")
    return harmless_test + helpful_test


def get_positive_class_proba(model_id: str, prompt: str) -> float:
    """Returns the probability of the positive class"""
    return 1.0


def get_pair_predicted_chosen(model_id: str, pair: AnthropicRawFormat) -> bool:
    """Returns true if the model predicts the chosen completion"""
    chosen_proba = get_positive_class_proba(model_id, pair.chosen)
    rejected_proba = get_positive_class_proba(model_id, pair.rejected)
    return chosen_proba > rejected_proba


def main(limit: int, model_id: str):
    # Load the test datasets
    all_test = get_harmless_helpful_test()
    sample = all_test.shuffle().take(limit)
    # Get the predictions
    predictions = sample.map(lambda pair: get_pair_predicted_chosen(model_id, pair))
    # Calculate the accuracy
    accuracy: Optional[float] = predictions.average()
    print(f"Accuracy: {accuracy} for {model_id} on {limit} samples")


if __name__ == "__main__":
    main(limit=100, model_id=...)
