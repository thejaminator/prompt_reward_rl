from enum import Enum
from typing import Optional

from slist import Slist

from api.dataset_paths import anthropic_harmless_test_path, anthropic_helpful_test_path
from api.set_key import set_openai_key
from evaluate.classification import get_pair_predicted_result
from parsing.parse_raw import AnthropicRawFormat, get_raw_anthropic
from settings import OPENAI_KEY


def get_harmless_test() -> Slist[AnthropicRawFormat]:
    harmless_test: Slist[AnthropicRawFormat] = get_raw_anthropic(
        anthropic_harmless_test_path
    )
    print(f"Loaded {len(harmless_test)} harmless test examples")
    return harmless_test


def get_helpful_test() -> Slist[AnthropicRawFormat]:
    helpful_test: Slist[AnthropicRawFormat] = get_raw_anthropic(
        anthropic_helpful_test_path
    )
    print(f"Loaded {len(helpful_test)} helpful test examples")
    return helpful_test


def get_harmless_helpful_test() -> Slist[AnthropicRawFormat]:
    return get_harmless_test() + get_helpful_test()


class TestDataset(str, Enum):
    HARMLESS = "harmless"
    HELPFUL = "helpful"
    HARMLESS_HELPFUL = "harmless_helpful"


def main(limit: int, test_dataset: TestDataset):
    # Load the test datasets
    all_test = (
        get_harmless_helpful_test()
        if test_dataset == TestDataset.HARMLESS_HELPFUL
        else (
            get_harmless_test()
            if test_dataset == TestDataset.HARMLESS
            else get_helpful_test()
        )
    )
    sample = all_test.shuffle(seed="123").take(limit)
    # Get the predictions
    predictions = sample.map(lambda pair: len(pair.chosen) >= len(pair.rejected))
    # Calculate the accuracy
    accuracy: Optional[float] = predictions.average()
    print(f"Accuracy: {accuracy} on {limit} samples")


if __name__ == "__main__":
    # Evaluate a strategy where we simply choose the longer one
    main(
        limit=1000,
        test_dataset=TestDataset.HARMLESS,
    )
