from enum import Enum
from typing import Optional

from slist import Slist

from api.dataset_paths import anthropic_harmless_test_path, anthropic_helpful_test_path
from api.set_key import set_openai_key
from calculate_reward import AnthropicRawFormat, get_raw_anthropic
from evaluate.classification import get_pair_predicted_chosen
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


def main(limit: int, model_id: str, openai_api_key: str, test_dataset: TestDataset):
    set_openai_key(openai_api_key)
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
    sample = all_test.shuffle().take(limit)
    # Get the predictions
    predictions = sample.map(lambda pair: get_pair_predicted_chosen(model_id, pair))
    # Calculate the accuracy
    accuracy: Optional[float] = predictions.average()
    print(f"Accuracy: {accuracy} for {model_id} on {limit} samples")


if __name__ == "__main__":
    # Joint model on 80k samples babbage:ft-leadiq:assistant-reward-model-2022-12-20-09-34-26 0.67
    # Joint model on 10k samples babbage:ft-leadiq:assistant-reward-model-2022-12-19-15-51-58 0.6
    #
    main(
        limit=1000,
        model_id="babbage:ft-leadiq:assistant-reward-model-2022-12-19-15-51-58",
        openai_api_key=OPENAI_KEY,
        test_dataset=TestDataset.HELPFUL,
    )
