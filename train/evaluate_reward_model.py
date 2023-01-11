from enum import Enum
from typing import Optional

from slist import Slist

from api.dataset_paths import anthropic_harmless_test_path, anthropic_helpful_test_path
from api.set_key import set_openai_key
from evaluate.classification import get_pair_predicted_result, PairPrediction
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


def plot_distribution_chosen(predictions: Slist[float]) -> None:
    import matplotlib.pyplot as plt

    plt.hist(predictions, bins=20)
    # Set title to "Probability distribution for chosen"
    plt.title("Probability distribution for chosen")
    # Save the plot to "chosen_distribution.png"
    plt.savefig("chosen_distribution.png")
    plt.show()


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
    sample = all_test.shuffle(seed="123").take(limit)
    # Get the predictions
    predictions: Slist[PairPrediction] = sample.map(lambda pair: get_pair_predicted_result(model_id, pair))
    # Calculate the accuracy
    accuracy: Optional[float] = predictions.map(lambda x: x.is_chosen_higher_proba).average()
    print(f"Accuracy: {accuracy} for {model_id} on {limit} samples")
    # Plot the distribution of predictions for the chosen
    plot_distribution_chosen(predictions.map(lambda x: x.chosen_proba))


if __name__ == "__main__":
    # Joint model on 80k samples babbage:ft-leadiq:assistant-reward-model-2022-12-20-09-34-26
    # batch size 32
    # 0.67 on joint, 0.67 on harmless, 0.6675 on helpful
    # Two epochs did not help babbage:ft-leadiq:leadiq-assistant-reward-model-2023-01-02-20-05-37
    # batch size 256 babbage:ft-leadiq:leadiq-assistant-reward-model-2023-01-03-14-24-13
    # 0.6125 on joint

    # Joint model on 10k pairs babbage:ft-leadiq:assistant-reward-model-2022-12-19-15-51-58 0.6
    # Helpful model on 43835 pairs babbage:ft-leadiq:helpful-reward-2022-12-22-08-04-46 0.721 on helpful
    # Harmless model on 42537 pairs babbage:ft-leadiq:harmless-reward-2022-12-22-08-55-12 0.717 on harmless
    main(
        limit=400,
        model_id="babbage:ft-leadiq:helpful-reward-2022-12-22-08-04-46",
        openai_api_key=OPENAI_KEY,
        test_dataset=TestDataset.HARMLESS_HELPFUL,
    )

    # 1,102,676 tokens for 6122 requests (samples)
    # 1,102,676 / 6122 * 1000 = 180,116 tokens per 1000 samples
    # 180,116 / 1000 * 0.0024 = $0.43 to evaluate 1000 samples
