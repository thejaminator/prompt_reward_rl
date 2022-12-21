from typing import Optional

from slist import Slist

from api.dataset_paths import anthropic_harmless_test_path, anthropic_helpful_test_path
from api.set_key import set_openai_key
from calculate_reward import AnthropicRawFormat, get_raw_anthropic
from evaluate.inference import (
    get_openai_completion,
    OpenaiInferenceConfig,
    TokenInfo,
    TokenProba,
)
from settings import OPENAI_KEY
from train.train_reward_model import (
    POSITIVE_TOKEN,
    NEGATIVE_TOKEN,
    format_dialogue_into_prompt,
)
import numpy as np


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
    # We just need 1 token
    config = OpenaiInferenceConfig(model=model_id, max_tokens=1)
    full_response = get_openai_completion(config=config, prompt=prompt)
    positive_token_info: TokenProba = full_response.completion_token_infos.first_or_raise().top_5_tokens.find_one_or_raise(
        lambda x: x.token == POSITIVE_TOKEN
    )
    negative_token_info: TokenProba = full_response.completion_token_infos.first_or_raise().top_5_tokens.find_one_or_raise(
        lambda x: x.token == NEGATIVE_TOKEN
    )
    # convert log_prob to prob
    positive_proba = np.exp(positive_token_info.log_prob)
    negative_proba = np.exp(negative_token_info.log_prob)
    # normalize the probabilities to equal to 1.0
    normalize_positive_proba = positive_proba / (positive_proba + negative_proba)
    return normalize_positive_proba


def get_pair_predicted_chosen(model_id: str, pair: AnthropicRawFormat) -> bool:
    """Returns true if the model predicts the chosen completion"""
    formatted_chosen_prompt = format_dialogue_into_prompt(pair.chosen)
    chosen_proba = get_positive_class_proba(model_id, formatted_chosen_prompt)
    formatted_rejected_prompt = format_dialogue_into_prompt(pair.rejected)
    rejected_proba = get_positive_class_proba(model_id, formatted_rejected_prompt)
    return chosen_proba > rejected_proba


def main(limit: int, model_id: str, openai_api_key: str):
    set_openai_key(openai_api_key)
    # Load the test datasets
    all_test = get_harmless_helpful_test()
    sample = all_test.shuffle().take(limit)
    # Get the predictions
    predictions = sample.map(lambda pair: get_pair_predicted_chosen(model_id, pair))
    # Calculate the accuracy
    accuracy: Optional[float] = predictions.average()
    print(f"Accuracy: {accuracy} for {model_id} on {limit} samples")


if __name__ == "__main__":
    # Model on 80k samples babbage:ft-leadiq:assistant-reward-model-2022-12-20-09-34-26 0.67
    # Model on 10k samples babbage:ft-leadiq:assistant-reward-model-2022-12-19-15-51-58 0.6
    main(
        limit=100,
        model_id="babbage:ft-leadiq:assistant-reward-model-2022-12-19-15-51-58",
        openai_api_key=OPENAI_KEY,
    )
