from typing import NewType

import numpy as np

from calculate_reward import AnthropicRawFormat
from evaluate.inference import OpenaiInferenceConfig, get_openai_completion, TokenProba
from train.separators import end_prompt_seperator
from train.train_joint_reward_model import POSITIVE_TOKEN, NEGATIVE_TOKEN

# A prompt that has been formatted properly for the reward model
PromptForRewardModel = NewType("PromptForRewardModel", str)


def format_conversation_into_reward_prompt(conversation: str) -> PromptForRewardModel:
    return PromptForRewardModel(conversation.strip() + end_prompt_seperator)


def get_positive_class_proba(model_id: str, prompt: PromptForRewardModel) -> float:
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
    formatted_chosen_prompt = format_conversation_into_reward_prompt(pair.chosen)
    chosen_proba = get_positive_class_proba(model_id, formatted_chosen_prompt)
    formatted_rejected_prompt = format_conversation_into_reward_prompt(pair.rejected)
    rejected_proba = get_positive_class_proba(model_id, formatted_rejected_prompt)
    return chosen_proba > rejected_proba
