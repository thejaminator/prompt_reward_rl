import os

from pydantic import BaseModel
from slist import Slist

from api.logged_fine_tune import logged_fine_tune
from api.openai_fine_tune import ModelId, FineTuneParams
from api.prompt_completion import PromptCompletion
from calculate_reward import AnthropicRawFormat
from evaluate.classification import (
    format_conversation_into_reward_prompt,
    get_positive_class_proba,
)
from evaluate.inference import OpenaiInferenceConfig
from train.reward_formatter import RewardFormatter
from train.reward_models import HelpfulHarmlessReward
from train.separators import END_TOKEN
from train.train_joint_reward_model import get_harmless_helpful_train


def get_offline_rollouts() -> Slist[AnthropicRawFormat]:
    # Get the already generated conversations
    return get_harmless_helpful_train()


class ConversationWithReward(BaseModel):
    conversation: str
    reward: HelpfulHarmlessReward


def get_offline_reward(
    conversation: str,
    helpful_model: ModelId,
    harmless_model: ModelId,
) -> ConversationWithReward:
    # Get the reward for the conversation
    formatted = format_conversation_into_reward_prompt(conversation)
    helpful_reward = get_positive_class_proba(model_id=helpful_model, prompt=formatted)
    harmless_reward = get_positive_class_proba(
        model_id=harmless_model, prompt=formatted
    )
    # Return the conversation with the reward
    return ConversationWithReward(
        conversation=conversation,
        reward=HelpfulHarmlessReward(
            helpful=helpful_reward,
            harmless=harmless_reward,
        ),
    )


def main(reward_formatter: RewardFormatter) -> None:
    # Get the prompts
    raw_prompts: Slist[AnthropicRawFormat] = get_offline_rollouts()
    # Get the helpful and harmless models
    helpful_model: ModelId = ModelId(
        "babbage:ft-leadiq:helpful-reward-2022-12-22-08-04-46"
    )
    harmless_model: ModelId = ModelId(
        "babbage:ft-leadiq:harmless-reward-2022-12-22-08-55-12"
    )
    # Get the rewards for the prompts
    prompt_with_rewards: Slist[ConversationWithReward] = raw_prompts.map(
        lambda raw: Slist(
            # Get rewards for both chosen and rejected
            [
                get_offline_reward(
                    conversation=raw.chosen,
                    helpful_model=helpful_model,
                    harmless_model=harmless_model,
                ),
                get_offline_reward(
                    conversation=raw.rejected,
                    helpful_model=helpful_model,
                    harmless_model=harmless_model,
                ),
            ]
        )
    ).flatten_list()
    # Convert to prompt completions
    prompt_completions: Slist[PromptCompletion] = prompt_with_rewards.map(
        reward_formatter.convo_reward_to_prompt_completion
    )
    finetune_params = FineTuneParams(
        model="babbage",
        n_epochs=1,
        learning_rate_multiplier=0.1,
        batch_size=32,
        prompt_loss_weight=0.1,
    )
    logged_fine_tune(
        train=prompt_completions,
        params=finetune_params,
        project_name="thejaminator/offline-assistant-policy",
        completion_start_token="",
        completion_end_token=END_TOKEN,
    )
