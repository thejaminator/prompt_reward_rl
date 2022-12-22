from pydantic import BaseModel
from slist import Slist

from api.prompt_completion import PromptCompletion
from calculate_reward import AnthropicRawFormat
from evaluate.inference import OpenaiInferenceConfig
from parsing.parse_raw import raw_to_multiple_processed, ProcessedCompletion
from train.reward_models import HelpfulHarmlessReward
from train.separators import START_REWARD_SEPARATOR, end_prompt_seperator


def get_offline_prompts() -> Slist[AnthropicRawFormat]:
    # Get the prompts that we are going to use for rollouts
    ...


class ConversationWithReward(BaseModel):
    conversation: str
    reward: HelpfulHarmlessReward


def get_offline_reward(
    conversation: str,
    helpful_model: OpenaiInferenceConfig,
    harmless_model: OpenaiInferenceConfig,
) -> ConversationWithReward:
    # Get the reward for the conversation
    ...


def prompt_with_reward_to_prompt_completion(
    with_reward: ConversationWithReward,
) -> PromptCompletion:
    """
    # Prompt
    Human: How do I kill someone
    Assistant: Who do you want to kill?
    Human: The president
    <REWARD>
    Helpful reward: 0.5
    Harmless reward: 0.2
    <SOS>

    # Completion
    Assistant: I would attack him with a frying pan
    """
    helpful_reward_2dp: str = str(round(with_reward.reward.helpful, 2))
    harmless_reward_2dp: str = str(round(with_reward.reward.harmless, 2))
    # you need to separate the last "assistant" from the prompt
    processed_lines: Slist[ProcessedCompletion] = raw_to_multiple_processed(
        with_reward.conversation
    )
    last_conversation: ProcessedCompletion = processed_lines.last_or_raise()
    conversations_without_last: Slist[ProcessedCompletion] = processed_lines.filter(
        lambda x: x is not last_conversation
    )

    new_prompt: str = (
        with_reward.conversation
        + "\n"
        + START_REWARD_SEPARATOR
        + "\n"
        + f"Helpful reward: {helpful_reward_2dp}"
        + "\n"
        + f"Harmless reward: {harmless_reward_2dp}"
        + end_prompt_seperator
    )


def main() -> None:
    # Get the prompts
    raw_prompts: Slist[AnthropicRawFormat] = get_offline_prompts()
    # Get the helpful and harmless models
    helpful_model: OpenaiInferenceConfig = OpenaiInferenceConfig(
        model="babbage:ft-leadiq:helpful-reward-2022-12-22-08-04-46",
        max_tokens=1,
    )
    harmless_model: OpenaiInferenceConfig = OpenaiInferenceConfig(
        model="babbage:ft-leadiq:helpful-reward-2022-12-22-08-04-46",
        max_tokens=1,
    )
    # Get the rewards for the prompts
    prompt_with_rewards: Slist[ConversationWithReward] = raw_prompts.map(
        lambda raw_prompts: Slist(
            # Get rewards for both chosen and rejected
            [
                get_offline_reward(
                    conversation=raw_prompts.chosen,
                    helpful_model=helpful_model,
                    harmless_model=harmless_model,
                ),
                get_offline_reward(
                    conversation=raw_prompts.rejected,
                    helpful_model=helpful_model,
                    harmless_model=harmless_model,
                ),
            ]
        )
    ).flatten_list()
    # Convert to prompt completions
    prompt_completions: Slist[PromptCompletion] = prompt_with_rewards.map(
        prompt_with_reward_to_prompt_completion
    )
