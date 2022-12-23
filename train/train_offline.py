from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel
from slist import Slist

from api.logged_fine_tune import logged_fine_tune
from api.openai_fine_tune import ModelId, FineTuneParams
from api.prompt_completion import PromptCompletion
from api.redis_cache import redis_cache
from evaluate.classification import (
    format_dialogue_into_reward_prompt,
    get_positive_class_proba,
)
from parsing.parse_raw import AnthropicRawFormat
from train.policy_prompt_formatter import (
    PolicyPromptFormatter,
    PolicyRewardAtBottomFormatter,
)
from train.reward_models import HelpfulHarmlessReward, DialogueWithReward
from train.separators import END_TOKEN
from train.train_joint_reward_model import get_harmless_helpful_train
from retry import retry
from openai.error import RateLimitError


@redis_cache(decode_dict=DialogueWithReward)
@retry(exceptions=RateLimitError, tries=5, delay=20)
def get_offline_reward(
    dialogue: str,
    helpful_model: ModelId,
    harmless_model: ModelId,
) -> DialogueWithReward:
    # Get the reward for the dialogue
    formatted = format_dialogue_into_reward_prompt(dialogue)
    helpful_reward = get_positive_class_proba(model_id=helpful_model, prompt=formatted)
    harmless_reward = get_positive_class_proba(
        model_id=harmless_model, prompt=formatted
    )
    # Return the dialogue with the reward
    return DialogueWithReward(
        dialogue=dialogue,
        target_reward=HelpfulHarmlessReward(
            helpful=helpful_reward,
            harmless=harmless_reward,
        ),
    )


def main(policy_formatter: PolicyPromptFormatter, pair_limit: int) -> None:
    # Get the prompts
    raw_prompts: Slist[AnthropicRawFormat] = (
        get_harmless_helpful_train().shuffle(seed="999").take(pair_limit)
    )
    # Get the helpful and harmless models
    helpful_model: ModelId = ModelId(
        "babbage:ft-leadiq:helpful-reward-2022-12-22-08-04-46"
    )
    harmless_model: ModelId = ModelId(
        "babbage:ft-leadiq:harmless-reward-2022-12-22-08-55-12"
    )
    thread_pool = ThreadPoolExecutor(max_workers=20)
    # Get the rewards for the prompts
    prompt_with_rewards: Slist[DialogueWithReward] = raw_prompts.par_map(
        lambda raw: Slist(
            # Get rewards for both chosen and rejected
            [
                get_offline_reward(
                    dialogue=raw.chosen,
                    helpful_model=helpful_model,
                    harmless_model=harmless_model,
                ),
                get_offline_reward(
                    dialogue=raw.rejected,
                    helpful_model=helpful_model,
                    harmless_model=harmless_model,
                ),
            ]
        ),
        executor=thread_pool,
    ).flatten_list()
    # Convert to prompt completions
    prompt_completions: Slist[PromptCompletion] = prompt_with_rewards.map(
        lambda x: policy_formatter.dialogue_reward_to_prompt_completion(
            x
        ).to_prompt_completion()
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


if __name__ == "__main__":
    policy_formatter = PolicyRewardAtBottomFormatter()
    # Run the main function
    main(policy_formatter, pair_limit=10000)
