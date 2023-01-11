from concurrent.futures import ThreadPoolExecutor

from neptune.new import Run
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
from settings import OFFLINE_SEPARATE_POLICY_NEPTUNE_PROJECT
from train.policy_prompt_formatter import (
    PolicyPromptFormatter,
    RewardAtBottomFormatter, RewardAtTopFormatter, DuplicateRewardAtBottomFormatter,
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


def train(
    policy_formatter: PolicyPromptFormatter,
    pair_limit: int,
    finetune_params: FineTuneParams,
) -> ModelId:
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
    counter = 0

    def get_rewards_and_count(raw: AnthropicRawFormat) -> Slist[DialogueWithReward]:
        nonlocal counter
        counter += 1
        if counter % 200 == 0:
            print(f"Processed {counter} pairs")
        return Slist(
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
        )

    # Get the rewards for the prompts
    prompt_with_rewards: Slist[DialogueWithReward] = raw_prompts.par_map(
        get_rewards_and_count,
        executor=thread_pool,
    ).flatten_list()
    # Convert to prompt completions
    prompt_completions: Slist[PromptCompletion] = prompt_with_rewards.map(
        lambda x: policy_formatter.dialogue_reward_to_prompt_completion(
            x
        ).to_prompt_completion()
    )
    # add formatter
    def neptune_pretrain_callable(run: Run) -> None:
        run["policy_formatter"] = policy_formatter.name
    model_id = logged_fine_tune(
        train=prompt_completions,
        params=finetune_params,
        project_name=OFFLINE_SEPARATE_POLICY_NEPTUNE_PROJECT,
        completion_start_token="",
        completion_end_token=END_TOKEN,
        neptune_pretrain_callable=neptune_pretrain_callable,
    )
    return model_id


if __name__ == "__main__":
    policy_formatter = RewardAtBottomFormatter()
    # policy_formatter = DuplicateRewardAtBottomFormatter()
    finetune_params = FineTuneParams(
        model="babbage",
        n_epochs=1,
        learning_rate_multiplier=0.1,
        batch_size=32,
        prompt_loss_weight=0.1,
    )
    # Run the main function
    # Try 1000, 10000, 25000, 50000, 75000
    train(policy_formatter, pair_limit=75000, finetune_params=finetune_params)
    # export PYTHONPATH=.; python train/train_offline.py
