from concurrent.futures import ThreadPoolExecutor
from typing import Type

from neptune.new import Run
from pydantic import BaseModel
from slist import Slist, identity

from api.logged_fine_tune import (
    logged_fine_tune,
    AlwaysContinueHandler,
    DefaultCLIHandler,
)
from api.openai_fine_tune import ModelId, FineTuneParams
from api.prompt_completion import PromptCompletion
from api.redis_cache import redis_cache
from evaluate.classification import (
    format_dialogue_into_reward_prompt,
    get_positive_class_proba,
)
from parsing.parse_raw import AnthropicRawFormat
from settings import OFFLINE_SEPARATE_POLICY_NEPTUNE_PROJECT, REWARD_NORMALIZER_NEPTUNE_KEY
from train.policy_prompt_formatter import (
    PolicyPromptFormatter,
    RewardAtBottomFormatter,
    RewardAtTopFormatter,
    DuplicateRewardAtBottomFormatter,
    RewardAtBottomTimes100Formatter,
)
from train.reward_models import HelpfulHarmlessReward, DialogueWithReward
from train.reward_normalizer import RewardNormalizer, StandardScaleNormalizer
from train.separators import END_TOKEN
from train.train_joint_reward_model import get_harmless_helpful_train
from retry import retry
from openai.error import RateLimitError


def replace_with_normalized(
        dialogue_with_reward: DialogueWithReward, normalizer: RewardNormalizer
) -> DialogueWithReward:
    return DialogueWithReward(
        dialogue=dialogue_with_reward.dialogue,
        target_reward=normalizer.normalize_reward(
            dialogue_with_reward.target_reward
        ),
    )

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
    chunk_size: int,
    normalizer_type: Type[RewardNormalizer],
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
    # Print maximum and minimum rewards
    print(
        f"Maximum helpful reward: {prompt_with_rewards.map(lambda x: x.target_reward.helpful).sort_by(identity, reverse=True).first_or_raise()}"
    )
    print(
        f"Minimum helpful reward: {prompt_with_rewards.map(lambda x: x.target_reward.helpful).sort_by(identity).first_or_raise()}"
    )
    # 95th percentile
    ninety_fifth_percentile_helpful = (
        prompt_with_rewards.map(lambda x: x.target_reward.helpful)
        .sort_by(identity)
        .take(int(len(prompt_with_rewards) * 0.95))
        .last_or_raise()
    )
    print(f"95th percentile helpful reward: {ninety_fifth_percentile_helpful}")
    # 5th percentile
    fifth_percentile_helpful = (
        prompt_with_rewards.map(lambda x: x.target_reward.helpful)
        .sort_by(identity)
        .take(int(len(prompt_with_rewards) * 0.05))
        .last_or_raise()
    )
    print(f"5th percentile helpful reward: {fifth_percentile_helpful}")

    print(
        f"Maximum harmless reward: {prompt_with_rewards.map(lambda x: x.target_reward.harmless).sort_by(identity, reverse=True).first_or_raise()}"
    )
    print(
        f"Minimum harmless reward: {prompt_with_rewards.map(lambda x: x.target_reward.harmless).sort_by(identity).first_or_raise()}"
    )

    # Create normalizer
    normalizer: RewardNormalizer = normalizer_type.from_rewards(
        rewards=prompt_with_rewards.map(lambda x: x.target_reward)
    )



    # Normalize the rewards
    prompt_with_normalized_rewards: Slist[DialogueWithReward] = prompt_with_rewards.map(
        lambda x: replace_with_normalized(x, normalizer=normalizer)
    )

    # Convert to prompt completions
    prompt_completions: Slist[PromptCompletion] = prompt_with_normalized_rewards.map(
        lambda x: policy_formatter.dialogue_reward_to_prompt_completion(
            x
        ).to_prompt_completion()
    )
    # add formatter
    # Split the prompts into chunks of 25k examples
    # This gets around the limitation that OpenAI doesn't save snapshots of your model
    training_chunks: Slist[Slist[PromptCompletion]] = prompt_completions.grouped(
        chunk_size
    )
    updated_fine_tune_params: FineTuneParams = finetune_params.copy()
    print(f"Normalizer: {normalizer.to_dict()}")
    for idx, chunk in training_chunks.enumerated():

        def neptune_pretrain_callable(run: Run) -> None:
            run["policy_formatter"] = policy_formatter.name
            run["train/total_train_examples"] = len(prompt_completions)
            run["train/chunk_number"] = idx + 1
            run["train/helpful_model"] = helpful_model
            run["train/harmless_model"] = harmless_model
            run[REWARD_NORMALIZER_NEPTUNE_KEY] = normalizer.to_dict()

        if idx > 0:
            updated_fine_tune_params.learning_rate_multiplier = (
                finetune_params.learning_rate_multiplier
            ) * 1
        print(f"Training chunk {idx + 1} of {len(training_chunks)}")
        new_model_id = logged_fine_tune(
            train=chunk,
            params=updated_fine_tune_params,
            project_name=OFFLINE_SEPARATE_POLICY_NEPTUNE_PROJECT,
            completion_start_token="",
            completion_end_token=END_TOKEN,
            neptune_pretrain_callable=neptune_pretrain_callable,
            should_continue_handler=AlwaysContinueHandler()
            if idx > 0
            else DefaultCLIHandler(),
        )
        # after training, update updated_fine_tune_params
        updated_fine_tune_params.model = new_model_id
    return ModelId(updated_fine_tune_params.model)


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
    normalizer: Type[StandardScaleNormalizer] = StandardScaleNormalizer
    # Run the main function
    # Try 1000, 10000, 25000, 50000, 75000
    train(
        policy_formatter,
        pair_limit=75000,
        finetune_params=finetune_params,
        chunk_size=99999999,
        normalizer_type=normalizer,
    )
    # export PYTHONPATH=.; python train/train_offline.py
