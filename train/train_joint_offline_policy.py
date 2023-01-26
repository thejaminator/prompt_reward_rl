from concurrent.futures import ThreadPoolExecutor
from typing import Type

from neptune.new import Run
from openai.error import RateLimitError, APIConnectionError
from retry import retry
from slist import Slist

from api.logged_fine_tune import (
    logged_fine_tune,
    AlwaysContinueHandler,
    DefaultCLIHandler,
)
from api.openai_fine_tune import ModelId, FineTuneParams
from api.prompt_completion import PromptCompletion
from api.redis_cache import redis_cache
from api.classification import (
    format_dialogue_into_reward_prompt,
    get_positive_class_proba,
)
from parsing.parse_raw import AnthropicRawFormat
from settings import (
    OFFLINE_JOINT_POLICY_NEPTUNE_PROJECT, REWARD_NORMALIZER_NEPTUNE_KEY,
)
from train.prompt_formatters.joint_policy_prompt_formatter import (
    JointPolicyPromptFormatter,
    JointRewardAtBottomFormatter,
)
from train.metrics.training_distribution import calculate_training_distribution_statistics, \
    TrainingDistributionStatistic
from train.normalizer.joint_reward_normalizer import (
    JointRewardNormalizer,
    JointStandardScaleNormalizer,
)
from train.rewards import (
    DialogueWithJointReward,
)
from train.separators import END_TOKEN
from train.train_joint_reward_model import get_all_train


def replace_with_normalized(
    dialogue_with_reward: DialogueWithJointReward, normalizer: JointRewardNormalizer
) -> DialogueWithJointReward:
    return DialogueWithJointReward(
        dialogue=dialogue_with_reward.dialogue,
        target_reward=normalizer.normalize_reward(dialogue_with_reward.target_reward),
    )


@redis_cache(decode_dict=DialogueWithJointReward)
@retry(exceptions=(RateLimitError, APIConnectionError), tries=5, delay=20)
def get_offline_reward(
    dialogue: str,
    joint_reward_model: ModelId,
) -> DialogueWithJointReward:
    # Get the reward for the dialogue
    formatted = format_dialogue_into_reward_prompt(dialogue)
    joint_reward = get_positive_class_proba(
        model_id=joint_reward_model, prompt=formatted
    )
    # Return the dialogue with the reward
    return DialogueWithJointReward(
        dialogue=dialogue,
        target_reward=joint_reward,
    )


def train(
    policy_formatter: JointPolicyPromptFormatter,
    pair_limit: int,
    finetune_params: FineTuneParams,
    chunks: int,
    normalizer_type: Type[JointRewardNormalizer],
) -> ModelId:
    # Get the prompts
    raw_prompts: Slist[AnthropicRawFormat] = (
        get_all_train().shuffle(seed="999").take(pair_limit)
    )
    reward_model: ModelId = ModelId(
        "babbage:ft-leadiq:assistant-reward-model-2022-12-20-09-34-26"
    )
    thread_pool = ThreadPoolExecutor(max_workers=20)
    counter = 0

    def get_rewards_and_count(
        raw: AnthropicRawFormat,
    ) -> Slist[DialogueWithJointReward]:
        nonlocal counter
        counter += 1
        if counter % 200 == 0:
            print(f"Processed {counter} pairs")
        return Slist(
            # Get rewards for both chosen and rejected
            [
                get_offline_reward(
                    dialogue=raw.chosen, joint_reward_model=reward_model
                ),
                get_offline_reward(
                    dialogue=raw.rejected,
                    joint_reward_model=reward_model,
                ),
            ]
        )

    # Get the rewards for the prompts
    prompt_with_rewards: Slist[DialogueWithJointReward] = raw_prompts.par_map(
        get_rewards_and_count,
        executor=thread_pool,
    ).flatten_list()

    reward_distribution_distribution: TrainingDistributionStatistic = calculate_training_distribution_statistics(
        prompt_with_rewards.map(lambda x: x.target_reward)
    )
    print(f"Reward distribution: {reward_distribution_distribution}")

    # Create the normalizer
    normalizer: JointRewardNormalizer = normalizer_type.from_rewards(
        prompt_with_rewards.map(lambda x: x.target_reward)
    )
    # replace_with_normalized
    normalized_rewards: Slist[DialogueWithJointReward] = prompt_with_rewards.map(
        lambda x: replace_with_normalized(dialogue_with_reward=x, normalizer=normalizer)
    )

    # Convert to prompt completions
    prompt_completions: Slist[PromptCompletion] = normalized_rewards.map(
        lambda x: policy_formatter.dialogue_reward_to_prompt_completion(
            x
        ).to_prompt_completion()
    )

    # Split the prompts into chunks of 25k examples
    # This gets around the limitation that OpenAI doesn't save snapshots of your model
    training_chunks: Slist[Slist[PromptCompletion]] = prompt_completions.grouped(chunks)
    updated_fine_tune_params: FineTuneParams = finetune_params.copy()
    print(normalizer.to_dict())
    for idx, chunk in training_chunks.enumerated():

        def neptune_pretrain_callable(run: Run) -> None:
            run["policy_formatter"] = policy_formatter.name
            run["train/total_train_examples"] = len(prompt_completions)
            run["train/chunk_number"] = idx + 1
            run["train/reward_model"] = reward_model
            run[REWARD_NORMALIZER_NEPTUNE_KEY] = normalizer.to_dict()
            run["train/reward_distribution"] = reward_distribution_distribution.dict()


        if idx > 0:
            updated_fine_tune_params.learning_rate_multiplier = (
                finetune_params.learning_rate_multiplier
            )
        print(f"Training chunk {idx + 1} of {len(training_chunks)}")
        new_model_id = logged_fine_tune(
            train=chunk,
            params=updated_fine_tune_params,
            project_name=OFFLINE_JOINT_POLICY_NEPTUNE_PROJECT,
            completion_start_token="",
            completion_end_token=END_TOKEN,
            neptune_pretrain_callable=neptune_pretrain_callable,
            should_continue_handler=AlwaysContinueHandler()
            if idx >= 1
            else DefaultCLIHandler(),
        )
        # after training, update updated_fine_tune_params
        updated_fine_tune_params.model = new_model_id
    # Return the final model id
    return ModelId(updated_fine_tune_params.model)


if __name__ == "__main__":
    policy_formatter = JointRewardAtBottomFormatter()
    finetune_params = FineTuneParams(
        model="babbage:ft-leadiq:thejaminator-offline-joint-policy-2023-01-20-06-03-15",
        n_epochs=1,
        learning_rate_multiplier=0.05,
        batch_size=128,
        prompt_loss_weight=0.1,
    )
    # Run the main function
    # Try 1000, 10000, 25000, 50000, 75000
    normalizer_type = JointStandardScaleNormalizer
    train(
        policy_formatter,
        pair_limit=75000,
        finetune_params=finetune_params,
        chunks=999999,
        normalizer_type=normalizer_type,
    )
    # export PYTHONPATH=.; python train/train_joint_offline.py
    """
    For joint model harmless + helpful
    Reward distribution: min=0.09957142308017183 max=0.945666965807259 mean=0.5001174086891287 median=0.4930572037643971 std=0.11485019425837607 one_percentile=0.23494138397650455 five_percentile=0.31770856940167097 twenty_five_percentile=0.4275376902366216 seventy_five_percentile=0.5685117348300421 ninety_five_percentile=0.7037077971606336 ninety_nine_percentile=0.8031785942122077 count=150000
    
    For single helpful
    min=0.19104335545627255 max=0.8914878180557255 mean=0.5126062869094005 median=0.5050833998426019 std=0.10006006577921583 one_percentile=0.30150982711480223 five_percentile=0.3602006440724758 twenty_five_percentile=0.4417482465071431 seventy_five_percentile=0.5765196231041 ninety_five_percentile=0.6935718686438797 ninety_nine_percentile=0.762532192437787 count=20000
    """

