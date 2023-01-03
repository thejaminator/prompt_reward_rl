from concurrent.futures import ThreadPoolExecutor

from neptune.new import Run
from openai.error import RateLimitError, APIConnectionError
from retry import retry
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
from train.joint_policy_prompt_formatter import (
    JointPolicyPromptFormatter,
    JointRewardAtBottomFormatter,
)
from train.policy_prompt_formatter import (
    PolicyPromptFormatter,
    RewardAtBottomFormatter,
)
from train.reward_models import (
    DialogueWithReward,
    DialogueWithJointReward,
)
from train.separators import END_TOKEN
from train.train_joint_reward_model import get_harmless_helpful_train


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
) -> ModelId:
    # Get the prompts
    raw_prompts: Slist[AnthropicRawFormat] = (
        get_harmless_helpful_train().shuffle(seed="999").take(pair_limit)
    )
    joint_model: ModelId = ModelId(
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
                get_offline_reward(dialogue=raw.chosen, joint_reward_model=joint_model),
                get_offline_reward(
                    dialogue=raw.rejected,
                    joint_reward_model=joint_model,
                ),
            ]
        )

    # Get the rewards for the prompts
    prompt_with_rewards: Slist[DialogueWithJointReward] = raw_prompts.par_map(
        get_rewards_and_count,
        executor=thread_pool,
    ).flatten_list()
    # Convert to prompt completions
    prompt_completions: Slist[PromptCompletion] = prompt_with_rewards.map(
        lambda x: policy_formatter.dialogue_reward_to_prompt_completion(
            x
        ).to_prompt_completion()
    )

    # Split the prompts into chunks of 25k examples
    # This gets around the limitation that OpenAI doesn't save snapshots of your model
    training_chunks: Slist[Slist[PromptCompletion]] = prompt_completions.grouped(25000)
    updated_fine_tune_params: FineTuneParams = finetune_params.copy()
    for idx, chunk in training_chunks.enumerated():

        def neptune_pretrain_callable(run: Run) -> None:
            run["policy_formatter"] = policy_formatter.name
            run["train/total_train_examples"] = len(prompt_completions)
            run["train/chunk_number"] = idx + 1

        # for idx greater than 0, we need to set a lower learning rate ( half )
        if idx > 0:
            updated_fine_tune_params.learning_rate_multiplier = (
                finetune_params.learning_rate_multiplier / 2
            )
        print(f"Training chunk {idx + 1} of {len(training_chunks)}")
        new_model_id = logged_fine_tune(
            train=chunk,
            params=updated_fine_tune_params,
            project_name=OFFLINE_SEPARATE_POLICY_NEPTUNE_PROJECT,
            completion_start_token="",
            completion_end_token=END_TOKEN,
            neptune_pretrain_callable=neptune_pretrain_callable,
        )
        # after training, update updated_fine_tune_params
        updated_fine_tune_params.model = new_model_id
    # Return the final model id
    return ModelId(updated_fine_tune_params.model)


if __name__ == "__main__":
    policy_formatter = JointRewardAtBottomFormatter()
    finetune_params = FineTuneParams(
        model="babbage",
        n_epochs=1,
        learning_rate_multiplier=0.1,
        batch_size=128,
        prompt_loss_weight=0.1,
    )
    # Run the main function
    # Try 1000, 10000, 25000, 50000, 75000
    train(policy_formatter, pair_limit=75000, finetune_params=finetune_params)
    # export PYTHONPATH=.; python train/train_joint_offline.py
