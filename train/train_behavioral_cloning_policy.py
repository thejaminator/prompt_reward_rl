from neptune.new import Run
from slist import Slist

from api.logged_fine_tune import (
    logged_fine_tune,
    AlwaysContinueHandler,
    DefaultCLIHandler,
)
from api.openai_fine_tune import ModelId, FineTuneParams
from api.prompt_completion import PromptCompletion
from parsing.parse_raw import AnthropicRawFormat
from settings import (
    BEHAVIORAL_CLONING_NEPTUNE_PROJECT,
)
from train.prompt_formatters.policy_prompt_formatter import (
    NoRewardFormatter,
)
from train.rewards import HelpfulHarmlessReward, DialogueWithReward
from train.separators import END_TOKEN
from train.train_helpful_reward_model import get_helpful_train


def train(
    pair_limit: int,
    finetune_params: FineTuneParams,
) -> ModelId:
    policy_formatter = NoRewardFormatter()
    # Get the prompts
    raw_prompts: Slist[AnthropicRawFormat] = (
        get_helpful_train().shuffle(seed="999").take(pair_limit)
    )
    # Convert to prompt completions
    prompt_completions: Slist[PromptCompletion] = raw_prompts.map(
        lambda raw: raw.chosen
    ).map(
        lambda chosen: policy_formatter.dialogue_reward_to_prompt_completion(
            with_reward=DialogueWithReward(
                dialogue=chosen,
                # these rewards aren't actually used since its a NoRewardFormatter
                target_reward=HelpfulHarmlessReward(helpful=0.00, harmless=0.00),
            )
        ).to_prompt_completion()
    )

    # Split the prompts into chunks of 25k examples
    # This gets around the limitation that OpenAI doesn't save snapshots of your model
    training_chunks: Slist[Slist[PromptCompletion]] = prompt_completions.grouped(100000)
    updated_fine_tune_params: FineTuneParams = finetune_params.copy()
    for idx, chunk in training_chunks.enumerated():

        def neptune_pretrain_callable(run: Run) -> None:
            run["policy_formatter"] = policy_formatter.name
            run["train/total_train_examples"] = len(prompt_completions)
            run["train/chunk_number"] = idx + 1
            run["train/policy_type"] = "behavioral_cloning"

        if idx > 0:
            updated_fine_tune_params.learning_rate_multiplier = (
                finetune_params.learning_rate_multiplier
            )
        print(f"Training chunk {idx + 1} of {len(training_chunks)}")
        new_model_id = logged_fine_tune(
            train=chunk,
            params=updated_fine_tune_params,
            project_name=BEHAVIORAL_CLONING_NEPTUNE_PROJECT,
            completion_start_token="",
            completion_end_token=END_TOKEN,
            neptune_pretrain_callable=neptune_pretrain_callable,
            # Ask to continue for the first chunk, but not the rest
            should_continue_handler=AlwaysContinueHandler()
            if idx > 0
            else DefaultCLIHandler(),
        )
        # after training, update updated_fine_tune_params
        updated_fine_tune_params.model = new_model_id
    # Return the final model id
    return ModelId(updated_fine_tune_params.model)


if __name__ == "__main__":
    # TODO: BC  on only helpful?
    finetune_params = FineTuneParams(
        model="babbage",
        n_epochs=2,
        learning_rate_multiplier=0.1,
        batch_size=32,
        prompt_loss_weight=0.0,
    )
    # Run the main function
    # Try 1000, 10000, 25000, 50000, 75000
    train(pair_limit=75000, finetune_params=finetune_params)
    # export PYTHONPATH=.; python train/train_offline.py
