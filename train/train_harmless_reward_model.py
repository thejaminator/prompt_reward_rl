from slist import Slist

from api.dataset_paths import (
    anthropic_harmless_train_path,
)
from api.logged_fine_tune import logged_fine_tune
from api.openai_fine_tune import FineTuneParams
from api.prompt_completion import PromptCompletion
from parsing.parse_raw import AnthropicRawFormat, get_raw_anthropic
from train.train_joint_reward_model import (
    format_raw_into_prompt_completion,
)


def get_harmless_train() -> Slist[AnthropicRawFormat]:
    harmless_train: Slist[AnthropicRawFormat] = get_raw_anthropic(
        anthropic_harmless_train_path
    )
    print(f"Loaded {len(harmless_train)} harmless train examples")
    return harmless_train


def main():
    limit = 100000
    harmless_train = get_harmless_train()
    print(f"Loaded {len(harmless_train)} harmless train examples")
    training_pairs: Slist[Slist[PromptCompletion]] = harmless_train.map(
        format_raw_into_prompt_completion
    )
    print(f"Created {len(training_pairs)} training_pairs")
    # Apply the limit here
    limited_pairs: Slist[Slist[PromptCompletion]] = training_pairs.shuffle().take(limit)
    print(f"Shuffled and limited. We now have {len(limited_pairs)} training pairs")

    finetune_params = FineTuneParams(
        model="babbage",
        n_epochs=1,
        learning_rate_multiplier=0.1,
        batch_size=32,
        prompt_loss_weight=0.1,
    )
    logged_fine_tune(
        train=limited_pairs.flatten_list(),
        params=finetune_params,
        project_name="harmless-reward",
        completion_start_token="",
        # no end token, we'll just call it using 1 token response length
        completion_end_token="",
    )


if __name__ == "__main__":
    main()
