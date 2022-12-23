from slist import Slist

from api.dataset_paths import (
    anthropic_harmless_train_path,
    anthropic_helpful_train_path,
)
from api.logged_fine_tune import logged_fine_tune
from api.openai_fine_tune import FineTuneParams
from api.prompt_completion import PromptCompletion
from evaluate.classification import format_dialogue_into_reward_prompt
from parsing.parse_raw import AnthropicRawFormat, get_raw_anthropic
from train.separators import POSITIVE_TOKEN, NEGATIVE_TOKEN


def get_harmless_helpful_train() -> Slist[AnthropicRawFormat]:
    harmless_train: Slist[AnthropicRawFormat] = get_raw_anthropic(
        anthropic_harmless_train_path
    )
    print(f"Loaded {len(harmless_train)} harmless train examples")
    helpful_train: Slist[AnthropicRawFormat] = get_raw_anthropic(
        anthropic_helpful_train_path
    )
    print(f"Loaded {len(helpful_train)} helpful train examples")
    return harmless_train + helpful_train


def format_raw_into_prompt_completion(
    raw: AnthropicRawFormat,
) -> Slist[PromptCompletion]:
    # the positive class is a good completion. aka chosen
    # the negative class is a bad completion. aka rejected
    positive_example = raw.chosen
    positive_token = POSITIVE_TOKEN
    negative_example = raw.rejected
    negative_token = NEGATIVE_TOKEN

    positive_prompt_completion = PromptCompletion(
        prompt=format_dialogue_into_reward_prompt(positive_example),
        completion=positive_token,
    )
    negative_prompt_completion = PromptCompletion(
        prompt=format_dialogue_into_reward_prompt(negative_example),
        completion=negative_token,
    )
    return Slist([positive_prompt_completion, negative_prompt_completion])


def main():
    limit = 100000
    harmless_helpful_train = get_harmless_helpful_train()
    print(f"Loaded {len(harmless_helpful_train)} harmless/helpful train examples")
    training_pairs: Slist[Slist[PromptCompletion]] = harmless_helpful_train.map(
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
        project_name="leadiq/assistant-reward-model",
        completion_start_token="",
        # no end token, we'll just call it using 1 token response length
        completion_end_token="",
    )


if __name__ == "__main__":
    main()
