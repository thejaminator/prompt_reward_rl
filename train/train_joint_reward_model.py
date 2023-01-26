from neptune.new import Run
from slist import Slist

from api.dataset_paths import (
    anthropic_harmless_train_path,
    anthropic_helpful_train_path,
    anthropic_online_train_path,
    anthropic_rejection_sampled_train_path,
)
from api.logged_fine_tune import (
    logged_fine_tune,
    DefaultCLIHandler,
    AlwaysContinueHandler,
)
from api.openai_fine_tune import FineTuneParams, ModelId
from api.prompt_completion import PromptCompletion
from evaluate.classification import format_dialogue_into_reward_prompt
from parsing.parse_raw import AnthropicRawFormat, get_raw_anthropic
from train.separators import POSITIVE_TOKEN, NEGATIVE_TOKEN


def get_harmless_helpful_train() -> Slist[AnthropicRawFormat]:
    harmless_train: Slist[AnthropicRawFormat] = get_raw_anthropic(
        anthropic_harmless_train_path
    )
    print(f"Loaded {len(harmless_train)} harmless train pairs")
    helpful_train: Slist[AnthropicRawFormat] = get_raw_anthropic(
        anthropic_helpful_train_path
    )
    print(f"Loaded {len(helpful_train)} helpful train pairs")
    return harmless_train + helpful_train


def get_online_and_rejection_sampling_train() -> Slist[AnthropicRawFormat]:
    online_train: Slist[AnthropicRawFormat] = get_raw_anthropic(
        anthropic_online_train_path
    )
    print(f"Loaded {len(online_train)} online train pairs")
    rejected_sampled_train: Slist[AnthropicRawFormat] = get_raw_anthropic(
        anthropic_rejection_sampled_train_path
    )
    print(f"Loaded {len(rejected_sampled_train)} rejected sampled train pairs")
    return online_train + rejected_sampled_train


def get_all_train() -> Slist[AnthropicRawFormat]:
    return get_harmless_helpful_train() + get_online_and_rejection_sampling_train()


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
    limit = 99999999999
    harmless_helpful_train = get_harmless_helpful_train()
    online_and_rejection_sampling_train = get_online_and_rejection_sampling_train()
    all_train = harmless_helpful_train + online_and_rejection_sampling_train
    print(f"Loaded {len(all_train)} train examples")
    training_pairs: Slist[Slist[PromptCompletion]] = all_train.map(
        format_raw_into_prompt_completion
    )
    print(f"Created {len(training_pairs)} training_pairs")
    # Apply the limit here
    limited_pairs: Slist[PromptCompletion] = (
        training_pairs.shuffle().take(limit).flatten_list()
    )
    print(f"Shuffled and limited. We now have {len(limited_pairs)} training examples")

    finetune_params = FineTuneParams(
        model="babbage",
        n_epochs=1,
        learning_rate_multiplier=0.1,
        batch_size=32,
        prompt_loss_weight=0.1,
    )
    training_chunks: Slist[Slist[PromptCompletion]] = limited_pairs.split_into_n(1)
    updated_fine_tune_params: FineTuneParams = finetune_params.copy()
    for idx, chunk in training_chunks.enumerated():
        if idx >= 1:
            # make learning rate half for second and more chunks
            updated_fine_tune_params.learning_rate_multiplier = (
                finetune_params.learning_rate_multiplier / 2
            )

        def pre_train_log(run: Run) -> None:
            run["train/total_train_examples"] = len(limited_pairs)
            run["train/chunk_number"] = idx + 1

        new_model_id: ModelId = logged_fine_tune(
            train=chunk,
            params=updated_fine_tune_params,
            project_name="leadiq/assistant-reward-model",
            completion_start_token="",
            # no end token, we'll just call it using 1 token response length
            completion_end_token="",
            neptune_pretrain_callable=pre_train_log,
            should_continue_handler=AlwaysContinueHandler()
            if idx >= 1
            else DefaultCLIHandler(),
        )
        print(f"Finished chunk {idx + 1} with model id {new_model_id}")
        # update with new model id
        updated_fine_tune_params.model = new_model_id


if __name__ == "__main__":
    main()


"""
For harmless+ helpful, we get:
Reward distribution: min=0.09957142308017183 max=0.945666965807259 mean=0.5001174086891287 median=0.4930572037643971 std=0.11485019425837607 five_percentile=0.31770856940167097 twenty_five_percentile=0.4275376902366216 seventy_five_percentile=0.5685117348300421 ninety_five_percentile=0.7037077971606336 count=150000
"""
