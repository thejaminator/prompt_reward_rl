from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Type, Optional

import pandas as pd
from neptune.new import Run
from openai import InvalidRequestError
from pydantic import BaseModel
from slist import Slist
from neptune.new.types import File

from api.cli import cli_input_list
from api.dataset_paths import anthropic_online_train_path
from api.logged_fine_tune import logged_fine_tune, AlwaysContinueHandler
from api.openai_fine_tune import ModelId, FineTuneParams
from api.prompt_completion import PromptCompletion
from api.type_check import should_not_happen
from evaluate.inference import OpenaiInferenceConfig, GPTFullResponse
from parsing.parse_raw import AnthropicRawFormat, get_raw_anthropic
from settings import ONLINE_POLICY_NEPTUNE_PROJECT
from train.assign_rewards import assign_separate_target_reward
from train.evaluate_offline import (
    get_policy_single_evaluation,
    HelpfulHarmlessEvaluationMetric,
    EvaluationWithGPTResponse,
)
from train.policy_prompt_formatter import (
    PolicyPromptFormatter,
    PolicyPromptInfo,
    RewardAtBottomFormatter,
)
from train.reward_models import HelpfulHarmlessReward, DialogueWithReward
from train.reward_normalizer import (
    RewardNormalizer,
    MinMaxNormalizer,
    OnlineTrainingData, DoNothingNormalizer,
)
from train.separators import END_TOKEN


def get_online_prompts() -> Slist[AnthropicRawFormat]:
    # Get the prompts that we are going to use for rollouts
    path = anthropic_online_train_path
    return get_raw_anthropic(path)


def rollout_and_evaluate(
    dialogue: str,
    formatter: PolicyPromptFormatter,
    policy_model: OpenaiInferenceConfig,
    target_reward: HelpfulHarmlessReward,
    helpful_model: ModelId,
    harmless_model: ModelId,
) -> Optional[EvaluationWithGPTResponse]:
    # do the same thing, but using the maximum target reward
    dialogue_with_maximum_reward: DialogueWithReward = assign_separate_target_reward(
        dialogue=dialogue,
        helpful_target=target_reward.helpful,
        harmless_target=target_reward.harmless,
    )
    formatted_maximum_reward: PolicyPromptInfo = (
        formatter.dialogue_reward_to_prompt_completion(dialogue_with_maximum_reward)
    )
    # Rollout the prompt using the policy model, with the target reward
    # This also evaluates the rollout and gets the actual reward
    try:
        rollout: EvaluationWithGPTResponse = get_policy_single_evaluation(
            policy_prompt_info=formatted_maximum_reward,
            policy_model=policy_model,
            helpful_model=helpful_model,
            harmless_model=harmless_model,
            cached=False,  # Don't cache this since we have multiple rollouts, all with the same target temperature
        )
    except InvalidRequestError as e:
        print(f"Invalid request error, probably token limit: {e}")
        return None
    return rollout


def evaluated_rollout_to_prompt_completion(
    evaluated_rollout: EvaluationWithGPTResponse,
    formatter: PolicyPromptFormatter,
    normalizer: RewardNormalizer,
) -> OnlineTrainingData:
    # Replace the policy prompt with the actual reward.
    completed_dialogue: str = evaluated_rollout.metrics.completed_dialogue
    non_normalized_reward: HelpfulHarmlessReward = HelpfulHarmlessReward(
        helpful=evaluated_rollout.metrics.actual_helpful,
        harmless=evaluated_rollout.metrics.actual_harmless,
    )
    normalized_reward: HelpfulHarmlessReward = normalizer.normalize_reward(
        non_normalized_reward
    )
    policy_prompt: PolicyPromptInfo = formatter.dialogue_reward_to_prompt_completion(
        with_reward=DialogueWithReward(
            dialogue=completed_dialogue,
            target_reward=normalized_reward,
        )
    )
    p_c = policy_prompt.to_prompt_completion()
    return OnlineTrainingData(
        rollout_metric=evaluated_rollout.metrics,
        normalized_helpful=normalized_reward.helpful,
        normalized_harmless=normalized_reward.harmless,
        training_prompt=p_c.prompt,
        training_completion=p_c.completion,
    )


class RolloutBufferMetrics(BaseModel):
    rollout_count: int
    prompt_unique_sample_count: int
    rollouts_per_prompt: int
    average_harmless_reward: float
    average_helpful_reward: float
    average_token_entropy: float
    average_token_proba: float
    average_rollout_length: float
    n_iteration: int


def finetune_online_with_neptune(
    project_name: str,
    fine_tune_params: FineTuneParams,
    rollout_buffer_metrics: RolloutBufferMetrics,
    normalizer_type: Type[RewardNormalizer],
    online_training_data: Slist[OnlineTrainingData],
    target_reward: HelpfulHarmlessReward,
) -> ModelId:
    # Fine-tune the model with the prompt completions

    def pre_train_log(run: Run) -> None:
        # Log the rollout metrics to neptune
        for k, v in rollout_buffer_metrics.dict().items():
            run[f"online_metrics/{k}"] = v

        # write the rollouts
        rollouts_df = pd.DataFrame(online_training_data.map(lambda x: x.to_flattened_dict()))
        # save the random rewards dataframe as jsonl
        results_jsonl = rollouts_df.to_json(orient="records", lines=True)
        # write the jsonl to a file
        evaluation_path = "rollouts_table.jsonl"
        with open(evaluation_path, "w") as f:
            f.write(results_jsonl)
        run[f"online_metrics/rollouts_table.jsonl"].upload(evaluation_path)
        # Save as html
        run[f"online_metrics/rollouts_table"].upload(File.as_html(rollouts_df))
        # normalizer name
        run["online_metrics/normalizer_type"] = normalizer_type.name()
        # target rewards
        run["online_metrics/target_helpful"] = target_reward.helpful
        run["online_metrics/target_harmless"] = target_reward.harmless

    return logged_fine_tune(
        train=online_training_data.map(lambda x: x.to_prompt_completion()),
        project_name=project_name,
        completion_start_token="",
        completion_end_token=END_TOKEN,
        params=fine_tune_params,
        neptune_pretrain_callable=pre_train_log,
        should_continue_handler=AlwaysContinueHandler(),
    )


threadpool = ThreadPoolExecutor(max_workers=30)


def single_iteration(
    dialogues: Slist[AnthropicRawFormat],
    target_reward: HelpfulHarmlessReward,
    formatter: PolicyPromptFormatter,
    policy_model: OpenaiInferenceConfig,
    helpful_model: ModelId,
    harmless_model: ModelId,
    project_name: str,
    fine_tune_params: FineTuneParams,
    normalizer_type: Type[RewardNormalizer],
    # for logging
    prompt_unique_sample_count: int,
    rollouts_per_prompt: int,
    n_iteration: int,
) -> ModelId:
    # Rollout and reward the prompts
    rollouts: Slist[EvaluationWithGPTResponse] = dialogues.par_map(
        lambda prompt: rollout_and_evaluate(
            dialogue=prompt.chosen,
            policy_model=policy_model,
            target_reward=target_reward,
            formatter=formatter,
            helpful_model=helpful_model,
            harmless_model=harmless_model,
        ),
        executor=threadpool,
    ).flatten_option()
    # Store additional metrics about the buffer of rollouts
    rollout_buffer_metrics: RolloutBufferMetrics = calculate_rollout_metrics(
        rollouts=rollouts,
        rollouts_per_prompt=rollouts_per_prompt,
        prompt_unique_sample_count=prompt_unique_sample_count,
        n_iteration=n_iteration,
    )
    print(f"Rollout metrics: {rollout_buffer_metrics}")

    # Create normalizer
    normalizer = normalizer_type.from_rewards(
        rewards=rollouts.map(lambda x: x.metrics.actual_rewards)
    )

    # Convert the rollouts to data for online training
    online_training_data: Slist[OnlineTrainingData] = rollouts.map(
        lambda r: evaluated_rollout_to_prompt_completion(
            evaluated_rollout=r, formatter=formatter, normalizer=normalizer
        )
    )
    # Fine-tune the model with the prompt completions
    return finetune_online_with_neptune(
        fine_tune_params=fine_tune_params,
        project_name=project_name,
        rollout_buffer_metrics=rollout_buffer_metrics,
        online_training_data=online_training_data,
        normalizer_type=normalizer_type,
        target_reward=target_reward,
    )


def calculate_rollout_metrics(
    rollouts: Slist[EvaluationWithGPTResponse],
    # for logging
    prompt_unique_sample_count: int,
    rollouts_per_prompt: int,
    n_iteration: int,
) -> RolloutBufferMetrics:
    rollout_count: int = len(rollouts)
    metrics: Slist[HelpfulHarmlessEvaluationMetric] = rollouts.map(lambda r: r.metrics)
    gpt_responses: Slist[GPTFullResponse] = rollouts.map(lambda m: m.full_response)

    average_harmless_reward: float = metrics.map(
        lambda m: m.actual_harmless
    ).average() or should_not_happen("Should not be empty")
    average_helpful_reward: float = metrics.map(
        lambda m: m.actual_helpful
    ).average() or should_not_happen("Should not be empty")
    average_rollout_length: float = gpt_responses.map(
        lambda r: r.completion_tokens_length
    ).average() or should_not_happen("Should not be empty")
    average_token_entropy: float = gpt_responses.map(
        lambda r: r.average_completion_total_log_prob or should_not_happen()
    ).average() or should_not_happen("Should not be empty")
    average_token_proba: float = gpt_responses.map(
        lambda r: r.average_completion_prob or should_not_happen()
    ).average() or should_not_happen("Should not be empty")
    return RolloutBufferMetrics(
        rollout_count=rollout_count,
        average_harmless_reward=average_harmless_reward,
        average_helpful_reward=average_helpful_reward,
        average_token_entropy=average_token_entropy,
        average_token_proba=average_token_proba,
        average_rollout_length=average_rollout_length,
        prompt_unique_sample_count=prompt_unique_sample_count,
        rollouts_per_prompt=rollouts_per_prompt,
        n_iteration=n_iteration,
    )


def main():
    online_project_name = ONLINE_POLICY_NEPTUNE_PROJECT
    formatter: PolicyPromptFormatter = RewardAtBottomFormatter()
    # Frozen reward models
    helpful_model = ModelId("babbage:ft-leadiq:helpful-reward-2022-12-22-08-04-46")
    harmless_model = ModelId("babbage:ft-leadiq:harmless-reward-2022-12-22-08-55-12")

    # Starting policy model
    # babbage:ft-leadiq:thejaminator-offline-assistant-policy-2022-12-28-17-51-17 150k samples
    policy_model = OpenaiInferenceConfig(
        model="babbage:ft-leadiq:thejaminator-online-assistant-policy-2023-01-14-08-43-41",
        top_p=1.0,
        temperature=1.0,
        max_tokens=600,
        stop=END_TOKEN,
    )
    # Get the target reward
    target_reward = HelpfulHarmlessReward(helpful=0.7, harmless=0.7)
    # Parameters to tweak
    normalizer_type: Type[DoNothingNormalizer] = DoNothingNormalizer
    n_iteration: int = 1
    prompt_sample_count = 128
    rollouts_per_prompt = 8
    all_dialogues: Slist[AnthropicRawFormat] = get_online_prompts()
    max_iterations = 20
    while True:
        if n_iteration > max_iterations:
            should_continue = cli_input_list(
                options=[True, False],
                start_text=f"Ran {max_iterations} iterations. Continue?",
            )
            if not should_continue:
                break
            else:
                number_more_iterations:int = cli_input_list(
                    options=[1, 2, 3, 4, 5, 10, 20, 50, 100],
                    start_text=f"How many more iterations?",
                )
                max_iterations += number_more_iterations
        print(f"Starting iteration {n_iteration}")
        # Sample dialogues
        sampled_dialogues: Slist[AnthropicRawFormat] = all_dialogues.sample(
            n=prompt_sample_count, seed=policy_model.model
        )
        # extend dialogues with the desired number of rollouts per prompt
        sampled_extended_dialogues: Slist[AnthropicRawFormat] = (
            sampled_dialogues * rollouts_per_prompt
        )
        # Params for the fine-tuning
        fine_tune_params = FineTuneParams(
            model=policy_model.model,
            n_epochs=1,
            # Lower than 0.1 because of scheduler?
            learning_rate_multiplier=0.05,
            prompt_loss_weight=0.0,
            batch_size=64,
        )
        # Single iteration
        new_model_id = single_iteration(
            dialogues=sampled_extended_dialogues,
            policy_model=policy_model,
            target_reward=target_reward,
            formatter=formatter,
            helpful_model=helpful_model,
            harmless_model=harmless_model,
            project_name=online_project_name,
            fine_tune_params=fine_tune_params,
            rollouts_per_prompt=rollouts_per_prompt,
            prompt_unique_sample_count=prompt_sample_count,
            n_iteration=n_iteration,
            normalizer_type=normalizer_type,
        )
        n_iteration += 1
        # Update the policy model
        policy_model.model = new_model_id
        print(f"Finished iteration {n_iteration}")


if __name__ == "__main__":
    main()
