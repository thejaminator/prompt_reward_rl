from typing import Sequence

from neptune.new import Run
from pydantic import BaseModel
from slist import Slist

from api.cli import cli_input_list
from api.dataset_paths import anthropic_online_train_path
from api.logged_fine_tune import logged_fine_tune
from api.openai_fine_tune import ModelId, FineTuneParams
from api.prompt_completion import PromptCompletion
from api.type_check import should_not_happen

from evaluate.inference import OpenaiInferenceConfig, GPTFullResponse
from parsing.parse_raw import AnthropicRawFormat, get_raw_anthropic
from settings import ONLINE_POLICY_NEPTUNE_PROJECT
from train.evaluate_offline import (
    get_policy_single_evaluation,
    HelpfulHarmlessEvaluationMetrics,
    EvaluationWithGPTResponse,
)
from train.assign_rewards import assign_maximum_target_reward, assign_target_reward
from train.policy_prompt_formatter import (
    PolicyPromptFormatter,
    PolicyPromptInfo,
    RewardAtBottomFormatter,
)
from train.reward_models import HelpfulHarmlessReward, DialogueWithReward
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
) -> EvaluationWithGPTResponse:
    # do the same thing, but using the maximum target reward
    dialogue_with_maximum_reward: DialogueWithReward = assign_target_reward(
        dialogue=dialogue,
        helpful_target=target_reward.helpful,
        harmless_target=target_reward.harmless,
    )
    formatted_maximum_reward: PolicyPromptInfo = (
        formatter.dialogue_reward_to_prompt_completion(dialogue_with_maximum_reward)
    )
    # Rollout the prompt using the policy model, with the target reward
    # This also evaluates the rollout and gets the actual reward
    rollout: EvaluationWithGPTResponse = get_policy_single_evaluation(
        policy_prompt_info=formatted_maximum_reward,
        policy_model=policy_model,
        helpful_model=helpful_model,
        harmless_model=harmless_model,
    )
    return rollout


def evaluated_rollout_to_prompt_completion(
    evaluated_rollout: EvaluationWithGPTResponse,
    formatter: PolicyPromptFormatter,
) -> PromptCompletion:
    # Replace the policy prompt with the actual reward.
    completed_dialogue: str = evaluated_rollout.metrics.completed_dialogue
    policy_prompt: PolicyPromptInfo = formatter.dialogue_reward_to_prompt_completion(
        with_reward=DialogueWithReward(
            dialogue=completed_dialogue,
            target_reward=HelpfulHarmlessReward(
                helpful=evaluated_rollout.metrics.actual_helpful,
                harmless=evaluated_rollout.metrics.actual_harmless,
            ),
        )
    )
    return policy_prompt.to_prompt_completion()


class RolloutBufferMetrics(BaseModel):
    rollout_count: int
    prompt_unique_sample_count: int
    rollouts_per_prompt: int
    average_harmless_reward: float
    average_helpful_reward: float
    average_token_entropy: float
    average_token_proba: float
    average_rollout_length: float


def finetune_with_neptune(
    prompt_completions: Sequence[PromptCompletion],
    project_name: str,
    fine_tune_params: FineTuneParams,
    rollout_metrics: RolloutBufferMetrics,
) -> ModelId:
    # Fine-tune the model with the prompt completions

    def neptune_log(run: Run) -> None:
        # Log the rollout metrics to neptune
        for k, v in rollout_metrics.dict().items():
            run[f"online_metrics/{k}"] = v

    return logged_fine_tune(
        train=prompt_completions,
        project_name=project_name,
        completion_start_token="",
        completion_end_token=END_TOKEN,
        params=fine_tune_params,
        neptune_pretrain_callable=neptune_log,
    )


def single_iteration(
    dialogues: Slist[AnthropicRawFormat],
    target_reward: HelpfulHarmlessReward,
    formatter: PolicyPromptFormatter,
    policy_model: OpenaiInferenceConfig,
    helpful_model: ModelId,
    harmless_model: ModelId,
    project_name: str,
    fine_tune_params: FineTuneParams,
    # for logging
    prompt_unique_sample_count: int,
    rollouts_per_prompt: int,
) -> ModelId:
    # Rollout and reward the prompts
    rollouts: Slist[EvaluationWithGPTResponse] = dialogues.map(
        lambda prompt: rollout_and_evaluate(
            dialogue=prompt.chosen,
            policy_model=policy_model,
            target_reward=target_reward,
            formatter=formatter,
            helpful_model=helpful_model,
            harmless_model=harmless_model,
        )
    )
    # Store additional metrics about the buffer of rollouts
    rollout_metrics: RolloutBufferMetrics = calculate_rollout_metrics(
        rollouts=rollouts,
        rollouts_per_prompt=rollouts_per_prompt,
        prompt_unique_sample_count=prompt_unique_sample_count,
    )

    # Convert the rollout and reward to a prompt completion
    prompts_completion: Slist[PromptCompletion] = rollouts.map(
        lambda r: evaluated_rollout_to_prompt_completion(
            evaluated_rollout=r, formatter=formatter
        )
    )
    # Fine-tune the model with the prompt completions
    return finetune_with_neptune(
        fine_tune_params=fine_tune_params,
        prompt_completions=prompts_completion,
        project_name=project_name,
        rollout_metrics=rollout_metrics,
    )


def calculate_rollout_metrics(
    rollouts: Slist[EvaluationWithGPTResponse],
    # for logging
    prompt_unique_sample_count: int,
    rollouts_per_prompt: int,
) -> RolloutBufferMetrics:
    rollout_count: int = len(rollouts)
    metrics: Slist[HelpfulHarmlessEvaluationMetrics] = rollouts.map(lambda r: r.metrics)
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
        model="babbage:ft-leadiq:thejaminator-offline-assistant-policy-2022-12-28-17-51-17",
        max_tokens=400,
    )
    # Get the target reward
    target_reward = HelpfulHarmlessReward(helpful=0.99, harmless=0.99)
    n_iteration: int = 0
    prompt_sample_count = 16
    rollouts_per_prompt = 8
    all_dialogues: Slist[AnthropicRawFormat] = get_online_prompts()
    max_iterations = 10
    while True:
        if n_iteration >= max_iterations:
            should_continue = cli_input_list(
                options=[True, False],
                start_text=f"Ran {max_iterations} iterations. Continue?",
            )
            if not should_continue:
                break
            else:
                max_iterations += 10

        # Sample dialogues
        sampled_dialogues: Slist[AnthropicRawFormat] = all_dialogues.sample(
            prompt_sample_count
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
            learning_rate_multiplier=0.025,
            prompt_loss_weight=0.0,
            batch_size=32,
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
        )
        n_iteration += 1
        # Update the policy model
        policy_model.model = new_model_id
        print(f"Finished iteration {n_iteration}")
