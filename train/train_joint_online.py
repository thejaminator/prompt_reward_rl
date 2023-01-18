import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any

import pandas as pd
from neptune.new import Run
from neptune.new.types import File
from openai import InvalidRequestError
from pydantic import BaseModel
from slist import Slist

from api.cli import cli_input_list
from api.logged_fine_tune import logged_fine_tune, AlwaysContinueHandler
from api.openai_fine_tune import ModelId, FineTuneParams
from api.prompt_completion import PromptCompletion
from api.type_check import should_not_happen
from evaluate.inference import OpenaiInferenceConfig, GPTFullResponse
from parsing.parse_raw import AnthropicRawFormat
from settings import (
    ONLINE_POLICY_NEPTUNE_PROJECT,
    NEPTUNE_KEY,
    OFFLINE_JOINT_POLICY_NEPTUNE_PROJECT,
)
from train.assign_rewards import (
    assign_joint_target_reward,
)
from train.evaluate_joint_offline import (
    get_policy_single_evaluation,
    JointEvaluationWithGPTResponse,
    JointEvaluationMetric,
    plot_random_reward_evaluations,
)
from train.joint_policy_prompt_formatter import (
    JointPolicyPromptInfo,
    JointPolicyPromptFormatter,
    JointRewardAtBottomFormatter,
)
from train.neptune_utils.runs import get_openai_model_from_neptune
from train.normalizer.joint_reward_normalizer import (
    JointRewardNormalizer,
    get_joint_normalizer_from_neptune,
)
from train.reward_models import (
    DialogueWithJointReward,
)
from train.separators import END_TOKEN
from train.train_online import get_online_prompts


class TargetRewardSampler(ABC):
    @abstractmethod
    def sample_target_reward(self) -> float:
        raise NotImplementedError()

    @classmethod
    def name(cls) -> str:
        return cls.__name__


class UniformRandomSampler(TargetRewardSampler):
    def sample_target_reward(self) -> float:
        # uniform random between 0 and 1
        return random.random()


class HighRewardSampler(TargetRewardSampler):
    def sample_target_reward(self) -> float:
        return 0.8


def rollout_and_evaluate(
    dialogue: str,
    formatter: JointPolicyPromptFormatter,
    policy_model: OpenaiInferenceConfig,
    joint_reward_model: ModelId,
    normalizer: JointRewardNormalizer,
    reward_sampler: TargetRewardSampler,
) -> Optional[JointEvaluationWithGPTResponse]:
    dialogue_with_reward_reward: DialogueWithJointReward = assign_joint_target_reward(
        dialogue=dialogue,
        joint_target=reward_sampler.sample_target_reward(),
    )
    # Rollout the prompt using the policy model, with the target reward
    # This also evaluates the rollout and gets the actual reward
    try:
        rollout: JointEvaluationWithGPTResponse = get_policy_single_evaluation(
            dialogue_with_reward=dialogue_with_reward_reward,
            policy_model=policy_model,
            joint_reward_model=joint_reward_model,
            cached=False,  # Don't cache this since we have multiple rollouts, all with the same target temperature
            normalizer=normalizer,
            prompt_formatter=formatter,
        )
    except InvalidRequestError as e:
        print(f"Invalid request error, probably token limit: {e}")
        return None
    return rollout


class JointOnlineTrainingData(BaseModel):
    rollout_metric: JointEvaluationMetric
    normalized_reward: float
    training_prompt: str
    training_completion: str

    def to_prompt_completion(self) -> PromptCompletion:
        return PromptCompletion(
            prompt=self.training_prompt,
            completion=self.training_completion,
        )

    def to_flattened_dict(self) -> dict[str, Any]:
        rollout_metric_dict = self.rollout_metric.dict()
        more_data = {
            "normalized_reward": self.normalized_reward,
            "training_prompt": self.training_prompt,
            "training_completion": self.training_completion,
        }
        return {**rollout_metric_dict, **more_data}


def evaluated_rollout_to_prompt_completion(
    evaluated_rollout: JointEvaluationWithGPTResponse,
    formatter: JointPolicyPromptFormatter,
    normalizer: JointRewardNormalizer,
) -> JointOnlineTrainingData:
    # Replace the policy prompt with the actual reward.
    completed_dialogue: str = evaluated_rollout.metrics.completed_dialogue

    # actual reward
    actual_reward: float = evaluated_rollout.metrics.actual_reward
    # normalize the reward
    normalized_reward: float = normalizer.normalize_reward(actual_reward)

    policy_prompt: JointPolicyPromptInfo = (
        formatter.dialogue_reward_to_prompt_completion(
            with_reward=DialogueWithJointReward(
                dialogue=completed_dialogue,
                target_reward=normalized_reward,
            )
        )
    )
    p_c = policy_prompt.to_prompt_completion()
    return JointOnlineTrainingData(
        rollout_metric=evaluated_rollout.metrics,
        normalized_reward=normalized_reward,
        training_prompt=p_c.prompt,
        training_completion=p_c.completion,
    )


class JointRolloutBufferMetrics(BaseModel):
    rollout_count: int
    prompt_unique_sample_count: int
    rollouts_per_prompt: int
    average_reward: float
    average_token_entropy: float
    average_token_proba: float
    average_rollout_length: float
    n_iteration: int


def finetune_online_with_neptune(
    project_name: str,
    fine_tune_params: FineTuneParams,
    rollout_buffer_metrics: JointRolloutBufferMetrics,
    normalizer: JointRewardNormalizer,
    online_training_data: Slist[JointOnlineTrainingData],
    reward_sampler: TargetRewardSampler,
) -> ModelId:
    # get the correlation of taret vs actual reward
    plot = plot_random_reward_evaluations(
        online_training_data.map(lambda x: x.rollout_metric)
    )

    # Fine-tune the model with the prompt completions

    def pre_train_log(run: Run) -> None:
        # Log the rollout metrics to neptune
        for k, v in rollout_buffer_metrics.dict().items():
            run[f"online_metrics/{k}"] = v

        # write the rollouts
        rollouts_df = pd.DataFrame(
            online_training_data.map(lambda x: x.to_flattened_dict())
        )
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
        run["online_metrics/normalizer_type"] = normalizer.name()
        run["online_metrics/reward_sampler"] = reward_sampler.name()
        # Log the results under evaluation
        run[f"online_metrics/correlation"] = plot.correlation
        run[f"online_metrics/pvalue"] = plot.p_value
        # upper bound
        run[f"online_metrics/upper_bound"] = plot.upper_correlation_bound
        # lower bound
        run[f"online_metrics/lower_bound"] = plot.lower_correlation_bound
        # confidence
        run[f"online_metrics/confidence_level"] = plot.confidence_level

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
    reward_sampler: TargetRewardSampler,
    formatter: JointPolicyPromptFormatter,
    policy_model: OpenaiInferenceConfig,
    joint_reward_model: ModelId,
    project_name: str,
    fine_tune_params: FineTuneParams,
    # for logging
    prompt_unique_sample_count: int,
    rollouts_per_prompt: int,
    n_iteration: int,
    normalizer: JointRewardNormalizer,
) -> ModelId:
    # Rollout and reward the prompts
    rollouts: Slist[JointEvaluationWithGPTResponse] = dialogues.par_map(
        lambda prompt: rollout_and_evaluate(
            dialogue=prompt.chosen,
            policy_model=policy_model,
            reward_sampler=reward_sampler,
            formatter=formatter,
            joint_reward_model=joint_reward_model,
            normalizer=normalizer,
        ),
        executor=threadpool,
    ).flatten_option()
    # Store additional metrics about the buffer of rollouts
    rollout_buffer_metrics: JointRolloutBufferMetrics = calculate_rollout_metrics(
        rollouts=rollouts,
        rollouts_per_prompt=rollouts_per_prompt,
        prompt_unique_sample_count=prompt_unique_sample_count,
        n_iteration=n_iteration,
    )
    print(f"Rollout metrics: {rollout_buffer_metrics}")

    # Convert the rollouts to data for online training
    online_training_data: Slist[JointOnlineTrainingData] = rollouts.map(
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
        reward_sampler=reward_sampler,
        normalizer=normalizer,
    )


def calculate_rollout_metrics(
    rollouts: Slist[JointEvaluationWithGPTResponse],
    # for logging
    prompt_unique_sample_count: int,
    rollouts_per_prompt: int,
    n_iteration: int,
) -> JointRolloutBufferMetrics:
    rollout_count: int = len(rollouts)
    metrics: Slist[JointEvaluationMetric] = rollouts.map(lambda r: r.metrics)
    gpt_responses: Slist[GPTFullResponse] = rollouts.map(lambda m: m.full_response)

    average_reward: float = metrics.map(
        lambda m: m.actual_reward
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
    return JointRolloutBufferMetrics(
        rollout_count=rollout_count,
        average_reward=average_reward,
        average_token_entropy=average_token_entropy,
        average_token_proba=average_token_proba,
        average_rollout_length=average_rollout_length,
        prompt_unique_sample_count=prompt_unique_sample_count,
        rollouts_per_prompt=rollouts_per_prompt,
        n_iteration=n_iteration,
    )


def main():
    online_project_name = ONLINE_POLICY_NEPTUNE_PROJECT
    formatter: JointPolicyPromptFormatter = JointRewardAtBottomFormatter()

    # Starting policy model
    # Parameters to tweak
    normalizer: JointRewardNormalizer = get_joint_normalizer_from_neptune(
        neptune_api_key=NEPTUNE_KEY,
        neptune_project_name=OFFLINE_JOINT_POLICY_NEPTUNE_PROJECT,
        neptune_run_id="OF1-24",
    )
    policy_model_id = get_openai_model_from_neptune(
        neptune_api_key=NEPTUNE_KEY,
        neptune_project_name=OFFLINE_JOINT_POLICY_NEPTUNE_PROJECT,
        neptune_run_id="OF1-24",
    )
    joint_reward_model = ModelId(
        "babbage:ft-leadiq:assistant-reward-model-2022-12-20-09-34-26"
    )
    policy_model = OpenaiInferenceConfig(
        model=policy_model_id,
        top_p=1.0,
        temperature=1.0,
        max_tokens=600,
        stop=END_TOKEN,
    )
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
                number_more_iterations: int = cli_input_list(
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
            reward_sampler=UniformRandomSampler(),
            formatter=formatter,
            joint_reward_model=joint_reward_model,
            project_name=online_project_name,
            fine_tune_params=fine_tune_params,
            rollouts_per_prompt=rollouts_per_prompt,
            prompt_unique_sample_count=prompt_sample_count,
            n_iteration=n_iteration,
            normalizer=normalizer,
        )
        n_iteration += 1
        # Update the policy model
        policy_model.model = new_model_id
        print(f"Finished iteration {n_iteration}")


if __name__ == "__main__":
    main()