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
from evaluate.inference import OpenaiInferenceConfig, GPTFullResponse, FinishReasons
from parsing.parse_raw import AnthropicRawFormat
from settings import (
    ONLINE_POLICY_NEPTUNE_PROJECT,
    NEPTUNE_KEY,
    OFFLINE_JOINT_POLICY_NEPTUNE_PROJECT,
    REWARD_NORMALIZER_NEPTUNE_KEY,
)
from train.assign_rewards import (
    assign_joint_target_reward,
)
from train.evaluate_joint_policy import (
    get_policy_single_evaluation,
    JointEvaluationWithGPTResponse,
    JointEvaluationMetric,
    plot_random_reward_evaluations, plot_random_reward_evaluations_like_paper,
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

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def from_rewards(rewards: Slist[float]) -> "TargetRewardSampler":
        raise NotImplementedError()




class UniformRandomSampler(TargetRewardSampler):
    def sample_target_reward(self) -> float:
        # uniform random between 0 and 1
        return random.random()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
        }

    @staticmethod
    def from_rewards(rewards: Slist[float]) -> "UniformRandomSampler":
        # Since we are sampling from a static distribution, we don't use the rewards
        return UniformRandomSampler()




class HighRewardSampler(TargetRewardSampler):
    def sample_target_reward(self) -> float:
        # rand between 0.7 and 1
        return random.random() * 0.3 + 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
        }

    @staticmethod
    def from_rewards(rewards: Slist[float]) -> "HighRewardSampler":
        # Since we are sampling from a static distribution, we don't use the rewards
        return HighRewardSampler()

class HighLowRandomSampler(TargetRewardSampler):
    def sample_target_reward(self) -> float:
        # sample between 0 and 0.3, and 0.7 and 1
        return random.random()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
        }

    @staticmethod
    def from_rewards(rewards: Slist[float]) -> "HighLowRandomSampler":
        # Since we are sampling from a static distribution, we don't use the rewards
        return HighLowRandomSampler()

class OneSDRewardSampler(TargetRewardSampler):
    # Samples one standard deviation above the mean
    def __init__(self, sd: float, mean: float):
        self.sd = sd
        self.mean = mean

    def sample_target_reward(self) -> float:
        # sample +- 0.05 around the sd
        return self.mean + self.sd + (random.random() * 0.1 - 0.05)

    @staticmethod
    def from_rewards(rewards: Slist[float]) -> "OneSDRewardSampler":
        std = rewards.standard_deviation()
        assert std is not None
        average = rewards.average()
        assert average is not None
        return OneSDRewardSampler(sd=std, mean=average)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
            "sd": self.sd,
            "mean": self.mean,
        }


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
    except RuntimeError as e:
        print(f"Runtime error: {e}")
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
    finish_reason: FinishReasons = evaluated_rollout.full_response.finish_reason
    return JointOnlineTrainingData(
        rollout_metric=evaluated_rollout.metrics,
        normalized_reward=normalized_reward,
        training_prompt=p_c.prompt,
        # Only add the end token if the rollout finished naturally
        training_completion=p_c.completion + END_TOKEN
        if finish_reason == FinishReasons.stop
        else p_c.completion,
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
    average_target_reward = online_training_data.map(
        lambda x: x.rollout_metric.target_reward
    ).average()

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
        run[REWARD_NORMALIZER_NEPTUNE_KEY] = normalizer.to_dict()
        run[f"online_metrics/reward_sampler"] = reward_sampler.to_dict()
        run[f"online_metrics/average_target_reward"] = average_target_reward

    return logged_fine_tune(
        train=online_training_data.map(lambda x: x.to_prompt_completion()),
        project_name=project_name,
        completion_start_token="",
        # Don't add the end token here as we only want to add it if it stopped naturally
        completion_end_token="",
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
) -> tuple[ModelId, TargetRewardSampler]:
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
    actual_rewards = online_training_data.map(lambda x: x.rollout_metric.actual_reward)
    new_reward_sampler: TargetRewardSampler = UniformRandomSampler.from_rewards(
        actual_rewards
    )
    print(f"New reward sampler: {new_reward_sampler.to_dict()}")
    # Fine-tune the model with the prompt completions
    return (
        finetune_online_with_neptune(
            fine_tune_params=fine_tune_params,
            project_name=project_name,
            rollout_buffer_metrics=rollout_buffer_metrics,
            online_training_data=online_training_data,
            reward_sampler=reward_sampler,
            normalizer=normalizer,
        ),
        new_reward_sampler,
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
    policy_model_id = ModelId(
        "babbage:ft-leadiq:thejaminator-online-assistant-policy-2023-01-20-08-05-05"
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
    prompt_sample_count = 256
    rollouts_per_prompt = 8
    all_dialogues: Slist[AnthropicRawFormat] = get_online_prompts()
    max_iterations = 20
    # Start with a UniformRandomSampler
    reward_sampler: TargetRewardSampler = UniformRandomSampler()
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
        new_model_id, new_reward_sampler = single_iteration(
            dialogues=sampled_extended_dialogues,
            policy_model=policy_model,
            reward_sampler=reward_sampler,
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
        # Update the reward sampler
        reward_sampler = new_reward_sampler
        print(f"Finished iteration {n_iteration}")


if __name__ == "__main__":
    main()
