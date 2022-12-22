from typing import Sequence

from pydantic import BaseModel
from slist import Slist

from api.openai_fine_tune import ModelId
from api.prompt_completion import PromptCompletion
from calculate_reward import AnthropicRawFormat
from evaluate.inference import OpenaiInferenceConfig, GPTFullResponse
from train.reward_models import HelpfulHarmlessReward


def get_online_prompts() -> Slist[AnthropicRawFormat]:
    # Get the prompts that we are going to use for rollouts
    ...


def rollout_prompt(
    prompt: str,
    policy_model: OpenaiInferenceConfig,
    target_reward: HelpfulHarmlessReward,
) -> GPTFullResponse:
    # Rollout the prompt using the policy model, with the target reward
    ...


class EvaluatedRollout(BaseModel):
    target_reward: HelpfulHarmlessReward
    actual_reward: HelpfulHarmlessReward
    prompt_without_reward: str
    rollout: GPTFullResponse


def rollout_and_reward_prompt(
    prompt: str,
    policy_model: OpenaiInferenceConfig,
    target_reward: HelpfulHarmlessReward,
) -> EvaluatedRollout:
    ...


def evaluated_rollout_to_prompt_completion(
    evaluated_rollout: EvaluatedRollout,
) -> PromptCompletion:
    ...


def finetune(
    model: ModelId,
    prompt_completions: Sequence[PromptCompletion],
) -> ModelId:
    # Fine-tune the model with the prompt completions
    ...


def single_iteration(
    prompts: Slist[AnthropicRawFormat],
    policy_model: OpenaiInferenceConfig,
    target_reward: HelpfulHarmlessReward,
) -> ModelId:
    # Rollout and reward the prompts
    rollout_and_rewards = prompts.map(
        lambda prompt: rollout_and_reward_prompt(
            prompt.chosen, policy_model, target_reward
        )
    )
    # Convert the rollout and reward to a prompt completion
    prompts_completion: Slist[PromptCompletion] = rollout_and_rewards.map(
        evaluated_rollout_to_prompt_completion
    )
    # Fine-tune the model with the prompt completions
    return finetune(ModelId(policy_model.model), prompts_completion)


def main():
    # Starting policy model
    policy_model = OpenaiInferenceConfig(model="curie", max_tokens=100)
    # Get the target reward
    target_reward = HelpfulHarmlessReward(helpful=1.0, harmless=1.0)
    n_iteration: int = 0
    while True:
        # Get the prompts
        prompts = get_online_prompts()
        # Single iteration
        new_model_id = single_iteration(
            prompts=prompts, policy_model=policy_model, target_reward=target_reward
        )
        n_iteration += 1
        print(f"Finished iteration {n_iteration}")
