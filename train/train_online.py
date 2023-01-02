from typing import Sequence

from pydantic import BaseModel
from slist import Slist

from api.dataset_paths import anthropic_online_train_path
from api.openai_fine_tune import ModelId
from api.prompt_completion import PromptCompletion

from evaluate.inference import OpenaiInferenceConfig, GPTFullResponse
from parsing.parse_raw import AnthropicRawFormat, get_raw_anthropic
from train.evaluate_offline import (
    get_policy_single_evaluation,
    HelpfulHarmlessEvaluationMetrics,
    HelpfulHarmlessEvaluation,
)
from train.assign_rewards import assign_maximum_target_reward, assign_target_reward
from train.policy_prompt_formatter import (
    PolicyPromptFormatter,
    PolicyPromptInfo,
    RewardAtBottomFormatter,
)
from train.reward_models import HelpfulHarmlessReward, DialogueWithReward


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
) -> HelpfulHarmlessEvaluation:
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
    rollout: HelpfulHarmlessEvaluation = get_policy_single_evaluation(
        policy_prompt_info=formatted_maximum_reward,
        policy_model=policy_model,
        helpful_model=helpful_model,
        harmless_model=harmless_model,
    )
    return rollout


def evaluated_rollout_to_prompt_completion(
    evaluated_rollout: HelpfulHarmlessEvaluation,
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


def finetune(
    model: ModelId,
    prompt_completions: Sequence[PromptCompletion],
) -> ModelId:
    # Fine-tune the model with the prompt completions
    raise NotImplementedError


def single_iteration(
    dialogues: Slist[AnthropicRawFormat],
    target_reward: HelpfulHarmlessReward,
    formatter: PolicyPromptFormatter,
    policy_model: OpenaiInferenceConfig,
    helpful_model: ModelId,
    harmless_model: ModelId,
) -> ModelId:
    # Rollout and reward the prompts
    rollouts: Slist[HelpfulHarmlessEvaluation] = dialogues.map(
        lambda prompt: rollout_and_evaluate(
            dialogue=prompt.chosen,
            policy_model=policy_model,
            target_reward=target_reward,
            formatter=formatter,
            helpful_model=helpful_model,
            harmless_model=harmless_model,
        )
    )
    # Convert the rollout and reward to a prompt completion
    prompts_completion: Slist[PromptCompletion] = rollouts.map(
        lambda r: evaluated_rollout_to_prompt_completion(
            evaluated_rollout=r, formatter=formatter
        )
    )
    # Fine-tune the model with the prompt completions
    return finetune(ModelId(policy_model.model), prompts_completion)


def main():
    formatter: PolicyPromptFormatter = RewardAtBottomFormatter()
    # Frozen reward models
    helpful_model = ModelId("babbage:ft-leadiq:helpful-reward-2022-12-22-08-04-46")
    harmless_model = ModelId("babbage:ft-leadiq:harmless-reward-2022-12-22-08-55-12")

    # Starting policy model
    policy_model = OpenaiInferenceConfig(model="babbage", max_tokens=400)
    # Get the target reward
    target_reward = HelpfulHarmlessReward(helpful=1.00, harmless=1.00)
    n_iteration: int = 0
    rollout_buffer_size = 128
    all_dialogues: Slist[AnthropicRawFormat] = get_online_prompts()
    while True:
        # Sample dialogues
        sampled_dialogues: Slist[AnthropicRawFormat] = all_dialogues.sample(
            rollout_buffer_size
        )
        # Single iteration
        new_model_id = single_iteration(
            dialogues=sampled_dialogues,
            policy_model=policy_model,
            target_reward=target_reward,
            formatter=formatter,
            helpful_model=helpful_model,
            harmless_model=harmless_model,
        )
        n_iteration += 1
        print(f"Finished iteration {n_iteration}")
