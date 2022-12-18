import pandas as pd
from pathlib import Path

from pydantic import BaseModel
from slist import Slist

from api.dataset_paths import moderated_harmless_rejected
from api.json_operations import read_jsonl_file_into_basemodel
from api.logged_fine_tune import logged_fine_tune
from api.moderations import OpenAIModeration
from api.openai_fine_tune import FineTuneParams
from api.prompt_completion import PromptCompletion
from calculate_reward import ProcessedWithModeration, ProcessedCompletion


def calculate_reward(moderation: OpenAIModeration) -> float:
    """
    Higher reward is scores less on these fields
    We'll take the 1 - max(scores)
    hate: ModerationField
    hate_threatening: ModerationField
    self_harm: ModerationField
    sexual: ModerationField
    sexual_minors: ModerationField
    violence: ModerationField
    violence_graphic: ModerationField
    """
    fields = moderation.fields
    scores = Slist(
        [
            fields.hate.proba_score,
            fields.hate_threatening.proba_score,
            fields.self_harm.proba_score,
            fields.sexual.proba_score,
            fields.sexual_minors.proba_score,
            fields.violence.proba_score,
            fields.violence_graphic.proba_score,
        ]
    )
    max_bad = max(scores)
    reward = 1 - max_bad
    return reward


class ProcessedWithReward(BaseModel):
    processed: ProcessedCompletion
    moderation: OpenAIModeration
    reward: float


def assign_reward(with_moderation: ProcessedWithModeration) -> ProcessedWithReward:
    moderation = with_moderation.moderation
    reward = calculate_reward(moderation)
    return ProcessedWithReward(
        processed=with_moderation.processed, moderation=moderation, reward=reward
    )


def format_rewarded_prompt_completion(
    prompt: str, completion: str, reward: float
) -> PromptCompletion:
    # 2 d.p
    reward_formatted = f"{reward:.2f}"
    new_prompt = f"Human: {prompt}\nReward: {reward_formatted}\n"
    new_completion = f"Assistant: {completion}"
    return PromptCompletion(prompt=new_prompt, completion=new_completion)


def format_no_reward_prompt_completion(
    prompt: str, completion: str
) -> PromptCompletion:
    # 2 d.p
    new_prompt = f"{prompt}\n"
    return PromptCompletion(prompt=new_prompt, completion=completion)


def reward_to_prompt_completion(with_reward: ProcessedWithReward) -> PromptCompletion:
    return format_rewarded_prompt_completion(
        prompt=with_reward.processed.last_human,
        completion=with_reward.processed.last_assistant,
        reward=with_reward.reward,
    )


def main() -> None:
    moderated_path: Path = moderated_harmless_rejected
    moderated: Slist[ProcessedWithModeration] = read_jsonl_file_into_basemodel(
        path=moderated_path, basemodel=ProcessedWithModeration
    )
    with_rewards: Slist[ProcessedWithReward] = moderated.map(assign_reward).shuffle()
    rewards_to_analyse: Slist[float] = with_rewards.map(lambda x: x.reward)
    print(
        f"Number with less than 0.9 reward: {len(rewards_to_analyse.filter(lambda x: x < 0.7))}"
    )
    # Use pandas to describe 25, 50, 75 percentiles
    print(f"Rewards summary for {rewards_to_analyse.length} completions")
    print(pd.Series(rewards_to_analyse).describe())
    # Make it roughly 50/50 for those that have rewards less than 0.9
    less_than_0_9, more_than_0_9 = with_rewards.split_by(lambda x: x.reward < 0.9)
    print(f"Less than 0.9: {less_than_0_9.length}")
    print(f"More than 0.9: {more_than_0_9.length}")
    truncated_more_than_0_9 = more_than_0_9.take(less_than_0_9.length)
    final_with_rewards = less_than_0_9 + truncated_more_than_0_9
    print(f"Final with rewards: {final_with_rewards.length}")
    print(pd.Series(final_with_rewards.map(lambda x: x.reward)).describe())

    prompt_completions: Slist[PromptCompletion] = final_with_rewards.map(
        reward_to_prompt_completion
    )
    finetune_params = FineTuneParams(
        model="babbage",
        n_epochs=4,
        learning_rate_multiplier=0.1,
        batch_size=16,
        prompt_loss_weight=0.1,
    )
    logged_fine_tune(
        train=prompt_completions,
        params=finetune_params,
        project_name="reverse-rl",
        completion_start_token="",
        completion_end_token=" END",
    )


if __name__ == "__main__":
    main()
