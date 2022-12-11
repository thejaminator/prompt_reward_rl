import pandas as pd
from pathlib import Path

from pydantic import BaseModel
from slist import Slist

from api.json import read_jsonl_file_into_basemodel
from api.openai import OpenAIModeration
from create_processed import ProcessedWithModeration, ProcessedCompletion
from dataset_paths import moderated_completions


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


class PromptCompletion(BaseModel):
    prompt: str
    completion: str


def assign_reward(with_moderation: ProcessedWithModeration) -> ProcessedWithReward:
    moderation = with_moderation.moderation
    reward = calculate_reward(moderation)
    return ProcessedWithReward(
        processed=with_moderation.processed, moderation=moderation, reward=reward
    )


def format_prompt_completion(
    prompt: str, completion: str, reward: float
) -> PromptCompletion:
    # 2 d.p
    reward_formatted = f"{reward:.2f}"
    new_prompt = f"{prompt}\nReward: {reward_formatted}"
    return PromptCompletion(prompt=new_prompt, completion=completion)


def reward_to_prompt_completion(with_reward: ProcessedWithReward) -> PromptCompletion:
    return format_prompt_completion(
        prompt=with_reward.processed.last_human,
        completion=with_reward.processed.last_assistant,
        reward=with_reward.reward,
    )


def main() -> None:
    moderated_path: Path = moderated_completions
    moderated: Slist[ProcessedWithModeration] = read_jsonl_file_into_basemodel(
        path=moderated_path, basemodel=ProcessedWithModeration
    )
    with_rewards: Slist[ProcessedWithReward] = moderated.map(assign_reward)
    rewards_to_analyse: Slist[float] = with_rewards.map(lambda x: x.reward)
    # Use pandas to describe 25, 50, 75 percentiles
    print(f"Rewards summary for {rewards_to_analyse.length} completions")
    print(pd.Series(rewards_to_analyse).describe())
    prompt_completions: Slist[PromptCompletion] = with_rewards.map(
        reward_to_prompt_completion
    )


if __name__ == "__main__":
    main()
