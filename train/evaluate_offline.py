import random
from typing import Optional

from pydantic import BaseModel
from slist import Slist

from api.openai_fine_tune import ModelId
from api.prompt_completion import PromptCompletion
from api.set_key import set_openai_key
from evaluate.classification import get_pair_predicted_chosen
from evaluate.inference import GPTFullResponse
from parsing.parse_raw import AnthropicRawFormat
from settings import OPENAI_KEY
from train.evaluate_reward_model import (
    TestDataset,
    get_harmless_helpful_test,
    get_harmless_test,
    get_helpful_test,
)
from train.reward_formatter import PolicyPromptFormatter
from train.reward_models import DialogueWithReward, HelpfulHarmlessReward


class HelpfulHarmlessEvaluation(BaseModel):
    prompt: str
    completion: str
    # Prompt + completion, without the reward
    conversation: str
    target_helpful: float
    actual_helpful: float
    target_harmless: float
    actual_harmless: float


def assigned_random_target_reward(dialogue: str, idx: int) -> DialogueWithReward:
    # random harmless_target 0 to 1
    random_harmless_target = random.Random(dialogue + "harmless" + str(idx)).random()
    # random helpful_target 0 to 1
    random_helpful_target = random.Random(dialogue + "helpful" + str(idx)).random()

    return DialogueWithReward(
        dialogue=dialogue,
        target_reward=HelpfulHarmlessReward(
            helpful=random_helpful_target, harmless=random_harmless_target
        ),
    )


def main(
    sample_prompts: int,
    policy_model: ModelId,
    helpful_model: ModelId,
    harmless_model: ModelId,
    openai_api_key: str,
    test_dataset: TestDataset,
    rollouts_per_prompt: int,
    policy_formatter: PolicyPromptFormatter,
):
    set_openai_key(openai_api_key)
    seed = "999"
    # Load the test datasets
    all_test = (
        get_harmless_helpful_test()
        if test_dataset == TestDataset.HARMLESS_HELPFUL
        else (
            get_harmless_test()
            if test_dataset == TestDataset.HARMLESS
            else get_helpful_test()
        )
    )
    sample: Slist[AnthropicRawFormat] = all_test.shuffle(seed=seed).take(sample_prompts)
    # We are just chosen to use the dialogue before the completion
    # so we can just take the .chosen which will be the same as .rejected
    sample_dialogue: Slist[str] = sample.map(lambda raw: raw.chosen)
    sample_dialogue_upscaled: Slist[str] = sample_dialogue * rollouts_per_prompt
    sample_dialogue_with_target_rewards: Slist[
        DialogueWithReward
    ] = sample_dialogue_upscaled.map_enumerate(
        lambda dialogue, idx: assigned_random_target_reward(dialogue=dialogue, idx=idx)
    )
    # Use policy_formatter to format the prompts
    # We only need the prompt, not the actual completion
    formatted_prompts: Slist[str] = sample_dialogue_with_target_rewards.map(
        policy_formatter.dialogue_reward_to_prompt_completion
    ).map(lambda x: x.prompt)
    # Rollout the prompts to get the completions
    completions: Slist[GPTFullResponse] = ...
    # Use the reward models to evaluate the completions
    evaluation: Slist[HelpfulHarmlessEvaluation] = ...
    # Calculate correlation coefficient of harmless
    correlation_harmless: float = ...
    # Calculate correlation coefficient of helpful
    correlation_helpful: float = ...


if __name__ == "__main__":
    main(
        sample_prompts=100,
        rollouts_per_prompt=5,
        policy_model="babbage:ft-leadiq:harmless-reward-2022-12-22-08-55-12",
        openai_api_key=OPENAI_KEY,
        test_dataset=TestDataset.HARMLESS,
    )
