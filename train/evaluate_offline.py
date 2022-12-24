import random
from concurrent.futures import ThreadPoolExecutor

import scipy.stats
from openai.error import RateLimitError
from retry import retry
from scipy.stats import pearsonr, _result_classes
from pydantic import BaseModel
from slist import Slist

from api.openai_fine_tune import ModelId
from api.set_key import set_openai_key
from evaluate.classification import (
    format_dialogue_into_reward_prompt,
    get_positive_class_proba,
    PromptForRewardModel,
)
from evaluate.inference import (
    GPTFullResponse,
    get_openai_completion,
    OpenaiInferenceConfig,
    cached_get_openai_completion,
)
from parsing.parse_raw import AnthropicRawFormat
from settings import OPENAI_KEY
from train.evaluate_reward_model import (
    TestDataset,
    get_harmless_helpful_test,
    get_harmless_test,
    get_helpful_test,
)
from train.policy_prompt_formatter import (
    PolicyPromptFormatter,
    PolicyPromptInfo,
    PolicyRewardAtBottomFormatter,
)
from train.reward_models import DialogueWithReward, HelpfulHarmlessReward
from train.separators import END_TOKEN
import pandas as pd


class HelpfulHarmlessEvaluation(BaseModel):
    policy_prompt: str
    reward_prompt: str
    completion: str
    # Prompt + completion, without the reward
    completed_dialogue: str
    target_helpful: float
    actual_helpful: float
    target_harmless: float
    actual_harmless: float


def assign_random_target_reward(dialogue: str, rollout_number: int) -> DialogueWithReward:
    # random harmless_target 0 to 1
    random_harmless_target = random.Random(dialogue + "harmless" + str(rollout_number)).random()
    # random helpful_target 0 to 1
    random_helpful_target = random.Random(dialogue + "helpful" + str(rollout_number)).random()

    return DialogueWithReward(
        dialogue=dialogue,
        target_reward=HelpfulHarmlessReward(
            helpful=random_helpful_target, harmless=random_harmless_target
        ),
    )


@retry(exceptions=RateLimitError, tries=5, delay=20)
def get_policy_single_evaluation(
    policy_prompt_info: PolicyPromptInfo,
    policy_model: OpenaiInferenceConfig,
    helpful_model: ModelId,
    harmless_model: ModelId,
) -> HelpfulHarmlessEvaluation:
    policy_prompt = policy_prompt_info.to_prompt_completion().prompt
    policy_completion: GPTFullResponse = cached_get_openai_completion(
        prompt=policy_prompt_info.to_prompt_completion().prompt, config=policy_model
    )
    # You need to get the prompt that does not have the target reward, and add the completion
    dialogue = (
        policy_prompt_info.dialogue_without_reward_without_completion
        + "\n\n"
        + policy_completion.completion
    )
    formatted_reward_prompt: PromptForRewardModel = format_dialogue_into_reward_prompt(
        dialogue
    )
    actual_helpful_reward = get_positive_class_proba(
        helpful_model, prompt=formatted_reward_prompt
    )
    actual_harmless_reward = get_positive_class_proba(
        harmless_model, prompt=formatted_reward_prompt
    )
    return HelpfulHarmlessEvaluation(
        policy_prompt=policy_prompt,
        reward_prompt=formatted_reward_prompt,
        completion=policy_completion.completion,
        completed_dialogue=dialogue,
        target_helpful=policy_prompt_info.target_reward.helpful,
        actual_helpful=actual_helpful_reward,
        target_harmless=policy_prompt_info.target_reward.harmless,
        actual_harmless=actual_harmless_reward,
    )


class TextWithRolloutNumber(BaseModel):
    text: str
    rollout_number: int


def extend_with_rollouts(text: str, rollouts: int) -> Slist[TextWithRolloutNumber]:
    # Add the rollout_number so that the random seed is different for each rollout
    # The target reward generated will be the same between runs
    # So now you can cache the rollout
    return Slist(
        [TextWithRolloutNumber(text=text, rollout_number=i) for i in range(rollouts)]
    )


def main(
    sample_prompts: int,
    policy_model: OpenaiInferenceConfig,
    helpful_model: ModelId,
    harmless_model: ModelId,
    openai_api_key: str,
    test_dataset: TestDataset,
    rollouts_per_prompt: int,
    policy_formatter: PolicyPromptFormatter,
):
    threadpool = ThreadPoolExecutor(max_workers=20)
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
    sample_dialogue_upscaled: Slist[TextWithRolloutNumber] = sample_dialogue.map(
        lambda x: extend_with_rollouts(text=x, rollouts=rollouts_per_prompt)
    ).flatten_list()
    sample_dialogue_with_target_rewards: Slist[
        DialogueWithReward
    ] = sample_dialogue_upscaled.map(
        lambda text_with_rollout: assign_random_target_reward(
            dialogue=text_with_rollout.text, rollout_number=text_with_rollout.rollout_number
        )
    )
    # Use policy_formatter to format the prompts
    formatted_prompts: Slist[
        PolicyPromptInfo
    ] = sample_dialogue_with_target_rewards.map(
        lambda x: policy_formatter.dialogue_reward_to_prompt_completion(x)
    )
    # Rollout the prompts to get the completions, and get the actual rewards
    evaluations: Slist[HelpfulHarmlessEvaluation] = formatted_prompts.par_map(
        lambda x: get_policy_single_evaluation(
            policy_prompt_info=x,
            policy_model=policy_model,
            helpful_model=helpful_model,
            harmless_model=harmless_model,
        ),
        executor=threadpool,
    )
    # Calculate correlation coefficient of harmless
    pearson_harmless: _result_classes.PearsonRResult = pearsonr(
        x=evaluations.map(lambda x: x.target_harmless),
        y=evaluations.map(lambda x: x.actual_harmless),
    )
    correlation_harmless = pearson_harmless.statistic
    pvalue_harmless = pearson_harmless.pvalue

    # Calculate correlation coefficient of helpful
    pearson_helpful: _result_classes.PearsonRResult = pearsonr(
        x=evaluations.map(lambda x: x.target_helpful),
        y=evaluations.map(lambda x: x.actual_helpful),
    )
    correlation_helpful = pearson_helpful.statistic
    pvalue_helpful = pearson_helpful.pvalue

    # Print the results
    print(f"Correlation of Harmless: {correlation_harmless}")
    print(f"P-value of Harmless: {pvalue_harmless}")
    print(f"Correlation of Helpful: {correlation_helpful}")
    print(f"P-value of Helpful: {pvalue_helpful}")

    # write the results to pandas csv
    df = pd.DataFrame(evaluations.map(lambda x: x.dict()))
    df.to_csv("evaluate_offline.csv", index=False)


if __name__ == "__main__":
    # 50k pairs babbage:ft-leadiq:thejaminator-offline-assistant-policy-2022-12-23-19-39-52
    # 25k pairs babbage:ft-leadiq:thejaminator-offline-assistant-policy-2022-12-24-13-56-20
    # 10k pairs babbage:ft-leadiq:thejaminator-offline-assistant-policy-2022-12-23-16-19-09
    # 1k pairs babbage:ft-leadiq:thejaminator-offline-assistant-policy-2022-12-23-08-30-50
    policy_config = OpenaiInferenceConfig(
        model="babbage:ft-leadiq:thejaminator-offline-assistant-policy-2022-12-23-19-39-52",
        temperature=1.0,  # try 0.6, 1.0
        max_tokens=400,
        top_p=1.0,
        stop=END_TOKEN,
    )

    main(
        sample_prompts=500,
        rollouts_per_prompt=1,
        policy_model=policy_config,
        openai_api_key=OPENAI_KEY,
        test_dataset=TestDataset.HARMLESS_HELPFUL,
        policy_formatter=PolicyRewardAtBottomFormatter(),
        helpful_model=ModelId("babbage:ft-leadiq:helpful-reward-2022-12-22-08-04-46"),
        harmless_model=ModelId("babbage:ft-leadiq:harmless-reward-2022-12-22-08-55-12"),
    )

    # 1,102,676 tokens for 6122 requests (samples)
    # 1,102,676 / 6122 * 1000 = 180,116 tokens per 1000 samples
    # 180,116 / 1000 * 0.0024 = $0.43 per 1000 samples

    # We need to run 2 reward models, and 1 policy model
    # 0.43 * 3 = $1.29 per 1000 samples
