from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import neptune
import neptune.new
import pandas as pd
from neptune.new import Run
from neptune.new.types import File
from openai.error import RateLimitError, APIConnectionError
from pydantic import BaseModel
from retry import retry
from slist import Slist
from slist.pydantic_compat import SlistPydantic

from api.openai_fine_tune import ModelId
from api.set_key import set_openai_key
from evaluate.classification import (
    format_dialogue_into_reward_prompt,
    get_positive_class_proba,
    PromptForRewardModel,
)
from evaluate.inference import (
    GPTFullResponse,
    OpenaiInferenceConfig,
    cached_get_openai_completion,
    get_openai_completion,
)
from parsing.parse_raw import AnthropicRawFormat
from settings import (
    OPENAI_KEY,
    NEPTUNE_KEY,
    OFFLINE_SEPARATE_POLICY_NEPTUNE_PROJECT,
    MODEL_ID_NEPTUNE_KEY,
)
from train.assign_rewards import (
    assign_random_joint_target_reward, assign_maximum_joint_target_reward,
)
from train.evaluate_offline import extend_with_rollout_number, ScatterplotResults, plot_scatterplot_and_correlation, \
    TextWithRolloutNumber
from train.evaluate_reward_model import (
    TestDataset,
    get_helpful_test, get_harmless_test, get_harmless_helpful_test,
)
from train.joint_policy_prompt_formatter import JointPolicyPromptInfo, JointPolicyPromptFormatter, \
    JointRewardAtBottomFormatter
from train.reward_models import DialogueWithJointReward
from train.separators import END_TOKEN


class JointEvaluationMetric(BaseModel):
    policy_prompt: str
    # prompt given to the reward model for determining the actual reward
    reward_prompt: str
    completion: str
    # Prompt + completion, without the reward
    completed_dialogue: str
    target_reward: float
    actual_reward: float



class JointEvaluationWithGPTResponse(BaseModel):
    metrics: JointEvaluationMetric
    full_response: GPTFullResponse


@retry(exceptions=(RateLimitError, APIConnectionError), tries=5, delay=20)
def get_policy_single_evaluation(
    policy_prompt_info: JointPolicyPromptInfo,
    policy_model: OpenaiInferenceConfig,
    joint_reward_model: ModelId,
    cached: bool = True,
) -> JointEvaluationWithGPTResponse:
    policy_prompt = policy_prompt_info.to_prompt_completion().prompt
    # rollout the policy
    policy_completion: GPTFullResponse = (
        cached_get_openai_completion(
            prompt=policy_prompt_info.to_prompt_completion().prompt, config=policy_model
        )
        if cached
        else get_openai_completion(
            prompt=policy_prompt_info.to_prompt_completion().prompt, config=policy_model
        )
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
    actual_reward = get_positive_class_proba(
        joint_reward_model, prompt=formatted_reward_prompt
    )
    metrics = JointEvaluationMetric(
        policy_prompt=policy_prompt,
        reward_prompt=formatted_reward_prompt,
        completion=policy_completion.completion,
        completed_dialogue=dialogue,
        target_reward=policy_prompt_info.target_reward,
        actual_reward=actual_reward,
    )
    return JointEvaluationWithGPTResponse(
        metrics=metrics,
        full_response=policy_completion,
    )

class EvaluationResults(BaseModel):
    random_target_rewards: SlistPydantic[JointEvaluationMetric]
    maximum_target_rewards: SlistPydantic[JointEvaluationMetric]


def run_evaluation(
    sample_prompts: int,
    policy_model: OpenaiInferenceConfig,
    joint_model: ModelId,
    openai_api_key: str,
    test_dataset: TestDataset,
    rollouts_per_prompt: int,
    policy_formatter: JointPolicyPromptFormatter,
) -> EvaluationResults:
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
    # We are just using `chosen` to use the dialogue before the completion
    # so we can just take the .chosen which will be the same as .rejected
    sample_dialogue: Slist[str] = sample.map(lambda raw: raw.chosen)
    sample_dialogue_upscaled: Slist[TextWithRolloutNumber] = sample_dialogue.map(
        lambda x: extend_with_rollout_number(text=x, rollouts=rollouts_per_prompt)
    ).flatten_list()
    # We want to see the performance correlation with a random target reward
    sample_dialogue_with_random_rewards: Slist[
        DialogueWithJointReward
    ] = sample_dialogue_upscaled.map(
        lambda text_with_rollout: assign_random_joint_target_reward(
            dialogue=text_with_rollout.text,
            rollout_number=text_with_rollout.rollout_number,
        )
    )
    # Use policy_formatter to format the prompts
    formatted_random_prompts: Slist[
        JointPolicyPromptInfo
    ] = sample_dialogue_with_random_rewards.map(
        lambda x: policy_formatter.dialogue_reward_to_prompt_completion(x)
    )

    # Rollout the prompts to get the completions, and get the actual rewards
    evaluated_random_prompts: Slist[
        JointEvaluationMetric
    ] = formatted_random_prompts.par_map(
        lambda x: get_policy_single_evaluation(
            policy_prompt_info=x,
            policy_model=policy_model,
            joint_reward_model=joint_model,
        ).metrics,
        executor=threadpool,
    )

    # do the same thing, but using the maximum target reward
    sample_dialogue_with_maximum_rewards: Slist[
        DialogueWithJointReward
    ] = sample_dialogue_upscaled.map(
        lambda text_with_rollout: assign_maximum_joint_target_reward(
            dialogue=text_with_rollout.text
        )
    )
    formatted_maximum_prompts: Slist[
        JointPolicyPromptInfo
    ] = sample_dialogue_with_maximum_rewards.map(
        lambda x: policy_formatter.dialogue_reward_to_prompt_completion(x)
    )
    evaluated_maximum_prompts: Slist[
        JointEvaluationMetric
    ] = formatted_maximum_prompts.par_map(
        lambda x: get_policy_single_evaluation(
            policy_prompt_info=x,
            policy_model=policy_model,
            joint_reward_model=joint_model,
        ).metrics,
        executor=threadpool,
    )

    return EvaluationResults(
        random_target_rewards=evaluated_random_prompts,
        maximum_target_rewards=evaluated_maximum_prompts,
    )


def plot_random_reward_evaluations(
    random_rewards_evaluations: Slist[JointEvaluationMetric],
) -> ScatterplotResults:
    # Calculate correlation coefficient of harmless
    target: Slist[float] = random_rewards_evaluations.map(
        lambda x: x.target_reward
    )
    actual: Slist[float] = random_rewards_evaluations.map(
        lambda x: x.actual_reward
    )
    results: ScatterplotResults = plot_scatterplot_and_correlation(
        x=actual,
        y=target,
        title="Harmless",
        xlabel="Actual",
        ylabel="Target",
    )
    return results



def log_results_to_neptune(
    scatter_plots: ScatterplotResults,
    neptune_api_key: str,
    neptune_project_name: str,
    neptune_run_id: str,
    policy_config: OpenaiInferenceConfig,
    joint_model: ModelId,
    formatter: JointPolicyPromptFormatter,
    number_samples: int,
    rollouts_per_sample: int,
    results: EvaluationResults,
) -> None:
    temperature = policy_config.temperature
    # add the temperature to the evaluation_key
    evaluation_key = f"evaluation_temp_{temperature}"

    print(f"Correlation coeffiicent of joint: {scatter_plots.correlation}")

    # Calculate the average reward when using the maximum target reward
    average_reward: Optional[float] = results.maximum_target_rewards.map(
        lambda x: x.actual_reward
    ).average()

    # print
    print(
        f"Average reward for maximum target reward: {average_reward}"
    )

    # convert to df
    # write the results to dataframe
    random_rewards_df = pd.DataFrame(
        results.random_target_rewards.map(lambda x: x.dict())
    )
    maximum_target_rewards_df = pd.DataFrame(
        results.maximum_target_rewards.map(lambda x: x.dict())
    )

    # Save to neptune
    run = neptune.new.init_run(
        with_id=neptune_run_id,
        project=f"{neptune_project_name}",
        api_token=neptune_api_key,
    )
    try:
        # Log the config
        run[f"{evaluation_key}/policy_config"] = policy_config.dict()
        run[f"{evaluation_key}/joint_model"] = joint_model
        run[f"{evaluation_key}/policy_formatter"] = formatter.name
        run[f"{evaluation_key}/number_samples"] = number_samples
        run[f"{evaluation_key}/rollouts_per_sample"] = rollouts_per_sample
        # Log the results under evaluation
        run[f"{evaluation_key}/correlation"] = scatter_plots.correlation
        run[f"{evaluation_key}/pvalue"] = scatter_plots.p_value
        # upper bound
        run[
            f"{evaluation_key}/upper_bound"
        ] = scatter_plots.upper_correlation_bound
        # lower bound
        run[
            f"{evaluation_key}/lower_bound"
        ] = scatter_plots.lower_correlation_bound
        # confidence
        run[
            f"{evaluation_key}/confidence_level"
        ] = scatter_plots.confidence_level

        # Log the average reward for maximum target reward
        run[
            f"{evaluation_key}/average_joint_maximum_target"
        ] = average_reward
        # Log the plots
        run[f"{evaluation_key}/joint_plot"].upload(scatter_plots.figure)

        # Save the results dataframe as html
        run[f"{evaluation_key}/random_rewards_html"].upload(
            File.as_html(random_rewards_df)
        )
        # save the random rewards dataframe as jsonl
        results_jsonl = random_rewards_df.to_json(orient="records", lines=True)
        # write the jsonl to a file
        evaluation_path = "random_rewards.jsonl"
        with open(evaluation_path, "w") as f:
            f.write(results_jsonl)
        run[f"{evaluation_key}/random_rewards_jsonl"].upload(evaluation_path)
        # Save the maximum target rewards dataframe as html
        run[f"{evaluation_key}/maximum_target_rewards_html"].upload(
            File.as_html(maximum_target_rewards_df)
        )
        # save the maximum target rewards dataframe as jsonl
        results_jsonl = maximum_target_rewards_df.to_json(orient="records", lines=True)
        # write the jsonl to a file
        evaluation_path = "maximum_target_rewards.jsonl"
        with open(evaluation_path, "w") as f:
            f.write(results_jsonl)
        run[f"{evaluation_key}/maximum_target_rewards_jsonl"].upload(evaluation_path)

    finally:
        run.stop()


def get_neptune_run(
    neptune_api_key: str,
    neptune_project_name: str,
    neptune_run_id: str,
) -> Run:
    run = neptune.new.init_run(
        with_id=neptune_run_id,
        project=f"{neptune_project_name}",
        api_token=neptune_api_key,
    )
    return run


def get_openai_model_from_neptune(
    neptune_api_key: str,
    neptune_project_name: str,
    neptune_run_id: str,
) -> ModelId:
    run = get_neptune_run(
        neptune_api_key=neptune_api_key,
        neptune_project_name=neptune_project_name,
        neptune_run_id=neptune_run_id,
    )
    model_id: ModelId = run[MODEL_ID_NEPTUNE_KEY].fetch()
    assert model_id is not None, "Model id is None"
    run.stop()
    return model_id


if __name__ == "__main__":
    # Optionally retrieve the openai model id from neptune
    run_id = "OF-17"
    policy_model_id = get_openai_model_from_neptune(
        neptune_api_key=NEPTUNE_KEY,
        neptune_project_name=OFFLINE_SEPARATE_POLICY_NEPTUNE_PROJECT,
        neptune_run_id=run_id,
    )
    policy_config = OpenaiInferenceConfig(
        model=policy_model_id,  # You can set this manually too
        temperature=1.0,  # try 0.6, 1.0
        max_tokens=400,
        top_p=1.0,
        stop=END_TOKEN,
    )
    joint_model = ModelId("babbage:ft-leadiq:thejaminator-offline-assistant-policy-2023-01-04-15-10-04")
    policy_formatter = JointRewardAtBottomFormatter()
    number_samples = 500
    rollouts_per_prompts = 1
    evaluations: EvaluationResults = run_evaluation(
        sample_prompts=number_samples,
        rollouts_per_prompt=rollouts_per_prompts,
        policy_model=policy_config,
        openai_api_key=OPENAI_KEY,
        test_dataset=TestDataset.HARMLESS_HELPFUL,
        policy_formatter=policy_formatter,
        joint_model=joint_model,
    )

    # plot the results
    scatter_plots: ScatterplotResults = plot_random_reward_evaluations(
        random_rewards_evaluations=evaluations.random_target_rewards
    )

    # save the results to neptune
    log_results_to_neptune(
        scatter_plots=scatter_plots,
        neptune_api_key=NEPTUNE_KEY,
        neptune_project_name=OFFLINE_SEPARATE_POLICY_NEPTUNE_PROJECT,
        neptune_run_id=run_id,
        policy_config=policy_config,
        joint_model=joint_model,
        formatter=policy_formatter,
        number_samples=number_samples,
        rollouts_per_sample=rollouts_per_prompts,
        results=evaluations,
    )
