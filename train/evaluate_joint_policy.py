from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import neptune
import neptune.new
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from neptune.new import Run
from neptune.new.types import File
from openai.error import RateLimitError, APIConnectionError
from pydantic import BaseModel
from retry import retry
from seaborn._core.plot import Plot
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
    OFFLINE_JOINT_POLICY_NEPTUNE_PROJECT,
    ONLINE_POLICY_NEPTUNE_PROJECT,
)
from train.assign_rewards import (
    assign_random_joint_target_reward,
    assign_high_joint_target_reward,
)
from train.evaluate_separate_policy import (
    extend_with_rollout_number,
    ScatterplotResults,
    plot_scatterplot_and_correlation,
    TextWithRolloutNumber,
    plot_linechart,
)
from train.evaluate_reward_model import (
    TestDataset,
    get_helpful_test,
    get_harmless_test,
    get_harmless_helpful_test,
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
    assert_not_none, JointDoNothingNormalizer,
)
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
    normalized_target_reward: float
    actual_reward: float


class JointEvaluationWithGPTResponse(BaseModel):
    metrics: JointEvaluationMetric
    full_response: GPTFullResponse


@retry(exceptions=(RateLimitError, APIConnectionError), tries=5, delay=20)
def get_policy_single_evaluation(
    dialogue_with_reward: DialogueWithJointReward,
    policy_model: OpenaiInferenceConfig,
    joint_reward_model: ModelId,
    normalizer: JointRewardNormalizer,
    prompt_formatter: JointPolicyPromptFormatter,
    cached: bool = True,
) -> JointEvaluationWithGPTResponse:
    normalizd_target_reward: float = normalizer.normalize_reward(
        dialogue_with_reward.target_reward
    )
    new_dialogue_with_reward = dialogue_with_reward.copy()
    new_dialogue_with_reward.target_reward = normalizd_target_reward
    prompt_info: JointPolicyPromptInfo = (
        prompt_formatter.dialogue_reward_to_prompt_completion(
            with_reward=new_dialogue_with_reward
        )
    )
    policy_prompt = prompt_info.to_prompt_completion().prompt
    # rollout the policy
    policy_completion: GPTFullResponse = (
        cached_get_openai_completion(prompt=policy_prompt, config=policy_model)
        if cached
        else get_openai_completion(prompt=policy_prompt, config=policy_model)
    )
    # You need to get the prompt that does not have the target reward, and add the completion
    dialogue = (
        prompt_info.dialogue_without_reward_without_completion
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
        target_reward=dialogue_with_reward.target_reward,
        actual_reward=actual_reward,
        normalized_target_reward=normalizd_target_reward,
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
    joint_reward_model: ModelId,
    openai_api_key: str,
    test_dataset: TestDataset,
    rollouts_per_prompt: int,
    policy_formatter: JointPolicyPromptFormatter,
    normalizer: JointRewardNormalizer,
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
    # Rollout the prompts to get the completions, and get the actual rewards
    evaluated_random_prompts: Slist[
        JointEvaluationMetric
    ] = sample_dialogue_with_random_rewards.par_map(
        lambda x: get_policy_single_evaluation(
            dialogue_with_reward=x,
            policy_model=policy_model,
            joint_reward_model=joint_reward_model,
            normalizer=normalizer,
            prompt_formatter=policy_formatter,
        ).metrics,
        executor=threadpool,
    )

    # do the same thing, but using the maximum target reward
    sample_dialogue_with_maximum_rewards: Slist[
        DialogueWithJointReward
    ] = sample_dialogue_upscaled.map(
        lambda text_with_rollout: assign_high_joint_target_reward(
            dialogue=text_with_rollout.text
        )
    )
    evaluated_maximum_prompts: Slist[
        JointEvaluationMetric
    ] = sample_dialogue_with_maximum_rewards.par_map(
        lambda x: get_policy_single_evaluation(
            dialogue_with_reward=x,
            policy_model=policy_model,
            joint_reward_model=joint_reward_model,
            normalizer=normalizer,
            prompt_formatter=policy_formatter,
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
    target: Slist[float] = random_rewards_evaluations.map(lambda x: x.target_reward)
    actual: Slist[float] = random_rewards_evaluations.map(lambda x: x.actual_reward)
    results: ScatterplotResults = plot_scatterplot_and_correlation(
        y=actual,
        x=target,
        title="Joint reward",
        ylabel="Actual",
        xlabel="Target",
    )
    return results


def to_nearest_zero_point_zero_five(x: float) -> float:
    assert x >= 0
    assert x <= 1
    return (round(x * 100 / 5) * 5) / 100


def plot_random_reward_evaluations_like_paper(
    random_rewards_evaluations: Slist[JointEvaluationMetric],
    one_percentile: float,  # calculated from offline dataset
    ninety_nine_percentile: float,  # calculated from offline dataset
) -> Figure:
    # Try to create a plot like the decision transformer's paper
    tups: Slist[
        tuple[float, Slist[JointEvaluationMetric]]
    ] = random_rewards_evaluations.group_by(
        lambda x: to_nearest_zero_point_zero_five(x.target_reward)
    )
    averaged_groups: Slist[tuple[float, float]] = tups.map(
        lambda x: (x[0], assert_not_none(x[1].map(lambda y: y.actual_reward).average()))
    )
    target = averaged_groups.map(lambda x: x[0])
    actual = averaged_groups.map(lambda x: x[1])
    # in the paper, target is x, actual is y
    plot: Axes = plot_linechart(
        x=target,
        y=actual,
        title="Actual vs Target Reward",
        xlabel="Target Reward",
        ylabel="Actual Reward",
    )
    # Add a vertical brown dashed line at ninety_nine_percentile
    plot.axvline(
        ninety_nine_percentile,
        color="brown",
        linestyle="dashed",
        label="99th percentile in train",
    )
    # Add a vertical gray dashed line at one_percentile
    plot.axvline(
        one_percentile,
        color="gray",
        linestyle="dashed",
        label="1st percentile in train",
    )
    # Add legends
    plot.legend()
    # write the plot to a file
    plot.figure.savefig(f"Actual vs Target Reward.png")
    return plot.figure


def log_results_to_neptune(
    scatter_plots: ScatterplotResults,
    paper_plots: Figure,
    neptune_api_key: str,
    neptune_project_name: str,
    neptune_run_id: str,
    policy_config: OpenaiInferenceConfig,
    joint_reward_model: ModelId,
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
    print(f"Average reward for maximum target reward: {average_reward}")

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
        run[f"{evaluation_key}/joint_model"] = joint_reward_model
        run[f"{evaluation_key}/policy_formatter"] = formatter.name
        run[f"{evaluation_key}/number_samples"] = number_samples
        run[f"{evaluation_key}/rollouts_per_sample"] = rollouts_per_sample
        # Log the results under evaluation
        run[f"{evaluation_key}/correlation"] = scatter_plots.correlation
        run[f"{evaluation_key}/pvalue"] = scatter_plots.p_value
        # upper bound
        run[f"{evaluation_key}/upper_bound"] = scatter_plots.upper_correlation_bound
        # lower bound
        run[f"{evaluation_key}/lower_bound"] = scatter_plots.lower_correlation_bound
        # confidence
        run[f"{evaluation_key}/confidence_level"] = scatter_plots.confidence_level

        # Log the average reward for maximum target reward
        run[f"{evaluation_key}/average_joint_maximum_target"] = average_reward
        # Log the plots
        run[f"{evaluation_key}/joint_plot"].upload(scatter_plots.figure)
        run[f"{evaluation_key}/plot_like_paper"].upload(paper_plots)

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


if __name__ == "__main__":
    # Optionally retrieve the openai model id from neptune
    run_id = "OF1-14"
    project_name = OFFLINE_JOINT_POLICY_NEPTUNE_PROJECT
    policy_model_id = get_openai_model_from_neptune(
        neptune_api_key=NEPTUNE_KEY,
        neptune_project_name=project_name,
        neptune_run_id=run_id,
    )
    # Optionally retrieve the normalizer from neptune
    normalizer = JointDoNothingNormalizer()
    policy_config = OpenaiInferenceConfig(
        model=policy_model_id,  # You can set this manually too
        temperature=1.0,  # try 0.6, 1.0
        max_tokens=400,
        top_p=1.0,
        stop=END_TOKEN,
    )
    joint_reward_model = ModelId(
        "babbage:ft-leadiq:helpful-reward-2022-12-22-08-04-46"
    )
    policy_formatter = JointRewardAtBottomFormatter()
    number_samples = 500
    rollouts_per_prompts = 1
    evaluations: EvaluationResults = run_evaluation(
        sample_prompts=number_samples,
        rollouts_per_prompt=rollouts_per_prompts,
        policy_model=policy_config,
        openai_api_key=OPENAI_KEY,
        test_dataset=TestDataset.HELPFUL,
        policy_formatter=policy_formatter,
        joint_reward_model=joint_reward_model,
        normalizer=normalizer,
    )

    # plot the results
    scatter_plots: ScatterplotResults = plot_random_reward_evaluations(
        random_rewards_evaluations=evaluations.random_target_rewards
    )
    paper_correlation_plot = plot_random_reward_evaluations_like_paper(
        random_rewards_evaluations=evaluations.random_target_rewards,
        one_percentile=0.30,
        ninety_nine_percentile=0.76,
    )

    # save the results to neptune
    log_results_to_neptune(
        scatter_plots=scatter_plots,
        neptune_api_key=NEPTUNE_KEY,
        neptune_project_name=project_name,
        neptune_run_id=run_id,
        policy_config=policy_config,
        joint_reward_model=joint_reward_model,
        formatter=policy_formatter,
        number_samples=number_samples,
        rollouts_per_sample=rollouts_per_prompts,
        results=evaluations,
        paper_plots=paper_correlation_plot,
    )
