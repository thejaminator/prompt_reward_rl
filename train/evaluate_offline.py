import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import neptune
import neptune.new
from neptune.new import Run
from neptune.new.types import File

import pandas as pd
import seaborn as sns
from openai.error import RateLimitError
from pydantic import BaseModel
from retry import retry
from scipy.stats import pearsonr, _result_classes
from scipy.stats._common import ConfidenceInterval
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
)
from parsing.parse_raw import AnthropicRawFormat
from settings import (
    OPENAI_KEY,
    NEPTUNE_KEY,
    OFFLINE_POLICY_NEPTUNE_PROJECT,
    MODEL_ID_NEPTUNE_KEY,
)
from train.assign_rewards import assign_random_target_reward, assign_maximum_target_reward
from train.evaluate_reward_model import (
    TestDataset,
    get_harmless_helpful_test,
    get_harmless_test,
    get_helpful_test,
)
from train.policy_prompt_formatter import (
    PolicyPromptFormatter,
    PolicyPromptInfo,
    RewardAtBottomFormatter,
)
from train.reward_models import DialogueWithReward
from train.separators import END_TOKEN


class HelpfulHarmlessEvaluationMetrics(BaseModel):
    policy_prompt: str
    # prompt given to the reward model for determining the actual reward
    reward_prompt: str
    completion: str
    # Prompt + completion, without the reward
    completed_dialogue: str
    target_helpful: float
    actual_helpful: float
    target_harmless: float
    actual_harmless: float

class HelpfulHarmlessEvaluation(BaseModel):
    metrics: HelpfulHarmlessEvaluationMetrics
    full_response: GPTFullResponse


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
    metrics= HelpfulHarmlessEvaluationMetrics(
        policy_prompt=policy_prompt,
        reward_prompt=formatted_reward_prompt,
        completion=policy_completion.completion,
        completed_dialogue=dialogue,
        target_helpful=policy_prompt_info.target_reward.helpful,
        actual_helpful=actual_helpful_reward,
        target_harmless=policy_prompt_info.target_reward.harmless,
        actual_harmless=actual_harmless_reward,
    )
    return HelpfulHarmlessEvaluation(
        metrics=metrics,
        full_response=policy_completion,
    )


class TextWithRolloutNumber(BaseModel):
    text: str
    rollout_number: int


def extend_with_rollout_number(
    text: str, rollouts: int
) -> Slist[TextWithRolloutNumber]:
    # Add the rollout_number so that the random seed is different for each rollout
    # The target reward generated will be the same between runs
    # So now you can cache the rollout
    return Slist(
        [TextWithRolloutNumber(text=text, rollout_number=i) for i in range(rollouts)]
    )


@dataclasses.dataclass
class ScatterplotResults:
    figure: Figure
    correlation: float
    p_value: float
    confidence_level: float
    upper_correlation_bound: float
    lower_correlation_bound: float


def plot_scatterplot_and_correlation(
    x: List[float], y: List[float], title: str, xlabel: str, ylabel: str
) -> ScatterplotResults:
    # clear seaborn
    sns.reset_orig()
    f, axes = plt.subplots(1)
    # use seaborn style defaults and set the default figure size
    sns.set(rc={"figure.figsize": (8, 8)})
    # use x and y as the data, assign to the variables called x and y
    # use the function regplot to make a scatterplot
    # color the scatterplot points blue
    # pass axes so you don't overwrite the same plots
    plot = sns.regplot(x=x, y=y, color="b", line_kws={"color": "red"}, ax=axes)
    # add a (1, 1) line to show perfect correlation
    plot.plot([0, 1], [0, 1], transform=plot.transAxes, ls="--", c=".3")
    # Calculate the correlation coefficient between x and y
    pearson: _result_classes.PearsonRResult = pearsonr(
        x=x,
        y=y,
    )
    confidence_level = 0.95
    confidence_interval: ConfidenceInterval = pearson.confidence_interval(
        confidence_level=confidence_level
    )
    lower_bound = confidence_interval[0]
    upper_bound = confidence_interval[1]
    correlation: float = pearson.statistic
    pvalue: float = pearson.pvalue

    # set a title for the regplot
    title_with_statistics = f"{title} Correlation: {correlation:.2f}, [{lower_bound:.2f}, {upper_bound:.2f}]"
    plot.figure.suptitle(title_with_statistics)
    # set the labels for the x and y axes
    plot.set(xlabel=xlabel, ylabel=ylabel)
    # set the x and y axis to (0, 1)
    plot.set(xlim=(0, 1), ylim=(0, 1))
    figure = plot.figure

    return ScatterplotResults(
        figure=figure,
        correlation=correlation,
        p_value=pvalue,
        confidence_level=confidence_level,
        upper_correlation_bound=upper_bound,
        lower_correlation_bound=lower_bound,
    )


class EvaluationResults(BaseModel):
    random_target_rewards: SlistPydantic[HelpfulHarmlessEvaluationMetrics]
    maximum_target_rewards: SlistPydantic[HelpfulHarmlessEvaluationMetrics]


def run_evaluation(
    sample_prompts: int,
    policy_model: OpenaiInferenceConfig,
    helpful_model: ModelId,
    harmless_model: ModelId,
    openai_api_key: str,
    test_dataset: TestDataset,
    rollouts_per_prompt: int,
    policy_formatter: PolicyPromptFormatter,
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
        DialogueWithReward
    ] = sample_dialogue_upscaled.map(
        lambda text_with_rollout: assign_random_target_reward(
            dialogue=text_with_rollout.text,
            rollout_number=text_with_rollout.rollout_number,
        )
    )
    # Use policy_formatter to format the prompts
    formatted_random_prompts: Slist[
        PolicyPromptInfo
    ] = sample_dialogue_with_random_rewards.map(
        lambda x: policy_formatter.dialogue_reward_to_prompt_completion(x)
    )

    # Rollout the prompts to get the completions, and get the actual rewards
    evaluated_random_prompts: Slist[
        HelpfulHarmlessEvaluationMetrics
    ] = formatted_random_prompts.par_map(
        lambda x: get_policy_single_evaluation(
            policy_prompt_info=x,
            policy_model=policy_model,
            helpful_model=helpful_model,
            harmless_model=harmless_model,
        ).metrics,
        executor=threadpool,
    )

    # do the same thing, but using the maximum target reward
    sample_dialogue_with_maximum_rewards: Slist[
        DialogueWithReward
    ] = sample_dialogue_upscaled.map(
        lambda text_with_rollout: assign_maximum_target_reward(
            dialogue=text_with_rollout.text
        )
    )
    formatted_maximum_prompts: Slist[
        PolicyPromptInfo
    ] = sample_dialogue_with_maximum_rewards.map(
        lambda x: policy_formatter.dialogue_reward_to_prompt_completion(x)
    )
    evaluated_maximum_prompts: Slist[
        HelpfulHarmlessEvaluationMetrics
    ] = formatted_maximum_prompts.par_map(
        lambda x: get_policy_single_evaluation(
            policy_prompt_info=x,
            policy_model=policy_model,
            helpful_model=helpful_model,
            harmless_model=harmless_model,
        ).metrics,
        executor=threadpool,
    )

    return EvaluationResults(
        random_target_rewards=evaluated_random_prompts,
        maximum_target_rewards=evaluated_maximum_prompts,
    )


@dataclasses.dataclass
class HelpfulHarmlessScatterplots:
    helpful: ScatterplotResults
    harmless: ScatterplotResults


def plot_random_reward_evaluations(
    random_rewards_evaluations: Slist[HelpfulHarmlessEvaluationMetrics],
) -> HelpfulHarmlessScatterplots:
    # Calculate correlation coefficient of harmless
    target_harmless: Slist[float] = random_rewards_evaluations.map(
        lambda x: x.target_harmless
    )
    actual_harmless: Slist[float] = random_rewards_evaluations.map(
        lambda x: x.actual_harmless
    )

    target_helpful: Slist[float] = random_rewards_evaluations.map(
        lambda x: x.target_helpful
    )
    actual_helpful: Slist[float] = random_rewards_evaluations.map(
        lambda x: x.actual_helpful
    )

    harmless_results: ScatterplotResults = plot_scatterplot_and_correlation(
        x=actual_harmless,
        y=target_harmless,
        title="Harmless",
        xlabel="Actual",
        ylabel="Target",
    )

    helpful_results = plot_scatterplot_and_correlation(
        x=actual_helpful,
        y=target_helpful,
        title="Helpful",
        xlabel="Actual",
        ylabel="Target",
    )
    return HelpfulHarmlessScatterplots(
        helpful=helpful_results, harmless=harmless_results
    )


def log_results_to_neptune(
    scatter_plots: HelpfulHarmlessScatterplots,
    neptune_api_key: str,
    neptune_project_name: str,
    neptune_run_id: str,
    policy_config: OpenaiInferenceConfig,
    helpful_model: ModelId,
    harmless_model: ModelId,
    formatter: PolicyPromptFormatter,
    number_samples: int,
    rollouts_per_sample: int,
    results: EvaluationResults,
) -> None:
    temperature = policy_config.temperature
    # add the temperature to the evaluation_key
    evaluation_key = f"evaluation_temp_{temperature}"
    helpful_results = scatter_plots.helpful
    harmless_results = scatter_plots.harmless

    # Print the results
    print(f"Correlation of Harmless: {harmless_results.correlation}")
    print(f"Correlation of Helpful: {helpful_results.correlation}")

    # Calculate the average reward when using the maximum target reward
    average_helpless_reward: Optional[float] = results.maximum_target_rewards.map(
        lambda x: x.actual_harmless
    ).average()
    average_helpful_reward: Optional[float] = results.maximum_target_rewards.map(
        lambda x: x.actual_helpful
    ).average()
    # print
    print(
        f"Average Harmless Reward for maximum target reward: {average_helpless_reward}"
    )
    print(f"Average Helpful Reward for maximum target reward: {average_helpful_reward}")

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
        run[f"{evaluation_key}/helpful_model"] = helpful_model
        run[f"{evaluation_key}/harmless_model"] = harmless_model
        run[f"{evaluation_key}/policy_formatter"] = formatter.name
        run[f"{evaluation_key}/number_samples"] = number_samples
        run[f"{evaluation_key}/rollouts_per_sample"] = rollouts_per_sample
        # Log the results under evaluation
        run[f"{evaluation_key}/correlation_harmless"] = harmless_results.correlation
        run[f"{evaluation_key}/pvalue_harmless"] = harmless_results.p_value
        # upper bound
        run[
            f"{evaluation_key}/upper_bound_harmless"
        ] = harmless_results.upper_correlation_bound
        # lower bound
        run[
            f"{evaluation_key}/lower_bound_harmless"
        ] = harmless_results.lower_correlation_bound
        # confidence
        run[
            f"{evaluation_key}/confidence_level_harmless"
        ] = harmless_results.confidence_level
        # Same thing for helpful
        run[f"{evaluation_key}/correlation_helpful"] = helpful_results.correlation
        run[f"{evaluation_key}/pvalue_helpful"] = helpful_results.p_value
        # upper bound
        run[
            f"{evaluation_key}/upper_bound_helpful"
        ] = helpful_results.upper_correlation_bound
        # lower bound
        run[
            f"{evaluation_key}/lower_bound_helpful"
        ] = helpful_results.lower_correlation_bound
        # confidence
        run[
            f"{evaluation_key}/confidence_level_helpful"
        ] = helpful_results.confidence_level
        # Log the average reward for maximum target reward
        run[
            f"{evaluation_key}/average_harmless_reward_maximum_target"
        ] = average_helpless_reward
        run[
            f"{evaluation_key}/average_helpful_reward_maximum_target"
        ] = average_helpful_reward
        # Log the plots
        run[f"{evaluation_key}/helpful_plot"].upload(helpful_results.figure)
        run[f"{evaluation_key}/harmless_plot"].upload(harmless_results.figure)
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
    run_id = "OF-8"
    policy_model_id = get_openai_model_from_neptune(
        neptune_api_key=NEPTUNE_KEY,
        neptune_project_name=OFFLINE_POLICY_NEPTUNE_PROJECT,
        neptune_run_id=run_id,
    )
    policy_config = OpenaiInferenceConfig(
        model=policy_model_id,  # You can set this manually too
        temperature=1.0,  # try 0.6, 1.0
        max_tokens=400,
        top_p=1.0,
        stop=[END_TOKEN, "\n\n"],
    )
    helpful_model = ModelId("babbage:ft-leadiq:helpful-reward-2022-12-22-08-04-46")
    harmless_model = ModelId("babbage:ft-leadiq:harmless-reward-2022-12-22-08-55-12")
    policy_formatter = RewardAtBottomFormatter()
    number_samples = 500
    rollouts_per_prompts = 1
    evaluations: EvaluationResults = run_evaluation(
        sample_prompts=number_samples,
        rollouts_per_prompt=rollouts_per_prompts,
        policy_model=policy_config,
        openai_api_key=OPENAI_KEY,
        test_dataset=TestDataset.HARMLESS_HELPFUL,
        policy_formatter=policy_formatter,
        helpful_model=helpful_model,
        harmless_model=harmless_model,
    )

    # plot the results
    scatter_plots: HelpfulHarmlessScatterplots = plot_random_reward_evaluations(
        random_rewards_evaluations=evaluations.random_target_rewards
    )

    # save the results to neptune
    log_results_to_neptune(
        scatter_plots=scatter_plots,
        neptune_api_key=NEPTUNE_KEY,
        neptune_project_name=OFFLINE_POLICY_NEPTUNE_PROJECT,
        neptune_run_id=run_id,
        policy_config=policy_config,
        helpful_model=helpful_model,
        harmless_model=harmless_model,
        formatter=policy_formatter,
        number_samples=number_samples,
        rollouts_per_sample=rollouts_per_prompts,
        results=evaluations,
    )
    """150k samples offline
    Target 0.99
    Average actual helpfulness: 0.5515753179993419
    Average actual harmlessness: 0.4696473063947702
    
    Target 1.00
    Average actual helpfulness: 0.42767436214852766
    Average actual harmlessness: 0.5496383727934258
    
    """

    """Control av. reward for vanilla babbage
    Average actual helpfulness: 0.45239336682231596
    Average actual harmlessness: 0.4870573758367717 
    """
