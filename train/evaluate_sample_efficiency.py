from typing import List

from pydantic import BaseModel
from slist import Slist

from api.redis_cache import redis_cache
from settings import (
    NEPTUNE_KEY,
    OFFLINE_POLICY_NEPTUNE_PROJECT,
    TRAIN_EXAMPLES_NEPTUNE_KEY,
)
from train.evaluate_offline import get_neptune_run

# Specify neptune run ids you want to plot
# these are runs that should already have been evaluated by train/evaluate_reward_model.py
run_ids: List[str] = ["OF-2", "OF-3", "OF-4", "OF-5", "OF-8"]


class SampleEfficencyMetric(BaseModel):
    training_examples: int
    harmless_correlation: float
    helpful_correlation: float


@redis_cache(decode_dict=SampleEfficencyMetric)
def get_sample_efficiency_metric(
    neptune_api_key: str, neptune_project_name: str, run_id: str
) -> SampleEfficencyMetric:
    run = get_neptune_run(
        neptune_api_key=neptune_api_key,
        neptune_project_name=neptune_project_name,
        neptune_run_id=run_id,
    )
    training_examples: int = int(run[TRAIN_EXAMPLES_NEPTUNE_KEY].fetch())
    harmless_correlation: float = float(
        run["evaluation_temp_1.0/correlation_harmless"].fetch()
    )
    helpful_correlation: float = float(
        run["evaluation_temp_1.0/correlation_helpful"].fetch()
    )
    metric = SampleEfficencyMetric(
        training_examples=training_examples,
        harmless_correlation=harmless_correlation,
        helpful_correlation=helpful_correlation,
    )
    run.stop()
    return metric


if __name__ == "__main__":
    metrics: Slist[SampleEfficencyMetric] = Slist()
    for run_id in run_ids:
        metric = get_sample_efficiency_metric(
            neptune_api_key=NEPTUNE_KEY,
            neptune_project_name=OFFLINE_POLICY_NEPTUNE_PROJECT,
            run_id=run_id,
        )
        metrics.append(metric)
    sorted_metrics = metrics.sort_by(key=lambda metric: metric.training_examples)
    # Plot training_examples as the x-axis and correlation as the y-axis
    import seaborn as sns

    sns.set_theme()
    # Plot harmless correlation as a blue dotted line
    sns.lineplot(
        x=metrics.map(lambda metric: metric.training_examples),
        y=metrics.map(lambda metric: metric.harmless_correlation),
        color="blue",
        label="harmless",
        linestyle="dotted",
    )
    # Plot helpful correlation as a red dottedline
    plot = sns.lineplot(
        data=sorted_metrics,
        x=metrics.map(lambda metric: metric.training_examples),
        y=metrics.map(lambda metric: metric.helpful_correlation),
        color="red",
        label="helpful",
        linestyle="dotted",
    )
    # Only show the x-axis ticks for the present x-axis values
    plot.set(xticks=metrics.map(lambda metric: metric.training_examples))
    plot.set(xlabel="Training Examples", ylabel="Correlation")
    # Save the plot to a file
    plot.figure.savefig("sample_efficiency.png")
