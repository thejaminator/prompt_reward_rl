from typing import List, Callable

from matplotlib import pyplot as plt
from pydantic import BaseModel
from slist import Slist

from api.redis_cache import redis_cache
from settings import (
    NEPTUNE_KEY,
    OFFLINE_POLICY_NEPTUNE_PROJECT,
    TRAIN_EXAMPLES_NEPTUNE_KEY,
)
from train.evaluate_offline import get_neptune_run
import seaborn as sns

# Specify neptune run ids you want to plot
# these are runs that should already have been evaluated by train/evaluate_reward_model.py
one_epoch_run_ids: List[str] = ["OF-2", "OF-3", "OF-4", "OF-5", "OF-8"]


class ModelCorrelationMetric(BaseModel):
    training_examples: int
    epochs: int
    total_training_examples: int  # epochs * training_examples
    harmless_correlation: float
    helpful_correlation: float


@redis_cache(decode_dict=ModelCorrelationMetric)
def get_sample_efficiency_metric(
    neptune_api_key: str, neptune_project_name: str, run_id: str
) -> ModelCorrelationMetric:
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
    epochs: int = int(run["parameters/n_epochs"].fetch())
    metric = ModelCorrelationMetric(
        training_examples=training_examples,
        harmless_correlation=harmless_correlation,
        helpful_correlation=helpful_correlation,
        epochs=epochs,
        total_training_examples=epochs * training_examples,
    )
    run.stop()
    return metric


def plot_training_examples_sample_efficiency() -> None:
    metrics: Slist[ModelCorrelationMetric] = Slist()
    for run_id in one_epoch_run_ids:
        metric = get_sample_efficiency_metric(
            neptune_api_key=NEPTUNE_KEY,
            neptune_project_name=OFFLINE_POLICY_NEPTUNE_PROJECT,
            run_id=run_id,
        )
        metrics.append(metric)
    sorted_metrics = metrics.sort_by(key=lambda metric: metric.training_examples)
    # Plot training_examples as the x-axis and correlation as the y-axis

    sns.set_theme()
    f, axes = plt.subplots(1)
    # Plot harmless correlation as a blue dotted line
    sns.lineplot(
        x=metrics.map(lambda metric: metric.training_examples),
        y=metrics.map(lambda metric: metric.harmless_correlation),
        color="blue",
        label="harmless",
        linestyle="dotted",
        axes=axes,
    )
    # Plot helpful correlation as a red dottedline
    plot = sns.lineplot(
        data=sorted_metrics,
        x=metrics.map(lambda metric: metric.training_examples),
        y=metrics.map(lambda metric: metric.helpful_correlation),
        color="red",
        label="helpful",
        linestyle="dotted",
        axes=axes,
    )
    # Only show the x-axis ticks for the present x-axis values
    plot.set(xticks=metrics.map(lambda metric: metric.training_examples))
    plot.set(xlabel="Training Examples", ylabel="Correlation")
    # Save the plot to a file
    plot.figure.savefig("sample_efficiency_n_training_examples.png")


def plot_metric_comparison(
    multiple_epoch_metrics: Slist[ModelCorrelationMetric],
    one_epoch_comparison_metrics: Slist[ModelCorrelationMetric],
    chosen_metric: Callable[[ModelCorrelationMetric], float],
    metric_name: str,
) -> None:
    # Plot the total_training_examples as the x-axis and correlation as the y-axis
    # We'll first plot the harmless correlation in one plot
    sns.set_theme()
    f, axes = plt.subplots(1)
    # Plot multiple epoch metrics as a blue dotted line
    sns.lineplot(
        x=multiple_epoch_metrics.map(lambda metric: metric.total_training_examples),
        y=multiple_epoch_metrics.map(chosen_metric),
        color="blue",
        label="Multiple epochs on 50k unique examples",
        linestyle="dotted",
        ax=axes,
    )
    # Plot one epoch metrics as a red dotted line
    plot = sns.lineplot(
        x=one_epoch_comparison_metrics.map(
            lambda metric: metric.total_training_examples
        ),
        y=one_epoch_comparison_metrics.map(chosen_metric),
        color="red",
        label="1 epoch on unique examples",
        linestyle="dotted",
        ax=axes,
    )
    # Only show the x-axis ticks for the present x-axis values
    plot.set(
        xticks=one_epoch_comparison_metrics.map(
            lambda metric: metric.total_training_examples
        )
    )
    # Set the y-axis scale to be between 0 and 1
    plot.set(ylim=(0, 1))
    plot.set(xlabel="Training Examples", ylabel="Correlation")
    # Set title to be "Harmless"
    plot.set_title(metric_name)
    # Save the plot to a file
    plot.figure.savefig(f"sample_efficiency_n_epochs_{metric_name}.png")


def plot_epochs_sample_efficiency() -> None:
    # 50k, 100k, 150k training samples
    multiple_epoch_run_ids: Slist[str] = Slist(["OF-5", "OF-6", "OF-9"])
    one_epoch_comparison_run_ids: Slist[str] = Slist(["OF-5", "OF-4", "OF-8"])
    # get metrics for the two sets of runs
    multiple_epoch_metrics: Slist[ModelCorrelationMetric] = multiple_epoch_run_ids.map(
        lambda run_id: get_sample_efficiency_metric(
            neptune_api_key=NEPTUNE_KEY,
            neptune_project_name=OFFLINE_POLICY_NEPTUNE_PROJECT,
            run_id=run_id,
        )
    )
    one_epoch_comparison_metrics: Slist[
        ModelCorrelationMetric
    ] = one_epoch_comparison_run_ids.map(
        lambda run_id: get_sample_efficiency_metric(
            neptune_api_key=NEPTUNE_KEY,
            neptune_project_name=OFFLINE_POLICY_NEPTUNE_PROJECT,
            run_id=run_id,
        )
    )
    plot_metric_comparison(
        multiple_epoch_metrics=multiple_epoch_metrics,
        one_epoch_comparison_metrics=one_epoch_comparison_metrics,
        chosen_metric=lambda metric: metric.harmless_correlation,
        metric_name="Harmless",
    )
    plot_metric_comparison(
        multiple_epoch_metrics=multiple_epoch_metrics,
        one_epoch_comparison_metrics=one_epoch_comparison_metrics,
        chosen_metric=lambda metric: metric.helpful_correlation,
        metric_name="Helpful",
    )


if __name__ == "__main__":
    # plot_training_examples_sample_efficiency()
    plot_epochs_sample_efficiency()
