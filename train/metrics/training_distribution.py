from typing import Optional

from pydantic import BaseModel
from slist import Slist, identity


class TrainingDistributionStatistic(BaseModel):
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    one_percentile: Optional[float]
    five_percentile: Optional[float]
    twenty_five_percentile: Optional[float]
    seventy_five_percentile: Optional[float]
    ninety_five_percentile: Optional[float]
    ninety_nine_percentile: Optional[float]
    count: int


def calculate_training_distribution_statistics(
    training_distributions: Slist[float],
) -> TrainingDistributionStatistic:
    return TrainingDistributionStatistic(
        min=training_distributions.min_by(identity),
        max=training_distributions.max_by(identity),
        mean=training_distributions.average(),
        median=training_distributions.median_by(identity),
        std=training_distributions.standard_deviation(),
        one_percentile=training_distributions.percentile_by(identity, 0.01),
        five_percentile=training_distributions.percentile_by(identity, 0.05),
        twenty_five_percentile=training_distributions.percentile_by(identity, 0.25),
        seventy_five_percentile=training_distributions.percentile_by(identity, 0.75),
        ninety_five_percentile=training_distributions.percentile_by(identity, 0.95),
        ninety_nine_percentile=training_distributions.percentile_by(identity, 0.99),
        count=len(training_distributions),
    )
