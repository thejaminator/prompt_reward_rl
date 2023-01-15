from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel
from slist import Slist, A

from api.prompt_completion import PromptCompletion
from train.metrics.reward_metric import HelpfulHarmlessEvaluationMetric
from train.reward_models import HelpfulHarmlessReward


class RewardNormalizer(ABC):
    @staticmethod
    @abstractmethod
    def from_rewards(rewards: Slist[HelpfulHarmlessReward]) -> "RewardNormalizer":
        raise NotImplementedError

    @abstractmethod
    def normalize_reward(self, reward: HelpfulHarmlessReward) -> HelpfulHarmlessReward:
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


    @staticmethod
    def create_from_dict(_dict: dict[str, Any]) -> "RewardNormalizer":
        name = _dict["name"]
        if name == MinMaxNormalizer.name():
            return MinMaxNormalizer.from_dict(_dict)
        elif name == StandardScaleNormalizer.name():
            return StandardScaleNormalizer.from_dict(_dict)
        elif name == DoNothingNormalizer.name():
            return DoNothingNormalizer()
        else:
            raise ValueError(f"Unknown normalizer name: {name}")


class MinMaxNormalizer(RewardNormalizer):
    # A normalizer that will normalize the rewards to be between 0 and 1
    def __init__(
        self,
        harmless_min: float,
        harmless_max: float,
        helpful_min: float,
        helpful_max: float,
    ):
        self.harmless_min = harmless_min
        self.harmless_max = harmless_max
        self.helpful_min = helpful_min
        self.helpful_max = helpful_max

    @staticmethod
    def from_rewards(rewards: Slist[HelpfulHarmlessReward])-> "MinMaxNormalizer":
        harmless_rewards: Slist[float] = rewards.map(lambda r: r.harmless)
        helpful_rewards: Slist[float] = rewards.map(lambda r: r.helpful)
        harmless_min: float = min(harmless_rewards)
        harmless_max: float = max(harmless_rewards)
        helpful_min: float = min(helpful_rewards)
        helpful_max: float = max(helpful_rewards)
        return MinMaxNormalizer(
            harmless_min=harmless_min,
            harmless_max=harmless_max,
            helpful_min=helpful_min,
            helpful_max=helpful_max,
        )

    def normalize_reward(self, reward: HelpfulHarmlessReward) -> HelpfulHarmlessReward:
        return HelpfulHarmlessReward(
            harmless=self.normalize_harmless(reward.harmless),
            helpful=self.normalize_helpful(reward.helpful),
        )

    def normalize_helpful(self, helpful: float) -> float:
        return (helpful - self.helpful_min) / (self.helpful_max - self.helpful_min)

    def normalize_harmless(self, harmless: float) -> float:
        return (harmless - self.harmless_min) / (self.harmless_max - self.harmless_min)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
            "harmless_min": self.harmless_min,
            "harmless_max": self.harmless_max,
            "helpful_min": self.helpful_min,
            "helpful_max": self.helpful_max,
        }

    @staticmethod
    def from_dict(_dict: dict[str, Any]) -> "MinMaxNormalizer":
        harmless_min = _dict["harmless_min"]
        harmless_max = _dict["harmless_max"]
        helpful_min = _dict["helpful_min"]
        helpful_max = _dict["helpful_max"]
        return MinMaxNormalizer(
            harmless_min=harmless_min,
            harmless_max=harmless_max,
            helpful_min=helpful_min,
            helpful_max=helpful_max,
        )


class DoNothingNormalizer(RewardNormalizer):
    def __init__(self):
        pass

    @staticmethod
    def from_rewards(rewards: Slist[HelpfulHarmlessReward]) -> "DoNothingNormalizer":
        return DoNothingNormalizer()
    def normalize_reward(self, reward: HelpfulHarmlessReward) -> HelpfulHarmlessReward:
        return reward

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name()}

    @staticmethod
    def from_dict(_dict: dict[str, Any]) -> "DoNothingNormalizer":
        return DoNothingNormalizer()


def assert_not_none(value: Optional[A], message: str = "Value should not be None") -> A:
    assert value is not None, message
    return value


class StandardScaleNormalizer(RewardNormalizer):
    def __init__(
        self,
        harmless_mean: float,
        harmless_std: float,
        helpful_mean: float,
        helpful_std: float,
    ):
        self.harmless_mean = harmless_mean
        self.harmless_std = harmless_std
        self.helpful_mean = helpful_mean
        self.helpful_std = helpful_std

    @staticmethod
    def from_rewards(rewards: Slist[HelpfulHarmlessReward]):
        harmless_rewards: Slist[float] = rewards.map(lambda r: r.harmless)
        helpful_rewards: Slist[float] = rewards.map(lambda r: r.helpful)
        harmless_mean: float = assert_not_none(harmless_rewards.average())
        harmless_std: float = assert_not_none(harmless_rewards.standard_deviation())
        helpful_mean: float = assert_not_none(helpful_rewards.average())
        helpful_std: float = assert_not_none(helpful_rewards.standard_deviation())
        return StandardScaleNormalizer(
            harmless_mean=harmless_mean,
            harmless_std=harmless_std,
            helpful_mean=helpful_mean,
            helpful_std=helpful_std,
        )

    def normalize_reward(self, reward: HelpfulHarmlessReward) -> HelpfulHarmlessReward:
        return HelpfulHarmlessReward(
            harmless=self.normalize_harmless(reward.harmless),
            helpful=self.normalize_helpful(reward.helpful),
        )

    def normalize_helpful(self, helpful: float) -> float:
        return (helpful - self.helpful_mean) / self.helpful_std

    def normalize_harmless(self, harmless: float) -> float:
        return (harmless - self.harmless_mean) / self.harmless_std

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
            "harmless_mean": self.harmless_mean,
            "harmless_std": self.harmless_std,
            "helpful_mean": self.helpful_mean,
            "helpful_std": self.helpful_std,
        }

    @staticmethod
    def from_dict(_dict: dict[str, Any]) -> "StandardScaleNormalizer":
        harmless_mean = _dict["harmless_mean"]
        harmless_std = _dict["harmless_std"]
        helpful_mean = _dict["helpful_mean"]
        helpful_std = _dict["helpful_std"]
        return StandardScaleNormalizer(
            harmless_mean=harmless_mean,
            harmless_std=harmless_std,
            helpful_mean=helpful_mean,
            helpful_std=helpful_std,
        )


class OnlineTrainingData(BaseModel):
    rollout_metric: HelpfulHarmlessEvaluationMetric
    normalized_helpful: float
    normalized_harmless: float
    training_prompt: str
    training_completion: str

    def to_prompt_completion(self) -> PromptCompletion:
        return PromptCompletion(
            prompt=self.training_prompt,
            completion=self.training_completion,
        )

    def to_flattened_dict(self) -> dict[str, Any]:
        rollout_metric_dict = self.rollout_metric.dict()
        more_data = {
            "normalized_helpful": self.normalized_helpful,
            "normalized_harmless": self.normalized_harmless,
            "training_prompt": self.training_prompt,
            "training_completion": self.training_completion,
        }
        return {**rollout_metric_dict, **more_data}
