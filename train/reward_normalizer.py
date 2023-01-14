from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel
from slist import Slist, A

from api.prompt_completion import PromptCompletion
from train.evaluate_offline import HelpfulHarmlessEvaluationMetric
from train.reward_models import HelpfulHarmlessReward


class RewardNormalizer(ABC):
    @abstractmethod
    def __init__(self, rewards: Slist[HelpfulHarmlessReward]):
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

    @abstractmethod
    def from_dict(self, d: dict[str, Any]) -> "RewardNormalizer":
        raise NotImplementedError


class MinMaxNormalizer(RewardNormalizer):
    # A normalizer that will normalize the rewards to be between 0 and 1
    def __init__(self, rewards: Slist[HelpfulHarmlessReward]):
        harmless_rewards: Slist[float] = rewards.map(lambda r: r.harmless)
        helpful_rewards: Slist[float] = rewards.map(lambda r: r.helpful)
        self.harmless_min: float = min(harmless_rewards)
        self.harmless_max: float = max(harmless_rewards)
        self.helpful_min: float = min(helpful_rewards)
        self.helpful_max: float = max(helpful_rewards)

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

    def from_dict(self, d: dict[str, Any]) -> "MinMaxNormalizer":
        normalizer = MinMaxNormalizer(
            rewards=Slist(),
        )
        normalizer.harmless_min = d["harmless_min"]
        normalizer.harmless_max = d["harmless_max"]
        normalizer.helpful_min = d["helpful_min"]
        normalizer.helpful_max = d["helpful_max"]
        return normalizer


class DoNothingNormalizer(RewardNormalizer):
    def __init__(self, rewards: Slist[HelpfulHarmlessReward]):
        pass

    def normalize_reward(self, reward: HelpfulHarmlessReward) -> HelpfulHarmlessReward:
        return reward

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name()}

    def from_dict(self, d: dict[str, Any]) -> "DoNothingNormalizer":
        return DoNothingNormalizer(rewards=Slist())

def assert_not_none(value: Optional[A], message: str = "Value should not be None") -> A:
    assert value is not None, message
    return value


class StandardScaleNormalizer(RewardNormalizer):
    def __init__(self, rewards: Slist[HelpfulHarmlessReward]):
        harmless_rewards: Slist[float] = rewards.map(lambda r: r.harmless)
        helpful_rewards: Slist[float] = rewards.map(lambda r: r.helpful)
        self.harmless_mean: float = assert_not_none(harmless_rewards.average())
        self.harmless_std: float = assert_not_none(
            harmless_rewards.standard_deviation()
        )
        self.helpful_mean: float = assert_not_none(helpful_rewards.average())
        self.helpful_std: float = assert_not_none(helpful_rewards.standard_deviation())

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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StandardScaleNormalizer":
        normalizer = StandardScaleNormalizer(
            rewards=Slist(),
        )
        normalizer.harmless_mean = d["harmless_mean"]
        normalizer.harmless_std = d["harmless_std"]
        normalizer.helpful_mean = d["helpful_mean"]
        normalizer.helpful_std = d["helpful_std"]
        return normalizer


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
