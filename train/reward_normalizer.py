from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel
from slist import Slist

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


class MinMaxNormalizer(RewardNormalizer):
    # A normalizer that will normalize the rewards to be between 0 and 1
    def __init__(self, rewards: Slist[HelpfulHarmlessReward]):
        self.harmless_rewards: Slist[float] = rewards.map(lambda r: r.harmless)
        self.helpful_rewards: Slist[float] = rewards.map(lambda r: r.helpful)
        self.harmless_min: float = min(self.harmless_rewards)
        self.harmless_max: float = max(self.harmless_rewards)
        self.helpful_min: float = min(self.helpful_rewards)
        self.helpful_max: float = max(self.helpful_rewards)

    def normalize_reward(self, reward: HelpfulHarmlessReward) -> HelpfulHarmlessReward:
        return HelpfulHarmlessReward(
            harmless=self.normalize_harmless(reward.harmless),
            helpful=self.normalize_helpful(reward.helpful),
        )

    def normalize_helpful(self, helpful: float) -> float:
        return (helpful - self.helpful_min) / (self.helpful_max - self.helpful_min)

    def normalize_harmless(self, harmless: float) -> float:
        return (harmless - self.harmless_min) / (self.harmless_max - self.harmless_min)


class DoNothingNormalizer(RewardNormalizer):
    def __init__(self, rewards: Slist[HelpfulHarmlessReward]):
        pass

    def normalize_reward(self, reward: HelpfulHarmlessReward) -> HelpfulHarmlessReward:
        return reward


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
