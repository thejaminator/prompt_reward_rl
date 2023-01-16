from abc import ABC, abstractmethod
from typing import Any, Optional

from slist import Slist, A

from settings import REWARD_NORMALIZER_NEPTUNE_KEY
from train.neptune_utils.runs import get_neptune_run


class JointRewardNormalizer(ABC):
    @staticmethod
    @abstractmethod
    def from_rewards(rewards: Slist[float]) -> "JointRewardNormalizer":
        raise NotImplementedError

    @abstractmethod
    def normalize_reward(self, reward: float) -> float:
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def create_from_dict(_dict: dict[str, Any]) -> "JointRewardNormalizer":
        name = _dict["name"]
        if name == JointMinMaxNormalizer.name():
            return JointMinMaxNormalizer.from_dict(_dict)
        elif name == JointStandardScaleNormalizer.name():
            return JointStandardScaleNormalizer.from_dict(_dict)
        elif name == JointDoNothingNormalizer.name():
            return JointDoNothingNormalizer()
        else:
            raise ValueError(f"Unknown normalizer name: {name}")


class JointMinMaxNormalizer(JointRewardNormalizer):
    # A normalizer that will normalize the rewards to be between 0 and 1
    def __init__(
        self,
        reward_min: float,
        reward_max: float,
    ):
        self.reward_min = reward_min
        self.reward_max = reward_max

    @staticmethod
    def from_rewards(rewards: Slist[float]) -> "JointMinMaxNormalizer":
        rewards_min: float = min(rewards)
        rewards_max: float = max(rewards)

        return JointMinMaxNormalizer(
            reward_min=rewards_min,
            reward_max=rewards_max,
        )

    def normalize_reward(self, reward: float) -> float:
        return (reward - self.reward_min) / (self.reward_max - self.reward_min)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
        }

    @staticmethod
    def from_dict(_dict: dict[str, Any]) -> "JointMinMaxNormalizer":
        reward_min = _dict["reward_min"]
        reward_max = _dict["reward_max"]
        return JointMinMaxNormalizer(
            reward_min=reward_min,
            reward_max=reward_max,
        )


class JointDoNothingNormalizer(JointRewardNormalizer):
    def __init__(self):
        pass

    @staticmethod
    def from_rewards(rewards: Slist[float]) -> "JointDoNothingNormalizer":
        return JointDoNothingNormalizer()

    def normalize_reward(self, reward: float) -> float:
        return reward

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name()}

    @staticmethod
    def from_dict(_dict: dict[str, Any]) -> "JointDoNothingNormalizer":
        return JointDoNothingNormalizer()


def assert_not_none(value: Optional[A], message: str = "Value should not be None") -> A:
    assert value is not None, message
    return value


class JointStandardScaleNormalizer(JointRewardNormalizer):
    def __init__(
        self,
        mean: float,
        std: float,

    ):
        self.mean = mean
        self.std = std

    @staticmethod
    def from_rewards(rewards: Slist[float]):
        mean: float = assert_not_none(rewards.average())
        std: float = assert_not_none(rewards.standard_deviation())
        return JointStandardScaleNormalizer(
            mean=mean,
            std=std,
        )

    def normalize_reward(self, reward: float) -> float:
        return (reward - self.mean) / self.std

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
            "mean": self.mean,
            "std": self.std,
        }

    @staticmethod
    def from_dict(_dict: dict[str, Any]) -> "JointStandardScaleNormalizer":
        mean = _dict["mean"]
        std = _dict["std"]
        return JointStandardScaleNormalizer(
            mean=mean,
            std=std,
        )


def get_joint_normalizer_from_neptune(
    neptune_api_key: str,
    neptune_project_name: str,
    neptune_run_id: str,
) -> JointRewardNormalizer:
    run = get_neptune_run(
        neptune_api_key=neptune_api_key,
        neptune_project_name=neptune_project_name,
        neptune_run_id=neptune_run_id,
    )
    normalizer_dict = run[REWARD_NORMALIZER_NEPTUNE_KEY].fetch()
    assert normalizer_dict is not None, "Normalizer is None"
    normalizer = JointRewardNormalizer.create_from_dict(normalizer_dict)
    run.stop()
    return normalizer
