from pydantic import BaseModel

from train.rewards import HelpfulHarmlessReward


class HelpfulHarmlessEvaluationMetric(BaseModel):
    policy_prompt: str
    # prompt given to the reward model for determining the actual reward
    reward_prompt: str
    completion: str
    # Prompt + completion, without the reward
    completed_dialogue: str
    target_helpful: float
    normalized_target_helpful: float
    actual_helpful: float # unnormalized

    target_harmless: float
    normalized_target_harmless: float
    actual_harmless: float # unnormalized

    @property
    def actual_rewards(self) -> HelpfulHarmlessReward:
        return HelpfulHarmlessReward(
            helpful=self.actual_helpful,
            harmless=self.actual_harmless,
        )
