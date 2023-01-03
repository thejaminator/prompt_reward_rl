from pydantic import BaseModel


class HelpfulHarmlessReward(BaseModel):
    helpful: float
    harmless: float


class DialogueWithReward(BaseModel):
    dialogue: str
    target_reward: HelpfulHarmlessReward

class DialogueWithJointReward(BaseModel):
    dialogue: str
    target_reward: float
