from pydantic import BaseModel


class HelpfulHarmlessReward(BaseModel):
    helpful: float
    harmless: float


class ConversationWithReward(BaseModel):
    conversation: str
    reward: HelpfulHarmlessReward
