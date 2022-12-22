from pydantic import BaseModel


class HelpfulHarmlessReward(BaseModel):
    helpful: float
    harmless: float
