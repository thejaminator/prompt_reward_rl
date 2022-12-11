from pydantic import BaseModel


class PromptCompletion(BaseModel):
    prompt: str
    completion: str
