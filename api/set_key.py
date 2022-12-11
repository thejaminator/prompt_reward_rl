import openai


def set_openai_key(key: str) -> None:
    """Sets OpenAI key."""
    openai.api_key = key
