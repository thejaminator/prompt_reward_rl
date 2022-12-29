import os

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

OPENAI_KEY = os.getenv("OPENAI_KEY", "")
NEPTUNE_KEY = os.getenv("NEPTUNE_KEY", "")
DEFAULT_OPENAI_KEY = OPENAI_KEY
OFFLINE_POLICY_NEPTUNE_PROJECT = os.getenv("thejaminator/offline-assistant-policy", "")
