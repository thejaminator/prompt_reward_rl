import os

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

OPENAI_KEY = os.getenv("OPENAI_KEY", "")
