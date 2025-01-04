import os

from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    BOT_TOKEN: str = os.getenv("BOT_TOKEN")
