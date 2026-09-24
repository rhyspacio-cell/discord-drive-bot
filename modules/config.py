import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def require_env(name: str):
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


DISCORD_BOT_TOKEN = require_env("DISCORD_BOT_TOKEN")
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID")
GOOGLE_CLIENT_SECRET_FILE = os.getenv("GOOGLE_CLIENT_SECRET_FILE", "client_secret.json")
OAUTH_REDIRECT_URI = require_env("OAUTH_REDIRECT_URI")
OAUTH_HOST = os.getenv("OAUTH_HOST", "127.0.0.1")
OAUTH_PORT = int(os.getenv("OAUTH_PORT", "8080"))
OAUTH_STATE_TTL = int(os.getenv("OAUTH_STATE_TTL", "600"))
FLASK_SECRET_KEY = require_env("FLASK_SECRET_KEY")
TOKEN_ENCRYPTION_KEY = require_env("TOKEN_ENCRYPTION_KEY").encode("utf-8")
DATABASE_PATH = os.getenv("DATABASE_PATH", "bot.sqlite3")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b")
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
MAX_FILES = int(os.getenv("MAX_FILES", "10"))
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", "10485760"))
MAX_CHARS_PER_FILE = int(os.getenv("MAX_CHARS_PER_FILE", "8000"))
MAX_TOTAL_CHARS = int(os.getenv("MAX_TOTAL_CHARS", "30000"))
MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "5000"))
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
