"""Application configuration loaded from environment variables.

This module centralizes configuration for the Discord bot, Google Drive
integration, OAuth server, encrypted credential storage, local LLM, and
Drive document processing.

Configuration values are loaded from the project's `.env` file when present.
The `.env` file is resolved relative to the project root rather than the
current working directory, so the application can be started from different
working directories without changing which configuration file is used.

Required variables are validated during module import. Optional variables
fall back to the defaults defined below.

Secrets such as bot tokens, Flask secret keys, and encryption keys should
never be committed to source control.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from the project's root `.env` file.
#
# `__file__` points to this configuration module. Moving two directory levels
# upward reaches the project root where `.env` is expected to be located.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def require_env(name: str):
    """Return a required environment variable or raise a clear error.

    Required configuration values are validated when this module is imported.
    An explicit error is raised immediately if the variable is missing or
    empty, rather than allowing the application to fail later with a less
    informative configuration error.

    Args:
        name: Name of the required environment variable.

    Returns:
        The value of the environment variable.

    Raises:
        RuntimeError: If the variable is missing or contains an empty value.
    """
    value = os.getenv(name)

    if value is None or value == "":
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


# ---------------------------------------------------------------------------
# Discord configuration
# ---------------------------------------------------------------------------

# Authentication token used by the Discord bot.
DISCORD_BOT_TOKEN = require_env("DISCORD_BOT_TOKEN")

# Optional Discord guild ID.
#
# When provided, slash commands can be synchronized to a specific guild.
# When omitted, the bot uses global command synchronization.
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID")


# ---------------------------------------------------------------------------
# Google OAuth configuration
# ---------------------------------------------------------------------------

# Path to the Google OAuth client secret JSON file.
GOOGLE_CLIENT_SECRET_FILE = os.getenv(
    "GOOGLE_CLIENT_SECRET_FILE",
    "client_secret.json",
)

# OAuth callback URL registered with the Google OAuth application.
OAUTH_REDIRECT_URI = require_env("OAUTH_REDIRECT_URI")

# Network interface used by the local OAuth callback server.
OAUTH_HOST = os.getenv(
    "OAUTH_HOST",
    "127.0.0.1",
)

# Port used by the local OAuth callback server.
OAUTH_PORT = int(
    os.getenv(
        "OAUTH_PORT",
        "8080",
    )
)

# Lifetime of an OAuth state value, in seconds.
OAUTH_STATE_TTL = int(
    os.getenv(
        "OAUTH_STATE_TTL",
        "600",
    )
)


# ---------------------------------------------------------------------------
# Security and credential storage
# ---------------------------------------------------------------------------

# Secret key used by the Flask OAuth application.
FLASK_SECRET_KEY = require_env("FLASK_SECRET_KEY")

# Encryption key used to protect stored Google OAuth credentials.
#
# The environment variable is stored as text and converted to UTF-8 bytes
# because the encryption layer expects a byte representation.
TOKEN_ENCRYPTION_KEY = require_env(
    "TOKEN_ENCRYPTION_KEY"
).encode("utf-8")

# SQLite database used for persistent application data.
DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    "bot.sqlite3",
)


# ---------------------------------------------------------------------------
# Local language model configuration
# ---------------------------------------------------------------------------

# Name of the local LLM model used to answer Drive questions.
LOCAL_LLM_MODEL = os.getenv(
    "LOCAL_LLM_MODEL",
    "llama3.2:3b",
)

# Base URL of the local LLM service.
LOCAL_LLM_BASE_URL = os.getenv(
    "LOCAL_LLM_BASE_URL",
    "http://localhost:11434",
)


# ---------------------------------------------------------------------------
# Google Drive document processing limits
# ---------------------------------------------------------------------------

# Maximum number of Drive files considered for extraction and LLM context.
MAX_FILES = int(
    os.getenv(
        "MAX_FILES",
        "10",
    )
)

# Maximum size, in bytes, of an individual Drive file that may be downloaded.
MAX_DOWNLOAD_BYTES = int(
    os.getenv(
        "MAX_DOWNLOAD_BYTES",
        "10485760",
    )
)

# Maximum number of characters retained from any single extracted file.
MAX_CHARS_PER_FILE = int(
    os.getenv(
        "MAX_CHARS_PER_FILE",
        "8000",
    )
)

# Maximum combined number of characters passed to the LLM from Drive files.
MAX_TOTAL_CHARS = int(
    os.getenv(
        "MAX_TOTAL_CHARS",
        "30000",
    )
)

# Maximum number of characters allowed in the generated answer.
MAX_ANSWER_CHARS = int(
    os.getenv(
        "MAX_ANSWER_CHARS",
        "5000",
    )
)


# ---------------------------------------------------------------------------
# Google Drive API permissions
# ---------------------------------------------------------------------------

# Read-only Drive access requested during Google OAuth authorization.
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly"
]