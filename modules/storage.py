import json
import sqlite3
import threading
import time

from cryptography.fernet import Fernet
from google.oauth2.credentials import Credentials

from modules.config import DATABASE_PATH, SCOPES, TOKEN_ENCRYPTION_KEY

fernet = Fernet(TOKEN_ENCRYPTION_KEY)
db_lock = threading.Lock()


def get_db():
    """Open/create the SQLite database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS google_tokens (
            discord_user_id TEXT PRIMARY KEY,
            token_blob BLOB NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def save_credentials(discord_user_id: int, credentials: Credentials):
    """Encrypt and store a user's Google OAuth credentials."""
    encrypted = fernet.encrypt(credentials.to_json().encode("utf-8"))

    with db_lock:
        connection = get_db()
        connection.execute(
            """
            INSERT INTO google_tokens (
                discord_user_id,
                token_blob,
                updated_at
            )
            VALUES (?, ?, ?)
            ON CONFLICT(discord_user_id)
            DO UPDATE SET
                token_blob = excluded.token_blob,
                updated_at = excluded.updated_at
            """,
            (str(discord_user_id), encrypted, int(time.time())),
        )
        connection.commit()
        connection.close()


def load_credentials(discord_user_id: int):
    """Load and decrypt a user's Google credentials."""
    with db_lock:
        connection = get_db()
        row = connection.execute(
            """
            SELECT token_blob
            FROM google_tokens
            WHERE discord_user_id = ?
            """,
            (str(discord_user_id),),
        ).fetchone()
        connection.close()

    if not row:
        return None

    try:
        decrypted = fernet.decrypt(row[0]).decode("utf-8")
        return Credentials.from_authorized_user_info(json.loads(decrypted), SCOPES)
    except Exception:
        return None


def delete_credentials(discord_user_id: int):
    """Remove a user's stored OAuth credentials."""
    with db_lock:
        connection = get_db()
        connection.execute(
            """
            DELETE FROM google_tokens
            WHERE discord_user_id = ?
            """,
            (str(discord_user_id),),
        )
        connection.commit()
        connection.close()
