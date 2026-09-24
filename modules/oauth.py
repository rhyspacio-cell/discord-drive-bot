import secrets
import threading
import time
from urllib.parse import urlencode

from flask import Flask, redirect, request, session
from google_auth_oauthlib.flow import Flow

from modules.config import GOOGLE_CLIENT_SECRET_FILE, OAUTH_REDIRECT_URI, OAUTH_STATE_TTL, SCOPES
from modules.storage import save_credentials


oauth_app = Flask(__name__)
oauth_app.secret_key = __import__("modules.config", fromlist=["FLASK_SECRET_KEY"]).FLASK_SECRET_KEY
oauth_states = {}
oauth_states_lock = threading.Lock()


def create_oauth_flow():
    """Create a Google OAuth flow."""
    return Flow.from_client_secrets_file(
        GOOGLE_CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=OAUTH_REDIRECT_URI,
    )


def create_oauth_url(discord_user_id: int):
    """Create the URL a Discord user opens to authorize Google."""
    oauth_state = secrets.token_urlsafe(32)
    with oauth_states_lock:
        prune_expired_oauth_states(oauth_states)
        oauth_states[oauth_state] = {
            "discord_user_id": str(discord_user_id),
            "created": time.time(),
            "started": False,
        }

    base_url = OAUTH_REDIRECT_URI.rsplit("/oauth2/callback", 1)[0]
    return base_url + "/oauth2/start?" + urlencode({"state": oauth_state})


def prune_expired_oauth_states(oauth_states):
    """Drop stale OAuth states so memory usage stays bounded."""
    now = time.time()
    expired = [
        state
        for state, data in oauth_states.items()
        if now - data["created"] > OAUTH_STATE_TTL
    ]
    for state in expired:
        oauth_states.pop(state, None)


def oauth_start(oauth_states, oauth_states_lock):
    """Begin the Google OAuth process."""
    state = request.args.get("state")
    if not state:
        return "Missing OAuth state.", 400

    with oauth_states_lock:
        prune_expired_oauth_states(oauth_states)
        transaction = oauth_states.get(state)
        if transaction and transaction["started"]:
            return "OAuth authorization has already started for this state.", 400
        if transaction:
            transaction["started"] = True
            transaction["browser_nonce"] = secrets.token_urlsafe(32)

    if not transaction:
        return "Invalid or expired OAuth state.", 400

    session["oauth_state"] = state
    session["oauth_nonce"] = transaction["browser_nonce"]

    flow = create_oauth_flow()
    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )
    return redirect(authorization_url)


def oauth_callback(oauth_states, oauth_states_lock):
    """Receive Google's OAuth callback and save credentials."""
    error = request.args.get("error")
    if error:
        print("[OAUTH ERROR]", "oauth_callback", error)
        return "Google authorization was cancelled or failed.", 400

    state = request.args.get("state")
    code = request.args.get("code")

    with oauth_states_lock:
        prune_expired_oauth_states(oauth_states)
        state_data = oauth_states.get(state) if state else None
        session_matches = bool(
            state_data
            and session.get("oauth_state") == state
            and session.get("oauth_nonce") == state_data.get("browser_nonce")
        )
        if session_matches:
            state_data = oauth_states.pop(state)

    if (
        not state_data
        or not state_data["started"]
        or not session_matches
        or time.time() - state_data["created"] > OAUTH_STATE_TTL
    ):
        return "Invalid or expired OAuth state.", 400

    if not code:
        return "Missing OAuth code.", 400

    try:
        flow = create_oauth_flow()
        flow.fetch_token(code=code)
        save_credentials(int(state_data["discord_user_id"]), flow.credentials)
    except Exception as exc:
        print("[OAUTH ERROR]", "oauth_callback", repr(exc))
        return "Unable to complete Google authorization.", 500

    return """
    <!doctype html>
    <html>
        <head>
            <title>Drive Connected</title>
        </head>
        <body>
            <h2>Google Drive connected successfully.</h2>
            <p>You can close this browser tab and return to Discord.</p>
        </body>
    </html>
    """


@oauth_app.get("/oauth2/start")
def oauth_start_route():
    return oauth_start(oauth_states, oauth_states_lock)


@oauth_app.get("/oauth2/callback")
def oauth_callback_route():
    return oauth_callback(oauth_states, oauth_states_lock)
