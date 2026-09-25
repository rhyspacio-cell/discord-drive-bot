"""Discord bot entrypoint for Google Drive question answering.

This module owns the Discord-facing portion of the application. It registers
slash commands, manages the Discord bot lifecycle, starts the OAuth callback
server, and coordinates the Drive search and local LLM modules.

The application is intentionally split into separate modules:

    config.py
        Loads application configuration and environment variables.

    storage.py
        Stores, loads, and deletes encrypted Google OAuth credentials.

    oauth.py
        Handles Google OAuth authorization and the callback server.

    search.py
        Converts a user's natural-language question into a structured
        Drive search plan.

    drive.py
        Searches Google Drive, extracts readable file content, ranks
        relevant files, and records a search audit.

    llm.py
        Sends the selected Drive documents to the local language model and
        generates the final answer.

The normal `/ask-drive` flow is:

    Discord question
        -> search plan
        -> Google Drive candidate search
        -> file extraction
        -> relevance filtering
        -> local LLM
        -> answer + file audit
        -> Discord response

All user-facing command responses are ephemeral so that Drive-related
information is only visible to the user who requested it.
"""

import asyncio
import threading

import discord
from discord import app_commands
from discord.ext import commands

from modules.config import (
    DISCORD_BOT_TOKEN,
    DISCORD_GUILD_ID,
    OAUTH_HOST,
    OAUTH_PORT,
)
from modules.drive import DriveError, collect_drive_documents
from modules.llm import LocalLLMError, answer_drive_question
from modules.oauth import create_oauth_url, oauth_app
from modules.search import interpret_search_request
from modules.storage import delete_credentials, load_credentials


class DriveBot(commands.Bot):
    """Discord bot responsible for registering and serving application commands."""

    def __init__(self):
        """Initialize the bot with Discord's default gateway intents."""
        super().__init__(
            command_prefix="!",
            intents=discord.Intents.default(),
        )

    async def setup_hook(self):
        """Synchronize slash commands with Discord.

        If a guild ID is configured, commands are copied to and synchronized
        with that guild. This is useful during development because guild
        commands become available immediately.

        Without a configured guild ID, commands are synchronized globally.
        """
        if DISCORD_GUILD_ID:
            guild = discord.Object(id=int(DISCORD_GUILD_ID))

            self.tree.copy_global_to(guild=guild)

            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()


bot = DriveBot()


def split_discord_message(
    text: str,
    limit: int = 1900,
):
    """Split long text into Discord-safe message chunks.

    Discord imposes a maximum message length. This helper keeps each chunk
    below the configured limit and attempts to split at a newline or space
    instead of cutting through a word.

    Args:
        text: Text that should be divided into multiple Discord messages.
        limit: Maximum number of characters per message chunk.

    Returns:
        A list of message strings suitable for sending to Discord.
    """
    chunks = []
    remaining = text.strip()

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        split_at = remaining.rfind(
            "\n",
            0,
            limit + 1,
        )

        if split_at <= 0:
            split_at = remaining.rfind(
                " ",
                0,
                limit + 1,
            )

        if split_at <= 0:
            split_at = limit

        chunks.append(
            remaining[:split_at].rstrip()
        )

        remaining = remaining[split_at:].lstrip()

    return chunks


def format_drive_error(reason: str) -> str:
    """Convert technical Drive errors into user-friendly explanations.

    The Drive module records the original exception so that it remains
    available in the bot's console logs. Discord users should not be exposed
    to raw Google API URLs, HTTP request details, or internal file IDs.

    Known errors are translated into explanations that describe what the
    user can reasonably understand or do next.

    Args:
        reason: Original technical error message recorded by the Drive module.

    Returns:
        A concise explanation suitable for display in Discord.
    """
    reason_lower = reason.lower()

    if (
        "httperror 404" in reason_lower
        or "file not found" in reason_lower
    ):
        return (
            "The file is a broken Drive shortcut. "
            "Its original file may have been deleted or you may no longer "
            "have access to it."
        )

    if (
        "403" in reason_lower
        or "permission" in reason_lower
    ):
        return (
            "The file could not be accessed because your Google account "
            "does not currently have permission to read it."
        )

    if (
        "401" in reason_lower
        or "unauthorized" in reason_lower
    ):
        return (
            "Google Drive authorization has expired or is no longer valid. "
            "Please reconnect your Drive."
        )

    if (
        "timeout" in reason_lower
        or "timed out" in reason_lower
    ):
        return (
            "The file took too long to download or process."
        )

    return (
        "The file could not be read or processed."
    )


def format_drive_answer(
    answer,
    search_audit,
):
    """Build the final Discord response for a Drive question.

    The response contains two distinct parts:

    1. The answer generated by the local LLM.
    2. A factual file audit generated by Python.

    The audit is deliberately generated outside the LLM so that filenames
    and processing statuses reflect what the application actually did rather
    than what the language model claims it used.

    The audit can contain:

        Reference files used
            Files whose extracted text was actually provided to the LLM.

        Files analyzed but not used
            Files that entered extraction successfully but were ultimately
            not selected for the LLM context.

        Files skipped
            Files intentionally excluded before extraction, for example
            because they exceeded configured size limits.

        Files that could not be read
            Files for which extraction failed.

        Files with no extractable text
            Files that were processed successfully but produced no usable
            text.

    Args:
        answer: Final answer returned by the local LLM.
        search_audit: Processing information returned by the Drive module.

    Returns:
        A formatted Discord message containing the answer and audit.
    """
    lines = []

    answer = answer.strip()

    # ---------------------------------------------------------
    # ACTUAL ANSWER FIRST
    # ---------------------------------------------------------

    if answer:
        lines.append(answer)

    used = search_audit.get(
        "used",
        [],
    )

    analyzed = search_audit.get(
        "analyzed",
        [],
    )

    skipped = search_audit.get(
        "skipped",
        [],
    )

    failed = search_audit.get(
        "failed",
        [],
    )

    empty = search_audit.get(
        "empty",
        [],
    )

    # ---------------------------------------------------------
    # FILES ACTUALLY USED BY THE LLM
    # ---------------------------------------------------------

    if used:
        lines.append("")
        lines.append("**Reference files used**")

        for name in used:
            lines.append(
                f"- `{name}`"
            )

    # ---------------------------------------------------------
    # FILES ANALYZED BUT NOT USED
    # ---------------------------------------------------------

    # A file can appear in several processing stages. Exclude files that
    # already have a more specific status so that the Discord audit does not
    # report the same file under multiple headings.

    used_set = set(used)

    skipped_names = {
        item["name"]
        for item in skipped
    }

    failed_names = {
        item["name"]
        for item in failed
    }

    empty_names = set(empty)

    analyzed_not_used = [
        name
        for name in analyzed
        if name not in used_set
        and name not in skipped_names
        and name not in failed_names
        and name not in empty_names
    ]

    if analyzed_not_used:
        lines.append("")
        lines.append(
            "**Files analyzed but not used**"
        )

        for name in analyzed_not_used:
            lines.append(
                f"- `{name}`"
            )

    # ---------------------------------------------------------
    # FILES SKIPPED
    # ---------------------------------------------------------

    if skipped:
        lines.append("")
        lines.append("**Files skipped**")

        for item in skipped:
            lines.append(
                f"- `{item['name']}` — {item['reason']}"
            )

    # ---------------------------------------------------------
    # FILES THAT FAILED EXTRACTION
    # ---------------------------------------------------------

    if failed:
        lines.append("")
        lines.append(
            "**Files that could not be read**"
        )

        for item in failed:
            friendly_reason = format_drive_error(
                item["reason"]
            )

            lines.append(
                f"- `{item['name']}` — {friendly_reason}"
            )

    # ---------------------------------------------------------
    # FILES WITH NO EXTRACTABLE TEXT
    # ---------------------------------------------------------

    if empty:
        lines.append("")
        lines.append(
            "**Files with no extractable text**"
        )

        for name in empty:
            lines.append(
                f"- `{name}`"
            )

    return "\n".join(lines)


@bot.event
async def on_ready():
    """Log a message when the Discord bot successfully connects."""
    print(
        "[BOT READY]",
        f"Logged in as {bot.user} ({bot.user.id})",
    )


@bot.tree.command(
    name="connect-drive",
    description="Connect your Google Drive",
)
async def connect_drive(
    interaction: discord.Interaction,
):
    """Create a Google OAuth link for the current Discord user.

    The generated authorization URL associates the user's Google Drive
    connection with their Discord user ID.
    """
    url = create_oauth_url(
        interaction.user.id
    )

    await interaction.response.send_message(
        "**Connect Google Drive**\n\n"
        "Open this link and authorize the bot to read your Drive:\n\n"
        f"{url}\n\n"
        "Only your Google account connection is associated "
        "with your Discord account.",
        ephemeral=True,
    )


@bot.tree.command(
    name="drive-status",
    description="Check your Google Drive connection",
)
async def drive_status(
    interaction: discord.Interaction,
):
    """Report whether the current Discord user has connected Google Drive."""
    credentials = load_credentials(
        interaction.user.id
    )

    if credentials:
        message = (
            "Your Google Drive is connected.\n\n"
            "Use `/ask-drive` to ask a question."
        )
    else:
        message = (
            "Your Google Drive is not connected.\n\n"
            "Use `/connect-drive` first."
        )

    await interaction.response.send_message(
        message,
        ephemeral=True,
    )


@bot.tree.command(
    name="ask-drive",
    description="Ask a question about your Google Drive",
)
@app_commands.describe(
    question="Question to answer using your Drive files",
)
async def ask_drive(
    interaction: discord.Interaction,
    question: str,
):
    """Answer a user's question using relevant Google Drive files.

    Processing is performed in several stages:

    1. Interpret the natural-language question into a search plan.
    2. Search the user's connected Google Drive.
    3. Extract text from candidate files.
    4. Rank and filter the extracted documents.
    5. Send the selected documents to the local LLM.
    6. Format the generated answer together with the Drive audit.
    7. Split the response into Discord-safe chunks and send them privately.

    CPU-bound or blocking Drive/LLM operations are executed in worker
    threads so that they do not block Discord's asynchronous event loop.

    Args:
        interaction: Discord interaction containing the user and question.
        question: Natural-language question to answer using Drive content.
    """
    await interaction.response.defer(
        ephemeral=True,
        thinking=True,
    )

    try:
        # -----------------------------------------------------
        # 1. Convert the question into a structured search plan.
        # -----------------------------------------------------

        search_plan = await asyncio.to_thread(
            interpret_search_request,
            question,
        )

        # -----------------------------------------------------
        # 2. Search Drive, extract relevant files, and collect
        #    an audit of what happened to each candidate.
        # -----------------------------------------------------

        documents, _, search_audit = (
            await asyncio.to_thread(
                collect_drive_documents,
                interaction.user.id,
                question,
                search_plan,
            )
        )

        # -----------------------------------------------------
        # 3. Ask the local LLM to answer using only the selected
        #    Drive documents.
        # -----------------------------------------------------

        answer = await asyncio.to_thread(
            answer_drive_question,
            question,
            documents,
        )

        # -----------------------------------------------------
        # 4. Combine the LLM answer with the factual Python-
        #    generated file audit.
        # -----------------------------------------------------

        message = format_drive_answer(
            answer,
            search_audit,
        )

        # -----------------------------------------------------
        # 5. Discord limits message length, so split long
        #    answers into multiple private messages.
        # -----------------------------------------------------

        message_chunks = split_discord_message(
            message
        )

        for message_chunk in message_chunks:
            await interaction.followup.send(
                message_chunk,
                ephemeral=True,
            )

    except LocalLLMError as exc:
        """Handle failures from the local language model."""
        print(
            "[LLM ERROR]",
            repr(exc),
        )

        await interaction.followup.send(
            "⚠️ I couldn't answer your Drive question "
            "because the local AI model is unavailable "
            f"or misconfigured.\n\n**Details:** {exc}",
            ephemeral=True,
        )

    except DriveError as exc:
        """Handle failures while accessing or processing Google Drive."""
        print(
            "[DRIVE ERROR]",
            repr(exc),
        )

        await interaction.followup.send(
            "⚠️ I couldn't read your Google Drive.\n\n"
            f"**Details:** {exc}",
            ephemeral=True,
        )

    except Exception as exc:
        """Handle unexpected failures without exposing internals to Discord."""
        print(
            "[UNEXPECTED ERROR]",
            "ask_drive",
            type(exc).__name__,
            repr(exc),
        )

        await interaction.followup.send(
            "⚠️ An unexpected error occurred while "
            "processing your Drive. Check the bot's "
            "console logs for details.",
            ephemeral=True,
        )


@bot.tree.command(
    name="disconnect-drive",
    description="Remove your stored Google Drive connection",
)
async def disconnect_drive(
    interaction: discord.Interaction,
):
    """Delete the current Discord user's stored Google credentials."""
    delete_credentials(
        interaction.user.id
    )

    await interaction.response.send_message(
        "Your stored Google Drive credentials have been deleted from this bot.",
        ephemeral=True,
    )


def run_oauth_server():
    """Start the local OAuth callback server.

    The OAuth server runs in a daemon thread so that the Discord bot and
    Google authorization callback endpoint can operate concurrently.
    """
    oauth_app.run(
        host=OAUTH_HOST,
        port=OAUTH_PORT,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    # Start the Google OAuth callback server in the background.
    oauth_thread = threading.Thread(
        target=run_oauth_server,
        daemon=True,
    )

    oauth_thread.start()

    # Start the Discord bot. This call blocks until the bot exits.
    bot.run(DISCORD_BOT_TOKEN)