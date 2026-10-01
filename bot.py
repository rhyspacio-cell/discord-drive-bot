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

All user-facing command responses are visible in the channel.
"""

import asyncio
import threading
from datetime import timedelta, timezone

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

_QA_MARKER_START = "\u2063"
_QA_MARKER_END = "\u2064"
_QA_MARKER_ZERO = "\u200b"
_QA_MARKER_ONE = "\u200c"
_QA_MARKER_BITS = 17 * 8


def _encode_qa_marker(interaction_id, user_id, role):
    role_value = {"question": 0, "answer": 1}.get(role)
    if role_value is None:
        raise ValueError("Unknown Q&A marker role.")
    payload = (
        int(interaction_id).to_bytes(8, "big")
        + int(user_id).to_bytes(8, "big")
        + bytes((role_value,))
    )
    bits = "".join(f"{value:08b}" for value in payload)
    hidden_bits = bits.translate(str.maketrans({
        "0": _QA_MARKER_ZERO,
        "1": _QA_MARKER_ONE,
    }))
    return f"{_QA_MARKER_START}{hidden_bits}{_QA_MARKER_END}"


def _qa_interaction_markers(interaction):
    return (
        _encode_qa_marker(interaction.id, interaction.user.id, "question"),
        _encode_qa_marker(interaction.id, interaction.user.id, "answer"),
    )


async def _send_marked_qa_message(interaction, text, marker):
    for chunk in split_discord_message(text, limit=1800):
        await interaction.followup.send(
            chunk + marker,
            ephemeral=False,
        )


def _decode_qa_marker(content):
    if not isinstance(content, str):
        return None
    start = content.rfind(_QA_MARKER_START)
    end = content.find(_QA_MARKER_END, start + 1)
    if start < 0 or end < 0:
        return None
    hidden_bits = content[start + 1:end]
    if len(hidden_bits) != _QA_MARKER_BITS:
        return None
    bit_values = {
        _QA_MARKER_ZERO: "0",
        _QA_MARKER_ONE: "1",
    }
    try:
        bits = "".join(bit_values[character] for character in hidden_bits)
    except KeyError:
        return None
    payload = int(bits, 2).to_bytes(17, "big")
    role = {0: "question", 1: "answer"}.get(payload[-1])
    if role is None:
        return None
    return (
        int.from_bytes(payload[:8], "big"),
        int.from_bytes(payload[8:16], "big"),
        role,
    )


def _is_in_time_window(message, cutoff, now):
    created_at = getattr(message, "created_at", None)
    if created_at is None:
        return False
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return cutoff <= created_at <= now


def _legacy_ask_drive_marker(message):
    metadata = (
        getattr(message, "interaction_metadata", None)
        or getattr(message, "interaction", None)
    )
    if getattr(metadata, "name", None) not in {"ask-drive", "/ask-drive"}:
        return None
    interaction_id = getattr(metadata, "id", None)
    metadata_user = getattr(metadata, "user", None)
    user_id = getattr(metadata, "user_id", None) or getattr(metadata_user, "id", None)
    if interaction_id is None or user_id is None:
        return None
    role = (
        "question"
        if getattr(message, "content", "").startswith("**Question:**")
        else "answer"
    )
    return int(interaction_id), int(user_id), role


def _find_qna_cleanup_groups(messages, bot_user_id, cutoff, now):
    recent_messages = [
        message
        for message in messages
        if _is_in_time_window(message, cutoff, now)
    ]
    groups = {}

    for message in recent_messages:
        author = getattr(message, "author", None)
        if getattr(author, "id", None) != bot_user_id:
            continue
        marker = _decode_qa_marker(getattr(message, "content", ""))
        if marker is None:
            marker = _legacy_ask_drive_marker(message)
            if marker is None:
                continue
        interaction_id, user_id, role = marker
        group = groups.setdefault(
            (interaction_id, user_id),
            {"questions": [], "answers": []},
        )
        group["questions" if role == "question" else "answers"].append(message)

    paired_groups = []
    skipped = 0
    for (interaction_id, user_id), group in groups.items():
        if not group["answers"]:
            skipped += len(group["answers"]) + len(group["questions"])
            continue

        question_messages = list(group["questions"])
        for message in recent_messages:
            author = getattr(message, "author", None)
            if getattr(author, "id", None) != user_id or getattr(author, "bot", False):
                continue
            metadata = (
                getattr(message, "interaction_metadata", None)
                or getattr(message, "interaction", None)
            )
            if getattr(metadata, "id", None) == interaction_id:
                question_messages.append(message)

        question_messages = list({
            message.id: message for message in question_messages
        }.values())
        if not question_messages:
            skipped += len(group["answers"])
            continue

        paired_groups.append({
            "answers": group["answers"],
            "questions": question_messages,
        })

    return paired_groups, skipped


async def _cleanup_channel_history(channel, bot_user_id, now=None):
    """Delete recent marked pairs one-by-one for precise age and failure handling."""
    now = now or discord.utils.utcnow()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff = now - timedelta(hours=24)

    try:
        messages = [
            message
            async for message in channel.history(
                limit=None,
                after=cutoff,
                oldest_first=True,
            )
        ]
    except Exception:
        return {
            "responses_deleted": 0,
            "questions_deleted": 0,
            "skipped": 0,
            "failures": 1,
            "history_failed": True,
        }

    groups, skipped = _find_qna_cleanup_groups(
        messages,
        bot_user_id,
        cutoff,
        now,
    )
    responses_deleted = 0
    questions_deleted = 0
    failures = 0

    for group in groups:
        response_group_failed = False
        for index, message in enumerate(group["answers"]):
            try:
                await message.delete()
                responses_deleted += 1
            except Exception:
                failures += 1
                skipped += len(group["answers"]) - index - 1
                skipped += len(group["questions"])
                response_group_failed = True
                break
        if response_group_failed:
            continue

        for message in group["questions"]:
            try:
                await message.delete()
                questions_deleted += 1
            except Exception:
                failures += 1

    return {
        "responses_deleted": responses_deleted,
        "questions_deleted": questions_deleted,
        "skipped": skipped,
        "failures": failures,
        "history_failed": False,
    }


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

    candidate_diagnostics = search_audit.get("candidate_diagnostics", [])
    validated_evidence_ids = set(
        search_audit.get("validated_evidence_document_ids", [])
    )
    validated_spans_by_id = {}
    for record in search_audit.get("validated_evidence_spans", []):
        if record.get("document_id") and record.get("evidence_span"):
            validated_spans_by_id.setdefault(record["document_id"], []).append(
                record["evidence_span"]
            )
    reference_evidence = []
    orphan_reference_count = 0
    for record in search_audit.get("reference_evidence", []):
        document_id = record.get("document_id")
        if (
            document_id not in validated_evidence_ids
            or not record.get("evidence_span")
            or not any(
                record["evidence_span"] in span
                for span in validated_spans_by_id.get(document_id, [])
            )
            or record.get("supports_final_claim") is not True
            or not record.get("file_name")
        ):
            orphan_reference_count += 1
            continue
        reference_evidence.append(record)
    if orphan_reference_count:
        search_audit["reference_firewall_rejected"] = orphan_reference_count
    used = list(dict.fromkeys(
        record["file_name"] for record in reference_evidence
    ))

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
    used_candidate_ids = {
        record["document_id"] for record in reference_evidence
    }

    skipped_names = {
        item["name"]
        for item in skipped
    }

    failed_names = {
        item["name"]
        for item in failed
    }

    empty_names = set(empty)

    if candidate_diagnostics and used_candidate_ids:
        analyzed_not_used = [
            item
            for item in candidate_diagnostics
            if item.get("extraction_status") == "SUCCESS"
            and item.get("candidate_id") not in used_candidate_ids
            and item.get("file_id") not in used_candidate_ids
            and item.get("file_name") not in used_set
        ]
    else:
        analyzed_not_used = [
            {"file_name": name}
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

        for item in analyzed_not_used:
            name = item.get("file_name", item.get("name", "Unnamed file"))
            lines.append(
                f"- `{name}`"
            )
            if search_audit.get("debug"):
                assessment = next(
                    (
                        assessment
                        for assessment in search_audit.get("assessment_records", [])
                        if assessment.get("candidate_id") == item.get("candidate_id")
                    ),
                    search_audit.get("assessments", {}).get(name, {}),
                )
                reasons = assessment.get("rejection_reasons", [])
                if reasons:
                    lines.append(
                        f"  reason: {', '.join(reasons)}"
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
    7. Post the original question and the answer as visible channel messages,
       splitting long content into Discord-safe chunks when necessary.

    CPU-bound or blocking Drive/LLM operations are executed in worker
    threads so that they do not block Discord's asynchronous event loop.

    Args:
        interaction: Discord interaction containing the user and question.
        question: Natural-language question to answer using Drive content.
    """
    await interaction.response.defer(
        ephemeral=False,
        thinking=True,
    )
    question_marker, answer_marker = _qa_interaction_markers(interaction)
    question_echo_sent = False

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
            search_plan,
            search_audit,
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
        # 5. Post the original question visibly in the channel,
        #    then send the answer as ordinary persistent messages.
        # -----------------------------------------------------

        await _send_marked_qa_message(
            interaction,
            f"**Question:** {question}",
            question_marker,
        )
        question_echo_sent = True

        await _send_marked_qa_message(
            interaction,
            message,
            answer_marker,
        )

    except LocalLLMError as exc:
        """Handle failures from the local language model."""
        print(
            "[LLM ERROR]",
            repr(exc),
        )

        if not question_echo_sent:
            await _send_marked_qa_message(
                interaction,
                f"**Question:** {question}",
                question_marker,
            )
        await _send_marked_qa_message(
            interaction,
            "⚠️ I couldn't answer your Drive question "
            "because the local AI model is unavailable "
            f"or misconfigured.\n\n**Details:** {exc}",
            answer_marker,
        )

    except DriveError as exc:
        """Handle failures while accessing or processing Google Drive."""
        print(
            "[DRIVE ERROR]",
            repr(exc),
        )

        if not question_echo_sent:
            await _send_marked_qa_message(
                interaction,
                f"**Question:** {question}",
                question_marker,
            )
        await _send_marked_qa_message(
            interaction,
            "⚠️ I couldn't read your Google Drive.\n\n"
            f"**Details:** {exc}",
            answer_marker,
        )

    except Exception as exc:
        """Handle unexpected failures without exposing internals to Discord."""
        print(
            "[UNEXPECTED ERROR]",
            "ask_drive",
            type(exc).__name__,
            repr(exc),
        )

        if not question_echo_sent:
            await _send_marked_qa_message(
                interaction,
                f"**Question:** {question}",
                question_marker,
            )
        await _send_marked_qa_message(
            interaction,
            "⚠️ An unexpected error occurred while "
            "processing your Drive. Check the bot's "
            "console logs for details.",
            answer_marker,
        )


@bot.tree.command(
    name="cleanup24h",
    description="Remove this bot's marked Q&A messages from the last 24 hours",
)
@app_commands.guild_only()
async def cleanup24h(interaction: discord.Interaction):
    """Clean recent Q&A pairs after an explicit Manage Messages check."""
    user_permissions = getattr(interaction.user, "guild_permissions", None)
    if not user_permissions or not user_permissions.manage_messages:
        await interaction.response.send_message(
            "You need the Manage Messages permission to run this cleanup.",
            ephemeral=True,
        )
        return

    channel = interaction.channel
    if channel is None or not hasattr(channel, "history"):
        await interaction.response.send_message(
            "This command can only clean a server text channel.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(ephemeral=True, thinking=True)
    result = await _cleanup_channel_history(channel, bot.user.id)
    if result["history_failed"]:
        summary = "Could not inspect channel history; no messages were deleted."
    else:
        summary = (
            "Cleanup finished for the last 24 hours: "
            f"{result['responses_deleted']} bot response messages deleted, "
            f"{result['questions_deleted']} accompanying question messages deleted, "
            f"{result['skipped']} skipped, "
            f"{result['failures']} deletion failures."
        )
    await interaction.followup.send(summary, ephemeral=True)


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