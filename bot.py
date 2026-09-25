"""Discord bot entrypoint for Google Drive question answering.

The application is split into modules for configuration, encrypted token
storage, Google OAuth, Drive extraction, and local LLM question answering.
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
    def __init__(self):
        super().__init__(command_prefix="!", intents=discord.Intents.default())

    async def setup_hook(self):
        if DISCORD_GUILD_ID:
            guild = discord.Object(id=int(DISCORD_GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()


bot = DriveBot()


def split_discord_message(text: str, limit: int = 1900):
    """Split text into Discord-safe chunks without cutting words when possible."""
    chunks = []
    remaining = text.strip()

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at <= 0:
            split_at = limit

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    return chunks


@bot.event
async def on_ready():
    print("[BOT READY]", f"Logged in as {bot.user} ({bot.user.id})")


@bot.tree.command(name="connect-drive", description="Connect your Google Drive")
async def connect_drive(interaction: discord.Interaction):
    url = create_oauth_url(interaction.user.id)
    await interaction.response.send_message(
        "**Connect Google Drive**\n\n"
        "Open this link and authorize the bot to read your Drive:\n\n"
        f"{url}\n\n"
        "Only your Google account connection is associated with your Discord account.",
        ephemeral=True,
    )


@bot.tree.command(name="drive-status", description="Check your Google Drive connection")
async def drive_status(interaction: discord.Interaction):
    credentials = load_credentials(interaction.user.id)
    if credentials:
        message = "Your Google Drive is connected.\n\nUse `/ask-drive` to ask a question."
    else:
        message = "Your Google Drive is not connected.\n\nUse `/connect-drive` first."
    await interaction.response.send_message(message, ephemeral=True)


@bot.tree.command(
    name="ask-drive",
    description="Ask a question about your Google Drive",
)
@app_commands.describe(
    question="Question to answer using your Drive files",
)
async def ask_drive(interaction: discord.Interaction, question: str):
    await interaction.response.defer(ephemeral=True, thinking=True)

    try:
        search_plan = await asyncio.to_thread(
            interpret_search_request,
            question,
        )
        documents, skipped_files = await asyncio.to_thread(
            collect_drive_documents,
            interaction.user.id,
            question,
            search_plan,
        )
        answer = await asyncio.to_thread(answer_drive_question, question, documents)
        skipped_notice = (
            f"\nSkipped oversized files: {len(skipped_files)}"
            if skipped_files
            else ""
        )
        message_chunks = split_discord_message(
            f"**Google Drive Answer**\nReadable matching files: {len(documents)}"
            f"{skipped_notice}\n\n{answer}"
        )
        for message_chunk in message_chunks:
            await interaction.followup.send(message_chunk, ephemeral=True)
    except LocalLLMError as exc:
        print("[LLM ERROR]", repr(exc))
        await interaction.followup.send(
            "⚠️ I couldn't answer your Drive question because the local AI model is unavailable "
            f"or misconfigured.\n\n**Details:** {exc}",
            ephemeral=True,
        )
    except DriveError as exc:
        print("[DRIVE ERROR]", repr(exc))
        await interaction.followup.send(
            f"⚠️ I couldn't read your Google Drive.\n\n**Details:** {exc}",
            ephemeral=True,
        )
    except Exception as exc:
        print("[UNEXPECTED ERROR]", "ask_drive", type(exc).__name__, repr(exc))
        await interaction.followup.send(
            "⚠️ An unexpected error occurred while processing your Drive. "
            "Check the bot's console logs for details.",
            ephemeral=True,
        )


@bot.tree.command(
    name="disconnect-drive",
    description="Remove your stored Google Drive connection",
)
async def disconnect_drive(interaction: discord.Interaction):
    delete_credentials(interaction.user.id)
    await interaction.response.send_message(
        "Your stored Google Drive credentials have been deleted from this bot.",
        ephemeral=True,
    )


def run_oauth_server():
    oauth_app.run(host=OAUTH_HOST, port=OAUTH_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    oauth_thread = threading.Thread(target=run_oauth_server, daemon=True)
    oauth_thread.start()
    bot.run(DISCORD_BOT_TOKEN)
