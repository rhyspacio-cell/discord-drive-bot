"""Discord bot entrypoint for the Google Drive summarizer.

The application is split into modules for configuration, encrypted token
storage, Google OAuth, Drive extraction, and local LLM summarization.
"""

import asyncio
import threading

import discord
from discord.ext import commands

from modules.config import (
    DISCORD_BOT_TOKEN,
    DISCORD_GUILD_ID,
    OAUTH_HOST,
    OAUTH_PORT,
)
from modules.drive import collect_drive_documents
from modules.llm import LocalLLMError, summarize_documents
from modules.oauth import create_oauth_url, oauth_app
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
        message = "Your Google Drive is connected.\n\nUse `/summarize-drive` to generate a summary."
    else:
        message = "Your Google Drive is not connected.\n\nUse `/connect-drive` first."
    await interaction.response.send_message(message, ephemeral=True)


@bot.tree.command(
    name="summarize-drive",
    description="Summarize readable files in your Google Drive",
)
async def summarize_drive(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)

    try:
        documents = await asyncio.to_thread(collect_drive_documents, interaction.user.id)
        summary = await asyncio.to_thread(summarize_documents, documents)
        await interaction.followup.send(
            f"**Google Drive Summary**\nReadable files: {len(documents)}\n\n{summary}",
            ephemeral=True,
        )
    except LocalLLMError as exc:
        print("[LLM ERROR]", repr(exc))
        await interaction.followup.send(
            "⚠️ I couldn't summarize your Drive because the local AI model is unavailable "
            f"or misconfigured.\n\n**Details:** {exc}",
            ephemeral=True,
        )
    except RuntimeError as exc:
        print("[DRIVE ERROR]", repr(exc))
        await interaction.followup.send(
            f"⚠️ I couldn't read your Google Drive.\n\n**Details:** {exc}",
            ephemeral=True,
        )
    except Exception as exc:
        print("[UNEXPECTED ERROR]", "summarize_drive", type(exc).__name__, repr(exc))
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
