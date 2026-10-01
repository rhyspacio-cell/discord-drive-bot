import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import bot as bot_module
from modules.drive import DriveError
from modules.llm import LocalLLMError


def make_interaction():
    return SimpleNamespace(
        user=SimpleNamespace(id=123),
        id=456,
        response=SimpleNamespace(
            send_message=AsyncMock(),
            defer=AsyncMock(),
        ),
        followup=SimpleNamespace(
            send=AsyncMock(),
        ),
    )


def test_connect_drive_response_is_not_ephemeral(monkeypatch):
    monkeypatch.setattr(
        bot_module,
        "create_oauth_url",
        lambda _user_id: "https://example.test/connect",
    )
    interaction = make_interaction()

    asyncio.run(bot_module.connect_drive.callback(interaction))

    interaction.response.send_message.assert_awaited_once()
    assert "ephemeral" not in interaction.response.send_message.await_args.kwargs


def test_drive_status_response_is_not_ephemeral(monkeypatch):
    monkeypatch.setattr(bot_module, "load_credentials", lambda _user_id: None)
    interaction = make_interaction()

    asyncio.run(bot_module.drive_status.callback(interaction))

    interaction.response.send_message.assert_awaited_once_with(
        "Your Google Drive is not connected.\n\n"
        "Use `/connect-drive` first."
    )


def test_disconnect_drive_response_is_not_ephemeral(monkeypatch):
    deleted_users = []
    monkeypatch.setattr(
        bot_module,
        "delete_credentials",
        deleted_users.append,
    )
    interaction = make_interaction()

    asyncio.run(bot_module.disconnect_drive.callback(interaction))

    assert deleted_users == [123]
    interaction.response.send_message.assert_awaited_once_with(
        "Your stored Google Drive credentials have been deleted from this bot."
    )


def test_ask_drive_posts_question_and_answer_once_publicly(monkeypatch):
    monkeypatch.setattr(
        bot_module,
        "interpret_search_request",
        lambda question: {"question": question},
    )
    monkeypatch.setattr(
        bot_module,
        "collect_drive_documents",
        lambda *_args: ([], None, {"used": []}),
    )
    monkeypatch.setattr(
        bot_module,
        "answer_drive_question",
        lambda *_args: "Generated answer.",
    )

    async def run_in_thread(function, *args):
        return function(*args)

    monkeypatch.setattr(bot_module.asyncio, "to_thread", run_in_thread)
    interaction = make_interaction()
    question = "Which companies entered IPI?"

    asyncio.run(bot_module.ask_drive.callback(interaction, question))

    interaction.response.defer.assert_awaited_once_with(
        ephemeral=False,
        thinking=True,
    )
    interaction.response.send_message.assert_not_awaited()
    assert interaction.followup.send.await_count == 2
    question_message = interaction.followup.send.await_args_list[0].args[0]
    answer_message = interaction.followup.send.await_args_list[1].args[0]
    assert question_message.startswith(f"**Question:** {question}")
    assert answer_message.startswith("Generated answer.")
    assert bot_module._decode_qa_marker(question_message) == (456, 123, "question")
    assert bot_module._decode_qa_marker(answer_message) == (456, 123, "answer")
    assert all(
        call.kwargs["ephemeral"] is False
        for call in interaction.followup.send.await_args_list
    )


@pytest.mark.parametrize(
    ("failing_stage", "error_type"),
    [
        ("planner", ValueError),
        ("drive", DriveError),
        ("llm", LocalLLMError),
    ],
)
def test_ask_drive_errors_remain_visible(
    monkeypatch,
    failing_stage,
    error_type,
):
    def planner(question):
        if failing_stage == "planner":
            raise error_type("test failure")
        return {"question": question}

    def collect(*_args):
        if failing_stage == "drive":
            raise error_type("test failure")
        return ([], None, {"used": []})

    def answer(*_args):
        if failing_stage == "llm":
            raise error_type("test failure")
        return "Generated answer."

    monkeypatch.setattr(bot_module, "interpret_search_request", planner)
    monkeypatch.setattr(bot_module, "collect_drive_documents", collect)
    monkeypatch.setattr(bot_module, "answer_drive_question", answer)

    async def run_in_thread(function, *args):
        return function(*args)

    monkeypatch.setattr(bot_module.asyncio, "to_thread", run_in_thread)
    interaction = make_interaction()

    asyncio.run(
        bot_module.ask_drive.callback(
            interaction,
            "Which companies entered IPI?",
        )
    )

    interaction.response.defer.assert_awaited_once_with(
        ephemeral=False,
        thinking=True,
    )
    assert interaction.followup.send.await_count == 2
    question_message = interaction.followup.send.await_args_list[0].args[0]
    failure_message = interaction.followup.send.await_args_list[1].args[0]
    assert bot_module._decode_qa_marker(question_message) == (456, 123, "question")
    assert bot_module._decode_qa_marker(failure_message) == (456, 123, "answer")
    assert all(
        call.kwargs["ephemeral"] is False
        for call in interaction.followup.send.await_args_list
    )


@pytest.mark.parametrize(
    "error_message",
    [
        "The local AI model is unavailable or misconfigured.",
        "The local AI model returned an empty search plan.",
        "A different local inference failure occurred.",
    ],
)
def test_ask_drive_failure_wording_does_not_change_cleanup_provenance(
    monkeypatch,
    error_message,
):
    monkeypatch.setattr(
        bot_module,
        "interpret_search_request",
        lambda _question: {"intent": "question"},
    )
    monkeypatch.setattr(
        bot_module,
        "collect_drive_documents",
        lambda *_args: ([], None, {"used": []}),
    )
    monkeypatch.setattr(
        bot_module,
        "answer_drive_question",
        lambda *_args: (_ for _ in ()).throw(LocalLLMError(error_message)),
    )

    async def run_in_thread(function, *args):
        return function(*args)

    monkeypatch.setattr(bot_module.asyncio, "to_thread", run_in_thread)
    interaction = make_interaction()

    asyncio.run(bot_module.ask_drive.callback(interaction, "Question text"))

    assert interaction.followup.send.await_count == 2
    question_echo = interaction.followup.send.await_args_list[0].args[0]
    response = interaction.followup.send.await_args_list[1].args[0]
    assert bot_module._decode_qa_marker(question_echo) == (456, 123, "question")
    assert bot_module._decode_qa_marker(response) == (456, 123, "answer")
    assert "Details:" in response