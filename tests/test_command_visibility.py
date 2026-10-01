import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

import bot as bot_module
from modules.drive import DriveError
from modules.llm import LocalLLMError


def make_interaction():
    return SimpleNamespace(
        user=SimpleNamespace(id=123),
        id=456,
        channel=SimpleNamespace(
            send=AsyncMock(),
        ),
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
    assert interaction.followup.send.await_count == 1
    submission = interaction.followup.send.await_args_list[0].args[0]
    assert submission.startswith(f"**Question:** {question}\n\n**Answer:**")
    assert "Generated answer." in submission
    assert bot_module._decode_qa_marker(submission) == (456, 123, "answer")
    assert all(
        call.kwargs["ephemeral"] is False
        for call in interaction.followup.send.await_args_list
    )


def test_primary_followup_send_never_uses_reference_keyword(monkeypatch):
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

    asyncio.run(bot_module.ask_drive.callback(interaction, "Short question"))

    assert interaction.followup.send.await_count == 1
    first_call = interaction.followup.send.await_args_list[0]
    assert first_call.kwargs.get("reference") is None
    assert first_call.kwargs.get("message_reference") is None
    assert first_call.kwargs.get("reply_to") is None
    assert interaction.channel.send.await_count == 0


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
    assert interaction.followup.send.await_count == 1
    submission = interaction.followup.send.await_args_list[0].args[0]
    assert submission.startswith("**Question:** Which companies entered IPI?")
    assert "⚠️" in submission or "Details:" in submission
    assert bot_module._decode_qa_marker(submission) == (456, 123, "answer")
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

    assert interaction.followup.send.await_count == 1
    response = interaction.followup.send.await_args_list[0].args[0]
    assert response.startswith("**Question:** Question text\n\n**Answer:**")
    assert bot_module._decode_qa_marker(response) == (456, 123, "answer")
    assert "Details:" in response


def test_bulk_question_parser_ignores_blank_lines():
    parsed = bot_module.parse_bulk_questions(
        "\n\nWhat are the risks identified for the project?\n\n"
        "What is John Smith's salary?\n\n"
    )

    assert parsed == [
        "What are the risks identified for the project?",
        "What is John Smith's salary?",
    ]


def test_ask_drive_bulk_posts_question_and_answer_per_question(monkeypatch):
    calls = []

    def planner(question):
        calls.append(("plan", question))
        return {"question": question}

    def collect(*args):
        calls.append(("collect", args[1]))
        return ([], None, {"used": []})

    def answer(*args):
        calls.append(("answer", args[0]))
        return f"Answer for: {args[0]}"

    monkeypatch.setattr(bot_module, "interpret_search_request", planner)
    monkeypatch.setattr(bot_module, "collect_drive_documents", collect)
    monkeypatch.setattr(bot_module, "answer_drive_question", answer)

    async def run_in_thread(function, *args):
        return function(*args)

    monkeypatch.setattr(bot_module.asyncio, "to_thread", run_in_thread)
    interaction = make_interaction()
    bulk_input = (
        "What are the risks identified for the project?\n"
        "What is the planned activity date for John Smith?\n"
        "Who is authorized to represent John Smith?"
    )

    asyncio.run(bot_module.ask_drive_bulk.callback(interaction, bulk_input))

    interaction.response.defer.assert_awaited_once_with(
        ephemeral=False,
        thinking=True,
    )
    assert interaction.followup.send.await_count == 3
    first_submission = interaction.followup.send.await_args_list[0].args[0]
    second_submission = interaction.followup.send.await_args_list[1].args[0]
    third_submission = interaction.followup.send.await_args_list[2].args[0]
    assert [bot_module._decode_qa_marker(message) for message in (
        first_submission,
        second_submission,
        third_submission,
    )] == [
        (bot_module._derive_question_lifecycle_id(interaction, index), 123, "answer")
        for index in range(3)
    ]
    assert first_submission.startswith("**Question:** What are the risks identified for the project?\n\n**Answer:**")
    assert "Answer for: What are the risks identified for the project?" in first_submission
    assert second_submission.startswith("**Question:** What is the planned activity date for John Smith?\n\n**Answer:**")
    assert third_submission.startswith("**Question:** Who is authorized to represent John Smith?\n\n**Answer:**")
    assert [call[1] for call in calls if call[0] == "plan"] == [
        "What are the risks identified for the project?",
        "What is the planned activity date for John Smith?",
        "Who is authorized to represent John Smith?",
    ]
    assert all(
        call.kwargs["ephemeral"] is False
        for call in interaction.followup.send.await_args_list
    )


def test_ask_drive_bulk_empty_input_returns_validation_message(monkeypatch):
    interaction = make_interaction()

    asyncio.run(bot_module.ask_drive_bulk.callback(interaction, "\n\n   \n"))

    interaction.response.defer.assert_awaited_once_with(
        ephemeral=False,
        thinking=True,
    )
    assert interaction.followup.send.await_count == 1
    message = interaction.followup.send.await_args_list[0].args[0]
    assert "No valid questions" in message


def test_ask_drive_long_answer_first_message_is_standalone_and_continuations_reply(monkeypatch):
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
        lambda *_args: "A" * 8000,
    )

    async def run_in_thread(function, *args):
        return function(*args)

    monkeypatch.setattr(bot_module.asyncio, "to_thread", run_in_thread)
    interaction = make_interaction()

    asyncio.run(bot_module.ask_drive.callback(interaction, "Question text"))

    assert interaction.followup.send.await_count >= 1
    assert all(
        bot_module._decode_qa_marker(call.args[0]) == (456, 123, "answer")
        for call in interaction.followup.send.await_args_list
    )
    first_call = interaction.followup.send.await_args_list[0]
    assert first_call.kwargs.get("reference") is None
    assert first_call.kwargs.get("message_reference") is None
    assert first_call.kwargs.get("reply_to") is None
    for call in interaction.followup.send.await_args_list[1:]:
        assert call.kwargs.get("reference") is None
        assert call.kwargs.get("message_reference") is None
        assert call.kwargs.get("reply_to") is None


def test_ask_drive_bulk_long_question_only_uses_continuation_replies(monkeypatch):
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

    def answer(*args):
        question = args[0]
        if question == "Q1":
            return "Short answer"
        if question == "Q2":
            return "B" * 8000
        return "Short answer 3"

    monkeypatch.setattr(bot_module, "answer_drive_question", answer)

    async def run_in_thread(function, *args):
        return function(*args)

    monkeypatch.setattr(bot_module.asyncio, "to_thread", run_in_thread)
    interaction = make_interaction()
    bulk_input = "Q1\nQ2\nQ3"

    asyncio.run(bot_module.ask_drive_bulk.callback(interaction, bulk_input))

    assert interaction.followup.send.await_count >= 3
    for call in interaction.followup.send.await_args_list:
        assert call.kwargs.get("reference") is None
        assert call.kwargs.get("message_reference") is None
        assert call.kwargs.get("reply_to") is None
    assert interaction.channel.send.await_count == 0


def test_invalid_webhook_falls_back_to_channel_delivery(monkeypatch):
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
    invalid_webhook = discord.HTTPException(
        response=SimpleNamespace(status=401, reason="Unauthorized"),
        message="Invalid Webhook Token",
    )
    invalid_webhook.code = 50027
    interaction.followup.send.side_effect = invalid_webhook

    asyncio.run(bot_module.ask_drive.callback(interaction, "Question text"))

    assert interaction.followup.send.await_count >= 1
    assert interaction.channel.send.await_count >= 1
    sent = interaction.channel.send.await_args_list[0].args[0]
    assert sent.startswith("**Question:** Question text\n\n**Answer:**")
