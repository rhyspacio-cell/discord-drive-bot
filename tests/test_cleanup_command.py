import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import bot as bot_module


BOT_ID = 900
USER_ID = 100
NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


class FakeMessage:
    def __init__(
        self,
        message_id,
        author_id,
        content,
        created_at=NOW,
        *,
        is_bot=False,
        interaction_id=None,
        interaction_name=None,
        interaction_user_id=None,
        delete_error=None,
    ):
        self.id = message_id
        self.author = SimpleNamespace(id=author_id, bot=is_bot)
        self.content = content
        self.created_at = created_at
        self.interaction_metadata = (
            SimpleNamespace(
                id=interaction_id,
                name=interaction_name,
                user_id=interaction_user_id,
            )
            if interaction_id is not None
            else None
        )
        self.delete_error = delete_error
        self.deleted = False

    async def delete(self):
        if self.delete_error:
            raise self.delete_error
        self.deleted = True


class FakeChannel:
    def __init__(self, messages):
        self.messages = messages
        self.history_args = None

    def history(self, **kwargs):
        self.history_args = kwargs

        async def iterate_messages():
            for message in self.messages:
                yield message

        return iterate_messages()


def marked_message(
    message_id,
    role,
    *,
    interaction_id=700,
    created_at=NOW,
    delete_error=None,
):
    marker = bot_module._encode_qa_marker(interaction_id, USER_ID, role)
    return FakeMessage(
        message_id,
        BOT_ID,
        f"Marked {role}{marker}",
        created_at,
        is_bot=True,
        delete_error=delete_error,
    )


def run_cleanup(messages, *, now=NOW):
    channel = FakeChannel(messages)
    result = asyncio.run(
        bot_module._cleanup_channel_history(channel, BOT_ID, now)
    )
    return channel, result


def test_recent_marked_response_and_its_question_are_deleted():
    question = marked_message(1, "question")
    answer = marked_message(2, "answer")

    channel, result = run_cleanup([question, answer])

    assert question.deleted and answer.deleted
    assert result["responses_deleted"] == 1
    assert result["questions_deleted"] == 1
    assert result["failures"] == 0
    assert channel.history_args["after"] == NOW - timedelta(hours=24)


def test_old_bot_response_is_not_deleted():
    old_time = NOW - timedelta(hours=24, seconds=1)
    question = marked_message(1, "question", created_at=old_time)
    answer = marked_message(2, "answer", created_at=old_time)

    _, result = run_cleanup([question, answer])

    assert not question.deleted and not answer.deleted
    assert result["responses_deleted"] == 0
    assert result["questions_deleted"] == 0


def test_recent_single_submission_is_deleted_without_old_separate_question():
    old_question = marked_message(
        1,
        "question",
        created_at=NOW - timedelta(hours=24, seconds=1),
    )
    recent_answer = marked_message(2, "answer")

    _, result = run_cleanup([old_question, recent_answer])

    assert not old_question.deleted
    assert recent_answer.deleted
    assert result["responses_deleted"] == 1
    assert result["questions_deleted"] == 0


def test_unrelated_user_message_before_unmarked_bot_message_remains():
    unrelated = FakeMessage(1, USER_ID, "This is unrelated.")
    other_bot_message = FakeMessage(2, BOT_ID, "A regular bot notification.", is_bot=True)

    _, result = run_cleanup([unrelated, other_bot_message])

    assert not unrelated.deleted
    assert not other_bot_message.deleted
    assert result["responses_deleted"] == 0
    assert result["questions_deleted"] == 0


def test_explicit_provenance_is_authoritative_across_bot_delivery_authors():
    other_bot_question = FakeMessage(
        1,
        901,
        "Question" + bot_module._encode_qa_marker(700, USER_ID, "question"),
        is_bot=True,
    )
    other_bot_answer = FakeMessage(
        2,
        901,
        "Answer" + bot_module._encode_qa_marker(700, USER_ID, "answer"),
        is_bot=True,
    )

    _, result = run_cleanup([other_bot_question, other_bot_answer])

    assert other_bot_question.deleted and other_bot_answer.deleted
    assert result["responses_deleted"] == 1
    assert result["questions_deleted"] == 1


def test_unmarked_bot_authored_message_is_not_treated_as_a_qa_response():
    notification = FakeMessage(1, BOT_ID, "Routine maintenance notice.", is_bot=True)

    _, result = run_cleanup([notification])

    assert not notification.deleted
    assert result["responses_deleted"] == 0


def test_marked_local_ai_unavailable_failure_deletes_response_and_invocation():
    invocation = FakeMessage(
        1,
        USER_ID,
        "Application command invocation",
        interaction_id=700,
        interaction_name="ask-drive",
        interaction_user_id=USER_ID,
    )
    failure = marked_message(2, "answer")
    failure.content = (
        "⚠️ I couldn't answer your Drive question because the local AI model "
        "is unavailable or misconfigured. Details: empty search plan"
        + bot_module._encode_qa_marker(700, USER_ID, "answer")
    )

    _, result = run_cleanup([invocation, failure])

    assert invocation.deleted and failure.deleted
    assert result["responses_deleted"] == 1
    assert result["questions_deleted"] == 1


def test_marked_empty_search_plan_failure_is_eligible_for_cleanup():
    question = marked_message(1, "question")
    failure = marked_message(2, "answer")
    failure.content = (
        "The local AI model returned an empty search plan."
        + bot_module._encode_qa_marker(700, USER_ID, "answer")
    )

    _, result = run_cleanup([question, failure])

    assert question.deleted and failure.deleted
    assert result["responses_deleted"] == 1
    assert result["questions_deleted"] == 1


def test_differently_worded_ask_drive_failure_uses_provenance_not_text():
    question = marked_message(1, "question")
    failure = marked_message(2, "answer")
    failure.content = (
        "A different inference provider failed unexpectedly."
        + bot_module._encode_qa_marker(700, USER_ID, "answer")
    )

    _, result = run_cleanup([question, failure])

    assert question.deleted and failure.deleted
    assert result["responses_deleted"] == 1


def test_unrelated_similar_error_warning_and_user_text_remain():
    bot_error = FakeMessage(
        1,
        BOT_ID,
        "⚠️ I couldn't answer your Drive question because the local AI model is unavailable.",
        is_bot=True,
    )
    bot_warning = FakeMessage(2, BOT_ID, "⚠️ Routine maintenance notice.", is_bot=True)
    user_error_text = FakeMessage(
        3,
        USER_ID,
        "The local AI model returned an empty search plan.",
    )

    _, result = run_cleanup([bot_error, bot_warning, user_error_text])

    assert not bot_error.deleted
    assert not bot_warning.deleted
    assert not user_error_text.deleted
    assert result["responses_deleted"] == 0
    assert result["questions_deleted"] == 0


def test_cleanup_rejects_user_without_manage_messages_permission():
    channel = FakeChannel([])
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=USER_ID,
            guild_permissions=SimpleNamespace(manage_messages=False),
        ),
        channel=channel,
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    asyncio.run(bot_module.cleanup24h.callback(interaction))

    interaction.response.send_message.assert_awaited_once()
    assert "Manage Messages" in interaction.response.send_message.await_args.args[0]
    assert channel.history_args is None


def test_cleanup_command_reports_deleted_skipped_and_failed_counts(monkeypatch):
    monkeypatch.setattr(
        bot_module,
        "bot",
        SimpleNamespace(user=SimpleNamespace(id=BOT_ID)),
    )
    monkeypatch.setattr(
        bot_module,
        "_cleanup_channel_history",
        AsyncMock(return_value={
            "responses_deleted": 3,
            "questions_deleted": 2,
            "skipped": 1,
            "failures": 1,
            "history_failed": False,
        }),
    )
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=USER_ID,
            guild_permissions=SimpleNamespace(manage_messages=True),
        ),
        channel=FakeChannel([]),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    asyncio.run(bot_module.cleanup24h.callback(interaction))

    interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
    summary = interaction.followup.send.await_args.args[0]
    assert "3 bot response messages deleted" in summary
    assert "2 accompanying question messages deleted" in summary
    assert "1 skipped" in summary
    assert "1 deletion failures" in summary


def test_all_recent_pairs_in_the_channel_are_deleted():
    pair_one_question = marked_message(1, "question", interaction_id=700)
    pair_one_answer = marked_message(2, "answer", interaction_id=700)
    pair_two_question = marked_message(3, "question", interaction_id=701)
    pair_two_answer = marked_message(4, "answer", interaction_id=701)

    _, result = run_cleanup([
        pair_one_question,
        pair_one_answer,
        pair_two_question,
        pair_two_answer,
    ])

    assert all(
        message.deleted
        for message in (
            pair_one_question,
            pair_one_answer,
            pair_two_question,
            pair_two_answer,
        )
    )
    assert result["responses_deleted"] == 2
    assert result["questions_deleted"] == 2


def test_api_failure_is_reported_and_keeps_question_when_answer_fails():
    question = marked_message(1, "question")
    answer = marked_message(2, "answer", delete_error=RuntimeError("rate limited"))

    _, result = run_cleanup([question, answer])

    assert not question.deleted
    assert not answer.deleted
    assert result["responses_deleted"] == 0
    assert result["questions_deleted"] == 0
    assert result["failures"] == 1
    assert result["skipped"] == 1


def test_partial_question_deletion_failure_reports_completed_response_count():
    question = marked_message(
        1,
        "question",
        delete_error=RuntimeError("permission failure"),
    )
    answer = marked_message(2, "answer")

    _, result = run_cleanup([question, answer])

    assert answer.deleted
    assert not question.deleted
    assert result["responses_deleted"] == 1
    assert result["questions_deleted"] == 0
    assert result["failures"] == 1
    assert result["skipped"] == 0


def test_cleanup_command_message_is_not_a_question_candidate():
    command_invocation = FakeMessage(1, USER_ID, "/cleanup24h")
    question = marked_message(2, "question")
    answer = marked_message(3, "answer")

    _, result = run_cleanup([command_invocation, question, answer])

    assert not command_invocation.deleted
    assert question.deleted and answer.deleted
    assert result["responses_deleted"] == 1
    assert result["questions_deleted"] == 1


def test_correlated_user_invocation_is_deleted_when_discord_exposes_metadata():
    user_invocation = FakeMessage(
        1,
        USER_ID,
        "Application command invocation",
        interaction_id=700,
    )
    question = marked_message(2, "question")
    answer = marked_message(3, "answer")

    _, result = run_cleanup([user_invocation, question, answer])

    assert user_invocation.deleted
    assert question.deleted and answer.deleted
    assert result["questions_deleted"] == 2
    assert result["responses_deleted"] == 1


def test_legacy_ask_drive_messages_pair_by_discord_interaction_metadata():
    question = FakeMessage(
        1,
        BOT_ID,
        "**Question:** What is the activity date?",
        is_bot=True,
        interaction_id=700,
        interaction_name="ask-drive",
        interaction_user_id=USER_ID,
    )
    answer = FakeMessage(
        2,
        BOT_ID,
        "⚠️ I couldn't answer because the local AI model is unavailable.",
        is_bot=True,
        interaction_id=700,
        interaction_name="ask-drive",
        interaction_user_id=USER_ID,
    )
    unrelated_bot_message = FakeMessage(
        3,
        BOT_ID,
        "A response from another slash command.",
        is_bot=True,
        interaction_id=701,
        interaction_name="drive-status",
        interaction_user_id=USER_ID,
    )

    _, result = run_cleanup([question, answer, unrelated_bot_message])

    assert question.deleted and answer.deleted
    assert not unrelated_bot_message.deleted
    assert result["responses_deleted"] == 1
    assert result["questions_deleted"] == 1


def test_old_ask_drive_failure_is_not_deleted():
    old_time = NOW - timedelta(hours=24, seconds=1)
    question = marked_message(1, "question", created_at=old_time)
    failure = marked_message(2, "answer", created_at=old_time)
    failure.content = (
        "⚠️ A failure occurred."
        + bot_module._encode_qa_marker(700, USER_ID, "answer")
    )

    _, result = run_cleanup([question, failure])

    assert not question.deleted and not failure.deleted
    assert result["responses_deleted"] == 0
    assert result["questions_deleted"] == 0


def test_current_sender_marker_is_recognized_by_cleanup():
    messages = []
    interaction = SimpleNamespace(
        id=700,
        user=SimpleNamespace(id=USER_ID),
        followup=SimpleNamespace(),
        channel=SimpleNamespace(),
    )

    async def send(content, **_kwargs):
        message = FakeMessage(len(messages) + 1, BOT_ID, content, is_bot=True)
        messages.append(message)
        return message

    interaction.followup.send = AsyncMock(side_effect=send)
    marker = bot_module._qa_interaction_markers(interaction)[1]
    submission = bot_module.format_question_answer_submission("Q1", "A1")

    asyncio.run(
        bot_module._send_marked_qa_message(interaction, submission, marker)
    )

    sent_content = messages[0].content
    assert sent_content.endswith(marker)
    assert bot_module._decode_qa_marker(sent_content) == (700, USER_ID, "answer")
    _, result = run_cleanup(messages)
    assert messages[0].deleted
    assert result["responses_deleted"] == 1


def test_webhook_fallback_message_is_cleanup_eligible():
    messages = []
    interaction = SimpleNamespace(
        id=700,
        user=SimpleNamespace(id=USER_ID),
        followup=SimpleNamespace(),
    )

    async def unavailable(_content, **_kwargs):
        raise bot_module.discord.HTTPException(
            response=SimpleNamespace(status=401, reason="Unauthorized"),
            message="Invalid Webhook Token",
        )

    async def channel_send(content, **_kwargs):
        message = FakeMessage(len(messages) + 1, BOT_ID, content, is_bot=True)
        messages.append(message)
        return message

    interaction.followup.send = AsyncMock(side_effect=unavailable)
    interaction.channel = SimpleNamespace(send=AsyncMock(side_effect=channel_send))
    marker = bot_module._qa_interaction_markers(interaction)[1]

    asyncio.run(
        bot_module._send_marked_qa_message(
            interaction,
            bot_module.format_question_answer_submission("Q1", "A1"),
            marker,
        )
    )

    assert bot_module._decode_qa_marker(messages[0].content) == (700, USER_ID, "answer")
    _, result = run_cleanup(messages)
    assert messages[0].deleted
    assert result["responses_deleted"] == 1


def test_primary_and_all_continuations_are_cleanup_eligible():
    messages = []
    interaction = SimpleNamespace(
        id=700,
        user=SimpleNamespace(id=USER_ID),
        followup=SimpleNamespace(),
    )

    async def record_message(content, **_kwargs):
        message = FakeMessage(
            len(messages) + 1,
            BOT_ID,
            content,
            is_bot=True,
        )
        messages.append(message)
        return message

    interaction.followup.send = AsyncMock(side_effect=record_message)
    interaction.channel = SimpleNamespace(
        send=AsyncMock(side_effect=record_message),
    )
    marker = bot_module._qa_interaction_markers(interaction)[1]

    asyncio.run(
        bot_module._send_marked_qa_message(
            interaction,
            "Long answer " + ("x" * 5000),
            marker,
        )
    )

    assert len(messages) > 1
    assert all(
        bot_module._decode_qa_marker(message.content) == (700, USER_ID, "answer")
        for message in messages
    )
    _, result = run_cleanup(messages)
    assert all(message.deleted for message in messages)
    assert result["responses_deleted"] == len(messages)


def test_bulk_question_provenance_is_independent_and_cleanup_ignores_replies():
    interaction = SimpleNamespace(id=700, user=SimpleNamespace(id=USER_ID))
    messages = []
    lifecycle_ids = []
    for index, question in enumerate(("Q1", "Q2", "Q3")):
        marker = bot_module._qa_interaction_markers(interaction, index)[1]
        lifecycle_id, _, role = bot_module._decode_qa_marker(marker)
        lifecycle_ids.append(lifecycle_id)
        messages.append(FakeMessage(
            index + 1,
            BOT_ID,
            bot_module._build_provenance_message(
                bot_module.format_question_answer_submission(question, f"A{index + 1}"),
                marker,
            ),
            is_bot=True,
        ))
    unrelated_reply = FakeMessage(4, USER_ID, "Unrelated reply", interaction_id=None)
    unrelated_reply.reference = SimpleNamespace(message_id=1)
    messages.append(unrelated_reply)

    _, result = run_cleanup(messages)

    assert len(set(lifecycle_ids)) == 3
    assert all(message.deleted for message in messages[:3])
    assert not unrelated_reply.deleted
    assert result["responses_deleted"] == 3


def test_bulk_empty_input_failure_response_has_cleanup_provenance():
    interaction = SimpleNamespace(
        id=700,
        user=SimpleNamespace(id=USER_ID),
        response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
        channel=SimpleNamespace(send=AsyncMock()),
    )

    asyncio.run(bot_module.ask_drive_bulk.callback(interaction, "\n   \n"))

    sent_content = interaction.followup.send.await_args.args[0]
    assert bot_module._decode_qa_marker(sent_content) == (700, USER_ID, "answer")
