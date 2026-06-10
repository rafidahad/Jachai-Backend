from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import BackgroundTasks

from app.api.routes_webhooks import (
    InboundMessageSchema,
    _normalize_inbound_message,
    inbound_message,
)
from app.utils.errors import AppError


def test_normalize_inbound_message_supports_telegram_text_payload() -> None:
    payload = InboundMessageSchema(
        platform="telegram",
        sender_id="12345",
        message_id="67890",
        text="Please verify this claim for me.",
        metadata={"chat_type": "private"},
    )

    normalized = _normalize_inbound_message(payload)

    assert normalized == {
        "message_type": "text",
        "message_text": "Please verify this claim for me.",
        "message_url": None,
        "supporting_text": None,
        "external_id": "telegram:12345:67890",
    }


def test_normalize_inbound_message_detects_url_and_supporting_text() -> None:
    payload = InboundMessageSchema(
        platform="telegram",
        sender_id="12345",
        message_id="67890",
        text="Please check this news link https://example.com/story/123 right now",
    )

    normalized = _normalize_inbound_message(payload)

    assert normalized == {
        "message_type": "url",
        "message_text": None,
        "message_url": "https://example.com/story/123",
        "supporting_text": "Please check this news link right now",
        "external_id": "telegram:12345:67890",
    }


def test_normalize_inbound_message_supports_explicit_url_payload() -> None:
    payload = InboundMessageSchema(
        message_type="url",
        message_text="Context before the link https://example.com/story/123",
    )

    normalized = _normalize_inbound_message(payload)

    assert normalized == {
        "message_type": "url",
        "message_text": None,
        "message_url": "https://example.com/story/123",
        "supporting_text": "Context before the link",
        "external_id": None,
    }


def test_normalize_inbound_message_rejects_missing_text_and_url() -> None:
    payload = InboundMessageSchema(metadata={"platform": "telegram"})

    with pytest.raises(AppError) as exc_info:
        _normalize_inbound_message(payload)

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "INVALID_WEBHOOK_PAYLOAD"


@pytest.mark.asyncio
async def test_inbound_message_enqueues_text_job() -> None:
    payload = InboundMessageSchema(message_type="text", message_text="Please verify this claim.")
    background_tasks = BackgroundTasks()
    session = object()
    expected_response = SimpleNamespace(job=SimpleNamespace(id="text-job"))

    with patch(
        "app.api.routes_webhooks.enqueue_text_claim",
        new_callable=AsyncMock,
        return_value=expected_response,
    ) as mock_enqueue:
        with patch("app.api.routes_webhooks.run_text_claim_job") as mock_runner:
            result = await inbound_message(
                payload,
                background_tasks=background_tasks,
                session=session,
            )

    assert result == expected_response
    mock_enqueue.assert_awaited_once_with(session, "Please verify this claim.", None)
    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].func is mock_runner
    assert background_tasks.tasks[0].args == ("text-job",)
    assert background_tasks.tasks[0].kwargs == {
        "text": "Please verify this claim.",
        "external_id": None,
    }


@pytest.mark.asyncio
async def test_inbound_message_enqueues_url_job_from_telegram_payload() -> None:
    payload = InboundMessageSchema(
        platform="telegram",
        sender_id="12345",
        message_id="67890",
        text="Context note before the link https://example.com/story/123",
    )
    background_tasks = BackgroundTasks()
    session = object()
    expected_response = SimpleNamespace(job=SimpleNamespace(id="url-job"))

    with patch(
        "app.api.routes_webhooks.enqueue_url_claim",
        new_callable=AsyncMock,
        return_value=expected_response,
    ) as mock_enqueue:
        with patch("app.api.routes_webhooks.run_url_claim_job") as mock_runner:
            result = await inbound_message(
                payload,
                background_tasks=background_tasks,
                session=session,
            )

    assert result == expected_response
    mock_enqueue.assert_awaited_once_with(
        session,
        "https://example.com/story/123",
        "telegram:12345:67890",
        supporting_text="Context note before the link",
    )
    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].func is mock_runner
    assert background_tasks.tasks[0].args == ("url-job",)
    assert background_tasks.tasks[0].kwargs == {
        "url": "https://example.com/story/123",
        "external_id": "telegram:12345:67890",
        "supporting_text": "Context note before the link",
    }
