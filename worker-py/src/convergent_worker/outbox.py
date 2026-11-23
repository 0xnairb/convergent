"""Transactional outbox relay publishing durable NATS JetStream events."""

from __future__ import annotations

import json
from typing import Any

import psycopg


async def relay_outbox_once(conn: psycopg.Connection, js: Any, batch_size: int = 50) -> int:
    """Publish unpublished outbox rows to NATS JetStream with Nats-Msg-Id deduplication."""
    published = 0
    with conn.transaction():
        rows = conn.execute(
            """
            SELECT id, msg_id, subject, payload
            FROM outbox_events
            WHERE published_at IS NULL
            ORDER BY id ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
            """,
            (batch_size,),
        ).fetchall()

        for row_id, msg_id, subject, payload in rows:
            data = json.dumps(payload).encode("utf-8")
            await js.publish(subject, data, headers={"Nats-Msg-Id": msg_id})
            conn.execute(
                "UPDATE outbox_events SET published_at = now() WHERE id = %s",
                (row_id,),
            )
            published += 1
    return published
