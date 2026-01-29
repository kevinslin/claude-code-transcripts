"""Tests for SQLite conversation storage."""

import os
import sqlite3
import time

from claude_code_transcripts import (
    build_conversation_record,
    init_conversations_db,
    index_conversations,
    reindex_conversations,
    store_conversation_record,
)


def test_init_conversations_db_creates_table_and_indexes(tmp_path):
    db_path = tmp_path / "conversations.db"
    init_conversations_db(db_path)

    with sqlite3.connect(db_path) as conn:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
        ).fetchone()
        assert table is not None
        state_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='conversation_index_state'"
        ).fetchone()
        assert state_table is not None

        index_rows = conn.execute("PRAGMA index_list('conversations')").fetchall()
        index_names = {row[1] for row in index_rows}
        assert "conversations_created_at_idx" in index_names
        assert "conversations_published_at_idx" in index_names


def test_store_conversation_record_inserts_row(tmp_path):
    db_path = tmp_path / "conversations.db"
    record = {
        "id": "session-123",
        "title": "Hello",
        "body": "User: Hello\n\nAssistant: Hi",
        "created_at": "2025-01-01T00:00:00Z",
        "published_at": None,
        "updated_at": "2025-01-01T00:00:10Z",
    }

    store_conversation_record(db_path, record)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, title, body, created_at, published_at, updated_at "
            "FROM conversations WHERE id = ?",
            (record["id"],),
        ).fetchone()

    assert row == (
        record["id"],
        record["title"],
        record["body"],
        record["created_at"],
        record["published_at"],
        record["updated_at"],
    )


def test_build_conversation_record_from_jsonl(tmp_path):
    session_file = tmp_path / "sample.jsonl"
    session_file.write_text(
        '{"type": "user", "timestamp": "2025-01-01T10:00:00Z", '
        '"message": {"role": "user", "content": "Hello"}}\n'
        '{"type": "assistant", "timestamp": "2025-01-01T10:00:05Z", '
        '"message": {"role": "assistant", "content": "Hi there"}}\n'
    )

    record = build_conversation_record(session_file, agent="claude")

    assert record["id"] == "sample"
    assert record["title"]
    assert "User: Hello" in record["body"]
    assert "Assistant: Hi there" in record["body"]
    assert record["created_at"] == "2025-01-01T10:00:00Z"
    assert record["updated_at"] == "2025-01-01T10:00:05Z"


def _write_session_file(folder, name, timestamp, mtime):
    session_file = folder / f"{name}.jsonl"
    session_file.write_text(
        f'{{"type": "user", "timestamp": "{timestamp}", '
        '"message": {"role": "user", "content": "Hello"}}\n'
        f'{{"type": "assistant", "timestamp": "{timestamp}", '
        '"message": {"role": "assistant", "content": "Hi"}}\n'
    )
    os.utime(session_file, (mtime, mtime))
    return session_file


def test_index_conversations_incremental(tmp_path):
    db_path = tmp_path / "conversations.db"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    base_time = time.time()
    _write_session_file(
        sessions_dir, "session-a", "2025-01-01T10:00:00Z", base_time
    )
    _write_session_file(
        sessions_dir,
        "session-b",
        "2025-01-02T10:00:00Z",
        base_time + 10,
    )

    first = index_conversations(db_path, sessions_dir, agent="claude")
    assert first["indexed_count"] == 2
    assert first["skipped_count"] == 0

    second = index_conversations(db_path, sessions_dir, agent="claude")
    assert second["indexed_count"] == 0
    assert second["skipped_count"] == 2

    updated_time = base_time + 20
    _write_session_file(
        sessions_dir, "session-b", "2025-01-03T10:00:00Z", updated_time
    )

    third = index_conversations(db_path, sessions_dir, agent="claude")
    assert third["indexed_count"] == 1
    assert third["skipped_count"] == 1


def test_reindex_conversations_forces_full_rebuild(tmp_path):
    db_path = tmp_path / "conversations.db"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_session_file(
        sessions_dir, "session-a", "2025-01-01T10:00:00Z", time.time()
    )

    index_conversations(db_path, sessions_dir, agent="claude")
    second = reindex_conversations(db_path, sessions_dir, agent="claude")

    assert second["indexed_count"] == 1
