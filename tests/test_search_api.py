"""Tests for conversation search API and fuzzy matching."""

import threading

import httpx

from claude_code_transcripts import create_search_server, search_conversations, store_conversation_record


def _store_record(db_path, conv_id, title, body):
    store_conversation_record(
        db_path,
        {
            "id": conv_id,
            "title": title,
            "body": body,
            "created_at": "2025-01-01T00:00:00Z",
            "published_at": None,
            "updated_at": "2025-01-01T00:00:00Z",
        },
    )


def test_search_conversations_fuzzy_match(tmp_path):
    db_path = tmp_path / "conversations.db"
    _store_record(
        db_path,
        "conv-1",
        "The Quick Fox",
        "The quick brown fox jumps over the lazy dog.",
    )

    results = search_conversations(db_path, "teh")
    assert results
    assert results[0]["id"] == "conv-1"
    assert results[0]["snippet"]


def test_search_conversations_empty_query_returns_empty(tmp_path):
    db_path = tmp_path / "conversations.db"
    _store_record(db_path, "conv-1", "Hello", "Hello world")
    assert search_conversations(db_path, "") == []


def test_search_api_endpoint_returns_results(tmp_path):
    db_path = tmp_path / "conversations.db"
    _store_record(db_path, "conv-1", "Hello", "The quick brown fox")

    server = create_search_server(db_path, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        response = httpx.get(f"http://{host}:{port}/api/search", params={"query": "teh"})
        assert response.status_code == 200
        payload = response.json()
        assert isinstance(payload, list)
        assert payload[0]["id"] == "conv-1"
    finally:
        server.shutdown()
        server.server_close()
