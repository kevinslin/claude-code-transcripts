"""Tests for the server index page listing."""

from claude_code_transcripts import (
    list_conversations,
    render_index_page,
    store_conversation_record,
)


def _seed_conversations(db_path):
    store_conversation_record(
        db_path,
        {
            "id": "alpha",
            "title": "Alpha convo",
            "body": "Body A",
            "created_at": "2025-01-02T10:00:00Z",
            "published_at": "2025-02-01T10:00:00Z",
            "updated_at": "2025-01-02T10:00:00Z",
        },
    )
    store_conversation_record(
        db_path,
        {
            "id": "bravo",
            "title": "Bravo convo",
            "body": "Body B",
            "created_at": "2025-01-03T10:00:00Z",
            "published_at": None,
            "updated_at": "2025-01-03T10:00:00Z",
        },
    )
    store_conversation_record(
        db_path,
        {
            "id": "charlie",
            "title": "Charlie convo",
            "body": "Body C",
            "created_at": "2025-01-01T10:00:00Z",
            "published_at": "2025-03-01T10:00:00Z",
            "updated_at": "2025-01-01T10:00:00Z",
        },
    )


def test_list_conversations_sorts_by_created_desc(tmp_path):
    db_path = tmp_path / "conversations.db"
    _seed_conversations(db_path)

    conversations = list_conversations(db_path, sort_by="created")
    ids = [conv["id"] for conv in conversations]

    assert ids[:3] == ["bravo", "alpha", "charlie"]


def test_list_conversations_sorts_by_published_desc(tmp_path):
    db_path = tmp_path / "conversations.db"
    _seed_conversations(db_path)

    conversations = list_conversations(db_path, sort_by="published")
    ids = [conv["id"] for conv in conversations]

    assert ids[:3] == ["charlie", "alpha", "bravo"]


def test_render_index_page_includes_sort_toggle_and_dates(tmp_path):
    db_path = tmp_path / "conversations.db"
    _seed_conversations(db_path)

    conversations = list_conversations(db_path, sort_by="published")
    page = render_index_page(conversations, sort_by="published")

    assert "Sort by published" in page
    assert "Sort by created" in page
    assert "Published:" in page
    assert "Unpublished" in page
    assert 'id="search-form"' in page
    assert 'id="search-input"' in page
    assert 'id="search-results"' in page
