"""Tests for Codex summary caching and refresh behavior."""

import asyncio
import os
from pathlib import Path
import threading

import pytest

from claude_code_transcripts import (
    AGENT_CODEX,
    find_local_sessions,
    codex_summary_path,
    read_codex_summary,
    build_codex_summary_prompt,
    _build_local_session_choices,
    _select_session_with_refresh,
    CODEX_SUMMARY_MAX_CHARS,
    CODEX_SUMMARY_SESSION_MARKER,
)


def _write_codex_session(path: Path, user_text: str, assistant_text: str = "") -> None:
    lines = [
        '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"%s"}]}}'
        % user_text,
    ]
    if assistant_text:
        lines.append(
            '{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"%s"}]}}'
            % assistant_text
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_codex_summary_path_and_read(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    session_file = tmp_path / "session-123.jsonl"
    session_file.write_text("{}\n", encoding="utf-8")

    summary_path = codex_summary_path(session_file)
    assert (
        summary_path
        == tmp_path / ".config" / "llm-transcripts" / "codex" / "session-123.summary.md"
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("Summary line\nSecond line", encoding="utf-8")

    assert read_codex_summary(session_file) == "Summary line Second line"


def test_find_local_sessions_codex_uses_cache_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    sessions_dir = tmp_path / ".codex" / "sessions" / "project"
    sessions_dir.mkdir(parents=True)

    session_a = sessions_dir / "session-a.jsonl"
    session_b = sessions_dir / "session-b.jsonl"

    _write_codex_session(session_a, "Hello A")
    _write_codex_session(session_b, "Hello B")

    summary_path = codex_summary_path(session_a)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("Cached summary", encoding="utf-8")

    results, missing = find_local_sessions(
        sessions_dir, limit=10, agent=AGENT_CODEX, include_missing=True
    )

    summaries = {path.name: summary for path, summary in results}
    assert summaries["session-a.jsonl"] == "Cached summary"
    assert summaries["session-b.jsonl"] == "session-b"
    assert missing == [session_b]


def test_local_choices_show_codex_id_and_summary_column(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    sessions_dir = tmp_path / ".codex" / "sessions" / "project"
    sessions_dir.mkdir(parents=True)

    session_id = "019bf728-cbca-7882-91e8-a1c992ef58eb"
    session_file = sessions_dir / "rollout-2026-01-25T19-48-48-019bf86b-1eea-79d1.jsonl"
    session_file.write_text(
        '{"type":"session_meta","payload":{"id":"%s","cwd":"/proj"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Hello"}]}}\n'
        % session_id,
        encoding="utf-8",
    )

    summary_path = codex_summary_path(session_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("Cached summary", encoding="utf-8")

    results = find_local_sessions(sessions_dir, limit=10, agent=AGENT_CODEX)
    choices = _build_local_session_choices(results, agent=AGENT_CODEX)

    title = choices[0].title
    assert session_id in title
    assert "Cached summary" in title
    assert "rollout-2026-01-25T19-48-48-019bf86b-1eea-79d1" not in title


def test_build_codex_summary_prompt_includes_roles(tmp_path):
    session_file = tmp_path / "session.jsonl"
    _write_codex_session(session_file, "User asks", "Assistant replies")

    prompt = build_codex_summary_prompt(session_file)
    assert CODEX_SUMMARY_SESSION_MARKER in prompt
    assert "User: User asks" in prompt
    assert "Assistant: Assistant replies" in prompt


def test_find_local_sessions_skips_summary_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    sessions_dir = tmp_path / ".codex" / "sessions" / "project"
    sessions_dir.mkdir(parents=True)

    summary_session = sessions_dir / "summary.jsonl"
    summary_session.write_text(
        '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Marker: llm-transcripts-summary"}]}}\n',
        encoding="utf-8",
    )

    normal_session = sessions_dir / "normal.jsonl"
    _write_codex_session(normal_session, "Hello")

    results, _ = find_local_sessions(
        sessions_dir, limit=10, agent=AGENT_CODEX, include_missing=True
    )

    paths = [path for path, _ in results]
    assert summary_session not in paths
    assert normal_session in paths


def test_summary_session_detection_handles_missing_instructions(tmp_path):
    session_file = tmp_path / "session.jsonl"
    session_file.write_text(
        '{"type":"session_meta","payload":{"instructions":null}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Hello"}]}}\n',
        encoding="utf-8",
    )

    results, _ = find_local_sessions(
        tmp_path, limit=10, agent=AGENT_CODEX, include_missing=True
    )
    assert any(path == session_file for path, _ in results)


def test_build_codex_summary_prompt_truncates_long_messages(tmp_path):
    session_file = tmp_path / "session.jsonl"
    long_text = "x" * (CODEX_SUMMARY_MAX_CHARS + 500)
    _write_codex_session(session_file, long_text, "Assistant reply")

    prompt = build_codex_summary_prompt(session_file)
    assert prompt is not None
    conversation = prompt.split("Conversation:\n", 1)[1].rsplit("\n\nSummary:", 1)[0]
    assert len(conversation) <= CODEX_SUMMARY_MAX_CHARS


def test_select_session_refreshes_after_background_completion():
    refresh_event = threading.Event()
    exit_event = asyncio.Event()
    build_calls = []

    def build_choices():
        build_calls.append(1)
        return ["choice"]

    class DummyApp:
        def __init__(self, exit_event):
            self._exit_event = exit_event

        def exit(self, result=None):
            if self._exit_event:
                self._exit_event.set()

    class DummyQuestion:
        def __init__(self, result, exit_event=None):
            self.application = DummyApp(exit_event)
            self._result = result
            self._exit_event = exit_event

        async def ask_async(self, *args, **kwargs):
            if self._exit_event is not None:
                await self._exit_event.wait()
            return self._result

    prompts = [DummyQuestion(None, exit_event), DummyQuestion("selected")]

    def prompt_factory(message, choices):
        return prompts.pop(0)

    async def run_select():
        async def trigger_refresh():
            await asyncio.sleep(0.01)
            refresh_event.set()

        asyncio.create_task(trigger_refresh())
        return await _select_session_with_refresh(
            "Pick", build_choices, refresh_event, prompt_factory
        )

    result = asyncio.run(run_select())
    assert result == "selected"
    assert len(build_calls) == 2
