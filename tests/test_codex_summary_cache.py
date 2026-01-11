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
    _select_session_with_refresh,
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


def test_build_codex_summary_prompt_includes_roles(tmp_path):
    session_file = tmp_path / "session.jsonl"
    _write_codex_session(session_file, "User asks", "Assistant replies")

    prompt = build_codex_summary_prompt(session_file)
    assert "User: User asks" in prompt
    assert "Assistant: Assistant replies" in prompt


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
