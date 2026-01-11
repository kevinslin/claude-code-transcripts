"""Pytest configuration and fixtures for claude-code-transcripts tests."""

import pytest


@pytest.fixture(autouse=True)
def mock_webbrowser_open(monkeypatch):
    """Automatically mock webbrowser.open to prevent browsers opening during tests."""
    opened_urls = []

    def mock_open(url):
        opened_urls.append(url)
        return True

    monkeypatch.setattr("claude_code_transcripts.webbrowser.open", mock_open)
    return opened_urls


@pytest.fixture(autouse=True)
def default_agent_env(monkeypatch):
    """Default to claude agent for CLI tests unless overridden."""
    monkeypatch.setenv("LLM_AGENT", "claude")
