# Architecture: llm-transcripts

**Date:** 2026-01-27
**Status:** Current

---

## Overview
llm-transcripts is a CLI that converts Claude Code and Codex session files into static, mobile-friendly HTML transcripts with pagination, optional gist publishing, and a local search UI.
It works by normalizing agent-specific JSON/JSONL logs into a shared logline schema, rendering HTML with Jinja2 templates, and indexing conversation summaries in SQLite for fuzzy search.

## Scope

### In scope
- Local session conversion from `~/.claude/projects` (Claude) and `~/.codex/sessions` (Codex).
- Claude web session import via the Anthropic API.
- Static HTML transcript generation (index + paginated pages).
- Local search server (`/` and `/api/search`) plus client-side transcript search.
- Optional gist publishing and browser launch.

### Out of scope
- Editing or mutating session data.
- Remote hosting or user authentication.
- Semantic/vector search.
- Full-text search across raw session files without indexing.

## Architecture goals
- Keep rendering logic agent-agnostic by normalizing inputs.
- Produce fully static HTML outputs that are shareable and offline-friendly.
- Keep dependencies small and the CLI easy to run.
- Provide incremental indexing for fast local search.

## System context
Actors and boundaries:
- User runs `llm-transcripts` commands locally.
- Session data comes from local files or the Claude API.
- Outputs are written to local directories or uploaded via `gh gist`.
- Optional local HTTP server provides search UI and API.

Conceptual diagram:
```
User
  |
  v
CLI (llm-transcripts)
  |-- read local session files (.jsonl/.json)
  |-- fetch web sessions (Claude API)
  |-- optional Codex summary via `codex` CLI
  |-- optional gist via `gh` CLI
  v
HTML output (index.html + page-XXX.html) --> Browser
  |
  v
SQLite index (~/.config/llm-transcripts/conversations.db)
  |
  v
Local search server (http://127.0.0.1:3010)
```

## High-level design
- Single Python module (`src/claude_code_transcripts/__init__.py`) contains CLI, parsing, rendering, indexing, and server logic.
- Jinja2 templates under `src/claude_code_transcripts/templates` render the UI; CSS/JS are inlined.
- A normalized "loglines" format is the internal contract between parsers and renderers.

## Components

### CLI and command layer
- Implementation: Click group `cli()` with `AgentDefaultGroup` to parse an optional agent prefix; entrypoint `main()` is exposed via the `llm-transcripts` script.
- Commands: `local`, `json`, `web`, `serve`, `reindex`, `all`.
- Agent resolution: `resolve_agent()` + `require_agent()` uses `LLM_AGENT` or prompts when absent.

### Session discovery and summarization
- Local session discovery: `find_local_sessions()` and `find_all_sessions()` using `default_sessions_root()` and filters for agent files and empty sessions.
- Project naming: `get_project_display_name()` converts Claude folder encodings into readable names.
- Codex summary cache: `codex_summary_path()` / `read_codex_summary()` store summaries in `~/.config/llm-transcripts/codex`.
- Background summary generation: `start_codex_summary_generation()` runs `codex exec` with prompts from `build_codex_summary_prompt()` and refreshes the picker when done.

### Parsing and normalization
- Entry point: `parse_session_file()` picks JSON vs JSONL and agent.
- Claude JSONL: `_parse_jsonl_file()` converts Claude loglines to `{"type","timestamp","message"}`.
- Codex JSONL: `_parse_codex_jsonl_file()` maps Codex `response_item` payloads into normalized loglines and converts function calls to tool blocks.
- Content normalization: `normalize_codex_content()` ensures content blocks match renderer expectations.

### Rendering pipeline
- Grouping: `generate_html()` and `generate_html_from_session_data()` group loglines by user prompts into "conversations".
- Rendering: `render_message()` -> `render_user_message_content()` / `render_assistant_message()` -> `render_content_block()` -> Jinja2 macros.
- Templates:
  - `base.html`, `page.html`, `index.html` for single-session output.
  - `master_index.html`, `project_index.html` for archives (`all` command).
  - `macros.html` defines markup for tool calls, commits, pagination, and more.
- Client-side search: `templates/search.js` scans generated transcript pages in the browser (disabled on `file://` due to CORS).
- Commit linking: `detect_github_repo()` finds repo URLs in tool output; commits render as cards with links when a repo is known.

### Search indexing and API
- SQLite DB at `~/.config/llm-transcripts/conversations.db` with tables `conversations` and `conversation_index_state`.
- Indexing: `index_conversations()` stores `build_conversation_record()` results and tracks `last_indexed_at` by file mtime.
- Fuzzy search: `_fuzzy_score()` uses token-level SequenceMatcher scoring; `search_conversations()` returns ranked results with snippets.
- Server: `create_search_server()` uses `ThreadingHTTPServer`, serving `/` (index page) and `/api/search`.

### Publishing and preview support
- Gist publishing: `create_gist()` shells out to `gh gist create` and returns the gist ID/URL.
- Gist preview fixes: `inject_gist_preview_js()` appends JS that rewrites links and fixes fragment navigation on gist preview hosts.
- Browser launch: `webbrowser.open()` is used when requested or implied by default output behavior.

### Batch archive generation
- `generate_batch_html()` iterates projects and sessions to build a full archive with a master index and per-project indexes.

## Data flows

### `local` command
1. Resolve agent (`claude` or `codex`) and determine the session root.
2. Incrementally index sessions into SQLite.
3. List recent sessions; for Codex, spawn background summary generation and refresh selection when done.
4. User selects a session via questionary.
5. Generate HTML output (and optional JSON copy).
6. Optional gist publishing and browser launch.

### `json` command
1. Accept path or URL; fetch URL to a temp file if needed.
2. Generate HTML output from the session file.
3. Optional JSON copy, gist publishing, and browser launch.

### `web` command (Claude only)
1. Resolve token/org UUID (keychain or CLI).
2. List sessions via `/sessions` and prompt for selection if needed.
3. Fetch a session via `/session_ingress/session/{id}`.
4. Generate HTML output from session data.
5. Optional JSON save, gist publishing, and browser launch.

### `serve` command
1. Index sessions into SQLite.
2. Start the HTTP server.
3. Serve index HTML at `/` and JSON search API at `/api/search`.

### `all` command
1. Scan all session files grouped by project.
2. Generate transcript HTML into `output/{project}/{session}/`.
3. Generate `project_index.html` per project and `master_index.html` at output root.

## Data model
- Normalized logline: `{"type": "user"|"assistant", "timestamp": str, "message": {"role": str, "content": [blocks]}}`.
- Content blocks: `text`, `thinking`, `tool_use`, `tool_result`, `image`.
- Conversation record (SQLite): `id`, `title`, `body`, `created_at`, `published_at`, `updated_at`.

## Storage and state
- Input sessions:
  - Claude: `~/.claude/projects/**/*.jsonl`
  - Codex: `~/.codex/sessions/**/*.jsonl`
- Claude web session config: `~/.claude.json` for org UUID.
- Codex summary cache: `~/.config/llm-transcripts/codex/*.summary.md`
- Search DB: `~/.config/llm-transcripts/conversations.db`
- Output:
  - Single session: `index.html`, `page-001.html`, etc.
  - Archive: `index.html` + `/{project}/{session}/` structure.

## External integrations
- Anthropic API for web sessions (Claude only).
- `codex` CLI for summary generation (optional).
- `gh` CLI for gist publishing (optional).
- Default web browser for preview.

## Configuration
- `LLM_AGENT` environment variable or CLI agent prefix (`llm-transcripts claude ...`).
- CLI flags for output directory, gist, JSON copy, open browser, and search server host/port.
- `--source` to override session root for `all`, `serve`, `reindex`.

## Performance characteristics
- Rendering is linear in logline count; pagination uses `PROMPTS_PER_PAGE` (default 5 prompts per page).
- Indexing is incremental based on file mtime; search scans all stored conversations in memory.
- Codex summary generation runs asynchronously in a background thread.

## Security and privacy
- All processing is local by default; session content only leaves the machine if the user chooses `web` or `--gist`.
- API tokens are retrieved from macOS keychain or CLI args; no token storage beyond local config.
- Gist publishing may expose transcripts publicly depending on `gh` defaults.

## Operational considerations
- Runtime requirements: Python 3.10+, click, jinja2, markdown, httpx, questionary.
- Tests use pytest with snapshot fixtures under `tests/`.
- No long-running daemon except the optional `serve` HTTP server.

## Risks and tradeoffs
- Fuzzy search uses a custom token matcher rather than FTS; it favors simplicity over speed at very large scale.
- The monolithic module simplifies distribution but concentrates responsibilities in one file.
- Gist publishing and Codex summaries depend on external CLIs and network access.
