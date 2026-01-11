# Execution Plan: Codex summary cache for local sessions

**Date:** 2026-01-10
**Status:** Completed

---

## Goal

Add a Codex summary cache so local session lists use stored conversation summaries,
generate missing summaries in the background, and refresh the local picker when
generation completes.

---

## Context

### Background
Codex local sessions are currently summarized on-the-fly from the session file
(first user message). We want persistent summaries to improve browsing and avoid
recomputing every time.

### Current State
- `local` uses `find_local_sessions()` -> `get_session_summary()` and renders a
  questionary picker.
- Codex summaries come from `_get_codex_jsonl_summary()` (first user message).
- No summary cache; no background generation.

### Constraints
- Python + Click CLI; keep Claude behavior unchanged.
- Summaries stored at `$HOME/.config/llm-transcripts/codex/{session-name}.summary.md`.
- If a summary is missing, use the filename as the displayed summary.
- Summaries should generate in the background and trigger a picker refresh when done.

---

## Technical Approach

### Architecture/Design
- Add a Codex summary cache layer (read/write to `~/.config/llm-transcripts/codex`).
- Introduce a summary generator that builds a conversation text prompt from parsed
  loglines and writes a Markdown summary file.
- Update `local` for agent `codex` to:
  - Prefer cached summaries.
  - Fallback to filename for missing summaries.
  - Kick off background generation for missing summaries.
  - Refresh the picker when background generation completes.

### Technology Stack
- Python 3, Click, questionary, existing JSONL parsing.
- Optional external summarizer if we choose to invoke an LLM CLI.

### Integration Points
- `local_cmd` flow (session list + questionary picker).
- `find_local_sessions()` / `get_session_summary()` for Codex paths.
- New cache directory under `~/.config/llm-transcripts/codex`.

### Design Patterns
- Cache-first lookup with fallback.
- Background work via a thread pool or background subprocesses.
- Atomic writes for summary files (write temp, rename).

---

## Steps

### Phase 1: Design + tests
- [x] Confirm summary generation approach (LLM CLI vs heuristic fallback).
- [x] Add failing tests for Codex summary cache lookup and fallback behavior.
- [x] Add tests around background generation triggers and picker refresh logic
      (mocked).

### Phase 2: Implementation + validation
- [x] Add cache helpers for Codex summary read/write and file paths.
- [x] Implement Codex summary generation from conversation text.
- [x] Wire Codex local session listing to use cache + fallback.
- [x] Add background generation for missing summaries and refresh behavior.
- [x] Update docs if needed and run `uv run pytest`.

**Dependencies between phases:**
- Phase 2 depends on summary generation approach from Phase 1.

---

## Dependencies

### External Services/APIs
- Optional: LLM summarizer (if we choose to call an external CLI).

### Libraries/Packages
- No new Python dependencies planned unless a summarizer is added.

### Tools/Infrastructure
- Local Codex sessions in `~/.codex/sessions`.
- Summary cache in `~/.config/llm-transcripts/codex`.

### Access Required
- [ ] Any credentials needed if we invoke an external summarizer CLI.

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|---------------------|
| Summarizer missing or slow | Med | Med | Provide a fallback summary (filename) and keep async generation optional |
| Picker refresh is clunky in terminal UI | Med | Med | Re-run selection once summaries finish and notify user |
| Summary generation reads too much data | Low | Med | Limit prompt length and strip to recent messages |

---

## Questions

### Technical Decisions Needed
- [x] Which summarizer should we use (external LLM CLI, API, or heuristic summary)?
      - run "codex" CLI to do summaries
- [x] Should we regenerate summaries if the session file changes, or only when missing?
      - only when missing

### Clarifications Required
- [x] Does "refresh the page" mean re-running the questionary picker after
      summaries finish, even if the user already selected a session?
      - only refresh if user has not selected
- [x] Should background generation run for all sessions or only those displayed
      by `--limit`?
      - only those displayed by limit

### Research Tasks
- [x] Inspect Codex JSONL content to decide how to build a summarization prompt
      (which roles/messages to include).

---

## Success Criteria

- [x] Codex summary files are stored under
      `~/.config/llm-transcripts/codex/{session-name}.summary.md`.
- [x] `local` displays cached summaries for Codex sessions when available.
- [x] Missing summaries show filename and are generated in the background.
- [x] Local picker refreshes once all background summaries complete.
- [x] Tests added and `uv run pytest` passes.

---

## Notes

- No `DESIGN.md` found; plan kept to two phases for simplicity.
- Recent commits primarily added Codex parsing and CLI agent support.
- Ran `uv run pytest`; no README updates needed.
