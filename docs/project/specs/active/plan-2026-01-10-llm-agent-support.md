# Execution Plan: Add agent-aware CLI and Codex support

**Date:** 2026-01-10
**Status:** Completed

---

## Goal

Extend the CLI to support multiple agents (Claude + Codex) with an optional agent
parameter (or `LLM_AGENT`), rename the CLI program, and ensure Codex sessions can be
parsed and rendered without breaking existing Claude behavior.

---

## Context

### Background
The tool currently targets Claude Code sessions only. We need a single CLI that can
convert both Claude and Codex transcripts, with an explicit agent selector and a
renamed entry point.

### Current State
- CLI program name: `claude-code-transcripts`
- Agent-specific assumptions baked in:
  - Local sessions in `~/.claude/projects`
  - JSONL parsing assumes Claude logline schema (`type=user|assistant`, `message`)
  - Web import uses Claude API credentials
- Renderer expects Claude-style content blocks (`text`, `tool_use`, `tool_result`)

### Constraints
- Python + Click CLI; preserve existing behavior for Claude
- TDD workflow, keep changes minimal and incremental
- Agent parameter optional only when `LLM_AGENT` is set
- Avoid breaking existing users (consider backward compatibility)

---

## Technical Approach

### Architecture/Design
- Introduce agent resolution (CLI arg -> env var -> fallback) to drive paths,
  parsing, and command behavior.
- Normalize Codex JSONL into the existing internal `loglines` shape so the
  renderer stays unchanged.
- Gate agent-specific features (e.g., Claude web API) when agent != `claude`.

### Technology Stack
- Python 3, Click, existing renderer + Jinja templates
- Codex parsing implemented in Python without new dependencies

### Integration Points
- CLI entry point and Click group (new program name + agent arg)
- Session discovery (`local`/`all`) and parse functions
- Documentation (README, help text, usage examples)
- Packaging metadata (`pyproject.toml` scripts entry)

### Design Patterns
- Per-agent configuration mapping (paths, parsers, command availability)
- Data normalization to keep renderer and templates stable

---

## Steps

### Phase 1: Research and decisions
- [x] Inspect Codex session JSONL format to map message/tool events to renderer blocks
- [x] Decide CLI naming (`llm-transcripts` vs `code-transcripts`) and compatibility strategy
- [x] Confirm agent resolution rules (required vs default `claude`)
- [x] Decide how Codex sessions should be grouped for `local`/`all`

### Phase 2: Implementation and validation
- [x] Add agent resolution helper and wire into CLI (agent arg + `LLM_AGENT`)
- [x] Update session discovery for Codex paths and grouping
- [x] Extend JSONL parsing to handle Codex loglines and normalize to existing schema
- [x] Update docs and help text for new CLI name + usage
- [x] Add tests for agent resolution and Codex parsing; run `uv run pytest`

**Dependencies between phases:**
- Phase 2 depends on decisions from Phase 1 (agent behavior + CLI naming)

---

## Dependencies

### External Services/APIs
- Claude API (existing `web` command, agent=`claude` only)

### Libraries/Packages
- None beyond current dependencies

### Tools/Infrastructure
- Local session directories: `~/.claude/projects`, `~/.codex/sessions`

### Access Required
- [ ] Claude API credentials if `web` remains Claude-only

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|---------------------|
| Codex JSONL schema differs from renderer assumptions | High | Med | Add a Codex normalization layer with fixture-based tests |
| CLI rename breaks existing users | Med | Med | Provide alias or clear migration path in docs |
| Agent mismatch leads to empty results | Med | Low | Validate agent values and surface friendly errors |

---

## Questions

### Technical Decisions Needed
- [x] Should we keep `claude-code-transcripts` as a backward-compatible alias?
  - no
- [x] Should agent default to `claude` when neither arg nor `LLM_AGENT` is set?
  - there should be no default. prompt user for an entry

### Clarifications Required
- [x] Confirm CLI name: `llm-transcripts` vs the example `code-transcripts`
  - llm-transcripts
- [x] Should `web` be disabled or error for `codex` agent?
  - disabled for now. we will implement it later
- [x] Expected grouping for Codex `local`/`all` (by date, by cwd, or flat list)?
  - should be same behavior as calude code

### Research Tasks
- [x] Identify Codex tool-call/result event shapes to map into renderer blocks

---

## Success Criteria

- [x] CLI supports `llm-transcripts [agent] <command>` with `LLM_AGENT` fallback
- [x] Claude workflows behave as before, Codex sessions render without errors
- [x] README and help text reflect new CLI name and agent usage
- [x] Tests updated/added and passing

---

## Notes

- No existing `docs/project/specs` structure; created and seeded from exec-plan template.
- Using `plan-YYYY-MM-DD-...` filename to match repo shortcut naming.
- Simplified to two phases to keep the plan lightweight.
- Agent selection is required unless `LLM_AGENT` is set; `web` is Claude-only for now.
