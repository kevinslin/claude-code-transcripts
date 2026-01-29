# Ralph Agent Instructions

## Overview

Ralph is an autonomous AI agent loop that runs AI coding tools (Amp, Claude Code, or OpenAI Codex) repeatedly until all PRD items are complete. Each iteration is a fresh instance with clean context.

## Commands

```bash
# Run the flowchart dev server
cd flowchart && npm run dev

# Build the flowchart
cd flowchart && npm run build

# Run Ralph with Amp (default)
./ralph.sh [max_iterations]

# Run Ralph with Claude Code
./ralph.sh --tool claude [max_iterations]

# Run Ralph with OpenAI Codex
./ralph.sh --tool codex [max_iterations]
```

## Key Files

- `ralph.sh` - The bash loop that spawns fresh AI instances (supports `--tool amp`, `--tool claude`, or `--tool codex`)
- `prompt.md` - Instructions given to each AMP instance
-  `CLAUDE.md` - Instructions given to each Claude Code instance
- `CODEX.md` - Instructions given to each OpenAI Codex instance
- `prd.json.example` - Example PRD format
- `flowchart/` - Interactive React Flow diagram explaining how Ralph works

## Flowchart

The `flowchart/` directory contains an interactive visualization built with React Flow. It's designed for presentations - click through to reveal each step with animations.

To run locally:
```bash
cd flowchart
npm install
npm run dev
```

## Patterns

- Each iteration spawns a fresh AI instance (Amp, Claude Code, or OpenAI Codex) with clean context
- For Codex, use `codex exec` for non-interactive runs; the interactive CLI requires a TTY
- Memory persists via git history, `progress.txt`, and `prd.json`
- Stories should be small enough to complete in one context window
- Always update AGENTS.md with discovered patterns for future iterations
- Core CLI and helper functions live in `src/claude_code_transcripts/__init__.py`
- The search API is served via `llm-transcripts serve` (defaults to 127.0.0.1:3010).
- The `/` route on `llm-transcripts serve` renders the conversations index page and supports `?sort=created|published`.
- The index page search UI is rendered in `render_index_page` and fetches `/api/search`.
- For global CLI availability, install with `uv tool install --editable .` (from repo) or `uv tool install claude-code-transcripts` (from PyPI) and ensure `~/.local/bin` is on PATH.
