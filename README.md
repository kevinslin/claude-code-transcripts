# llm-transcripts

[![PyPI](https://img.shields.io/pypi/v/claude-code-transcripts.svg)](https://pypi.org/project/claude-code-transcripts/)
[![Changelog](https://img.shields.io/github/v/release/simonw/claude-code-transcripts?include_prereleases&label=changelog)](https://github.com/simonw/claude-code-transcripts/releases)
[![Tests](https://github.com/simonw/claude-code-transcripts/workflows/Test/badge.svg)](https://github.com/simonw/claude-code-transcripts/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/claude-code-transcripts/blob/main/LICENSE)

Convert Claude Code and Codex session files (JSON or JSONL) to clean, mobile-friendly HTML pages with pagination.

[Example transcript](https://static.simonwillison.net/static/2025/claude-code-microjs/index.html) produced using this tool.

Read [A new way to extract detailed transcripts from Claude Code](https://simonwillison.net/2025/Dec/25/claude-code-transcripts/) for background on this project.

## Installation

Install this tool using `uv` (installs the `llm-transcripts` command):
```bash
uv tool install claude-code-transcripts
```
Or run it without installing:
```bash
uvx --from claude-code-transcripts llm-transcripts --help
```

## Usage

This tool converts Claude Code or Codex session files into browseable multi-page HTML transcripts.

Select an agent by passing it before the command, or set `LLM_AGENT`:

```bash
llm-transcripts claude local
llm-transcripts codex local

# Or set once for all commands
export LLM_AGENT=claude
llm-transcripts local
```

Examples below assume `LLM_AGENT` is set; otherwise pass the agent first (e.g., `llm-transcripts codex local`).

There are four commands available:

- `local` (default) - select from local sessions (`~/.claude/projects` or `~/.codex/sessions`)
- `web` - select from web sessions via the Claude API (Claude only)
- `json` - convert a specific JSON or JSONL session file
- `all` - convert all local sessions to a browsable HTML archive

The quickest way to view a recent local session:

```bash
LLM_AGENT=claude llm-transcripts
```

This shows an interactive picker to select a session, generates HTML, and opens it in your default browser.

### Output options

All commands support these options:

- `-o, --output DIRECTORY` - output directory (default: writes to temp dir and opens browser)
- `-a, --output-auto` - auto-name output subdirectory based on session ID or filename
- `--repo OWNER/NAME` - GitHub repo for commit links (auto-detected from git push output if not specified)
- `--open` - open the generated `index.html` in your default browser (default if no `-o` specified)
- `--gist` - upload the generated HTML files to a GitHub Gist and output a preview URL
- `--json` - include the original session file in the output directory

The generated output includes:
- `index.html` - an index page with a timeline of prompts and commits
- `page-001.html`, `page-002.html`, etc. - paginated transcript pages

### Local sessions

Local sessions are stored as JSONL files in `~/.claude/projects` (Claude) or `~/.codex/sessions` (Codex). Run with no arguments to select from recent sessions:

```bash
llm-transcripts
# or explicitly:
llm-transcripts local
```

Use `--limit` to control how many sessions are shown (default: 10):

```bash
llm-transcripts local --limit 20
```

### Web sessions

Import sessions directly from the Claude API:

```bash
# Interactive session picker
llm-transcripts web

# Import a specific session by ID
llm-transcripts web SESSION_ID

# Import and publish to gist
llm-transcripts web SESSION_ID --gist
```

Web sessions are only available for the Claude agent.

On macOS, API credentials are automatically retrieved from your keychain (requires being logged into Claude Code). On other platforms, provide `--token` and `--org-uuid` manually.

### Publishing to GitHub Gist

Use the `--gist` option to automatically upload your transcript to a GitHub Gist and get a shareable preview URL:

```bash
llm-transcripts --gist
llm-transcripts web --gist
llm-transcripts json session.json --gist
```

This will output something like:
```
Gist: https://gist.github.com/username/abc123def456
Preview: https://gisthost.github.io/?abc123def456/index.html
Files: /var/folders/.../session-id
```

The preview URL uses [gisthost.github.io](https://gisthost.github.io/) to render your HTML gist. The tool automatically injects JavaScript to fix relative links when served through gisthost.

Combine with `-o` to keep a local copy:

```bash
llm-transcripts json session.json -o ./my-transcript --gist
```

**Requirements:** The `--gist` option requires the [GitHub CLI](https://cli.github.com/) (`gh`) to be installed and authenticated (`gh auth login`).

### Auto-naming output directories

Use `-a/--output-auto` to automatically create a subdirectory named after the session:

```bash
# Creates ./session_ABC123/ subdirectory
llm-transcripts web SESSION_ABC123 -a

# Creates ./transcripts/session_ABC123/ subdirectory
llm-transcripts web SESSION_ABC123 -o ./transcripts -a
```

### Including the source file

Use the `--json` option to include the original session file in the output directory:

```bash
llm-transcripts json session.json -o ./my-transcript --json
```

This will output:
```
JSON: ./my-transcript/session_ABC.json (245.3 KB)
```

This is useful for archiving the source data alongside the HTML output.

### Converting from JSON/JSONL files

Convert a specific session file directly:

```bash
llm-transcripts json session.json -o output-directory/
llm-transcripts json session.jsonl --open
```
This works with JSONL files in `~/.claude/projects/` or `~/.codex/sessions/` and JSON session files extracted from Claude Code for web.

The `json` command can take a URL to a JSON or JSONL file as an alternative to a path on disk.

### Converting all sessions

Convert all your local sessions to a browsable HTML archive:

```bash
llm-transcripts all
```

This creates a directory structure with:
- A master index listing all projects
- Per-project pages listing sessions
- Individual session transcripts

Options:

- `-s, --source DIRECTORY` - source directory (default: `~/.claude/projects` or `~/.codex/sessions`)
- `-o, --output DIRECTORY` - output directory (default: `./claude-archive` or `./codex-archive`)
- `--include-agents` - include agent session files (excluded by default)
- `--dry-run` - show what would be converted without creating files
- `--open` - open the generated archive in your default browser
- `-q, --quiet` - suppress all output except errors

Examples:

```bash
# Preview what would be converted
llm-transcripts all --dry-run

# Convert all sessions and open in browser
llm-transcripts all --open

# Convert to a specific directory
llm-transcripts all -o ./my-archive

# Include agent sessions
llm-transcripts all --include-agents
```

## Development

To contribute to this tool, first checkout the code. You can run the tests using `uv run`:
```bash
cd claude-code-transcripts
uv run pytest
```
And run your local development copy of the tool like this:
```bash
uv run llm-transcripts --help
```
