# PRD: Conversation Search (Fuzzy Match)

## Introduction/Overview

Build a conversation search feature that supports fuzzy matching across all stored conversations. The feature includes an index page listing all conversations (sortable by created or published date), a search UI, and an API. The app is available at `localhost:3010`, and conversations are stored in a SQLite database with automatic, incremental indexing.

## Goals

- Provide fast fuzzy search across all conversations.
- Automatically index all conversations incrementally without full reindexing.
- Offer an index page listing all conversations, sortable by created or published date.
- Serve the UI and API from `localhost:3010` in development.

## User Stories

### US-001: Store conversations in SQLite
**Description:** As a developer, I want conversations stored in SQLite so they persist and can be indexed for search.

**Acceptance Criteria:**
- [ ] Conversations are persisted in a SQLite database table (e.g., `conversations`).
- [ ] Each conversation has: `id`, `title`, `body` (or transcript), `created_at`, `published_at` (nullable), `updated_at`.
- [ ] Add indexes for `created_at` and `published_at` to support sorting.
- [ ] Typecheck/lint passes.

### US-002: Incremental indexing on startup
**Description:** As a user, I want new or updated conversations to be indexed automatically so search results are up to date.

**Acceptance Criteria:**
- [ ] On app startup, an indexing job runs automatically.
- [ ] Indexing only processes conversations that are new or have changed since last index run (incremental).
- [ ] Indexing is idempotent; running it twice with no changes does not modify the index.
- [ ] Provide a manual “Reindex” action for recovery/debugging.
- [ ] Typecheck/lint passes.

### US-003: Fuzzy search API
**Description:** As a user, I want search results even when I mistype so I can still find the right conversation.

**Acceptance Criteria:**
- [ ] `GET /api/search?query=...` returns matching conversations ordered by relevance.
- [ ] Fuzzy matching tolerates small typos (e.g., query “teh” matches “the”).
- [ ] Empty query returns an empty list (not all conversations).
- [ ] Response includes `id`, `title`, `created_at`, `published_at`, and a short snippet.
- [ ] Typecheck/lint passes.

### US-004: Index page with sort toggle
**Description:** As a user, I want to browse all conversations and sort by created or published date.

**Acceptance Criteria:**
- [ ] `GET /` shows an index page listing all conversations.
- [ ] Default sort order is by `created_at` descending.
- [ ] User can toggle sort by `published_at` descending.
- [ ] Each list item shows title and date (created or published, depending on sort).
- [ ] Typecheck/lint passes.
- [ ] Verify in browser using dev-browser skill.

### US-005: Search UI
**Description:** As a user, I want a search box so I can search conversations directly in the UI.

**Acceptance Criteria:**
- [ ] Index page includes a search input.
- [ ] Submitting a query shows search results without leaving the page.
- [ ] Results show title, date, and snippet.
- [ ] No results state is shown when nothing matches.
- [ ] Typecheck/lint passes.
- [ ] Verify in browser using dev-browser skill.

## Functional Requirements

- FR-1: The system must use SQLite as the primary storage for conversations.
- FR-2: The system must provide incremental indexing of conversations on startup.
- FR-3: The system must support fuzzy matching across all conversations.
- FR-4: The system must expose a search API endpoint at `/api/search`.
- FR-5: The UI must be available at `localhost:3010` in development.
- FR-6: The index page must list all conversations sorted by `created_at` descending by default.
- FR-7: The index page must allow sorting by `published_at` descending.
- FR-8: The UI must include a search input and display results in-page.

## Non-Goals (Out of Scope)

- No authentication or user accounts.
- No semantic/vector search or AI-generated summaries.
- No tagging, filtering, or faceted search beyond sorting.
- No remote sync; all data is local SQLite.
- No editing or deletion of conversations from the UI.

## Design Considerations (Optional)

- Index page layout: search bar at top, sort toggle near the list header.
- Date display should be consistent and readable (e.g., `YYYY-MM-DD`).
- Snippets should highlight or clearly indicate matched text if feasible.

## Technical Considerations (Optional)

- SQLite FTS5 or trigram-based search is acceptable for fuzzy matching.
- Track incremental indexing via `updated_at` or content hash per conversation.
- Consider a separate FTS table to avoid duplicating data in the primary table.
- Ensure index build does not block server startup longer than necessary.

## Success Metrics

- A user can find a known conversation with a 1–2 character typo.
- Search results return in under 200ms for 10k conversations on a typical dev machine.
- New or updated conversations are searchable within one startup cycle.

## Open Questions

- What should be displayed as the “date” when `published_at` is null?
- Should search results be sorted purely by relevance or allow secondary date sort?
- Do we need to expose conversation detail pages, or only list/search views?
