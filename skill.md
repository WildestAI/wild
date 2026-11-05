# skill: wild-diff

Use this skill to analyse git diffs with WildestAI's `wild` CLI and get a structural understanding of code changes.

## What it does

`wild diff` deterministically analyses your current Git diff and prints a ranked
review path to stdout. Canonical HTML and JSON reports are available explicitly:
- A dependency graph of changed components and their relationships
- An offline self-contained HTML report (`--format html` or `-o report.html`)
- A schema-v2 JSON artifact for IDEs and CI (`--format json`)
- Optional deprecated AI HTML compatibility (`--format legacy-html`)

## Prerequisites

```bash
# Install from source (Python 3.10+)
git clone https://github.com/WildestAI/wild && cd wild && pip install -e .
```

No API key is needed: the terminal, HTML, and JSON formats are local and
offline. AI summaries are opt-in (`--llm`, see below). The deprecated
`--format legacy-html` is the exception: it sends code to OpenAI and loads
external CDN assets, so avoid it for private code.

## Commands

```bash
# Analyse unstaged + untracked changes (most common)
wild diff

# Analyse only staged changes
wild diff --staged

# Compare a commit (or branch) with the working tree, like `git diff REF`
wild diff HEAD
wild diff main

# Compare two commits, or a branch since it forked from main
wild diff HEAD~3..HEAD
wild diff main...feature

# Limit to paths: put them after `--`
wild diff main -- path/to/file.py

# Machine-readable artifact (schema 2.1) for agents and CI
wild diff --format json main...feature

# Interactive offline HTML report, don't auto-open (`-o` alone selects HTML)
wild diff -o report.html --no-open

# Opt-in one-sentence AI summaries per changed file and symbol
wild diff --llm claude:fast        # Haiku; also claude, gpt, gemini, gpt:fast, gemini:fast

# All other git commands pass through transparently
wild log
wild blame path/to/file.py
wild status
```

## Output

By default, writes the terminal review path to stdout without creating files or
opening a browser. Use `--format html` for the interactive offline report; it
opens automatically unless `--no-open` is passed. `--output <path>` without an
explicit format is an HTML shorthand.

## MCP Server

If you need programmatic access (agent-to-agent), start the MCP server:

```bash
cd path/to/DiffGraph-CLI
python mcp_server.py
```

The MCP server exposes tools:
- `run_wild_diff(repo_path, args, output_file)` — run wild diff and return structured analysis; `output_file` is optional (defaults to `<repo_path>/diffgraph.html`)
- `get_docs(page)` — retrieve a documentation page
- `list_docs()` — list available documentation pages
- `search_docs(query)` — search across docs

## Configuration

AI summaries (`--llm`) go through BYOK Relay. Provider keys come from the
shell environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`)
or a hidden prompt; they are never read from a repository's `.env`.
`BYOK_RELAY_TOKEN` / `BYOK_RELAY_URL` (shell only) select an existing relay
token or a self-hosted relay.

## Notes

- Requires Python 3.10+; tested on macOS and Linux
- Structural analysis covers Python; other files are listed with their diffs
- The CLI wraps `git` — it must be run inside a git repository
- Exit codes: 0 success (including no changes), 1 runtime failure, 2 usage error
