# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2026-10-26

### Added
- `wild diff REF` compares a commit with the working tree (like
  `git diff REF`), and `wild diff --staged REF` with the index. A single
  operand that is not a commit is a usage error instead of an empty diff.
- `--llm` tiers: `claude:fast` (Haiku), `claude:balanced`, `gpt:fast`,
  `gpt:balanced`, `gemini:fast`, and `:best` (the default flagship).
- HTML graph: "Fit to view" and "Fit width" controls; the wheel zooms
  around the cursor, and the graph refits as its panel resizes.
- `inherits` edges for bare-name base classes, only to a confirmed class:
  local, imported from a changed file, or a changed or deleted class
  subclassed outside the diff. External bases keep only their import symbol.
- Callers outside the diff: unchanged files that call or import changed or
  deleted declarations are found at the analyzed snapshot and included as
  context files with their call sites (terminal "called outside the diff",
  dashed boxes in the HTML graph).
- Schema 2.1: changed symbols carry their exact before/after source, and
  text files carry a capped unified diff.
- `wild diff --llm MODEL`: opt-in AI summaries for each changed file and
  symbol through BYOK Relay (Claude, GPT, and Gemini models), rendered in
  JSON, HTML, and terminal output. Schema 2.1 adds optional per-node
  `summary` fields.
- Resolved `from module import name` bindings are recorded as file →
  declaration `imports` edges.

### Changed
- The HTML report is graph-first: a full-window graph, a details panel for the
  selected node, and a top bar. The textual file/symbol/topology/metadata
  dumps are gone (use `--format json`), roughly halving report size.
- Graph files show changed symbols by default; unchanged symbols and import
  statements expand per file, or all at once with "Show unchanged".
- The graph reads top to bottom: modules, files, and symbols are ordered so
  callers sit above the code they call (changed source first, then files
  outside the diff, then tests), and "◆ start here" marks the entry point.
- Modules are collapsible (collapsed by default from five modules, keeping
  the first open); a collapsed module summarizes its files and changed
  symbols, and edges into it attach to its header.
- Graph edges are drawn on top as arcs beside the boxes: symbol-to-symbol
  when both are visible, otherwise lifted to file headers and merged with a
  count; hovering or selecting a node highlights its connections.
- Selecting a symbol shows its exact before/after diff (folded context,
  unified or side by side); selecting a file shows its unified diff.

### Security
- `--llm` never takes the relay URL, relay token, or provider keys from a
  `.env` file, which may come from an untrusted repository; a self-hosted relay
  is accepted only from the shell environment or `~/.config/wild/byok-relay.json`.
- AI summary text is stripped of control characters before terminal output.

### Fixed
- Terminal "imported by" is credited per symbol instead of to every changed
  symbol in an imported file; import statements are no longer ranked as
  imported.
- The HTML graph is readable in dark mode.
- Terminal warnings render as readable lines.

## [1.2.0] - 2026-10-09

### Added
- Deterministic, offline DiffGraph v2 reports for Python changes, with JSON,
  terminal, and self-contained HTML output.
- Structural evidence for Python symbols, imports, calls, bindings, constants,
  type aliases, decorators, and Python stub files.
- Exact Git provenance for requested ranges and snapshots, including SHA-256
  repositories, unsupported files, parser failures, and binary changes.
- Canonical HTML topology links and an offline content-security policy.

### Changed
- Improved structural handling for local-import rebinding, comprehensions,
  match patterns, `with`/`except` bindings, and named expressions.
- Strengthened Git diff validation for ambiguous merge bases, incomplete object
  IDs, and paths outside the repository.

### Fixed
- Preserve explicit warnings and provenance where structural analysis falls
  back or encounters malformed Python.

## [1.1.1] - 2026-09-13

### Fixed
- Treat incomplete GitHub pull-request metadata as non-matching release evidence instead of failing the release-eligibility classifier.

## [1.0.0] - 2025-08-06

### Changed
- **BREAKING CHANGE**: Renamed CLI command from `diffgraph-ai` to `wild`
  - Updated all documentation to reflect the new command name
  - Updated setup.py entry point to use `wild` command
  - All usage examples now use `wild` instead of `diffgraph-ai`
  - This is a breaking change - users will need to update their scripts and documentation

### Fixed
- Fixed `--version` command not working after CLI rename by explicitly specifying `package_name='wild'` in `@click.version_option()`
- Improved path handling in environment loader by wrapping paths with `os.path.normpath()` for better cross-platform compatibility
- Removed redundant `hasattr(sys, 'executable')` check as `sys.executable` is always available in Python

## [0.1.0] - 2025-06-06

### Added
- Initial release of DiffGraph CLI tool
- AI-powered code change analysis
- HTML report generation with Mermaid.js dependency graphs
- Support for both tracked and untracked files
- Dark mode support in HTML reports
- Syntax highlighting for code blocks
- Responsive design for all screen sizes