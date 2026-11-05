# DiffGraph-CLI

DiffGraph-CLI visualizes code changes from a validated canonical artifact. Its terminal-default, HTML, and JSON reports are deterministic and local; the former AI-driven HTML report remains available for one compatibility release under an explicit deprecated option.

## 🌟 Features

- 📊 Visualizes code changes as a dependency graph
- 🤖 Deprecated AI HTML compatibility mode for one release
- 🌙 Dark mode support
- 📝 Markdown-formatted summaries
- 🔍 Syntax highlighting for code blocks
- 📱 Responsive design
- 🔄 Works with both tracked and untracked files

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wild.git
cd wild
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Set up your OpenAI API key:
```bash
# Create a .env file in the project root
cp .env.example .env  # On Windows, use: type nul > .env
```
Add your OpenAI API key to the .env file

> Note: The `.env` file is git-ignored for security reasons. Make sure to keep your API key secure and never commit it to version control.

## 💻 Usage

Basic usage:
```bash
wild diff
```

This will resolve the selected local Git snapshot once, build and validate one
canonical DiffGraph artifact, then print the ranked review path to stdout. It
writes no report and does not open a browser.

To generate the interactive self-contained HTML report, choose HTML explicitly:
```bash
wild diff --format html
# or use an HTML output path shorthand:
wild diff -o diffgraph.html
```

### Command-line Options

- `--api-key`: Specify the OpenAI API key used only by `--format legacy-html`
  (defaults to `OPENAI_API_KEY`)
- `--format`: Select canonical `terminal` (default), `html`, or `json` output. `legacy-html` temporarily selects the deprecated AI report.
- `--output` or `-o`: Specify an HTML or JSON output path. Without an explicit
  `--format`, an output path selects HTML. JSON defaults to stdout. Terminal
  output is always stdout and never writes a file.
- `--no-open`: Don't automatically open the HTML report in browser
- `--structural-json`: Write a local Python structural DiffGraph v2 artifact to the given path (`-` for stdout). Applies to `wild diff` only.
- `--llm MODEL`: Opt in to AI summaries for each changed file and symbol via
  [BYOK Relay](https://byokrelay.com). `MODEL` is a provider (`claude`, `gpt`,
  or `gemini`, which selects that provider's latest flagship) or an exact model
  id such as `gemini-3.8-flash`. Sends the changed code to the model provider.
  See [AI summaries](#ai-summaries-opt-in).
- `--version`: Show version information

Example:
```bash
wild diff --output my-report.html --no-open
```

### Canonical local output (experimental)

A deterministic, network-free Python baseline can be rendered from one validated
DiffGraph v2 artifact:

```bash
wild diff                         # terminal review path on stdout
wild diff --format html --no-open  # self-contained visual report
wild diff -o diffgraph.html        # HTML shorthand; opens unless --no-open
wild diff --format json
wild diff --format json --output diffgraph.json
wild diff --format terminal
# Compatibility spelling retained for existing scripts:
wild --structural-json diffgraph.json diff
wild --structural-json staged.json diff --staged -- src/
wild --structural-json - diff -- path/to/file.py
wild diff --format json HEAD~2..HEAD
wild diff --format html main...feature -- src/
```

`--structural-json PATH` remains a compatibility alias for canonical JSON with
that destination. Do not combine it with `--format` or `--output`; ambiguous
combinations are usage errors. Likewise, `--format terminal` cannot use
`--output`. Terminal-only `--compact` and `--all` flags follow `diff`.

Canonical output supports local unstaged (`index` → working tree), staged
(`HEAD` → index), one commit against the working tree (`wild diff REF`, like
`git diff REF`, including untracked files) or the index (`wild diff --staged
REF`), explicit two-dot (`BASE..HEAD`), and explicit three-dot
(`BASE...HEAD`) comparisons. Two-dot resolves both refs to immutable commits;
three-dot resolves their merge base and compares it with the immutable head,
matching Git diff semantics. For commit ranges, `diff_ref` records both the
user-requested base/head refs (`requested_base_ref`, `requested_head_ref`) and
the resolved immutable comparison endpoints (`base_ref`, `head_ref`), plus
`comparison_mode`; every file records its pre/post blob identities. Invalid refs and unavailable merge
bases produce structured warnings instead of false changes.

Put pathspecs after `--`. Pathspecs are interpreted relative to the directory
where `wild` is invoked, matching Git's command-line behavior. Both endpoints
of a commit range are required. A single operand that is not a commit
(e.g. a mistyped ref or a pathspec without `--`) is a usage error, never an
empty diff.

Python (`.py`) is the only language with structural symbol/import extraction in
this baseline. Other changed files remain in `files[]` and receive a scoped
`UNSUPPORTED_LANGUAGE` warning. Syntax/decoding failures receive a scoped
`PARSE_FAILURE` warning and do not produce invented symbol changes. Imports
of other changed Python files resolve to that file (and, for named imports, to
the imported declaration); other imports are labeled unresolved/external.
Python call edges are emitted only for bare calls that resolve unambiguously
to a function or class in the same file or through such a resolved import.
Attribute dispatch (`obj.method()`, `module.func()`) and names shadowed by
parameters or local assignments remain absent
rather than being guessed. Every file records old/new paths, modes, Git object
IDs, and content SHA-256 values in structural evidence, while
symbol/relationship evidence names the parser package, query revision, source
blob identity, and exact call site where applicable.

**Callers outside the diff.** Unchanged Python files anywhere in the
repository (not limited by pathspecs) that call or import a changed or deleted
top-level declaration are found at the same snapshot: the head commit for
ranges, the index for staged and unstaged diffs. They are resolved with the
same rules and added as context files (`change_kind: "unchanged"`) carrying
only their calling symbols and the `calls`/`imports` edges into the change.
The terminal lists them as "called outside the diff"; the HTML graph shows
them as dashed boxes. The search covers at most 300 candidate files, with a
`PARTIAL_ANALYSIS` warning beyond that.

Unstaged analysis includes non-ignored ordinary untracked files as exact
one-sided working-tree snapshots without modifying the index. Git ignore rules
and caller-relative pathspecs are preserved.

#### CLI and offline contract

- Each canonical invocation resolves the requested Git snapshot once, builds
  one artifact, validates it against the packaged schema, and passes that same
  validated object to the HTML, JSON, or terminal consumer. Consumers never
  re-read repository files or rebuild the artifact.
- The flagless `wild diff` path uses terminal output only: it writes no files and
  launches no browser. JSON sent to stdout contains only the artifact. Terminal
  output also uses stdout. A successful JSON file write reports its path on stderr; diagnostics,
  usage help, and errors use stderr. Explicit JSON paths are replaced atomically
  and parent directories are not created implicitly.
- Exit code `0` means success, including a snapshot with no changes. Empty JSON
  has empty `files`, `symbols`, and `relationships` arrays; terminal output says
  that the selected snapshot has no changes. Runtime, validation, output, and
  cancellation failures return `1`; option/command usage errors return `2`.
  Ctrl-C prints Click's `Aborted!` diagnostic and does not dispatch an artifact
  or print a success message.
- Without `--llm`, canonical HTML, JSON, and terminal modes are local/offline. They import or
  invoke no AI or network module, make no network calls, report
  `privacy_tier: local` and `llm_calls: 0`, and use only local
  Git/object/worktree data plus packaged parser/schema resources. Canonical HTML
  is self-contained and has no external asset dependency.

#### AI summaries (opt-in)

`wild diff --llm MODEL` adds a one-sentence AI summary to every changed file
and non-import symbol, in all three formats (JSON `summary` fields, HTML graph
boxes and lists, terminal `AI:` lines). Requests go through
[BYOK Relay](https://byokrelay.com)'s OpenAI-format endpoint, which routes
`claude-*`, `gpt-*`, and `gemini-*` models (or an explicit
`anthropic/`, `openai/`, `google/` prefix) to that provider.

- **Models:** `--llm claude`, `--llm gpt`, and `--llm gemini` use each
  provider's flagship (`claude-opus-5-5`, `gpt-6-astra`, `gemini-4-argon`).
  Add a tier for speed or cost: `claude:fast` (`claude-haiku-5-5`),
  `claude:balanced` (`claude-sonnet-5-5`), `gpt:fast` (`gpt-6-luna`),
  `gpt:balanced` (`gpt-6.1-sol`), `gemini:fast` (`gemini-3.8-flash`); `:best`
  is the flagship. Any exact model id also works (e.g. `--llm
  gemini-3.8-flash`).
- **Keys:** on first use, `wild` registers with the relay and sends the
  provider key once; the relay stores it encrypted. The key comes from
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`/`GOOGLE_API_KEY`
  in your shell environment when set, otherwise from a hidden prompt in an
  interactive terminal. `wild` saves only the relay token, in
  `~/.config/wild/byok-relay.json` (mode 0600), or reads it from
  `BYOK_RELAY_TOKEN`. A self-hosted relay is selected by `BYOK_RELAY_URL` in
  the shell environment or `"relay_url"` in that config file.
- **Trust boundary:** `wild` auto-loads a `.env` from the current directory,
  which may belong to an untrusted repository. Values from any `.env` are
  never used for the relay URL, the relay token, or provider keys, so a
  cloned repository cannot redirect your credentials or code.
- **What is sent:** the unified diff of each changed text file at the exact
  analyzed snapshot (working-tree content is re-verified against its recorded
  blob id; binary files are skipped), plus node ids and names.
- **Contract:** the model only writes prose for existing nodes. Unknown ids
  are discarded, topology is never changed, and the artifact reports
  `schema_version: "2.1"`, `privacy_tier: cloud_llm`, the provider, model, and
  call count. Failed requests become scoped `LLM_ERROR`/`LLM_TIMEOUT`
  warnings rather than failing the run.

#### Deprecated AI HTML compatibility

For this compatibility release only, `wild diff --format legacy-html` retains
old AI analysis, progress output, default `diffgraph.html` destination,
`--output`, `--no-open`, and browser-opening behavior. This option is deprecated
and scheduled for removal after one release. It may call the configured AI
provider, and its old renderer is isolated in `diffgraph/html_report.py`; that
legacy renderer still loads Mermaid, Tailwind, Highlight.js, and Marked from
external CDNs. Canonical `--format html` does not import that module or any AI
SDK.

### Artifact compatibility

DiffGraph artifacts use a `MAJOR.MINOR` `schema_version`. Consumers reject
malformed versions and unknown major versions. Minor releases within major 2
are additive: a consumer accepts them only when the complete artifact still
validates against its packaged v2 schema. This fail-closed rule lets producers
add optional data without weakening validation for existing consumers. The
canonical schema and a complete local-only example are packaged under
`diffgraph/schema/`; neither contains AI-derived symbols or relationships.

Snapshots containing NUL bytes produce a scoped `PARTIAL_ANALYSIS` warning.
Their exact pre/post object IDs and content hashes remain in file evidence,
while language, line counts, symbols, and relationships are left unset rather
than treating binary bytes as source code.

## 📊 Example Output

The canonical HTML report is the interactive graph:
- A full-window graph of changed files and symbols (unchanged symbols one
  click away per file), with optional AI summaries in each box
- A details panel for the selected file, symbol, or edge: summary, location,
  calls, callers, importers, and evidence
- A top bar with the snapshot, change counts, privacy/AI status, and warnings
- Light and dark themes; self-contained and offline

The raw artifact is not shown in the page; use `wild diff --format json`.

The deprecated `legacy-html` report retains its Mermaid.js diagram,
syntax-highlighted code blocks, and AI-generated summary during the one-release
compatibility window.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Mermaid.js](https://mermaid.js.org/) for graph visualization
- [Highlight.js](https://highlightjs.org/) for syntax highlighting
- [Tailwind CSS](https://tailwindcss.com/) for styling
- [OpenAI](https://openai.com/) for AI capabilities
