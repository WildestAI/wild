import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import click
from click.core import ParameterSource
from typing import Dict, List, Optional, Tuple
import os
from diffgraph import __version__
from diffgraph.artifact import (
    build_validated_artifact,
    render_canonical_json,
    write_canonical_json,
)
from diffgraph.contract import DiffGraphContractError, validate_artifact
from diffgraph.env_loader import load_env_file, debug_environment
from diffgraph.git_snapshot import GitSnapshotError, UnknownRevisionError
from diffgraph.utils import sanitize_diff_args, involves_working_tree
from diffgraph.structural import StructuralDependencyError

# Load environment variables
load_env_file()

def is_git_repo() -> bool:
    """Check if current directory is a git repository."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _open_html_report(html_path: Path) -> None:
    """Best-effort browser launch; report failures without failing generation."""
    try:
        if sys.platform == 'darwin':
            result = subprocess.run(['open', str(html_path)], check=False)
        elif sys.platform == 'win32':
            os.startfile(html_path)
            return
        else:
            result = subprocess.run(['xdg-open', str(html_path)], check=False)
    except OSError as error:
        click.echo(f"⚠️ Could not open report in browser: {error}", err=True)
        return

    if result.returncode != 0:
        click.echo(
            "⚠️ Could not open report in browser: "
            f"opener exited with status {result.returncode}",
            err=True,
        )


def get_changed_files(diff_args: List[str] = None) -> List[Dict[str, str]]:
    """
    Get list of changed and untracked files.
    Returns a list of dicts with 'path' and 'status' keys.
    """
    if diff_args is None:
        diff_args = []

    changed_files = []

    # Get modified/staged files
    try:
        sanitized_args, pathspecs = sanitize_diff_args(diff_args)
        cmd = ["git", "diff", "--name-only"] + sanitized_args
        if pathspecs:
            cmd += ["--"] + pathspecs
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        for file_path in result.stdout.strip().split('\n'):
            if file_path:  # Skip empty lines
                changed_files.append({
                    'path': file_path,
                    'status': 'modified'
                })
    except subprocess.CalledProcessError as e:
        click.echo(f"Error getting modified files: {e}", err=True)
        sys.exit(1)

    # Decide if we should include untracked files
    if involves_working_tree(diff_args):
        try:
            # Use git ls-files for native untracked file detection
            # --others: show untracked files
            # --exclude-standard: respect .gitignore patterns
            # -z: null-byte separated output for reliable parsing
            result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                check=True,
                capture_output=True,
                text=True
            )

            # Split on null byte and filter out empty strings
            untracked_files = [path for path in result.stdout.split('\0') if path.strip()]

            for file_path in untracked_files:
                changed_files.append({
                    'path': file_path,
                    'status': 'untracked'
                })
        except subprocess.CalledProcessError as e:
            click.echo(f"Error getting untracked files: {e}", err=True)
            sys.exit(1)

    return changed_files

class _RawArgsCommand(click.Command):
    """Retain the raw separator that Click removes from variadic arguments."""

    def parse_args(self, ctx, args):
        ctx.meta["raw_args"] = tuple(args)
        return super().parse_args(ctx, args)


def _terminal_options(
    diff_args, explicit_pathspecs: Optional[Tuple[str, ...]] = None
):
    """Remove terminal-only flags without consuming pathspecs after ``--``."""
    option_args = list(diff_args)
    pathspec_args = []
    if explicit_pathspecs:
        pathspec_count = len(explicit_pathspecs)
        if tuple(option_args[-pathspec_count:]) == explicit_pathspecs:
            option_args, pathspec_args = (
                option_args[:-pathspec_count],
                option_args[-pathspec_count:],
            )

    remaining = []
    compact = False
    show_all = False
    for arg in option_args:
        if arg == "--compact":
            compact = True
        elif arg == "--all":
            show_all = True
        else:
            remaining.append(arg)
    return remaining + pathspec_args, compact, show_all


def _pathspecs_after_diff_separator(raw_args) -> Optional[Tuple[str, ...]]:
    """Return raw pathspecs after ``diff --``, preserving range-like names."""

    value_options = {"--api-key", "--output", "-o", "--structural-json", "--format"}
    index = 0
    while index < len(raw_args):
        argument = raw_args[index]
        if argument in value_options:
            index += 2
            continue
        if any(argument.startswith(option + "=") for option in value_options):
            index += 1
            continue
        if argument.startswith("-o") and argument != "-o":
            index += 1
            continue
        if argument == "diff":
            trailing = raw_args[index + 1 :]
            if "--" not in trailing:
                return None
            separator_index = trailing.index("--")
            return tuple(trailing[separator_index + 1 :])
        index += 1
    return None


@dataclass(frozen=True)
class _StructuralScope:
    staged: bool = False
    pathspecs: Tuple[str, ...] = ()
    base_ref: Optional[str] = None
    head_ref: Optional[str] = None
    three_dot: bool = False
    against: Optional[str] = None


def _range_operand(argument: str):
    """Return exact two/three-dot endpoints, or ``None`` for another operand."""
    if "...." in argument:
        raise click.UsageError(
            "commit ranges require explicit non-empty BASE..HEAD or BASE...HEAD refs"
        )
    separator = "..." if "..." in argument else ".." if ".." in argument else None
    if separator is None:
        return None
    endpoints = argument.split(separator)
    if len(endpoints) != 2 or not all(endpoints):
        raise click.UsageError(
            "commit ranges require explicit non-empty BASE..HEAD or BASE...HEAD refs"
        )
    return endpoints[0], endpoints[1], separator == "..."


def _structural_scope(
    diff_args: List[str], explicit_pathspecs: Optional[Tuple[str, ...]] = None
):
    """Parse the deterministic local or immutable commit comparison scope."""
    arguments = list(diff_args)
    if "--" in arguments:
        separator_index = arguments.index("--")
        explicit_pathspecs = tuple(arguments[separator_index + 1 :])
        arguments = arguments[:separator_index]
    elif explicit_pathspecs is not None and explicit_pathspecs:
        count = len(explicit_pathspecs)
        if tuple(arguments[-count:]) != explicit_pathspecs:
            raise click.UsageError("could not preserve pathspec scope after '--'")
        arguments = arguments[:-count]

    staged = False
    base_ref = head_ref = against = None
    three_dot = False
    for argument in arguments:
        if argument in ("--staged", "--cached"):
            if base_ref is not None:
                raise click.UsageError(
                    "--staged/--cached cannot be combined with a commit range"
                )
            staged = True
            continue
        commit_range = _range_operand(argument)
        if commit_range is None:
            # One commit, like ``git diff REF`` (or ``--staged REF``).
            if argument.startswith("-") or against is not None or base_ref is not None:
                raise click.UsageError(
                    "canonical output supports unstaged, --staged/--cached, REF, "
                    "--staged REF, or BASE..HEAD/BASE...HEAD diffs; put pathspecs after '--'"
                )
            against = argument
            continue
        if staged or base_ref is not None or against is not None:
            raise click.UsageError("use exactly one local mode or commit range")
        base_ref, head_ref, three_dot = commit_range

    return _StructuralScope(
        staged,
        explicit_pathspecs or (),
        base_ref,
        head_ref,
        three_dot,
        against,
    )


def _validate_structural_artifact(artifact):
    """Backward-compatible validation helper for callers outside the CLI path."""
    try:
        validate_artifact(artifact)
    except DiffGraphContractError as error:
        raise click.ClickException(f"structural artifact validation failed: {error}") from error


def load_file_contents(changed_files: List[Dict[str, str]], diff_args: List[str] = None) -> List[Dict[str, str]]:
    """
    Load contents of changed files.
    For modified files, gets the diff content.
    For untracked files, reads the entire file.
    """
    if diff_args is None:
        diff_args = []

    files_with_content = []

    for file_info in changed_files:
        file_path = file_info['path']
        status = file_info['status']

        try:
            if status == 'modified':
                # Get diff content for modified files with sanitized args and proper separator
                sanitized_args, _ = sanitize_diff_args(diff_args)
                cmd = ["git", "diff"] + sanitized_args + ["--", file_path]
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                content = result.stdout
            else:  # untracked
                # Read entire file for untracked files
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

            files_with_content.append({
                'path': file_path,
                'status': status,
                'content': content
            })
        except (subprocess.CalledProcessError, IOError) as e:
            click.echo(f"Error reading file {file_path}: {e}", err=True)
            continue

    return files_with_content

def _stdin_is_interactive() -> bool:
    return sys.stdin.isatty()


def _enrich_with_llm(artifact, model: str):
    """Add per-node AI summaries through BYOK Relay (explicit opt-in only).

    Network-capable modules are imported here, lazily, so canonical output
    without --llm never loads them. All progress goes to stderr so JSON on
    stdout stays a pure artifact.
    """
    from diffgraph.contract import enrich_with_node_summaries
    from diffgraph.llm_summaries import PROMPT_REF, summarize_artifact
    from diffgraph.relay import (
        MissingProviderKey,
        RelayClient,
        RelayError,
        load_token,
        provider_key_candidates,
        relay_url,
        resolve_model,
        save_token,
    )

    try:
        provider, model = resolve_model(model)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    interactive = _stdin_is_interactive()
    base_url = relay_url()
    client = RelayClient(base_url, load_token(base_url))

    def require_interactive(what: str) -> None:
        if not interactive:
            raise click.ClickException(
                f"{what}; run 'wild diff --llm {model}' once in an interactive "
                "terminal to set up BYOK Relay, or set BYOK_RELAY_TOKEN"
            )

    candidates = provider_key_candidates(provider)

    def store_provider_key() -> None:
        click.echo(
            f"🔑 wild sends your {provider} key once to BYOK Relay ({base_url}), "
            "which stores it encrypted and never returns it. wild keeps only "
            "the relay token.",
            err=True,
        )
        key = None
        if len(candidates) > 1:
            labels = "; ".join(candidate.label for candidate in candidates)
            if not interactive:
                raise click.ClickException(
                    f"found different {provider} keys: {labels}. Unset one, or "
                    "run interactively to choose"
                )
            for index, candidate in enumerate(candidates, 1):
                click.echo(f"  {index}. {candidate.label}", err=True)
            choice = click.prompt(
                "Which key should be sent",
                type=click.IntRange(1, len(candidates)),
                err=True,
            )
            key = candidates[choice - 1].key
        elif candidates:
            candidate = candidates[0]
            if not interactive or click.confirm(
                f"Send {candidate.label} to {base_url}?", default=True, err=True
            ):
                click.echo(f"🔑 Using {candidate.label}.", err=True)
                key = candidate.key
        if key is None:
            require_interactive(f"no {provider} key is stored on BYOK Relay")
            key = click.prompt(f"{provider} API key", hide_input=True, err=True)
        client.store_key(provider, key)

    try:
        if not client.token:
            if not candidates:
                require_interactive("no BYOK Relay token is configured")
            client.register()
            path = save_token(base_url, client.token)
            click.echo(f"🔐 BYOK Relay token saved to {path}", err=True)
            store_provider_key()

        def progress(done: int, total: int) -> None:
            click.echo(f"🧠 Summarizing with {model} ({done}/{total})...", err=True)

        click.echo(
            f"📤 Sending the changed code to {provider} via BYOK Relay...", err=True
        )
        try:
            result = summarize_artifact(artifact, client, model, progress=progress)
        except MissingProviderKey:
            store_provider_key()
            result = summarize_artifact(artifact, client, model, progress=progress)
    except MissingProviderKey as error:
        raise click.ClickException(f"BYOK Relay: {error}") from error
    except RelayError as error:
        raise click.ClickException(f"BYOK Relay: {error}") from error

    try:
        return enrich_with_node_summaries(
            artifact,
            result.summaries,
            provider=provider,
            model=model,
            llm_calls=result.llm_calls,
            prompt_ref=PROMPT_REF,
            warnings=result.warnings,
        )
    except (DiffGraphContractError, ValueError) as error:
        raise click.ClickException(f"could not attach AI summaries: {error}") from error


@click.command(
    cls=_RawArgsCommand,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.version_option(version=__version__, prog_name='wild')
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
@click.option(
    '--api-key',
    envvar='OPENAI_API_KEY',
    help='OpenAI API key (legacy-html only)',
)
@click.option(
    '--output', '-o', default=None,
    help='Output path (HTML default: diffgraph.html; JSON default: stdout)',
)
@click.option(
    '--format',
    'output_format',
    type=click.Choice(
        ['html', 'terminal', 'json', 'legacy-html'], case_sensitive=False
    ),
    default='terminal',
    show_default=True,
    help=(
        'Render a canonical local terminal, HTML, or JSON artifact; '
        'legacy-html retains the deprecated AI report for one compatibility release'
    ),
)
@click.option('--no-open', is_flag=True, help='Do not open the HTML report automatically')
@click.option('--debug-env', is_flag=True, help='Debug environment variable loading')
@click.option(
    '--structural-json',
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the local Python structural DiffGraph v2 artifact ('-' for stdout)",
)
@click.option(
    '--llm',
    'llm_model',
    metavar='MODEL',
    default=None,
    help=(
        'Add AI summaries to each changed file and symbol through BYOK Relay. '
        'MODEL is a provider (claude, gpt, gemini: its latest flagship), a '
        'provider tier (e.g. claude:fast for Haiku; tiers: fast, balanced, '
        'best), or an exact model id (e.g. gemini-3.8-flash). Sends the '
        'changed code to the model provider.'
    ),
)
def main(
    args,
    api_key: str,
    output: str,
    output_format: str,
    no_open: bool,
    debug_env: bool,
    structural_json: Path,
    llm_model: Optional[str],
):
    """wild - Git wrapper CLI with DiffGraph for diff commands."""

    context = click.get_current_context()
    format_was_explicit = (
        context.get_parameter_source("output_format") != ParameterSource.DEFAULT
    )
    # A destination implies the self-contained HTML report unless the caller
    # explicitly selected another format. This keeps `wild diff -o report.html`
    # ergonomic while the flagless command remains terminal-only.
    if output is not None and not format_was_explicit:
        output_format = "html"
    canonical_output_requested = (
        format_was_explicit or output is not None or structural_json is not None
    )
    if structural_json is not None and (not args or args[0] != "diff"):
        raise click.UsageError("--structural-json can only be used with 'diff'")
    if canonical_output_requested and (not args or args[0] != "diff"):
        raise click.UsageError(f"--format {output_format} can only be used with 'diff'")
    if structural_json is not None and format_was_explicit:
        raise click.UsageError("--structural-json cannot be combined with --format")
    if structural_json is not None and output is not None:
        raise click.UsageError("--structural-json cannot be combined with --output")
    if output_format == "terminal" and output is not None:
        raise click.UsageError("--format terminal writes to stdout and cannot use --output")
    if llm_model is not None and (not args or args[0] != "diff"):
        raise click.UsageError("--llm can only be used with 'diff'")
    if llm_model is not None and output_format == "legacy-html":
        raise click.UsageError("--llm cannot be combined with --format legacy-html")
    if debug_env and canonical_output_requested:
        raise click.UsageError(
            "--debug-env cannot be combined with canonical artifact output"
        )

    # Check if this is a diff command
    if args and args[0] == 'diff':
        # Handle diff command with custom logic
        diff_args = list(args[1:])  # Skip 'diff' and pass remaining args

        # Debug environment variable loading if requested
        if debug_env:
            debug_environment(api_key)
            return

        if not is_git_repo():
            click.echo("❌ Error: Not a git repository", err=True)
            sys.exit(1)

        if structural_json is not None or output_format in ("html", "terminal", "json"):
            compact = False
            show_all = False
            raw_args = click.get_current_context().meta.get("raw_args", ())
            explicit_pathspecs = _pathspecs_after_diff_separator(raw_args)
            if output_format == "terminal":
                diff_args, compact, show_all = _terminal_options(
                    diff_args, explicit_pathspecs=explicit_pathspecs
                )
            scope = _structural_scope(
                diff_args, explicit_pathspecs=explicit_pathspecs
            )
            try:
                artifact = build_validated_artifact(
                    ".",
                    staged=scope.staged,
                    pathspecs=scope.pathspecs,
                    base_ref=scope.base_ref,
                    head_ref=scope.head_ref,
                    three_dot=scope.three_dot,
                    against=scope.against,
                    wild_version=__version__,
                )
            except UnknownRevisionError as error:
                raise click.UsageError(str(error)) from error
            except (
                GitSnapshotError,
                StructuralDependencyError,
                DiffGraphContractError,
            ) as error:
                raise click.ClickException(str(error)) from error
            if llm_model is not None:
                artifact = _enrich_with_llm(artifact, llm_model)
            if structural_json is None and output_format == "terminal":
                from diffgraph.formatters.terminal import TerminalFormatter

                try:
                    TerminalFormatter(
                        artifact,
                        compact=compact,
                        max_items=(
                            None if show_all else TerminalFormatter.DEFAULT_MAX_ITEMS
                        ),
                    ).render()
                except ValueError as error:
                    raise click.ClickException(str(error)) from error
            elif structural_json is None and output_format == "html":
                from diffgraph.formatters.html import HtmlFormatter

                destination = Path(output or "diffgraph.html")
                try:
                    html_path = HtmlFormatter(artifact).write(destination)
                except (OSError, TypeError, ValueError) as error:
                    raise click.ClickException(
                        f"could not write {destination}: {error}"
                    ) from error
                click.echo(f"✅ HTML report generated: {html_path}")
                if not no_open:
                    click.echo("🌐 Opening report in browser...")
                    _open_html_report(html_path)
            else:
                destination = structural_json if structural_json is not None else output
                if destination is None or str(destination) == "-":
                    rendered = render_canonical_json(artifact)
                    click.echo(rendered, nl=False)
                else:
                    destination = Path(destination)
                    try:
                        write_canonical_json(artifact, destination)
                    except OSError as error:
                        raise click.ClickException(
                            f"could not write {destination}: {error}"
                        ) from error
                    label = "Structural" if structural_json is not None else "Canonical"
                    click.echo(f"✅ {label} DiffGraph written: {destination}", err=True)
            return

        # One-release compatibility path. Keep it lazy so canonical output never
        # imports a network-capable SDK. Remove after the documented transition.
        try:
            from click_spinner import spinner
            from diffgraph.ai_analysis import CodeAnalysisAgent
            from diffgraph.html_report import generate_html_report, AnalysisResult
        except ImportError as error:
            raise click.ClickException(
                f"The AI report path requires additional dependencies: {error}"
            ) from error

        click.echo("🔍 Scanning for changed files...")
        changed_files = get_changed_files(diff_args)

        if not changed_files:
            click.echo("ℹ️ No changes to analyze")
            sys.exit(0)

        click.echo(f"📝 Found {len(changed_files)} changed files")

        # Load contents of changed files with progress bar
        with click.progressbar(changed_files, label='📖 Loading file contents') as files:
            files_with_content = load_file_contents(files, diff_args)

        try:
            # Initialize the AI analysis agent
            click.echo("🤖 Initializing AI analysis...")
            agent = CodeAnalysisAgent(api_key=api_key)

            # Define progress callback
            def progress_callback(current_file, total_files, status):
                if current_file is None:
                    click.echo("📊 Generating final diagram...")
                    return

                file_name = os.path.basename(current_file)
                current_index = len(agent.graph_manager.processed_files) + 1

                if status == "processing":
                    click.echo(f"🔄 Processing {file_name} ({current_index}/{total_files})...")
                elif status == "analyzing":
                    click.echo(f"🧠 Analyzing {file_name} with AI ({current_index}/{total_files})...")
                elif status == "processing_components":
                    click.echo(f"🔍 Processing components in {file_name} ({current_index}/{total_files})...")
                elif status == "completed":
                    click.echo(f"✅ Completed analysis of {file_name} ({current_index}/{total_files})...")
                elif status == "error":
                    click.echo(f"❌ Error analyzing {file_name} ({current_index}/{total_files})...")

            # Analyze the changes with progress updates
            click.echo("🧠 Starting code analysis...")
            analysis = agent.analyze_changes(files_with_content, progress_callback)

            # Create analysis result
            click.echo("📊 Creating analysis result...")
            analysis_result = AnalysisResult(
                summary=analysis.summary,
                mermaid_diagram=analysis.mermaid_diagram
            )

            # Generate HTML report
            click.echo("🖨️ Generating HTML report...")
            html_path = generate_html_report(analysis_result, output or "diffgraph.html")
            click.echo(f"✅ HTML report generated: {html_path}")

            # Open the HTML report in the default browser
            if not no_open:
                click.echo("🌐 Opening report in browser...")
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', html_path])
                elif sys.platform == 'win32':  # Windows
                    os.startfile(html_path)
                else:  # Linux
                    subprocess.run(['xdg-open', html_path])

        except ValueError as e:
            click.echo(f"❌ Error: {e}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"❌ Error during analysis: {e}", err=True)
            sys.exit(1)
    else:
        # Pass through to git for all other commands
        try:
            result = subprocess.run(["git"] + list(args))
            sys.exit(result.returncode)
        except Exception as e:
            click.secho(f"❌ Error running git command: {e}", fg="red", err=True)
            sys.exit(1)

if __name__ == "__main__":
    main()
