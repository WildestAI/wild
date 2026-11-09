"""Contract tests for one-build validated canonical CLI dispatch."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import diffgraph.artifact as artifact_service
import diffgraph.cli as cli
from diffgraph.contract import ValidatedArtifact
from diffgraph.formatters.terminal import TerminalFormatter


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )


def changed_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.name", "Artifact Tests")
    git(root, "config", "user.email", "artifact@example.test")
    (root / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    git(root, "add", "app.py")
    git(root, "commit", "-m", "base")
    (root / "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    return root


def test_builder_constructs_and_validates_once_then_consumers_share_wrapper(monkeypatch):
    value = {"identity": "one object"}
    calls = []

    def analyze(*args, **kwargs):
        calls.append(("build", args, kwargs))
        return value

    def validate(candidate):
        calls.append(("validate", candidate))
        assert candidate == value
        assert candidate is not value

    monkeypatch.setattr(artifact_service, "analyze_local_diff", analyze)
    monkeypatch.setattr("diffgraph.contract.validate_artifact", validate)

    artifact = artifact_service.build_validated_artifact(
        ".", staged=True, pathspecs=("src",), wild_version="test"
    )

    assert artifact.value == value
    assert artifact.value is not value
    assert [call[0] for call in calls] == ["build", "validate"]
    assert TerminalFormatter.__new__(TerminalFormatter) is not None  # class is importable locally
    assert artifact_service.render_canonical_json(artifact).endswith("\n")


def test_json_terminal_and_html_receive_the_exact_once_built_artifact(
    golden_artifact, monkeypatch, tmp_path
):
    artifact = ValidatedArtifact.from_value(golden_artifact)
    build_calls = []
    received = {}

    def build(*args, **kwargs):
        build_calls.append((args, kwargs))
        return artifact

    class RecordingTerminal:
        DEFAULT_MAX_ITEMS = 10

        def __init__(self, candidate, **kwargs):
            received["terminal"] = candidate

        def render(self):
            pass

    class RecordingHtml:
        def __init__(self, candidate):
            received["html"] = candidate

        def write(self, destination):
            return Path(destination).absolute()

    def render_json(candidate):
        received["json"] = candidate
        return "{}\n"

    monkeypatch.setattr(cli, "is_git_repo", lambda: True)
    monkeypatch.setattr(cli, "build_validated_artifact", build)
    monkeypatch.setattr(cli, "render_canonical_json", render_json)
    monkeypatch.setattr("diffgraph.formatters.terminal.TerminalFormatter", RecordingTerminal)
    monkeypatch.setattr("diffgraph.formatters.html.HtmlFormatter", RecordingHtml)

    invocations = [
        ["diff", "--format", "json"],
        ["diff", "--format", "terminal"],
        [
            "diff",
            "--format",
            "html",
            "--output",
            str(tmp_path / "report.html"),
            "--no-open",
        ],
    ]
    for arguments in invocations:
        result = CliRunner().invoke(cli.main, arguments)
        assert result.exit_code == 0, result.output

    assert len(build_calls) == 3
    assert received == {"json": artifact, "terminal": artifact, "html": artifact}
    assert all(candidate is artifact for candidate in received.values())


def test_output_path_without_explicit_format_selects_html(golden_artifact, monkeypatch, tmp_path):
    artifact = ValidatedArtifact.from_value(golden_artifact)
    received = {}

    monkeypatch.setattr(cli, "is_git_repo", lambda: True)
    monkeypatch.setattr(cli, "build_validated_artifact", lambda *args, **kwargs: artifact)

    class RecordingHtml:
        def __init__(self, candidate):
            received["artifact"] = candidate

        def write(self, destination):
            received["destination"] = Path(destination)
            return Path(destination).absolute()

    monkeypatch.setattr("diffgraph.formatters.html.HtmlFormatter", RecordingHtml)
    destination = tmp_path / "report.html"
    result = CliRunner().invoke(cli.main, ["diff", "-o", str(destination), "--no-open"])

    assert result.exit_code == 0, result.output
    assert received == {"artifact": artifact, "destination": destination}


def test_terminal_consumer_does_not_revalidate_branded_artifact(golden_artifact, monkeypatch):
    artifact = ValidatedArtifact.from_value(golden_artifact)

    def unexpected_validation(*args, **kwargs):
        raise AssertionError("branded CLI artifact must not be validated twice")

    monkeypatch.setattr("diffgraph.contract.validate_artifact", unexpected_validation)
    formatter = TerminalFormatter(artifact, color=False)

    assert formatter.artifact is artifact
    assert formatter.dg == artifact.value


@pytest.fixture
def golden_artifact():
    return json.loads(
        (Path(__file__).parents[1] / "diffgraph/schema/diffgraph-v2.structural.example.json")
        .read_text(encoding="utf-8")
    )


def test_format_json_stdout_is_artifact_only(tmp_path, monkeypatch):
    root = changed_repo(tmp_path)
    monkeypatch.chdir(root)

    result = CliRunner().invoke(cli.main, ["diff", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    parsed = json.loads(result.stdout)
    assert parsed["files"][0]["path"] == "app.py"


def test_format_json_explicit_output_is_atomic_and_reports_on_stderr(tmp_path, monkeypatch):
    root = changed_repo(tmp_path)
    destination = root / "artifact.json"
    monkeypatch.chdir(root)

    result = CliRunner().invoke(
        cli.main, ["diff", "--format", "json", "--output", str(destination)]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "Canonical DiffGraph written" in result.stderr
    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"] == "2.1"
    assert not list(root.glob(".artifact.json.*.tmp"))


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--structural-json", "legacy.json", "--format", "json", "diff"], "cannot be combined"),
        (["--structural-json", "legacy.json", "--output", "other.json", "diff"], "cannot be combined"),
        (["diff", "--format", "terminal", "--output", "terminal.txt"], "writes to stdout"),
        # wild options apply before the command; after it they belong to git.
        (["--format", "json", "status"], "can only be used with 'diff'"),
    ],
)
def test_canonical_option_conflicts_are_usage_errors(arguments, message):
    result = CliRunner().invoke(cli.main, arguments)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert message in result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ["diff", "--format", "json", "--debug-env"],
        ["diff", "--format", "terminal", "--debug-env"],
        ["diff", "--format", "html", "--debug-env"],
        ["diff", "--structural-json", "-", "--debug-env"],
    ],
)
def test_debug_env_cannot_preempt_canonical_artifact_output(arguments):
    result = CliRunner().invoke(cli.main, arguments)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "--debug-env cannot be combined with canonical artifact output" in result.stderr


def test_no_change_json_and_terminal_succeed(tmp_path, monkeypatch):
    root = changed_repo(tmp_path)
    git(root, "checkout", "--", "app.py")
    monkeypatch.chdir(root)

    json_result = CliRunner().invoke(cli.main, ["diff", "--format", "json"])
    terminal_result = CliRunner().invoke(cli.main, ["diff", "--format", "terminal"])

    assert json_result.exit_code == 0
    assert json.loads(json_result.stdout)["files"] == []
    assert terminal_result.exit_code == 0
    assert "No changes in the selected snapshot." in terminal_result.stdout
    assert terminal_result.stderr == ""


@pytest.mark.parametrize("output_format", ["json", "terminal", "html"])
def test_canonical_paths_are_offline_and_do_not_import_ai_modules(
    output_format, tmp_path, monkeypatch
):
    root = changed_repo(tmp_path)
    monkeypatch.chdir(root)
    # delitem restores the modules afterwards, so later tests keep patching
    # the same module objects the CLI imports.
    for module in ("diffgraph.ai_analysis", "agents", "diffgraph.relay", "diffgraph.llm_summaries"):
        monkeypatch.delitem(sys.modules, module, raising=False)

    import socket

    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network access")),
    )
    result = CliRunner().invoke(
        cli.main, ["diff", "--format", output_format, "--no-open"]
    )

    assert result.exit_code == 0, result.output
    assert "diffgraph.ai_analysis" not in sys.modules
    assert "agents" not in sys.modules
    assert "diffgraph.relay" not in sys.modules
    assert "diffgraph.llm_summaries" not in sys.modules


def test_ctrl_c_is_exit_one_on_stderr_and_does_not_create_output(tmp_path, monkeypatch):
    root = changed_repo(tmp_path)
    destination = root / "cancelled.json"
    monkeypatch.chdir(root)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_validated_artifact", interrupt)
    result = CliRunner().invoke(
        cli.main, ["diff", "--format", "json", "--output", str(destination)]
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Aborted!" in result.stderr
    assert not destination.exists()
