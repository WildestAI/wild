"""Non-diff commands go to git verbatim, even with options wild also defines."""

import subprocess

import pytest
from click.testing import CliRunner

import diffgraph.cli as cli


@pytest.fixture
def git_calls(monkeypatch):
    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize(
    "argv",
    [
        ["log", "-1", "--format=%h %s"],
        ["log", "--format", "oneline"],
        ["show", "--no-patch", "--format=%an", "HEAD"],
        ["format-patch", "-1", "-o", "out-dir"],
        ["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
        ["status", "--short"],
        ["log", "--", "--output"],
    ],
)
def test_non_diff_commands_reach_git_verbatim(argv, git_calls):
    result = CliRunner().invoke(cli.main, argv)

    assert result.exit_code == 0, result.output
    assert git_calls == [["git", *argv]]


def test_wild_options_before_the_command_are_still_wild_options(git_calls):
    # A wild-only option before a non-diff command is a usage error, as before.
    result = CliRunner().invoke(cli.main, ["--format", "json", "status"])

    assert result.exit_code == 2
    assert git_calls == []


def test_version_and_help_are_unaffected():
    assert CliRunner().invoke(cli.main, ["--version"]).exit_code == 0
    assert "Usage" in CliRunner().invoke(cli.main, ["--help"]).output


@pytest.mark.parametrize(
    "argv",
    [
        ["-C", "/repo", "log", "--format=%h"],
        ["-c", "core.pager=cat", "log", "-o", "x"],
        ["--no-pager", "show", "--format=%s"],
        ["-C", "/repo", "diff", "--stat"],
    ],
)
def test_git_global_options_start_a_verbatim_passthrough(argv, git_calls):
    result = CliRunner().invoke(cli.main, argv)

    assert result.exit_code == 0, result.output
    assert git_calls == [["git", *argv]]
