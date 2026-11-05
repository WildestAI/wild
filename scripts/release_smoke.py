#!/usr/bin/env python3
"""End-to-end scenario suite for a ``wild`` executable (native or installed).

Usage: python scripts/release_smoke.py <path-to-wild>

Runs every prominent use case against a throwaway repository: the exact
commands the VS Code extension issues (health probe, working tree, staged, and
history views), the core CLI forms, and the error/exit-code contract. Every
JSON artifact is validated against the packaged DiffGraph schema when this
checkout's dependencies are importable (they are in CI), and structurally
otherwise. All scenarios run; failures are listed together.

``--help`` alone cannot catch a binary that is missing runtime data: the
cli-v1.3.0 binaries could not load their DiffGraph schema.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
try:  # Full schema validation with the checkout's own contract.
    sys.path.insert(0, str(REPO_ROOT))
    from diffgraph.contract import validate_artifact as _schema_validate
except Exception:  # pragma: no cover - stdlib-only environments
    _schema_validate = None


class ScenarioFailure(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioFailure(message)


class Workspace:
    """A repository with history, a branch, and staged/unstaged/untracked work."""

    def __init__(self, root: Path, executable: str) -> None:
        self.root = root
        self.repo = root / "repo"
        self.executable = executable
        home = root / "home"
        home.mkdir()
        self.env = {
            **os.environ,
            # No user config, relay token, or .env can influence the run.
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Release Smoke",
            "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
            "GIT_COMMITTER_NAME": "Release Smoke",
            "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
        }
        for key in ("BYOK_RELAY_TOKEN", "BYOK_RELAY_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
            self.env.pop(key, None)
        self._build()

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.repo, env=self.env, check=True, capture_output=True, text=True
        ).stdout.strip()

    def write(self, path: str, text: str) -> None:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def _build(self) -> None:
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("checkout", "-q", "-b", "main")
        self.write("pkg/__init__.py", "")
        self.write("pkg/lib.py", "class Base:\n    pass\n\ndef f(a):\n    return a\n")
        self.write("pkg/use.py", "from pkg.lib import Base, f\n\nclass Child(Base):\n    def run(self):\n        return f(1)\n")
        self.write("README.md", "# Smoke\n")
        (self.repo / "data.bin").write_bytes(b"\x00\x01binary\x00")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "root")
        self.root_commit = self.git("rev-parse", "HEAD")
        # A feature branch from the root, for three-dot comparisons.
        self.git("checkout", "-q", "-b", "feature")
        self.write("pkg/feature.py", "from pkg.lib import f\n\ndef feature():\n    return f(3)\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "feature")
        self.git("checkout", "-q", "main")
        # Change a signature and a base class; pkg/use.py (untouched) calls/subclasses them.
        self.write("pkg/lib.py", "class Base:\n    x = 1\n\ndef f(a, b=0):\n    return a + b\n")
        self.git("commit", "-q", "-am", "change signature")
        self.change_commit = self.git("rev-parse", "HEAD")
        # Working tree: staged README, unstaged use.py and data.bin, untracked new.py.
        self.write("README.md", "# Smoke\n\nStaged.\n")
        self.git("add", "README.md")
        self.write("pkg/use.py", "from pkg.lib import Base, f\n\nclass Child(Base):\n    def run(self):\n        return f(2)\n")
        (self.repo / "data.bin").write_bytes(b"\x00\x02binary\x00")
        self.write("new.py", "def fresh():\n    return 0\n")

    def wild(self, *args: str, expect: int = 0) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [self.executable, *args], cwd=self.repo, env=self.env, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != expect:
            tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-5:])
            raise ScenarioFailure(f"`wild {' '.join(args)}` exited {result.returncode}, expected {expect}:\n{tail}")
        return result

    def out(self, name: str) -> Path:
        return self.root / name

    def json_file(self, *args: str, name: str) -> dict:
        path = self.out(name)
        self.wild(*args, "--output", str(path))
        return validated(json.loads(path.read_text(encoding="utf-8")), " ".join(args))

    def html_file(self, *args: str, name: str) -> str:
        path = self.out(name)
        self.wild(*args, "--output", str(path))
        report = path.read_text(encoding="utf-8")
        check(report.lower().startswith("<!doctype html>"), f"{name}: not an HTML document")
        check('id="diffgraph-artifact"' in report and "Content-Security-Policy" in report,
              f"{name}: missing the embedded artifact or CSP")
        return report


def validated(artifact: dict, label: str) -> dict:
    check(re.fullmatch(r"2\.\d+", str(artifact.get("schema_version"))) is not None,
          f"{label}: schema_version {artifact.get('schema_version')!r}")
    for key in ("files", "symbols", "relationships", "metadata", "diff_ref"):
        check(key in artifact, f"{label}: missing {key}")
    if _schema_validate is not None:
        try:
            _schema_validate(artifact)
        except Exception as error:  # noqa: BLE001 - report any contract failure
            raise ScenarioFailure(f"{label}: schema validation failed: {error}") from error
    return artifact


def paths(artifact: dict, *, changed_only: bool = True) -> set:
    return {item["path"] for item in artifact["files"]
            if not changed_only or item["change_kind"] != "unchanged"}


def warning_codes(artifact: dict) -> set:
    return {item["code"] for item in artifact["metadata"].get("warnings", [])}


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

def vscode_version(ws: Workspace) -> str:
    stdout = ws.wild("--version").stdout.strip()
    check(re.fullmatch(r"wild, version \d+\.\d+\.\d+", stdout) is not None, f"unexpected version line {stdout!r}")
    return stdout


def vscode_health_probe(ws: Workspace) -> str:
    """CliService.probeRuntime: `wild diff --format json` on stdout."""
    artifact = validated(json.loads(ws.wild("diff", "--format", "json").stdout), "health probe")
    check(artifact["diff_ref"]["kind"] == "unstaged", "probe artifact is not an unstaged diff")
    check({"pkg/use.py", "new.py"} <= paths(artifact), f"probe files {sorted(paths(artifact))}")
    return f"{len(artifact['files'])} files on stdout"


def vscode_working_tree_html(ws: Workspace) -> str:
    path = ws.out("working.html")
    ws.wild("diff", "--format", "html", "--output", str(path), "--no-open")
    report = path.read_text(encoding="utf-8")
    check(report.lower().startswith("<!doctype html>") and 'id="diffgraph-artifact"' in report, "not a DiffGraph report")
    return f"{len(report)} bytes"


def vscode_staged_html(ws: Workspace) -> str:
    path = ws.out("staged.html")
    ws.wild("diff", "--format", "html", "--output", str(path), "--no-open", "--staged")
    report = path.read_text(encoding="utf-8")
    check('id="diffgraph-artifact"' in report and "Staged changes" in report, "staged report missing its snapshot label")
    return "ok"


def vscode_working_tree_json(ws: Workspace) -> str:
    artifact = ws.json_file("diff", "--format", "json", name="working.json")
    check({"pkg/use.py", "new.py", "data.bin"} <= paths(artifact), f"files {sorted(paths(artifact))}")
    check("README.md" not in paths(artifact), "a staged-only file appeared in the unstaged view")
    return f"{len(artifact['files'])} files"


def vscode_staged_json(ws: Workspace) -> str:
    path = ws.out("staged.json")
    ws.wild("diff", "--format", "json", "--output", str(path), "--staged")
    artifact = validated(json.loads(path.read_text(encoding="utf-8")), "staged json")
    check(artifact["diff_ref"]["kind"] == "staged", "not a staged diff")
    check(paths(artifact) == {"README.md"}, f"staged files {sorted(paths(artifact))}")
    check("UNSUPPORTED_LANGUAGE" in warning_codes(artifact), "no UNSUPPORTED_LANGUAGE warning for Markdown")
    return "README.md only"


def vscode_history_commit(ws: Workspace) -> str:
    """HistoryViewProvider: `<sha>~1..<sha>` as HTML and JSON."""
    commit = ws.change_commit
    artifact = ws.json_file("diff", f"{commit}~1..{commit}", "--format", "json", name="history.json")
    check(artifact["diff_ref"]["kind"] == "commit_range", "not a commit range")
    check(paths(artifact) == {"pkg/lib.py"}, f"commit files {sorted(paths(artifact))}")
    context = paths(artifact, changed_only=False) - paths(artifact)
    check("pkg/use.py" in context, "the untouched caller pkg/use.py is not reported outside the diff")
    kinds = {item["kind"] for item in artifact["relationships"]}
    check({"calls", "inherits"} <= kinds, f"expected calls and inherits edges, got {sorted(kinds)}")
    symbols = {item["name"]: item for item in artifact["symbols"]}
    check(symbols["f"].get("source", {}).get("after", {}).get("text", "").startswith("def f(a, b=0)"),
          "the changed symbol carries no exact after-source")
    path = ws.out("history.html")
    ws.wild("diff", f"{commit}~1..{commit}", "--format", "html", "--output", str(path), "--no-open")
    check('id="diffgraph-artifact"' in path.read_text(encoding="utf-8"), "history HTML is not a DiffGraph report")
    return "callers and subclasses outside the diff found"


def vscode_history_root_commit(ws: Workspace) -> str:
    """The first commit has no parent: `<root>~1..<root>` must not crash."""
    commit = ws.root_commit
    artifact = ws.json_file("diff", f"{commit}~1..{commit}", "--format", "json", name="root.json")
    check(artifact["files"] == [], "a parentless range invented changes")
    check("invalid_base_ref" in warning_codes(artifact), f"warnings {sorted(warning_codes(artifact))}")
    return "exit 0 with an invalid_base_ref warning"


def cli_terminal_default(ws: Workspace) -> str:
    stdout = ws.wild("diff").stdout
    check("wild diff" in stdout and "pkg/use.py" in stdout, "terminal review path is missing the change")
    ws.wild("diff", "--compact")
    ws.wild("diff", "--all")
    return "default, --compact, --all"


def cli_ref_against_working_tree(ws: Workspace) -> str:
    artifact = ws.json_file("diff", "HEAD~1", "--format", "json", name="ref.json")
    check(artifact["diff_ref"]["kind"] == "working_tree", f"kind {artifact['diff_ref']['kind']}")
    check(artifact["diff_ref"]["requested_base_ref"] == "HEAD~1", "requested ref not recorded")
    check({"pkg/lib.py", "pkg/use.py", "README.md", "new.py"} <= paths(artifact), f"files {sorted(paths(artifact))}")
    return "wild diff HEAD~1"


def cli_staged_against_ref(ws: Workspace) -> str:
    artifact = ws.json_file("diff", "--staged", "HEAD~1", "--format", "json", name="staged-ref.json")
    check(artifact["diff_ref"]["kind"] == "staged" and artifact["diff_ref"]["requested_base_ref"] == "HEAD~1",
          "not a staged diff against HEAD~1")
    check(paths(artifact) == {"pkg/lib.py", "README.md"}, f"files {sorted(paths(artifact))}")
    return "wild diff --staged HEAD~1"


def cli_three_dot_branch(ws: Workspace) -> str:
    artifact = ws.json_file("diff", "main...feature", "--format", "json", name="three-dot.json")
    check(artifact["diff_ref"]["comparison_mode"] == "three_dot", "not a three-dot comparison")
    check(paths(artifact) == {"pkg/feature.py"}, f"files {sorted(paths(artifact))}")
    return "main...feature"


def cli_pathspec(ws: Workspace) -> str:
    # Options go before `--`; everything after it is a pathspec.
    path = ws.out("pathspec.json")
    ws.wild("diff", "--format", "json", "--output", str(path), "--", "pkg/use.py")
    artifact = validated(json.loads(path.read_text(encoding="utf-8")), "pathspec")
    check(paths(artifact) == {"pkg/use.py"}, f"files {sorted(paths(artifact))}")
    return "-- pkg/use.py"


def cli_binary_file(ws: Workspace) -> str:
    artifact = json.loads(ws.out("working.json").read_text(encoding="utf-8")) if ws.out("working.json").exists() \
        else ws.json_file("diff", "--format", "json", name="working.json")
    check("PARTIAL_ANALYSIS" in warning_codes(artifact), "no PARTIAL_ANALYSIS warning for the binary file")
    return "data.bin reported, not parsed"


def cli_no_changes(ws: Workspace) -> str:
    artifact = json.loads(ws.wild("diff", "HEAD..HEAD", "--format", "json").stdout)
    validated(artifact, "no changes")
    check(artifact["files"] == [] and artifact["symbols"] == [], "an empty range reported changes")
    return "exit 0, empty artifact"


def errors_usage(ws: Workspace) -> str:
    unknown = ws.wild("diff", "nosuchref", expect=2)
    check("unknown revision 'nosuchref'" in unknown.stderr, "unknown revision message missing")
    ws.wild("diff", "--format", "terminal", "--output", str(ws.out("x.txt")), expect=2)
    ws.wild("diff", "--llm", "claude:turbo", expect=2)
    return "unknown revision, terminal+--output, bad --llm tier -> exit 2"


def errors_runtime(ws: Workspace) -> str:
    target = ws.out("missing-dir") / "out.json"
    ws.wild("diff", "--format", "json", "--output", str(target), expect=1)
    check(not target.exists(), "a failed write left a file behind")
    return "unwritable output -> exit 1, nothing written"


def git_passthrough(ws: Workspace) -> str:
    status = ws.wild("status", "--short").stdout
    check("new.py" in status, "`wild status` is not passed through to git")
    # git's own options that share wild's names must reach git untouched.
    log = ws.wild("log", "-1", "--format=%s").stdout.strip()
    check(log == "change signature", f"`wild log --format=%s` returned {log!r}")
    shown = ws.wild("show", "--no-patch", "--format=%an", "HEAD").stdout.strip()
    check(shown == "Release Smoke", f"`wild show --format=%an` returned {shown!r}")
    patch_dir = ws.out("patches")
    ws.wild("format-patch", "-1", "-o", str(patch_dir), "--quiet")
    check(any(patch_dir.glob("*.patch")), "`wild format-patch -o DIR` wrote no patch")
    return "status, log --format, format-patch -o"


SCENARIOS: List[tuple] = [
    ("vscode: --version", vscode_version),
    ("vscode: health probe (JSON on stdout)", vscode_health_probe),
    ("vscode: working tree HTML", vscode_working_tree_html),
    ("vscode: working tree JSON", vscode_working_tree_json),
    ("vscode: staged HTML", vscode_staged_html),
    ("vscode: staged JSON", vscode_staged_json),
    ("vscode: history commit (<sha>~1..<sha>)", vscode_history_commit),
    ("vscode: history root commit", vscode_history_root_commit),
    ("cli: terminal review path", cli_terminal_default),
    ("cli: REF -> working tree", cli_ref_against_working_tree),
    ("cli: --staged REF", cli_staged_against_ref),
    ("cli: three-dot branch comparison", cli_three_dot_branch),
    ("cli: pathspec", cli_pathspec),
    ("cli: binary file", cli_binary_file),
    ("cli: no changes", cli_no_changes),
    ("errors: usage (exit 2)", errors_usage),
    ("errors: runtime (exit 1)", errors_runtime),
    ("git passthrough", git_passthrough),
]


def main(binary: str, only: Optional[Callable] = None) -> int:
    executable = str(Path(binary).resolve()) if Path(binary).exists() else binary
    failures = []
    print(f"Scenario suite for {executable} "
          f"({'schema-validated' if _schema_validate else 'structural checks only'})")
    with tempfile.TemporaryDirectory() as root:
        workspace = Workspace(Path(root), executable)
        for name, scenario in SCENARIOS:
            try:
                detail = scenario(workspace)
                print(f"  ✔ {name}: {detail}")
            except Exception as error:  # noqa: BLE001 - keep running, report all
                failures.append(name)
                print(f"  ✘ {name}: {error}")
    if failures:
        print(f"{len(failures)} of {len(SCENARIOS)} scenarios failed: {', '.join(failures)}")
        return 1
    print(f"All {len(SCENARIOS)} scenarios passed.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: release_smoke.py <path-to-wild>")
    sys.exit(main(sys.argv[1]))
