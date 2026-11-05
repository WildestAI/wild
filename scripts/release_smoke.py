#!/usr/bin/env python3
"""Smoke-test a native ``wild`` executable on a real diff (stdlib only).

Usage: python scripts/release_smoke.py <path-to-wild>

``--help`` alone cannot catch a binary that is missing runtime data (the
cli-v1.3.0 binaries could not load their DiffGraph schema), so this runs the
canonical JSON and HTML outputs in a throwaway repository.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main(binary: str) -> int:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Release Smoke",
        "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
        "GIT_COMMITTER_NAME": "Release Smoke",
        "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
    }
    executable = str(Path(binary).resolve())
    with tempfile.TemporaryDirectory() as workspace:
        repo = Path(workspace) / "repo"
        repo.mkdir()

        def git(*args: str) -> None:
            subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)

        def wild(*args: str) -> str:
            result = subprocess.run(
                [executable, *args], cwd=repo, env=env, capture_output=True, text=True
            )
            if result.returncode != 0:
                tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-5:])
                raise SystemExit(f"`wild {' '.join(args)}` exited with {result.returncode}:\n{tail}")
            return result.stdout

        git("init", "-q")
        (repo / "lib.py").write_text("def f(a):\n    return a\n", encoding="utf-8")
        (repo / "use.py").write_text("from lib import f\n\ndef g():\n    return f(1)\n", encoding="utf-8")
        git("add", ".")
        git("commit", "-q", "-m", "base")
        (repo / "lib.py").write_text("def f(a, b=0):\n    return a + b\n", encoding="utf-8")

        version = wild("--version").strip()
        print(version)

        json_path = Path(workspace) / "diffgraph.json"
        wild("diff", "--format", "json", "--output", str(json_path))
        artifact = json.loads(json_path.read_text(encoding="utf-8"))
        if not re.fullmatch(r"2\.\d+", str(artifact.get("schema_version"))):
            raise SystemExit(f"unexpected schema_version {artifact.get('schema_version')!r}")
        if not artifact.get("files") or not artifact.get("symbols"):
            raise SystemExit("the JSON artifact has no files or symbols for a real change")
        print(f"json: schema {artifact['schema_version']}, {len(artifact['files'])} files, "
              f"{len(artifact['symbols'])} symbols")

        html_path = Path(workspace) / "diffgraph.html"
        wild("diff", "--format", "html", "--output", str(html_path), "--no-open")
        report = html_path.read_text(encoding="utf-8")
        if not report.lower().startswith("<!doctype html>") or 'id="diffgraph-artifact"' not in report:
            raise SystemExit("the HTML report is not a self-contained DiffGraph document")
        print(f"html: {len(report)} bytes")

        wild("diff")  # terminal output
        print("terminal: ok")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: release_smoke.py <path-to-wild>")
    sys.exit(main(sys.argv[1]))
