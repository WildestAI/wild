#!/usr/bin/env python3
"""Build a native, single-file ``wild`` executable with PyInstaller.

The only data bundled is ``PACKAGED_DATA_FILES``: an explicit allowlist of
exact package files the CLI loads at runtime (the DiffGraph schema every
canonical artifact is validated against). Nothing else is ever passed to
PyInstaller as data, and a local ``.env`` is rejected before the build starts
so credentials cannot become part of a release executable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
ENTRYPOINT = ROOT / "diffgraph" / "cli.py"
FORBIDDEN_BUNDLE_FILE_NAMES = frozenset({".env"})
# Runtime data loaded through importlib.resources. A missing entry here breaks
# every `wild diff` in the native binary (cli-v1.3.0 shipped without it).
PACKAGED_DATA_FILES = (
    ROOT / "diffgraph" / "schema" / "diffgraph-v2.schema.json",
)


def find_forbidden_files(root: Path) -> list[Path]:
    """Return credential-like dotfiles that must not be present for a build."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name in FORBIDDEN_BUNDLE_FILE_NAMES
        and ".git" not in path.parts
        and ".venv" not in path.parts
    )


def assert_release_workspace_safe(root: Path) -> None:
    """Fail before PyInstaller runs if a local credential file is present."""
    forbidden_files = find_forbidden_files(root)
    if forbidden_files:
        rendered_paths = ", ".join(str(path.relative_to(root)) for path in forbidden_files)
        raise SystemExit(
            "refusing to build from a workspace containing forbidden file(s): "
            f"{rendered_paths}. Release builds never package .env files; "
            "use process environment variables instead."
        )


def packaged_data_arguments() -> list[str]:
    """``--add-data`` arguments for exactly the allowlisted package files."""
    arguments: list[str] = []
    package_root = (ROOT / "diffgraph").resolve()
    for listed in PACKAGED_DATA_FILES:
        # Resolve symlinks first: a link inside the package must not pull an
        # outside file into the release binary.
        path = listed.resolve()
        if not path.is_file() or package_root not in path.parents:
            raise SystemExit(f"packaged data file is missing or outside the package: {listed}")
        destination = path.parent.relative_to(ROOT.resolve()).as_posix()
        arguments += ["--add-data", f"{path}{os.pathsep}{destination}"]
    return arguments


def build_command(*, output_dir: Path, work_dir: Path, spec_dir: Path, name: str) -> list[str]:
    """Return the native PyInstaller invocation; data is only the allowlist."""
    return [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--console",
        "--name",
        name,
        "--distpath",
        str(output_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        *packaged_data_arguments(),
        str(ENTRYPOINT),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="wild", help="name of the generated executable")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "dist",
        help="directory that receives the executable (default: %(default)s)",
    )
    return parser.parse_args(argv)


def assert_output_dir_safe(output_dir: Path) -> None:
    """Reject an output directory that would remove the checkout during cleanup."""
    if ROOT.is_relative_to(output_dir):
        raise SystemExit(
            "refusing to use an output directory that is the repository root or an "
            "ancestor of it"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    work_dir = ROOT / "build" / "pyinstaller"
    spec_dir = ROOT / "build" / "spec"

    assert_output_dir_safe(output_dir)
    assert_release_workspace_safe(ROOT)
    if not ENTRYPOINT.is_file():
        raise SystemExit(f"build entrypoint is missing: {ENTRYPOINT}")

    for directory in (output_dir, work_dir, spec_dir):
        shutil.rmtree(directory, ignore_errors=True)

    try:
        import PyInstaller  # noqa: F401
    except ImportError as error:
        raise SystemExit("PyInstaller is not installed; install requirements.txt first") from error

    command = build_command(
        output_dir=output_dir,
        work_dir=work_dir,
        spec_dir=spec_dir,
        name=args.name,
    )
    subprocess.run(command, cwd=ROOT, check=True)

    executable = output_dir / (f"{args.name}.exe" if sys.platform == "win32" else args.name)
    if not executable.is_file():
        raise SystemExit(f"build completed without expected executable: {executable}")
    print(executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
