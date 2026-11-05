from pathlib import Path
import subprocess
import sys


import pytest

import build
from diffgraph import __version__


def test_release_build_bundles_only_the_allowlisted_schema(tmp_path):
    command = build.build_command(
        output_dir=tmp_path / "release",
        work_dir=tmp_path / "work",
        spec_dir=tmp_path / "spec",
        name="wild",
    )

    assert "--onefile" in command
    assert "--add-binary" not in command
    assert all(".env" not in argument for argument in command)
    data = [command[index + 1] for index, argument in enumerate(command) if argument == "--add-data"]
    # Exactly the runtime schema, as one file into its package directory.
    assert data == [
        f"{build.ROOT / 'diffgraph' / 'schema' / 'diffgraph-v2.schema.json'}{build.os.pathsep}diffgraph/schema"
    ]


def test_release_build_bundles_every_runtime_resource():
    """Every importlib.resources path the CLI reads must be bundled, or the
    native binary fails at runtime (cli-v1.3.0 could not load its schema)."""
    from diffgraph import contract

    bundled = {path.relative_to(build.ROOT / "diffgraph").as_posix() for path in build.PACKAGED_DATA_FILES}
    assert contract._SCHEMA_RESOURCE in bundled


def test_release_build_refuses_data_outside_the_package(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "PACKAGED_DATA_FILES", (build.ROOT / "README.md",))
    with pytest.raises(SystemExit, match="missing or outside the package"):
        build.packaged_data_arguments()


def test_release_build_rejects_local_dotenv(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=must-not-bundle\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="never package .env files"):
        build.assert_release_workspace_safe(tmp_path)


def test_release_build_allows_env_example(tmp_path):
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=\n", encoding="utf-8")

    assert build.find_forbidden_files(tmp_path) == []


def test_release_build_rejects_nested_dotenv(tmp_path):
    nested_env = tmp_path / "package" / ".env"
    nested_env.parent.mkdir()
    nested_env.write_text("OPENAI_API_KEY=must-not-bundle\n", encoding="utf-8")

    assert build.find_forbidden_files(tmp_path) == [nested_env]

@pytest.mark.parametrize("output_dir", [build.ROOT, build.ROOT.parent])
def test_release_build_rejects_checkout_or_ancestor_output_dir_before_cleanup(
    monkeypatch, output_dir
):
    monkeypatch.setattr(
        build.shutil,
        "rmtree",
        lambda *args, **kwargs: pytest.fail("cleanup called for unsafe output directory"),
    )

    with pytest.raises(SystemExit, match="repository root or an ancestor"):
        build.main(["--output-dir", str(output_dir)])


def test_native_binary_reports_version(tmp_path: Path) -> None:
    pytest.importorskip("PyInstaller")
    output_dir = tmp_path / "release"
    subprocess.run(
        [sys.executable, "build.py", "--output-dir", str(output_dir), "--name", "wild"],
        check=True,
        cwd=build.ROOT,
    )
    binary = output_dir / ("wild.exe" if sys.platform == "win32" else "wild")
    result = subprocess.run([str(binary), "--version"], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == f"wild, version {__version__}"
    # A real diff exercises the bundled runtime data (the schema).
    smoke = subprocess.run(
        [sys.executable, "scripts/release_smoke.py", str(binary)],
        cwd=build.ROOT, capture_output=True, text=True,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
    assert "json: schema 2." in smoke.stdout



def test_release_smoke_script_checks_a_real_diff():
    """The release smoke test passes on a working CLI (the dev install)."""
    import shutil

    wild = shutil.which("wild")
    if wild is None:
        pytest.skip("the wild console script is not installed")
    smoke = subprocess.run(
        [sys.executable, "scripts/release_smoke.py", wild],
        cwd=build.ROOT, capture_output=True, text=True,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
    assert "json: schema 2." in smoke.stdout and "html:" in smoke.stdout
