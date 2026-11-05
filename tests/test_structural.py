import hashlib
import json
import os
import subprocess
from pathlib import Path

import jsonschema
import pytest

from diffgraph.git_snapshot import (
    CommitRangeResolution,
    GitSnapshotError,
    ResolutionWarning,
    SnapshotResolution,
)
from diffgraph.structural import analyze_local_diff

SCHEMA = json.loads((Path(__file__).parents[1] / "diffgraph/schema/diffgraph-v2.schema.json").read_text())


def git(repo, *args, input_bytes=None):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.decode().strip()


def write(repo, path, text):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.name", "Structural Tests")
    git(root, "config", "user.email", "structural@example.test")
    return root


def commit(root):
    git(root, "add", "-A")
    git(root, "commit", "-m", "snapshot")


def stable(artifact):
    copy = json.loads(json.dumps(artifact))
    copy["generated_at"] = "<run>"
    copy["metadata"]["analysis_duration_ms"] = None
    copy["diff_ref"]["repo_root"] = "<repo>"
    return copy


def assert_valid(artifact):
    jsonschema.validate(artifact, SCHEMA)


def test_staged_add_modify_delete_rename_import_is_schema_valid_and_golden(tmp_path):
    root = repo(tmp_path)
    write(root, "modify.py", "def retained():\n    return 1\n\ndef removed():\n    return 0\n")
    write(root, "delete.py", "class Deleted:\n    pass\n")
    write(root, "rename.py", "def moved():\n    return 1\n")
    commit(root)

    write(root, "add.py", "import os\nfrom external.pkg import value\n\ndef added():\n    def child():\n        return value\n    return child()\n")
    write(root, "modify.py", "def retained():\n    return 2\n\ndef created():\n    return 3\n")
    os.unlink(root / "delete.py")
    git(root, "mv", "rename.py", "renamed.py")
    write(root, "renamed.py", "def moved():\n    return 2\n")
    git(root, "add", "-A")

    artifact = analyze_local_diff(str(root), staged=True)
    assert_valid(artifact)
    assert [f["path"] for f in artifact["files"]] == ["add.py", "delete.py", "modify.py", "renamed.py"]
    assert {f["path"]: f["change_kind"] for f in artifact["files"]} == {
        "add.py": "added", "delete.py": "deleted", "modify.py": "modified", "renamed.py": "renamed_modified"
    }
    changes = {s["id"]: s["change_kind"] for s in artifact["symbols"]}
    assert changes["sym::add.py::added"] == "added"
    assert changes["sym::delete.py::Deleted"] == "deleted"
    assert changes["sym::modify.py::retained"] == "modified"
    assert changes["sym::modify.py::removed"] == "deleted"
    assert changes["sym::modify.py::created"] == "added"
    assert changes["sym::renamed.py::moved"] == "modified"
    assert artifact["metadata"]["files_analyzed"] == 4
    imports = [r for r in artifact["relationships"] if r["kind"] == "imports"]
    assert [r["label"] for r in imports] == [
        "unresolved/external module: external.pkg", "unresolved/external module: os"
    ]
    assert {(r["kind"], r["source_id"], r["target_id"]) for r in artifact["relationships"] if r["kind"] in ("contains", "calls")} == {
        ("contains", "sym::add.py::added", "sym::add.py::added.child"),
        ("calls", "sym::add.py::added", "sym::add.py::added.child"),
    }
    for file_entry in artifact["files"]:
        provenance = json.loads(file_entry["evidence"][0]["detail"])
        assert provenance["old_oid"] is None or len(provenance["old_oid"]) == 40
        assert provenance["new_oid"] is None or len(provenance["new_oid"]) == 40
        assert provenance["old_sha256"] is None or len(provenance["old_sha256"]) == 64
        assert provenance["new_sha256"] is None or len(provenance["new_sha256"]) == 64

    golden_path = Path(__file__).parent / "fixtures/python_topology.json"
    expected = json.loads(golden_path.read_text())
    actual = {
        "files": [{"path": f["path"], "old_path": f["old_path"], "change_kind": f["change_kind"]} for f in artifact["files"]],
        "symbols": [{"id": s["id"], "kind": s["kind"], "change_kind": s["change_kind"]} for s in artifact["symbols"]],
        "relationships": [{"id": r["id"], "kind": r["kind"]} for r in artifact["relationships"]],
    }
    if os.environ.get("UPDATE_GOLDEN"):
        golden_path.write_text(json.dumps(actual, indent=2) + "\n")
        pytest.skip("golden fixture regenerated")
    assert actual == expected


def test_unstaged_uses_index_to_worktree_exact_identity_and_is_stable(tmp_path):
    root = repo(tmp_path)
    write(root, "app.py", "def value():\n    return 1\n")
    commit(root)
    write(root, "app.py", "def value():\n    return 2\n")

    first = analyze_local_diff(str(root))
    second = analyze_local_diff(str(root))
    assert_valid(first)
    assert stable(first) == stable(second)
    provenance = json.loads(first["files"][0]["evidence"][0]["detail"])
    expected = git(root, "hash-object", "--path=app.py", "app.py")
    assert provenance["new_oid"] == expected
    assert first["symbols"][0]["change_kind"] == "modified"


def test_sha256_repository_preserves_structural_blob_provenance(tmp_path):
    """Canonical artifacts must retain full SHA-256 Git blob identities."""
    root = tmp_path / "sha256-repo"
    root.mkdir()
    git(root, "init", "--object-format=sha256")
    git(root, "config", "user.name", "Structural Tests")
    git(root, "config", "user.email", "structural@example.test")
    write(root, "app.py", "def value():\n    return 1\n")
    commit(root)

    old_oid = git(root, "rev-parse", "HEAD:app.py")
    write(root, "app.py", "def value():\n    return 2\n")
    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    provenance = json.loads(artifact["files"][0]["evidence"][0]["detail"])
    assert provenance["old_oid"] == old_oid
    assert provenance["new_oid"] == git(root, "hash-object", "app.py")
    assert len(provenance["old_oid"]) == len(provenance["new_oid"]) == 64
    assert artifact["symbols"][0]["change_kind"] == "modified"


@pytest.mark.parametrize("staged", [False, True])
def test_worktree_diffs_do_not_record_commit_range_mode(tmp_path, staged):
    root = repo(tmp_path)
    write(root, "app.py", "def value():\n    return 1\n")
    commit(root)
    write(root, "app.py", "def value():\n    return 2\n")
    if staged:
        git(root, "add", "app.py")

    artifact = analyze_local_diff(str(root), staged=staged, three_dot=True)

    assert_valid(artifact)
    assert artifact["diff_ref"]["kind"] == ("staged" if staged else "unstaged")
    assert artifact["diff_ref"]["comparison_mode"] is None


def test_untracked_python_is_an_exact_added_snapshot(tmp_path):
    root = repo(tmp_path)
    write(root, "tracked.txt", "baseline\n")
    commit(root)
    write(root, "new.py", "def added():\n    return 1\n")

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    assert [item["path"] for item in artifact["files"]] == ["new.py"]
    file_entry = artifact["files"][0]
    provenance = json.loads(file_entry["evidence"][0]["detail"])
    assert file_entry["change_kind"] == "added"
    assert provenance["old_oid"] is None
    assert provenance["new_oid"] == git(
        root, "hash-object", "--path=new.py", "new.py"
    )
    assert [item["name"] for item in artifact["symbols"]] == ["added"]
    assert artifact["symbols"][0]["change_kind"] == "added"


def test_pathspec_scope_is_not_widened(tmp_path):
    root = repo(tmp_path)
    write(root, "inside/a.py", "def a():\n    return 1\n")
    write(root, "outside/b.py", "def b():\n    return 1\n")
    commit(root)
    write(root, "inside/a.py", "def a():\n    return 2\n")
    write(root, "outside/b.py", "def b():\n    return 2\n")
    artifact = analyze_local_diff(str(root), pathspecs=["inside"])
    assert [item["path"] for item in artifact["files"]] == ["inside/a.py"]


def test_python_stub_files_are_analyzed_with_tree_sitter_provenance(tmp_path):
    """Typed Python stubs are part of the deterministic Python baseline."""
    root = repo(tmp_path)
    write(root, "api.pyi", "def load(value: str) -> int: ...\n")
    commit(root)
    write(root, "api.pyi", "def load(value: str, retries: int = 0) -> int: ...\n")

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    assert artifact["files"][0]["language"] == "python"
    assert artifact["metadata"]["files_analyzed"] == 1
    assert artifact["metadata"]["warnings"] == []
    symbol = artifact["symbols"][0]
    assert (symbol["id"], symbol["name"], symbol["change_kind"]) == (
        "sym::api.pyi::load", "load", "modified"
    )
    assert "parser=tree-sitter-language-pack@" in symbol["evidence"][0]["detail"]


def test_unsupported_language_is_explicit_and_not_overclaimed(tmp_path):
    """Unsupported snapshots keep their explicit capability boundary and blobs."""
    root = repo(tmp_path)
    write(root, "main.js", "export function value() { return 1; }\n")
    commit(root)
    write(root, "main.js", "export function value() { return 2; }\n")
    old_oid = git(root, "rev-parse", "HEAD:main.js")
    new_oid = git(root, "hash-object", "main.js")
    artifact = analyze_local_diff(str(root))
    assert_valid(artifact)
    assert artifact["symbols"] == []
    assert artifact["metadata"]["files_skipped"] == 1
    assert artifact["metadata"]["warnings"][0]["code"] == "UNSUPPORTED_LANGUAGE"
    assert artifact["metadata"]["warnings"][0]["detail"] == (
        "Deterministic extraction currently supports Python (.py and .pyi) only; "
        "pre-change blob {}; post-change blob {}."
    ).format(old_oid, new_oid)


@pytest.mark.parametrize("change_kind", ["added", "deleted"])
def test_unsupported_language_one_sided_blob_is_absent(tmp_path, change_kind):
    """Added and deleted unsupported snapshots identify their missing blob side."""
    root = repo(tmp_path)
    write(root, "main.js", "export function value() { return 1; }\n")
    commit(root)

    if change_kind == "added":
        write(root, "added.js", "export function value() { return 2; }\n")
        old_oid = "absent"
        new_oid = git(root, "hash-object", "added.js")
    else:
        old_oid = git(root, "rev-parse", "HEAD:main.js")
        (root / "main.js").unlink()
        new_oid = "absent"

    artifact = analyze_local_diff(str(root))
    assert_valid(artifact)
    assert artifact["metadata"]["warnings"][0]["detail"] == (
        "Deterministic extraction currently supports Python (.py and .pyi) only; "
        "pre-change blob {}; post-change blob {}."
    ).format(old_oid, new_oid)


@pytest.mark.parametrize("staged", [False, True])
def test_binary_python_snapshot_preserves_identity_without_parsing(tmp_path, staged):
    """Binary snapshots retain exact evidence without source-level claims."""
    root = repo(tmp_path)
    write(root, "payload.py", "def previous():\n    return 1\n")
    commit(root)
    original = b"def previous():\n    return 1\n"
    binary = b"\x89PNG\r\n\x1a\n\x00not-python\xff\n"
    (root / "payload.py").write_bytes(binary)
    if staged:
        git(root, "add", "--", "payload.py")

    artifact = analyze_local_diff(str(root), staged=staged)

    assert_valid(artifact)
    assert artifact["schema_version"] == "2.1"
    assert artifact["symbols"] == []
    assert artifact["relationships"] == []
    assert artifact["metadata"]["files_analyzed"] == 0
    assert artifact["metadata"]["files_skipped"] == 1
    file_entry = artifact["files"][0]
    assert file_entry["language"] is None
    assert file_entry["lines_added"] is None
    assert file_entry["lines_removed"] is None
    assert "diff" not in file_entry  # binary content is never rendered as text
    provenance = json.loads(file_entry["evidence"][0]["detail"])
    assert provenance["old_oid"] == git(root, "rev-parse", "HEAD:payload.py")
    assert provenance["new_oid"] == git(root, "hash-object", "--path=payload.py", "payload.py")
    assert provenance["old_mode"] == "100644"
    assert provenance["new_mode"] == "100644"
    assert provenance["old_sha256"] == hashlib.sha256(original).hexdigest()
    assert provenance["new_sha256"] == hashlib.sha256(binary).hexdigest()
    assert artifact["metadata"]["warnings"] == [{
        "code": "PARTIAL_ANALYSIS",
        "file": "payload.py",
        "detail": "Binary content detected in post-change snapshot; structural parsing and line counts were skipped.",
    }]


@pytest.mark.parametrize(
    ("old_content", "new_content", "binary_sides"),
    [
        (b"\x00old-binary\xff\n", b"def readable():\n    return 1\n", "pre-change"),
        (b"\x00old-binary\xff\n", b"\x00new-binary\xfe\n", "pre-change and post-change"),
    ],
    ids=["binary-to-text", "binary-to-binary"],
)

def test_pre_change_binary_snapshots_skip_all_source_analysis(
    tmp_path, old_content, new_content, binary_sides
):
    """Pre-change binary content suppresses analysis for either or both sides."""
    root = repo(tmp_path)
    path = root / "payload.py"
    path.write_bytes(old_content)
    commit(root)
    path.write_bytes(new_content)

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    assert artifact["symbols"] == []
    assert artifact["relationships"] == []
    assert artifact["metadata"]["files_analyzed"] == 0
    assert artifact["metadata"]["files_skipped"] == 1
    file_entry = artifact["files"][0]
    assert file_entry["language"] is None
    assert file_entry["lines_added"] is None
    assert file_entry["lines_removed"] is None
    provenance = json.loads(file_entry["evidence"][0]["detail"])
    assert provenance["old_oid"] == git(root, "rev-parse", "HEAD:payload.py")
    assert provenance["new_oid"] == git(root, "hash-object", "--path=payload.py", "payload.py")
    assert provenance["old_sha256"] == hashlib.sha256(old_content).hexdigest()
    assert provenance["new_sha256"] == hashlib.sha256(new_content).hexdigest()
    assert artifact["metadata"]["warnings"] == [{
        "code": "PARTIAL_ANALYSIS",
        "file": "payload.py",
        "detail": (
            f"Binary content detected in {binary_sides} snapshot; "
            "structural parsing and line counts were skipped."
        ),
    }]


@pytest.mark.parametrize("change_kind", ["added", "deleted"])
def test_added_and_deleted_binary_snapshots_preserve_one_sided_identity(
    tmp_path, change_kind
):
    """One-sided binary changes preserve evidence only for the present side."""
    root = repo(tmp_path)
    binary = b"\x00binary-snapshot\xff\n"
    path = root / "payload.py"
    write(root, "anchor.txt", "committed\n")
    if change_kind == "deleted":
        path.write_bytes(binary)
    commit(root)

    if change_kind == "added":
        path.write_bytes(binary)
        git(root, "add", "--", "payload.py")
    else:
        path.unlink()

    artifact = analyze_local_diff(str(root), staged=change_kind == "added")

    assert_valid(artifact)
    assert artifact["symbols"] == []
    assert artifact["relationships"] == []
    assert artifact["metadata"]["files_analyzed"] == 0
    assert artifact["metadata"]["files_skipped"] == 1
    file_entry = artifact["files"][0]
    assert file_entry["change_kind"] == change_kind
    assert file_entry["language"] is None
    assert file_entry["lines_added"] is None
    assert file_entry["lines_removed"] is None
    provenance = json.loads(file_entry["evidence"][0]["detail"])
    binary_oid = (
        git(root, "hash-object", "--path=payload.py", "payload.py")
        if change_kind == "added"
        else git(root, "rev-parse", "HEAD:payload.py")
    )
    binary_side = "new" if change_kind == "added" else "old"
    absent_side = "old" if change_kind == "added" else "new"
    assert provenance[f"{binary_side}_oid"] == binary_oid
    assert provenance[f"{binary_side}_mode"] == "100644"
    assert provenance[f"{binary_side}_sha256"] == hashlib.sha256(binary).hexdigest()
    assert provenance[f"{absent_side}_oid"] is None
    assert provenance[f"{absent_side}_mode"] is None
    assert provenance[f"{absent_side}_sha256"] is None
    warning_side = "post-change" if change_kind == "added" else "pre-change"
    assert artifact["metadata"]["warnings"] == [{
        "code": "PARTIAL_ANALYSIS",
        "file": "payload.py",
        "detail": (
            f"Binary content detected in {warning_side} snapshot; "
            "structural parsing and line counts were skipped."
        ),
    }]


def test_unstaged_binary_snapshot_uses_index_before_worktree(tmp_path):
    """Unstaged evidence compares the index snapshot with the worktree."""
    root = repo(tmp_path)
    path = root / "payload.py"
    committed = b"def committed():\n    return 1\n"
    staged_binary = b"\x00staged-binary\xff\n"
    worktree_text = b"def worktree():\n    return 2\n"
    path.write_bytes(committed)
    commit(root)
    path.write_bytes(staged_binary)
    git(root, "add", "--", "payload.py")
    index_oid = git(root, "rev-parse", ":payload.py")
    path.write_bytes(worktree_text)

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    assert artifact["symbols"] == []
    assert artifact["relationships"] == []
    assert artifact["metadata"]["files_analyzed"] == 0
    assert artifact["metadata"]["files_skipped"] == 1
    file_entry = artifact["files"][0]
    assert file_entry["language"] is None
    assert file_entry["lines_added"] is None
    assert file_entry["lines_removed"] is None
    provenance = json.loads(file_entry["evidence"][0]["detail"])
    assert provenance["old_oid"] == index_oid
    assert provenance["new_oid"] == git(
        root, "hash-object", "--path=payload.py", "payload.py"
    )
    assert provenance["old_oid"] != git(root, "rev-parse", "HEAD:payload.py")
    assert provenance["old_mode"] == "100644"
    assert provenance["new_mode"] == "100644"
    assert provenance["old_sha256"] == hashlib.sha256(staged_binary).hexdigest()
    assert provenance["new_sha256"] == hashlib.sha256(worktree_text).hexdigest()
    assert artifact["metadata"]["warnings"] == [{
        "code": "PARTIAL_ANALYSIS",
        "file": "payload.py",
        "detail": "Binary content detected in pre-change snapshot; structural parsing and line counts were skipped.",
    }]


def test_file_fallback_reports_structural_line_statistics(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "notes.txt", "first\nsecond\n")
    commit(root)
    write(root, "notes.txt", "changed\nsecond\nthird\n")

    artifact = analyze_local_diff(str(root))
    file_entry = artifact["files"][0]
    assert file_entry["lines_added"] == 2
    assert file_entry["lines_removed"] == 1

    monkeypatch.chdir(root)
    result = CliRunner().invoke(main, ["diff", "--format", "terminal"])

    assert result.exit_code == 0, result.output
    assert "notes.txt  +2 / -1" in result.stdout


def test_parse_failure_is_scoped_and_matches_the_golden_fixture(tmp_path):
    """A syntax error has a stable, scoped artifact instead of false topology."""
    root = repo(tmp_path)
    write(root, "broken.py", "def valid():\n    return 1\n")
    commit(root)
    write(root, "broken.py", "def broken(:\n")

    artifact = analyze_local_diff(str(root))
    assert_valid(artifact)
    golden_path = Path(__file__).parent / "fixtures/python_parser_failure.json"
    expected = json.loads(golden_path.read_text())
    actual = {
        "files": [{
            key: file_entry[key]
            for key in ("path", "old_path", "language", "change_kind")
        } for file_entry in artifact["files"]],
        "symbols": artifact["symbols"],
        "relationships": artifact["relationships"],
        "metadata": {
            key: artifact["metadata"][key]
            for key in ("files_analyzed", "files_skipped", "warnings")
        },
    }
    assert actual == expected


def test_pre_change_parse_failure_names_the_immutable_blob(tmp_path):
    """A broken pre-change snapshot remains reproducible after it is fixed."""
    root = repo(tmp_path)
    write(root, "recovered.py", "def broken(:\n")
    commit(root)
    old_oid = git(root, "rev-parse", "HEAD:recovered.py")
    write(root, "recovered.py", "def recovered():\n    return True\n")

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    assert artifact["metadata"]["files_analyzed"] == 0
    assert artifact["metadata"]["files_skipped"] == 1
    assert artifact["symbols"] == []
    assert artifact["metadata"]["warnings"] == [{
        "code": "PARSE_FAILURE",
        "file": "recovered.py",
        "detail": (
            "pre-change blob {}: ValueError: Tree-sitter reported a syntax error "
            "at line 1, column 12"
        ).format(old_oid),
    }]


def test_no_network_calls(monkeypatch, tmp_path):
    root = repo(tmp_path)
    write(root, "a.py", "def a():\n    pass\n")
    git(root, "add", "a.py")

    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    artifact = analyze_local_diff(str(root), staged=True)
    assert_valid(artifact)
    assert artifact["metadata"]["privacy_tier"] == "local"
    assert artifact["metadata"]["llm_calls"] == 0


def test_cli_structural_json_is_additive_and_stdout_is_valid_json(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "cli.py", "def old():\n    return 1\n")
    commit(root)
    write(root, "cli.py", "def old():\n    return 2\n")
    monkeypatch.chdir(root)

    result = CliRunner().invoke(main, ["--structural-json", "-", "diff"])
    assert result.exit_code == 0, result.output
    artifact = json.loads(result.output)
    assert_valid(artifact)
    assert artifact["symbols"][0]["change_kind"] == "modified"


def test_cli_terminal_format_renders_validated_local_artifact(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "cli.py", "def value():\n    return 1\n")
    commit(root)
    write(root, "cli.py", "def value():\n    return 2\n")
    monkeypatch.chdir(root)

    result = CliRunner().invoke(main, ["diff", "--format", "terminal"])

    assert result.exit_code == 0, result.output
    assert "wild diff" in result.output
    assert "cli.py" in result.output
    assert "value" in result.output
    assert "Analysis: structural" in result.output


def test_cli_terminal_compact_hides_context(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "cli.py", "def changed():\n    return 1\n\ndef context():\n    return 1\n")
    commit(root)
    write(root, "cli.py", "def changed():\n    return 2\n\ndef context():\n    return 1\n")
    monkeypatch.chdir(root)

    result = CliRunner().invoke(main, ["diff", "--format", "terminal", "--compact"])

    assert result.exit_code == 0, result.output
    assert "REVIEW NEXT" in result.output
    assert "CONTEXT" not in result.output


def test_cli_terminal_all_disables_review_item_cap(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    names = [f"item_{index:02d}" for index in range(11)]
    before = "\n\n".join(f"def {name}():\n    return 1" for name in names) + "\n"
    after = "\n\n".join(f"def {name}():\n    return 2" for name in names) + "\n"
    write(root, "cli.py", before)
    commit(root)
    write(root, "cli.py", after)
    monkeypatch.chdir(root)

    result = CliRunner().invoke(main, ["diff", "--format", "terminal", "--all"])

    assert result.exit_code == 0, result.output
    assert "item_10" in result.output
    assert "more" not in result.output

def test_symlink_type_change_is_an_opaque_snapshot_not_python_source(tmp_path):
    """A symlink target ending in .py must not create false topology."""
    root = repo(tmp_path)
    write(root, "module.py", "def previous():\n    return 1\n")
    commit(root)

    (root / "module.py").unlink()
    os.symlink("replacement.py", root / "module.py")
    git(root, "add", "-A")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    file_entry = artifact["files"][0]
    provenance = json.loads(file_entry["evidence"][0]["detail"])
    assert provenance["old_mode"] == "100644"
    assert provenance["new_mode"] == "120000"
    assert file_entry["language"] is None
    assert file_entry["lines_added"] is None
    assert file_entry["lines_removed"] is None
    assert artifact["symbols"] == []
    assert artifact["relationships"] == []
    assert artifact["metadata"]["files_skipped"] == 1
    assert artifact["metadata"]["warnings"] == [{
        "code": "PARTIAL_ANALYSIS",
        "file": "module.py",
        "detail": "Non-regular Git mode 120000 detected; structural parsing and line counts were skipped.",
    }]


def assert_opaque_gitlink(artifact, expected_old_oid, expected_new_oid):
    """Assert that a gitlink keeps commit provenance without source claims."""
    assert_valid(artifact)
    file_entry = next(item for item in artifact["files"] if item["path"] == "vendor/tool")
    provenance = json.loads(file_entry["evidence"][0]["detail"])
    assert provenance["old_mode"] in (None, "160000")
    assert provenance["new_mode"] in (None, "160000")
    assert provenance["old_oid"] == expected_old_oid
    assert provenance["new_oid"] == expected_new_oid
    assert file_entry["language"] is None
    assert file_entry["lines_added"] is None
    assert file_entry["lines_removed"] is None
    assert artifact["symbols"] == []
    assert artifact["relationships"] == []
    assert artifact["metadata"]["files_skipped"] == 1
    assert artifact["metadata"]["warnings"] == [{
        "code": "PARTIAL_ANALYSIS",
        "file": "vendor/tool",
        "detail": "Non-regular Git mode 160000 detected; structural parsing and line counts were skipped.",
    }]


def test_staged_gitlink_is_an_opaque_commit_snapshot(tmp_path):
    """A staged gitlink addition must preserve its commit ID without a blob read."""
    root = repo(tmp_path)
    write(root, "tracked.txt", "baseline\n")
    commit(root)
    gitlink_oid = git(root, "rev-parse", "HEAD")
    git(root, "update-index", "--add", "--cacheinfo", "160000,{},vendor/tool".format(gitlink_oid))

    artifact = analyze_local_diff(str(root), staged=True)

    assert_opaque_gitlink(artifact, None, gitlink_oid)


def test_unstaged_gitlink_uses_checked_out_submodule_commit(tmp_path):
    """An unstaged gitlink change must resolve the checked-out submodule HEAD."""
    child = tmp_path / "child"
    child.mkdir()
    git(child, "init")
    git(child, "config", "user.name", "Structural Tests")
    git(child, "config", "user.email", "structural@example.test")
    write(child, "value.txt", "one\n")
    commit(child)

    root = repo(tmp_path)
    git(root, "-c", "protocol.file.allow=always", "submodule", "add", str(child), "vendor/tool")
    commit(root)
    old_oid = git(root / "vendor/tool", "rev-parse", "HEAD")
    git(root / "vendor/tool", "config", "user.name", "Structural Tests")
    git(root / "vendor/tool", "config", "user.email", "structural@example.test")
    write(root / "vendor/tool", "value.txt", "two\n")
    commit(root / "vendor/tool")
    new_oid = git(root / "vendor/tool", "rev-parse", "HEAD")

    artifact = analyze_local_diff(str(root))

    assert_opaque_gitlink(artifact, old_oid, new_oid)


def test_commit_range_gitlink_is_an_opaque_commit_snapshot(tmp_path):
    """A commit-range gitlink change must preserve both commit IDs without reads."""
    root = repo(tmp_path)
    write(root, "tracked.txt", "first\n")
    commit(root)
    first_oid = git(root, "rev-parse", "HEAD")
    write(root, "tracked.txt", "second\n")
    commit(root)
    second_oid = git(root, "rev-parse", "HEAD")

    git(root, "update-index", "--add", "--cacheinfo", "160000,{},vendor/tool".format(first_oid))
    git(root, "commit", "-m", "add gitlink")
    base_oid = git(root, "rev-parse", "HEAD")
    git(root, "update-index", "--cacheinfo", "160000,{},vendor/tool".format(second_oid))
    git(root, "commit", "-m", "update gitlink")
    head_oid = git(root, "rev-parse", "HEAD")

    artifact = analyze_local_diff(
        str(root), base_ref=base_oid, head_ref=head_oid
    )

    assert_opaque_gitlink(artifact, first_oid, second_oid)


@pytest.mark.parametrize("pathspec", ["--compact", "--all"])
def test_cli_terminal_preserves_flag_like_pathspec_after_separator(
    pathspec, tmp_path, monkeypatch
):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, pathspec, "before\n")
    commit(root)
    write(root, pathspec, "after\n")
    monkeypatch.chdir(root)

    result = CliRunner().invoke(
        main, ["diff", "--format", "terminal", "--", pathspec]
    )

    assert result.exit_code == 0, result.output
    assert pathspec in result.output


def test_cli_git_passthrough_preserves_all(monkeypatch):
    from click.testing import CliRunner
    import diffgraph.cli as cli

    calls = []

    def run(command, *args, **kwargs):
        calls.append(command)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(cli.subprocess, "run", run)

    result = CliRunner().invoke(cli.main, ["branch", "--all"])

    assert result.exit_code == 0, result.output
    assert calls == [["git", "branch", "--all"]]


def test_cli_default_format_uses_terminal_stdout_without_creating_html(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import diffgraph.cli as cli

    root = repo(tmp_path)
    monkeypatch.chdir(root)

    result = CliRunner().invoke(cli.main, ["diff"])

    assert result.exit_code == 0, result.output
    assert "No changes in the selected snapshot." in result.stdout
    assert not (root / "diffgraph.html").exists()


def test_cli_legacy_html_retains_no_change_compatibility(monkeypatch):
    import sys
    from types import ModuleType

    from click.testing import CliRunner
    import diffgraph.cli as cli

    spinner_module = ModuleType("click_spinner")
    spinner_module.spinner = object()
    ai_module = ModuleType("diffgraph.ai_analysis")
    ai_module.CodeAnalysisAgent = object
    html_module = ModuleType("diffgraph.html_report")
    html_module.generate_html_report = lambda *args, **kwargs: None
    html_module.AnalysisResult = object
    monkeypatch.setitem(sys.modules, "click_spinner", spinner_module)
    monkeypatch.setitem(sys.modules, "diffgraph.ai_analysis", ai_module)
    monkeypatch.setitem(sys.modules, "diffgraph.html_report", html_module)
    monkeypatch.setattr(cli, "is_git_repo", lambda: True)
    monkeypatch.setattr(cli, "get_changed_files", lambda diff_args: [])

    result = CliRunner().invoke(cli.main, ["diff", "--format", "legacy-html"])

    assert result.exit_code == 0, result.output
    assert "No changes to analyze" in result.output


def test_cli_structural_json_resolves_immutable_two_dot_range(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "cli.py", "def value():\n    return 1\n")
    commit(root)
    base_oid = git(root, "rev-parse", "HEAD")
    old_blob = git(root, "rev-parse", "HEAD:cli.py")
    write(root, "cli.py", "def value():\n    return 2\n")
    commit(root)
    head_oid = git(root, "rev-parse", "HEAD")
    new_blob = git(root, "rev-parse", "HEAD:cli.py")
    monkeypatch.chdir(root)

    result = CliRunner().invoke(main, ["--structural-json", "-", "diff", "HEAD~1..HEAD"])

    assert result.exit_code == 0, result.output
    artifact = json.loads(result.output)
    assert artifact["diff_ref"]["kind"] == "commit_range"
    assert artifact["diff_ref"]["base_ref"] == base_oid
    assert artifact["diff_ref"]["head_ref"] == head_oid
    assert artifact["diff_ref"]["requested_base_ref"] == "HEAD~1"
    assert artifact["diff_ref"]["requested_head_ref"] == "HEAD"
    assert artifact["diff_ref"]["comparison_mode"] == "two_dot"
    provenance = json.loads(artifact["files"][0]["evidence"][0]["detail"])
    assert provenance["old_oid"] == old_blob
    assert provenance["new_oid"] == new_blob


def test_cli_structural_json_three_dot_uses_merge_base_and_pathspec(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "included.py", "def value():\n    return 1\n")
    write(root, "ignored.py", "def ignored():\n    return 1\n")
    commit(root)
    merge_base_oid = git(root, "rev-parse", "HEAD")

    git(root, "checkout", "-b", "left")
    write(root, "left.py", "def left():\n    return 1\n")
    commit(root)
    left_oid = git(root, "rev-parse", "HEAD")

    git(root, "checkout", "-b", "right", merge_base_oid)
    write(root, "included.py", "def value():\n    return 2\n")
    write(root, "ignored.py", "def ignored():\n    return 2\n")
    commit(root)
    head_oid = git(root, "rev-parse", "HEAD")
    monkeypatch.chdir(root)

    result = CliRunner().invoke(
        main,
        ["--structural-json", "-", "diff", "left...right", "--", "included.py"],
    )

    assert result.exit_code == 0, result.output
    artifact = json.loads(result.output)
    assert artifact["diff_ref"] == {
        "kind": "commit_range",
        "base_ref": merge_base_oid,
        "head_ref": head_oid,
        "requested_base_ref": "left",
        "requested_head_ref": "right",
        "comparison_mode": "three_dot",
        "pathspecs": ["included.py"],
        "repo_root": str(root),
    }
    assert artifact["diff_ref"]["base_ref"] != left_oid
    assert [item["path"] for item in artifact["files"]] == ["included.py"]


def test_cli_structural_json_preserves_range_like_pathspec(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "name..with.py", "def value():\n    return 1\n")
    commit(root)
    write(root, "name..with.py", "def value():\n    return 2\n")
    monkeypatch.chdir(root)

    result = CliRunner().invoke(
        main, ["--structural-json", "-", "diff", "--", "name..with.py"]
    )

    assert result.exit_code == 0, result.output
    artifact = json.loads(result.output)
    assert artifact["diff_ref"]["kind"] == "unstaged"
    assert artifact["diff_ref"]["pathspecs"] == ["name..with.py"]


def test_cli_structural_json_rejects_four_dot_range(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "app.py", "def value():\n    return 1\n")
    commit(root)
    monkeypatch.chdir(root)

    result = CliRunner().invoke(
        main, ["--structural-json", "-", "diff", "HEAD....HEAD"]
    )

    assert result.exit_code == 2
    assert "explicit non-empty BASE..HEAD or BASE...HEAD refs" in result.output


def test_cli_structural_json_preserves_invalid_ref_warning_code(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "app.py", "def value():\n    return 1\n")
    commit(root)
    monkeypatch.chdir(root)

    result = CliRunner().invoke(
        main, ["--structural-json", "-", "diff", "missing-ref..HEAD"]
    )

    assert result.exit_code == 0, result.output
    artifact = json.loads(result.output)
    assert artifact["files"] == []
    warning = artifact["metadata"]["warnings"][0]
    assert warning["code"] == "invalid_base_ref"
    assert warning["detail"].startswith("invalid_base_ref:")


def test_cli_structural_json_preserves_scoped_pathspec_warning_code(tmp_path, monkeypatch):
    """Pathspec failures remain actionable structured artifact warnings."""
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "app.py", "def value():\n    return 1\n")
    commit(root)
    monkeypatch.chdir(root)

    outside = tmp_path / "outside.py"
    result = CliRunner().invoke(
        main, ["--structural-json", "-", "diff", "--", str(outside)]
    )

    assert result.exit_code == 0, result.output
    artifact = json.loads(result.output)
    assert artifact["files"] == []
    assert artifact["metadata"]["warnings"] == [{
        "code": "pathspec_outside_repository",
        "file": str(outside),
        "detail": "pathspec_outside_repository: Absolute pathspec is outside the repository and was not resolved",
    }]


def test_cli_structural_json_requires_separator_before_pathspecs(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "app.py", "def value():\n    return 1\n")
    commit(root)
    write(root, "app.py", "def value():\n    return 2\n")
    monkeypatch.chdir(root)

    result = CliRunner().invoke(
        main, ["--structural-json", "-", "diff", "app.py"]
    )

    assert result.exit_code == 2
    assert "put pathspecs after '--'" in result.output


def test_methods_nested_functions_and_deleted_imports_are_not_overclaimed(tmp_path):
    root = repo(tmp_path)
    write(
        root,
        "nested.py",
        "import removed_pkg\n\nclass Container:\n    def method(self):\n        return 1\n\ndef outer():\n    def inner():\n        return 1\n    return inner()\n",
    )
    commit(root)
    write(
        root,
        "nested.py",
        "class Container:\n    def method(self):\n        return 2\n\ndef outer():\n    def inner():\n        return 2\n    return inner()\n",
    )

    artifact = analyze_local_diff(str(root))
    assert_valid(artifact)
    symbols = {item["qualified_name"]: item for item in artifact["symbols"]}
    assert symbols["Container.method"]["kind"] == "method"
    assert symbols["outer.inner"]["kind"] == "function"
    assert symbols["import::removed_pkg"]["change_kind"] == "deleted"
    assert symbols["import::removed_pkg"]["location"] is None
    assert not any(
        item["kind"] == "imports"
        and item["target_id"] == symbols["import::removed_pkg"]["id"]
        for item in artifact["relationships"]
    )


def test_async_python_functions_are_structural_symbols_and_callers(tmp_path):
    """Async declarations must be represented like their synchronous peers."""
    root = repo(tmp_path)
    write(
        root,
        "service.py",
        "async def helper():\n    return 1\n\n"
        "async def orchestrate():\n"
        "    return await helper()\n\n"
        "class Service:\n"
        "    async def run(self):\n"
        "        return await helper()\n",
    )
    commit(root)
    write(
        root,
        "service.py",
        "async def helper():\n    return 2\n\n"
        "async def orchestrate():\n"
        "    return await helper()\n\n"
        "class Service:\n"
        "    async def run(self):\n"
        "        return await helper()\n",
    )

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    symbols = {item["id"]: item for item in artifact["symbols"]}
    assert symbols["sym::service.py::helper"]["kind"] == "function"
    assert symbols["sym::service.py::helper"]["change_kind"] == "modified"
    assert symbols["sym::service.py::orchestrate"]["kind"] == "function"
    assert symbols["sym::service.py::Service.run"]["kind"] == "method"
    assert symbols["sym::service.py::Service.run"]["parent_id"] == "sym::service.py::Service"
    relationships = {
        (item["kind"], item["source_id"], item["target_id"])
        for item in artifact["relationships"]
    }
    assert ("contains", "sym::service.py::Service", "sym::service.py::Service.run") in relationships
    assert ("calls", "sym::service.py::orchestrate", "sym::service.py::helper") in relationships
    assert ("calls", "sym::service.py::Service.run", "sym::service.py::helper") in relationships


def test_module_constants_and_type_aliases_are_deterministic_symbols(tmp_path):
    """Python's unambiguous module declarations belong in the local graph."""
    root = repo(tmp_path)
    write(
        root,
        "settings.py",
        "from typing import TypeAlias\n\n"
        "MAX_RETRIES = 3\n"
        "UserId: TypeAlias = str\n"
        "type Payload = dict[str, int]\n"
        "runtime_value = 1\n",
    )
    commit(root)
    write(
        root,
        "settings.py",
        "from typing import TypeAlias\n\n"
        "MAX_RETRIES = 5\n"
        "DEFAULT_TIMEOUT = 30\n"
        "type Payload = dict[str, str]\n"
        "runtime_value = 2\n",
    )

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    symbols = {item["qualified_name"]: item for item in artifact["symbols"]}
    assert symbols["MAX_RETRIES"]["kind"] == "constant"
    assert symbols["MAX_RETRIES"]["change_kind"] == "modified"
    assert symbols["DEFAULT_TIMEOUT"]["kind"] == "constant"
    assert symbols["DEFAULT_TIMEOUT"]["change_kind"] == "added"
    assert symbols["UserId"]["kind"] == "type_alias"
    assert symbols["UserId"]["change_kind"] == "deleted"
    assert symbols["Payload"]["kind"] == "type_alias"
    assert symbols["Payload"]["change_kind"] == "modified"
    assert "runtime_value" not in symbols
    defines = {
        (item["source_id"], item["target_id"])
        for item in artifact["relationships"]
        if item["kind"] == "defines"
    }
    assert ("file::settings.py", "sym::settings.py::MAX_RETRIES") in defines
    assert ("file::settings.py", "sym::settings.py::DEFAULT_TIMEOUT") in defines
    assert ("file::settings.py", "sym::settings.py::Payload") in defines
    assert ("file::settings.py", "sym::settings.py::UserId") not in defines


def test_duplicate_symbol_occurrences_are_preserved(tmp_path):
    root = repo(tmp_path)
    write(
        root,
        "properties.py",
        "class Item:\n"
        "    @property\n"
        "    def value(self):\n"
        "        return 1\n\n"
        "    @value.setter\n"
        "    def value(self, new):\n"
        "        self._value = new\n",
    )
    commit(root)
    write(
        root,
        "properties.py",
        "class Item:\n"
        "    @property\n"
        "    def value(self):\n"
        "        return 1\n\n"
        "    @value.setter\n"
        "    def value(self, new):\n"
        "        self._value = new + 1\n",
    )

    artifact = analyze_local_diff(str(root))
    assert_valid(artifact)
    values = {
        item["qualified_name"]: item
        for item in artifact["symbols"]
        if item["name"] == "value"
    }
    assert set(values) == {"Item.value", "Item.value#1"}
    assert values["Item.value"]["change_kind"] == "unchanged"
    assert values["Item.value#1"]["change_kind"] == "modified"


def test_aliased_import_uses_name_field_and_reports_alias_edit_as_modified(tmp_path):
    root = repo(tmp_path)
    write(root, "imports.py", "import package \\\n    as old_alias\n")
    commit(root)
    write(root, "imports.py", "import package \\\n    as new_alias\n")

    artifact = analyze_local_diff(str(root))
    assert_valid(artifact)
    imported = next(item for item in artifact["symbols"] if item["kind"] == "import")
    assert imported["name"] == "package"
    assert imported["qualified_name"] == "import::package"
    assert imported["change_kind"] == "modified"


def test_worktree_symlink_preserves_exact_link_bytes_without_source_analysis(tmp_path):
    root = repo(tmp_path)
    os.symlink("original.py", root / "link.py")
    commit(root)
    os.unlink(root / "link.py")
    os.symlink("replacement.py", root / "link.py")

    artifact = analyze_local_diff(str(root))
    assert_valid(artifact)
    assert artifact["metadata"]["files_analyzed"] == 0
    assert artifact["metadata"]["files_skipped"] == 1
    file_entry = artifact["files"][0]
    assert file_entry["language"] is None
    assert file_entry["lines_added"] is None
    assert file_entry["lines_removed"] is None
    assert artifact["metadata"]["warnings"] == [{
        "code": "PARTIAL_ANALYSIS",
        "file": "link.py",
        "detail": "Non-regular Git mode 120000 detected; structural parsing and line counts were skipped.",
    }]
    provenance = json.loads(file_entry["evidence"][0]["detail"])
    assert provenance["new_oid"] == git(
        root, "hash-object", "--stdin", input_bytes=b"replacement.py"
    )


def test_snapshot_read_failure_is_partial_analysis(monkeypatch, tmp_path):
    root = repo(tmp_path)
    write(root, "app.py", "def value():\n    return 1\n")
    commit(root)
    write(root, "app.py", "def value():\n    return 2\n")

    def fail_read(*args, **kwargs):
        raise GitSnapshotError("simulated read race")

    monkeypatch.setattr("diffgraph.structural.read_worktree_blob", fail_read)
    artifact = analyze_local_diff(str(root))
    assert_valid(artifact)
    assert artifact["metadata"]["files_analyzed"] == 0
    assert artifact["metadata"]["files_skipped"] == 1
    warning = artifact["metadata"]["warnings"][0]
    assert warning["code"] == "PARTIAL_ANALYSIS"
    assert "simulated read race" in warning["detail"]


def test_resolution_warning_preserves_machine_readable_code(monkeypatch, tmp_path):
    root = repo(tmp_path)
    warning = ResolutionWarning("hash_object_failed", "simulated failure", "app.py")
    monkeypatch.setattr(
        "diffgraph.structural.resolve_unstaged",
        lambda repository, pathspecs: SnapshotResolution((), (warning,)),
    )

    artifact = analyze_local_diff(str(root))
    assert_valid(artifact)
    assert artifact["metadata"]["warnings"] == [
        {
            "code": "hash_object_failed",
            "file": "app.py",
            "detail": "hash_object_failed: simulated failure",
        }
    ]


def test_ambiguous_merge_base_warning_is_not_downgraded(monkeypatch, tmp_path):
    """Three-dot ambiguity remains actionable for JSON consumers."""
    root = repo(tmp_path)
    warning = ResolutionWarning(
        "ambiguous_merge_base", "Git returned 2 merge bases; choose a base", None
    )
    monkeypatch.setattr(
        "diffgraph.structural.resolve_commit_range",
        lambda *args, **kwargs: CommitRangeResolution(
            (), (warning,), "left", "right", None, None, None, True
        ),
    )

    artifact = analyze_local_diff(
        str(root), base_ref="left", head_ref="right", three_dot=True
    )

    assert_valid(artifact)
    assert artifact["metadata"]["warnings"] == [{
        "code": "ambiguous_merge_base",
        "detail": "ambiguous_merge_base: Git returned 2 merge bases; choose a base",
    }]


def test_undecodable_path_warning_is_schema_valid(monkeypatch, tmp_path):
    root = repo(tmp_path)
    warning = ResolutionWarning(
        "undecodable_path", "Git reported a non-UTF-8 filename", None
    )
    monkeypatch.setattr(
        "diffgraph.structural.resolve_unstaged",
        lambda repository, pathspecs: SnapshotResolution((), (warning,)),
    )

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    assert artifact["metadata"]["warnings"] == [{
        "code": "undecodable_path",
        "detail": "undecodable_path: Git reported a non-UTF-8 filename",
    }]


def test_unmerged_index_warning_preserves_machine_readable_code(monkeypatch, tmp_path):
    root = repo(tmp_path)
    warning = ResolutionWarning("unmerged_index_entry", "conflict remains", "app.py")
    monkeypatch.setattr(
        "diffgraph.structural.resolve_unstaged",
        lambda repository, pathspecs: SnapshotResolution((), (warning,)),
    )

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    assert artifact["metadata"]["warnings"] == [
        {
            "code": "unmerged_index_entry",
            "file": "app.py",
            "detail": "unmerged_index_entry: conflict remains",
        }
    ]


def test_cli_structural_json_rejects_non_diff_command():
    from click.testing import CliRunner
    from diffgraph.cli import main

    result = CliRunner().invoke(main, ["--structural-json", "out.json", "status"])
    assert result.exit_code == 2
    assert "can only be used with 'diff'" in result.output


def test_cli_pathspec_is_relative_to_calling_subdirectory(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "src/app.py", "def value():\n    return 1\n")
    write(root, "app.py", "def root_value():\n    return 1\n")
    commit(root)
    write(root, "src/app.py", "def value():\n    return 2\n")
    write(root, "app.py", "def root_value():\n    return 2\n")
    monkeypatch.chdir(root / "src")

    result = CliRunner().invoke(
        main, ["--structural-json", "-", "diff", "--", "app.py"]
    )
    assert result.exit_code == 0, result.output
    artifact = json.loads(result.output)
    assert [item["path"] for item in artifact["files"]] == ["src/app.py"]


def test_cli_missing_structural_output_parent_is_a_click_error(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    write(root, "app.py", "def value():\n    return 1\n")
    git(root, "add", "app.py")
    monkeypatch.chdir(root)
    output = root / "nested" / "artifact.json"

    result = CliRunner().invoke(
        main, ["--structural-json", str(output), "diff", "--staged"]
    )
    assert result.exit_code == 1
    assert "could not write" in result.output
    assert "Traceback" not in result.output
    assert not output.exists()


def test_schema_errors_become_click_errors(monkeypatch):
    from click import ClickException
    from diffgraph.contract import DiffGraphContractError
    import diffgraph.cli as cli

    def invalid_artifact(*args, **kwargs):
        raise DiffGraphContractError("invalid schema")

    monkeypatch.setattr(cli, "validate_artifact", invalid_artifact)
    with pytest.raises(ClickException, match="structural artifact validation failed"):
        cli._validate_structural_artifact({})


def test_missing_parser_dependency_is_a_scoped_structural_warning(tmp_path, monkeypatch):
    """A parser installation failure must not turn exact Git input into a false change."""
    from diffgraph.structural import StructuralDependencyError

    root = repo(tmp_path)
    write(root, "app.py", "def value():\n    return 1\n")
    git(root, "add", "app.py")

    def unavailable():
        raise StructuralDependencyError("parser dependency is unavailable")

    monkeypatch.setattr("diffgraph.structural._parser", unavailable)
    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    assert artifact["symbols"] == []
    assert artifact["relationships"] == []
    assert artifact["metadata"]["files_analyzed"] == 0
    assert artifact["metadata"]["files_skipped"] == 1
    assert artifact["metadata"]["warnings"] == [{
        "code": "PARSE_FAILURE",
        "file": "app.py",
        "detail": (
            "post-change blob {}: StructuralDependencyError: "
            "parser dependency is unavailable"
        ).format(git(root, "hash-object", "app.py")),
    }]


def test_cli_emits_valid_artifact_when_parser_dependency_is_unavailable(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from diffgraph.cli import main
    from diffgraph.structural import StructuralDependencyError

    root = repo(tmp_path)
    write(root, "app.py", "def value():\n    return 1\n")
    git(root, "add", "app.py")
    monkeypatch.chdir(root)

    def unavailable():
        raise StructuralDependencyError("parser dependency is unavailable")

    monkeypatch.setattr("diffgraph.structural._parser", unavailable)
    result = CliRunner().invoke(
        main, ["--structural-json", "-", "diff", "--staged"]
    )

    assert result.exit_code == 0, result.output
    artifact = json.loads(result.output)
    assert_valid(artifact)
    assert artifact["metadata"]["warnings"][0]["code"] == "PARSE_FAILURE"


def test_missing_ai_dependency_is_a_click_error(tmp_path, monkeypatch):
    import builtins
    from click.testing import CliRunner
    from diffgraph.cli import main

    root = repo(tmp_path)
    monkeypatch.chdir(root)
    real_import = builtins.__import__

    def without_spinner(name, *args, **kwargs):
        if name == "click_spinner":
            raise ImportError("simulated missing spinner")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_spinner)
    result = CliRunner().invoke(main, ["diff", "--format", "legacy-html"])
    assert result.exit_code == 1
    assert "requires additional dependencies" in result.output
    assert "Traceback" not in result.output


def test_python_calls_are_conservative_schema_valid_and_golden(tmp_path):
    root = repo(tmp_path)
    write(
        root,
        "calls.py",
        "def helper():\n"
        "    return 1\n\n"
        "def caller():\n"
        "    helper()\n"
        "    helper()\n\n"
        "def outer():\n"
        "    def nested():\n"
        "        return 1\n"
        "    nested()\n\n"
        "def parameter_shadow(helper):\n"
        "    helper()\n\n"
        "def assignment_shadow():\n"
        "    helper = lambda: 2\n"
        "    helper()\n\n"
        "def attribute_call(service):\n"
        "    service.helper()\n\n"
        "def default_call(value=helper()):\n"
        "    return value\n\n"
        "helper()\n\n"
        "def closure_shadow():\n"
        "    helper = lambda: 3\n"
        "    def inner():\n"
        "        helper()\n\n"
        "class MethodShadow:\n"
        "    def method(self):\n"
        "        helper = lambda: 4\n"
        "        def inner():\n"
        "            helper()\n",
    )
    git(root, "add", "calls.py")

    artifact = analyze_local_diff(str(root), staged=True)
    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    actual = [
        {
            "id": item["id"],
            "source_id": item["source_id"],
            "target_id": item["target_id"],
            "resolution_method": item["resolution_method"],
            "line": item["evidence"][0]["line_start"],
            "snippet": item["evidence"][0]["snippet"],
        }
        for item in calls
    ]
    golden_path = Path(__file__).parent / "fixtures/python_calls.json"
    expected = json.loads(golden_path.read_text())
    if os.environ.get("UPDATE_GOLDEN"):
        golden_path.write_text(json.dumps(actual, indent=2) + "\n")
        pytest.skip("golden fixture regenerated")
    assert actual == expected

    # Parameter/local bindings and attribute dispatch are intentionally not
    # guessed. Every emitted edge has exact call-site/parser/blob evidence.
    assert len(calls) == 5
    assert all(item["analysis_source"] == "structural" for item in calls)
    assert all(item["confidence"] is None for item in calls)
    shadowed_callers = {
        "sym::calls.py::closure_shadow.inner",
        "sym::calls.py::MethodShadow.method.inner",
    }
    symbol_ids = {item["id"] for item in artifact["symbols"]}
    assert shadowed_callers <= symbol_ids
    assert shadowed_callers.isdisjoint(item["source_id"] for item in calls)
    assert all("query=python-structure-v2" in item["evidence"][0]["detail"] for item in calls)
    assert all("blob=" in item["evidence"][0]["detail"] for item in calls)


def test_explicit_from_import_creates_import_grounded_call_edge(tmp_path):
    root = repo(tmp_path)
    write(
        root,
        "external_calls.py",
        "from remote.worker import execute as run_remote\n\n"
        "def caller():\n"
        "    run_remote()\n",
    )
    git(root, "add", "external_calls.py")

    artifact = analyze_local_diff(str(root), staged=True)
    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    call = calls[0]
    assert call["source_id"] == "sym::external_calls.py::caller"
    assert call["target_id"] == "sym::external_calls.py::import::remote.worker"
    assert call["resolution_method"] == "import_grounded"
    assert call["confidence"] is None
    assert call["evidence"][0]["kind"] == "call_site"
    assert call["evidence"][0]["snippet"] == "run_remote()"
    assert "query=python-structure-v2" in call["evidence"][0]["detail"]


def test_function_local_imports_ground_only_their_lexical_calls(tmp_path):
    """A local import is exact evidence for its own function, not its siblings."""
    root = repo(tmp_path)
    write(
        root,
        "local_import.py",
        "def imports_and_calls():\n"
        "    from remote.worker import execute as run_remote\n"
        "    run_remote()\n\n"
        "def sibling():\n"
        "    run_remote()\n",
    )
    git(root, "add", "local_import.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["source_id"] == "sym::local_import.py::imports_and_calls"
    assert calls[0]["target_id"] == "sym::local_import.py::import::remote.worker"
    assert calls[0]["resolution_method"] == "import_grounded"
    assert calls[0]["evidence"][0]["snippet"] == "run_remote()"


def test_class_body_imports_do_not_shadow_nested_method_lookups(tmp_path):
    """Class-body imports resolve there but remain outside method lexical scope."""
    root = repo(tmp_path)
    write(
        root,
        "class_import.py",
        "def enclosing():\n"
        "    from outer.worker import execute as run_remote\n\n"
        "    class Worker:\n"
        "        from class_body.worker import execute as run_remote\n"
        "        run_remote()\n\n"
        "        def method(self):\n"
        "            run_remote()\n",
    )
    git(root, "add", "class_import.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 2
    assert [item["source_id"] for item in calls] == [
        "sym::class_import.py::enclosing.Worker",
        "sym::class_import.py::enclosing.Worker.method",
    ]
    assert [item["target_id"] for item in calls] == [
        "sym::class_import.py::import::class_body.worker",
        "sym::class_import.py::import::outer.worker",
    ]
    assert all(item["resolution_method"] == "import_grounded" for item in calls)


def test_relative_from_imports_preserve_package_evidence_and_call_bindings(tmp_path):
    """Relative imports are explicit package-local external dependencies.

    The extractor intentionally does not guess whether the target module is in
    the changed snapshot, but it must preserve the exact relative spelling and
    alias binding so clients can distinguish ``..pkg`` from an absolute module.
    """
    root = repo(tmp_path)
    write(
        root,
        "relative_calls.py",
        "from ..pkg import execute as run_local\n\n"
        "def caller():\n"
        "    run_local()\n",
    )
    git(root, "add", "relative_calls.py")

    artifact = analyze_local_diff(str(root), staged=True)
    assert_valid(artifact)
    imported = next(
        item
        for item in artifact["symbols"]
        if item["id"] == "sym::relative_calls.py::import::..pkg"
    )
    assert imported["name"] == "..pkg"
    assert imported["change_kind"] == "added"
    import_edge = next(
        item
        for item in artifact["relationships"]
        if item["kind"] == "imports"
    )
    assert import_edge["label"] == "unresolved/external module: ..pkg"
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["source_id"] == "sym::relative_calls.py::caller"
    assert calls[0]["target_id"] == "sym::relative_calls.py::import::..pkg"
    assert calls[0]["resolution_method"] == "import_grounded"
    assert calls[0]["evidence"][0]["snippet"] == "run_local()"


def test_changeset_python_imports_resolve_across_files_conservatively(tmp_path):
    """Changed package modules provide deterministic import and call targets."""
    root = repo(tmp_path)
    write(
        root,
        "package/worker.py",
        "def run():\n"
        "    return 1\n",
    )
    write(
        root,
        "package/client.py",
        "from .worker import run as run_worker\n"
        "from remote.worker import execute as run_remote\n\n"
        "def caller():\n"
        "    run_worker()\n"
        "    run_remote()\n\n"
        "def shadowed():\n"
        "    run_worker = lambda: None\n"
        "    run_worker()\n",
    )
    write(
        root,
        "fallback.py",
        "from package.worker import not_a_declaration\n\n"
        "def caller():\n"
        "    not_a_declaration()\n",
    )
    git(root, "add", "package", "fallback.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    relationships = {
        (item["kind"], item["source_id"], item["target_id"]): item
        for item in artifact["relationships"]
    }
    assert (
        "imports", "file::package/client.py", "file::package/worker.py"
    ) in relationships
    assert relationships[(
        "imports", "file::package/client.py", "file::package/worker.py"
    )]["label"] == "resolved module: .worker"
    assert (
        "calls", "sym::package/client.py::caller", "sym::package/worker.py::run"
    ) in relationships
    assert relationships[(
        "calls", "sym::package/client.py::caller", "sym::package/worker.py::run"
    )]["resolution_method"] == "resolved"
    assert (
        "imports", "file::fallback.py", "file::package/worker.py"
    ) in relationships
    # A name that is not a callable declaration must retain the local import
    # stub; a call edge must never point at a file node.
    assert (
        "calls", "sym::fallback.py::caller",
        "sym::fallback.py::import::package.worker",
    ) in relationships
    assert relationships[(
        "calls", "sym::fallback.py::caller",
        "sym::fallback.py::import::package.worker",
    )]["resolution_method"] == "import_grounded"
    # The explicit external import and non-callable import stay represented by
    # local stubs, while callable internal imports resolve directly.
    symbol_ids = {item["id"] for item in artifact["symbols"]}
    assert "sym::package/client.py::import::remote.worker" in symbol_ids
    assert "sym::package/client.py::import::.worker" in symbol_ids
    assert "sym::fallback.py::import::package.worker" in symbol_ids
    assert not any(
        item["kind"] == "calls"
        and item["source_id"] == "sym::package/client.py::shadowed"
        for item in artifact["relationships"]
    )


def test_resolved_import_bindings_link_files_to_imported_declarations(tmp_path):
    """Named imports record which changed declaration they bind, and only that."""
    root = repo(tmp_path)
    write(
        root,
        "package/worker.py",
        "LIMIT = 3\n\n"
        "def run():\n"
        "    return LIMIT\n\n"
        "def unused():\n"
        "    return 0\n\n"
        "class Engine:\n"
        "    def start(self):\n"
        "        return run()\n",
    )
    write(
        root,
        "package/client.py",
        "from .worker import run as go, LIMIT, Engine, missing\n"
        "from .worker import run\n"
        "import package.worker\n",
    )
    git(root, "add", "package")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    bindings = {
        (item["source_id"], item["target_id"]): item["label"]
        for item in artifact["relationships"]
        if item["kind"] == "imports" and item["label"].startswith("resolved binding")
    }
    # One edge per imported declaration, even when imported twice; undeclared
    # names, methods, and whole-module imports bind no declaration.
    assert bindings == {
        ("file::package/client.py", "sym::package/worker.py::run"): "resolved binding: go",
        ("file::package/client.py", "sym::package/worker.py::LIMIT"): "resolved binding: LIMIT",
        ("file::package/client.py", "sym::package/worker.py::Engine"): "resolved binding: Engine",
    }
    relationship_ids = [item["id"] for item in artifact["relationships"]]
    assert len(relationship_ids) == len(set(relationship_ids))


def test_namespace_package_child_imports_resolve_without_init_files(tmp_path):
    """Absolute and relative child imports resolve across namespace packages."""
    root = repo(tmp_path)
    write(root, "package/worker.py", "def run():\n    return 1\n")
    write(root, "absolute.py", "from package import worker\n")
    write(root, "package/relative.py", "from . import worker\n")
    git(root, "add", "package", "absolute.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    imports = {
        (item["source_id"], item["target_id"]): item
        for item in artifact["relationships"]
        if item["kind"] == "imports"
    }
    assert ("file::absolute.py", "file::package/worker.py") in imports
    assert ("file::package/relative.py", "file::package/worker.py") in imports
    assert imports[("file::absolute.py", "file::package/worker.py")]["label"] == (
        "resolved module: package"
    )
    assert imports[("file::package/relative.py", "file::package/worker.py")]["label"] == (
        "resolved module: ."
    )


def test_cross_file_import_resolution_handles_module_named_callables(tmp_path):
    """A callable with the same name as its module resolves to its symbol."""
    root = repo(tmp_path)
    write(root, "helpers.py", "def helpers():\n    return 1\n")
    write(
        root,
        "consumer.py",
        "from helpers import helpers\n\n"
        "def caller():\n"
        "    helpers()\n",
    )
    git(root, "add", "helpers.py", "consumer.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    assert (
        "calls", "sym::consumer.py::caller", "sym::helpers.py::helpers"
    ) in {
        (item["kind"], item["source_id"], item["target_id"])
        for item in artifact["relationships"]
    }


def test_cross_file_import_resolution_keeps_duplicate_callables_ambiguous(tmp_path):
    """Repeated declarations must not select the first callable arbitrarily."""
    root = repo(tmp_path)
    write(root, "helpers.py", "def run():\n    return 1\n\ndef run():\n    return 2\n")
    write(
        root,
        "consumer.py",
        "from helpers import run\n\n"
        "def caller():\n"
        "    run()\n",
    )
    git(root, "add", "helpers.py", "consumer.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    call_targets = {
        item["target_id"]
        for item in artifact["relationships"]
        if item["kind"] == "calls" and item["source_id"] == "sym::consumer.py::caller"
    }
    assert call_targets == {"sym::consumer.py::import::helpers"}


def test_resolved_import_relationship_ids_are_unique(tmp_path):
    """Multiple imports from one changed module retain distinct edge IDs."""
    root = repo(tmp_path)
    write(root, "package/worker.py", "def run():\n    return 1\n\ndef other():\n    return 2\n")
    write(
        root,
        "package/client.py",
        "from .worker import run\n"
        "from .worker import other\n",
    )
    git(root, "add", "package")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    resolved_imports = [
        item for item in artifact["relationships"]
        if item["kind"] == "imports"
        and item["source_id"] == "file::package/client.py"
        and item["target_id"] == "file::package/worker.py"
    ]
    assert len(resolved_imports) == 2
    assert {item["id"] for item in resolved_imports} == {
        "rel::file::package/client.py->file::package/worker.py",
        "rel::file::package/client.py->file::package/worker.py#1",
    }
    assert len({item["id"] for item in artifact["relationships"]}) == len(
        artifact["relationships"]
    )


def test_as_pattern_bindings_do_not_create_import_grounded_call_edges(tmp_path):
    """with/except aliases shadow imports before their bodies execute."""
    root = repo(tmp_path)
    write(
        root,
        "as_pattern_bindings.py",
        "from remote.worker import execute as run_remote\n\n"
        "def with_shadow(resource):\n"
        "    with resource() as run_remote:\n"
        "        run_remote()\n\n"
        "def except_shadow():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as run_remote:\n"
        "        run_remote()\n",
    )
    git(root, "add", "as_pattern_bindings.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert calls == []


def test_as_binding_keeps_import_visible_in_context_expression(tmp_path):
    """An ``as`` alias binds only after its context expression runs."""
    root = repo(tmp_path)
    write(
        root,
        "as_binding_order.py",
        "from remote.worker import execute as run_remote\n\n"
        "with run_remote() as run_remote:\n"
        "    run_remote()\n",
    )
    git(root, "add", "as_binding_order.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["resolution_method"] == "import_grounded"
    assert calls[0]["evidence"][0]["line_start"] == 3


def test_local_import_rebinding_stops_import_grounded_call_edges(tmp_path):
    """A local assignment must replace a same-scope imported call target."""
    root = repo(tmp_path)
    write(
        root,
        "local_import_rebinding.py",
        "def caller():\n"
        "    from remote.worker import execute as run_remote\n"
        "    run_remote()\n"
        "    run_remote = lambda: None\n"
        "    run_remote()\n",
    )
    git(root, "add", "local_import_rebinding.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["source_id"] == "sym::local_import_rebinding.py::caller"
    assert calls[0]["resolution_method"] == "import_grounded"
    assert calls[0]["evidence"][0]["line_start"] == 3


def test_attribute_and_annotation_only_assignments_do_not_shadow_imports(tmp_path):
    """Only assignment targets with a value replace an imported binding."""
    root = repo(tmp_path)
    write(
        root,
        "assignment_binding_precision.py",
        "from remote.worker import execute as run_remote\n\n"
        "run_remote.result = None\n"
        "run_remote: object\n"
        "run_remote()\n",
    )
    git(root, "add", "assignment_binding_precision.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["resolution_method"] == "import_grounded"


@pytest.mark.parametrize("declaration", ["def run_remote():\n    pass", "class run_remote:\n    pass"])
def test_nested_declaration_rebinds_function_local_import(tmp_path, declaration):
    """Nested declarations replace same-scope imported call targets."""
    root = repo(tmp_path)
    write(
        root,
        "nested_declaration_rebind.py",
        "def caller():\n"
        "    from remote.worker import execute as run_remote\n\n"
        + "    " + declaration.replace("\n", "\n    ") + "\n\n"
        "    run_remote()\n",
    )
    git(root, "add", "nested_declaration_rebind.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert calls == []


def test_named_expression_bindings_do_not_create_import_grounded_call_edges(tmp_path):
    """A walrus target shadows an import for calls after the assignment."""
    root = repo(tmp_path)
    write(
        root,
        "named_expression_bindings.py",
        "from remote.worker import execute as run_remote\n\n"
        "def factory():\n"
        "    return lambda: None\n\n"
        "def caller():\n"
        "    if (run_remote := factory()):\n"
        "        run_remote()\n",
    )
    git(root, "add", "named_expression_bindings.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert [(item["source_id"], item["target_id"]) for item in calls] == [
        ("sym::named_expression_bindings.py::caller", "sym::named_expression_bindings.py::factory")
    ]


def test_module_named_expression_keeps_import_visible_during_its_value(tmp_path):
    """A walrus rebind takes effect after its value expression is evaluated."""
    root = repo(tmp_path)
    write(
        root,
        "named_expression_order.py",
        "from remote.worker import execute as run_remote\n\n"
        "if (run_remote := run_remote()):\n"
        "    pass\n",
    )
    git(root, "add", "named_expression_order.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["resolution_method"] == "import_grounded"


def test_module_for_target_keeps_import_visible_during_iterable_evaluation(tmp_path):
    """A loop target binds after its iterable expression is evaluated."""
    root = repo(tmp_path)
    write(
        root,
        "for_binding_order.py",
        "from remote.worker import execute as run_remote\n\n"
        "for run_remote in range(run_remote()):\n"
        "    pass\n",
    )
    git(root, "add", "for_binding_order.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["resolution_method"] == "import_grounded"


def test_comprehension_targets_shadow_imports_only_inside_comprehensions(tmp_path):
    """Comprehension targets shadow imports without leaking into their function."""
    root = repo(tmp_path)
    write(
        root,
        "comprehension_bindings.py",
        "from remote.worker import execute as run_remote\n\n"
        "def build(values):\n"
        "    run_remote()\n"
        "    result = [run_remote() for run_remote in values]\n"
        "    run_remote()\n"
        "    return result\n",
    )
    git(root, "add", "comprehension_bindings.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 2
    assert {item["evidence"][0]["line_start"] for item in calls} == {4, 6}
    assert all(item["resolution_method"] == "import_grounded" for item in calls)


def test_comprehension_clauses_bind_targets_in_evaluation_order(tmp_path):
    """Comprehension iterables see only targets from earlier clauses."""
    root = repo(tmp_path)
    write(
        root,
        "comprehension_clause_order.py",
        "from remote.worker import execute as run_remote\n\n"
        "def build(values):\n"
        "    own_iterable = [item for run_remote in run_remote()]\n"
        "    earlier_iterable = [item for item in run_remote() for run_remote in values]\n"
        "    later_iterable = [item for run_remote in values for item in run_remote()]\n"
        "    return own_iterable, earlier_iterable, later_iterable\n",
    )
    git(root, "add", "comprehension_clause_order.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert {item["evidence"][0]["line_start"] for item in calls} == {4, 5}
    assert all(item["resolution_method"] == "import_grounded" for item in calls)


def test_comprehension_filter_does_not_bind_later_targets(tmp_path):
    """A filter sees prior targets but not names bound by later clauses."""
    root = repo(tmp_path)
    write(
        root,
        "comprehension_filter_order.py",
        "from remote.worker import execute as run_remote\n\n"
        "def build(values, sources):\n"
        "    return [item for item in values "
        "if run_remote() for run_remote in sources]\n",
    )
    git(root, "add", "comprehension_filter_order.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["evidence"][0]["line_start"] == 4
    assert calls[0]["resolution_method"] == "import_grounded"


def test_match_pattern_captures_do_not_create_import_grounded_call_edges(tmp_path):
    """Python match captures, including splats, shadow imports in case bodies."""
    root = repo(tmp_path)
    write(
        root,
        "match_pattern_bindings.py",
        "from remote.worker import execute as run_remote\n\n"
        "def bare_capture(value):\n"
        "    match value:\n"
        "        case run_remote:\n"
        "            run_remote()\n\n"
        "def sequence_capture(value):\n"
        "    match value:\n"
        "        case [run_remote]:\n"
        "            run_remote()\n\n"
        "def splat_capture(value):\n"
        "    match value:\n"
        "        case [*run_remote]:\n"
        "            run_remote()\n\n"
        "def keyword_capture(value):\n"
        "    match value:\n"
        "        case Point(callback=run_remote):\n"
        "            run_remote()\n",
    )
    git(root, "add", "match_pattern_bindings.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert calls == []


def test_rebound_import_does_not_create_import_grounded_call_edge(tmp_path):
    root = repo(tmp_path)
    write(
        root,
        "rebound_import.py",
        "from remote.worker import execute as run_remote\n\n"
        "run_remote = lambda: None\n\n"
        "def caller():\n"
        "    run_remote()\n",
    )
    git(root, "add", "rebound_import.py")

    artifact = analyze_local_diff(str(root), staged=True)
    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert calls == []


@pytest.mark.parametrize(
    "binding_statement",
    [
        "with context() as run_remote:\n    pass",
        "try:\n    pass\nexcept Exception as run_remote:\n    pass",
    ],
)
def test_module_as_binding_rebinds_imported_alias(tmp_path, binding_statement):
    """Module ``as`` bindings suppress later import-grounded call edges."""
    root = repo(tmp_path)
    write(
        root,
        "as_binding_rebind.py",
        "from remote.worker import execute as run_remote\n\n"
        + binding_statement
        + "\n\nrun_remote()\n",
    )
    git(root, "add", "as_binding_rebind.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert calls == []


def test_as_binding_attribute_target_does_not_shadow_import(tmp_path):
    """An attribute ``as`` target does not assign its object name."""
    root = repo(tmp_path)
    write(
        root,
        "as_attribute_target.py",
        "from remote.worker import execute as run_remote\n\n"
        "with context() as run_remote.result:\n"
        "    pass\n\n"
        "run_remote()\n",
    )
    git(root, "add", "as_attribute_target.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["resolution_method"] == "import_grounded"


def test_as_binding_destructuring_rebinds_each_imported_alias(tmp_path):
    """A destructured ``as`` target binds each assigned identifier."""
    root = repo(tmp_path)
    write(
        root,
        "as_destructure.py",
        "from remote.worker import execute as run_remote\n"
        "from remote.worker import summarize as summarize_remote\n\n"
        "with context() as (run_remote, summarize_remote):\n"
        "    pass\n\n"
        "run_remote()\n"
        "summarize_remote()\n",
    )
    git(root, "add", "as_destructure.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert calls == []


def test_lambda_named_expression_does_not_shadow_an_enclosing_import(tmp_path):
    """A lambda-local walrus target must not leak into the enclosing function."""
    root = repo(tmp_path)
    write(
        root,
        "lambda_named_expression.py",
        "from remote.worker import execute as run_remote\n\n"
        "def caller():\n"
        "    thunk = lambda: (run_remote := 1)\n"
        "    return run_remote()\n",
    )
    git(root, "add", "lambda_named_expression.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["resolution_method"] == "import_grounded"


def test_declaration_headers_keep_imports_visible_before_rebinding(tmp_path):
    """Default and base expressions run before their top-level names bind."""
    root = repo(tmp_path)
    write(
        root,
        "declaration_header_order.py",
        "from remote.worker import execute as run_remote\n\n"
        "def function_rebind(value=run_remote()):\n"
        "    pass\n\n"
        "class class_rebind(run_remote()):\n"
        "    pass\n",
    )
    git(root, "add", "declaration_header_order.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 2
    assert all(item["resolution_method"] == "import_grounded" for item in calls)


@pytest.mark.parametrize("declaration", ["def run_remote():\n    return None", "class run_remote:\n    pass"])
def test_module_declaration_rebinds_imported_alias(tmp_path, declaration):
    root = repo(tmp_path)
    write(
        root,
        "declaration_rebind.py",
        "from remote.worker import execute as run_remote\n\n"
        + declaration + "\n\n"
        "def caller():\n"
        "    run_remote()\n",
    )
    git(root, "add", "declaration_rebind.py")

    artifact = analyze_local_diff(str(root), staged=True)
    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert all(item["resolution_method"] != "import_grounded" for item in calls)


def test_pep_695_type_alias_rebinds_imported_alias(tmp_path):
    """A module alias shadows an earlier import for conservative call edges."""
    root = repo(tmp_path)
    write(
        root,
        "type_alias_rebind.py",
        "from remote.worker import execute as Alias\n\n"
        "type Alias = int\n\n"
        "def caller():\n"
        "    Alias()\n",
    )
    git(root, "add", "type_alias_rebind.py")

    artifact = analyze_local_diff(str(root), staged=True)

    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert calls == []


def test_import_binding_remains_visible_before_later_rebind(tmp_path):
    root = repo(tmp_path)
    write(
        root,
        "line_aware_rebind.py",
        "from remote.worker import execute as run_remote\n\n"
        "run_remote()\n\n"
        "run_remote = lambda: None\n",
    )
    git(root, "add", "line_aware_rebind.py")

    artifact = analyze_local_diff(str(root), staged=True)
    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["target_id"] == "sym::line_aware_rebind.py::import::remote.worker"
    assert calls[0]["resolution_method"] == "import_grounded"


def test_later_import_does_not_hide_earlier_local_call(tmp_path):
    root = repo(tmp_path)
    write(
        root,
        "local_before_import.py",
        "def run():\n"
        "    return None\n\n"
        "run()\n\n"
        "from remote.worker import execute as run\n",
    )
    git(root, "add", "local_before_import.py")

    artifact = analyze_local_diff(str(root), staged=True)
    assert_valid(artifact)
    calls = [item for item in artifact["relationships"] if item["kind"] == "calls"]
    assert len(calls) == 1
    assert calls[0]["target_id"] == "sym::local_before_import.py::run"
    assert calls[0]["resolution_method"] == "resolved"


def test_decorator_change_is_part_of_python_declaration_evidence(tmp_path):
    """Decorator edits modify the wrapped declaration rather than disappearing."""
    root = repo(tmp_path)
    write(
        root,
        "decorated.py",
        "@first\ndef decorated():\n    return 1\n",
    )
    commit(root)
    write(
        root,
        "decorated.py",
        "@second\ndef decorated():\n    return 1\n",
    )

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    symbol = next(item for item in artifact["symbols"] if item["name"] == "decorated")
    assert symbol["kind"] == "function"
    assert symbol["change_kind"] == "modified"
    assert symbol["evidence"] == [{
        "kind": "ast_parse",
        "file": "decorated.py",
        "line_start": 1,
        "line_end": 3,
        "detail": symbol["evidence"][0]["detail"],
    }]


def test_changed_symbols_carry_exact_before_and_after_source(tmp_path):
    root = repo(tmp_path)
    write(root, "mod.py", "def keep():\n    return 0\n\ndef edit():\n    return 1\n\ndef gone():\n    return 2\n")
    write(root, "notes.md", "one\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")
    write(root, "mod.py", "def keep():\n    return 0\n\ndef edit():\n    return 10\n\ndef new():\n    return 3\n")
    write(root, "notes.md", "one\ntwo\n")

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    symbols = {item["name"]: item for item in artifact["symbols"]}
    assert "source" not in symbols["keep"]
    assert symbols["edit"]["source"] == {
        "before": {"line_start": 4, "text": "def edit():\n    return 1\n", "truncated": False},
        "after": {"line_start": 4, "text": "def edit():\n    return 10\n", "truncated": False},
    }
    assert symbols["gone"]["source"]["after"] is None
    assert symbols["gone"]["source"]["before"]["text"] == "def gone():\n    return 2\n"
    assert symbols["new"]["source"]["before"] is None

    files = {item["path"]: item for item in artifact["files"]}
    assert files["notes.md"]["diff"]["truncated"] is False
    assert "+two\n" in files["notes.md"]["diff"]["unified"]
    assert files["notes.md"]["diff"]["unified"].startswith("--- a/notes.md\n+++ b/notes.md\n")
    assert "-    return 1\n+    return 10\n" in files["mod.py"]["diff"]["unified"]


def _callers_repo(tmp_path):
    root = repo(tmp_path)
    write(root, "pkg/__init__.py", "")
    write(root, "pkg/lib.py", "def f(a):\n    return a\n\ndef g():\n    return 0\n\ndef stable():\n    return 1\n")
    write(root, "pkg/caller.py", "from pkg.lib import f as run\n\ndef use():\n    return run(1)\n")
    write(root, "pkg/broken.py", "from .lib import g\n\nclass Box:\n    def open(self):\n        return g()\n")
    write(root, "pkg/mention.py", "# f is mentioned here but never imported\ndef f2():\n    return 'f'\n")
    write(root, "pkg/other.py", "from pkg.lib import stable\n\ndef keep():\n    return stable()\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")
    # Change f's signature and delete g; callers stay untouched.
    write(root, "pkg/lib.py", "def f(a, b):\n    return a + b\n\ndef stable():\n    return 1\n")
    return root


def _context(artifact):
    files = {item["path"]: item for item in artifact["files"] if item["change_kind"] == "unchanged"}
    edges = {
        (item["kind"], item["source_id"], item["target_id"])
        for item in artifact["relationships"]
        if any(item["source_id"] in (fid, ) or item["source_id"].startswith("sym::" + path + "::")
               for path, fid in ((p, f["id"]) for p, f in files.items()))
    }
    return files, edges


def test_callers_outside_the_diff_are_found_including_callers_of_deleted_functions(tmp_path):
    root = _callers_repo(tmp_path)

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    files, edges = _context(artifact)
    # Only files that really call or import changed declarations; a textual
    # mention (mention.py) and a caller of an unchanged function (other.py) are not context.
    assert set(files) == {"pkg/caller.py", "pkg/broken.py"}
    assert ("calls", "sym::pkg/caller.py::use", "sym::pkg/lib.py::f") in edges
    assert ("imports", "file::pkg/caller.py", "sym::pkg/lib.py::f") in edges
    # The call site of a deleted function is exactly what breaks.
    assert ("calls", "sym::pkg/broken.py::Box.open", "sym::pkg/lib.py::g") in edges
    symbols = {item["id"]: item for item in artifact["symbols"]}
    assert symbols["sym::pkg/lib.py::g"]["change_kind"] == "deleted"
    # Callers keep their enclosing declarations so parent ids resolve.
    assert symbols["sym::pkg/broken.py::Box.open"]["parent_id"] == "sym::pkg/broken.py::Box"
    assert symbols["sym::pkg/broken.py::Box"]["change_kind"] == "unchanged"
    call = next(item for item in artifact["relationships"] if item["source_id"] == "sym::pkg/caller.py::use")
    assert call["evidence"][0]["line_start"] == 4 and call["evidence"][0]["snippet"] == "run(1)"
    assert files["pkg/caller.py"]["lines_added"] == 0


def test_callers_outside_the_diff_use_the_head_commit_of_a_range(tmp_path):
    root = _callers_repo(tmp_path)
    git(root, "commit", "-am", "change")
    # A later working-tree edit must not leak into a commit-range analysis.
    write(root, "pkg/caller.py", "def use():\n    return 0\n")

    artifact = analyze_local_diff(str(root), base_ref="HEAD~1", head_ref="HEAD")

    files, edges = _context(artifact)
    assert ("calls", "sym::pkg/caller.py::use", "sym::pkg/lib.py::f") in edges


def test_caller_search_is_capped_with_a_warning(tmp_path, monkeypatch):
    import diffgraph.structural as structural

    root = _callers_repo(tmp_path)
    monkeypatch.setattr(structural, "MAX_CONTEXT_FILES", 1)

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    assert any(
        item["code"] == "PARTIAL_ANALYSIS" and "first 1 of" in item["detail"]
        for item in artifact["metadata"]["warnings"]
    )


def test_inherits_edges_resolve_local_imported_and_outside_the_diff(tmp_path):
    root = repo(tmp_path)
    write(root, "pkg/__init__.py", "")
    write(root, "pkg/base.py", "class Base:\n    pass\n")
    write(root, "pkg/sub.py", "class Plain:\n    pass\n")
    write(root, "pkg/user.py", "from pkg.base import Base\n\nclass Outside(Base):\n    pass\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")
    write(root, "pkg/base.py", "class Base:\n    x = 1\n")
    write(
        root, "pkg/sub.py",
        "import abc\nfrom pkg.base import Base\n\n"
        "def helper():\n    pass\n\n"
        "class Local:\n    pass\n\n"
        "class Child(Base, Local, abc.ABC, helper, metaclass=abc.ABCMeta):\n    pass\n",
    )

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    inherits = {
        (item["source_id"], item["target_id"]): item
        for item in artifact["relationships"] if item["kind"] == "inherits"
    }
    # Imported base in another changed file, a local base, and a base in the
    # diff subclassed by an untouched file. Attribute bases (abc.ABC), a
    # function used as a base, and metaclass keywords are not inheritance edges.
    assert set(inherits) == {
        ("sym::pkg/sub.py::Child", "sym::pkg/base.py::Base"),
        ("sym::pkg/sub.py::Child", "sym::pkg/sub.py::Local"),
        ("sym::pkg/user.py::Outside", "sym::pkg/base.py::Base"),
    }
    evidence = inherits[("sym::pkg/sub.py::Child", "sym::pkg/base.py::Base")]["evidence"][0]
    assert evidence["line_start"] == 10 and evidence["snippet"].startswith("class Child(Base")
    outside = {item["path"]: item for item in artifact["files"]}
    assert outside["pkg/user.py"]["change_kind"] == "unchanged"


def test_symbol_source_and_file_diff_split_lines_like_git(tmp_path):
    """Form feeds must not shift spans; a missing final newline gets Git's marker."""
    root = repo(tmp_path)
    write(root, "mod.py", "\x0c\ndef page():\n    return 1\n\ndef last():\n    return 2")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")
    write(root, "mod.py", "\x0c\ndef page():\n    return 1\n\ndef last():\n    return 3")

    artifact = analyze_local_diff(str(root))

    symbols = {item["name"]: item for item in artifact["symbols"]}
    assert symbols["last"]["source"]["after"] == {
        "line_start": 5, "text": "def last():\n    return 3", "truncated": False,
    }
    unified = next(item for item in artifact["files"] if item["path"] == "mod.py")["diff"]["unified"]
    assert "-    return 2\n\\ No newline at end of file\n+    return 3\n\\ No newline at end of file\n" in unified


def test_inherits_edges_require_a_confirmed_class_target(tmp_path):
    """External bases, and functions or constants used as bases, are not
    inheritance edges; the import symbol and its evidence remain."""
    root = repo(tmp_path)
    write(root, "pkg/__init__.py", "")
    write(root, "pkg/base.py", "def factory():\n    return object\n\nclass Base:\n    pass\n\nclass Old:\n    pass\n")
    write(root, "pkg/outside.py",
          "from pkg.base import factory, Base, Old\n\n"
          "class FromFunction(factory):\n    pass\n\n"
          "class FromClass(Base):\n    pass\n\n"
          "class FromDeleted(Old):\n    pass\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")
    # factory and Base change, Old is deleted; outside.py is untouched.
    write(root, "pkg/base.py", "def factory():\n    return dict\n\nclass Base:\n    x = 1\n")
    write(root, "pkg/inside.py",
          "from external import Framework\nfrom pkg.base import factory\n\n"
          "class UsesExternal(Framework):\n    pass\n\n"
          "class UsesFunction(factory):\n    pass\n")

    artifact = analyze_local_diff(str(root))

    assert_valid(artifact)
    inherits = {
        (item["source_id"], item["target_id"])
        for item in artifact["relationships"] if item["kind"] == "inherits"
    }
    assert inherits == {
        ("sym::pkg/outside.py::FromClass", "sym::pkg/base.py::Base"),
        ("sym::pkg/outside.py::FromDeleted", "sym::pkg/base.py::Old"),
    }
    # The external import is still recorded as an import symbol with evidence.
    stub = next(item for item in artifact["symbols"] if item["id"] == "sym::pkg/inside.py::import::external")
    assert stub["kind"] == "import" and stub["evidence"][0]["snippet"] == "from external import Framework"


def _ref_repo(tmp_path):
    root = repo(tmp_path)
    write(root, "app.py", "def one():\n    return 1\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")
    write(root, "app.py", "def one():\n    return 1\n\ndef two():\n    return 2\n")
    git(root, "commit", "-am", "second")
    write(root, "app.py", "def one():\n    return 10\n\ndef two():\n    return 2\n")
    write(root, "new.py", "def fresh():\n    return 0\n")  # untracked
    return root


def test_single_ref_compares_that_commit_with_the_working_tree(tmp_path):
    root = _ref_repo(tmp_path)

    artifact = analyze_local_diff(str(root), against="HEAD~1")

    assert_valid(artifact)
    ref = artifact["diff_ref"]
    assert ref["kind"] == "working_tree" and ref["requested_base_ref"] == "HEAD~1"
    assert ref["base_ref"] == git(root, "rev-parse", "HEAD~1").strip()
    changes = {item["name"]: item["change_kind"] for item in artifact["symbols"] if item["kind"] != "import"}
    # Committed (two), uncommitted (one), and untracked (fresh) changes all count.
    assert changes == {"one": "modified", "two": "added", "fresh": "added"}


def test_staged_single_ref_compares_that_commit_with_the_index(tmp_path):
    root = _ref_repo(tmp_path)
    git(root, "add", "app.py")

    artifact = analyze_local_diff(str(root), staged=True, against="HEAD~1")

    ref = artifact["diff_ref"]
    assert ref["kind"] == "staged" and ref["requested_base_ref"] == "HEAD~1"
    names = {item["name"] for item in artifact["symbols"] if item["change_kind"] != "unchanged"}
    assert names == {"one", "two"}  # untracked new.py is not in the index


def test_cli_single_ref_and_unknown_revision(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import diffgraph.cli as cli

    root = _ref_repo(tmp_path)
    monkeypatch.chdir(root)

    ok = CliRunner().invoke(cli.main, ["diff", "--format", "json", "HEAD"])
    assert ok.exit_code == 0, ok.output
    assert json.loads(ok.stdout)["diff_ref"]["kind"] == "working_tree"

    terminal = CliRunner().invoke(cli.main, ["diff", "HEAD~1"])
    assert "HEAD~1 → working tree" in terminal.stdout

    typo = CliRunner().invoke(cli.main, ["diff", "mian"])
    assert typo.exit_code == 2
    assert "unknown revision 'mian'; put pathspecs after '--'" in typo.output

    both = CliRunner().invoke(cli.main, ["diff", "HEAD~1..HEAD", "HEAD"])
    assert both.exit_code == 2
