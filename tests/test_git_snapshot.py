import errno
import os
import subprocess

import pytest

import diffgraph.git_snapshot as git_snapshot
from diffgraph.git_snapshot import (
    resolve_commit_range,
    resolve_staged,
    resolve_unstaged,
)


def git(repo, *args, input_bytes=None):
    completed = subprocess.run(
        ["git"] + list(args),
        cwd=str(repo),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout


def write(repo, path, content):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def write_raw_path(raw_path, content):
    """Write a non-UTF-8 path, skipping where the filesystem rejects it (APFS)."""
    try:
        with open(raw_path, "wb") as handle:
            handle.write(content)
    except OSError as error:
        if error.errno == errno.EILSEQ:
            pytest.skip("filesystem rejects non-UTF-8 file names")
        raise


def oid(repo, revision):
    return git(repo, "rev-parse", revision).decode("ascii").strip()


def index_oid(repo, path):
    output = git(repo, "ls-files", "-s", "--", path)
    return output.split()[1].decode("ascii")


def commit_all(repo, message="baseline"):
    git(repo, "add", "-A")
    git(repo, "commit", "-m", message)


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.name", "Snapshot Tests")
    git(repo, "config", "user.email", "snapshot@example.test")
    return repo


def make_sha256_repo(tmp_path):
    """Create a repository whose immutable object IDs use SHA-256."""
    repo = tmp_path / "sha256-repo"
    repo.mkdir()
    git(repo, "init", "--object-format=sha256")
    git(repo, "config", "user.name", "Snapshot Tests")
    git(repo, "config", "user.email", "snapshot@example.test")
    return repo


def make_conflicted_repo(tmp_path):
    """Create one unresolved merge conflict in an otherwise valid repository."""

    repo = make_repo(tmp_path)
    write(repo, "conflict.py", b"base = True\n")
    commit_all(repo, "base")

    git(repo, "switch", "-c", "side")
    write(repo, "conflict.py", b"side = True\n")
    commit_all(repo, "side")

    git(repo, "switch", "master")
    write(repo, "conflict.py", b"main = True\n")
    commit_all(repo, "main")
    completed = subprocess.run(
        ["git", "merge", "side"], cwd=str(repo), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode != 0
    return repo


def test_staged_add_modify_delete_and_rename_have_exact_identities(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "modify.txt", b"old modify\n")
    write(repo, "delete.txt", b"old delete\n")
    write(repo, "rename-old.txt", b"rename content\n" * 20)
    commit_all(repo)

    old_modify = oid(repo, "HEAD:modify.txt")
    old_delete = oid(repo, "HEAD:delete.txt")
    old_rename = oid(repo, "HEAD:rename-old.txt")

    write(repo, "added.txt", b"new file\n")
    write(repo, "modify.txt", b"new modify\n")
    os.unlink(repo / "delete.txt")
    git(repo, "mv", "rename-old.txt", "rename-new.txt")
    git(repo, "add", "-A")

    result = resolve_staged(str(repo))
    entries = {entry.new_path or entry.old_path: entry for entry in result.entries}

    assert result.warnings == ()
    assert set(entries) == {"added.txt", "modify.txt", "delete.txt", "rename-new.txt"}

    added = entries["added.txt"]
    assert (added.status, added.old_path, added.new_path) == ("A", None, "added.txt")
    assert (added.old_mode, added.old_oid) == (None, None)
    assert (added.new_mode, added.new_oid) == ("100644", index_oid(repo, "added.txt"))

    modified = entries["modify.txt"]
    assert (modified.status, modified.old_path, modified.new_path) == (
        "M",
        "modify.txt",
        "modify.txt",
    )
    assert modified.old_oid == old_modify
    assert modified.new_oid == index_oid(repo, "modify.txt")

    deleted = entries["delete.txt"]
    assert (deleted.status, deleted.old_path, deleted.new_path) == (
        "D",
        "delete.txt",
        None,
    )
    assert (deleted.old_oid, deleted.new_oid) == (old_delete, None)
    assert (deleted.old_mode, deleted.new_mode) == ("100644", None)

    renamed = entries["rename-new.txt"]
    assert (renamed.status, renamed.old_path, renamed.new_path) == (
        "R",
        "rename-old.txt",
        "rename-new.txt",
    )
    assert (renamed.old_oid, renamed.new_oid) == (old_rename, old_rename)
    assert (renamed.old_mode, renamed.new_mode) == ("100644", "100644")


def test_unstaged_modify_and_delete_have_exact_identities(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "modify.txt", b"old\n")
    write(repo, "delete.txt", b"delete me\n")
    commit_all(repo)
    old_modify = oid(repo, "HEAD:modify.txt")
    old_delete = oid(repo, "HEAD:delete.txt")

    write(repo, "modify.txt", b"new working tree bytes\n")
    os.unlink(repo / "delete.txt")

    result = resolve_unstaged(str(repo))
    entries = {entry.new_path or entry.old_path: entry for entry in result.entries}

    assert result.warnings == ()
    assert set(entries) == {"modify.txt", "delete.txt"}
    modified = entries["modify.txt"]
    assert modified.status == "M"
    assert modified.old_oid == old_modify
    assert modified.new_oid == git(
        repo, "hash-object", "--stdin", "--path=modify.txt", input_bytes=b"new working tree bytes\n"
    ).decode("ascii").strip()
    assert len(modified.new_oid) == 40

    deleted = entries["delete.txt"]
    assert deleted.status == "D"
    assert (deleted.old_oid, deleted.new_oid) == (old_delete, None)
    assert (deleted.old_mode, deleted.new_mode) == ("100644", None)


def test_sha256_repository_preserves_full_immutable_blob_identities(tmp_path):
    """Exact snapshot provenance must not assume Git's legacy SHA-1 format."""
    repo = make_sha256_repo(tmp_path)
    write(repo, "module.py", b"value = 1\n")
    commit_all(repo, "base")
    git(repo, "branch", "before")
    old_oid = oid(repo, "HEAD:module.py")

    write(repo, "module.py", b"value = 2\n")
    unstaged = resolve_unstaged(str(repo), ["module.py"])

    assert unstaged.warnings == ()
    assert len(unstaged.entries) == 1
    unstaged_entry = unstaged.entries[0]
    assert (unstaged_entry.old_oid, unstaged_entry.new_oid) == (
        old_oid,
        git(
            repo,
            "hash-object",
            "--stdin",
            "--path=module.py",
            input_bytes=b"value = 2\n",
        ).decode("ascii").strip(),
    )
    assert all(len(item) == 64 for item in (unstaged_entry.old_oid, unstaged_entry.new_oid))

    git(repo, "add", "--", "module.py")
    commit_all(repo, "update")
    ranged = resolve_commit_range(str(repo), "before", "HEAD")

    assert ranged.warnings == ()
    assert ranged.base_oid == oid(repo, "before")
    assert ranged.head_oid == oid(repo, "HEAD")
    assert ranged.comparison_base_oid == ranged.base_oid
    assert all(len(item) == 64 for item in (ranged.base_oid, ranged.head_oid, ranged.comparison_base_oid))
    assert [(entry.old_oid, entry.new_oid) for entry in ranged.entries] == [
        (old_oid, oid(repo, "HEAD:module.py"))
    ]


def test_sha256_staged_snapshot_preserves_full_blob_identities(tmp_path):
    """Staged SHA-256 changes retain complete immutable endpoint identities."""
    repo = make_sha256_repo(tmp_path)
    write(repo, "modified.bin", b"\x00before\xff\n")
    write(repo, "deleted.txt", b"remove me\n")
    commit_all(repo, "base")

    old_modified = oid(repo, "HEAD:modified.bin")
    old_deleted = oid(repo, "HEAD:deleted.txt")
    write(repo, "modified.bin", b"\x00after\xff\n")
    os.unlink(repo / "deleted.txt")
    write(repo, "added.bin", b"\x1f\x8bnew archive")
    git(repo, "add", "-A")
    staged_modified = index_oid(repo, "modified.bin")
    write(repo, "modified.bin", b"\x00worktree-only\xff\n")

    result = resolve_staged(str(repo))
    entries = {entry.new_path or entry.old_path: entry for entry in result.entries}

    assert result.warnings == ()
    assert set(entries) == {"added.bin", "deleted.txt", "modified.bin"}
    assert (entries["modified.bin"].old_oid, entries["modified.bin"].new_oid) == (
        old_modified, staged_modified,
    )
    assert (entries["deleted.txt"].old_oid, entries["deleted.txt"].new_oid) == (
        old_deleted, None,
    )
    assert (entries["added.bin"].old_oid, entries["added.bin"].new_oid) == (
        None, index_oid(repo, "added.bin"),
    )
    for entry in entries.values():
        for object_id in (entry.old_oid, entry.new_oid):
            if object_id is not None:
                assert len(object_id) == 64


def test_unstaged_hash_matches_git_add_clean_filter_semantics(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, ".gitattributes", b"filtered.txt text eol=lf\n")
    write(repo, "filtered.txt", b"old\n")
    commit_all(repo)

    write(repo, "filtered.txt", b"line one\r\nline two\r\n")
    result = resolve_unstaged(str(repo), ["filtered.txt"])
    assert result.warnings == ()
    assert len(result.entries) == 1
    resolved_oid = result.entries[0].new_oid

    git(repo, "add", "--", "filtered.txt")
    assert resolved_oid == index_oid(repo, "filtered.txt")


def test_explicit_pathspec_scope_is_not_widened(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "inside/a.txt", b"old a\n")
    write(repo, "outside/b.txt", b"old b\n")
    commit_all(repo)

    write(repo, "inside/a.txt", b"staged a\n")
    write(repo, "outside/b.txt", b"staged b\n")
    git(repo, "add", "-A")
    write(repo, "inside/a.txt", b"unstaged a\n")
    write(repo, "outside/b.txt", b"unstaged b\n")

    staged = resolve_staged(str(repo), ["inside"])
    unstaged = resolve_unstaged(str(repo), ["inside"])
    no_match = resolve_staged(str(repo), ["does-not-exist"])

    assert [entry.new_path for entry in staged.entries] == ["inside/a.txt"]
    assert [entry.new_path for entry in unstaged.entries] == ["inside/a.txt"]
    assert no_match.entries == ()
    assert no_match.warnings == ()


def test_absolute_pathspec_outside_repository_is_a_scoped_warning(tmp_path):
    """An invalid absolute scope must not turn into an unscoped diff."""
    repo = make_repo(tmp_path)
    write(repo, "tracked.txt", b"old\n")
    commit_all(repo)
    write(repo, "tracked.txt", b"new\n")

    outside = tmp_path / "outside"
    staged = resolve_staged(str(repo), [str(outside)])
    unstaged = resolve_unstaged(str(repo), [str(outside)])
    ranged = resolve_commit_range(
        str(repo), "HEAD", "HEAD", pathspecs=[str(outside)]
    )

    for result in (staged, unstaged, ranged):
        assert result.entries == ()
        assert [(warning.code, warning.path) for warning in result.warnings] == [
            ("pathspec_outside_repository", str(outside))
        ]


def test_relative_pathspec_outside_repository_is_a_scoped_warning(tmp_path):
    """A subdirectory caller must not pass an escaping scope to Git's root."""
    repo = make_repo(tmp_path)
    write(repo, "inside/tracked.txt", b"old\n")
    commit_all(repo)
    write(repo, "inside/tracked.txt", b"new\n")

    caller = repo / "inside"
    pathspec = "../../outside"
    staged = resolve_staged(str(caller), [pathspec])
    unstaged = resolve_unstaged(str(caller), [pathspec])
    ranged = resolve_commit_range(
        str(caller), "HEAD", "HEAD", pathspecs=[pathspec]
    )

    for result in (staged, unstaged, ranged):
        assert result.entries == ()
        assert [(warning.code, warning.path) for warning in result.warnings] == [
            ("pathspec_outside_repository", pathspec)
        ]


def test_windows_relative_pathspec_outside_repository_is_detected(monkeypatch):
    """Windows separators must not bypass the repository-boundary check."""
    monkeypatch.setattr(git_snapshot.os, "sep", "\\")

    assert git_snapshot._pathspec_escapes_repository("inside/..\\..\\outside")


def test_absolute_pathspec_via_symlink_alias_is_in_repository(tmp_path):
    """A symlinked repository path resolves to the canonical repository scope."""
    repo = make_repo(tmp_path)
    write(repo, "tracked.txt", b"old\n")
    commit_all(repo)
    write(repo, "tracked.txt", b"new\n")
    alias = tmp_path / "repo-alias"
    os.symlink(repo, alias, target_is_directory=True)

    result = resolve_unstaged(str(alias), [str(alias / "tracked.txt")])

    assert result.warnings == ()
    assert [entry.new_path for entry in result.entries] == ["tracked.txt"]


def test_nul_parsing_preserves_tabs_and_newlines_in_paths(tmp_path):
    repo = make_repo(tmp_path)
    old_name = "old\tname\npart.txt"
    new_name = "new\nname\tpart.txt"
    write(repo, old_name, b"unusual path\n" * 10)
    commit_all(repo)

    git(repo, "mv", old_name, new_name)
    result = resolve_staged(str(repo))

    assert result.warnings == ()
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.status == "R"
    assert entry.old_path == old_name
    assert entry.new_path == new_name
    assert entry.old_oid == entry.new_oid == oid(repo, "HEAD:" + old_name)


def test_undecodable_untracked_path_is_a_scoped_warning(tmp_path):
    """One non-UTF-8 path must not leak surrogate escapes or hide valid files."""
    repo = make_repo(tmp_path)
    write(repo, "good.py", b"value = 1\n")
    raw_path = os.fsencode(repo) + b"/bad-\xff.py"
    write_raw_path(raw_path, b"value = 2\n")

    result = resolve_unstaged(str(repo))

    assert [entry.new_path for entry in result.entries] == ["good.py"]
    assert [(warning.code, warning.path) for warning in result.warnings] == [
        ("undecodable_path", "bad-\\xff.py")
    ]


def test_staged_undecodable_tracked_path_is_a_scoped_warning(tmp_path):
    """One invalid tracked pathname must not hide valid staged provenance."""
    repo = make_repo(tmp_path)
    write(repo, "good.py", b"before = 1\n")
    raw_path = os.fsencode(repo) + b"/bad-\xff.py"
    write_raw_path(raw_path, b"before = 2\n")
    commit_all(repo)

    write(repo, "good.py", b"after = 1\n")
    write_raw_path(raw_path, b"after = 2\n")
    git(repo, "add", "-A")

    result = resolve_staged(str(repo))

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.status == "M"
    assert entry.old_path == entry.new_path == "good.py"
    assert entry.old_oid == oid(repo, "HEAD:good.py")
    assert entry.new_oid == index_oid(repo, "good.py")
    assert [(warning.code, warning.path) for warning in result.warnings] == [
        ("undecodable_path", "bad-\\xff.py")
    ]


def test_repeated_resolution_is_deterministic(tmp_path):
    repo = make_repo(tmp_path)
    for name in ("z.txt", "a.txt", "middle.txt"):
        write(repo, name, ("old " + name + "\n").encode("ascii"))
    commit_all(repo)
    for name in ("z.txt", "a.txt", "middle.txt"):
        write(repo, name, ("new " + name + "\n").encode("ascii"))

    first = resolve_unstaged(str(repo))
    second = resolve_unstaged(str(repo))

    assert first == second
    assert [entry.new_path for entry in first.entries] == ["a.txt", "middle.txt", "z.txt"]


def test_git_failures_are_warnings_not_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    result = resolve_staged(str(tmp_path))

    assert result.entries == ()
    assert len(result.warnings) == 1
    assert result.warnings[0].code == "not_a_git_repository"


def test_unmerged_index_entry_is_a_warning_not_a_fabricated_snapshot(tmp_path):
    repo = make_conflicted_repo(tmp_path)

    staged = resolve_staged(str(repo))
    unstaged = resolve_unstaged(str(repo))

    assert staged.entries == ()
    assert unstaged.entries == ()
    expected_warning = ("unmerged_index_entry", "conflict.py")
    assert [(warning.code, warning.path) for warning in staged.warnings] == [
        expected_warning
    ]
    assert [(warning.code, warning.path) for warning in unstaged.warnings] == [
        expected_warning
    ]


def test_unstaged_intent_to_add_and_rename_have_exact_identities(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "rename-old.py", b"def moved():\n    return 1\n" * 20)
    commit_all(repo)
    old_oid = oid(repo, "HEAD:rename-old.py")

    os.rename(repo / "rename-old.py", repo / "rename-new.py")
    write(repo, "added.py", b"def added():\n    return 1\n")
    # Intent-to-add lets Git represent working-tree additions without putting
    # their content in the index; the resolver must still derive exact IDs.
    git(repo, "add", "-N", "--", "rename-new.py", "added.py")

    result = resolve_unstaged(str(repo))
    entries = {entry.new_path or entry.old_path: entry for entry in result.entries}
    assert result.warnings == ()
    assert set(entries) == {"rename-new.py", "added.py"}

    renamed = entries["rename-new.py"]
    assert (renamed.status, renamed.old_path, renamed.new_path) == (
        "R",
        "rename-old.py",
        "rename-new.py",
    )
    assert renamed.old_oid == old_oid
    assert renamed.new_oid == git(
        repo,
        "hash-object",
        "--stdin",
        "--path=rename-new.py",
        input_bytes=(repo / "rename-new.py").read_bytes(),
    ).decode("ascii").strip()

    added = entries["added.py"]
    assert (added.status, added.old_path, added.new_path) == ("A", None, "added.py")
    assert added.old_oid is None
    assert added.new_oid == git(
        repo,
        "hash-object",
        "--stdin",
        "--path=added.py",
        input_bytes=(repo / "added.py").read_bytes(),
    ).decode("ascii").strip()


def test_unstaged_includes_non_ignored_untracked_files_with_exact_identities(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, ".gitignore", b"ignored.py\n")
    write(repo, "tracked.py", b"def tracked():\n    return 1\n")
    commit_all(repo)

    write(repo, "new.py", b"def added():\n    return 1\n")
    write(repo, "ignored.py", b"def ignored():\n    return 1\n")
    write(repo, "bin/tool", b"#!/bin/sh\nexit 0\n")
    os.chmod(repo / "bin/tool", 0o700)
    os.symlink("new.py", repo / "new-link")

    result = resolve_unstaged(str(repo))
    entries = {entry.new_path: entry for entry in result.entries}

    assert result.warnings == ()
    assert set(entries) == {"new.py", "bin/tool", "new-link"}
    assert entries["new.py"].status == "A"
    assert entries["new.py"].old_oid is None
    assert entries["new.py"].new_mode == "100644"
    assert entries["new.py"].new_oid == git(
        repo,
        "hash-object",
        "--stdin",
        "--path=new.py",
        input_bytes=(repo / "new.py").read_bytes(),
    ).decode("ascii").strip()
    assert entries["bin/tool"].new_mode == "100755"
    assert entries["new-link"].new_mode == "120000"
    assert entries["new-link"].new_oid == git(
        repo, "hash-object", "--stdin", input_bytes=b"new.py"
    ).decode("ascii").strip()


def test_untracked_pathspec_scope_is_relative_and_not_widened(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "tracked.txt", b"baseline\n")
    commit_all(repo)
    write(repo, "inside/new.py", b"inside = True\n")
    write(repo, "outside.py", b"outside = True\n")

    scoped = resolve_unstaged(str(repo / "inside"), ["new.py"])
    no_match = resolve_unstaged(str(repo), ["missing"])

    assert scoped.warnings == ()
    assert [entry.new_path for entry in scoped.entries] == ["inside/new.py"]
    assert no_match.entries == ()
    assert no_match.warnings == ()


def test_pathspecs_are_relative_to_the_calling_subdirectory(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "src/app.py", b"def value():\n    return 1\n")
    write(repo, "app.py", b"def root_value():\n    return 1\n")
    commit_all(repo)
    write(repo, "src/app.py", b"def value():\n    return 2\n")
    write(repo, "app.py", b"def root_value():\n    return 2\n")

    result = resolve_unstaged(str(repo / "src"), ["app.py"])

    assert result.warnings == ()
    assert [entry.new_path for entry in result.entries] == ["src/app.py"]


def test_unstaged_symlink_hashes_raw_target_without_attributes(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, ".gitattributes", b"*.py filter=decorate\n")
    git(repo, "config", "filter.decorate.clean", "sed 's/^/filtered:/'")
    os.symlink("original.py", repo / "link.py")
    commit_all(repo)

    os.unlink(repo / "link.py")
    os.symlink("replacement.py", repo / "link.py")
    result = resolve_unstaged(str(repo), ["link.py"])

    assert result.warnings == ()
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.new_mode == "120000"
    assert entry.new_oid == git(
        repo, "hash-object", "--stdin", input_bytes=b"replacement.py"
    ).decode("ascii").strip()


def test_two_dot_range_records_exact_endpoints_and_tree_identities(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "shared.txt", b"common\n")
    commit_all(repo, "common")
    git(repo, "branch", "base")

    write(repo, "head-only.txt", b"head\n")
    commit_all(repo, "head change")
    git(repo, "branch", "topic")
    head_oid = oid(repo, "topic")

    git(repo, "switch", "base")
    write(repo, "base-only.txt", b"base\n")
    commit_all(repo, "base change")
    base_oid = oid(repo, "base")

    result = resolve_commit_range(str(repo), "base", "topic")
    entries = {entry.new_path or entry.old_path: entry for entry in result.entries}

    assert result.warnings == ()
    assert result.base_oid == result.comparison_base_oid == base_oid
    assert result.head_oid == head_oid
    assert result.three_dot is False
    assert set(entries) == {"base-only.txt", "head-only.txt"}
    assert entries["base-only.txt"].status == "D"
    assert entries["base-only.txt"].old_oid == oid(repo, "base:base-only.txt")
    assert entries["base-only.txt"].new_oid is None
    assert entries["head-only.txt"].status == "A"
    assert entries["head-only.txt"].old_oid is None
    assert entries["head-only.txt"].new_oid == oid(repo, "topic:head-only.txt")


def test_commit_range_rename_preserves_both_paths_modes_and_blob_identities(tmp_path):
    """A renamed-and-modified file keeps immutable provenance on both sides."""
    repo = make_repo(tmp_path)
    old_path = "package/old_name.py"
    new_path = "package/new_name.py"
    old_content = b"def moved():\n    return 'stable'\n" * 20
    new_content = old_content + b"# post-rename change\n"
    write(repo, old_path, old_content)
    commit_all(repo, "before rename")
    git(repo, "branch", "before")
    old_oid = oid(repo, "HEAD:" + old_path)

    git(repo, "mv", old_path, new_path)
    write(repo, new_path, new_content)
    commit_all(repo, "rename and modify")
    new_oid = oid(repo, "HEAD:" + new_path)

    result = resolve_commit_range(str(repo), "before", "HEAD")

    assert result.warnings == ()
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert (entry.status, entry.old_path, entry.new_path) == (
        "R", old_path, new_path,
    )
    assert (entry.old_mode, entry.new_mode) == ("100644", "100644")
    assert (entry.old_oid, entry.new_oid) == (old_oid, new_oid)
    assert entry.old_oid != entry.new_oid


def test_three_dot_range_uses_merge_base_and_excludes_base_only_changes(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "shared.txt", b"common\n")
    commit_all(repo, "common")
    common_oid = oid(repo, "HEAD")
    git(repo, "branch", "base")

    write(repo, "head-only.txt", b"head\n")
    commit_all(repo, "head change")
    git(repo, "branch", "topic")

    git(repo, "switch", "base")
    write(repo, "base-only.txt", b"base\n")
    commit_all(repo, "base change")

    result = resolve_commit_range(str(repo), "base", "topic", three_dot=True)

    assert result.warnings == ()
    assert result.comparison_base_oid == common_oid
    assert result.base_oid == oid(repo, "base")
    assert result.head_oid == oid(repo, "topic")
    assert result.three_dot is True
    assert [entry.new_path or entry.old_path for entry in result.entries] == [
        "head-only.txt"
    ]


def test_commit_range_pathspec_is_relative_to_calling_subdirectory(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "inside/a.txt", b"old a\n")
    write(repo, "outside/b.txt", b"old b\n")
    commit_all(repo)
    git(repo, "branch", "before")
    write(repo, "inside/a.txt", b"new a\n")
    write(repo, "outside/b.txt", b"new b\n")
    commit_all(repo, "both changed")

    result = resolve_commit_range(
        str(repo / "inside"), "before", "HEAD", pathspecs=["a.txt"]
    )

    assert result.warnings == ()
    assert [entry.new_path for entry in result.entries] == ["inside/a.txt"]


def test_invalid_commit_range_ref_is_a_warning_not_a_change(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "tracked.txt", b"content\n")
    commit_all(repo)

    result = resolve_commit_range(str(repo), "missing-ref", "HEAD")

    assert result.entries == ()
    assert result.base_oid is None
    assert result.head_oid == oid(repo, "HEAD")
    assert [warning.code for warning in result.warnings] == ["invalid_base_ref"]


def test_three_dot_without_merge_base_is_a_warning_not_a_change(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "first.txt", b"first\n")
    commit_all(repo)
    git(repo, "branch", "first")
    git(repo, "switch", "--orphan", "second")
    first_path = repo / "first.txt"
    if first_path.exists():
        first_path.unlink()
    write(repo, "second.txt", b"second\n")
    commit_all(repo, "unrelated root")

    result = resolve_commit_range(str(repo), "first", "second", three_dot=True)

    assert result.entries == ()
    assert result.comparison_base_oid is None
    assert [warning.code for warning in result.warnings] == ["merge_base_failed"]


def test_three_dot_with_multiple_merge_bases_is_a_warning_not_a_change(tmp_path):
    """A criss-cross history must not silently select one merge base."""
    repo = make_repo(tmp_path)
    write(repo, "shared.txt", b"common\n")
    commit_all(repo, "common")
    git(repo, "branch", "left-start")

    write(repo, "left.txt", b"left\n")
    commit_all(repo, "left change")
    left_change = oid(repo, "HEAD")
    git(repo, "branch", "left-change")

    git(repo, "switch", "left-start")
    write(repo, "right.txt", b"right\n")
    commit_all(repo, "right change")
    right_change = oid(repo, "HEAD")
    git(repo, "branch", "right-change")

    git(repo, "switch", "left-change")
    git(repo, "merge", "--no-ff", "right-change", "-m", "left merge")
    git(repo, "branch", "left")

    git(repo, "switch", "right-change")
    git(repo, "merge", "--no-ff", left_change, "-m", "right merge")
    git(repo, "branch", "right")

    assert set(git(repo, "merge-base", "--all", "left", "right").splitlines()) == {
        left_change.encode(), right_change.encode(),
    }
    result = resolve_commit_range(str(repo), "left", "right", three_dot=True)

    assert result.entries == ()
    assert result.comparison_base_oid is None
    assert [(warning.code, warning.path) for warning in result.warnings] == [
        ("ambiguous_merge_base", None),
    ]


def test_incomplete_raw_object_ids_are_not_reported_as_exact_snapshots():
    raw = git_snapshot._RawEntry(
        status="M",
        old_path="tracked.py",
        new_path="tracked.py",
        old_mode="100644",
        new_mode="100644",
        old_oid="a" * 39,
        new_oid="b" * 40,
    )
    warnings = []

    assert git_snapshot._exact_staged_entry(raw, warnings) is None
    assert [(warning.code, warning.path) for warning in warnings] == [
        ("missing_object_id", "tracked.py")
    ]


def test_malformed_raw_record_discards_preceding_partial_snapshot():
    """A malformed raw stream must not be surfaced as a complete snapshot."""

    warnings = []
    raw = (
        b":100644 100644 "
        b"1111111111111111111111111111111111111111 "
        b"2222222222222222222222222222222222222222 M\0"
        b"valid.py\0"
        b":100644 incomplete\0"
    )

    assert git_snapshot._parse_raw(raw, warnings) == []
    assert [(warning.code, warning.path) for warning in warnings] == [
        ("malformed_git_output", None)
    ]


def test_complete_git_object_ids_support_sha1_and_sha256_only():
    assert git_snapshot._is_hex_oid("a" * 40)
    assert git_snapshot._is_hex_oid("b" * 64)
    assert not git_snapshot._is_hex_oid("c" * 39)
    assert not git_snapshot._is_hex_oid("d" * 65)
    assert not git_snapshot._is_hex_oid("A" * 40)


def test_file_mode_transitions_preserve_exact_blob_provenance(tmp_path):
    """Executable-bit-only changes retain both endpoint modes and blob IDs."""
    repo = make_repo(tmp_path)
    write(repo, "tool.py", b"#!/usr/bin/env python3\nprint('wild')\n")
    commit_all(repo)
    original_oid = oid(repo, "HEAD:tool.py")

    os.chmod(repo / "tool.py", 0o755)
    unstaged = resolve_unstaged(str(repo), ["tool.py"])

    assert unstaged.warnings == ()
    assert len(unstaged.entries) == 1
    unstaged_entry = unstaged.entries[0]
    assert (unstaged_entry.status, unstaged_entry.old_path, unstaged_entry.new_path) == (
        "M", "tool.py", "tool.py",
    )
    assert (unstaged_entry.old_mode, unstaged_entry.new_mode) == ("100644", "100755")
    assert (unstaged_entry.old_oid, unstaged_entry.new_oid) == (original_oid, original_oid)

    git(repo, "add", "--", "tool.py")
    staged = resolve_staged(str(repo), ["tool.py"])

    assert staged.warnings == ()
    assert len(staged.entries) == 1
    staged_entry = staged.entries[0]
    assert (staged_entry.old_mode, staged_entry.new_mode) == ("100644", "100755")
    assert (staged_entry.old_oid, staged_entry.new_oid) == (original_oid, original_oid)


def test_regular_file_to_symlink_transition_preserves_exact_provenance(tmp_path):
    """Type changes retain endpoint modes and raw symlink blob identities."""
    repo = make_repo(tmp_path)
    path = "current.py"
    previous_content = b"print('previous')\n"
    link_target = "next.py"
    write(repo, path, previous_content)
    commit_all(repo, "regular file")
    git(repo, "branch", "before-type-change")
    previous_oid = oid(repo, "HEAD:" + path)
    link_oid = git(
        repo, "hash-object", "--stdin", input_bytes=link_target.encode("utf-8")
    ).decode("ascii").strip()

    (repo / path).unlink()
    os.symlink(link_target, repo / path)

    unstaged = resolve_unstaged(str(repo), [path])

    assert unstaged.warnings == ()
    assert len(unstaged.entries) == 1
    assert unstaged.entries[0] == git_snapshot.SnapshotEntry(
        status="T",
        old_path=path,
        new_path=path,
        old_mode="100644",
        new_mode="120000",
        old_oid=previous_oid,
        new_oid=link_oid,
    )

    git(repo, "add", "--", path)
    staged = resolve_staged(str(repo), [path])

    assert staged.warnings == ()
    assert staged.entries == unstaged.entries

    commit_all(repo, "symlink file")
    ranged = resolve_commit_range(str(repo), "before-type-change", "HEAD", pathspecs=[path])

    assert ranged.warnings == ()
    assert ranged.entries == staged.entries


def test_commit_range_binary_blobs_preserve_exact_endpoint_provenance(tmp_path):
    """Binary additions, edits, and deletions retain immutable blob IDs."""
    repo = make_repo(tmp_path)
    original = b"\x00\xffbefore\x00payload\n"
    removed = b"\x89PNG\r\n\x1a\nold-image"
    write(repo, "assets/data.bin", original)
    write(repo, "assets/removed.bin", removed)
    commit_all(repo, "binary baseline")
    git(repo, "branch", "before-binary-change")

    updated = b"\x00\xffafter\x00payload\x01\n"
    added = b"\x1f\x8b\x08\x00new-archive"
    write(repo, "assets/data.bin", updated)
    os.unlink(repo / "assets/removed.bin")
    write(repo, "assets/added.bin", added)
    commit_all(repo, "binary changes")

    result = resolve_commit_range(str(repo), "before-binary-change", "HEAD")
    entries = {entry.new_path or entry.old_path: entry for entry in result.entries}

    assert result.warnings == ()
    assert set(entries) == {
        "assets/added.bin", "assets/data.bin", "assets/removed.bin",
    }
    assert (entries["assets/data.bin"].old_oid, entries["assets/data.bin"].new_oid) == (
        git(
            repo, "rev-parse", "before-binary-change:assets/data.bin"
        ).decode("ascii").strip(),
        oid(repo, "HEAD:assets/data.bin"),
    )
    assert (entries["assets/added.bin"].old_oid, entries["assets/added.bin"].new_oid) == (
        None, oid(repo, "HEAD:assets/added.bin"),
    )
    assert (entries["assets/removed.bin"].old_oid, entries["assets/removed.bin"].new_oid) == (
        oid(repo, "before-binary-change:assets/removed.bin"), None,
    )
