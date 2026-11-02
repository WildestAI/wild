"""Resolve exact Git object identities for index and working-tree changes.

This module models the two local snapshot pairs and immutable commit ranges:

* staged: ``HEAD`` -> index
* unstaged: index -> working tree, including ordinary untracked files
* two-dot: the requested base commit -> the requested head commit
* three-dot: the merge base of the requested commits -> the requested head

Paths returned by Git are read using its NUL-delimited raw format, so tabs,
newlines, and other unusual filename bytes are not delimiters.
"""

from __future__ import annotations

import os
import posixpath
import stat
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SnapshotEntry:
    """One changed path pair and its exact pre/post Git identities.

    ``status`` is Git's one-letter raw status (``A``, ``M``, ``D``, ``R``,
    ``C``, or ``T``).  A path, mode, or object ID is ``None`` when that side of
    the change does not exist.  Object IDs are full-length hexadecimal IDs.
    """

    status: str
    old_path: Optional[str]
    new_path: Optional[str]
    old_mode: Optional[str]
    new_mode: Optional[str]
    old_oid: Optional[str]
    new_oid: Optional[str]


@dataclass(frozen=True)
class ResolutionWarning:
    """A structured, non-fatal reason a snapshot entry was not resolved."""

    code: str
    message: str
    path: Optional[str] = None


@dataclass(frozen=True)
class SnapshotResolution:
    """Deterministically ordered entries plus any resolver warnings."""

    entries: Tuple[SnapshotEntry, ...]
    warnings: Tuple[ResolutionWarning, ...]


@dataclass(frozen=True)
class CommitRangeResolution:
    """Exact endpoints and entries for a two-dot or three-dot comparison.

    ``comparison_base_oid`` equals ``base_oid`` for two-dot comparisons and
    the resolved merge-base object ID for three-dot comparisons. Recording
    both prevents display refs from being mistaken for immutable provenance.
    """

    entries: Tuple[SnapshotEntry, ...]
    warnings: Tuple[ResolutionWarning, ...]
    base_ref: str
    head_ref: str
    base_oid: Optional[str]
    head_oid: Optional[str]
    comparison_base_oid: Optional[str]
    three_dot: bool


@dataclass(frozen=True)
class RefComparisonResolution:
    """Entries comparing one resolved commit with the index or working tree.

    Matches ``git diff <ref>`` (working tree, including ordinary untracked
    files like an unstaged snapshot) and ``git diff --cached <ref>`` (index).
    """

    entries: Tuple[SnapshotEntry, ...]
    warnings: Tuple[ResolutionWarning, ...]
    base_ref: str
    base_oid: str
    staged: bool


class GitSnapshotError(RuntimeError):
    """A Git or working-tree operation required for exact snapshots failed."""


class UnknownRevisionError(GitSnapshotError):
    """A single diff operand is not a commit (usually a pathspec without ``--``)."""


@dataclass(frozen=True)
class _RawEntry:
    status: str
    old_path: Optional[str]
    new_path: Optional[str]
    old_mode: Optional[str]
    new_mode: Optional[str]
    old_oid: Optional[str]
    new_oid: Optional[str]


def resolve_staged(
    repository: str, pathspecs: Optional[Sequence[str]] = None
) -> SnapshotResolution:
    """Resolve changes from ``HEAD`` to the index.

    ``pathspecs`` are passed verbatim after Git's ``--`` separator.  In
    particular, a non-empty scope is never replaced with a repository-wide
    query if it has no matches.
    """

    return _resolve(repository, pathspecs, staged=True)


def resolve_unstaged(
    repository: str, pathspecs: Optional[Sequence[str]] = None
) -> SnapshotResolution:
    """Resolve tracked changes from the index to the working tree.

    Ordinary untracked files are added as one-sided working-tree snapshots;
    ignored files remain excluded.  For each post-change regular file, the
    object ID is computed with ``git
    hash-object --path`` so clean filters and attributes match ``git add``
    semantics without modifying the index.
    """

    return _resolve(repository, pathspecs, staged=False)


def resolve_against_commit(
    repository: str,
    ref: str,
    *,
    staged: bool = False,
    pathspecs: Optional[Sequence[str]] = None,
) -> RefComparisonResolution:
    """Resolve changes from one commit to the working tree (or the index).

    An operand that is not a commit is an error rather than an empty diff:
    pathspecs must follow ``--``, so a mistyped ref must never look like a
    snapshot without changes.
    """

    root = repository_root(repository)
    warnings: List[ResolutionWarning] = []
    base_oid = _resolve_commit(root, ref, "invalid_base_ref", warnings)
    if base_oid is None:
        raise UnknownRevisionError(
            "unknown revision {!r}; put pathspecs after '--'".format(ref)
        )
    resolution = _resolve(repository, pathspecs, staged=staged, base_oid=base_oid)
    return RefComparisonResolution(
        resolution.entries, resolution.warnings, ref, base_oid, staged
    )


def resolve_commit_range(
    repository: str,
    base_ref: str,
    head_ref: str,
    *,
    three_dot: bool = False,
    pathspecs: Optional[Sequence[str]] = None,
) -> CommitRangeResolution:
    """Resolve exact tree identities for a two-dot or three-dot comparison.

    Refs are resolved to commits before diffing. Two-dot compares those two
    commits directly; three-dot compares their merge base with the resolved
    head, matching ``git diff base...head`` semantics. Failures are returned
    as structured warnings and never as fabricated changes.
    """

    warnings: List[ResolutionWarning] = []
    root = _repository_root(repository, warnings)
    if root is None:
        return _commit_range_result(base_ref, head_ref, three_dot, warnings=warnings)

    base_oid = _resolve_commit(root, base_ref, "invalid_base_ref", warnings)
    head_oid = _resolve_commit(root, head_ref, "invalid_head_ref", warnings)
    if base_oid is None or head_oid is None:
        return _commit_range_result(
            base_ref, head_ref, three_dot, warnings=warnings,
            base_oid=base_oid, head_oid=head_oid,
        )

    comparison_base_oid = base_oid
    if three_dot:
        output = _run(
            ["git", "merge-base", "--all", base_oid, head_oid], root, warnings,
            "merge_base_failed",
        )
        if output is None:
            return _commit_range_result(
                base_ref, head_ref, three_dot, warnings=warnings,
                base_oid=base_oid, head_oid=head_oid,
            )
        merge_bases = sorted(set(os.fsdecode(output).splitlines()))
        if not merge_bases or any(not _is_hex_oid(oid) for oid in merge_bases):
            warnings.append(ResolutionWarning(
                "malformed_merge_base",
                "Git returned an invalid merge-base object ID",
            ))
            return _commit_range_result(
                base_ref, head_ref, three_dot, warnings=warnings,
                base_oid=base_oid, head_oid=head_oid,
            )
        if len(merge_bases) != 1:
            warnings.append(ResolutionWarning(
                "ambiguous_merge_base",
                "Git returned {} merge bases; a three-dot comparison requires one exact base".format(
                    len(merge_bases)
                ),
            ))
            return _commit_range_result(
                base_ref, head_ref, three_dot, warnings=warnings,
                base_oid=base_oid, head_oid=head_oid,
            )
        comparison_base_oid = merge_bases[0]

    command = [
        "git", "diff", "--raw", "-z", "--no-abbrev", "--no-ext-diff",
        "--find-renames=50%", comparison_base_oid, head_oid,
    ]
    scoped_pathspecs = _root_relative_pathspecs(
        repository, root, pathspecs, warnings
    )
    if scoped_pathspecs is None:
        return _commit_range_result(
            base_ref, head_ref, three_dot, warnings=warnings,
            base_oid=base_oid, head_oid=head_oid,
            comparison_base_oid=comparison_base_oid,
        )
    if scoped_pathspecs:
        command.append("--")
        command.extend(scoped_pathspecs)
    output = _run(command, root, warnings, "git_diff_failed")
    entries: List[SnapshotEntry] = []
    if output is not None:
        for raw in _parse_raw(output, warnings):
            entry = _exact_staged_entry(raw, warnings)
            if entry is not None:
                entries.append(entry)
        entries.sort(key=_entry_sort_key)

    return _commit_range_result(
        base_ref, head_ref, three_dot, entries=entries, warnings=warnings,
        base_oid=base_oid, head_oid=head_oid,
        comparison_base_oid=comparison_base_oid,
    )


def _commit_range_result(
    base_ref: str,
    head_ref: str,
    three_dot: bool,
    *,
    entries: Sequence[SnapshotEntry] = (),
    warnings: Sequence[ResolutionWarning] = (),
    base_oid: Optional[str] = None,
    head_oid: Optional[str] = None,
    comparison_base_oid: Optional[str] = None,
) -> CommitRangeResolution:
    return CommitRangeResolution(
        tuple(entries), tuple(warnings), base_ref, head_ref, base_oid, head_oid,
        comparison_base_oid, three_dot,
    )


def _resolve_commit(
    root: str, ref: str, warning_code: str,
    warnings: List[ResolutionWarning],
) -> Optional[str]:
    output = _run(
        [
            "git",
            "rev-parse",
            "--verify",
            "--end-of-options",
            "{}^{{commit}}".format(ref),
        ],
        root, warnings, warning_code,
    )
    if output is None:
        return None
    oid = os.fsdecode(output).strip()
    if not _is_hex_oid(oid):
        warnings.append(ResolutionWarning(
            warning_code, "Git returned an invalid commit object ID"
        ))
        return None
    return oid


def _is_hex_oid(value: str) -> bool:
    """Return whether *value* is a complete Git SHA-1 or SHA-256 object ID."""

    return len(value) in (40, 64) and all(
        character in "0123456789abcdef" for character in value
    )


def _resolve(
    repository: str, pathspecs: Optional[Sequence[str]], staged: bool,
    base_oid: Optional[str] = None,
) -> SnapshotResolution:
    warnings: List[ResolutionWarning] = []
    root = _repository_root(repository, warnings)
    if root is None:
        return SnapshotResolution((), tuple(warnings))

    command = ["git", "diff"]
    if staged:
        command.append("--cached")
    command.extend(
        ["--raw", "-z", "--no-abbrev", "--no-ext-diff", "--find-renames=50%"]
    )
    if base_oid is not None:
        # A full object ID cannot be mistaken for an option or a path.
        command.append(base_oid)
    scoped_pathspecs = _root_relative_pathspecs(
        repository, root, pathspecs, warnings
    )
    if scoped_pathspecs is None:
        return SnapshotResolution((), tuple(warnings))
    if scoped_pathspecs:
        command.append("--")
        command.extend(scoped_pathspecs)

    output = _run(command, root, warnings, "git_diff_failed")
    if output is None:
        return SnapshotResolution((), tuple(warnings))

    raw_entries = _parse_raw(output, warnings)
    # A conflicted index has no single pre/post pair. ``git diff --raw`` can
    # emit an unmerged ``U`` record and a second record for the same path;
    # hashing that second record against an arbitrary conflict stage would
    # fabricate a snapshot. Omit every record touching an unmerged path until
    # the user resolves it, while retaining an actionable warning.
    unmerged_paths = {
        path
        for raw in raw_entries
        if raw.status == "U"
        for path in (raw.old_path, raw.new_path)
        if path is not None
    }
    for path in sorted(unmerged_paths, key=os.fsencode):
        warnings.append(ResolutionWarning(
            "unmerged_index_entry",
            "Cannot resolve an exact pre/post snapshot while the index is unmerged",
            path,
        ))

    entries: List[SnapshotEntry] = []
    for raw in raw_entries:
        if raw.status == "U" or (
            raw.old_path in unmerged_paths or raw.new_path in unmerged_paths
        ):
            continue
        if staged:
            entry = _exact_staged_entry(raw, warnings)
        else:
            entry = _exact_unstaged_entry(root, raw, warnings)
        if entry is not None:
            entries.append(entry)

    if not staged:
        entries.extend(_resolve_untracked(root, scoped_pathspecs, warnings))

    entries.sort(key=_entry_sort_key)
    return SnapshotResolution(tuple(entries), tuple(warnings))


def _resolve_untracked(
    root: str,
    pathspecs: Sequence[str],
    warnings: List[ResolutionWarning],
) -> List[SnapshotEntry]:
    """Resolve non-ignored untracked paths without modifying the index."""

    command = ["git", "ls-files", "--others", "--exclude-standard", "-z"]
    if pathspecs:
        command.append("--")
        command.extend(pathspecs)
    output = _run(command, root, warnings, "git_untracked_failed")
    if output is None:
        return []

    entries: List[SnapshotEntry] = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        path = _decode_git_path(raw_path, warnings)
        if path is None:
            continue
        full_path = os.path.join(root, path)
        try:
            file_stat = os.lstat(full_path)
        except OSError as error:
            warnings.append(ResolutionWarning("worktree_read_failed", str(error), path))
            continue

        if stat.S_ISLNK(file_stat.st_mode):
            mode = "120000"
        elif stat.S_ISREG(file_stat.st_mode):
            mode = "100755" if file_stat.st_mode & stat.S_IXUSR else "100644"
        else:
            warnings.append(ResolutionWarning(
                "unsupported_worktree_entry",
                "Cannot derive a Git blob ID for untracked filesystem entry",
                path,
            ))
            continue

        new_oid = _working_tree_oid(root, path, mode, warnings)
        if new_oid is not None:
            entries.append(SnapshotEntry(
                status="A",
                old_path=None,
                new_path=path,
                old_mode=None,
                new_mode=mode,
                old_oid=None,
                new_oid=new_oid,
            ))
    return entries


def _repository_root(
    repository: str, warnings: List[ResolutionWarning]
) -> Optional[str]:
    output = _run(
        ["git", "rev-parse", "--show-toplevel"],
        os.fspath(repository),
        warnings,
        "not_a_git_repository",
    )
    if output is None:
        return None
    return os.fsdecode(output.rstrip(b"\n"))


def repository_root(repository: str) -> str:
    """Return the repository root or raise a typed, user-facing error."""

    output = run_git(repository, "rev-parse", "--show-toplevel")
    return os.fsdecode(output.rstrip(b"\n"))


def run_git(
    repository: str, *args: str, input_bytes: Optional[bytes] = None
) -> bytes:
    """Run Git with captured output and raise :class:`GitSnapshotError`."""

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=os.fspath(repository),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except (OSError, ValueError) as error:
        raise GitSnapshotError(str(error)) from error

    if completed.returncode != 0:
        detail = os.fsdecode(completed.stderr).strip()
        message = detail or "Git command exited with status {}".format(
            completed.returncode
        )
        raise GitSnapshotError(message)
    return completed.stdout


def _root_relative_pathspecs(
    repository: str,
    root: str,
    pathspecs: Optional[Sequence[str]],
    warnings: List[ResolutionWarning],
) -> Optional[List[str]]:
    """Translate caller-relative pathspecs for a Git process run at ``root``.

    A scope outside the repository cannot be passed to Git safely. Reject it
    rather than normalising it to ``../...`` and relying on a command failure,
    so callers receive an actionable warning and never risk falling back to a
    broader query. This applies to relative scopes from a subdirectory too.
    """

    if not pathspecs:
        return []
    canonical_root = os.path.realpath(root)
    caller = os.path.realpath(os.fspath(repository))
    prefix = os.path.relpath(caller, canonical_root)
    prefix = "" if prefix == "." else prefix.replace(os.sep, "/")
    scoped: List[str] = []
    for pathspec in pathspecs:
        if os.path.isabs(pathspec):
            absolute_path = os.path.realpath(pathspec)
            try:
                inside_root = (
                    os.path.commonpath([canonical_root, absolute_path])
                    == canonical_root
                )
            except ValueError:
                inside_root = False
            if not inside_root:
                warnings.append(ResolutionWarning(
                    "pathspec_outside_repository",
                    "Absolute pathspec is outside the repository and was not resolved",
                    pathspec,
                ))
                return None
            scoped.append(
                os.path.relpath(absolute_path, canonical_root).replace(os.sep, "/")
            )
        else:
            prefixed = _prefix_pathspec(pathspec, prefix)
            if _pathspec_escapes_repository(prefixed):
                warnings.append(ResolutionWarning(
                    "pathspec_outside_repository",
                    "Relative pathspec escapes the repository and was not resolved",
                    pathspec,
                ))
                return None
            scoped.append(prefixed)
    return scoped


def _prefix_pathspec(pathspec: str, prefix: str) -> str:
    """Prefix a Git pathspec while preserving common pathspec magic."""

    def prefixed(pattern: str) -> str:
        return posixpath.normpath(posixpath.join(prefix, pattern))

    if pathspec.startswith(":/"):
        return pathspec
    if pathspec.startswith(":("):
        end = pathspec.find(")")
        if end != -1:
            magic = pathspec[2:end].split(",")
            if "top" in magic:
                return pathspec
            return pathspec[: end + 1] + prefixed(pathspec[end + 1 :])
    if pathspec.startswith((":!", ":^")):
        return pathspec[:2] + prefixed(pathspec[2:])
    return prefixed(pathspec)


def _pathspec_escapes_repository(pathspec: str) -> bool:
    """Return whether a root-relative pathspec traverses above the repository.

    ``_prefix_pathspec`` preserves Git's long and short pathspec magic. Strip
    only that prefix before checking the path portion so exclusions and glob
    pathspecs cannot turn a caller-relative ``../...`` scope into a Git
    command rooted outside the requested repository.
    """

    if pathspec.startswith(":/"):
        return False
    if pathspec.startswith(":("):
        end = pathspec.find(")")
        if end != -1:
            magic = pathspec[2:end].split(",")
            if "top" in magic:
                return False
            pathspec = pathspec[end + 1 :]
    elif pathspec.startswith((":!", ":^")):
        pathspec = pathspec[2:]

    if os.sep == "\\":
        pathspec = pathspec.replace("\\", "/")
    pathspec = posixpath.normpath(pathspec)
    return pathspec == ".." or pathspec.startswith("../")


def _run(
    command: Sequence[str],
    cwd: str,
    warnings: List[ResolutionWarning],
    code: str,
    input_bytes: Optional[bytes] = None,
    path: Optional[str] = None,
) -> Optional[bytes]:
    try:
        if not command or command[0] != "git":
            raise ValueError("only Git commands are supported")
        return run_git(cwd, *command[1:], input_bytes=input_bytes)
    except (GitSnapshotError, ValueError) as error:
        warnings.append(ResolutionWarning(code, str(error), path))
        return None


def _parse_raw(data: bytes, warnings: List[ResolutionWarning]) -> List[_RawEntry]:
    fields = data.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()

    parsed: List[_RawEntry] = []
    index = 0
    while index < len(fields):
        header = fields[index]
        index += 1
        try:
            metadata = header[1:].split(b" ") if header.startswith(b":") else []
            if len(metadata) != 5:
                raise ValueError("malformed raw-diff metadata")
            old_mode_b, new_mode_b, old_oid_b, new_oid_b, status_b = metadata
            status_text = status_b.decode("ascii")
            status = status_text[:1]
            path_count = 2 if status in ("R", "C") else 1
            if not status or index + path_count > len(fields):
                raise ValueError("malformed raw-diff path fields")
            paths = [
                _decode_git_path(value, warnings)
                for value in fields[index : index + path_count]
            ]
            index += path_count

            # Git permits arbitrary bytes in a pathname, but artifact paths
            # must be valid Unicode strings. Skip only this record while
            # retaining its NUL-delimited boundaries, so a single invalid
            # filename cannot poison the rest of the snapshot.
            if any(path is None for path in paths):
                continue

            if status in ("R", "C"):
                old_path, new_path = paths
            elif status == "A":
                old_path, new_path = None, paths[0]
            elif status == "D":
                old_path, new_path = paths[0], None
            else:
                old_path = new_path = paths[0]

            parsed.append(
                _RawEntry(
                    status=status,
                    old_path=old_path,
                    new_path=new_path,
                    old_mode=_mode(old_mode_b),
                    new_mode=_mode(new_mode_b),
                    old_oid=_oid(old_oid_b),
                    new_oid=_oid(new_oid_b),
                )
            )
        except (UnicodeDecodeError, ValueError) as error:
            warnings.append(
                ResolutionWarning("malformed_git_output", str(error), None)
            )
            # Record boundaries are no longer trustworthy.  A successfully
            # parsed prefix is not an exact snapshot of the requested Git
            # state, so omit it rather than presenting a partial change set
            # as complete.
            return []
    return parsed


def _decode_git_path(
    value: bytes, warnings: List[ResolutionWarning]
) -> Optional[str]:
    """Decode a Git pathname without leaking surrogate escapes into artifacts."""

    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        warnings.append(ResolutionWarning(
            "undecodable_path",
            "Git reported a path that is not valid UTF-8; the entry was skipped",
            value.decode("utf-8", "backslashreplace"),
        ))
        return None


def _mode(value: bytes) -> Optional[str]:
    text = value.decode("ascii")
    return None if not text or set(text) == {"0"} else text


def _oid(value: bytes) -> Optional[str]:
    text = value.decode("ascii")
    return None if not text or set(text) == {"0"} else text


def _exact_staged_entry(
    raw: _RawEntry, warnings: List[ResolutionWarning]
) -> Optional[SnapshotEntry]:
    object_ids = (
        (raw.old_path, raw.old_oid),
        (raw.new_path, raw.new_oid),
    )
    if any(path is not None and (oid is None or not _is_hex_oid(oid))
           for path, oid in object_ids):
        warnings.append(
            ResolutionWarning(
                "missing_object_id",
                "Git did not provide a complete exact staged object ID",
                raw.new_path or raw.old_path,
            )
        )
        return None
    return SnapshotEntry(**raw.__dict__)


def _exact_unstaged_entry(
    root: str, raw: _RawEntry, warnings: List[ResolutionWarning]
) -> Optional[SnapshotEntry]:
    if raw.old_path is not None and (
        raw.old_oid is None or not _is_hex_oid(raw.old_oid)
    ):
        warnings.append(
            ResolutionWarning(
                "missing_object_id",
                "Git did not provide a complete exact index object ID",
                raw.old_path,
            )
        )
        return None

    new_oid = raw.new_oid
    if raw.new_path is not None:
        new_oid = _working_tree_oid(root, raw.new_path, raw.new_mode, warnings)
        if new_oid is None:
            return None

    return SnapshotEntry(
        status=raw.status,
        old_path=raw.old_path,
        new_path=raw.new_path,
        old_mode=raw.old_mode,
        new_mode=raw.new_mode,
        old_oid=raw.old_oid,
        new_oid=new_oid,
    )


def _working_tree_oid(
    root: str,
    path: str,
    mode: Optional[str],
    warnings: List[ResolutionWarning],
) -> Optional[str]:
    """Return the exact object ID for a worktree file, link, or gitlink."""

    if mode == "160000":
        output = _run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            os.path.join(root, path),
            warnings,
            "gitlink_head_failed",
            path=path,
        )
        if output is None:
            return None
        oid = os.fsdecode(output).strip()
        if not _is_hex_oid(oid):
            warnings.append(ResolutionWarning(
                "malformed_gitlink_head",
                "Git returned an invalid gitlink commit object ID",
                path,
            ))
            return None
        return oid

    result = _working_tree_blob(root, path, mode, warnings)
    return result[1] if result is not None else None


def _working_tree_blob(
    root: str,
    path: str,
    mode: Optional[str],
    warnings: List[ResolutionWarning],
) -> Optional[Tuple[bytes, str]]:
    full_path = os.path.join(root, path)
    try:
        before = os.lstat(full_path)
        if mode == "120000" and stat.S_ISLNK(before.st_mode):
            content = os.fsencode(os.readlink(full_path))
        elif mode in ("100644", "100755") and stat.S_ISREG(before.st_mode):
            with open(full_path, "rb") as handle:
                content = handle.read()
                opened = os.fstat(handle.fileno())
            if _stat_identity(opened) != _stat_identity(before):
                raise OSError("working-tree file changed while it was opened")
        else:
            warnings.append(
                ResolutionWarning(
                    "unsupported_worktree_entry",
                    "Cannot derive a Git blob ID for working-tree mode {}".format(mode),
                    path,
                )
            )
            return None
        after = os.lstat(full_path)
        if _stat_identity(before) != _stat_identity(after):
            raise OSError("working-tree file changed while it was hashed")
    except OSError as error:
        warnings.append(ResolutionWarning("worktree_read_failed", str(error), path))
        return None

    command = ["git", "hash-object", "--stdin"]
    if mode != "120000":
        command.append("--path={}".format(path))
    output = _run(
        command,
        root,
        warnings,
        "hash_object_failed",
        input_bytes=content,
        path=path,
    )
    if output is None:
        return None
    oid = os.fsdecode(output).strip()
    if not _is_hex_oid(oid):
        warnings.append(
            ResolutionWarning("malformed_hash_object_output", "Git returned an invalid object ID", path)
        )
        return None
    return content, oid


def read_worktree_blob(
    root: str, path: str, mode: Optional[str], expected_oid: Optional[str] = None
) -> bytes:
    """Read exact regular-file or symlink bytes and verify their Git identity."""

    warnings: List[ResolutionWarning] = []
    result = _working_tree_blob(root, path, mode, warnings)
    if result is None:
        warning = warnings[0] if warnings else ResolutionWarning(
            "worktree_read_failed", "working-tree content could not be read", path
        )
        raise GitSnapshotError("{}: {}".format(warning.code, warning.message))
    content, oid = result
    if expected_oid is not None and oid != expected_oid:
        raise GitSnapshotError(
            "working-tree content no longer matches resolved Git identity"
        )
    return content


def _stat_identity(value: os.stat_result) -> Tuple[int, int, int, int, int]:
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )


def _entry_sort_key(entry: SnapshotEntry) -> Tuple[bytes, bytes, str]:
    return (
        os.fsencode(entry.old_path or ""),
        os.fsencode(entry.new_path or ""),
        entry.status,
    )
