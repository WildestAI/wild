"""Deterministic, local DiffGraph v2 extraction from exact Git snapshots.

This first baseline intentionally supports Python source and stub files only.
Other languages remain in ``files`` and produce scoped ``UNSUPPORTED_LANGUAGE``
warnings; no capability is inferred from a filename beyond that explicit boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from difflib import SequenceMatcher, unified_diff
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from diffgraph import __version__ as package_version
from diffgraph.contract import CURRENT_SCHEMA_VERSION
from diffgraph.git_snapshot import (
    GitSnapshotError,
    ResolutionWarning,
    SnapshotEntry,
    read_worktree_blob,
    repository_root,
    resolve_against_commit,
    resolve_commit_range,
    resolve_staged,
    resolve_unstaged,
    run_git,
)

ANALYZER = "diffgraph-python-tree-sitter"
QUERY_VERSION = "python-structure-v2"
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
_PARSER_STATE = threading.local()

# Resolver failures are part of the public artifact contract. Keep their
# machine-readable identity instead of collapsing an actionable Git failure
# into UNKNOWN, so clients can offer the right remediation without parsing
# the human-oriented detail string.
RESOLUTION_WARNING_CODES = frozenset({
    "not_a_git_repository",
    "git_diff_failed",
    "git_untracked_failed",
    "invalid_base_ref",
    "invalid_head_ref",
    "merge_base_failed",
    "malformed_merge_base",
    "ambiguous_merge_base",
    "malformed_git_output",
    "undecodable_path",
    "missing_object_id",
    "unsupported_worktree_entry",
    "worktree_read_failed",
    "hash_object_failed",
    "malformed_hash_object_output",
    "unmerged_index_entry",
    "pathspec_outside_repository",
    "gitlink_head_failed",
    "malformed_gitlink_head",
})


class StructuralDependencyError(ImportError):
    """A required local structural parser dependency is unavailable."""


@dataclass(frozen=True)
class _Symbol:
    name: str
    qualified_name: str
    kind: str
    parent: Optional[str]
    start_line: int
    end_line: int
    text_hash: str


@dataclass(frozen=True)
class _Import:
    module: str
    line: int
    position: int
    snippet: str
    bindings: Tuple[str, ...]
    imported_names: Tuple[str, ...]
    scope: Optional[str]


@dataclass(frozen=True)
class _ParsedFile:
    """Exact post-change parse state retained for cross-file resolution."""

    path: str
    new_symbols: List[_Symbol]
    new_imports: List[_Import]


@dataclass(frozen=True)
class _Call:
    caller: Optional[str]
    name: str
    line: int
    position: int
    snippet: str
    comprehension_bindings: Tuple[str, ...] = ()
    # "inherits": a base class in ``class subject(name)``. Bases are evaluated
    # in the class's enclosing scope, so ``caller`` is that scope and the
    # resolved edge starts at ``subject`` (the class).
    kind: str = "call"
    subject: Optional[str] = None


def _blob(repository: str, oid: Optional[str]) -> Optional[bytes]:
    if oid is None:
        return None
    return run_git(repository, "cat-file", "blob", oid)


def _worktree_bytes(repository: str, entry: SnapshotEntry) -> Optional[bytes]:
    if entry.new_path is None:
        return None
    return read_worktree_blob(
        repository, entry.new_path, entry.new_mode, expected_oid=entry.new_oid
    )


def _line_counts(
    old: Optional[bytes], new: Optional[bytes]
) -> Tuple[Optional[int], Optional[int]]:
    """Return deterministic added/removed line counts, or nulls for binaries."""
    old_bytes = old or b""
    new_bytes = new or b""
    if b"\0" in old_bytes or b"\0" in new_bytes:
        return None, None

    matcher = SequenceMatcher(
        None,
        old_bytes.splitlines(keepends=True),
        new_bytes.splitlines(keepends=True),
        autojunk=False,
    )
    added = removed = 0
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += old_end - old_start
        if tag in ("replace", "insert"):
            added += new_end - new_start
    return added, removed


MAX_SOURCE_LINES = 2000
MAX_FILE_DIFF_CHARS = 200_000


def split_lines(text: str) -> List[str]:
    """Split on newline only, keeping the newline, as Git and tree-sitter do.

    ``str.splitlines`` also breaks on form feeds, lone ``\\r``, ``\\x85``, and
    other separators, which would misalign declaration spans and hunks.
    """
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _decoded_lines(data: Optional[bytes]) -> Optional[List[str]]:
    """Exact UTF-8 lines, or None when a side is absent or not valid UTF-8."""
    if data is None:
        return None
    try:
        return split_lines(data.decode("utf-8"))
    except UnicodeDecodeError:
        return None


def _file_diff(
    old_path: Optional[str], new_path: Optional[str],
    old: Optional[bytes], new: Optional[bytes],
) -> Optional[Dict]:
    """A capped unified diff of one text file, so consumers can show the
    micro-level change without reading the repository."""
    old_lines, new_lines = _decoded_lines(old), _decoded_lines(new)
    if (old is not None and old_lines is None) or (new is not None and new_lines is None):
        return None
    text = "".join(
        # A last line without a newline gets Git's marker instead of being
        # glued to the next diff line.
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
        for line in unified_diff(
            old_lines or [], new_lines or [],
            fromfile="a/" + old_path if old is not None and old_path else "/dev/null",
            tofile="b/" + new_path if new is not None and new_path else "/dev/null",
            n=3,
        )
    )
    if not text:
        return None
    truncated = len(text) > MAX_FILE_DIFF_CHARS
    if truncated:
        text = text[: text.rfind("\n", 0, MAX_FILE_DIFF_CHARS) + 1]
    return {"unified": text, "truncated": truncated}


def _symbol_source(data: Optional[bytes], symbol: Optional[_Symbol]) -> Optional[Dict]:
    """The exact declaration text of one side of a changed symbol."""
    lines = _decoded_lines(data)
    if lines is None or symbol is None:
        return None
    span = lines[symbol.start_line - 1:symbol.end_line]
    return {
        "line_start": symbol.start_line,
        "text": "".join(span[:MAX_SOURCE_LINES]),
        "truncated": len(span) > MAX_SOURCE_LINES,
    }


def _binary_sides(old: Optional[bytes], new: Optional[bytes]) -> Tuple[str, ...]:
    """Return snapshot sides matching the deterministic NUL-byte heuristic."""
    sides = []
    if old is not None and b"\0" in old:
        sides.append("pre-change")
    if new is not None and b"\0" in new:
        sides.append("post-change")
    return tuple(sides)


def _non_regular_modes(entry: SnapshotEntry) -> Tuple[str, ...]:
    """Return Git modes whose contents must not be treated as source text.

    Symlink targets and gitlinks are represented by object-like values in a
    raw Git diff, but neither is a source-file snapshot. Parsing a symlink
    target such as ``replacement.py`` as Python can otherwise yield a
    misleading successful analysis with empty topology.
    """
    return tuple(sorted({
        mode
        for mode in (entry.old_mode, entry.new_mode)
        if mode is not None and mode not in ("100644", "100755")
    }))


def _parser():
    parser = getattr(_PARSER_STATE, "python_parser", None)
    if parser is not None:
        return parser
    try:
        import tree_sitter
        import tree_sitter_language_pack
    except ImportError as error:
        raise StructuralDependencyError(
            "Python structural analysis requires tree-sitter and "
            "tree-sitter-language-pack"
        ) from error

    # Construct the official parser directly so byte offsets and byte input
    # remain exact across language-pack releases.
    parser = tree_sitter.Parser(tree_sitter_language_pack.get_language("python"))
    _PARSER_STATE.python_parser = parser
    return parser


def _node_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8")


def _name_child(node):
    child = node.child_by_field_name("name")
    if child is not None:
        return child
    return next((item for item in node.children if item.type == "identifier"), None)


def _first_identifier(node):
    """Return the first identifier below *node*, preserving source order."""
    if node is None:
        return None
    if node.type == "identifier":
        return node
    for child in node.children:
        identifier = _first_identifier(child)
        if identifier is not None:
            return identifier
    return None


def _is_type_alias_annotation(source: bytes, node) -> bool:
    """Recognize the conservative ``TypeAlias`` assignment spelling.

    PEP 613 aliases use an annotated assignment rather than a dedicated
    Tree-sitter node. Treat only a literal ``TypeAlias`` annotation (including
    the explicit ``typing.TypeAlias`` spelling) as a type alias; ordinary type
    annotations remain values and must not become invented topology.
    """
    annotation = node.child_by_field_name("type")
    if annotation is None:
        return False
    text = _node_text(source, annotation)
    return text in ("TypeAlias", "typing.TypeAlias")


def _syntax_error_detail(root) -> str:
    """Return the first parser-reported error location in source order.

    Tree-sitter's root error flag tells us a snapshot is unsafe to turn into
    topology, but users still need a deterministic, file-scoped clue for the
    repair.  Traverse children in source order and report the first explicit
    ERROR or missing node without attempting error recovery or inventing a
    structural claim.
    """
    pending = [root]
    while pending:
        node = pending.pop()
        if node.is_error or node.is_missing:
            line, column = node.start_point
            return "Tree-sitter reported a syntax error at line {}, column {}".format(
                line + 1, column + 1
            )
        pending.extend(reversed(node.children))
    return "Tree-sitter reported a syntax error"


def _parse_python(
    content: bytes,
) -> Tuple[
    List[_Symbol],
    List[_Import],
    List[_Call],
    Dict[Optional[str], set],
    List[Tuple[str, int]],
]:
    # Do not silently replace undecodable source: the warning must identify the
    # exact side that could not be structurally analyzed.
    content.decode("utf-8")
    tree = _parser().parse(content)
    if tree.root_node.has_error:
        raise ValueError(_syntax_error_detail(tree.root_node))

    symbols: List[_Symbol] = []
    imports: List[_Import] = []
    calls: List[_Call] = []
    bindings: Dict[Optional[str], set] = {}
    rebindings: List[Tuple[Optional[str], str, int]] = []
    symbol_occurrences: Dict[str, int] = {}

    def identifiers(node) -> set:
        found = set()
        if node.type == "identifier":
            found.add(_node_text(content, node))
        for child in node.children:
            found.update(identifiers(child))
        return found

    def as_target_identifiers(node) -> set:
        """Return names assigned by an ``as`` target, excluding object references."""
        if node.type in ("attribute", "subscript"):
            return set()
        if node.type == "identifier":
            return {_node_text(content, node)}
        found = set()
        for child in node.children:
            found.update(as_target_identifiers(child))
        return found

    def assignment_target_identifiers(node) -> set:
        """Return names assigned by an assignment target, not names it reads."""
        if node.type in ("attribute", "subscript"):
            return set()
        if node.type == "identifier":
            return {_node_text(content, node)}
        found = set()
        for child in node.children:
            found.update(assignment_target_identifiers(child))
        return found

    def case_pattern_identifiers(node) -> set:
        """Return capture names assigned by a Python structural match pattern.

        Bare names in ``case`` patterns capture, while qualified names and the
        leading name in a class pattern are value lookups. Record only the
        former so imported call targets cannot leak into a capture's case body.
        """
        found = set()
        if node.type == "case_pattern":
            dotted_name = next(
                (child for child in node.children if child.type == "dotted_name"),
                None,
            )
            if dotted_name is not None:
                name = _node_text(content, dotted_name)
                if "." not in name and name not in ("_", "False", "None", "True"):
                    found.add(name)
        elif node.type == "keyword_pattern":
            # The leading identifier is the class attribute label; only the
            # value pattern to the right of ``=`` can capture a local name.
            dotted_name = next(
                (child for child in node.children if child.type == "dotted_name"),
                None,
            )
            if dotted_name is not None:
                name = _node_text(content, dotted_name)
                if "." not in name and name not in ("_", "False", "None", "True"):
                    found.add(name)
        elif node.type == "as_pattern":
            target = next(
                (child for child in node.children if child.type == "as_pattern_target"),
                None,
            )
            if target is not None:
                found.update(as_target_identifiers(target))
        elif node.type == "splat_pattern":
            target = next(
                (child for child in node.children if child.type == "identifier"),
                None,
            )
            if target is not None:
                found.add(_node_text(content, target))
        for child in node.children:
            found.update(case_pattern_identifiers(child))
        return found

    def enclosing_comprehension_bindings(node) -> Tuple[str, ...]:
        """Return comprehension targets bound before ``node`` executes.

        A comprehension target is not visible in its own iterable or in an
        earlier clause's iterable. Tree-sitter stores each ``for_in_clause``
        beside the expression it governs, so derive visibility from the call's
        position within the direct clauses rather than treating every target as
        an enclosing-function binding.
        """
        found = set()
        ancestor = node.parent
        while ancestor is not None:
            if ancestor.type in (
                "list_comprehension",
                "set_comprehension",
                "dictionary_comprehension",
                "generator_expression",
            ):
                clauses = [
                    child
                    for child in ancestor.children
                    if child.type in ("for_in_clause", "if_clause")
                ]
                visible_clauses = clauses
                for index, clause in enumerate(clauses):
                    if not (
                        clause.start_byte <= node.start_byte
                        and node.end_byte <= clause.end_byte
                    ):
                        continue
                    iterable = clause.child_by_field_name("right")
                    if (
                        iterable is not None
                        and iterable.start_byte <= node.start_byte
                        and node.end_byte <= iterable.end_byte
                    ):
                        visible_clauses = clauses[:index]
                    else:
                        visible_clauses = clauses[: index + 1]
                    break
                for clause in visible_clauses:
                    if clause.type != "for_in_clause":
                        continue
                    left = clause.child_by_field_name("left")
                    if left is not None:
                        found.update(identifiers(left))
            ancestor = ancestor.parent
        return tuple(sorted(found))

    def is_inside_lambda(node) -> bool:
        """Return whether node belongs to a lambda's lexical scope."""
        ancestor = node.parent
        while ancestor is not None:
            if ancestor.type == "lambda":
                return True
            ancestor = ancestor.parent
        return False

    def is_declaration_header(node) -> bool:
        """Return whether node is evaluated before its declaration binds."""
        ancestor = node.parent
        while ancestor is not None:
            if ancestor.type not in ("class_definition", "function_definition"):
                ancestor = ancestor.parent
                continue
            body = ancestor.child_by_field_name("body")
            return body is None or not (
                body.start_byte <= node.start_byte and node.end_byte <= body.end_byte
            )
        return False

    def record_rebindings(
        scope: Optional[str], names: set, position: int
    ) -> None:
        """Record lexical rebindings after the expression that creates them."""
        rebindings.extend((scope, name, position) for name in names)

    def visit(node, parents: Tuple[Tuple[str, str], ...] = ()) -> None:
        next_parents = parents
        if node.type in ("class_definition", "function_definition"):
            name_node = _name_child(node)
            if name_node is not None:
                name = _node_text(content, name_node)
                parent = parents[-1][0] if parents else None
                is_method = bool(parents and parents[-1][1] == "class")
                kind = (
                    "class"
                    if node.type == "class_definition"
                    else "method"
                    if is_method
                    else "function"
                )
                base_qname = "{}.{}".format(parent, name) if parent else name
                occurrence = symbol_occurrences.get(base_qname, 0)
                symbol_occurrences[base_qname] = occurrence + 1
                qname = (
                    base_qname
                    if occurrence == 0
                    else "{}#{}".format(base_qname, occurrence)
                )
                # Tree-sitter wraps decorated declarations in a
                # ``decorated_definition`` node. The decorator is part of the
                # declaration's exact source-level behavior, so include that
                # wrapper in the symbol span and hash. Otherwise changing
                # ``@decorator`` could incorrectly leave the declaration
                # unchanged in a deterministic DiffGraph artifact.
                declaration = (
                    node.parent
                    if node.parent is not None
                    and node.parent.type == "decorated_definition"
                    else node
                )
                body = content[declaration.start_byte:declaration.end_byte]
                symbols.append(
                    _Symbol(
                        name,
                        qname,
                        kind,
                        parent,
                        declaration.start_point[0] + 1,
                        declaration.end_point[0] + 1,
                        hashlib.sha256(body).hexdigest(),
                    )
                )
                next_parents = (*parents, (qname, kind))
                superclasses = (
                    node.child_by_field_name("superclasses")
                    if node.type == "class_definition" else None
                )
                if superclasses is not None:
                    header = content[node.start_byte:superclasses.end_byte].decode("utf-8")
                    for base in superclasses.named_children:
                        # Bare names only: ``pkg.Base`` is attribute dispatch,
                        # and ``metaclass=...`` is a keyword argument.
                        if base.type == "identifier":
                            calls.append(_Call(
                                parent, _node_text(content, base), node.start_point[0] + 1,
                                node.start_byte, header, (), "inherits", qname,
                            ))
                # A declaration overwrites an imported binding in its enclosing
                # scope, but only after its declaration header is evaluated.
                body = node.child_by_field_name("body")
                binding_position = (
                    body.start_byte
                    if node.type == "function_definition" and body is not None
                    else node.end_byte
                )
                record_rebindings(parent, {name}, binding_position)
        elif node.type in ("import_statement", "import_from_statement"):
            snippet = _node_text(content, node)
            if node.type == "import_statement":
                names = [c for c in node.children if c.type in ("dotted_name", "aliased_import")]
                for item in names:
                    if item.type == "aliased_import":
                        name_node = item.child_by_field_name("name")
                        if name_node is None:
                            raise ValueError("aliased import has no name")
                        raw = _node_text(content, name_node)
                    else:
                        raw = _node_text(content, item)
                    if item.type == "aliased_import":
                        binding = _node_text(content, item.children[-1])
                    else:
                        # ``import package.submodule`` binds ``package``.
                        binding = raw.split(".", 1)[0]
                    imports.append(_Import(
                        raw, node.start_point[0] + 1, node.end_byte, snippet,
                        (binding,), (raw,), parents[-1][0] if parents else None,
                    ))
            else:
                # Tree-sitter exposes absolute modules through ``module_name``
                # but represents ``from .pkg import value`` as a
                # ``relative_import`` child. Keep the leading dots instead of
                # dropping the import: they are deterministic source evidence
                # and distinguish a package-local dependency from an unrelated
                # absolute module with the same name.
                module_node = (
                    node.child_by_field_name("module_name")
                    or next(
                        (child for child in node.children if child.type == "relative_import"),
                        None,
                    )
                )
                if module_node is not None:
                    bindings_for_import = []
                    imported_names = []
                    after_import = False
                    for child in node.children:
                        if child.type == "import":
                            after_import = True
                            continue
                        if not after_import or child.type not in (
                            "dotted_name", "aliased_import"
                        ):
                            continue
                        if child.type == "aliased_import":
                            name_node = child.child_by_field_name("name")
                            if name_node is None:
                                raise ValueError("aliased import has no name")
                            imported_names.append(_node_text(content, name_node))
                            bindings_for_import.append(_node_text(content, child.children[-1]))
                        else:
                            name = _node_text(content, child)
                            imported_names.append(name)
                            bindings_for_import.append(name.split(".", 1)[0])
                    imports.append(_Import(
                        _node_text(content, module_node), node.start_point[0] + 1,
                        node.end_byte, snippet, tuple(bindings_for_import),
                        tuple(imported_names), parents[-1][0] if parents else None,
                    ))
        elif not parents and node.type in (
            "assignment", "annotated_assignment", "type_alias_statement"
        ):
            # The schema already has ``constant`` and ``type_alias`` symbol
            # kinds. Surface only module-level declarations whose syntax is
            # unambiguous: ALL_CAPS value bindings and PEP 613/695 aliases.
            # Local assignments remain bindings used for call-resolution
            # safety, not promoted to structural symbols.
            if node.type == "type_alias_statement":
                name_node = _first_identifier(
                    node.children[1] if len(node.children) > 1 else None
                )
                kind = "type_alias"
            else:
                name_node = node.child_by_field_name("left")
                kind = (
                    "type_alias"
                    if _is_type_alias_annotation(content, node)
                    else "constant"
                )
                if (
                    kind == "constant"
                    and (name_node is None or name_node.type != "identifier"
                         or not _node_text(content, name_node).isupper())
                ):
                    name_node = None
            if name_node is not None and name_node.type == "identifier":
                name = _node_text(content, name_node)
                occurrence = symbol_occurrences.get(name, 0)
                symbol_occurrences[name] = occurrence + 1
                qname = name if occurrence == 0 else "{}#{}".format(name, occurrence)
                body = content[node.start_byte:node.end_byte]
                symbols.append(
                    _Symbol(
                        name,
                        qname,
                        kind,
                        None,
                        node.start_point[0] + 1,
                        node.end_point[0] + 1,
                        hashlib.sha256(body).hexdigest(),
                    )
                )
        elif node.type == "call":
            function = node.child_by_field_name("function")
            if function is not None and function.type == "identifier":
                caller = parents[-1][0] if parents else None
                ancestor = node.parent
                while ancestor is not None:
                    if ancestor.type in ("class_definition", "function_definition"):
                        body = ancestor.child_by_field_name("body")
                        if body is not None and not (
                            body.start_byte <= node.start_byte
                            and node.end_byte <= body.end_byte
                        ):
                            caller = parents[-2][0] if len(parents) > 1 else None
                        break
                    ancestor = ancestor.parent
                calls.append(
                    _Call(
                        caller,
                        _node_text(content, function),
                        node.start_point[0] + 1,
                        node.start_byte,
                        _node_text(content, node),
                        enclosing_comprehension_bindings(node),
                    )
                )

        scope = parents[-1][0] if parents else None
        if node.type == "parameters" or node.type in (
            "import_statement", "import_from_statement"
        ):
            bindings.setdefault(scope, set()).update(identifiers(node))
        elif node.type == "type_alias_statement":
            name_node = _first_identifier(
                node.children[1] if len(node.children) > 1 else None
            )
            if name_node is not None and name_node.type == "identifier":
                name = _node_text(content, name_node)
                bindings.setdefault(scope, set()).add(name)
                record_rebindings(scope, {name}, node.start_byte)
        elif node.type == "case_clause":
            # Capture patterns bind their names before the case body executes.
            # Keep them lexical and line-aware just like assignment bindings.
            bound_names = case_pattern_identifiers(node)
            bindings.setdefault(scope, set()).update(bound_names)
            record_rebindings(scope, bound_names, node.start_byte)
        elif node.type in (
            "assignment", "annotated_assignment", "for_statement", "named_expression"
        ):
            # Assignment expressions (``name := value``) bind their target in
            # the enclosing scope before later expressions execute. Treat them
            # like ordinary assignments so an imported name cannot produce a
            # false import-grounded call edge after it has been rebound.
            left = node.child_by_field_name(
                "name" if node.type == "named_expression" else "left"
            )
            if left is not None:
                bound_names = assignment_target_identifiers(left)
                binding_scope = scope
                if node.type == "named_expression":
                    # Lambdas have their own lexical scope, but are not
                    # structural symbols. Do not let their local targets leak
                    # into the enclosing function or module.
                    if is_inside_lambda(node):
                        bound_names = set()
                    elif is_declaration_header(node):
                        binding_scope = parents[-2][0] if len(parents) > 1 else None
                bindings.setdefault(binding_scope, set()).update(bound_names)
                binding_position = node.end_byte
                if node.type == "for_statement":
                    iterable = node.child_by_field_name("right")
                    if iterable is not None:
                        binding_position = iterable.end_byte
                annotation_only = node.type in (
                    "assignment", "annotated_assignment"
                ) and not any(child.type == "=" for child in node.children)
                if not annotation_only:
                    record_rebindings(binding_scope, bound_names, binding_position)
        elif (
            node.type == "as_pattern"
            and node.parent is not None
            and node.parent.type in ("with_item", "except_clause")
        ):
            # ``with resource() as name`` and ``except Error as name`` bind
            # their target in the enclosing lexical scope before the body is
            # evaluated. Treat it like other local bindings so an imported
            # function of the same name cannot produce a false call edge.
            target = next(
                (child for child in node.children if child.type == "as_pattern_target"),
                None,
            )
            if target is not None:
                bound_names = as_target_identifiers(target)
                bindings.setdefault(scope, set()).update(bound_names)
                record_rebindings(scope, bound_names, target.start_byte)
        for child in node.children:
            visit(child, next_parents)

    visit(tree.root_node)
    return (
        sorted(symbols, key=lambda s: (s.qualified_name, s.start_line)),
        sorted(imports, key=lambda item: (item.line, item.module, item.snippet)),
        sorted(calls, key=lambda item: (item.line, item.caller or "", item.name, item.snippet)),
        bindings,
        rebindings,
    )


def _change_kind(status: str, old_oid: Optional[str], new_oid: Optional[str]) -> str:
    if status == "A":
        return "added"
    if status == "D":
        return "deleted"
    if status == "R":
        return "renamed" if old_oid == new_oid else "renamed_modified"
    return "modified"


def _provenance(entry: SnapshotEntry, old: Optional[bytes], new: Optional[bytes]) -> str:
    values = {
        "status": entry.status, "old_path": entry.old_path, "new_path": entry.new_path,
        "old_mode": entry.old_mode, "new_mode": entry.new_mode,
        "old_oid": entry.old_oid, "new_oid": entry.new_oid,
        "old_sha256": hashlib.sha256(old).hexdigest() if old is not None else None,
        "new_sha256": hashlib.sha256(new).hexdigest() if new is not None else None,
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=1)
def _parser_version() -> str:
    try:
        return version("tree-sitter-language-pack")
    except PackageNotFoundError:
        return "unknown"


def _parser_provenance(oid: Optional[str]) -> str:
    return "analyzer={};parser=tree-sitter-language-pack@{};query={};blob={}".format(
        ANALYZER, _parser_version(), QUERY_VERSION, oid or "absent"
    )


def _warning(code: str, path: Optional[str], detail: str) -> Dict[str, str]:
    result = {"code": code, "detail": detail}
    if path is not None:
        result["file"] = path
    return result


def _resolution_warning(item: ResolutionWarning) -> Dict[str, str]:
    code = item.code if item.code in RESOLUTION_WARNING_CODES else "UNKNOWN"
    return _warning(code, item.path, "{}: {}".format(item.code, item.message))


def _parse_failure_warning(
    side: str, path: str, oid: Optional[str], error: Exception,
) -> Dict[str, str]:
    """Preserve the exact immutable snapshot that Tree-sitter could not parse."""
    return _warning(
        "PARSE_FAILURE",
        path,
        "{}-change blob {}: {}: {}".format(
            side, oid or "absent", type(error).__name__, error,
        ),
    )


def _unsupported_language_warning(entry: SnapshotEntry, path: str) -> Dict[str, str]:
    """Record the immutable snapshots outside the deterministic baseline."""
    return _warning(
        "UNSUPPORTED_LANGUAGE",
        path,
        "Deterministic extraction currently supports Python (.py and .pyi) only; "
        "pre-change blob {}; post-change blob {}.".format(
            entry.old_oid or "absent", entry.new_oid or "absent",
        ),
    )


def _symbol_id(path: str, qualified_name: str) -> str:
    return "sym::{}::{}".format(path, qualified_name)


def _evidence(path: str, symbol: _Symbol, oid: Optional[str]) -> List[Dict]:
    return [{
        "kind": "ast_parse", "file": path, "line_start": symbol.start_line,
        "line_end": symbol.end_line, "detail": _parser_provenance(oid),
    }]


def _keyed_imports(items: List[_Import]) -> Dict[Tuple[str, int], _Import]:
    occurrences: Dict[str, int] = {}
    result: Dict[Tuple[str, int], _Import] = {}
    for import_item in items:
        occurrence = occurrences.get(import_item.module, 0)
        occurrences[import_item.module] = occurrence + 1
        result[(import_item.module, occurrence)] = import_item
    return result


def _module_name_for_path(path: str) -> str:
    """Return the conventional dotted module spelling for a Python path."""
    candidate = Path(path)
    parts = list(candidate.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_map(parsed_files: Sequence[_ParsedFile]) -> Dict[str, Optional[str]]:
    """Map changed Python modules to one path, or ``None`` when ambiguous."""
    paths: Dict[str, List[str]] = {}
    for parsed in parsed_files:
        module = _module_name_for_path(parsed.path)
        if module:
            paths.setdefault(module, []).append(parsed.path)
    return {
        module: candidates[0] if len(candidates) == 1 else None
        for module, candidates in paths.items()
    }


def _relative_module_name(source_path: str, module: str) -> Optional[str]:
    """Resolve a relative import spelling against its source package path."""
    dots = len(module) - len(module.lstrip("."))
    if not dots:
        return module
    source_parts = list(Path(source_path).with_suffix("").parts)
    package = source_parts if source_parts[-1] == "__init__" else source_parts[:-1]
    if dots - 1 > len(package):
        return None
    prefix = package[: len(package) - (dots - 1)]
    suffix = module[dots:].split(".") if module[dots:] else []
    return ".".join((*prefix, *suffix))


def _import_module_path(
    modules: Dict[str, Optional[str]], module: str, item: _Import
) -> Optional[str]:
    """Resolve an import's module, including one changed namespace child.

    ``from package import child`` is a valid import of ``package.child`` even
    when no ``package/__init__.py`` appears in the selected changeset.  A
    single child path is safe to use for the statement-level import edge;
    multiple children remain unresolved rather than being collapsed together.
    """
    direct_path = modules.get(module)
    if direct_path is not None:
        return direct_path
    child_paths = {
        modules.get("{}.{}".format(module, imported_name))
        for imported_name in item.imported_names
        if modules.get("{}.{}".format(module, imported_name)) is not None
    }
    return next(iter(child_paths)) if len(child_paths) == 1 else None


def _resolved_import_paths(
    parsed_files: Sequence[_ParsedFile],
) -> Dict[Tuple[str, Tuple[str, int]], str]:
    """Resolve import modules to unique changed Python paths."""
    modules = _module_map(parsed_files)
    result: Dict[Tuple[str, Tuple[str, int]], str] = {}
    for parsed in parsed_files:
        for key, item in _keyed_imports(parsed.new_imports).items():
            module = _relative_module_name(parsed.path, item.module)
            target_path = _import_module_path(modules, module, item) if module else None
            if target_path is not None:
                result[(parsed.path, key)] = target_path
    return result


def _resolved_import_targets(
    parsed_files: Sequence[_ParsedFile],
    kinds: Tuple[str, ...] = ("function", "class"),
) -> Dict[Tuple[str, Tuple[str, int]], Dict[str, str]]:
    """Resolve in-changeset ``from`` bindings without guessing package state.

    The map deliberately contains only a unique changed module and a unique
    top-level declaration of one of ``kinds`` (when one is named).  The caller
    can otherwise retain its local import stub, preserving the existing
    external-import behavior.
    """
    modules = _module_map(parsed_files)
    import_paths = _resolved_import_paths(parsed_files)
    symbols_by_path = {
        parsed.path: {symbol.qualified_name: symbol for symbol in parsed.new_symbols}
        for parsed in parsed_files
    }
    resolved: Dict[Tuple[str, Tuple[str, int]], Dict[str, str]] = {}
    for parsed in parsed_files:
        for key, item in _keyed_imports(parsed.new_imports).items():
            module = _relative_module_name(parsed.path, item.module)
            module_path = import_paths.get((parsed.path, key))
            if not module or module_path is None:
                continue
            bindings: Dict[str, str] = {}
            declarations = symbols_by_path[module_path]
            for binding, imported_name in zip(item.bindings, item.imported_names):
                declaration = declarations.get(imported_name)
                has_duplicate = any(
                    qualified_name.startswith(imported_name + "#")
                    for qualified_name in declarations
                )
                if (
                    declaration is not None
                    and not has_duplicate
                    and declaration.kind in kinds
                ):
                    bindings[binding] = _symbol_id(module_path, imported_name)
                    continue
                # Only unique callable declarations can be targets of a
                # ``calls`` edge. Non-declarations, module objects, and
                # ambiguous declarations retain their local import stubs.
            if bindings:
                resolved[(parsed.path, key)] = bindings
    return resolved


def _resolve_call_target(
    call: _Call,
    symbols: Dict[str, _Symbol],
    bindings: Dict[Optional[str], set],
    imported_targets: Dict[Optional[str], Dict[str, List[Tuple[int, Optional[str]]]]],
) -> Optional[str]:
    """Resolve only syntax-grounded, same-file Python calls.

    Bare identifiers shadowed by a parameter, assignment, loop target, or import
    are deliberately left unresolved. Attribute calls and ambiguous duplicate
    definitions are likewise omitted rather than guessed.
    """
    if call.name in call.comprehension_bindings:
        return None
    candidates: List[str] = []
    current_name = call.caller
    while current_name is not None:
        current = symbols.get(current_name)
        if current is None:
            break
        # Function and method scopes participate in lexical lookup. A class
        # body also needs its own imports, but a method must never resolve
        # through its enclosing class namespace.
        is_initial_class_body = (
            current_name == call.caller and current.kind == "class"
        )
        if current.kind in ("function", "method") or is_initial_class_body:
            if call.name in bindings.get(current_name, set()):
                history = imported_targets.get(current_name, {}).get(call.name, [])
                visible = [target for position, target in history if position <= call.position]
                if visible:
                    return visible[-1]
                return None
            if current.kind in ("function", "method"):
                candidates.append("{}.{}".format(current_name, call.name))
        current_name = current.parent

    if call.name in bindings.get(None, set()):
        # An explicit import is a deterministic external target. Other global
        # bindings (for example an assignment) remain intentionally unresolved.
        # Select the binding visible at this call site rather than applying a
        # later top-level rebind retroactively.
        history = imported_targets.get(None, {}).get(call.name, [])
        visible = [target for position, target in history if position <= call.position]
        if visible and visible[-1] is not None:
            return visible[-1]
        # A later import must not hide a declaration that was already visible
        # at this call site. Likewise, a declaration that replaces an import
        # can still resolve through the ordinary local-symbol path. Assignments
        # and loop targets have no matching local symbol and remain unresolved.
    candidates.append(call.name)

    for candidate in candidates:
        matches = [
            qname
            for qname, symbol in symbols.items()
            if symbol.kind in ("function", "class")
            and (qname == candidate or qname.startswith(candidate + "#"))
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
    return None


def _imported_call_targets(
    imports: Dict[Tuple[str, int], _Import],
    rebindings: List[Tuple[Optional[str], str, int]],
    resolved_targets: Optional[Dict[Tuple[str, int], Dict[str, str]]] = None,
) -> Dict[Optional[str], Dict[str, List[Tuple[int, Optional[str]]]]]:
    """Map import bindings to conservative, lexical and line-aware histories.

    Imports inside a function bind only that function's local scope. Keeping
    that scope alongside the source line permits a direct call to an imported
    name without leaking the binding into sibling functions or the module.
    """

    targets: Dict[Optional[str], Dict[str, List[Tuple[int, Optional[str]]]]] = {}
    imported_bindings: Dict[Optional[str], set] = {}
    for (module, occurrence), item in imports.items():
        suffix = "" if occurrence == 0 else "#{}".format(occurrence)
        target = "import::{}{}".format(module, suffix)
        resolved_bindings = (resolved_targets or {}).get((module, occurrence), {})
        scope_targets = targets.setdefault(item.scope, {})
        scope_bindings = imported_bindings.setdefault(item.scope, set())
        for binding in item.bindings:
            # A later import of the same local name is intentionally
            # unresolved, but calls before it retain the earlier binding.
            scope_targets.setdefault(binding, []).append((
                item.position,
                None if binding in scope_bindings else resolved_bindings.get(binding, target),
            ))
            scope_bindings.add(binding)
    for scope, binding, position in rebindings:
        # A declaration, assignment, or loop target replaces the imported
        # binding after its source expression has been evaluated.
        targets.setdefault(scope, {}).setdefault(binding, []).append((position, None))
    for scope_targets in targets.values():
        for history in scope_targets.values():
            history.sort(key=lambda item: item[0])
    return targets


MAX_CONTEXT_FILES = 300
CONTEXT_TARGET_KINDS = ("function", "class", "constant", "type_alias")


def _grep_candidate_paths(
    root: str, names: Sequence[str], revision: Optional[str]
) -> Optional[List[str]]:
    """Python paths in the analyzed snapshot that mention any of ``names``.

    ``revision`` is the head commit for ranges, or None for the index (staged
    and unstaged snapshots: an unchanged tracked file's index content is its
    working-tree content). Returns None when Git cannot search.
    """
    args = ["git", "grep", "-l", "-z", "-I", "-w", "-F"]
    for name in sorted(set(names)):
        args += ["-e", name]
    args += [revision] if revision else ["--cached"]
    args += ["--", "*.py", "*.pyi"]
    try:
        completed = subprocess.run(args, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except (OSError, ValueError):
        return None
    if completed.returncode == 1:  # git grep: no match
        return []
    if completed.returncode != 0:
        return None
    paths = []
    prefix = (revision + ":").encode("ascii") if revision else b""
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        if prefix and raw.startswith(prefix):
            raw = raw[len(prefix):]
        try:
            paths.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            continue
    return sorted(set(paths))


def _read_snapshot_blobs(
    root: str, paths: Sequence[str], revision: Optional[str]
) -> Dict[str, Tuple[str, bytes]]:
    """Read ``{path: (blob oid, bytes)}`` at the snapshot with one Git process."""
    if not paths:
        return {}
    spec = (revision or "") + ":"
    request = "".join(spec + path + "\n" for path in paths).encode("utf-8")
    try:
        completed = subprocess.run(
            ["git", "cat-file", "--batch"], cwd=root, input=request,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except (OSError, ValueError):
        return {}
    if completed.returncode != 0:
        return {}
    data, offset, blobs = completed.stdout, 0, {}
    for path in paths:
        end = data.find(b"\n", offset)
        if end < 0:
            break
        header = data[offset:end].split()
        offset = end + 1
        if len(header) != 3 or header[1] != b"blob":
            continue  # "<spec> missing" or a non-blob entry
        size = int(header[2])
        blobs[path] = (header[0].decode("ascii"), data[offset:offset + size])
        offset += size + 1
    return blobs


def _merge_bindings(
    primary: Dict[Tuple[str, int], Dict[str, str]],
    fallback: Dict[Tuple[str, int], Dict[str, str]],
    allowed: set,
) -> Dict[Tuple[str, int], Dict[str, str]]:
    """Add fallback bindings (deleted declarations) not resolved by primary."""
    merged = {key: dict(value) for key, value in primary.items()}
    for key, bindings in fallback.items():
        for binding, target in bindings.items():
            if target in allowed and binding not in merged.get(key, {}):
                merged.setdefault(key, {})[binding] = target
    return merged


def _context_callers(
    root: str,
    revision: Optional[str],
    changed_paths: set,
    post_parses: List[_ParsedFile],
    pre_parses: List[_ParsedFile],
    pre_id_paths: Dict[str, str],
    symbols: List[Dict],
    warnings: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Find callers and importers of changed declarations in unchanged files.

    A changed signature is usually broken by a call site the diff did not
    touch. Unchanged Python files mentioning a changed (or deleted) top-level
    name are parsed at the same snapshot and resolved with the same
    conservative rules as changed files; only resolved calls and named imports
    of changed declarations are kept. Matching files are returned as context
    files (``change_kind: "unchanged"``) with just their calling symbols.
    """
    target_ids = {
        item["id"] for item in symbols
        if item["change_kind"] != "unchanged" and item["kind"] != "import"
    }
    deleted_ids = {item["id"] for item in symbols if item["change_kind"] == "deleted"}
    # Deleted classes stay valid inheritance targets through pre-change parses.
    class_ids = {item["id"] for item in symbols if item["id"] in target_ids and item["kind"] == "class"}
    names = sorted({
        item["name"] for item in symbols
        if item["id"] in target_ids and item["parent_id"] is None
        and item["kind"] in CONTEXT_TARGET_KINDS
    })
    if not names:
        return [], [], []
    candidates = _grep_candidate_paths(root, names, revision)
    if candidates is None:
        warnings.append(_warning("PARTIAL_ANALYSIS", None, "Callers outside the diff were not searched: git grep failed."))
        return [], [], []
    candidates = [
        path for path in candidates
        if path not in changed_paths and Path(path).suffix.lower() in PYTHON_SUFFIXES
    ]
    if len(candidates) > MAX_CONTEXT_FILES:
        warnings.append(_warning(
            "PARTIAL_ANALYSIS", None,
            "Callers outside the diff were searched in the first {} of {} candidate files.".format(
                MAX_CONTEXT_FILES, len(candidates)
            ),
        ))
        candidates = candidates[:MAX_CONTEXT_FILES]

    def translate(bindings: Dict[Tuple[str, int], Dict[str, str]]) -> Dict[Tuple[str, int], Dict[str, str]]:
        # Pre-image ids use the old path; artifact ids use the output path.
        out = {}
        for key, values in bindings.items():
            out[key] = {}
            for binding, target in values.items():
                for old_prefix, new_prefix in pre_id_paths.items():
                    if target.startswith(old_prefix):
                        target = new_prefix + target[len(old_prefix):]
                        break
                out[key][binding] = target
        return out

    files: List[Dict] = []
    context_symbols: List[Dict] = []
    relationships: List[Dict] = []
    blobs = _read_snapshot_blobs(root, candidates, revision)
    for path in candidates:
        if path not in blobs:
            continue
        oid, content = blobs[path]
        try:
            new_symbols, new_imports, calls, bindings, rebindings = _parse_python(content)
        except (StructuralDependencyError, UnicodeDecodeError, ValueError, RuntimeError, TypeError):
            continue  # an unparseable unchanged file says nothing about the diff
        parsed = _ParsedFile(path, new_symbols, new_imports)

        def own(resolved: Dict) -> Dict[Tuple[str, int], Dict[str, str]]:
            return {key: value for (source, key), value in resolved.items() if source == path}

        call_bindings = _merge_bindings(
            own(_resolved_import_targets(post_parses + [parsed])),
            translate(own(_resolved_import_targets(pre_parses + [parsed]))),
            deleted_ids,
        )
        declaration_bindings = _merge_bindings(
            own(_resolved_import_targets(post_parses + [parsed], kinds=CONTEXT_TARGET_KINDS)),
            translate(own(_resolved_import_targets(pre_parses + [parsed], kinds=CONTEXT_TARGET_KINDS))),
            deleted_ids,
        )
        import_map = _keyed_imports(new_imports)
        imported = _imported_call_targets(import_map, rebindings, call_bindings)
        symbol_map = {item.qualified_name: item for item in new_symbols}

        edges: List[Dict] = []
        callers: set = set()
        occurrences: Dict[Tuple[str, str], int] = {}
        for call in calls:
            target = _resolve_call_target(call, symbol_map, bindings, imported)
            if target not in (class_ids if call.kind == "inherits" else target_ids):
                continue
            subject = call.subject if call.kind == "inherits" else call.caller
            source = _symbol_id(path, subject) if subject is not None else "file::" + path
            if subject is not None:
                callers.add(subject)
            occurrence = occurrences.get((source, target), 0)
            occurrences[(source, target)] = occurrence + 1
            edges.append({
                "id": "rel::{}->{}{}".format(source, target, "" if occurrence == 0 else "#{}".format(occurrence)),
                "kind": "inherits" if call.kind == "inherits" else "calls",
                "source_id": source, "target_id": target,
                "analysis_source": "structural", "resolution_method": "resolved", "confidence": None,
                "evidence": [{
                    "kind": "call_site", "file": path, "line_start": call.line, "line_end": call.line,
                    "snippet": call.snippet, "detail": _parser_provenance(oid),
                }],
            })
        seen_imports: set = set()
        for key, item in sorted(import_map.items()):
            for binding, target in sorted(declaration_bindings.get(key, {}).items()):
                edge_id = "rel::file::{}->{}".format(path, target)
                if target not in target_ids or edge_id in seen_imports:
                    continue
                seen_imports.add(edge_id)
                edges.append({
                    "id": edge_id, "kind": "imports", "source_id": "file::" + path, "target_id": target,
                    "analysis_source": "structural", "label": "resolved binding: " + binding,
                    "evidence": [{
                        "kind": "import_statement", "file": path, "line_start": item.line,
                        "line_end": item.line, "snippet": item.snippet, "detail": _parser_provenance(oid),
                    }],
                })
        if not edges:
            continue

        # Keep each caller's enclosing declarations so parent ids resolve.
        keep: set = set()
        for qname in callers:
            while qname is not None and qname not in keep and qname in symbol_map:
                keep.add(qname)
                qname = symbol_map[qname].parent
        for qname in sorted(keep):
            item = symbol_map[qname]
            context_symbols.append({
                "id": _symbol_id(path, qname), "name": item.name, "qualified_name": qname,
                "file_id": "file::" + path, "kind": item.kind,
                "parent_id": _symbol_id(path, item.parent) if item.parent else None,
                "change_kind": "unchanged", "analysis_source": "structural",
                "location": {"file": path, "line_start": item.start_line, "line_end": item.end_line},
                "evidence": _evidence(path, item, oid),
            })
        files.append({
            "id": "file::" + path, "path": path, "old_path": None, "language": "python",
            "change_kind": "unchanged", "lines_added": 0, "lines_removed": 0,
            "analysis_source": "structural",
            "evidence": [{"kind": "ast_parse", "file": path, "detail": _parser_provenance(oid)}],
        })
        relationships.extend(edges)
    return files, context_symbols, relationships


def analyze_local_diff(
    repository: str = ".", *, staged: bool = False,
    pathspecs: Optional[Sequence[str]] = None,
    base_ref: Optional[str] = None,
    head_ref: Optional[str] = None,
    three_dot: bool = False,
    against: Optional[str] = None,
    wild_version: str = package_version,
) -> Dict:
    """Build a schema-v2 artifact for a local snapshot or immutable commit range.

    ``against`` compares one commit with the working tree (``git diff REF``)
    or, with ``staged``, with the index (``git diff --cached REF``).
    """
    started = time.monotonic()
    root = repository_root(repository)
    if (base_ref is None) != (head_ref is None):
        raise ValueError("base_ref and head_ref must be supplied together")
    is_commit_range = base_ref is not None
    if is_commit_range and staged:
        raise ValueError("staged and commit-range comparisons are mutually exclusive")
    if is_commit_range and against is not None:
        raise ValueError("a commit range and a single base ref are mutually exclusive")
    resolution = (
        resolve_against_commit(repository, against, staged=staged, pathspecs=pathspecs)
        if against is not None
        else resolve_commit_range(
            repository,
            base_ref,
            head_ref,
            three_dot=three_dot,
            pathspecs=pathspecs,
        )
        if is_commit_range
        else (resolve_staged if staged else resolve_unstaged)(repository, pathspecs)
    )
    warnings = [_resolution_warning(item) for item in resolution.warnings]
    files: List[Dict] = []
    symbols: List[Dict] = []
    relationships: List[Dict] = []
    analyzed = skipped = 0

    # Build a closed-world module map before emitting relationships.  It covers
    # every changed Python post-image, so a source file can resolve an import
    # regardless of diff ordering without reading arbitrary repository files.
    # This preserves the artifact's changeset scope and makes duplicate module
    # names deliberately unresolved rather than guessed.
    cross_file_parses: List[_ParsedFile] = []
    for candidate in resolution.entries:
        candidate_path = candidate.new_path
        if (
            candidate_path is None
            or candidate.new_mode == "160000"
            or Path(candidate_path).suffix.lower() not in PYTHON_SUFFIXES
            or _non_regular_modes(candidate)
        ):
            continue
        try:
            candidate_new = (
                _blob(root, candidate.new_oid)
                if staged or is_commit_range
                else _worktree_bytes(root, candidate)
            )
            if candidate_new is None:
                continue
            new_symbols, new_imports, _new_calls, _new_bindings, _new_rebindings = (
                _parse_python(candidate_new)
            )
        except (
            OSError, GitSnapshotError, StructuralDependencyError,
            UnicodeDecodeError, ValueError, RuntimeError, TypeError,
        ):
            continue
        cross_file_parses.append(
            _ParsedFile(candidate_path, new_symbols, new_imports)
        )
    # Pre-image parses let callers outside the diff resolve to deleted
    # declarations, the most direct way an untouched call site breaks.
    pre_change_parses: List[_ParsedFile] = []
    pre_change_id_paths: Dict[str, str] = {}
    for candidate in resolution.entries:
        old_path = candidate.old_path
        if (
            old_path is None or candidate.old_oid is None
            or candidate.old_mode == "160000"
            or Path(old_path).suffix.lower() not in PYTHON_SUFFIXES
        ):
            continue
        try:
            old_symbols, old_imports, _c, _b, _r = _parse_python(_blob(root, candidate.old_oid))
        except (
            OSError, GitSnapshotError, StructuralDependencyError,
            UnicodeDecodeError, ValueError, RuntimeError, TypeError,
        ):
            continue
        pre_change_parses.append(_ParsedFile(old_path, old_symbols, old_imports))
        output = candidate.new_path or old_path
        if output != old_path:
            pre_change_id_paths["sym::{}::".format(old_path)] = "sym::{}::".format(output)
    cross_import_paths = _resolved_import_paths(cross_file_parses)
    cross_import_targets = _resolved_import_targets(cross_file_parses)
    changed_class_ids = {
        _symbol_id(parsed.path, item.qualified_name)
        for parsed in cross_file_parses
        for item in parsed.new_symbols
        if item.kind == "class"
    }
    # Import attribution is broader than call resolution: any unique top-level
    # declaration can be imported by name, but only callables are call targets.
    cross_import_declarations = _resolved_import_targets(
        cross_file_parses, kinds=("function", "class", "constant", "type_alias")
    )

    for entry in resolution.entries:
        path = entry.new_path or entry.old_path
        if path is None:
            raise RuntimeError("snapshot entry has neither an old nor a new path")
        # Gitlink (mode 160000) OIDs identify commits, not blobs, so content
        # reads would fail or be misleading. Skip reads for gitlink sides and
        # let _non_regular_modes() report the opaque entry cleanly; continue
        # reading symlink sides normally.
        try:
            old = (
                None if entry.old_mode == "160000" else _blob(root, entry.old_oid)
            )
            if entry.new_mode == "160000":
                new = None
            elif staged or is_commit_range:
                new = _blob(root, entry.new_oid)
            else:
                new = _worktree_bytes(root, entry)
        except (OSError, GitSnapshotError) as error:
            old = new = None
            warnings.append(_warning("PARTIAL_ANALYSIS", path, "snapshot read failed: {}".format(error)))

        snapshot_missing = (
            old is None and entry.old_oid is not None
        ) or (
            new is None and entry.new_oid is not None
        )
        binary_sides = () if snapshot_missing else _binary_sides(old, new)
        non_regular_modes = _non_regular_modes(entry)
        lines_added, lines_removed = (
            (None, None)
            if snapshot_missing or binary_sides or non_regular_modes
            else _line_counts(old, new)
        )
        file_entry = {
            "id": "file::" + path, "path": path,
            "old_path": entry.old_path if entry.status in ("R", "C") else None,
            "language": (
                "python"
                if not binary_sides
                and not non_regular_modes
                and Path(path).suffix.lower() in PYTHON_SUFFIXES
                else None
            ),
            "change_kind": _change_kind(entry.status, entry.old_oid, entry.new_oid),
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "analysis_source": "structural",
            "evidence": [{"kind": "git_diff_name_status", "detail": _provenance(entry, old, new)}],
        }
        if not (snapshot_missing or binary_sides or non_regular_modes):
            diff = _file_diff(entry.old_path or path, path, old, new)
            if diff is not None:
                file_entry["diff"] = diff
        files.append(file_entry)
        if binary_sides:
            skipped += 1
            warnings.append(_warning(
                "PARTIAL_ANALYSIS",
                path,
                "Binary content detected in {} snapshot; structural parsing and line counts were skipped.".format(
                    " and ".join(binary_sides)
                ),
            ))
            continue
        if non_regular_modes:
            skipped += 1
            warnings.append(_warning(
                "PARTIAL_ANALYSIS",
                path,
                "Non-regular Git mode {} detected; structural parsing and line "
                "counts were skipped.".format(", ".join(non_regular_modes)),
            ))
            continue
        if file_entry["language"] != "python":
            skipped += 1
            warnings.append(_unsupported_language_warning(entry, path))
            continue
        if (old is None and entry.old_oid is not None) or (
            new is None and entry.new_oid is not None
        ):
            skipped += 1
            continue

        parser_errors = (
            StructuralDependencyError,
            UnicodeDecodeError,
            ValueError,
            RuntimeError,
            OSError,
            TypeError,
        )
        try:
            old_symbols, old_imports, _old_calls, _old_bindings, _old_rebindings = (
                _parse_python(old) if old is not None else ([], [], [], {}, [])
            )
        except parser_errors as error:
            warnings.append(_parse_failure_warning(
                "pre", entry.old_path or path, entry.old_oid, error,
            ))
            skipped += 1
            continue
        try:
            new_symbols, new_imports, new_calls, new_bindings, new_rebindings = (
                _parse_python(new) if new is not None else ([], [], [], {}, [])
            )
        except parser_errors as error:
            warnings.append(_parse_failure_warning(
                "post", path, entry.new_oid, error,
            ))
            skipped += 1
            continue
        analyzed += 1
        old_map = {item.qualified_name: item for item in old_symbols}
        new_map = {item.qualified_name: item for item in new_symbols}
        output_path = path
        for qname in sorted(set(old_map) | set(new_map)):
            before, after = old_map.get(qname), new_map.get(qname)
            current = after or before
            if current is None:
                raise RuntimeError("symbol comparison produced no symbol")
            kind = "added" if before is None else "deleted" if after is None else (
                "unchanged" if before.text_hash == after.text_hash else "modified"
            )
            oid = entry.new_oid if after is not None else entry.old_oid
            location = None if after is None else {
                "file": output_path, "line_start": current.start_line, "line_end": current.end_line,
            }
            symbol_entry = {
                "id": _symbol_id(output_path, qname), "name": current.name,
                "qualified_name": qname, "file_id": "file::" + output_path,
                "kind": current.kind,
                "parent_id": _symbol_id(output_path, current.parent) if current.parent else None,
                "change_kind": kind, "analysis_source": "structural",
                "location": location, "evidence": _evidence(output_path, current, oid),
            }
            if kind != "unchanged":
                symbol_entry["source"] = {
                    "before": _symbol_source(old, before),
                    "after": _symbol_source(new, after),
                }
            symbols.append(symbol_entry)

        for qname, item in sorted(new_map.items()):
            source = _symbol_id(output_path, qname)
            if item.parent:
                target = _symbol_id(output_path, item.parent)
                relationships.append({
                    "id": "rel::{}->{}".format(target, source), "kind": "contains",
                    "source_id": target, "target_id": source, "analysis_source": "structural",
                    "evidence": _evidence(output_path, item, entry.new_oid),
                })
            elif item.kind in ("function", "class", "constant", "type_alias"):
                relationships.append({
                    "id": "rel::file::{}->{}".format(output_path, source), "kind": "defines",
                    "source_id": "file::" + output_path, "target_id": source,
                    "analysis_source": "structural", "evidence": _evidence(output_path, item, entry.new_oid),
                })

        old_import_map = _keyed_imports(old_imports)
        new_import_map = _keyed_imports(new_imports)
        imported_call_targets = _imported_call_targets(
            new_import_map,
            new_rebindings,
            {
                key: targets
                for (source_path, key), targets in cross_import_targets.items()
                if source_path == output_path
            },
        )
        resolved_import_occurrences: Dict[Tuple[str, str], int] = {}
        resolved_binding_edges: set = set()
        for import_key in sorted(set(old_import_map) | set(new_import_map)):
            before = old_import_map.get(import_key)
            after = new_import_map.get(import_key)
            item = after or before
            if item is None:
                raise RuntimeError("import comparison produced no import")
            module, occurrence = import_key
            suffix = "" if occurrence == 0 else "#{}".format(occurrence)
            qualified_name = "import::{}{}".format(module, suffix)
            target = _symbol_id(output_path, qualified_name)
            oid = entry.new_oid if after is not None else entry.old_oid
            evidence = [
                {
                    "kind": "import_statement",
                    "file": output_path,
                    "line_start": item.line,
                    "line_end": item.line,
                    "snippet": item.snippet,
                    "detail": _parser_provenance(oid),
                }
            ]
            import_kind = (
                "added"
                if before is None
                else "deleted"
                if after is None
                else "unchanged"
                if before.snippet == after.snippet
                else "modified"
            )
            location = (
                {
                    "file": output_path,
                    "line_start": item.line,
                    "line_end": item.line,
                }
                if after is not None
                else None
            )
            resolved_path = (
                cross_import_paths.get((output_path, import_key))
                if after is not None
                else None
            )
            # Retain the local import evidence node even for an internal
            # module edge: unresolved/non-callable bindings may still refer to
            # it as a conservative call target.
            symbols.append(
                {
                    "id": target,
                    "name": module,
                    "qualified_name": qualified_name,
                    "file_id": "file::" + output_path,
                    "kind": "import",
                    "parent_id": None,
                    "change_kind": import_kind,
                    "analysis_source": "structural",
                    "location": location,
                    "evidence": evidence,
                }
            )
            if resolved_path is not None:
                edge_key = (output_path, resolved_path)
                occurrence = resolved_import_occurrences.get(edge_key, 0)
                resolved_import_occurrences[edge_key] = occurrence + 1
                edge_suffix = "" if occurrence == 0 else "#{}".format(occurrence)
                relationships.append(
                    {
                        "id": "rel::file::{}->file::{}{}".format(
                            output_path, resolved_path, edge_suffix
                        ),
                        "kind": "imports",
                        "source_id": "file::" + output_path,
                        "target_id": "file::" + resolved_path,
                        "analysis_source": "structural",
                        "evidence": evidence,
                        "label": "resolved module: " + module,
                    }
                )
                declarations = cross_import_declarations.get(
                    (output_path, import_key), {}
                )
                for binding, declaration_id in sorted(declarations.items()):
                    edge_id = "rel::file::{}->{}".format(output_path, declaration_id)
                    if edge_id in resolved_binding_edges:
                        continue
                    resolved_binding_edges.add(edge_id)
                    relationships.append(
                        {
                            "id": edge_id,
                            "kind": "imports",
                            "source_id": "file::" + output_path,
                            "target_id": declaration_id,
                            "analysis_source": "structural",
                            "evidence": evidence,
                            "label": "resolved binding: " + binding,
                        }
                    )
            elif after is not None:
                relationships.append(
                    {
                        "id": "rel::file::{}->{}".format(output_path, target),
                        "kind": "imports",
                        "source_id": "file::" + output_path,
                        "target_id": target,
                        "analysis_source": "structural",
                        "evidence": evidence,
                        "label": "unresolved/external module: " + module,
                    }
                )

        call_occurrences: Dict[Tuple[str, str], int] = {}
        for call in new_calls:
            target_qname = _resolve_call_target(
                call, new_map, new_bindings, imported_call_targets
            )
            if target_qname is None:
                continue
            # Inheritance edges need a confirmed class: local, or in a changed
            # file. An external base keeps only its import symbol and evidence.
            if call.kind == "inherits" and not (
                (target_qname in new_map and new_map[target_qname].kind == "class")
                or target_qname in changed_class_ids
            ):
                continue
            subject = call.subject if call.kind == "inherits" else call.caller
            source = (
                _symbol_id(output_path, subject)
                if subject is not None
                else "file::" + output_path
            )
            target = (
                target_qname
                if target_qname.startswith(("sym::", "file::"))
                else _symbol_id(output_path, target_qname)
            )
            edge = (source, target)
            occurrence = call_occurrences.get(edge, 0)
            call_occurrences[edge] = occurrence + 1
            suffix = "" if occurrence == 0 else "#{}".format(occurrence)
            relationships.append(
                {
                    "id": "rel::{}->{}{}".format(source, target, suffix),
                    "kind": "inherits" if call.kind == "inherits" else "calls",
                    "source_id": source,
                    "target_id": target,
                    "analysis_source": "structural",
                    "resolution_method": (
                        "import_grounded"
                        if target_qname.startswith("import::")
                        else "resolved"
                    ),
                    "confidence": None,
                    "evidence": [
                        {
                            "kind": "call_site",
                            "file": output_path,
                            "line_start": call.line,
                            "line_end": call.line,
                            "snippet": call.snippet,
                            "detail": _parser_provenance(entry.new_oid),
                        }
                    ],
                }
            )

    context_files, context_symbols, context_relationships = _context_callers(
        root,
        resolution.head_oid if is_commit_range else None,
        {path for entry in resolution.entries for path in (entry.old_path, entry.new_path) if path},
        cross_file_parses,
        pre_change_parses,
        pre_change_id_paths,
        symbols,
        warnings,
    )
    files.extend(context_files)
    symbols.extend(context_symbols)
    relationships.extend(context_relationships)

    files.sort(key=lambda item: item["path"])
    symbols.sort(key=lambda item: item["id"])
    relationships.sort(key=lambda item: (item["id"], item["kind"]))
    warnings.sort(key=lambda item: (item.get("file", ""), item["code"], item.get("detail", "")))
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wild_version": wild_version,
        "diff_ref": {
            "kind": (
                "commit_range" if is_commit_range
                else "staged" if staged
                else "working_tree" if against is not None
                else "unstaged"
            ),
            "base_ref": (
                resolution.comparison_base_oid if is_commit_range
                else resolution.base_oid if against is not None
                else "HEAD" if staged else None
            ),
            "head_ref": resolution.head_oid if is_commit_range else None,
            "requested_base_ref": (
                resolution.base_ref if is_commit_range or against is not None else None
            ),
            "requested_head_ref": resolution.head_ref if is_commit_range else None,
            "comparison_mode": (
                ("three_dot" if three_dot else "two_dot") if is_commit_range else None
            ),
            "pathspecs": list(pathspecs or []), "repo_root": root,
        },
        "files": files, "symbols": symbols, "relationships": relationships,
        "summary": None,
        "metadata": {
            "privacy_tier": "local", "cloud_providers_used": [],
            "analysis_duration_ms": int((time.monotonic() - started) * 1000),
            "languages_detected": ["python"] if any(f["language"] == "python" for f in files) else [],
            "files_analyzed": analyzed, "files_skipped": skipped, "llm_calls": 0,
            "llm_model": None, "tiers_used": ["structural"], "warnings": warnings,
        },
    }
