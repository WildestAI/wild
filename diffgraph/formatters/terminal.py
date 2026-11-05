"""
Terminal formatter for DiffGraph v2 output.

Renders a schema v2 DiffGraph dict as a ranked review path in the terminal.
Pure consumer — knows nothing about git, tree-sitter, or LLMs.

Usage:
    formatter = TerminalFormatter(diffgraph_v2_dict)
    formatter.render(sys.stdout)

Flags (set via constructor):
    compact      Omit the CONTEXT section (unchanged symbols)
    max_items    Cap on REVIEW FIRST + REVIEW NEXT (None = --all, no limit)
    color        Override auto-detection (True/False); default: detect from TTY + NO_COLOR
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional

from diffgraph.contract import ValidatedArtifact
from diffgraph.formatters.common import collapsed as _collapsed
from diffgraph.formatters.common import warning_lines as _warning_lines


# ---------------------------------------------------------------------------
# Color / terminal helpers
# ---------------------------------------------------------------------------

def use_color(out=None) -> bool:
    """Return True if ANSI color codes should be emitted."""
    if "NO_COLOR" in os.environ:
        return False
    stream = out if out is not None else sys.stdout
    if not hasattr(stream, "isatty"):
        return False
    return stream.isatty()


def _is_dumb_terminal() -> bool:
    return os.environ.get("TERM", "") == "dumb"


# ANSI escape helpers
def _ansi(code: str, text: str, color: bool) -> str:
    if not color:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(text: str, color: bool) -> str:
    return _ansi("1", text, color)


def dim(text: str, color: bool) -> str:
    return _ansi("2", text, color)


def yellow(text: str, color: bool) -> str:
    return _ansi("33", text, color)


def green(text: str, color: bool) -> str:
    return _ansi("32", text, color)


def red(text: str, color: bool) -> str:
    return _ansi("31", text, color)


def bold_yellow(text: str, color: bool) -> str:
    return _ansi("1;33", text, color)


def dim_red(text: str, color: bool) -> str:
    return _ansi("2;31", text, color)


_SECTION_STYLE_FNS = {
    "bold": bold,
    "bold_yellow": bold_yellow,
    "dim": dim,
}


def _change_label(change_kind: str, color: bool) -> str:
    """Return colored [label] for a change_kind value."""
    labels = {
        "modified": ("[modified]", yellow),
        "added": ("[added]", green),
        "deleted": ("[deleted]", red),
        "unchanged": ("[unchanged]", dim),
    }
    label, colorize = labels.get(change_kind, ("[unknown]", lambda t, c: t))
    return colorize(label, color)


def _summary_text(node: dict) -> Optional[str]:
    """Return a node's AI summary text, if the artifact carries one."""
    summary = node.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("text"), str):
        # Model text is untrusted: an escape sequence must not reach the terminal.
        cleaned = "".join(ch if ch.isprintable() else " " for ch in summary["text"])
        return " ".join(cleaned.split()) or None
    return None


def _write_summary(out, node: dict, color: bool) -> None:
    text = _summary_text(node)
    if text:
        out.write("    " + dim(f"AI: {text}", color) + "\n")


# ---------------------------------------------------------------------------
# Ranking data structures
# ---------------------------------------------------------------------------

@dataclass
class _RankedSymbol:
    """Enriched symbol entry ready for rendering."""
    symbol: dict
    importer_paths: list[str] = field(default_factory=list)
    outside_callers: list[str] = field(default_factory=list)
    lines_changed: int = 0
    score: int = 0


@dataclass
class RankedSymbols:
    """Three-bucket output from _rank_symbols()."""
    review_first: list[_RankedSymbol] = field(default_factory=list)
    review_next: list[_RankedSymbol] = field(default_factory=list)
    context: list[_RankedSymbol] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TerminalFormatter
# ---------------------------------------------------------------------------

class TerminalFormatter:
    """
    Render a schema v2 DiffGraph dict as a ranked terminal review path.

    Args:
        diffgraph:  Schema v2 dict (from TreeSitterProcessor.analyze_changes() or any v2 source)
        compact:    Omit the CONTEXT section when True
        max_items:  Cap per section (REVIEW FIRST, REVIEW NEXT); None = no cap
        color:      Force color on/off; None = auto-detect from TTY + NO_COLOR
    """

    DEFAULT_MAX_ITEMS = 10
    def __init__(
        self,
        diffgraph: dict | ValidatedArtifact,
        *,
        compact: bool = False,
        max_items: Optional[int] = DEFAULT_MAX_ITEMS,
        color: Optional[bool] = None,
    ):
        validated = (
            diffgraph
            if isinstance(diffgraph, ValidatedArtifact)
            else ValidatedArtifact.from_value(diffgraph)
        )
        self.artifact = validated
        self.dg = validated.value
        self.compact = compact
        self.max_items = max_items
        self._color_override = color

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, out=None) -> None:
        """Write the full ranked review path to `out` (default: sys.stdout)."""
        if out is None:
            out = sys.stdout
        color = self._color_override if self._color_override is not None else use_color(out)

        ranked = self._rank_symbols()
        self._write_header(out, color)
        self._write_warnings(out, color)
        if not self.dg.get("files"):
            out.write("No changes in the selected snapshot.\n\n")
        self._write_section("REVIEW FIRST", ranked.review_first, out, color, section_style="bold_yellow")
        self._write_section("REVIEW NEXT", ranked.review_next, out, color, section_style="bold")
        if not self.compact:
            self._write_section("CONTEXT", ranked.context, out, color, section_style="dim")
        self._write_footer(out, color)

    # ------------------------------------------------------------------
    # Ranking (pure function — no I/O)
    # ------------------------------------------------------------------

    def _rank_symbols(self) -> RankedSymbols:
        """
        Pure function: DiffGraph v2 dict → RankedSymbols.
        No I/O, no side effects. Directly testable.
        """
        symbols = self.dg.get("symbols", [])
        relationships = self.dg.get("relationships", [])
        files = self.dg.get("files", [])

        # Build lookup: file_id → file path
        file_id_to_path: dict[str, str] = {
            f["id"]: f.get("path", f["id"]) for f in files
        }

        # Context files sit outside the diff; they are present only because
        # they call or import changed declarations.
        context_file_ids = {
            f["id"] for f in files if f.get("change_kind", "unchanged") == "unchanged"
        }

        def importer_label(file_id: str) -> str:
            path = file_id_to_path.get(file_id, file_id)
            return f"{path} (outside diff)" if file_id in context_file_ids else path

        # Build import relationship index: target symbol id → importing file ids.
        # Only symbol-targeted edges count: a file-level import says the module
        # is used, not which of its symbols, so it must not be credited to every
        # changed symbol in that file. Edges from a file to its own import stubs
        # are excluded because they describe the import, not a dependent.
        symbols_by_id = {sym.get("id"): sym for sym in symbols}
        import_index: dict[str, set[str]] = {}
        for rel in relationships:
            target = rel.get("target_id", "")
            target_symbol = symbols_by_id.get(target)
            if (
                rel.get("kind") == "imports"
                and rel.get("analysis_source") in ("structural", "derived")
                and rel.get("source_id") in file_id_to_path
                and target_symbol is not None
                and target_symbol.get("file_id") != rel.get("source_id")
            ):
                import_index.setdefault(target, set()).add(rel["source_id"])

        # Call sites outside the diff: the places a changed signature can
        # break without appearing in the diff at all.
        outside_calls: dict[str, list[str]] = {}
        for rel in relationships:
            source = rel.get("source_id", "")
            source_symbol = symbols_by_id.get(source)
            source_file = source_symbol.get("file_id") if source_symbol else source
            if rel.get("kind") not in ("calls", "inherits") or source_file not in context_file_ids:
                continue
            line = next(
                (e.get("line_start") for e in rel.get("evidence") or [] if e.get("kind") == "call_site"),
                None,
            )
            where = file_id_to_path.get(source_file, source_file)
            if source_symbol:
                where += f":{source_symbol.get('qualified_name') or source_symbol.get('name')}"
            if line:
                where += f" (L{line})"
            if rel.get("kind") == "inherits":
                where += " (subclass)"
            outside_calls.setdefault(rel.get("target_id", ""), []).append(where)

        ranked = RankedSymbols()

        for sym in symbols:
            change_kind = sym.get("change_kind", "unchanged")

            if change_kind == "unchanged":
                if sym.get("file_id") not in context_file_ids:
                    ranked.context.append(_RankedSymbol(symbol=sym))
                continue

            # Files in the diff that import this symbol, or the top-level
            # declaration containing it (importing a class exposes its methods).
            importer_file_ids: set[str] = set()
            current = sym
            seen: set = set()
            while current is not None and current.get("id") not in seen:
                seen.add(current.get("id"))
                importer_file_ids |= import_index.get(current.get("id"), set())
                current = symbols_by_id.get(current.get("parent_id"))
            importer_paths = [importer_label(fid) for fid in sorted(importer_file_ids)]
            outside_callers = sorted(set(outside_calls.get(sym.get("id"), [])))

            # DiffGraph v2 currently exposes declaration spans, not per-symbol
            # changed-line evidence. Do not present or rank a declaration's size
            # as though every line changed; importer count is the honest signal
            # available to this pure artifact consumer.
            lines_changed = 0
            score = (len(importer_paths) + len(outside_callers)) * 3

            rs = _RankedSymbol(
                symbol=sym,
                importer_paths=importer_paths,
                outside_callers=outside_callers,
                lines_changed=lines_changed,
                score=score,
            )

            if importer_paths or outside_callers:
                ranked.review_first.append(rs)
            else:
                ranked.review_next.append(rs)

        # Sort each bucket
        ranked.review_first.sort(
            key=lambda r: (-r.score, r.symbol.get("name", ""))
        )
        ranked.review_next.sort(key=lambda r: r.symbol.get("name", ""))
        ranked.context.sort(
            key=lambda r: (
                file_id_to_path.get(r.symbol.get("file_id", ""), ""),
                r.symbol.get("location", {}).get("line_start", 0),
            )
        )

        return ranked

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _write_header(self, out, color: bool) -> None:
        files = self.dg.get("files", [])
        symbols = self.dg.get("symbols", [])

        changed_files = sum(1 for f in files if f.get("change_kind", "unchanged") != "unchanged")
        modified = sum(1 for s in symbols if s.get("change_kind") == "modified")
        added = sum(1 for s in symbols if s.get("change_kind") == "added")
        deleted = sum(1 for s in symbols if s.get("change_kind") == "deleted")

        # Detect fallback (no symbols extracted)
        if not symbols and files:
            parts = [f"{changed_files} file{'s' if changed_files != 1 else ''} changed"]
            parts.append("symbol extraction unavailable")
            line = bold("wild diff", color) + " — " + "  ·  ".join(parts)
        else:
            sym_parts = []
            if modified:
                sym_parts.append(f"{modified} symbol{'s' if modified != 1 else ''} modified")
            if added:
                sym_parts.append(f"{added} added")
            if deleted:
                sym_parts.append(f"{deleted} deleted")

            counts = f"{changed_files} file{'s' if changed_files != 1 else ''} changed"
            if sym_parts:
                counts += "  ·  " + "  ·  ".join(sym_parts)

            line = bold("wild diff", color) + " — " + counts

        out.write(line + "\n\n")

    def _write_warnings(self, out, color: bool) -> None:
        """Write any per-file warnings (e.g. unsupported language fallbacks)."""
        metadata = self.dg.get("metadata", {})
        warnings = metadata.get("warnings", [])
        for line in _warning_lines(warnings):
            out.write(dim_red("⚠", color) + f"  {line}\n")
        if warnings:
            out.write("\n")

        # File-level fallback section when no symbols
        files = self.dg.get("files", [])
        symbols = self.dg.get("symbols", [])
        if not symbols and files:
            changed_files = [f for f in files if f.get("change_kind", "unchanged") != "unchanged"]
            if changed_files:
                out.write(bold("▶ FILES CHANGED", color) + "\n")
                for f in changed_files:
                    path = f.get("path", f.get("id", "?"))
                    additions = f.get("lines_added") or 0
                    deletions = f.get("lines_removed") or 0
                    out.write(f"  {path}  +{additions} / -{deletions}\n")
                    _write_summary(out, f, color)
                out.write("\n")
        elif any(_summary_text(f) for f in files):
            # Symbol sections cover code; this keeps per-file summaries,
            # including non-Python files, visible too.
            out.write(bold("▶ FILES", color) + "\n")
            for f in files:
                if _summary_text(f):
                    path = f.get("path", f.get("id", "?"))
                    label = _change_label(f.get("change_kind", "unknown"), color)
                    out.write(f"  {path} {label}\n")
                    _write_summary(out, f, color)
            out.write("\n")

    def _write_section(
        self,
        title: str,
        items: list[_RankedSymbol],
        out,
        color: bool,
        section_style: str = "bold",
    ) -> None:
        if not items:
            return  # Never show empty sections

        # Apply cap
        total = len(items)
        capped = items if self.max_items is None else items[: self.max_items]
        hidden = total - len(capped)

        # Section header
        prefix = "▶ " if not _is_dumb_terminal() else "> "
        header_text = f"{prefix}{title}"
        style_fn = _SECTION_STYLE_FNS.get(section_style, bold)
        header = style_fn(header_text, color)

        if hidden > 0:
            hint = f"  (showing top {len(capped)} of {total} · run: wild diff --all to see all)"
            out.write(header + dim(hint, color) + "\n")
        else:
            out.write(header + "\n")

        for rs in capped:
            self._write_symbol_entry(rs, out, color, title)

        out.write("\n")

    def _write_symbol_entry(
        self, rs: _RankedSymbol, out, color: bool, section_title: str
    ) -> None:
        sym = rs.symbol
        name = sym.get("name", "<unknown>")
        change_kind = sym.get("change_kind", "unknown")
        file_id = sym.get("file_id", "")

        # Resolve file path from files[]
        files = self.dg.get("files", [])
        file_path = next(
            (f.get("path", f.get("id", file_id)) for f in files if f["id"] == file_id),
            file_id,
        )

        change_label = _change_label(change_kind, color)

        # Main line: "  file/path.py  SymbolName [modified]  · 29 lines"
        line = f"  {file_path}  {bold(name, color)} {change_label}"
        if rs.lines_changed:
            line += f"  · {rs.lines_changed} lines"
        elif change_kind in ("added", "deleted") and rs.symbol.get("location"):
            loc = rs.symbol["location"]
            loc_lines = max(0, loc.get("line_end", 0) - loc.get("line_start", 0) + 1)
            if loc_lines:
                line += f"  · {loc_lines} lines"
        out.write(line + "\n")
        _write_summary(out, sym, color)

        # Importer line (REVIEW FIRST only)
        if rs.importer_paths and section_title == "REVIEW FIRST":
            importer_str = _collapsed(rs.importer_paths)
            arrow = "↳" if not _is_dumb_terminal() else "+->"
            out.write(
                "    " + dim(f"{arrow} imported by: {importer_str}", color) + "\n"
            )
        if rs.outside_callers and section_title == "REVIEW FIRST":
            arrow = "↳" if not _is_dumb_terminal() else "+->"
            out.write(
                "    "
                + dim(f"{arrow} called outside the diff: {_collapsed(rs.outside_callers)}", color)
                + "\n"
            )

    def _write_footer(self, out, color: bool) -> None:
        metadata = self.dg.get("metadata", {})
        analysis_source = metadata.get("analysis_source", "structural")
        duration_ms = metadata.get("analysis_duration_ms")
        diff_ref = self.dg.get("diff_ref", {})
        diff_kind = diff_ref.get("kind", "unstaged")

        # Detect languages from files[]
        files = self.dg.get("files", [])
        langs = sorted({f.get("language") for f in files if f.get("language")})
        lang_str = " · ".join(langs) if langs else ""

        # Diff context
        diff_context_map = {
            "unstaged": "unstaged changes",
            "staged": "staged changes",
        }
        requested_base = diff_ref.get("requested_base_ref")
        if diff_kind == "working_tree":
            diff_context = f"{requested_base or diff_ref.get('base_ref', '')} → working tree"
        elif diff_kind == "staged" and requested_base:
            diff_context = f"{requested_base} → index (staged)"
        elif diff_kind in diff_context_map:
            diff_context = diff_context_map[diff_kind]
        elif diff_kind == "commit_range":
            base = diff_ref.get("base_ref", "")
            head = diff_ref.get("head_ref", "")
            diff_context = f"{base}..{head}" if base and head else "commit range"
        elif diff_kind == "file_scope":
            pathspec = diff_ref.get("pathspec", "")
            diff_context = pathspec or "file scope"
        else:
            diff_context = diff_kind

        parts = [analysis_source]
        if lang_str:
            parts.append(lang_str)
        if duration_ms is not None:
            parts.append(f"{duration_ms}ms")
        parts.append(diff_context)

        sep = "─" * 64 if not _is_dumb_terminal() else "-" * 64
        out.write(dim(sep, color) + "\n")
        out.write(dim("Analysis: " + " · ".join(parts), color) + "\n")

        # Optional LLM upgrade hint (shown only for local-structural analysis)
        privacy_tier = metadata.get("privacy_tier", "local")
        if privacy_tier == "local":
            hint = (
                "Optional: wild diff --llm claude|gpt|gemini  "
                "(adds AI summaries · code sent to the model provider via BYOK Relay)"
            )
            out.write(dim(hint, color) + "\n")
        elif metadata.get("llm_model"):
            calls = metadata.get("llm_calls") or 0
            providers = ", ".join(metadata.get("cloud_providers_used") or [])
            note = (
                f"AI summaries: {metadata['llm_model']} via BYOK Relay"
                f"{f' ({providers})' if providers else ''}"
                f" · {calls} call{'s' if calls != 1 else ''}"
            )
            out.write(dim(note, color) + "\n")
