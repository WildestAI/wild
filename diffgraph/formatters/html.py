"""Deterministic, offline HTML adapter for canonical DiffGraph artifacts.

The report is the interactive graph: a small inline SVG renderer (no runtime
dependency) fills the window, a top bar summarizes the snapshot, and a details
panel shows the selected node's evidence. The validated artifact is embedded
as inert JSON for the script; it is not rendered as text (use
``--format json`` for the raw artifact). The renderer performs no repository
reads, structural analysis, AI calls, or network access.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import tempfile
import textwrap
from pathlib import Path

from diffgraph.contract import ValidatedArtifact
from diffgraph.formatters.common import warning_lines


def _text(value: object) -> str:
    """Escape one artifact value for HTML text content."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return html.escape(str(value), quote=True)


def _embedded_json(value: object) -> str:
    """Embed compact JSON without allowing artifact strings to terminate the script tag."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _short_ref(value: object) -> str:
    text = str(value or "")
    return text[:12] if len(text) in (40, 64) else text


def _diff_label(diff_ref: dict) -> str:
    """Human label for the analyzed snapshot (e.g. ``main...feature``)."""
    kind = diff_ref.get("kind")
    if kind == "commit_range":
        base = diff_ref.get("requested_base_ref") or _short_ref(diff_ref.get("base_ref"))
        head = diff_ref.get("requested_head_ref") or _short_ref(diff_ref.get("head_ref"))
        separator = "..." if diff_ref.get("comparison_mode") == "three_dot" else ".."
        label = f"{base}{separator}{head}"
    elif kind == "working_tree":
        label = f"{diff_ref.get('requested_base_ref') or _short_ref(diff_ref.get('base_ref'))} → working tree"
    elif kind == "staged" and diff_ref.get("requested_base_ref"):
        label = f"{diff_ref['requested_base_ref']} → index (staged)"
    else:
        label = {"unstaged": "Unstaged changes", "staged": "Staged changes"}.get(kind, str(kind))
    pathspecs = diff_ref.get("pathspecs") or []
    if pathspecs:
        label += " — " + ", ".join(str(item) for item in pathspecs)
    return label


# This code is static: artifact values remain inert JSON until it is parsed, and
# all text is assigned through textContent (never innerHTML).  Its CSP hash is
# calculated at render time so the report can permit exactly this script.
_INTERACTIVE_SCRIPT = textwrap.dedent(
    """\
    (() => {
      'use strict';
      const NS = 'http://www.w3.org/2000/svg';
      const source = document.getElementById('diffgraph-artifact');
      const host = document.getElementById('diffgraph-graph');
      const layout = document.getElementById('diffgraph-layout');
      const panel = document.getElementById('diffgraph-details');
      if (!source || !host || !panel || !layout) return;
      let artifact;
      try { artifact = JSON.parse(source.textContent); } catch (_) { return; }
      const files = [...(artifact.files || [])].sort((a, b) => a.id.localeCompare(b.id));
      const symbols = [...(artifact.symbols || [])].sort((a, b) => a.id.localeCompare(b.id));
      const relationships = [...(artifact.relationships || [])].sort((a, b) => a.id.localeCompare(b.id));
      if (!files.length) return;

      const svg = document.createElementNS(NS, 'svg');
      // The viewBox tracks the host's pixel size, so pan/zoom work in screen
      // pixels; the layout itself is LAYOUT_WIDTH units wide.
      svg.setAttribute('preserveAspectRatio', 'xMinYMin meet');
      svg.setAttribute('role', 'img');
      svg.setAttribute('aria-label', 'Interactive DiffGraph visualization');
      svg.setAttribute('tabindex', '0');
      // No SVG <title>: browsers show it as a native tooltip over every child,
      // hiding the graph's own tooltip. aria-label already names the graph.
      const viewport = document.createElementNS(NS, 'g');
      svg.appendChild(viewport);
      host.appendChild(svg);

      const tooltip = document.createElement('div');
      tooltip.className = 'graph-tooltip';
      tooltip.hidden = true;
      host.appendChild(tooltip);
      const state = { scale: 1, x: 0, y: 0, dragging: false, collapsed: new Set(), expanded: new Set(), collapsedModules: new Set(), showUnchanged: false, selected: null, wide: false };
      const DRAWN_KINDS = new Set(['calls', 'imports', 'inherits', 'implements']);
      const SYMBOL_RIGHT = 700;
      const FILE_RIGHT = 1128;
      const MODULE_RIGHT = 1152;
      const edgeIndex = new Map();
      // Colors live in the stylesheet so both light and dark themes apply.
      const changeKinds = new Set(['added', 'deleted', 'modified', 'unchanged', 'untracked', 'renamed']);
      const changeClass = (kind) => `ck-${changeKinds.has(kind) ? kind : 'unchanged'}`;
      const svgEl = (name, attrs = {}) => {
        const node = document.createElementNS(NS, name);
        Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
        return node;
      };
      const addText = (parent, value, x, y, cls) => {
        const node = svgEl('text', { x, y, class: cls || '' });
        node.textContent = value == null ? '' : String(value);
        parent.appendChild(node);
        return node;
      };
      const htmlEl = (tag, cls, text) => {
        const node = document.createElement(tag);
        if (cls) node.className = cls;
        if (text != null) node.textContent = String(text);
        return node;
      };
      // Optional inferred summary: untrusted model text, only ever set via textContent.
      const summaryOf = (item) => {
        const summary = item && item.summary;
        if (!summary || typeof summary.text !== 'string' || !summary.text.trim()) return null;
        const inference = (summary.evidence || []).find((entry) => entry && entry.kind === 'llm_inference');
        return { text: summary.text.trim().replace(/\\s+/g, ' '), model: inference && inference.model ? String(inference.model) : null };
      };
      const truncate = (value, max) => (value.length > max ? value.slice(0, max - 1) + '…' : value);
      // Keep the end of a long path (the file name matters most), cut at a '/'
      // where possible. Full paths stay in tooltips and the details panel.
      const truncatePath = (value, max) => {
        const text = String(value);
        if (text.length <= max) return text;
        const tail = text.slice(text.length - (max - 1));
        const slash = tail.indexOf('/');
        return '…' + (slash > 0 && slash < tail.length - 12 ? tail.slice(slash) : tail);
      };
      const summaryLine = (summary, max) => truncate(`AI · ${summary.text}`, max);

      const fileIndex = new Map(files.map((file) => [file.id, file]));
      const symbolIndex = new Map(symbols.map((symbol) => [symbol.id, symbol]));
      const nameOf = (id) => {
        const file = fileIndex.get(id);
        if (file) return file.path || id;
        const symbol = symbolIndex.get(id);
        if (symbol) return symbol.qualified_name || symbol.name || id;
        if (String(id).startsWith('module::')) { const module = moduleList.find((entry) => entry.id === id); return `module ${module ? module.name : id}`; }
        return id;
      };

      // Default view: changed symbols plus their containing declarations.
      // Call neighbours are not shown by default: one hub function can call,
      // or be called by, dozens of unchanged symbols. They are counted in the
      // per-file toggle so the change's dependency graph stays one click away.
      const changedSymbolIds = new Set(symbols.filter((symbol) => symbol.change_kind && symbol.change_kind !== 'unchanged').map((symbol) => symbol.id));
      // Import statements are real changes but low-signal; they wait behind ▾.
      // Context files (outside the diff) exist only for their callers, so those show too.
      const contextFileIds = new Set(files.filter((file) => file.change_kind === 'unchanged').map((file) => file.id));
      const relevantSymbolIds = new Set([...changedSymbolIds].filter((id) => symbolIndex.get(id).kind !== 'import'));
      symbols.forEach((symbol) => { if (contextFileIds.has(symbol.file_id)) relevantSymbolIds.add(symbol.id); });
      [...relevantSymbolIds].forEach((id) => {
        let parent = symbolIndex.get(id).parent_id;
        while (parent && symbolIndex.has(parent) && !relevantSymbolIds.has(parent)) { relevantSymbolIds.add(parent); parent = symbolIndex.get(parent).parent_id; }
      });
      const callNeighbourIds = new Set();
      relationships.forEach((relationship) => {
        if (relationship.kind !== 'calls') return;
        if (changedSymbolIds.has(relationship.source_id)) callNeighbourIds.add(relationship.target_id);
        if (changedSymbolIds.has(relationship.target_id)) callNeighbourIds.add(relationship.source_id);
      });

      // ---- Hover: a short label; full details live in the panel. ----
      const showTooltip = (text, event) => {
        tooltip.textContent = text;
        tooltip.hidden = false;
        const rect = host.getBoundingClientRect();
        tooltip.style.left = Math.max(8, event.clientX - rect.left + 12) + 'px';
        tooltip.style.top = Math.max(8, event.clientY - rect.top + 12) + 'px';
      };
      const hideTooltip = () => { tooltip.hidden = true; };
      const hoverText = (item) => {
        if (item.relationships) return edgeLabel(item);
        const summary = summaryOf(item);
        const head = `${item.path ? 'file' : item.kind || 'symbol'} · ${nameOf(item.id)} · ${item.change_kind || 'unknown'}`;
        return summary ? `${head}\\n${truncate(summary.text, 240)}` : head;
      };

      const callLine = (relationship) => {
        const site = (relationship.evidence || []).find((entry) => entry && entry.kind === 'call_site' && entry.line_start);
        return site ? site.line_start : null;
      };
      const edgeLabel = (edge) => {
        const rels = edge.relationships;
        const kinds = [...new Set(rels.map((r) => r.kind))].join('/');
        const ends = rels.length === 1 ? `${nameOf(rels[0].source_id)} → ${nameOf(rels[0].target_id)}` : `${nameOf(edge.from.id)} → ${nameOf(edge.to.id)}`;
        const lines = [...new Set(rels.map(callLine).filter(Boolean))].sort((a, b) => a - b);
        return `${rels.length} ${kinds}: ${ends}${lines.length ? `\\nat line${lines.length > 1 ? 's' : ''} ${lines.slice(0, 8).join(', ')}${lines.length > 8 ? ', …' : ''}` : ''}`;
      };

      // ---- Micro diff: exact before/after source embedded by the producer ----
      const splitLines = (text) => { const lines = String(text || '').split('\\n'); if (lines.length && lines[lines.length - 1] === '') lines.pop(); return lines; };
      const diffOps = (beforeSpan, afterSpan) => {
        const before = beforeSpan ? splitLines(beforeSpan.text) : [];
        const after = afterSpan ? splitLines(afterSpan.text) : [];
        const a0 = beforeSpan ? beforeSpan.line_start : 0;
        const b0 = afterSpan ? afterSpan.line_start : 0;
        const n = before.length, m = after.length;
        const ops = [];
        if (n * m > 4000000) {
          before.forEach((text, i) => ops.push({ op: '-', a: a0 + i, text }));
          after.forEach((text, j) => ops.push({ op: '+', b: b0 + j, text }));
          return ops;
        }
        const w = m + 1;
        const table = new Uint32Array((n + 1) * w);
        for (let i = n - 1; i >= 0; i -= 1) for (let j = m - 1; j >= 0; j -= 1) {
          table[i * w + j] = before[i] === after[j] ? table[(i + 1) * w + j + 1] + 1 : Math.max(table[(i + 1) * w + j], table[i * w + j + 1]);
        }
        let i = 0, j = 0;
        while (i < n && j < m) {
          if (before[i] === after[j]) { ops.push({ op: ' ', a: a0 + i, b: b0 + j, text: before[i] }); i += 1; j += 1; }
          else if (table[(i + 1) * w + j] >= table[i * w + j + 1]) { ops.push({ op: '-', a: a0 + i, text: before[i] }); i += 1; }
          else { ops.push({ op: '+', b: b0 + j, text: after[j] }); j += 1; }
        }
        for (; i < n; i += 1) ops.push({ op: '-', a: a0 + i, text: before[i] });
        for (; j < m; j += 1) ops.push({ op: '+', b: b0 + j, text: after[j] });
        return ops;
      };
      const parseUnified = (text) => {
        const ops = [];
        let a = 0, b = 0;
        let inHunk = false;
        splitLines(text).forEach((line) => {
          // File headers precede the first hunk; inside a hunk, '--- x' is a
          // removed line that starts with '--' (e.g. a SQL comment).
          if (!inHunk && (line.startsWith('--- ') || line.startsWith('+++ '))) return;
          const hunk = /^@@ -(\\d+)(?:,\\d+)? \\+(\\d+)(?:,\\d+)? @@/.exec(line);
          if (hunk) { inHunk = true; a = Number(hunk[1]); b = Number(hunk[2]); ops.push({ op: '@', text: line }); return; }
          const sign = line[0];
          const body = line.slice(1);
          if (sign === '-') { ops.push({ op: '-', a, text: body }); a += 1; }
          else if (sign === '+') { ops.push({ op: '+', b, text: body }); b += 1; }
          else if (sign === ' ') { ops.push({ op: ' ', a, b, text: body }); a += 1; b += 1; }
        });
        return ops;
      };
      const foldContext = (ops, context = 3) => {
        const keep = ops.map(() => false);
        ops.forEach((entry, index) => {
          if (entry.op === ' ') return;
          for (let k = Math.max(0, index - context); k <= Math.min(ops.length - 1, index + context); k += 1) keep[k] = true;
        });
        const rows = [];
        let run = [];
        const flush = () => { if (run.length) { rows.push(run.length > 1 ? { op: 'fold', ops: run } : run[0]); run = []; } };
        ops.forEach((entry, index) => { if (keep[index] || entry.op !== ' ') { flush(); rows.push(entry); } else run.push(entry); });
        flush();
        return rows;
      };
      const cell = (cls, text) => htmlEl('span', cls, text == null ? '' : text);
      const renderDiff = (ops, options = {}) => {
        const container = htmlEl('div', `diff ${state.wide ? 'diff-split' : 'diff-unified'}`);
        const paint = (rows) => {
          const out = [];
          const foldRow = (entry) => {
            const button = htmlEl('button', 'diff-fold', `⋯ ${entry.ops.length} unchanged lines`);
            button.type = 'button';
            button.addEventListener('click', () => { const index = rows.indexOf(entry); rows.splice(index, 1, ...entry.ops); container.replaceChildren(...paint(rows)); });
            return button;
          };
          if (!state.wide) {
            rows.forEach((entry) => {
              if (entry.op === 'fold') { out.push(foldRow(entry)); return; }
              if (entry.op === '@') { out.push(htmlEl('div', 'diff-hunk', entry.text)); return; }
              const row = htmlEl('div', `diff-row op-${entry.op === '+' ? 'add' : entry.op === '-' ? 'del' : 'eq'}`);
              row.append(cell('ln', entry.a), cell('ln', entry.b), cell('sign', entry.op === ' ' ? '' : entry.op), cell('code', entry.text));
              out.push(row);
            });
            return out;
          }
          for (let index = 0; index < rows.length;) {
            const entry = rows[index];
            if (entry.op === 'fold') { out.push(foldRow(entry)); index += 1; continue; }
            if (entry.op === '@') { out.push(htmlEl('div', 'diff-hunk', entry.text)); index += 1; continue; }
            if (entry.op === ' ') {
              const row = htmlEl('div', 'diff-row op-eq');
              row.append(cell('ln', entry.a), cell('code', entry.text), cell('ln', entry.b), cell('code', entry.text));
              out.push(row); index += 1; continue;
            }
            const dels = [], adds = [];
            while (index < rows.length && (rows[index].op === '-' || rows[index].op === '+')) { (rows[index].op === '-' ? dels : adds).push(rows[index]); index += 1; }
            for (let k = 0; k < Math.max(dels.length, adds.length); k += 1) {
              const left = dels[k], right = adds[k];
              const row = htmlEl('div', 'diff-row');
              row.append(
                cell(`ln${left ? ' op-del' : ''}`, left && left.a), cell(`code${left ? ' op-del' : ' empty'}`, left && left.text),
                cell(`ln${right ? ' op-add' : ''}`, right && right.b), cell(`code${right ? ' op-add' : ' empty'}`, right && right.text),
              );
              out.push(row);
            }
          }
          return out;
        };
        const rows = options.fold ? foldContext(ops) : ops.slice();
        const limit = options.limit || Infinity;
        if (rows.length > limit) {
          const visible = rows.slice(0, limit);
          container.replaceChildren(...paint(visible));
          const more = htmlEl('button', 'show-all', `Show all ${rows.length} diff lines`);
          more.type = 'button';
          more.addEventListener('click', () => { container.replaceChildren(...paint(rows)); });
          const wrapper = htmlEl('div');
          wrapper.append(container, more);
          return wrapper;
        }
        container.replaceChildren(...paint(rows));
        return container;
      };
      const diffSection = (heading, ops, options, note) => {
        if (!ops.length) return null;
        const wrapper = htmlEl('section', 'details-section');
        const head = htmlEl('div', 'diff-head');
        head.appendChild(htmlEl('h3', null, heading));
        const wide = htmlEl('button', 'diff-wide', state.wide ? '⤡ Narrow' : '⤢ Side by side');
        wide.type = 'button';
        wide.addEventListener('click', () => { state.wide = !state.wide; layout.classList.toggle('details-wide', state.wide); renderDetails(); });
        head.appendChild(wide);
        wrapper.appendChild(head);
        if (note) wrapper.appendChild(htmlEl('p', 'details-meta', note));
        wrapper.appendChild(renderDiff(ops, options));
        return wrapper;
      };
      const symbolDiff = (symbol) => {
        const source = symbol.source;
        if (!source || (!source.before && !source.after)) return null;
        const truncated = (source.before && source.before.truncated) || (source.after && source.after.truncated);
        return diffSection('Change', diffOps(source.before, source.after), { fold: true }, truncated ? 'Declaration truncated at the size limit.' : null);
      };
      const fileDiff = (file) => {
        if (!file.diff || !file.diff.unified) return null;
        return diffSection('Diff', parseUnified(file.diff.unified), { limit: 400 }, file.diff.truncated ? 'Diff truncated at the size limit.' : null);
      };

      // ---- Details panel ----
      const help = panel.querySelector('.details-help');
      const body = panel.querySelector('.details-body');
      const reveal = (id) => {
        const symbol = symbolIndex.get(id);
        const fileId = symbol ? symbol.file_id : id;
        state.collapsed.delete(fileId);
        state.collapsedModules.delete(moduleIdOf.get(fileId));
        if (symbol && !relevantSymbolIds.has(id)) state.expanded.add(fileId);
      };
      const linkTo = (id) => {
        if (!fileIndex.has(id) && !symbolIndex.has(id)) return htmlEl('code', null, id);
        const button = htmlEl('button', 'node-link', nameOf(id));
        button.type = 'button';
        button.title = nameOf(id);
        button.addEventListener('click', () => select(id));
        return button;
      };
      const section = (heading, ids) => {
        const unique = [...new Set(ids)];
        if (!unique.length) return null;
        const wrapper = htmlEl('section', 'details-section');
        wrapper.appendChild(htmlEl('h3', null, `${heading} (${unique.length})`));
        const list = htmlEl('ul');
        const fill = (ids) => list.replaceChildren(...ids.map((id) => { const item = htmlEl('li'); item.appendChild(linkTo(id)); return item; }));
        const limit = 12;
        fill(unique.slice(0, limit));
        wrapper.appendChild(list);
        if (unique.length > limit) {
          const more = htmlEl('button', 'show-all', `Show all ${unique.length}`);
          more.type = 'button';
          more.addEventListener('click', () => { fill(unique); more.remove(); });
          wrapper.appendChild(more);
        }
        return wrapper;
      };
      const evidenceSection = (evidence) => {
        const entries = (evidence || []).filter((entry) => entry && entry.kind !== 'structural_basis' && entry.kind !== 'llm_inference');
        if (!entries.length) return null;
        const wrapper = htmlEl('section', 'details-section');
        wrapper.appendChild(htmlEl('h3', null, 'Evidence'));
        entries.forEach((entry) => {
          const lines = entry.line_start ? `:${entry.line_start}${entry.line_end && entry.line_end !== entry.line_start ? `-${entry.line_end}` : ''}` : '';
          wrapper.appendChild(htmlEl('p', 'evidence-kind', `${entry.kind}${entry.file ? ` · ${entry.file}${lines}` : ''}`));
          if (entry.snippet) wrapper.appendChild(htmlEl('pre', 'evidence-snippet', entry.snippet));
        });
        return wrapper;
      };
      const summaryBlock = (item) => {
        const summary = summaryOf(item);
        if (!summary) return null;
        const block = htmlEl('div', 'ai-summary');
        block.appendChild(htmlEl('span', 'ai-badge', 'AI'));
        block.appendChild(htmlEl('span', 'ai-model', summary.model || 'summary'));
        block.appendChild(htmlEl('p', null, summary.text));
        return block;
      };
      const header = (kindLabel, name, changeKind) => {
        const wrapper = htmlEl('header', 'details-header');
        wrapper.appendChild(htmlEl('p', 'details-kind', kindLabel));
        wrapper.appendChild(htmlEl('h2', null, name));
        if (changeKind) wrapper.appendChild(htmlEl('span', `badge ${changeClass(changeKind)}`, changeKind));
        return wrapper;
      };
      const incoming = (id, kind) => relationships.filter((r) => r.target_id === id && r.kind === kind).map((r) => r.source_id);
      const outgoing = (id, kind) => relationships.filter((r) => r.source_id === id && r.kind === kind).map((r) => r.target_id);
      const renderDetails = () => {
        const selected = state.selected;
        if (!selected) { help.hidden = false; body.hidden = true; body.replaceChildren(); return; }
        const parts = [];
        if (selected.type === 'edge') {
          const edge = selected.value;
          parts.push(header(`${edge.relationships.length} relationship${edge.relationships.length === 1 ? '' : 's'}`, `${nameOf(edge.from.id)} → ${nameOf(edge.to.id)}`, null));
          edge.relationships.forEach((rel) => {
            const block = htmlEl('section', 'details-section');
            const ends = htmlEl('p', 'details-edge');
            ends.append(htmlEl('span', 'badge', rel.kind), document.createTextNode(' '), linkTo(rel.source_id), htmlEl('span', 'edge-arrow-text', ' → '), linkTo(rel.target_id));
            block.appendChild(ends);
            const meta = [rel.label, rel.resolution_method && `resolution: ${rel.resolution_method}`, callLine(rel) && `line ${callLine(rel)}`].filter(Boolean);
            if (meta.length) block.appendChild(htmlEl('p', 'details-meta', meta.join(' · ')));
            (rel.evidence || []).filter((entry) => entry && entry.snippet).forEach((entry) => block.appendChild(htmlEl('pre', 'evidence-snippet', entry.snippet)));
            parts.push(block);
          });
        } else if (fileIndex.has(selected.id)) {
          const file = fileIndex.get(selected.id);
          parts.push(header(contextFileIds.has(file.id) ? 'file outside the diff' : 'file', file.path || file.id, contextFileIds.has(file.id) ? null : file.change_kind));
          if (contextFileIds.has(file.id)) parts.push(htmlEl('p', 'details-meta', 'Unchanged. Shown because it calls or imports changed code.'));
          const stats = [];
          if (file.old_path && file.old_path !== file.path) stats.push(`renamed from ${file.old_path}`);
          if (file.language) stats.push(file.language);
          if (file.lines_added != null || file.lines_removed != null) stats.push(`+${file.lines_added || 0} / -${file.lines_removed || 0}`);
          const fileSymbols = symbols.filter((symbol) => symbol.file_id === file.id);
          if (fileSymbols.length) stats.push(`${fileSymbols.filter((symbol) => changedSymbolIds.has(symbol.id)).length} of ${fileSymbols.length} symbols changed`);
          parts.push(htmlEl('p', 'details-meta', stats.join(' · ')));
          parts.push(summaryBlock(file));
          parts.push(section('Changed symbols', fileSymbols.filter((symbol) => changedSymbolIds.has(symbol.id)).map((symbol) => symbol.id)));
          parts.push(section('Imported by', incoming(file.id, 'imports').concat(...fileSymbols.map((symbol) => incoming(symbol.id, 'imports'))).filter((id) => id !== file.id)));
          parts.push(section('Imports', outgoing(file.id, 'imports').filter((id) => fileIndex.has(id) || (symbolIndex.has(id) && symbolIndex.get(id).file_id !== file.id))));
          parts.push(fileDiff(file));
        } else {
          const symbol = symbolIndex.get(selected.id);
          parts.push(header(symbol.kind || 'symbol', symbol.qualified_name || symbol.name || symbol.id, symbol.change_kind));
          const location = symbol.location ? `${symbol.location.file}:${symbol.location.line_start}-${symbol.location.line_end}` : nameOf(symbol.file_id);
          parts.push(htmlEl('p', 'details-meta', location));
          parts.push(summaryBlock(symbol));
          if (symbol.source) parts.push(symbolDiff(symbol));
          else if (!changedSymbolIds.has(symbol.id)) parts.push(htmlEl('p', 'details-meta', 'Unchanged in this snapshot.'));
          if (symbol.parent_id) parts.push(section('Defined in', [symbol.parent_id]));
          parts.push(section('Inherits from', outgoing(symbol.id, 'inherits')));
          parts.push(section('Subclassed by', incoming(symbol.id, 'inherits')));
          parts.push(section('Calls', outgoing(symbol.id, 'calls')));
          parts.push(section('Called by', incoming(symbol.id, 'calls')));
          parts.push(section('Imported by', incoming(symbol.id, 'imports').filter((id) => id !== symbol.file_id)));
          parts.push(evidenceSection(symbol.evidence));
        }
        body.replaceChildren(...parts.filter(Boolean));
        help.hidden = true;
        body.hidden = false;
        layout.classList.remove('details-closed');
      };
      const select = (id) => {
        reveal(id);
        state.selected = { type: 'node', id };
        renderDetails();
        draw();
      };
      const selectEdge = (edge) => {
        state.selected = { type: 'edge', id: edge.key, value: edge };
        renderDetails();
        draw();
      };
      const clearSelection = () => { state.selected = null; renderDetails(); draw(); };
      const closeButton = panel.querySelector('.details-close');
      if (closeButton) closeButton.addEventListener('click', () => { layout.classList.add('details-closed'); clearSelection(); });
      document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && state.selected) clearSelection(); });

      const highlight = (item, on) => {
        if (item.relationships) return;
        (edgeIndex.get(item.id) || new Set()).forEach((edge) => edge.classList.toggle('hover', on));
      };
      const interactive = (node, item, onSelect) => {
        node.addEventListener('pointerenter', (event) => { highlight(item, true); showTooltip(hoverText(item), event); });
        node.addEventListener('pointermove', (event) => showTooltip(hoverText(item), event));
        node.addEventListener('pointerleave', () => { highlight(item, false); hideTooltip(); });
        node.addEventListener('click', (event) => { event.stopPropagation(); onSelect(); });
        node.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(); } });
      };
      const isSelected = (id) => Boolean(state.selected && state.selected.id === id);

      // ---- Reading order: top to bottom, callers above the code they call ----
      // Changed source comes first (the change's entry points at the very top),
      // then files outside the diff, then tests, which call everything.
      const fileOf = (id) => (symbolIndex.has(id) ? symbolIndex.get(id).file_id : id);
      const isTestPath = (path) => {
        const parts = String(path || '').split('/');
        const base = parts[parts.length - 1];
        return parts.slice(0, -1).some((part) => part === 'tests' || part === 'test') || base.startsWith('test_') || base.endsWith('_test.py');
      };
      const category = (file) => (isTestPath(file.path) ? 2 : file.change_kind === 'unchanged' ? 1 : 0);
      const dependencyEdges = relationships.filter((r) => DRAWN_KINDS.has(r.kind) && !(r.kind === 'imports' && fileOf(r.target_id) === r.source_id));
      // Kahn's algorithm: a node is placed once everything that depends on it
      // (its callers) is placed. A cycle releases the node with the fewest
      // unplaced callers. Ties use ``compare``.
      const topoOrder = (ids, pairs, compare) => {
        const idSet = new Set(ids);
        const callers = new Map(ids.map((id) => [id, new Set()]));
        const callees = new Map(ids.map((id) => [id, new Set()]));
        pairs.forEach(([from, to]) => {
          if (from === to || !idSet.has(from) || !idSet.has(to)) return;
          callees.get(from).add(to);
          callers.get(to).add(from);
        });
        const placed = new Set();
        const order = [];
        const pending = (id) => [...callers.get(id)].filter((from) => !placed.has(from)).length;
        const tie = (a, b) => compare(a, b, callees);
        while (order.length < ids.length) {
          const open = ids.filter((id) => !placed.has(id));
          const ready = open.filter((id) => pending(id) === 0);
          const next = (ready.length ? ready.sort(tie) : open.sort((a, b) => pending(a) - pending(b) || tie(a, b)))[0];
          placed.add(next);
          order.push(next);
        }
        return order;
      };
      const filePairs = dependencyEdges.map((r) => [fileOf(r.source_id), fileOf(r.target_id)]);
      const byOutdegreeThenPath = (a, b, callees) => callees.get(b).size - callees.get(a).size || nameOf(a).localeCompare(nameOf(b));
      const fileOrder = [0, 1, 2].flatMap((group) => topoOrder(files.filter((file) => category(file) === group).map((file) => file.id), filePairs, byOutdegreeThenPath));
      const fileRank = new Map(fileOrder.map((id, index) => [id, index]));
      const moduleNameOf = (file) => { const bits = String(file.path || file.id).split('/'); return bits.length > 1 ? bits.slice(0, -1).join('/') : '(root)'; };
      const moduleList = [];
      const moduleIdOf = new Map();
      // A module is a directory within one reading group (changed source,
      // outside the diff, tests), so grouping never pulls a later group above
      // an earlier one, and every module id is unique.
      const GROUP_SUFFIX = ['', ' · outside the diff', ' · tests'];
      fileOrder.forEach((id) => {
        const file = fileIndex.get(id);
        const directory = moduleNameOf(file);
        const group = category(file);
        let module = moduleList.find((entry) => entry.directory === directory && entry.group === group);
        if (!module) {
          const testsDirectory = directory.split('/').some((part) => part === 'tests' || part === 'test');
          const suffix = group === 2 && testsDirectory ? '' : GROUP_SUFFIX[group];
          module = { directory, group, name: `${directory}${suffix}`, id: `module::${group}::${directory}`, files: [] };
          moduleList.push(module);
        }
        module.files.push(file);
        moduleIdOf.set(file.id, module.id);
      });
      // Many modules start collapsed, except the first: there is always a place to start.
      if (moduleList.length >= 5) moduleList.slice(1).forEach((module) => state.collapsedModules.add(module.id));
      const lineOf = (symbol) => (symbol.location && symbol.location.line_start) || ((symbol.evidence || [])[0] || {}).line_start || 0;
      const topOf = (id) => { let current = symbolIndex.get(id); while (current && current.parent_id && symbolIndex.has(current.parent_id)) current = symbolIndex.get(current.parent_id); return current ? current.id : id; };
      const symbolRank = new Map();
      files.forEach((file) => {
        const own = symbols.filter((symbol) => symbol.file_id === file.id);
        const tops = own.filter((symbol) => !symbol.parent_id || !symbolIndex.has(symbol.parent_id)).map((symbol) => symbol.id);
        const pairs = dependencyEdges.filter((r) => symbolIndex.has(r.source_id) && symbolIndex.has(r.target_id) && fileOf(r.source_id) === file.id && fileOf(r.target_id) === file.id).map((r) => [topOf(r.source_id), topOf(r.target_id)]);
        // Ties favour symbols that call more of the file (entry points), then source order.
        const byLine = (a, b, callees) => callees.get(b).size - callees.get(a).size || lineOf(symbolIndex.get(a)) - lineOf(symbolIndex.get(b));
        let rank = 0;
        topoOrder(tops, pairs, byLine).forEach((topId) => {
          symbolRank.set(topId, rank++);
          own.filter((symbol) => symbol.id !== topId && topOf(symbol.id) === topId).sort((a, b) => lineOf(a) - lineOf(b)).forEach((symbol) => symbolRank.set(symbol.id, rank++));
        });
      });
      const startFileId = fileOrder.length ? fileOrder[0] : null;

      // ---- Graph toolbar ----
      const toolbar = htmlEl('div', 'graph-toolbar');
      const tool = (label, onClick) => { const button = htmlEl('button', 'tool', label); button.type = 'button'; button.addEventListener('click', onClick); toolbar.appendChild(button); return button; };
      const unchangedButton = tool('Show unchanged', () => {
        state.showUnchanged = !state.showUnchanged;
        unchangedButton.textContent = state.showUnchanged ? 'Hide unchanged' : 'Show unchanged';
        unchangedButton.setAttribute('aria-pressed', String(state.showUnchanged));
        draw();
      });
      unchangedButton.setAttribute('aria-pressed', 'false');
      if (moduleList.length > 1) {
        tool('Expand modules', () => { state.collapsedModules.clear(); draw(); });
        tool('Collapse modules', () => { moduleList.forEach((module) => state.collapsedModules.add(module.id)); draw(); });
      }
      host.appendChild(toolbar);
      const setTransform = () => viewport.setAttribute('transform', `translate(${state.x} ${state.y}) scale(${state.scale})`);

      // ---- Viewport: fit width by default, fit everything on request, zoom at the cursor ----
      const LAYOUT_WIDTH = 1400;
      const MIN_SCALE = 0.1;
      const MAX_SCALE = 3;
      // Fits may go below the wheel minimum (tall or very narrow graphs).
      const FIT_FLOOR = 0.005;
      const clampScale = (value) => Math.max(MIN_SCALE, Math.min(MAX_SCALE, value));
      const fitScale = (value) => Math.max(FIT_FLOOR, Math.min(1.25, value));
      const hostSize = () => ({ width: Math.max(1, host.clientWidth), height: Math.max(1, host.clientHeight) });
      // The toolbar wraps on narrow panels; reserve its real height.
      const toolbarSpace = () => toolbar.offsetTop + toolbar.offsetHeight + 8;
      const syncViewBox = () => { const { width, height } = hostSize(); svg.setAttribute('viewBox', `0 0 ${width} ${height}`); };
      // Drawn bounds, so long paths or names wider than the nominal layout fit too.
      const drawnBox = () => {
        try { const box = viewport.getBBox(); if (box.width && box.height) return box; } catch (_) { /* not rendered yet */ }
        return { x: 0, y: 0, width: LAYOUT_WIDTH, height: 0 };
      };
      const fitWidth = () => {
        const { width } = hostSize();
        const box = drawnBox();
        const right = Math.max(LAYOUT_WIDTH, box.x + box.width);
        state.scale = fitScale((width - 16) / right);
        state.x = 8;
        state.y = toolbarSpace();
        setTransform();
      };
      // The scale at which the whole drawn graph fits the panel.
      const wholeGraphScale = () => {
        const box = drawnBox();
        if (!box.height) return MIN_SCALE;
        const { width, height } = hostSize();
        return fitScale(Math.min((width - 24) / box.width, (height - toolbarSpace() - 12) / box.height));
      };
      // Zoom out until the whole graph is visible (or to the wheel minimum,
      // whichever is smaller), never further: the graph cannot vanish, and a
      // narrow width fit can still zoom out to show everything.
      const zoomFloor = () => Math.min(MIN_SCALE, wholeGraphScale());
      const fitAll = () => {
        const box = drawnBox();
        if (!box.height) { fitWidth(); return; }
        const { width } = hostSize();
        const top = toolbarSpace();
        state.scale = wholeGraphScale();
        state.x = (width - box.width * state.scale) / 2 - box.x * state.scale;
        state.y = top - box.y * state.scale;
        state.userMoved = true;
        setTransform();
      };
      tool('Fit to view', fitAll);
      tool('Fit width', () => { state.userMoved = false; fitWidth(); });
      const draw = () => {
        viewport.replaceChildren();
        const positions = new Map();
        let y = 8;
        moduleList.forEach((module) => {
          const moduleY = y;
          const group = module.files;
          const moduleGroup = svgEl('g', { class: 'module-container' });
          viewport.appendChild(moduleGroup);
          const moduleCollapsed = state.collapsedModules.has(module.id);
          const moduleToggle = svgEl('g', { class: 'module-toggle', tabindex: '0', role: 'button', 'aria-expanded': String(!moduleCollapsed), 'aria-label': `${moduleCollapsed ? 'Expand' : 'Collapse'} module ${module.name}` });
          moduleToggle.appendChild(svgEl('rect', { x: 28, y: y + 4, width: 1120, height: 26, rx: 6, class: 'toggle-hit' }));
          addText(moduleToggle, `${moduleCollapsed ? '▶' : '▼'}  module: ${truncatePath(module.name, 120)}`, 34, y + 22, 'module-label');
          if (moduleCollapsed) {
            const changedCount = symbols.filter((symbol) => moduleIdOf.get(symbol.file_id) === module.id && changedSymbolIds.has(symbol.id)).length;
            const outside = group.filter((file) => file.change_kind === 'unchanged').length;
            const summaryText = `${group.length} file${group.length === 1 ? '' : 's'} · ${changedCount} changed symbol${changedCount === 1 ? '' : 's'}${outside ? ` · ${outside} outside the diff` : ''}`;
            addText(moduleToggle, summaryText, 34, y + 44, 'change-label');
          }
          const flipModule = (event) => { event.stopPropagation(); if (state.collapsedModules.has(module.id)) state.collapsedModules.delete(module.id); else state.collapsedModules.add(module.id); draw(); };
          moduleToggle.addEventListener('click', flipModule);
          moduleToggle.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); flipModule(event); } });
          moduleGroup.appendChild(moduleToggle);
          if (moduleCollapsed) {
            positions.set(module.id, { y: y + 22, visible: true });
            group.forEach((file) => { positions.set(file.id, { visible: false }); });
            y += 60;
            const outline = svgEl('rect', { x: 24, y: moduleY, width: 1128, height: 56, rx: 14, class: 'module-outline collapsed' });
            moduleGroup.insertBefore(outline, moduleGroup.firstChild);
            y += 18;
            return;
          }
          y += 34;
          group.forEach((file) => {
            const fileSymbols = symbols.filter((symbol) => symbol.file_id === file.id);
            const collapsed = state.collapsed.has(file.id);
            const fileSummary = summaryOf(file);
            const headerHeight = fileSummary ? 66 : 44;
            const expanded = state.expanded.has(file.id);
            const hiddenSymbols = fileSymbols.filter((symbol) => !relevantSymbolIds.has(symbol.id));
            const hiddenCount = hiddenSymbols.length;
            const relatedCount = hiddenSymbols.filter((symbol) => callNeighbourIds.has(symbol.id)).length;
            const showAll = expanded || state.showUnchanged;
            const shownSymbols = (showAll ? fileSymbols : fileSymbols.filter((symbol) => relevantSymbolIds.has(symbol.id)))
              .sort((a, b) => (symbolRank.get(a.id) || 0) - (symbolRank.get(b.id) || 0));
            const symbolRows = shownSymbols.map((symbol) => ({ symbol, summary: summaryOf(symbol) }));
            const moreHeight = hiddenCount && !state.showUnchanged ? 34 : 0;
            const symbolsHeight = symbolRows.reduce((total, row) => total + (row.summary ? 54 : 38), 0) + moreHeight;
            const height = collapsed || !fileSymbols.length ? headerHeight + 12 : Math.max(headerHeight + 28, headerHeight + symbolsHeight);
            const groupNode = svgEl('g', { class: 'file-container', tabindex: '0', role: 'button', 'aria-label': `Details for ${file.path || file.id}` });
            const context = contextFileIds.has(file.id);
            groupNode.appendChild(svgEl('rect', { x: 48, y, width: 1080, height, rx: 10, class: `file-box ${changeClass(file.change_kind)}${context ? ' context' : ''}${isSelected(file.id) ? ' selected' : ''}` }));
            addText(groupNode, truncatePath(file.path || file.id, file.id === startFileId ? 84 : 96), 88, y + 29, 'file-label');
            if (file.id === startFileId) addText(groupNode, '◆ start here', 880, y + 29, 'start-label');
            addText(groupNode, context ? 'outside diff' : file.change_kind || 'unknown', context ? 1000 : 1014, y + 29, 'change-label');
            if (fileSummary) addText(groupNode, summaryLine(fileSummary, 150), 88, y + 52, 'summary-label');
            interactive(groupNode, file, () => select(file.id));
            viewport.appendChild(groupNode);
            if (fileSymbols.length) {
              const toggle = svgEl('g', { class: 'collapse-toggle', tabindex: '0', role: 'button', 'aria-expanded': String(!collapsed), 'aria-label': `${collapsed ? 'Expand' : 'Collapse'} ${file.path || file.id}` });
              toggle.appendChild(svgEl('rect', { x: 56, y: y + 12, width: 24, height: 24, rx: 5, class: 'toggle-hit' }));
              addText(toggle, collapsed ? '▶' : '▼', 63, y + 29, 'toggle');
              const flip = (event) => { event.stopPropagation(); if (state.collapsed.has(file.id)) state.collapsed.delete(file.id); else state.collapsed.add(file.id); draw(); };
              toggle.addEventListener('click', flip);
              toggle.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); flip(event); } });
              viewport.appendChild(toggle);
            }
            positions.set(file.id, { x: 92, y: y + 29, visible: true });
            let sy = y + headerHeight + 4;
            if (!collapsed) symbolRows.forEach(({ symbol, summary }) => {
              const boxHeight = summary ? 44 : 28;
              const node = svgEl('g', { class: 'symbol-node', tabindex: '0', role: 'button', 'aria-label': `Details for ${symbol.kind || 'symbol'} ${symbol.qualified_name || symbol.name || symbol.id}` });
              node.appendChild(svgEl('rect', { x: 80, y: sy, width: 620, height: boxHeight, rx: 7, class: `symbol-box ${changeClass(symbol.change_kind)}${isSelected(symbol.id) ? ' selected' : ''}` }));
              addText(node, truncate(`${symbol.kind || 'symbol'}: ${symbol.qualified_name || symbol.name || symbol.id}`, 78), 94, sy + 19, 'symbol-label');
              if (summary) addText(node, summaryLine(summary, 84), 94, sy + 36, 'summary-label');
              node.appendChild(svgEl('circle', { cx: 678, cy: sy + 14, r: 5, class: `change-dot ${changeClass(symbol.change_kind)}` }));
              interactive(node, symbol, () => select(symbol.id));
              viewport.appendChild(node);
              positions.set(symbol.id, { x: 390, y: sy + 14, visible: true });
              sy += boxHeight + 10;
            });
            if (!collapsed && hiddenCount && !state.showUnchanged) {
              const more = svgEl('g', { class: 'more-toggle', tabindex: '0', role: 'button', 'aria-expanded': String(expanded) });
              const related = relatedCount ? ` · ${relatedCount} call or called by changed code` : '';
              const label = expanded ? `▴ Show fewer (hide ${hiddenCount})` : `▾ ${hiddenCount} more symbol${hiddenCount === 1 ? '' : 's'} (unchanged or imports)${related}`;
              more.setAttribute('aria-label', label);
              more.appendChild(svgEl('rect', { x: 80, y: sy, width: 620, height: 24, rx: 7, class: 'more-box' }));
              addText(more, label, 94, sy + 16, 'more-label');
              const toggleMore = (event) => {
                event.stopPropagation();
                if (state.expanded.has(file.id)) state.expanded.delete(file.id); else state.expanded.add(file.id);
                draw();
              };
              more.addEventListener('click', toggleMore);
              more.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggleMore(event); } });
              viewport.appendChild(more);
            }
            fileSymbols.forEach((symbol) => { if (collapsed || !positions.has(symbol.id)) positions.set(symbol.id, { visible: false }); });
            y += height + 14;
          });
          const moduleHeight = y - moduleY - 14;
          const outline = svgEl('rect', { x: 24, y: moduleY, width: 1128, height: moduleHeight, rx: 14, class: 'module-outline' });
          moduleGroup.insertBefore(outline, moduleGroup.firstChild);
          y += 18;
        });
        // Edges are drawn last (on top) as arcs in the empty space right of the
        // symbol rows (symbol ↔ symbol) or right of the file boxes (anything
        // touching a file header), so boxes never hide them. An endpoint hidden
        // by a collapsed file or the ▾ toggle is lifted to its file header, and
        // parallel edges between the same two anchors merge into one with a count.
        // Lift an endpoint to the nearest visible anchor: symbol, file, module.
        const anchorOf = (id) => {
          const pos = positions.get(id);
          if (pos && pos.visible) return { id, y: pos.y, x: symbolIndex.has(id) ? SYMBOL_RIGHT : FILE_RIGHT, symbol: symbolIndex.has(id) };
          const fileId = fileOf(id);
          const filePos = fileId !== id ? positions.get(fileId) : null;
          if (filePos && filePos.visible) return { id: fileId, y: filePos.y, x: FILE_RIGHT, symbol: false };
          const moduleId = moduleIdOf.get(fileId);
          const modulePos = moduleId && positions.get(moduleId);
          return modulePos && modulePos.visible ? { id: moduleId, y: modulePos.y, x: MODULE_RIGHT, symbol: false } : null;
        };
        const merged = new Map();
        relationships.forEach((relationship) => {
          if (!DRAWN_KINDS.has(relationship.kind)) return;
          // A file's edge to its own import statement describes the import, not a dependency.
          if (relationship.kind === 'imports' && fileOf(relationship.target_id) === relationship.source_id) return;
          const from = anchorOf(relationship.source_id);
          const to = anchorOf(relationship.target_id);
          if (!from || !to || from.id === to.id) return;
          const key = `${from.id} -> ${to.id}`;
          if (!merged.has(key)) merged.set(key, { key, from, to, relationships: [] });
          merged.get(key).relationships.push(relationship);
        });
        const touches = (edge, id) => [edge.from.id, edge.to.id].some((end) => end === id || fileOf(end) === id)
          || edge.relationships.some((r) => r.source_id === id || r.target_id === id);
        const edges = svgEl('g', { class: 'relationship-edges' });
        edgeIndex.clear();
        [...merged.values()].sort((a, b) => a.key.localeCompare(b.key)).forEach((edge) => {
          const inner = edge.from.symbol && edge.to.symbol;
          const x = Math.max(edge.from.x, edge.to.x);
          const dy = Math.abs(edge.to.y - edge.from.y);
          const bulge = inner ? Math.min(290, 28 + dy * 0.35) : Math.min(220, 28 + dy * 0.3);
          const d = `M ${edge.from.x} ${edge.from.y} C ${x + bulge} ${edge.from.y}, ${x + bulge} ${edge.to.y}, ${edge.to.x} ${edge.to.y}`;
          const kinds = new Set(edge.relationships.map((r) => r.kind));
          const classes = ['edge', kinds.has('calls') ? 'kind-calls' : kinds.has('inherits') || kinds.has('implements') ? 'kind-inherits' : 'kind-imports'];
          if (isSelected(edge.key)) classes.push('selected');
          else if (state.selected && state.selected.type === 'node') classes.push(touches(edge, state.selected.id) ? 'active' : 'dim');
          const group = svgEl('g', { class: classes.join(' '), tabindex: '0', role: 'button', 'aria-label': edgeLabel(edge) });
          group.appendChild(svgEl('path', { d, class: 'edge-hit' }));
          group.appendChild(svgEl('path', { d, class: 'edge-line', 'marker-end': 'url(#arrow)' }));
          if (edge.relationships.length > 1) addText(group, String(edge.relationships.length), x + bulge * 0.75 + 3, (edge.from.y + edge.to.y) / 2 + 4, 'edge-count');
          interactive(group, edge, () => selectEdge(edge));
          edges.appendChild(group);
          [edge.from.id, edge.to.id, fileOf(edge.from.id), fileOf(edge.to.id)].forEach((id) => {
            if (!edgeIndex.has(id)) edgeIndex.set(id, new Set());
            edgeIndex.get(id).add(group);
          });
        });
        const defs = svgEl('defs');
        const marker = svgEl('marker', { id: 'arrow', markerWidth: 8, markerHeight: 8, refX: 7, refY: 3, orient: 'auto' });
        marker.appendChild(svgEl('path', { d: 'M0,0 L0,6 L8,3 z', class: 'edge-arrow' }));
        defs.appendChild(marker);
        edges.insertBefore(defs, edges.firstChild);
        viewport.appendChild(edges);
        // Expanding or collapsing changes the drawn bounds; keep the fit until
        // the reader zooms or pans.
        if (state.userMoved) setTransform(); else fitWidth();
      };
      svg.addEventListener('wheel', (event) => {
        event.preventDefault();
        // Keep the layout point under the cursor fixed while zooming.
        const rect = svg.getBoundingClientRect();
        const px = event.clientX - rect.left;
        const py = event.clientY - rect.top;
        const factor = event.deltaY < 0 ? 1.12 : 0.89;
        const next = Math.max(zoomFloor(), Math.min(MAX_SCALE, state.scale * factor));
        state.x = px - (px - state.x) * (next / state.scale);
        state.y = py - (py - state.y) * (next / state.scale);
        state.scale = next;
        state.userMoved = true;
        setTransform();
      }, { passive: false });
      svg.addEventListener('pointerdown', (event) => { if (event.target === svg) { state.dragging = true; state.startX = event.clientX - state.x; state.startY = event.clientY - state.y; svg.setPointerCapture(event.pointerId); } });
      svg.addEventListener('pointermove', (event) => { if (state.dragging) { state.x = event.clientX - state.startX; state.y = event.clientY - state.startY; state.userMoved = true; setTransform(); } });
      svg.addEventListener('pointerup', () => { state.dragging = false; });
      syncViewBox();
      fitWidth();
      draw();
      // Resizing (window, details panel, side-by-side diff) keeps the fit
      // until the reader zooms or pans themselves.
      if (typeof ResizeObserver === 'function') {
        new ResizeObserver(() => { syncViewBox(); if (!state.userMoved) fitWidth(); }).observe(host);
      }
    })();
    """
).strip()


def _script_csp_hash(script: str) -> str:
    digest = hashlib.sha256(script.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


class HtmlFormatter:
    """Render one already-validated DiffGraph artifact as self-contained HTML."""

    def __init__(self, artifact: ValidatedArtifact):
        if not isinstance(artifact, ValidatedArtifact):
            raise TypeError("HtmlFormatter requires a ValidatedArtifact")
        self.artifact = artifact
        self.dg = artifact.value

    def render(self) -> str:
        """Return deterministic HTML: the graph plus a compact snapshot bar."""
        files = self.dg["files"]
        symbols = self.dg["symbols"]
        metadata = self.dg["metadata"]
        self._validate_unique_object_ids(files, symbols)
        script_hash = _script_csp_hash(_INTERACTIVE_SCRIPT)
        topbar = self._topbar(files, symbols, metadata)
        overall = self._overall_summary()
        empty = (
            '<p class="graph-empty">No changes in the selected snapshot.</p>'
            if not files
            else ""
        )

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'sha256-{script_hash}'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
  <title>wild diff — {_text(_diff_label(self.dg["diff_ref"]))}</title>
  <style>
    :root {{
      color-scheme: light dark; font-family: system-ui, sans-serif;
      --page-bg: #ffffff; --graph-bg: #f8fafc; --node-bg: #ffffff; --symbol-bg: #f2f4f7;
      --text-strong: #101828; --text: #344054; --text-muted: #475467;
      --edge: #475467; --outline: #98a2b3; --focus: #101828; --selected: #7a5af8;
      --tooltip-bg: #101828; --tooltip-text: #f9fafb; --tooltip-border: #667085;
      --ck-added: #18794e; --ck-deleted: #b42318; --ck-modified: #175cd3;
      --ck-unchanged: #667085; --ck-untracked: #7a5af8; --ck-renamed: #b54708;
      --diff-add-bg: #e3f6e8; --diff-del-bg: #fde8e7; --diff-hunk-bg: #eef2f6;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --page-bg: #0c111d; --graph-bg: #0c111d; --node-bg: #161b26; --symbol-bg: #1f242f;
        --text-strong: #f5f5f6; --text: #cecfd2; --text-muted: #94969c;
        --edge: #94969c; --outline: #61646c; --focus: #f5f5f6; --selected: #bdb4fe;
        --tooltip-bg: #1f242f; --tooltip-text: #f5f5f6; --tooltip-border: #61646c;
        --ck-added: #47cd89; --ck-deleted: #f97066; --ck-modified: #84adff;
        --ck-unchanged: #94969c; --ck-untracked: #bdb4fe; --ck-renamed: #fdb022;
        --diff-add-bg: #113a22; --diff-del-bg: #431816; --diff-hunk-bg: #1b2433;
      }}
    }}
    html, body {{ height: 100%; margin: 0; }}
    body {{ display: flex; flex-direction: column; background: var(--page-bg); color: var(--text); line-height: 1.4; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
    .topbar {{ display: flex; flex-wrap: wrap; gap: .5rem 1rem; align-items: center; padding: .55rem 1rem; border-bottom: 1px solid var(--outline); background: var(--node-bg); font-size: .875rem; }}
    .topbar .brand {{ font-weight: 700; color: var(--text-strong); }}
    .topbar .range {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; color: var(--text-strong); }}
    .topbar .spacer {{ flex: 1; }}
    .badge {{ display: inline-block; padding: .05rem .55rem; border: 1px solid var(--outline); border-radius: 999px; font-size: .78rem; white-space: nowrap; }}
    .badge.ck-added {{ color: var(--ck-added); border-color: var(--ck-added); }}
    .badge.ck-deleted {{ color: var(--ck-deleted); border-color: var(--ck-deleted); }}
    .badge.ck-modified {{ color: var(--ck-modified); border-color: var(--ck-modified); }}
    .badge.ck-unchanged {{ color: var(--ck-unchanged); border-color: var(--ck-unchanged); }}
    .badge.ck-untracked {{ color: var(--ck-untracked); border-color: var(--ck-untracked); }}
    .badge.ck-renamed {{ color: var(--ck-renamed); border-color: var(--ck-renamed); }}
    .badge.ai {{ color: var(--selected); border-color: var(--selected); }}
    .popover {{ position: relative; }}
    .popover > summary {{ cursor: pointer; list-style: none; }}
    .popover > summary::-webkit-details-marker {{ display: none; }}
    .popover > .popover-body {{ position: absolute; right: 0; z-index: 3; width: min(32rem, 90vw); max-height: 60vh; overflow: auto; margin-top: .4rem; padding: .75rem 1rem; border: 1px solid var(--outline); border-radius: .5rem; background: var(--node-bg); box-shadow: 0 8px 24px #0003; }}
    .popover-body ul {{ margin: 0; padding-left: 1.1rem; }}
    .popover-body dl {{ display: grid; grid-template-columns: auto 1fr; gap: .2rem .8rem; margin: 0; }}
    .popover-body dt {{ color: var(--text-muted); }} .popover-body dd {{ margin: 0; overflow-wrap: anywhere; }}
    .overall-summary {{ margin: 0; padding: .45rem 1rem; border-bottom: 1px solid var(--outline); font-size: .875rem; }}
    .layout {{ flex: 1; display: grid; grid-template-columns: minmax(0, 1fr) 22rem; min-height: 0; }}
    .layout.details-wide {{ grid-template-columns: minmax(0, 1fr) min(64rem, 58%); }}
    .layout.details-closed {{ grid-template-columns: minmax(0, 1fr) 0; }}
    .layout.details-closed #diffgraph-details {{ display: none; }}
    #diffgraph-graph {{ position: relative; min-height: 0; overflow: hidden; background: var(--graph-bg); }}
    #diffgraph-graph svg {{ display: block; width: 100%; height: 100%; cursor: grab; }}
    #diffgraph-graph svg:active {{ cursor: grabbing; }}
    .graph-empty {{ margin: 3rem auto; text-align: center; color: var(--text-muted); }}
    #diffgraph-details {{ position: relative; overflow: auto; padding: 1rem 1.1rem; border-left: 1px solid var(--outline); background: var(--node-bg); font-size: .875rem; }}
    .details-close {{ position: absolute; top: .5rem; right: .6rem; border: 0; background: none; color: var(--text-muted); font-size: 1.1rem; cursor: pointer; }}
    .details-help h2 {{ margin: 0 0 .5rem; font-size: 1rem; color: var(--text-strong); }}
    .details-help ul {{ padding-left: 1.1rem; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: .35rem; }}
    .details-header h2 {{ margin: .1rem 0 .4rem; font-size: 1rem; color: var(--text-strong); overflow-wrap: anywhere; }}
    .details-kind {{ margin: 0; color: var(--text-muted); text-transform: uppercase; font-size: .7rem; letter-spacing: .04em; }}
    .details-meta {{ margin: .45rem 0; color: var(--text-muted); overflow-wrap: anywhere; }}
    .details-section h3 {{ margin: 1rem 0 .3rem; font-size: .8rem; color: var(--text-strong); }}
    .details-section ul {{ margin: 0; padding-left: 1.1rem; }}
    .node-link {{ display: block; max-width: 100%; padding: 0; border: 0; background: none; color: var(--ck-modified); font: inherit; text-align: left; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .show-all {{ margin-top: .3rem; padding: .1rem .5rem; border: 1px solid var(--outline); border-radius: .3rem; background: none; color: var(--text-muted); font: inherit; font-size: .78rem; cursor: pointer; }}
    .details-edge .node-link {{ display: inline; white-space: normal; overflow-wrap: anywhere; }}
    .node-link:hover {{ text-decoration: underline; }}
    .ai-summary {{ margin: .6rem 0; padding: .55rem .7rem; border-left: 3px solid var(--selected); background: var(--symbol-bg); border-radius: .3rem; }}
    .ai-summary p {{ margin: .3rem 0 0; }}
    .ai-badge {{ display: inline-block; margin-right: .4rem; padding: 0 .35rem; border: 1px solid var(--selected); border-radius: .3rem; color: var(--selected); font-size: .7rem; font-weight: 700; }}
    .ai-model {{ color: var(--text-muted); font-size: .75rem; }}
    .evidence-kind {{ margin: .35rem 0 .15rem; color: var(--text-muted); font-size: .78rem; overflow-wrap: anywhere; }}
    .evidence-snippet {{ margin: 0 0 .4rem; padding: .4rem .5rem; overflow-x: auto; white-space: pre-wrap; background: var(--symbol-bg); border-radius: .3rem; font-size: .75rem; }}
    .module-label {{ font-size: 16px; font-weight: 700; fill: var(--text); }}
    .file-label {{ font-size: 16px; font-weight: 700; fill: var(--text-strong); }}
    .symbol-label {{ font-size: 14px; fill: var(--text); }}
    .change-label {{ font-size: 12px; fill: var(--text-muted); }}
    .summary-label {{ font-size: 12px; font-style: italic; fill: var(--text-muted); }}
    .toggle {{ font-size: 13px; fill: var(--text); pointer-events: none; }}
    .toggle-hit {{ fill: transparent; stroke: none; }}
    .collapse-toggle {{ cursor: pointer; outline: none; }}
    .collapse-toggle:hover .toggle-hit, .collapse-toggle:focus .toggle-hit {{ fill: var(--symbol-bg); }}
    .module-outline {{ fill: none; stroke: var(--outline); stroke-dasharray: 6 5; }}
    .file-box {{ fill: var(--node-bg); stroke: var(--ck-unchanged); stroke-width: 3; }}
    .symbol-box {{ fill: var(--symbol-bg); stroke: var(--ck-unchanged); stroke-width: 1; }}
    .change-dot {{ fill: var(--ck-unchanged); stroke: none; }}
    .ck-added {{ stroke: var(--ck-added); }} .change-dot.ck-added {{ fill: var(--ck-added); }}
    .ck-deleted {{ stroke: var(--ck-deleted); }} .change-dot.ck-deleted {{ fill: var(--ck-deleted); }}
    .ck-modified {{ stroke: var(--ck-modified); }} .change-dot.ck-modified {{ fill: var(--ck-modified); }}
    .ck-unchanged {{ stroke: var(--ck-unchanged); }} .change-dot.ck-unchanged {{ fill: var(--ck-unchanged); }}
    .ck-untracked {{ stroke: var(--ck-untracked); }} .change-dot.ck-untracked {{ fill: var(--ck-untracked); }}
    .ck-renamed {{ stroke: var(--ck-renamed); }} .change-dot.ck-renamed {{ fill: var(--ck-renamed); }}
    .file-box.context {{ stroke-width: 2; stroke-dasharray: 7 5; fill: var(--graph-bg); }}
    .file-box.selected, .symbol-box.selected {{ stroke: var(--selected); stroke-width: 3; }}
    .edge {{ cursor: pointer; outline: none; }}
    .edge-hit {{ fill: none; stroke: transparent; stroke-width: 12; pointer-events: stroke; }}
    .edge-line {{ fill: none; stroke: var(--edge); stroke-width: 1.6; opacity: .85; pointer-events: none; }}
    .edge.kind-imports .edge-line {{ stroke-dasharray: 6 4; }}
    .edge.kind-inherits .edge-line {{ stroke-dasharray: 2 3; }}
    .edge.dim {{ opacity: .14; }}
    .edge:hover .edge-line, .edge:focus .edge-line, .edge.hover .edge-line, .edge.active .edge-line, .edge.selected .edge-line {{ stroke: var(--selected); stroke-width: 2.8; opacity: 1; }}
    .edge-count {{ font-size: 11px; font-weight: 700; fill: var(--text-muted); pointer-events: none; }}
    .edge-arrow {{ fill: var(--edge); }}
    .diff-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: .5rem; }}
    .diff-wide {{ padding: .05rem .45rem; border: 1px solid var(--outline); border-radius: .3rem; background: none; color: var(--text-muted); font: inherit; font-size: .72rem; cursor: pointer; }}
    .diff {{ overflow-x: auto; border: 1px solid var(--outline); border-radius: .35rem; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .74rem; line-height: 1.45; }}
    .diff-row {{ display: grid; min-width: max-content; }}
    .diff-unified .diff-row {{ grid-template-columns: 3.2em 3.2em 1.2em 1fr; }}
    .diff-split .diff-row {{ min-width: 0; grid-template-columns: 3.2em minmax(0, 1fr) 3.2em minmax(0, 1fr); }}
    .diff.diff-split .code {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    .diff .ln {{ padding: 0 .4em; text-align: right; color: var(--text-muted); user-select: none; }}
    .diff .sign {{ color: var(--text-muted); user-select: none; }}
    .diff .code {{ padding-right: .8em; white-space: pre; }}
    .diff-row.op-add, .diff .op-add {{ background: var(--diff-add-bg); }}
    .diff-row.op-del, .diff .op-del {{ background: var(--diff-del-bg); }}
    .diff-hunk {{ padding: .1rem .5rem; background: var(--diff-hunk-bg); color: var(--text-muted); white-space: pre; }}
    .diff-fold {{ display: block; width: 100%; padding: .1rem .5rem; border: 0; background: var(--diff-hunk-bg); color: var(--text-muted); font: inherit; text-align: left; cursor: pointer; }}
    .file-container, .symbol-node {{ cursor: pointer; outline: none; }}
    .file-container:hover > rect, .symbol-node:hover > rect {{ stroke-width: 3; }}
    .file-container:focus > rect, .symbol-node:focus > rect {{ stroke: var(--focus); stroke-width: 3; }}
    .more-toggle {{ cursor: pointer; outline: none; }}
    .more-box {{ fill: none; stroke: var(--outline); stroke-dasharray: 4 4; }}
    .more-label {{ font-size: 13px; fill: var(--text-muted); }}
    .more-toggle:hover .more-label, .more-toggle:focus .more-label {{ fill: var(--text-strong); }}
    .more-toggle:focus > rect {{ stroke: var(--focus); stroke-dasharray: none; }}
    .graph-toolbar {{ position: absolute; top: .55rem; left: .6rem; right: .6rem; z-index: 2; display: flex; flex-wrap: wrap; gap: .35rem; pointer-events: none; }}
    .graph-toolbar .tool {{ pointer-events: auto; }}
    .tool {{ padding: .2rem .6rem; border: 1px solid var(--outline); border-radius: .35rem; background: var(--node-bg); color: var(--text); font: inherit; font-size: .78rem; cursor: pointer; }}
    .tool[aria-pressed="true"] {{ border-color: var(--selected); color: var(--selected); }}
    .module-toggle {{ cursor: pointer; outline: none; }}
    .module-toggle:hover .toggle-hit, .module-toggle:focus .toggle-hit {{ fill: var(--symbol-bg); }}
    .module-outline.collapsed {{ fill: var(--node-bg); }}
    .start-label {{ font-size: 12px; font-weight: 700; fill: var(--selected); }}
    .graph-tooltip {{ position: absolute; z-index: 2; max-width: min(28rem, 80%); margin: 0; padding: .45rem .6rem; white-space: pre-line; border: 1px solid var(--tooltip-border); border-radius: .4rem; background: var(--tooltip-bg); color: var(--tooltip-text); font-size: .75rem; pointer-events: none; }}
  </style>
</head>
<body>
  {topbar}
  {overall}
  <div class="layout" id="diffgraph-layout">
    <main id="diffgraph-graph" aria-label="Interactive graph">{empty}</main>
    <aside id="diffgraph-details" aria-label="Details">
      <button type="button" class="details-close" aria-label="Close details">×</button>
      <div class="details-help">
        <h2>Explore the change</h2>
        <ul>
          <li>Select a file, symbol, or edge to see its details and exact diff here.</li>
          <li>Scroll to zoom; drag empty space to pan.</li>
          <li>▶/▼ collapses a file; ▾/▴ shows or hides its unchanged symbols.</li>
          <li>Dashed files are outside the diff: they call or import changed code.</li>
          <li>Read top to bottom: code that calls is above the code it calls (◆ marks where to start); tests and files outside the diff come last.</li>
        </ul>
        <div class="legend">
          <span class="badge ck-added">added</span><span class="badge ck-modified">modified</span><span class="badge ck-deleted">deleted</span><span class="badge ck-renamed">renamed</span><span class="badge ck-unchanged">unchanged</span>
        </div>
        <p class="details-meta">Solid arcs are calls, dashed arcs imports, dotted arcs inheritance; a number counts merged edges. Hover a box to highlight its connections.</p>
      </div>
      <div class="details-body" hidden></div>
    </aside>
  </div>
  <script type="application/json" id="diffgraph-artifact">{_embedded_json(self.dg)}</script>
  <script>{_INTERACTIVE_SCRIPT}</script>
</body>
</html>
"""

    def _topbar(self, files: list[dict], symbols: list[dict], metadata: dict) -> str:
        changed_files = sum(1 for item in files if item.get("change_kind") != "unchanged")
        context_files = len(files) - changed_files
        context_note = (
            f" · {_plural(context_files, 'caller file')} outside the diff" if context_files else ""
        )
        counts = {"added": 0, "modified": 0, "deleted": 0}
        for symbol in symbols:
            if symbol.get("change_kind") in counts and symbol.get("kind") != "import":
                counts[symbol["change_kind"]] += 1
        symbol_counts = " · ".join(
            f"{count} {kind}" for kind, count in counts.items() if count
        ) or "no symbol changes"
        if metadata.get("privacy_tier") == "local":
            privacy = '<span class="badge" title="No data left this machine">Local only</span>'
        else:
            calls = metadata.get("llm_calls") or 0
            model = metadata.get("llm_model")
            label = f"AI: {model} · {_plural(calls, 'call')}" if model else str(metadata.get("privacy_tier"))
            providers = ", ".join(metadata.get("cloud_providers_used") or [])
            title = f"Code was sent to {providers}" if providers else "Data left this machine"
            privacy = f'<span class="badge ai" title="{_text(title)}">{_text(label)}</span>'
        raw_warnings = metadata.get("warnings", [])
        lines = warning_lines(raw_warnings)
        warnings = (
            '<details class="popover"><summary><span class="badge">⚠ '
            f"{_text(_plural(len(lines), 'warning'))}</span></summary>"
            '<div class="popover-body"><ul>'
            + "".join(f"<li>{_text(line)}</li>" for line in lines)
            + "</ul></div></details>"
            if lines
            else ""
        )
        about = (
            '<details class="popover"><summary><span class="badge">About</span></summary>'
            '<div class="popover-body"><dl>'
            f"<dt>Snapshot</dt><dd>{_text(_diff_label(self.dg['diff_ref']))}</dd>"
            f"<dt>Generated</dt><dd>{_text(self.dg['generated_at'])}</dd>"
            f"<dt>wild</dt><dd>{_text(self.dg['wild_version'])}</dd>"
            f"<dt>Schema</dt><dd>{_text(self.dg['schema_version'])}</dd>"
            "<dt>Raw data</dt><dd><code>wild diff --format json</code></dd>"
            "</dl></div></details>"
        )
        return (
            '<header class="topbar">'
            '<span class="brand">wild diff</span>'
            f'<span class="range">{_text(_diff_label(self.dg["diff_ref"]))}</span>'
            f"<span>{_text(_plural(changed_files, 'file'))} · {_text(symbol_counts)}{context_note}</span>"
            '<span class="spacer"></span>'
            f"{privacy}{warnings}{about}"
            "</header>"
        )

    def _overall_summary(self) -> str:
        summary = self.dg.get("summary")
        if not isinstance(summary, dict) or not str(summary.get("text") or "").strip():
            return ""
        return (
            '<p class="overall-summary"><span class="ai-badge">AI</span>'
            f"{_text(summary['text'])}</p>"
        )

    @staticmethod
    def _validate_unique_object_ids(files: list[dict], symbols: list[dict]) -> None:
        """Reject duplicate object IDs; the graph keys positions and selection by ID."""
        object_ids = [item["id"] for item in (*files, *symbols)]
        duplicate_ids = sorted(item_id for item_id in set(object_ids) if object_ids.count(item_id) > 1)
        if duplicate_ids:
            raise ValueError("DiffGraph HTML rendering requires unique file and symbol IDs; duplicates: " + ", ".join(duplicate_ids))

    def write(self, destination: Path) -> Path:
        """Atomically write a complete report and return its absolute path."""
        rendered = self.render()
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(rendered)
            os.replace(temporary_path, destination)
        except BaseException:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
            raise
        return destination.absolute()
