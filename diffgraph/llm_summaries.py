"""Summarize each changed file and symbol of a DiffGraph with an LLM.

The LLM only writes prose for nodes that already exist in a validated
artifact; it never adds, removes, or edits files, symbols, or relationships.
Source text is read at the exact Git identities recorded in the artifact
(working-tree content is re-verified against its recorded blob id), so the
model sees the same snapshot that was analyzed.

Network-capable via :mod:`diffgraph.relay`; import lazily, only after the
caller has opted into LLM output.
"""
from __future__ import annotations

import difflib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from diffgraph.contract import ValidatedArtifact
from diffgraph.git_snapshot import GitSnapshotError, read_worktree_blob, run_git
from diffgraph.structural import split_lines
from diffgraph.relay import MissingProviderKey, RelayAccountError, RelayError, RelayTimeout

PROMPT_REF = "wild-node-summaries-v1"
MAX_FILE_DIFF_CHARS = 12_000
MAX_BATCH_CHARS = 24_000
MAX_SUMMARY_CHARS = 300
MAX_WORKERS = 4

_SYSTEM_PROMPT = """You summarize code changes for a reviewer.
For every node id listed in the request, write one plain sentence (at most 25 words) \
describing what changed in that node and why it matters, based only on the diff shown. \
Do not speculate beyond the diff. Reply with only a JSON object mapping each node id \
to its sentence, e.g. {"file::a.py": "...", "sym::a.py::f": "..."}."""


class ChatClient(Protocol):
    """Any backend that turns chat messages into reply text.

    :class:`diffgraph.relay.RelayClient` is the shipped implementation; other
    backends only need this method and should raise :class:`RelayError`
    subclasses on failure so warnings stay scoped and typed.
    """

    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int = ...) -> str:
        ...


@dataclass
class _FileJob:
    file_id: str
    path: str
    targets: List[dict]
    diff: str

    def render(self) -> str:
        lines = [f"## File {self.path} ({self.targets[0]['change']})", "Nodes:"]
        for target in self.targets:
            lines.append(f"- {target['id']}: {target['label']}")
        lines.extend(["Diff:", self.diff or "(no textual diff)"])
        return "\n".join(lines)


@dataclass
class SummaryResult:
    summaries: Dict[str, str] = field(default_factory=dict)
    llm_calls: int = 0
    warnings: List[Dict[str, str]] = field(default_factory=list)


def _file_identity(file_entry: dict) -> Optional[dict]:
    for evidence in file_entry.get("evidence") or []:
        if evidence.get("kind") == "git_diff_name_status":
            try:
                identity = json.loads(evidence.get("detail", ""))
            except ValueError:
                return None
            return identity if isinstance(identity, dict) else None
    return None


def _side_text(root: str, identity: dict, side: str, worktree: bool) -> Optional[str]:
    oid = identity.get(f"{side}_oid")
    if oid is None or identity.get(f"{side}_mode") == "160000":
        return None
    if side == "new" and worktree:
        data = read_worktree_blob(
            root, identity["new_path"], identity.get("new_mode"), expected_oid=oid
        )
    else:
        data = run_git(root, "cat-file", "blob", oid)
    if b"\0" in data:
        raise ValueError("binary content")
    return data.decode("utf-8")


def _unified_diff(path: str, old: Optional[str], new: Optional[str]) -> str:
    diff = "".join(
        difflib.unified_diff(
            split_lines(old or ""),
            split_lines(new or ""),
            fromfile=f"a/{path}" if old is not None else "/dev/null",
            tofile=f"b/{path}" if new is not None else "/dev/null",
            n=3,
        )
    )
    if len(diff) > MAX_FILE_DIFF_CHARS:
        diff = diff[:MAX_FILE_DIFF_CHARS] + "\n... (diff truncated)\n"
    return diff


def _jobs(dg: dict) -> Tuple[List[_FileJob], List[Dict[str, str]]]:
    root = dg["diff_ref"].get("repo_root") or "."
    # The post-change side lives in the working tree (re-verified against its
    # recorded blob id) for unstaged diffs and for ``wild diff REF``.
    worktree = dg["diff_ref"].get("kind") in ("unstaged", "working_tree")
    symbols_by_file: Dict[str, List[dict]] = {}
    for symbol in dg["symbols"]:
        if symbol["change_kind"] != "unchanged" and symbol["kind"] != "import":
            symbols_by_file.setdefault(symbol["file_id"], []).append(symbol)

    jobs: List[_FileJob] = []
    warnings: List[Dict[str, str]] = []
    for file_entry in dg["files"]:
        if file_entry["change_kind"] == "unchanged":
            continue  # context file outside the diff: nothing changed to summarize
        path = file_entry["path"]
        identity = _file_identity(file_entry)
        if identity is None:
            continue
        try:
            old = _side_text(root, identity, "old", worktree)
            new = _side_text(root, identity, "new", worktree)
        except (GitSnapshotError, OSError, UnicodeDecodeError, ValueError) as error:
            # Binary, undecodable, or changed-since-analysis content is not
            # sent; the node keeps its structural data without prose.
            warnings.append({
                "code": "LLM_ERROR",
                "file": path,
                "detail": f"not summarized: content unavailable as text ({error})",
            })
            continue
        targets = [{
            "id": file_entry["id"],
            "label": f"file, {file_entry['change_kind']}",
            "change": file_entry["change_kind"],
        }]
        for symbol in symbols_by_file.get(file_entry["id"], []):
            location = symbol.get("location") or {}
            span = (
                f", lines {location['line_start']}-{location['line_end']}"
                if location else ""
            )
            targets.append({
                "id": symbol["id"],
                "label": (
                    f"{symbol['kind']} {symbol.get('qualified_name') or symbol['name']}, "
                    f"{symbol['change_kind']}{span}"
                ),
                "change": symbol["change_kind"],
            })
        jobs.append(_FileJob(file_entry["id"], path, targets, _unified_diff(path, old, new)))
    return jobs, warnings


def _batches(jobs: Sequence[_FileJob]) -> List[List[_FileJob]]:
    batches: List[List[_FileJob]] = []
    current: List[_FileJob] = []
    size = 0
    for job in jobs:
        job_size = len(job.render())
        if current and size + job_size > MAX_BATCH_CHARS:
            batches.append(current)
            current, size = [], 0
        current.append(job)
        size += job_size
    if current:
        batches.append(current)
    return batches


def parse_summaries(text: str, requested: Sequence[str]) -> Dict[str, str]:
    """Extract ``{node_id: summary}`` for requested ids from a model reply."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return {}
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    allowed = set(requested)
    result: Dict[str, str] = {}
    for node_id, summary in data.items():
        if node_id in allowed and isinstance(summary, str) and summary.strip():
            # Untrusted model text: drop control characters (e.g. ANSI escapes).
            summary = "".join(ch if ch.isprintable() else " " for ch in summary)
            summary = " ".join(summary.split())
            if len(summary) > MAX_SUMMARY_CHARS:
                summary = summary[: MAX_SUMMARY_CHARS - 1].rstrip() + "…"
            result[node_id] = summary
    return result


def summarize_artifact(
    artifact: ValidatedArtifact,
    client: ChatClient,
    model: str,
    *,
    progress: Optional[Callable[[int, int], None]] = None,
) -> SummaryResult:
    """Request summaries for every changed file and non-import symbol.

    Raises :class:`MissingProviderKey`, :class:`RelayAccountError`, or any
    other client (4xx) error from the first request, before any others are
    sent, so the caller can register a key or report the problem once. Other relay failures become scoped
    ``LLM_ERROR``/``LLM_TIMEOUT`` warnings for the files in that request.
    """
    jobs, warnings = _jobs(artifact.value)
    result = SummaryResult(warnings=warnings)
    batches = _batches(jobs)
    if not batches:
        return result

    def run(batch: List[_FileJob]) -> str:
        prompt = "\n\n".join(job.render() for job in batch)
        return client.chat(
            model,
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

    # The first request runs alone so a missing key or refused account
    # surfaces once, before parallel requests multiply the failure.
    done = 0
    outcomes: List[Tuple[List[_FileJob], object]] = []
    try:
        outcomes.append((batches[0], run(batches[0])))
    except (MissingProviderKey, RelayAccountError):
        raise
    except RelayError as error:
        # A client error on the first request (bad model id, unsupported
        # parameter, billing) would repeat for every request.
        if error.status is not None and 400 <= error.status < 500 and error.status not in (408, 429):
            raise
        outcomes.append((batches[0], error))
    done += 1
    if progress:
        progress(done, len(batches))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [(batch, pool.submit(run, batch)) for batch in batches[1:]]
        for batch, future in futures:
            try:
                outcomes.append((batch, future.result()))
            except RelayError as error:
                outcomes.append((batch, error))
            done += 1
            if progress:
                progress(done, len(batches))

    for batch, outcome in outcomes:
        result.llm_calls += 1
        requested = [target["id"] for job in batch for target in job.targets]
        if isinstance(outcome, RelayError):
            code = "LLM_TIMEOUT" if isinstance(outcome, RelayTimeout) else "LLM_ERROR"
            result.warnings.extend(
                {"code": code, "file": job.path, "detail": f"not summarized: {outcome}"}
                for job in batch
            )
            continue
        parsed = parse_summaries(outcome, requested)
        result.summaries.update(parsed)
        for job in batch:
            if job.file_id not in parsed:
                result.warnings.append({
                    "code": "LLM_ERROR",
                    "file": job.path,
                    "detail": "not summarized: the model reply had no summary for this file",
                })
    return result
