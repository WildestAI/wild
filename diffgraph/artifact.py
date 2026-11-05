"""Build and dispatch validated canonical DiffGraph artifacts.

The structural producer and contract validator meet here so a CLI invocation
constructs one artifact, validates it once, and then hands the same validated
value to its selected consumer.  This module is deliberately local-only: it
imports no AI SDK or network-capable DiffGraph module.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from diffgraph.contract import ValidatedArtifact
from diffgraph.structural import analyze_local_diff


def build_validated_artifact(
    repository: str,
    *,
    staged: bool = False,
    pathspecs: Sequence[str] = (),
    base_ref: Optional[str] = None,
    head_ref: Optional[str] = None,
    three_dot: bool = False,
    against: Optional[str] = None,
    wild_version: str,
) -> ValidatedArtifact:
    """Construct and validate exactly one local structural artifact."""
    artifact = analyze_local_diff(
        repository,
        staged=staged,
        pathspecs=pathspecs,
        base_ref=base_ref,
        head_ref=head_ref,
        three_dot=three_dot,
        against=against,
        wild_version=wild_version,
    )
    return ValidatedArtifact.from_value(artifact)


def render_canonical_json(artifact: ValidatedArtifact) -> str:
    """Return stable, human-readable canonical JSON with a trailing newline."""
    return json.dumps(artifact.value, indent=2, sort_keys=True) + "\n"


def write_canonical_json(artifact: ValidatedArtifact, destination: Path) -> None:
    """Atomically write canonical JSON without creating missing directories."""
    rendered = render_canonical_json(artifact)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
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
