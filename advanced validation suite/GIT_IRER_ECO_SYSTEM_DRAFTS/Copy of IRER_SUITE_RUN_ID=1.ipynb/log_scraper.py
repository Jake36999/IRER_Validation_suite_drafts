"""
log_scraper.py
CLASSIFICATION: Artifact Collection Utility (Sprint 3)
GOAL: Scans project directories to extract stack traces and error snippets,
      preparing a structured input payload for the Agnostic AI Debugging Core.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any


# --- Configuration defaults ---
DEFAULT_LOG_EXTENSIONS = (".log", ".txt", ".err", ".out", ".json")
DEFAULT_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


# --- Core Helpers ---

def iter_log_files(
    roots: Iterable[str],
    extensions: Tuple[str, ...] = DEFAULT_LOG_EXTENSIONS,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> Iterable[Path]:
    """Recursively yields paths to log-like files under the given roots."""
    for root in roots:
        root_path = Path(root)
        if not root_path.exists(): continue

        if root_path.is_file():
            if root_path.suffix.lower() in extensions and root_path.stat().st_size <= max_file_size_bytes:
                yield root_path
            continue

        for path in root_path.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in extensions
                and path.stat().st_size <= max_file_size_bytes
            ):
                yield path


def _extract_python_tracebacks(lines: List[str]) -> List[str]:
    """Extracts full Python-style tracebacks."""
    tracebacks: List[str] = []
    current_block: List[str] = []
    in_traceback = False

    for line in lines:
        if not in_traceback:
            if line.startswith("Traceback (most recent call last):"):
                in_traceback = True
                current_block = [line]
            continue

        stripped = line.rstrip("\n")
        if stripped == "" and current_block:
            tracebacks.append("".join(current_block))
            current_block = []
            in_traceback = False
            continue

        if (stripped.startswith("  File") or stripped.startswith("    ") or
            re.match(r"^\w+Error[: ]", stripped) or re.match(r"^\w+Exception[: ]", stripped)):
            current_block.append(line)
        else:
            if current_block:
                tracebacks.append("".join(current_block))
            current_block = []
            in_traceback = False

    if in_traceback and current_block:
        tracebacks.append("".join(current_block))

    return tracebacks


def _extract_error_snippets(lines: List[str], context_lines: int = 3) -> List[str]:
    """Extracts generic error snippets (JAX, CUDA, etc.) with context."""
    error_keywords = ["Traceback (most recent call last):", "Error:", "Exception:", "TypeError:", "ValueError:", "RuntimeError:", "SyntaxError:", "ImportError:", "CUDA error", "JAX"]

    error_indices: List[int] = []
    for i, line in enumerate(lines):
        if any(kw in line for kw in error_keywords):
            error_indices.append(i)

    snippets: List[str] = []
    for idx in error_indices:
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + context_lines + 1)
        block = lines[start:end]

        snippet_text = "".join(block)
        # Avoid duplicating tracebacks already captured
        if "Traceback (most recent call last):" not in snippet_text:
             snippets.append(snippet_text)

    # Deduplicate unique snippets
    unique_snippets: List[str] = []
    seen = set()
    for snippet in snippets:
        key = snippet.strip()
        if key and key not in seen:
            seen.add(key)
            unique_snippets.append(snippet)

    return unique_snippets


def extract_stack_traces_from_file(
    path: Path,
    context_lines_for_snippets: int = 3,
) -> Tuple[List[str], List[str]]:
    """Extracts stack traces and error snippets from a single file."""
    try:
        text = path.read_text(errors="replace")
    except Exception as exc:
        return [], [f"Failed to read {path}: {exc}"]

    lines = text.splitlines(keepends=True)

    tracebacks = _extract_python_tracebacks(lines)
    snippets = _extract_error_snippets(lines, context_lines=context_lines)

    # Attach filename context
    tracebacks_with_header = [f"=== FILE: {path} ===\n" + tb for tb in tracebacks]
    snippets_with_header = [f"=== FILE: {path} (snippet) ===\n" + sn for sn in snippets]

    return tracebacks_with_header, snippets_with_header


# --- High-level aggregation helper ---


def collect_debug_inputs(
    roots: Iterable[str],
    max_files: int = 100,
    context_lines: int = 3,
    extensions: Tuple[str, ...] = DEFAULT_LOG_EXTENSIONS,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> Dict[str, List[str] | str]:
    """
    Collects stack traces and error snippets, preparing inputs for the debugging core.
    """
    log_files = list(iter_log_files(roots, extensions=extensions, max_file_size_bytes=max_file_size_bytes))
    log_files = log_files[:max_files]

    bug_instances: List[str] = []
    transcripts: List[str] = []

    for path in log_files:
        tracebacks, snippets = extract_stack_traces_from_file(path, context_lines_for_snippets=context_lines)
        bug_instances.extend(tracebacks)
        transcripts.extend(snippets) # Use snippets as supplementary transcript data

    artifacts_lines = [
        "Log Scraper Artifacts Summary",
        "------------------------------",
        f"Roots scanned: {', '.join(str(r) for r in roots)}",
        f"Log files considered: {len(log_files)}",
    ]
    if log_files:
        artifacts_lines.append("")
        artifacts_lines.append("Files:")
        for path in log_files:
            artifacts_lines.append(f"- {path} ({path.stat().st_size} bytes)")

    artifacts_summary = "\n".join(artifacts_lines)

    return {
        "bug_instances": bug_instances,
        "transcripts": transcripts,
        "artifacts_summary": artifacts_summary,
    }
