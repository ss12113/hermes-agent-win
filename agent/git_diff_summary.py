"""Bounded Git diff summaries for slash commands.

Hermes exposes this through `/diff` so users can inspect the current repository
state without dumping a huge raw patch into chat.  The helper intentionally
uses Git metadata (`--shortstat`, `--numstat`, untracked filenames) and never
reads untracked file contents.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_GIT_TIMEOUT_SECONDS = 5
_DEFAULT_MAX_FILES = 50
_MAX_FILES_FOR_DETAILS = 500
_TRANSIENT_STATE_FILES: tuple[tuple[str, str], ...] = (
    ("MERGE_HEAD", "merge"),
    ("REBASE_HEAD", "rebase"),
    ("CHERRY_PICK_HEAD", "cherry-pick"),
    ("REVERT_HEAD", "revert"),
)


@dataclass(frozen=True)
class NumstatEntry:
    path: str
    additions: int | None
    deletions: int | None

    @property
    def binary(self) -> bool:
        return self.additions is None or self.deletions is None


@dataclass(frozen=True)
class GitDiffSummary:
    cwd: str
    repo_root: str | None
    shortstat: dict[str, int]
    files: tuple[NumstatEntry, ...]
    untracked: tuple[str, ...]
    details_omitted: bool = False
    transient_state: str | None = None
    error: str | None = None

    @property
    def clean(self) -> bool:
        return (
            not self.error
            and not self.transient_state
            and not self.details_omitted
            and not self.files
            and not self.untracked
        )


def _run_git(args: Iterable[str], cwd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=_GIT_TIMEOUT_SECONDS,
        check=False,
    )


def parse_shortstat(stdout: str) -> dict[str, int]:
    """Parse `git diff --shortstat` output into integer counters."""
    text = (stdout or "").strip()
    if not text:
        return {"files": 0, "insertions": 0, "deletions": 0}

    def _find(pattern: str) -> int:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else 0

    return {
        "files": _find(r"(\d+)\s+files?\s+changed"),
        "insertions": _find(r"(\d+)\s+insertions?\(\+\)"),
        "deletions": _find(r"(\d+)\s+deletions?\(-\)"),
    }


def parse_numstat(stdout: str) -> tuple[NumstatEntry, ...]:
    """Parse `git diff --numstat` output, preserving filenames containing tabs."""
    entries: list[NumstatEntry] = []
    for raw_line in (stdout or "").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t", 2)
        if len(parts) != 3:
            continue
        add_raw, del_raw, path = parts
        if add_raw == "-" or del_raw == "-":
            entries.append(NumstatEntry(path=path, additions=None, deletions=None))
            continue
        try:
            entries.append(NumstatEntry(path=path, additions=int(add_raw), deletions=int(del_raw)))
        except ValueError:
            continue
    return tuple(entries)


def _repo_root(cwd: str) -> str | None:
    result = _run_git(["rev-parse", "--show-toplevel"], cwd)
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return root or None


def _git_dir(cwd: str) -> Path | None:
    result = _run_git(["rev-parse", "--git-dir"], cwd)
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path(cwd) / path
    return path.resolve()


def _transient_state(cwd: str) -> str | None:
    git_dir = _git_dir(cwd)
    if git_dir is None:
        return None
    for filename, label in _TRANSIENT_STATE_FILES:
        if (git_dir / filename).exists():
            return label
    # During an interactive rebase Git commonly uses a directory instead of
    # REBASE_HEAD, so check those too.
    if (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists():
        return "rebase"
    return None


def collect_git_diff_summary(cwd: str | os.PathLike[str] | None = None) -> GitDiffSummary:
    """Collect a bounded, metadata-only summary for the repository at `cwd`."""
    resolved_cwd = str(Path(cwd or os.getcwd()).expanduser().resolve())
    if not Path(resolved_cwd).exists():
        return GitDiffSummary(
            cwd=resolved_cwd,
            repo_root=None,
            shortstat={"files": 0, "insertions": 0, "deletions": 0},
            files=(),
            untracked=(),
            error=f"Directory does not exist: {resolved_cwd}",
        )

    try:
        root = _repo_root(resolved_cwd)
    except (OSError, subprocess.SubprocessError) as exc:
        return GitDiffSummary(
            cwd=resolved_cwd,
            repo_root=None,
            shortstat={"files": 0, "insertions": 0, "deletions": 0},
            files=(),
            untracked=(),
            error=str(exc),
        )
    if not root:
        return GitDiffSummary(
            cwd=resolved_cwd,
            repo_root=None,
            shortstat={"files": 0, "insertions": 0, "deletions": 0},
            files=(),
            untracked=(),
            error="Not in a git repository.",
        )

    state = _transient_state(resolved_cwd)
    if state:
        return GitDiffSummary(
            cwd=resolved_cwd,
            repo_root=root,
            shortstat={"files": 0, "insertions": 0, "deletions": 0},
            files=(),
            untracked=(),
            transient_state=state,
        )

    short = _run_git(["diff", "HEAD", "--shortstat"], resolved_cwd)

    error = None
    if short.returncode not in {0, 1}:
        error = (short.stderr or short.stdout or "git diff failed").strip()

    shortstat = parse_shortstat(short.stdout)
    if error:
        return GitDiffSummary(
            cwd=resolved_cwd,
            repo_root=root,
            shortstat=shortstat,
            files=(),
            untracked=(),
            error=error,
        )

    # Claude Code probes --shortstat before fetching per-file details.  If a
    # tracked diff is huge, keep /diff responsive by returning accurate totals
    # without loading every path from --numstat.
    if shortstat.get("files", 0) > _MAX_FILES_FOR_DETAILS:
        return GitDiffSummary(
            cwd=resolved_cwd,
            repo_root=root,
            shortstat=shortstat,
            files=(),
            untracked=(),
            details_omitted=True,
        )

    num = _run_git(["diff", "HEAD", "--numstat"], resolved_cwd)
    untracked_result = _run_git(["ls-files", "--others", "--exclude-standard"], resolved_cwd)

    files = parse_numstat(num.stdout if num.returncode in {0, 1} else "")
    untracked = tuple(
        line for line in untracked_result.stdout.splitlines()
        if line.strip()
    ) if untracked_result.returncode == 0 else ()

    return GitDiffSummary(
        cwd=resolved_cwd,
        repo_root=root,
        shortstat=shortstat,
        files=files,
        untracked=untracked,
        error=error,
    )


def _plural(value: int, singular: str, plural: str | None = None) -> str:
    return f"{value} {singular if value == 1 else (plural or singular + 's')}"


def format_git_diff_summary(
    cwd: str | os.PathLike[str] | None = None,
    *,
    max_files: int = _DEFAULT_MAX_FILES,
) -> str:
    """Return a human-readable bounded Git diff summary."""
    summary = collect_git_diff_summary(cwd)
    if summary.error:
        return summary.error
    if summary.transient_state:
        return (
            f"Git {summary.transient_state} in progress. Resolve it before using /diff; "
            "the worktree may include transient conflict state."
        )
    if summary.clean:
        return "Working tree is clean."

    root_label = summary.repo_root or summary.cwd
    stat = summary.shortstat
    tracked_files = stat.get("files", 0) or len(summary.files)
    insertions = stat.get("insertions", 0)
    deletions = stat.get("deletions", 0)
    untracked_count = len(summary.untracked)

    lines = [f"Git diff summary for {root_label}"]
    stat_parts = []
    if tracked_files:
        stat_parts.append(_plural(tracked_files, "tracked file"))
    if insertions:
        stat_parts.append(_plural(insertions, "insertion"))
    if deletions:
        stat_parts.append(_plural(deletions, "deletion"))
    if untracked_count:
        stat_parts.append(_plural(untracked_count, "untracked file"))
    lines.append(", ".join(stat_parts) if stat_parts else "No tracked diff.")

    if summary.details_omitted:
        limit = _plural(_MAX_FILES_FOR_DETAILS, "tracked file")
        lines.append(f"Too many files to display details: exceeds {limit}.")
        return "\n".join(lines)

    detail_limit = max(0, max_files)
    shown_files = summary.files[:detail_limit]
    remaining_slots = max(0, detail_limit - len(shown_files))
    shown_untracked = summary.untracked[:remaining_slots]
    shown_count = len(shown_files) + len(shown_untracked)
    detail_total = len(summary.files) + len(summary.untracked)

    if shown_files:
        lines.append("")
        lines.append("Tracked changes:")
        for entry in shown_files:
            if entry.binary:
                suffix = "binary"
            else:
                suffix = f"+{entry.additions}/-{entry.deletions}"
            lines.append(f"  M {entry.path} ({suffix})")

    if shown_untracked:
        lines.append("")
        lines.append("Untracked files:")
        for path in shown_untracked:
            lines.append(f"  ?? {path}")

    if detail_total > shown_count:
        lines.append("")
        if shown_count:
            lines.append(
                f"Per-file details limited to first {shown_count} of {detail_total} changed files."
            )
        else:
            lines.append(f"Per-file details omitted: {detail_total} files exceeds limit {detail_limit}.")

    return "\n".join(lines)
