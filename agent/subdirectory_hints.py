"""Progressive subdirectory hint discovery: as the agent navigates into
subdirectories via tool calls, load project context files (AGENTS.md, CLAUDE.md,
.cursorrules) from them and append to the tool result — context arrives without
touching the system prompt (prompt caching preserved). Complements the startup
CWD-only loading in ``prompt_builder.py``."""

import hashlib
import logging
import os
import re
import shlex
from pathlib import Path
from threading import RLock
from typing import Dict, Any, Optional, Set

from agent.prompt_builder import (
    _get_context_file_max_chars,
    _read_text_with_timeout,
    _scan_context_content,
    _truncate_content,
)
from agent.project_rules import DEFAULT_MAX_LAZY_CHARS, ProjectRuleSet
from agent.search_policy import SEARCH_PRUNE_DIR_NAMES
from agent.skill_path_triggers import format_touched_skill_hints

logger = logging.getLogger(__name__)

# Same filenames as prompt_builder.py, in priority order (first match wins per dir).
_HINT_FILENAMES = ["AGENTS.override.md", "AGENTS.md", "agents.md", "CLAUDE.md", "claude.md", ".cursorrules"]
# Per-file ceiling for on-demand subdirectory hints. 32 KiB matches Codex's `project_doc_max_bytes` default
# (Claude Code and Cursor apply none); it is a guard against a stray huge CLAUDE.md in a vendored tree, not a
# target — keep area AGENTS.md files well under it (~8k) because this text lands in a tool result on the first
# touch of that directory. Over the ceiling: head+tail kept, marker with the path so the agent can read_file it,
# and a WARNING in the log (the old 8k silent tail-chop cut apps/desktop/AGENTS.md for months unnoticed).
_MAX_HINT_CHARS = 32_000
_PATH_ARG_KEYS = {"path", "file_path", "workdir"}
_COMMAND_TOOLS = {"terminal"}
_MAX_ANCESTOR_WALK = 5  # ancestor levels walked per path — bounds deep-path scans

# Shared with broad recursive search probes so context discovery and search never drift into
# different dependency/cache/build trees (those hold *copies* of context files, never authoritative ones).
_EXCLUDED_DIR_NAMES = SEARCH_PRUNE_DIR_NAMES


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _resolved_hint_target(hint_path: Path, working_dir: Path) -> Optional[Path]:
    """Resolved hint-file target, or None when the file must not be loaded.

    ``is_file()`` and ``read_text`` follow symlinks, so a checked-in
    ``sub/AGENTS.md -> ~/.aws/credentials`` would inject an out-of-tree file
    into the tool result. The resolved target must stay inside the resolved
    working dir (same containment ``_within_working_dir`` applies to the
    directory itself) and pass the canonical read deny-list
    (``context_references`` applies it to explicit @-references), which also
    catches in-tree targets like a symlink to the project ``.env``.
    """
    try:
        resolved = hint_path.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        inside = resolved.is_relative_to(working_dir)
    except (OSError, ValueError, RuntimeError):
        inside = False
    if not inside:
        return None
    try:
        from agent.file_safety import get_read_block_error
        blocked = get_read_block_error(str(resolved)) is not None
    except Exception:
        # Mirror context_references: a deny-list lookup that fails re-opens the
        # exact hole the check closes, so fail closed.
        return None
    return None if blocked else resolved


def _first_hint_file(directory: Path):
    """``(path, stripped content)`` of the first readable non-empty hint file
    in *directory* (priority order), or None. Unreadable files are skipped."""
    for filename in _HINT_FILENAMES:
        candidate = directory / filename
        try:
            if not candidate.is_file() or (target := _resolved_hint_target(candidate, directory)) is None:
                continue
            # Read the resolved target (not the link path) so a symlink swapped
            # between check and read still lands on the vetted file.
            content = target.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeDecodeError):
            continue
        return candidate, content
    return None


_NAV_COMMANDS = frozenset({"cd", "pushd"})
_SHELL_OPERATORS = frozenset({"&&", "||", "|", ";", "&", ";;", "|&", "(", ")"})


def _nav_targets(cmd: str) -> list:
    """Operands of `cd` / `pushd` that begin a shell segment. `cd -` and bare `cd` yield nothing."""
    lexer = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return []
    targets, segment_start = [], True
    for idx, token in enumerate(tokens):
        if token in _SHELL_OPERATORS:
            segment_start = True
            continue
        if segment_start and token in _NAV_COMMANDS:
            operand = next((t for t in tokens[idx + 1:] if t in _SHELL_OPERATORS or not t.startswith("-")), None)
            if operand and operand not in _SHELL_OPERATORS:
                targets.append(operand)
        segment_start = False
    return targets


class SubdirectoryHintTracker:
    """Track which directories the agent visits and load hints on first access.

    Usage: after each tool call, ``hints = tracker.check_tool_call(name, args)``
    and append the returned text to the tool result.
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        *,
        enabled: bool = True,
        context_length: Optional[int] = None,
        context_source: Optional[Any] = None,
    ):
        # ``enabled=False`` mirrors ``skip_context_files``: a session that opted out of
        # AGENTS.md/CLAUDE.md injection at startup must not get the same files spliced into
        # tool results later — cron jobs relaying exact stdout leaked them to chat (#9441).
        self.enabled = enabled
        self.working_dir = Path(working_dir or os.getcwd()).resolve()
        self._context_length = context_length
        self._context_source = context_source
        # Serialize discovery and claims for concurrent tool-result handling.
        self._lock = RLock()
        # The working dir is pre-marked loaded (startup context handles it).
        self._loaded_dirs: Set[Path] = {self.working_dir}
        # Content digests already injected: the same file reached through
        # symlinks/hardlinks/copies is never re-sent. Seeded with the CWD hint
        # file prompt_builder already loaded.
        self._loaded_digests: Set[str] = set()
        found = _first_hint_file(self.working_dir)
        if found and found[1]:
            self._loaded_digests.add(_digest(found[1]))
        # Scoped (``paths:``-carrying) rule files claimed lazily as tool calls touch matching
        # workspace paths — the same attachment slot the directory hints use.
        self._project_rules = (
            ProjectRuleSet.discover(self.working_dir, scanner=_scan_context_content) if enabled else None
        )
        # Skill names already surfaced by a path hint: the pointer fires once per session,
        # mirroring the scoped-rule claim bookkeeping (repeat touches stay free).
        self._hinted_skills: Set[str] = set()

    def _current_hint_budget(self) -> int:
        """Resolve the live aggregate budget for one lazy context attachment.

        One attachment can carry a directory hint plus the scoped rule files for the same
        touch, so the aggregate is the larger of the context-file cap and the lazy-rules
        ceiling — a default-config project must not starve its own rules out of the same
        tool result.
        """
        context_length = self._context_length
        if self._context_source is not None:
            try:
                context_length = getattr(self._context_source, "context_length", context_length)
            except Exception:
                pass
        return max(_get_context_file_max_chars(context_length), DEFAULT_MAX_LAZY_CHARS)

    def check_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
        """Return formatted hint text for newly visited directories, or None."""
        if not self.enabled:
            return None
        with self._lock:
            return self._check_tool_call_unlocked(tool_name, tool_args)

    def _check_tool_call_unlocked(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
        """Discover directory hints and claim scoped rules for this tool call.

        One aggregate attachment budget (``context_file_max_chars``) covers every section
        this call appends: discovered directory hints first, then scoped rule files whose
        ``paths:`` match a touched workspace path.
        """
        total_budget = self._current_hint_budget()
        # ``check_tool_call`` prefixes every non-empty attachment with two newlines.
        remaining = max(0, total_budget - 2)
        if remaining <= 0:
            return None

        all_hints = []
        for directory in self._extract_directories(tool_name, tool_args):
            separator_chars = 2 if all_hints else 0
            available = max(0, remaining - separator_chars)
            if available <= 0:
                break
            hints = self._load_hints_for_directory(directory, max_chars=available)
            if hints:
                all_hints.append(hints)
                remaining -= separator_chars + len(hints)

        touched_paths = self._extract_touched_paths(tool_name, tool_args)
        separator_chars = 2 if all_hints else 0
        available = max(0, remaining - separator_chars)
        project_rules = (
            self._project_rules.claim_matches(
                touched_paths,
                max_chars=available,
                defer_oversize=bool(all_hints),
            )
            if self._project_rules is not None
            else ""
        )
        if project_rules:
            all_hints.append(project_rules)
            remaining -= separator_chars + len(project_rules)

        # Path-triggered skill hints ride the same attachment, after directory hints and
        # scoped rules: a pointer to a skill whose ``paths:`` covers a touched path.
        if touched_paths:
            separator_chars = 2 if all_hints else 0
            available = max(0, remaining - separator_chars)
            skill_hints = format_touched_skill_hints(
                touched_paths, self.working_dir, max_chars=available,
                claimed=self._hinted_skills)
            if skill_hints:
                all_hints.append(skill_hints)

        return "\n\n" + "\n\n".join(all_hints) if all_hints else None

    def _extract_directories(self, tool_name: str, args: Dict[str, Any]) -> list:
        """Extract directory paths from tool call arguments."""
        candidates: Set[Path] = set()
        for key in _PATH_ARG_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                self._add_path_candidate(val, candidates)
        cmd = args.get("command", "") if tool_name in _COMMAND_TOOLS else None
        if isinstance(cmd, str):
            self._extract_paths_from_command(cmd, candidates)
        # Most-specific directory first: when several ancestors carry hint files they share
        # one bounded attachment, so the closest context must get the budget first.
        return sorted(candidates, key=lambda path: (-len(path.parts), str(path)))

    def _extract_touched_paths(self, tool_name: str, args: Dict[str, Any]) -> list:
        """Extract concrete workspace paths for scoped-rule matching.

        Intentionally separate from ``_extract_directories``: a nonexistent extensionless
        file is still a concrete rule candidate, not necessarily a directory. V4A
        multi-file patches keep their targets in patch text, so those headers are parsed
        explicitly.
        """
        touched: list = []
        seen: Set[Path] = set()

        base_dir = self.working_dir
        raw_workdir = args.get("workdir")
        if isinstance(raw_workdir, str) and raw_workdir.strip():
            resolved_workdir = self._resolve_touched_path(raw_workdir, self.working_dir)
            if resolved_workdir is not None:
                base_dir = resolved_workdir
                seen.add(resolved_workdir)
                touched.append(resolved_workdir)

        for key in ("path", "file_path"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                candidate = self._resolve_touched_path(value, base_dir)
                if candidate is not None and candidate not in seen:
                    seen.add(candidate)
                    touched.append(candidate)

        if tool_name == "patch" and args.get("mode") == "patch":
            patch_text = args.get("patch", "")
            if isinstance(patch_text, str):
                for raw_path in re.findall(
                    r"^\*\*\* (?:(?:Add|Update|Delete) File:|Move to:)\s*(.+?)\s*$",
                    patch_text,
                    flags=re.MULTILINE,
                ):
                    candidate = self._resolve_touched_path(raw_path, base_dir)
                    if candidate is not None and candidate not in seen:
                        seen.add(candidate)
                        touched.append(candidate)

        if tool_name in _COMMAND_TOOLS:
            command = args.get("command", "")
            if isinstance(command, str):
                for candidate in self._command_paths(command, base_dir):
                    if candidate not in seen:
                        seen.add(candidate)
                        touched.append(candidate)

        return touched

    @staticmethod
    def _resolve_touched_path(raw_path: str, base_dir: Path) -> Optional[Path]:
        try:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            return path.resolve(strict=False)
        except (OSError, ValueError, RuntimeError):
            return None

    def _command_paths(self, command: str, base_dir: Path) -> list:
        """Return path-like shell operands, including existing bare directories."""
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()

        paths: list = []
        seen: Set[Path] = set()
        for token in tokens:
            if token.startswith("-") or token in {"&&", "||", ";", "|"}:
                continue
            if token.startswith(("http://", "https://", "git@")):
                continue
            candidate = self._resolve_touched_path(token, base_dir)
            if candidate is None:
                continue
            looks_like_path = "/" in token or "." in token
            try:
                exists = candidate.exists()
            except OSError:
                exists = False
            if not looks_like_path and not exists:
                continue
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
        return paths

    def _add_path_candidate(self, raw_path: str, candidates: Set[Path]):
        """Add a raw path's directory and its ancestors (up to ``_MAX_ANCESTOR_WALK``
        levels, stopping at the first already-loaded dir) so reading
        ``project/src/main.py`` still discovers ``project/AGENTS.md``."""
        try:
            p = Path(raw_path).expanduser()
            if not p.is_absolute():
                p = self.working_dir / p
            p = p.resolve()
            if p.suffix or (p.exists() and p.is_file()):
                p = p.parent
            for _ in range(_MAX_ANCESTOR_WALK):
                if p in self._loaded_dirs:
                    # A deeper hint can consume the aggregate attachment budget before its
                    # ancestors are visited; keep walking so deferred ancestors stay
                    # discoverable on a later call.
                    if p == self.working_dir:
                        break
                    parent = p.parent
                    if parent == p:
                        break
                    p = parent
                    continue
                if self._is_valid_subdir(p):
                    candidates.add(p)
                parent = p.parent
                if parent == p:
                    break  # filesystem root
                p = parent
        except (OSError, ValueError, RuntimeError):
            pass

    def _extract_paths_from_command(self, cmd: str, candidates: Set[Path]):
        """Extract path-like tokens (contain / or .; not flags or URLs) from a shell command."""
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()
        # `cd backend && ls`: a bare directory name has no `/` or `.`, so the generic filter below drops
        # it; the operand of a navigation command is a path by construction (#11032). Only a `cd` at the
        # START of a shell segment counts (`echo cd backend` is prose); punctuation-aware tokenizing keeps
        # a quoted `'backend;'` literal while splitting bare `backend;ls` at the operator.
        for target in _nav_targets(cmd):
            self._add_path_candidate(target, candidates)
        for token in tokens:
            if token.startswith(("-", "http://", "https://", "git@")) or ("/" not in token and "." not in token):
                continue
            self._add_path_candidate(token, candidates)

    def _within_working_dir(self, path: Path) -> bool:
        """Reject paths outside the working-dir tree: loading ~/.codex/AGENTS.md
        or ~/.claude/CLAUDE.md would mix another agent's instructions into this
        session. Falls back to an ancestor check when ``is_relative_to`` fails."""
        try:
            return path.is_relative_to(self.working_dir)
        except (OSError, ValueError):
            try:
                path.relative_to(self.working_dir)
                return True
            except ValueError:
                return False

    def _is_valid_subdir(self, path: Path) -> bool:
        """Directory inside the working-dir tree, not yet loaded, not an excluded copy dir."""
        try:
            if not path.is_dir():
                return False
        except OSError:
            return False
        return path not in self._loaded_dirs and self._within_working_dir(path) and not self._is_excluded(path)

    def _is_excluded(self, path: Path) -> bool:
        """True when a segment *below* the working dir is an excluded copy dir
        (a user deliberately working inside ``vendor/`` keeps that segment legitimate)."""
        try:
            rel_parts = path.relative_to(self.working_dir).parts
        except ValueError:
            return True  # outside the tree — already rejected upstream
        return any(part in _EXCLUDED_DIR_NAMES for part in rel_parts)

    def _load_hints_for_directory(self, directory: Path, *, max_chars: Optional[int] = None) -> Optional[str]:
        """Load the first hint file in *directory*; formatted text or None.

        ``max_chars`` is this directory's share of the call's aggregate attachment budget;
        it can be lower than ``_MAX_HINT_CHARS`` when earlier sections used the budget.
        """
        if max_chars is None:
            max_chars = max(0, self._current_hint_budget() - 2)
        else:
            max_chars = max(0, int(max_chars))
        max_chars = min(max_chars, _MAX_HINT_CHARS)
        if max_chars <= 0:
            return None
        self._loaded_dirs.add(directory)
        if not self._within_working_dir(directory):
            logger.debug("Skipping hint files in %s — outside working_dir %s", directory, self.working_dir)
            return None
        for filename in _HINT_FILENAMES:
            hint_path = directory / filename
            try:
                if not hint_path.is_file():
                    continue
            except OSError:
                continue
            if (target := _resolved_hint_target(hint_path, self.working_dir)) is None:
                continue
            try:
                content = (_read_text_with_timeout(target) or "").strip()
                if not content:
                    continue
                digest = _digest(content)
                if digest in self._loaded_digests:
                    logger.debug("Skipping duplicate hint content at %s (digest %s)", hint_path, digest[:12])
                    return None
                self._loaded_digests.add(digest)
                # Same security scan as startup context loading.
                content = _scan_context_content(content, filename)
                rel_path = self._display_path(hint_path)
                content = _truncate_content(
                    content, filename, max_chars=max_chars, read_path=rel_path, queue_warning=False,
                )
                logger.debug("Loaded subdirectory hints from %s: %s", directory, [rel_path])
                return f"[Subdirectory context discovered: {rel_path}]\n{content}"  # first match wins per directory
            except Exception as exc:
                logger.debug("Could not read %s: %s", hint_path, exc)
        return None

    def _display_path(self, hint_path: Path) -> str:
        """Working-dir-relative, else ``~/``-relative (POSIX rendering so Windows
        never shows ``~/AppData\\Local\\...`` chimeras), else absolute."""
        try:
            return str(hint_path.relative_to(self.working_dir))
        except (ValueError, RuntimeError):
            pass
        try:
            return "~/" + hint_path.relative_to(Path.home()).as_posix()
        except (ValueError, RuntimeError):
            return str(hint_path)
