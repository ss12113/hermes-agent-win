"""Scoped project-instruction rules.

This module implements the small, useful part of Claude Code's project-memory
model that Hermes can support without changing the system-prompt cache:

* ``.hermes/rules/**/*.md`` (and compatible ``.claude/rules/**/*.md``) files
  without ``paths:`` are loaded at startup;
* files with a valid ``paths:`` string/list are claimed lazily when a tool
  touches a matching workspace path;
* rule bodies are snapshotted, threat-scanned, capped, and deduplicated per
  agent session.

The module deliberately has no dependency on prompt assembly or tool execution.
Callers provide the context scanner so the same security policy is used for
startup and lazy attachments without creating an import cycle.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable, Optional

from pathspec import PathSpec

from agent.skill_utils import yaml_load

logger = logging.getLogger(__name__)

RULE_DIRECTORY_NAMES = (
    Path(".hermes") / "rules",
    Path(".claude") / "rules",
)
DEFAULT_MAX_RULE_CHARS = 12_000
DEFAULT_MAX_RULE_COUNT = 256
DEFAULT_MAX_STARTUP_CHARS = 48_000
DEFAULT_MAX_LAZY_CHARS = 32_000

RuleScanner = Callable[[str, str], str]


@dataclass(frozen=True)
class ProjectRule:
    """A validated, immutable rule snapshot."""

    identity: Path
    display_path: str
    # Path patterns are authored relative to the project that owns the rule
    # directory, which may be nested below the tracker's outer workspace.
    match_root: Path
    body: str
    # None means unconditional; an empty tuple means invalid/never-match.
    patterns: Optional[tuple[str, ...]]
    spec: Optional[PathSpec]


def _rule_specificity(rule: ProjectRule) -> int:
    """Prefer narrow path rules before broad catch-all rules.

    Claude-style rule sets commonly combine a broad code-quality rule with a
    smaller subsystem rule. Loading the narrow rule first makes the lazy budget
    useful instead of allowing a large catch-all document to crowd out the
    subsystem instructions that explain the file being touched.
    """
    if not rule.patterns:
        return 0
    return sum(
        len(re.sub(r"[*!?\[\]{}#]", "", pattern))
        for pattern in rule.patterns
    )


def find_workspace_root(cwd: Path | str) -> Path:
    """Return the nearest git root, or *cwd* without walking its parents.

    The no-git behavior is intentional: a random ``.hermes/rules`` in a parent
    directory must not contaminate an unrelated checkout or temporary worker.
    ``.git`` may be a directory or a worktree file; ``exists()`` handles both.
    """

    start = Path(cwd).expanduser().resolve()
    if not start.is_dir():
        start = start.parent
    for candidate in (start, *start.parents):
        try:
            if (candidate / ".git").exists():
                return candidate
        except OSError:
            continue
    return start


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _strict_frontmatter(content: str) -> tuple[dict[str, Any], str] | None:
    """Parse rule frontmatter strictly; malformed metadata fails closed.

    ``skill_utils.parse_frontmatter`` intentionally has a permissive fallback
    for skill discovery. Rule activation is different: a malformed ``paths``
    header must never accidentally turn a scoped rule into a global rule.
    """

    if content.startswith("\ufeff"):
        content = content[1:]
    if not re.match(r"\A---[ \t]*(?:\r?\n|\Z)", content):
        # A frontmatter-looking delimiter after whitespace is almost always a
        # malformed attempt at metadata. Fail closed rather than treating the
        # apparent scoped rule as unconditional startup content.
        if content.lstrip(" \t\r\n").startswith("---"):
            return None
        return {}, content

    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None
    closing_index: Optional[int] = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        return None

    yaml_text = "".join(lines[1:closing_index])
    try:
        metadata = yaml_load(yaml_text)
    except Exception as exc:
        logger.warning("Skipping project rule with invalid YAML frontmatter: %s", exc)
        return None
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        logger.warning("Skipping project rule whose frontmatter is not a mapping")
        return None

    return metadata, "".join(lines[closing_index + 1 :])


def _normalise_patterns(metadata: dict[str, Any]) -> Optional[tuple[str, ...]]:
    """Return path patterns, with invalid scoped metadata as ``()``."""

    if "paths" not in metadata:
        return None
    raw = metadata.get("paths")
    if isinstance(raw, str):
        values: list[Any] = [raw]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        return ()
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        return ()
    return tuple(value.strip() for value in values)


def _normalise_workspace_path(raw: Path | str, workspace_root: Path) -> Optional[Path]:
    """Resolve a touched path and reject traversal/symlink escapes."""

    try:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        # strict=False still resolves existing parent symlinks, which is the
        # important part for rejecting ``workspace/link-outside/file.py``.
        candidate = candidate.resolve(strict=False)
        root = workspace_root.resolve(strict=True)
        if not candidate.is_relative_to(root):
            return None
        return candidate
    except (OSError, ValueError, RuntimeError):
        return None


def _relative_posix(path: Path, root: Path) -> Optional[str]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    value = relative.as_posix()
    return "" if value == "." else value


class ProjectRuleSet:
    """Per-session immutable-ish project rule index with atomic claims."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        scanner: Optional[RuleScanner] = None,
        max_rule_chars: int = DEFAULT_MAX_RULE_CHARS,
        max_rule_count: int = DEFAULT_MAX_RULE_COUNT,
        max_startup_chars: int = DEFAULT_MAX_STARTUP_CHARS,
        max_lazy_chars: int = DEFAULT_MAX_LAZY_CHARS,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.scanner = scanner or (lambda body, _filename: body)
        self.max_rule_chars = max(1, int(max_rule_chars))
        self.max_rule_count = max(1, int(max_rule_count))
        self.max_startup_chars = max(0, int(max_startup_chars))
        self.max_lazy_chars = max(0, int(max_lazy_chars))
        self._rules: dict[Path, ProjectRule] = {}
        self._known_rule_roots: set[Path] = set()
        self._claimed: set[Path] = set()
        self._unindexed_rule_paths: set[str] = set()
        self._reported_unindexed_paths: set[str] = set()
        self._lock = RLock()
        self._discover_roots_for_directory(self.workspace_root)

    @classmethod
    def discover(
        cls,
        cwd: Path | str,
        *,
        scanner: Optional[RuleScanner] = None,
        max_rule_chars: int = DEFAULT_MAX_RULE_CHARS,
        max_rule_count: int = DEFAULT_MAX_RULE_COUNT,
        max_startup_chars: int = DEFAULT_MAX_STARTUP_CHARS,
        max_lazy_chars: int = DEFAULT_MAX_LAZY_CHARS,
    ) -> "ProjectRuleSet":
        """Build a ruleset rooted at the nearest repository or cwd."""

        rules = cls(
            find_workspace_root(cwd),
            scanner=scanner,
            max_rule_chars=max_rule_chars,
            max_rule_count=max_rule_count,
            max_startup_chars=max_startup_chars,
            max_lazy_chars=max_lazy_chars,
        )
        initial_path = Path(cwd).expanduser().resolve(strict=False)
        rules._discover_roots_for_directory(
            initial_path if initial_path.is_dir() else initial_path.parent
        )
        return rules

    @property
    def rules(self) -> tuple[ProjectRule, ...]:
        """Deterministic snapshot useful for diagnostics and tests."""

        with self._lock:
            return tuple(sorted(self._rules.values(), key=lambda rule: rule.display_path))

    def startup_text(self) -> str:
        """Format unconditional rules for the initial system prompt."""

        with self._lock:
            rules = [rule for rule in self._rules.values() if rule.patterns is None]
            rules.sort(key=lambda rule: rule.display_path)
            return self._bounded_text(rules, self.max_startup_chars, "startup")

    def claim_matches(
        self,
        paths: Iterable[Path | str],
        *,
        max_chars: Optional[int] = None,
        defer_oversize: bool = False,
    ) -> str:
        """Claim and format newly matching scoped rules in stable order.

        Only rules that fit the lazy aggregate budget are claimed. A later tool
        call can therefore claim any rules left over instead of silently losing
        them forever.
        """

        candidates: list[tuple[Path, bool]] = []
        with self._lock:
            for raw in paths:
                candidate = _normalise_workspace_path(raw, self.workspace_root)
                if candidate is None:
                    continue
                self._discover_roots_for_path(candidate)
                is_directory = False
                try:
                    if candidate.is_dir():
                        is_directory = True
                except OSError:
                    pass
                candidates.append((candidate, is_directory))

            matched: list[ProjectRule] = []
            for rule in sorted(
                self._rules.values(),
                key=lambda item: (-_rule_specificity(item), item.display_path),
            ):
                if rule.patterns in (None, ()) or rule.spec is None:
                    continue
                if rule.identity in self._claimed:
                    continue
                values: list[str] = []
                for candidate, is_directory in candidates:
                    relative = _relative_posix(candidate, rule.match_root)
                    if relative is None or relative == "":
                        continue
                    values.append(relative)
                    if is_directory:
                        values.append(relative.rstrip("/") + "/")
                if any(rule.spec.match_file(value) for value in values):
                    matched.append(rule)
            return self._claim_and_format(
                matched,
                max_chars=max_chars,
                defer_oversize=defer_oversize,
            )

    def _claim_and_format(
        self,
        rules: list[ProjectRule],
        *,
        max_chars: Optional[int] = None,
        defer_oversize: bool = False,
    ) -> str:
        limit = (
            self.max_lazy_chars
            if max_chars is None
            else max(0, int(max_chars))
        )
        if limit <= 0:
            return ""

        selected: list[tuple[ProjectRule, str]] = []
        deferred: list[ProjectRule] = []
        for rule in rules:
            if defer_oversize:
                normal = self._format_rule(
                    rule,
                    max_chars=min(self.max_rule_chars, self.max_lazy_chars),
                )
                if len(normal) > limit:
                    deferred.append(rule)
                    continue
            formatted = self._format_rule(rule, max_chars=min(self.max_rule_chars, limit))
            candidate = "\n\n".join([*(text for _item, text in selected), formatted])
            if formatted and len(candidate) <= limit:
                selected.append((rule, formatted))
            else:
                deferred.append(rule)

        unindexed = sorted(self._unindexed_rule_paths - self._reported_unindexed_paths)
        while True:
            deferred_paths = [rule.display_path for rule in deferred] + unindexed
            marker = self._format_recovery_marker(deferred_paths, "lazy", limit)
            body = "\n\n".join(text for _rule, text in selected)
            candidate = body + ("\n\n" if body and marker else "") + marker
            if len(candidate) <= limit:
                break
            if selected:
                rule, _formatted = selected.pop()
                deferred.insert(0, rule)
                continue
            candidate = self._format_recovery_marker(deferred_paths, "lazy", limit)
            break

        self._claimed.update(rule.identity for rule, _formatted in selected)
        if marker and unindexed:
            self._reported_unindexed_paths.update(unindexed)
        return candidate

    def _bounded_text(self, rules: list[ProjectRule], limit: int, label: str) -> str:
        if limit <= 0:
            return ""

        selected: list[tuple[ProjectRule, str]] = []
        deferred: list[ProjectRule] = []
        for rule in rules:
            formatted = self._format_rule(rule, max_chars=min(self.max_rule_chars, limit))
            candidate = "\n\n".join([*(text for _item, text in selected), formatted])
            if formatted and len(candidate) <= limit:
                selected.append((rule, formatted))
            else:
                deferred.append(rule)

        unindexed = sorted(self._unindexed_rule_paths - self._reported_unindexed_paths)
        while True:
            deferred_paths = [rule.display_path for rule in deferred] + unindexed
            marker = self._format_recovery_marker(deferred_paths, label, limit)
            body = "\n\n".join(text for _rule, text in selected)
            candidate = body + ("\n\n" if body and marker else "") + marker
            if len(candidate) <= limit:
                break
            if selected:
                rule, _formatted = selected.pop()
                deferred.insert(0, rule)
                continue
            candidate = self._format_recovery_marker(deferred_paths, label, limit)
            break

        if marker and unindexed:
            self._reported_unindexed_paths.update(unindexed)
        return candidate

    @staticmethod
    def _format_recovery_marker(paths: list[str], label: str, limit: int) -> str:
        if not paths or limit <= 0:
            return ""
        unique_paths = list(dict.fromkeys(paths))
        prefix = f"[Project rules deferred by {label} budget; use read_file: "
        suffix = "]"
        full = prefix + "; ".join(unique_paths) + suffix
        if len(full) <= limit:
            return full

        compact_prefix = "[deferred; read_file: "
        compact = compact_prefix + "; ".join(unique_paths) + suffix
        if len(compact) <= limit:
            return compact

        included: list[str] = []
        for index, path in enumerate(unique_paths):
            remaining = len(unique_paths) - index - 1
            tail = (
                f"; +{remaining} more under .claude/rules or .hermes/rules]"
                if remaining
                else suffix
            )
            candidate = prefix + "; ".join([*included, path]) + tail
            if len(candidate) > limit:
                break
            included.append(path)
        if included:
            remaining = len(unique_paths) - len(included)
            tail = (
                f"; +{remaining} more under .claude/rules or .hermes/rules]"
                if remaining
                else suffix
            )
            return prefix + "; ".join(included) + tail

        fallback = (
            f"[Project rules deferred by {label} budget: {len(unique_paths)} rule(s); "
            "inspect .claude/rules and .hermes/rules with search_files, then read_file.]"
        )
        return fallback[:limit]

    def _format_rule(self, rule: ProjectRule, *, max_chars: Optional[int] = None) -> str:
        limit = self.max_rule_chars if max_chars is None else max(0, max_chars)
        if limit <= 0:
            return ""
        header = f"[Project context rule: {rule.display_path}]\n"
        full = header + rule.body
        if len(full) <= limit:
            return full

        marker = (
            f"\n\n[Rule body truncated: {len(rule.body):,} chars; "
            f"use read_file {rule.display_path} for the full rule.]"
        )
        available = limit - len(header) - len(marker)
        if available <= 0:
            recovery = (
                f"[Project rule exceeds attachment budget; "
                f"use read_file {rule.display_path}.]"
            )
            return recovery[:limit]
        return header + rule.body[:available] + marker

    def _discover_roots_for_directory(self, directory: Path) -> None:
        """Discover rule directories at *directory* without leaving workspace."""

        try:
            directory = directory.resolve(strict=False)
            relative = directory.relative_to(self.workspace_root)
        except (OSError, ValueError, RuntimeError):
            return
        current = self.workspace_root
        self._discover_rule_roots_at(current)
        for part in relative.parts:
            current = current / part
            self._discover_rule_roots_at(current)

    def _discover_roots_for_path(self, path: Path) -> None:
        try:
            directory = path if path.is_dir() else path.parent
        except OSError:
            directory = path.parent
        self._discover_roots_for_directory(directory)

    def _discover_rule_roots_at(self, directory: Path) -> None:
        for relative_name in RULE_DIRECTORY_NAMES:
            self._discover_rule_root(directory / relative_name)

    def _discover_rule_root(self, rule_root: Path) -> None:
        try:
            canonical_root = rule_root.resolve(strict=False)
        except (OSError, RuntimeError):
            return
        if canonical_root in self._known_rule_roots:
            return
        self._known_rule_roots.add(canonical_root)
        try:
            if not rule_root.is_dir():
                return
        except OSError:
            return
        if not _is_within(canonical_root, self.workspace_root):
            return
        try:
            # Match semantics are relative to the logical project containing
            # ``.claude``/``.hermes``, not to a symlink target used for storage.
            match_root = rule_root.parent.parent.resolve(strict=True)
        except (OSError, RuntimeError):
            return
        if not _is_within(match_root, self.workspace_root):
            return

        try:
            files = sorted(
                (path for path in rule_root.rglob("*") if path.suffix.lower() in {".md", ".mdc"}),
                key=lambda path: path.as_posix(),
            )
        except OSError as exc:
            logger.debug("Could not scan project rules under %s: %s", rule_root, exc)
            return

        for file_index, path in enumerate(files):
            if len(self._rules) >= self.max_rule_count:
                for unindexed_path in files[file_index:]:
                    display = (
                        _relative_posix(unindexed_path, match_root)
                        or unindexed_path.as_posix()
                    )
                    self._unindexed_rule_paths.add(display)
                logger.warning("Project rule limit reached at %d files", self.max_rule_count)
                break
            try:
                if not path.is_file():
                    continue
                identity = path.resolve(strict=True)
                if not _is_within(identity, self.workspace_root) or not _is_within(identity, canonical_root):
                    logger.warning("Skipping project rule outside its rule root: %s", path)
                    continue
                if identity in self._rules:
                    continue
                raw = path.read_text(encoding="utf-8")
                parsed = _strict_frontmatter(raw)
                if parsed is None:
                    logger.warning("Skipping malformed project rule: %s", path)
                    continue
                metadata, body = parsed
                body = body.strip()
                if not body:
                    continue
                patterns = _normalise_patterns(metadata)
                spec = None
                if patterns:
                    try:
                        spec = PathSpec.from_lines("gitignore", patterns)
                    except Exception as exc:
                        logger.warning("Skipping project rule with invalid paths in %s: %s", path, exc)
                        patterns = ()
                elif patterns == ():
                    spec = None
                display = _relative_posix(path, match_root) or path.as_posix()
                try:
                    body = self.scanner(body, display)
                except Exception as exc:
                    logger.warning("Project rule scanner failed for %s: %s", display, exc)
                    continue
                body = body.strip()
                if not body:
                    continue
                self._rules[identity] = ProjectRule(
                    identity=identity,
                    display_path=display,
                    match_root=match_root,
                    body=body,
                    patterns=patterns,
                    spec=spec,
                )
            except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
                logger.debug("Could not load project rule %s: %s", path, exc)
