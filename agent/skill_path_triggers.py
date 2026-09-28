"""Path-triggered skill hints: a skill's SKILL.md may declare ``paths:`` — a string or a
list of gitignore-style patterns. When a tool call touches a matching path, the lazy
context attachment names the skill so it can be loaded with ``skill_view``. This mirrors
scoped project rules (``agent/project_rules.py``): the same gitwildmatch semantics and the
same attachment slot, but the payload is a pointer to a skill, never the skill body — the
resident skills index already carries descriptions, so the hint only has to connect
"this path" to "that skill" at the moment it becomes relevant.

Patterns match the touched path's absolute POSIX form and, when the path lives under the
workspace, its workspace-relative form — so both ``/srv/app/**`` and ``src/**`` authoring
styles work. A malformed ``paths:`` value skips that skill only; a bad pattern must never
promote a skill to matching everything.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, Optional

from pathspec import PathSpec

from agent.skill_utils import iter_skill_index_files, parse_frontmatter

logger = logging.getLogger(__name__)

#: Ceiling for one hint section; the resident index already carries descriptions, so the
#: hint stays a pointer (a few names), never a second index.
MAX_SECTION_CHARS = 320
#: At most this many skills are named per tool call; more would read as an index dump.
MAX_HINTED_SKILLS = 3
#: Below this many remaining characters the section is skipped instead of truncated.
MIN_REMAINING_CHARS = 60
_CACHE_TTL_SECONDS = 30.0

#: ``str(skills_dir)`` -> ``(loaded_at, [(skill_name, spec, pattern_text)])``
_CACHE: dict = {}


def reset_cache() -> None:
    """Drop the scan cache (tests, or a caller that just edited a skill's ``paths:``)."""
    _CACHE.clear()


def _collect_patterns(raw) -> list:
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        return [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    return []


def _load_specs(skills_dir: Path) -> list:
    """Scan the skills tree for ``paths:`` declarations, cached for a short TTL."""
    key = str(skills_dir)
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    specs: list = []
    if skills_dir.exists():
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            try:
                frontmatter, _ = parse_frontmatter(
                    skill_file.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            patterns = _collect_patterns(frontmatter.get("paths"))
            if not patterns:
                continue
            try:
                spec = PathSpec.from_lines("gitwildmatch", patterns)
            except Exception:
                logger.debug("Ignoring malformed paths: in %s", skill_file, exc_info=True)
                continue
            name = str(frontmatter.get("name") or "").strip() or skill_file.parent.name
            specs.append((name, spec, ", ".join(patterns)))
    specs.sort(key=lambda item: item[0])
    _CACHE[key] = (now, specs)
    return specs


def _candidate_forms(path: Path, working_dir: Path) -> list:
    forms = [path.as_posix()]
    try:
        forms.append(path.relative_to(working_dir).as_posix())
    except ValueError:
        pass
    return forms


def format_touched_skill_hints(
    touched: Iterable[Path],
    working_dir: Path,
    *,
    skills_dir: Optional[Path] = None,
    max_chars: int = MAX_SECTION_CHARS,
    claimed: Optional[set] = None,
) -> str:
    """Return the skills hint section for *touched* paths ('' when nothing matches).

    ``claimed`` gives the caller session-scoped exactly-once semantics (matching the
    scoped-rule claim bookkeeping): names already in the set are skipped, and newly
    emitted names are added to it.
    """
    limit = min(max_chars, MAX_SECTION_CHARS)
    if limit < MIN_REMAINING_CHARS:
        return ""
    paths = list(touched)
    if not paths:
        return ""
    if skills_dir is None:
        from hermes_constants import get_skills_dir
        skills_dir = get_skills_dir()
    specs = _load_specs(Path(skills_dir))
    if not specs:
        return ""

    hits: list = []
    seen: set = set(claimed) if claimed else set()
    for name, spec, pattern_text in specs:
        if name in seen:
            continue
        for path in paths:
            if any(spec.match_file(form) for form in _candidate_forms(path, working_dir)):
                hits.append((name, pattern_text))
                seen.add(name)
                break
        if len(hits) >= MAX_HINTED_SKILLS:
            break
    if not hits:
        return ""
    if claimed is not None:
        claimed.update(name for name, _ in hits)
    lines = ["Skills declared for these paths (load with skill_view(name=...)):"]
    lines.extend(f"- {name}  [{pattern_text}]" for name, pattern_text in hits)
    return "\n".join(lines)[:limit]
