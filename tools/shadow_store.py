"""Shadow-layer file revisions — read/restore companion for the pre-edit hook store.

The ``pre_tool_call`` hook (``shadow_checkpoint.py``) records the PRE-edit bytes
of every file the agent is about to modify::

    <shadow-root>/<session>/manifest.jsonl      one JSON line per snapshot
    <shadow-root>/<session>/objects/<key>/<ts>.snap

That layer is the only history for files built-in checkpoints refuse to track
(e.g. trees under the service home or over the 50k-file cap), so ``/rollback
shadow`` surfaces it when the checkpoint store has nothing for a directory.
Restores snapshot the CURRENT content first (``tool=pre-restore``), so a wrong
rollback is itself reversible. Set ``HERMES_SHADOW_HOME`` to override the store
root (same contract as the hook).
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from hermes_constants import get_hermes_home

_LIST_CAP = 40
_SHOW_MAX_LINES = 80
_MAX_FILE_BYTES = 16 * 1024 * 1024  # mirror the hook's skip threshold
_SAFETY_SESSION = "rollback-safety"


def store_root(root=None) -> Path:
    if root is not None:
        return Path(root)
    override = os.environ.get("HERMES_SHADOW_HOME")
    return Path(override) if override else get_hermes_home() / "shadow"


def load_rows(root=None) -> List[dict]:
    """Every manifest record across sessions, oldest → newest, session-decorated."""
    rows: List[dict] = []
    for manifest in sorted(store_root(root).glob("*/manifest.jsonl")):
        session = manifest.parent.name
        try:
            with open(manifest, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    record["session"] = session
                    rows.append(record)
        except OSError:
            continue
    rows.sort(key=lambda r: r.get("ts", 0.0))
    return rows


def filter_rows(rows: List[dict], *, cwd: Optional[str] = None,
                path_filter: Optional[str] = None) -> List[dict]:
    out = rows
    if cwd:
        prefix = str(Path(cwd).resolve()).rstrip("/") + "/"
        out = [r for r in out if str(r.get("path", "")).startswith(prefix)]
    if path_filter:
        out = [r for r in out if path_filter in str(r.get("path", ""))]
    return out


def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _row_line(index: int, record: dict) -> str:
    return (f"  rev {index:>3}  {_fmt_ts(record.get('ts', 0.0))}  "
            f"{record.get('bytes', 0):>9}B  {str(record.get('tool', '?')):<9} "
            f"{record.get('path', '?')}")


def format_rows(rows: List[dict]) -> str:
    return "\n".join(_row_line(i, r) for i, r in enumerate(rows))


def _store_snapshot(path: Path, session: str, tool: str, *, root=None) -> bool:
    """Mirror of the hook's writer: same key, object naming and manifest shape."""
    try:
        if not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
            return False
        data = path.read_bytes()
        key = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        base = store_root(root) / session
        objdir = base / "objects" / key
        objdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"-{int(time.time_ns() % 10**9):09d}"
        obj = objdir / f"{stamp}.snap"
        obj.write_bytes(data)
        line = {"ts": time.time(), "tool": tool, "path": str(path), "key": key,
                "obj": str(obj), "bytes": len(data)}
        with open(base / "manifest.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def restore_row(record: dict, *, root=None, dry_run: bool = False) -> str:
    """Write one snapshot back. Current content is snapshotted first so the
    rollback itself is reversible; ``dry_run`` only reports the plan."""
    obj = Path(str(record.get("obj", "")))
    target = Path(str(record.get("path", "")))
    label = f"rev {record.get('tool', '?')} @ {_fmt_ts(record.get('ts', 0.0))}"
    if not obj.is_file():
        return f"  Snapshot object missing (pruned?): {obj}"
    if dry_run:
        return f"  [dry] would restore {label} -> {target}"
    if target.is_file():
        _store_snapshot(target, _SAFETY_SESSION, "pre-restore", root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(obj.read_bytes())
    return f"  restored {label} -> {target} (previous content kept as a pre-restore snapshot)"


def shadow_hint(cwd: str, *, root=None) -> str:
    """One-line pointer for checkpoint listings that found nothing; '' when the
    shadow layer has no revisions under *cwd* either."""
    rows = filter_rows(load_rows(root), cwd=cwd)
    if not rows:
        return ""
    return (f"  Shadow-layer revisions exist here "
            f"({len(rows)} across {len({r.get('path') for r in rows})} file(s)) — "
            f"see /rollback shadow")


def run_shadow_command(args: List[str], cwd: str, *, root=None) -> str:
    """Render a /rollback shadow invocation: list, show, or restore."""
    sub = args[0].lower() if args else ""
    if sub in ("restore", "show"):
        tail = list(args[1:])
        rev: Optional[int] = None
        if tail and tail[0].lstrip("+-").isdigit():
            rev = int(tail.pop(0))
        path_filter = " ".join(tail) or None
        # Same scope as the listing the revs came from: a bare restore/show resolves
        # within the cwd set the bare listing shows; a path substring searches the
        # whole store. (A live-store E2E caught the earlier version grabbing the
        # global rev 0 — a different directory's file.)
        if path_filter:
            rows = filter_rows(load_rows(root), path_filter=path_filter)
        else:
            rows = filter_rows(load_rows(root), cwd=cwd)
        if not rows:
            return "  No shadow revisions found" + (
                f" for {path_filter!r}" if path_filter else f" under {Path(cwd).resolve()}")
        if rev is None:
            rev = len(rows) - 1
        if not (0 <= rev < len(rows)):
            return f"  rev out of range 0..{len(rows) - 1}"
        record = rows[rev]
        if sub == "restore":
            return restore_row(record, root=root)
        obj = Path(str(record.get("obj", "")))
        if not obj.is_file():
            return f"  Snapshot object missing (pruned?): {obj}"
        lines = obj.read_text(encoding="utf-8", errors="replace").splitlines()
        head = "\n".join(lines[:_SHOW_MAX_LINES])
        more = (f"\n  … (+{len(lines) - _SHOW_MAX_LINES} more lines — "
                f"restore with /rollback shadow restore {rev})") if len(lines) > _SHOW_MAX_LINES else ""
        return f"  {record.get('path', '?')}  (rev {rev}, {_fmt_ts(record.get('ts', 0.0))}):\n{head}{more}"

    path_filter = " ".join(args) or None
    if path_filter:
        rows = filter_rows(load_rows(root), path_filter=path_filter)
        scope = f"paths matching {path_filter!r}"
    else:
        rows = filter_rows(load_rows(root), cwd=cwd)
        scope = str(Path(cwd).resolve())
    if not rows:
        overall = load_rows(root)
        if not overall:
            return "  No shadow revisions recorded yet (the pre-edit hook store is empty)."
        newest = overall[-10:]
        start = len(overall) - len(newest)
        lines = "\n".join(_row_line(start + i, r) for i, r in enumerate(newest))
        return (f"  No shadow revisions under {scope}.\n"
                f"  Newest overall (rev 0 = oldest):\n{lines}\n"
                f"  Filter with /rollback shadow <path-substr>")
    start = max(0, len(rows) - _LIST_CAP)
    lines = "\n".join(_row_line(i, rows[i]) for i in range(start, len(rows)))
    older = f"  … (+{start} older omitted)\n" if start else ""
    return (f"  Shadow revisions under {scope} (rev 0 = oldest):\n{older}{lines}\n"
            f"  Restore: /rollback shadow restore <rev>   Show: /rollback shadow show <rev>")
