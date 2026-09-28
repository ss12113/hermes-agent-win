"""Contract tests for tools.shadow_store — the /rollback shadow reader/restorer.

The store fixture mirrors the hook's on-disk shape exactly (manifest.jsonl +
objects/<sha256(path)[:16]>/<ts>.snap), so these tests pin the cross-module
contract: what shadow_checkpoint.py writes is what /rollback shadow reads.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from tools import shadow_store


def _seed(store: Path, session: str, path: str, content: bytes,
          tool: str = "write_file", ts: Optional[float] = None) -> None:
    """Append one snapshot record the way the pre-edit hook does."""
    key = hashlib.sha256(path.encode()).hexdigest()[:16]
    objdir = store / session / "objects" / key
    objdir.mkdir(parents=True, exist_ok=True)
    obj = objdir / f"seed-{time.time_ns()}.snap"
    obj.write_bytes(content)
    rec = {"ts": ts if ts is not None else time.time(), "tool": tool, "path": path,
           "key": key, "obj": str(obj), "bytes": len(content)}
    with open(store / session / "manifest.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


class TestLoadAndFilter:
    def test_rows_across_sessions_sorted_oldest_first(self, tmp_path):
        store = tmp_path / "shadow"
        _seed(store, "s2", "/work/b.py", b"b2", ts=3.0)
        _seed(store, "s1", "/work/a.py", b"a1", ts=1.0)
        _seed(store, "s2", "/work/a.py", b"a2", ts=2.0)
        rows = shadow_store.load_rows(store)
        assert [r["bytes"] for r in rows] == [2, 2, 2]
        assert [r["ts"] for r in rows] == [1.0, 2.0, 3.0]
        assert {r["session"] for r in rows} == {"s1", "s2"}

    def test_cwd_prefix_and_substring_filters(self, tmp_path):
        store = tmp_path / "shadow"
        _seed(store, "s", str(tmp_path / "proj" / "x.py"), b"x", ts=1.0)
        _seed(store, "s", str(tmp_path / "other" / "y.py"), b"y", ts=2.0)
        rows = shadow_store.load_rows(store)
        under_cwd = shadow_store.filter_rows(rows, cwd=str(tmp_path / "proj"))
        assert [r["path"] for r in under_cwd] == [str(tmp_path / "proj" / "x.py")]
        by_substr = shadow_store.filter_rows(rows, path_filter="y.py")
        assert [r["path"] for r in by_substr] == [str(tmp_path / "other" / "y.py")]


class TestRunCommand:
    def test_listing_shows_revs_with_true_indices_when_capped(self, tmp_path):
        store = tmp_path / "shadow"
        target = str(tmp_path / "f.py")
        for i in range(shadow_store._LIST_CAP + 3):
            _seed(store, "s", target, f"v{i}".encode(), ts=float(i))
        out = shadow_store.run_shadow_command([], str(tmp_path), root=store)
        assert "(+3 older omitted)" in out
        # The newest row keeps its true rev index even though older rows are hidden.
        assert f"rev {shadow_store._LIST_CAP + 2:>3}" in out

    def test_empty_store_message(self, tmp_path):
        out = shadow_store.run_shadow_command([], str(tmp_path), root=tmp_path / "shadow")
        assert "No shadow revisions recorded yet" in out

    def test_no_cwd_rows_falls_back_to_overall(self, tmp_path):
        store = tmp_path / "shadow"
        _seed(store, "s", "/elsewhere/z.py", b"z", ts=1.0)
        out = shadow_store.run_shadow_command([], str(tmp_path), root=store)
        assert "No shadow revisions under" in out
        assert "/elsewhere/z.py" in out


class TestRestore:
    def test_restore_writes_snapshot_and_keeps_current_as_pre_restore(self, tmp_path):
        store = tmp_path / "shadow"
        live = tmp_path / "live.py"
        live.write_bytes(b"NEW-version")
        _seed(store, "s", str(live), b"OLD-version", ts=1.0)
        out = shadow_store.run_shadow_command(["restore"], str(tmp_path), root=store)
        assert "restored" in out
        assert live.read_bytes() == b"OLD-version"  # rolled back
        rows = shadow_store.load_rows(store)
        safety = [r for r in rows if r["tool"] == "pre-restore"]
        assert len(safety) == 1
        assert Path(safety[0]["obj"]).read_bytes() == b"NEW-version"  # reversal kept

    def test_wrong_rollback_is_itself_reversible(self, tmp_path):
        store = tmp_path / "shadow"
        live = tmp_path / "live.py"
        live.write_bytes(b"v3")
        _seed(store, "s", str(live), b"v1", ts=1.0)
        _seed(store, "s", str(live), b"v2", ts=2.0)
        shadow_store.run_shadow_command(["restore", "0"], str(tmp_path), root=store)
        assert live.read_bytes() == b"v1"
        shadow_store.run_shadow_command(["restore"], str(tmp_path), root=store)  # newest = pre-restore v3
        assert live.read_bytes() == b"v3"

    def test_bare_restore_scope_matches_listing(self, tmp_path):
        """A bare restore resolves within the cwd set the listing showed, never the
        global rev 0 (a live-store E2E caught that grabbing another directory's file)."""
        store = tmp_path / "shadow"
        cwd_file = tmp_path / "proj" / "mine.py"
        other_file = tmp_path / "other" / "other.py"
        _seed(store, "s", str(cwd_file), b"mine-v1", ts=1.0)
        _seed(store, "s", str(other_file), b"other-v1", ts=2.0)
        out = shadow_store.run_shadow_command(["restore", "1"], str(tmp_path / "proj"), root=store)
        assert "rev out of range 0..0" in out
        out = shadow_store.run_shadow_command(["restore", "0"], str(tmp_path / "proj"), root=store)
        assert "restored" in out
        assert cwd_file.read_bytes() == b"mine-v1"
        assert not other_file.exists()

    def test_restore_out_of_range(self, tmp_path):
        store = tmp_path / "shadow"
        _seed(store, "s", str(tmp_path / "a.py"), b"a", ts=1.0)
        out = shadow_store.run_shadow_command(["restore", "5"], str(tmp_path), root=store)
        assert "rev out of range 0..0" in out

    def test_missing_object_reported(self, tmp_path):
        store = tmp_path / "shadow"
        live = tmp_path / "live.py"
        _seed(store, "s", str(live), b"x", ts=1.0)
        for obj in (store / "s" / "objects").rglob("*.snap"):
            obj.unlink()
        out = shadow_store.run_shadow_command(["restore"], str(tmp_path), root=store)
        assert "Snapshot object missing" in out


class TestShow:
    def test_show_prints_content_and_truncates(self, tmp_path):
        store = tmp_path / "shadow"
        big = "\n".join(f"line {i}" for i in range(shadow_store._SHOW_MAX_LINES + 10))
        _seed(store, "s", str(tmp_path / "f.py"), big.encode(), ts=1.0)
        out = shadow_store.run_shadow_command(["show"], str(tmp_path), root=store)
        assert "line 0" in out
        assert f"+{10} more lines" in out
        assert "line 89" not in out


class TestHint:
    def test_hint_mentions_counts_and_command(self, tmp_path):
        store = tmp_path / "shadow"
        _seed(store, "s", str(tmp_path / "proj" / "a.py"), b"a", ts=1.0)
        _seed(store, "s", str(tmp_path / "proj" / "a.py"), b"b", ts=2.0)
        hint = shadow_store.shadow_hint(str(tmp_path / "proj"), root=store)
        assert "/rollback shadow" in hint
        assert "2" in hint

    def test_hint_empty_outside_scope(self, tmp_path):
        store = tmp_path / "shadow"
        _seed(store, "s", "/elsewhere/a.py", b"a", ts=1.0)
        assert shadow_store.shadow_hint(str(tmp_path), root=store) == ""
