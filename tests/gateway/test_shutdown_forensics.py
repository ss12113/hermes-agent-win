"""Tests for gateway.shutdown_forensics — fast snapshot + async diag spawn."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from gateway import shutdown_forensics as sf

# ---------------------------------------------------------------------------
# _signal_name
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# snapshot_shutdown_context
# ---------------------------------------------------------------------------

class TestSnapshotShutdownContext:

    def test_detects_takeover_marker_for_self(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        marker = tmp_path / ".gateway-takeover.json"
        marker.write_text(
            f'{{"target_pid": {os.getpid()}, "replacer_pid": 99999}}',
            encoding="utf-8",
        )
        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        assert "takeover_marker" in ctx
        assert ctx["takeover_marker_for_self"] is True

# ---------------------------------------------------------------------------
# format_context_for_log / context_as_json
# ---------------------------------------------------------------------------

class TestFormatters:

    def test_context_as_json_handles_unserialisable_values(self):
        ctx = {"signal": "SIGTERM", "weird": object()}
        payload = sf.context_as_json(ctx)
        # default=str means objects get repr'd, JSON stays valid
        decoded = json.loads(payload)
        assert decoded["signal"] == "SIGTERM"
        assert "weird" in decoded

# ---------------------------------------------------------------------------
# persisted snapshots must never include process argv (#112459)
# ---------------------------------------------------------------------------

_ARGV_CANARY = "lin_api_CANARY_SHUTDOWN_FORENSICS_9f3a2c"

@pytest.fixture
def child_with_secret_argv():
    """A live child whose argv carries a token-shaped value, like ``docker exec -e KEY=...``."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", f"--token={_ARGV_CANARY}"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait()

class TestArgvFreePersistence:

    @pytest.mark.platforms("linux")
    def test_snapshot_and_log_line_identify_process_without_argv(self, child_with_secret_argv):
        """/proc-backed summaries keep pid/name/ppid/state but never the command line, so neither
        the JSON snapshot nor the warning line can carry a credential from a parent's argv."""
        summary = sf._proc_summary(child_with_secret_argv.pid)
        assert summary["pid"] == child_with_secret_argv.pid
        assert summary["name"]  # identity survives
        assert "cmdline" not in summary

        ctx = sf.snapshot_shutdown_context(signal.SIGTERM)
        ctx["parent"] = summary
        line = sf.format_context_for_log(ctx)
        assert _ARGV_CANARY not in line and _ARGV_CANARY not in sf.context_as_json(ctx)
        assert f"parent_pid={child_with_secret_argv.pid}" in line

# ---------------------------------------------------------------------------
# spawn_async_diagnostic
# ---------------------------------------------------------------------------

class TestSpawnAsyncDiagnostic:
    @pytest.mark.platforms("linux")
    def test_spawns_subprocess_and_writes_output(self, tmp_path):
        self._assert_diagnostic_written(tmp_path)

    @pytest.mark.platforms("macos")
    def test_spawns_without_gnu_timeout_on_macos(self, tmp_path):
        """Stock macOS has no ``timeout`` binary and BSD ``ps``; the diagnostic still lands."""
        self._assert_diagnostic_written(tmp_path)

    @staticmethod
    def _assert_diagnostic_written(tmp_path):
        log_path = tmp_path / "diag.log"
        pid = sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=3.0)
        assert pid is not None and pid > 0

        # Wait briefly for the subprocess to write — bounded by its own timeout.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if log_path.exists() and log_path.stat().st_size > 0:
                # Wait a touch longer for the script to finish writing
                time.sleep(0.2)
                break
            time.sleep(0.1)

        # Reap the subprocess so it doesn't show up as a zombie.
        try:
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            pass

        assert log_path.exists()
        contents = log_path.read_text(encoding="utf-8", errors="replace")
        assert "shutdown diagnostic" in contents
        assert "SIGTERM" in contents
        lines = contents.splitlines()
        ps_section = lines[lines.index("--- ps (top 60 by cpu, comm only) ---") + 1:]
        assert ps_section and ps_section[0].split()[:2] == ["PID", "PPID"], \
            "ps column header must lead the listing, not sort as a 0.0-cpu row"

    @pytest.mark.platforms("linux")
    def test_diagnostic_log_omits_child_argv_and_is_owner_only(self, tmp_path, child_with_secret_argv):
        """The detached ps/pstree walk must not write any process's argv to disk, and the log
        (even one created 0644 by an earlier release) ends up owner-only."""
        log_path = tmp_path / "diag.log"
        log_path.write_text("prior\n", encoding="utf-8")
        os.chmod(log_path, 0o644)

        pid = sf.spawn_async_diagnostic(log_path, "SIGTERM", timeout_seconds=5.0)
        assert pid is not None
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    break
            except ChildProcessError:
                break
            time.sleep(0.1)

        contents = log_path.read_text(encoding="utf-8", errors="replace")
        assert "shutdown diagnostic" in contents
        assert _ARGV_CANARY not in contents
        assert (log_path.stat().st_mode & 0o777) == 0o600

# ---------------------------------------------------------------------------
# parse_systemd_duration_to_us
# ---------------------------------------------------------------------------

class TestParseSystemdDuration:
    def test_seconds(self):
        assert sf.parse_systemd_duration_to_us("90s") == 90 * 1_000_000

    def test_minutes(self):
        assert sf.parse_systemd_duration_to_us("3min") == 180 * 1_000_000

# ---------------------------------------------------------------------------
# check_systemd_timing_alignment
# ---------------------------------------------------------------------------

class TestCheckSystemdTimingAlignment:

    def test_returns_none_when_unit_undeterminable(self, monkeypatch):
        monkeypatch.setenv("INVOCATION_ID", "abc")
        # /proc/self/cgroup likely doesn't end in .service for the test runner
        result = sf.check_systemd_timing_alignment(180.0)
        # Either None (we couldn't find a unit) or a dict with mismatch info
        # for whatever unit pytest IS in.  Both are valid; we just ensure
        # the function doesn't raise.
        assert result is None or isinstance(result, dict)


def _patch_open_for_cgroup(monkeypatch, cgroup_line):
    """Route /proc/self/cgroup reads to a synthetic line so the unit-name
    detection is deterministic regardless of the test runner's cgroup."""
    import builtins
    import io

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/self/cgroup":
            return io.StringIO(cgroup_line + "\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def _patch_systemctl(monkeypatch, responses):
    """Fake systemctl(1) output keyed by (domain, property).

    responses: dict[str, tuple[str, int]] where key is
    'user|LoadState' / 'user|TimeoutStopUSec' / 'system|...' and value
    is (stdout, returncode).  Any unmatched call raises so a regression
    that stops querying a domain fails loudly instead of silently passing.
    """
    import subprocess
    from types import SimpleNamespace

    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        flags = "user" if "--user" in cmd else "system"
        prop = None
        for arg in cmd:
            if arg.startswith("--property="):
                prop = arg.split("=", 1)[1]
        if prop is None:
            raise AssertionError(f"unexpected systemctl call: {cmd}")
        key = f"{flags}|{prop}"
        if key not in responses:
            raise AssertionError(f"unexpected systemctl call: {cmd} (key={key})")
        stdout, rc = responses[key]
        return SimpleNamespace(stdout=stdout, returncode=rc)

    monkeypatch.setattr(subprocess, "run", fake_run)


class TestCheckSystemdTimingAlignmentSystemdUnits:

    CGROUP = "0::/system.slice/hermes-gateway.service"

    def test_system_owned_unit_reports_real_timeout_not_user_default(
        self, monkeypatch
    ):
        """Regression: systemd returns a *default* TimeoutStopUSec (1min 30s)
        with exit 0 for unloaded units.  A system-owned unit with no user
        file used to be misread as 90s and logged a stale-unit warning even
        though the system unit was correct (630s).  The checker must only
        trust a manager that actually has the unit loaded."""
        monkeypatch.setenv("INVOCATION_ID", "synthetic-invocation")
        _patch_open_for_cgroup(monkeypatch, self.CGROUP)
        _patch_systemctl(
            monkeypatch,
            {
                # user domain: unit NOT loaded -> systemd default value
                "user|LoadState": ("LoadState=not-found\n", 0),
                "user|TimeoutStopUSec": ("TimeoutStopUSec=1min 30s\n", 0),
                # system domain: the real owner, correctly configured
                "system|LoadState": ("LoadState=loaded\n", 0),
                "system|TimeoutStopUSec": ("TimeoutStopUSec=10min 30s\n", 0),
            },
        )
        result = sf.check_systemd_timing_alignment(600.0)
        assert result is not None
        assert result["mismatch"] is False
        assert result["timeout_stop_sec"] == 630.0

    def test_user_loaded_stale_unit_still_reports_mismatch(self, monkeypatch):
        """A genuinely stale user-domain unit (loaded, 90s) must still be
        caught: the LoadState gate is not a blanket 'ignore user domain'."""
        monkeypatch.setenv("INVOCATION_ID", "synthetic-invocation")
        _patch_open_for_cgroup(monkeypatch, self.CGROUP)
        _patch_systemctl(
            monkeypatch,
            {
                "user|LoadState": ("LoadState=loaded\n", 0),
                "user|TimeoutStopUSec": ("TimeoutStopUSec=1min 30s\n", 0),
            },
        )
        result = sf.check_systemd_timing_alignment(600.0)
        assert result is not None
        assert result["mismatch"] is True
        assert result["timeout_stop_sec"] == 90.0

    def test_no_loaded_manager_returns_none(self, monkeypatch):
        """If neither manager has the unit loaded, the defaults must not be
        trusted: report nothing rather than a false stale-unit warning."""
        monkeypatch.setenv("INVOCATION_ID", "synthetic-invocation")
        _patch_open_for_cgroup(monkeypatch, self.CGROUP)
        _patch_systemctl(
            monkeypatch,
            {
                "user|LoadState": ("LoadState=not-found\n", 0),
                "system|LoadState": ("LoadState=not-found\n", 0),
            },
        )
        result = sf.check_systemd_timing_alignment(600.0)
        assert result is None
