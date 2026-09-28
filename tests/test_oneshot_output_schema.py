"""--output-schema on the one-shot path: flag surface, offline failures, exit codes."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Launch through the interpreter running this test — the same entry point tests/e2e/** uses.
# A gate candidate lives in a git worktree that has no ``venv/bin/hermes``, so pinning the
# console script made these three subprocess tests fail there while passing in the deployed
# tree (they only exercised flag parsing and the two pre-agent exit paths).
HERMES = [sys.executable, "-m", "hermes_cli.main"]

SCHEMA = json.dumps({
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
    "additionalProperties": False,
})


def _run(*args, hermes_home, timeout=180):
    # A throwaway HERMES_HOME keeps the child CLI off the real ~/.hermes (profiles, caches).
    env = {**os.environ, "HERMES_HOME": str(hermes_home)}
    return subprocess.run([*HERMES, *args], capture_output=True, text=True,
                          timeout=timeout, cwd=str(ROOT), env=env)


def test_flag_is_advertised(tmp_path):
    out = _run("--help", hermes_home=tmp_path).stdout
    assert "--output-schema" in out


def test_inline_non_json_schema_exits_2(tmp_path):
    r = _run("-z", "hi", "--output-schema", "{not json", hermes_home=tmp_path)
    assert r.returncode == 2
    assert "invalid --output-schema" in r.stderr


def test_missing_schema_file_exits_2(tmp_path):
    r = _run("-z", "hi", "--output-schema", "/nope/missing.json", hermes_home=tmp_path)
    assert r.returncode == 2
    assert "cannot read --output-schema file" in r.stderr


def test_schema_failure_maps_to_exit_3():
    sys.path.insert(0, str(ROOT))
    from hermes_cli.oneshot import _oneshot_exit_code

    assert _oneshot_exit_code("some text", {"output_schema_ok": False, "completed": True}) == 3


def test_schema_success_keeps_exit_0():
    sys.path.insert(0, str(ROOT))
    from hermes_cli.oneshot import _oneshot_exit_code

    assert _oneshot_exit_code('{"answer": "x"}', {"output_schema_ok": True, "completed": True}) == 0
