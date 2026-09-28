"""Tests for ``hermes sessions fork`` — the CLI face of ``SessionDB.fork_session``.

The command copies a session into a new child and prints the new id + name; the parent must
stay as it was. Error paths (unknown id, blank ``--name``, title in use) print a friendly
message and return a non-zero exit code so scripts can tell.

The seeded store is ``hermes_state._default_db_path()`` on purpose: tests/conftest.py pins
that constant per test (``<tmp>/hermes_test/state.db``), and it is the store ``cmd_sessions``
opens for a writing action — a ``tmp_path/state.db`` seeded by hand is a DIFFERENT file and
the command then (correctly) reports "no such session".
"""

from argparse import Namespace

import hermes_cli.sessions_cmd as sc


def _args(**kw):
    base = dict(sessions_action="fork", session_id=None, name=None)
    base.update(kw)
    return Namespace(**base)


def _store():
    """The session store ``cmd_sessions`` opens (conftest pins the default path per test)."""
    from hermes_state import SessionDB, _default_db_path

    return SessionDB(_default_db_path())


def _seed(db, session_id="20260101_120000_parent"):
    db.create_session(session_id, source="cli", model="test-model")
    db.append_message(session_id, "user", content="hello")
    db.append_message(session_id, "assistant", content="hi")
    return session_id


def _only_child(db, parent):
    children = [row["id"] for row in db.search_sessions() if row.get("parent_session_id") == parent]
    assert len(children) == 1, children
    return children[0]


def test_fork_prints_new_session_id_and_name(capsys):
    db = _store()
    parent = _seed(db)

    rc = sc.cmd_sessions(_args(session_id=parent, name="forked name"))

    out = capsys.readouterr().out
    child = _only_child(db, parent)
    assert rc is None
    assert parent in out and child in out
    assert "forked name" in out
    assert db.get_session_title(child) == "forked name"
    assert db.get_messages_as_conversation(child) == db.get_messages_as_conversation(parent)
    db.close()


def test_fork_accepts_a_unique_id_prefix_and_derives_the_name(capsys):
    db = _store()
    parent = _seed(db)
    db.set_session_title(parent, "my chat")

    rc = sc.cmd_sessions(_args(session_id=parent[:12]))

    out = capsys.readouterr().out
    child = _only_child(db, parent)
    assert rc is None
    assert child in out
    # No --name: the child is numbered into the parent's title lineage.
    assert db.get_session_title(child) == "my chat #2"
    assert "my chat #2" in out
    assert db.get_session_title(parent) == "my chat"
    db.close()


def test_fork_missing_session_returns_1(capsys):
    _store()

    rc = sc.cmd_sessions(_args(session_id="nope_xyz"))

    out = capsys.readouterr().out
    assert rc == 1
    assert "No session 'nope_xyz'" in out and "hermes sessions list" in out


def test_fork_title_conflict_returns_1(capsys):
    db = _store()
    parent = _seed(db)
    db.create_session("20260101_130000_other", source="cli")
    db.set_session_title("20260101_130000_other", "taken name")

    rc = sc.cmd_sessions(_args(session_id=parent, name="taken name"))

    out = capsys.readouterr().out
    assert rc == 1
    assert "already in use" in out
    # The whole fork is rolled back: no unnamed half-child left behind.
    assert db.session_count() == 2
    db.close()


def test_fork_rejects_a_blank_name(capsys):
    db = _store()
    parent = _seed(db)

    rc = sc.cmd_sessions(_args(session_id=parent, name="   "))

    out = capsys.readouterr().out
    assert rc == 1
    assert "--name" in out
    assert db.session_count() == 1
    db.close()


def test_fork_notes_the_live_continuation_of_a_rotated_session(capsys):
    """A compression-rotated session forks its own (pre-rotation) transcript, and the note says
    where the conversation's live continuation is."""
    import json

    db = _store()
    rotated = "20260101_150000_rotated"
    db.create_session(rotated, source="cli")
    db.append_message(rotated, "user", content="pre-rotation turn")
    db.create_session("20260101_150100_tip", source="cli", parent_session_id=rotated)
    db.append_message("20260101_150100_tip", "user", content="post-rotation turn")
    db.end_session(rotated, "compression")

    rc = sc.cmd_sessions(_args(session_id=rotated))

    out = capsys.readouterr().out
    assert rc is None
    assert "rotated by context compression" in out and "20260101_150100_tip" in out
    forked_ids = [r["id"] for r in db.search_sessions()
                  if json.loads(r.get("model_config") or "{}").get("_branched_from") == rotated]
    assert len(forked_ids) == 1
    assert [m["content"] for m in db.get_messages(forked_ids[0])] == ["pre-rotation turn"]
    db.close()


def test_sessions_fork_parses_id_and_name_flag():
    """`hermes sessions fork <id> --name <name>` reaches the handler with both arguments."""
    import argparse

    from hermes_cli.subcommands.sessions import build_sessions_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    calls = []
    build_sessions_parser(subparsers, cmd_sessions=lambda a, **kw: calls.append(a) or 0)

    ns = parser.parse_args(["sessions", "fork", "20260101_120000_parent", "--name", "alt line"])

    assert ns.sessions_action == "fork"
    assert ns.session_id == "20260101_120000_parent"
    assert ns.name == "alt line"
    assert ns.func(ns) == 0
    assert [a.session_id for a in calls] == ["20260101_120000_parent"]


def test_sessions_fork_allows_omitting_the_name():
    import argparse

    from hermes_cli.subcommands.sessions import build_sessions_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_sessions_parser(subparsers, cmd_sessions=lambda a, **kw: 0)

    ns = parser.parse_args(["sessions", "fork", "20260101_120000_parent"])

    assert ns.name is None
