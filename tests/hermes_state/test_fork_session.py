"""Tests for ``SessionDB.fork_session`` — user-facing session fork.

A fork must be a faithful row-level copy of the parent's LIVE transcript, must stamp the
child as an explicit fork child (``model_config._branched_from`` + ``parent_session_id`` —
the marker ``agent/prompt_cache_scope.py`` reads to keep the child's prompt-cache scope out
of the parent's), and must leave the parent byte-identical: the store-level fork never ends
or rewrites the original.
"""

import json

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed(db, session_id="20260101_120000_parent", **kwargs):
    """A parent with a user / assistant(tool call) / tool / assistant / user round trip."""
    db.create_session(
        session_id, source="cli", model="test-model", cwd="/work/proj",
        system_prompt="You are a fixture.",
        model_config={"reasoning_config": {"effort": "high"}}, **kwargs)
    db.append_message(session_id, "user", content="first question", api_content="first question")
    db.append_message(
        session_id, "assistant", content="let me look", reasoning="because",
        tool_calls=[{"id": "call_1", "type": "function",
                     "function": {"name": "read_file", "arguments": "{}"}}])
    db.append_message(session_id, "tool", content="file contents", tool_call_id="call_1")
    db.append_message(session_id, "assistant", content="here it is", api_content="here it is api")
    db.append_message(session_id, "user", content="thanks")
    return session_id


def test_fork_copies_transcript_metadata_and_marks_the_child(db):
    parent = _seed(db)
    forked = db.fork_session(parent, title="forked run")

    child = forked["session_id"]
    assert child and child != parent
    assert forked["parent_session_id"] == parent
    assert forked["title"] == "forked run"
    assert forked["message_count"] == 5
    assert forked["tool_call_count"] == 1

    parent_row, child_row = db.get_session(parent), db.get_session(child)
    assert child_row["parent_session_id"] == parent
    assert child_row["source"] == parent_row["source"]
    assert child_row["model"] == parent_row["model"]
    assert child_row["cwd"] == parent_row["cwd"]
    assert child_row["system_prompt"] == parent_row["system_prompt"] == "You are a fixture."
    assert child_row["ended_at"] is None and child_row["end_reason"] is None
    assert child_row["message_count"] == 5

    child_config = json.loads(child_row["model_config"])
    assert child_config["_branched_from"] == parent
    assert child_config["reasoning_config"] == {"effort": "high"}  # per-session state copied

    # Transcript rows land byte-faithfully, in insertion order, sidecars included.
    parent_msgs, child_msgs = db.get_messages(parent), db.get_messages(child)
    assert [m["role"] for m in child_msgs] == [m["role"] for m in parent_msgs]
    assert [m["content"] for m in child_msgs] == [m["content"] for m in parent_msgs]
    assert [m["timestamp"] for m in child_msgs] == [m["timestamp"] for m in parent_msgs]
    assert [m.get("api_content") for m in child_msgs] == [m.get("api_content") for m in parent_msgs]
    assert child_msgs[1]["reasoning"] == "because"
    assert child_msgs[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert child_msgs[2]["tool_call_id"] == "call_1"
    assert db.get_messages_as_conversation(child) == db.get_messages_as_conversation(parent)


def test_fork_leaves_the_parent_untouched(db):
    parent = _seed(db)
    before_row = db.get_session(parent)
    before_msgs = db.get_messages(parent, include_inactive=True)

    db.fork_session(parent)

    assert db.get_session(parent) == before_row
    assert db.get_messages(parent, include_inactive=True) == before_msgs
    assert db.get_session(parent)["ended_at"] is None
    assert db.get_session(parent)["end_reason"] is None
    assert db.is_explicit_fork_child(parent) is False


def test_fork_child_gets_its_own_cache_scope(db):
    parent = _seed(db)
    child = db.fork_session(parent)["session_id"]

    assert db.is_explicit_fork_child(child) is True
    assert db.declared_scope_identity(child) == (True, "cli")
    assert db.declared_scope_identity(parent) == (False, "cli")
    # Both are their own lineage/cache root: the child's scope never resolves through the parent.
    assert db.get_compression_lineage(child) == [child]
    assert db.get_compression_lineage(parent) == [parent]


def test_fork_does_not_inherit_the_parents_own_lineage_markers(db):
    db.create_session("20260101_100000_grandparent", source="cli")
    db.create_session(
        "20260101_110000_parent", source="cli", parent_session_id="20260101_100000_grandparent",
        model_config={"_branched_from": "20260101_100000_grandparent", "_reset_from": "20260101_100000_grandparent",
                      "_delegate_from": "20260101_100000_grandparent", "provider": "openrouter"})

    child = db.fork_session("20260101_110000_parent")["session_id"]
    config = json.loads(db.get_session(child)["model_config"])

    assert config["_branched_from"] == "20260101_110000_parent"
    assert "_reset_from" not in config and "_delegate_from" not in config
    assert config["provider"] == "openrouter"  # route state still copied
    assert db.get_compression_lineage(child) == [child]


def test_fork_default_title_numbers_into_the_lineage(db):
    parent = _seed(db)
    db.set_session_title(parent, "my chat")

    first = db.fork_session(parent)
    second = db.fork_session(parent)

    assert first["title"] == "my chat #2"
    assert second["title"] == "my chat #3"
    assert db.get_session_title(parent) == "my chat"  # the parent keeps its own name
    assert db.get_session_title(first["session_id"]) == "my chat #2"


def test_fork_copies_only_the_active_transcript(db):
    parent = _seed(db)
    db.replace_messages(parent, db.get_messages(parent)[:3], archive_dropped=True)
    assert len(db.get_messages(parent)) == 3

    child = db.fork_session(parent)["session_id"]

    assert [m["content"] for m in db.get_messages(child)] == \
        [m["content"] for m in db.get_messages(parent)]
    assert db.get_session(child)["message_count"] == 3


def test_fork_copies_an_empty_session(db):
    db.create_session("20260101_140000_empty", source="cli")

    forked = db.fork_session("20260101_140000_empty")

    assert forked["message_count"] == 0
    assert db.get_session(forked["session_id"])["parent_session_id"] == "20260101_140000_empty"
    assert db.get_messages(forked["session_id"]) == []


def test_fork_missing_parent_raises(db):
    with pytest.raises(ValueError, match="No session"):
        db.fork_session("20260101_000000_missing")


def test_fork_refuses_a_taken_or_identical_child_id(db):
    parent = _seed(db)
    assert db.fork_session(parent, new_session_id="forked_child_1")["session_id"] == "forked_child_1"

    with pytest.raises(ValueError, match="already exists"):
        db.fork_session(parent, new_session_id="forked_child_1")
    with pytest.raises(ValueError, match="new session id"):
        db.fork_session(parent, new_session_id=parent)

    # The refused forks left the first child intact.
    assert db.get_session("forked_child_1")["message_count"] == 5
    assert len(db.get_messages("forked_child_1")) == 5


def test_fork_conflicting_title_rolls_the_child_back(db):
    parent = _seed(db)
    _seed(db, "20260101_130000_other")
    db.set_session_title("20260101_130000_other", "taken name")
    count_before = db.session_count()

    with pytest.raises(ValueError, match="already in use"):
        db.fork_session(parent, title="taken name")

    assert db.session_count() == count_before  # no partial child left behind
