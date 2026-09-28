"""Fan-out prompt-cache parity: same-route delegate children must share ONE cache bucket.

A batch of siblings renders byte-identical cached prefixes (same toolsets/role/workspace, memory
and context files skipped, ``Conversation started:`` from the lineage ROOT stamp), so N separate
buckets mean N cold full-context requests for the same bytes.
"""

from agent.prompt_cache_scope import resolve_prompt_cache_scope
from tools.delegate_tool import _apply_child_cache_parity


class _Agent:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _parent(**kw):
    # A real parent has resolved its scope on its first API request; the memo is what makes the
    # inheritance a pure attribute read (no lineage DB walk on the spawn path).
    return _Agent(session_id="parent-sid", model="m-1", provider="p-1",
                  _prompt_cache_scope_memo=(("parent-sid", False), "parent-sid"), **kw)


def test_same_route_child_inherits_parent_scope():
    parent, child = _parent(), _Agent(session_id="child-sid", model="m-1", provider="p-1")
    _apply_child_cache_parity(child, parent, {"model": "m-1", "provider": "p-1"})
    assert getattr(child, "_inherited_cache_scope", None) == resolve_prompt_cache_scope(parent)
    # the consumer resolves the inherited value, not the child's own id
    assert resolve_prompt_cache_scope(child) == "parent-sid"


def test_child_model_from_runtime_when_unset():
    parent, child = _parent(), _Agent(session_id="child-sid")
    _apply_child_cache_parity(child, parent, {"model": "m-1", "provider": "p-1"})
    assert getattr(child, "_inherited_cache_scope", None) == "parent-sid"


def test_routed_child_keeps_its_own_bucket():
    parent, child = _parent(), _Agent(session_id="child-sid", model="m-2", provider="p-1")
    _apply_child_cache_parity(child, parent, {"model": "m-2", "provider": "p-1"})
    assert not getattr(child, "_inherited_cache_scope", None)


def test_different_provider_does_not_inherit():
    parent, child = _parent(), _Agent(session_id="child-sid", model="m-1", provider="p-2")
    _apply_child_cache_parity(child, parent, {"model": "m-1", "provider": "p-2"})
    assert not getattr(child, "_inherited_cache_scope", None)


def test_unresolved_parent_is_skipped():
    """No memo yet = nothing to inherit. Resolving here would put a lineage DB walk on the
    spawn path, where a sub-second liveness cap can feel it under parallel load."""
    parent = _Agent(session_id="parent-sid", model="m-1", provider="p-1")
    child = _Agent(session_id="child-sid", model="m-1", provider="p-1")
    _apply_child_cache_parity(child, parent, {"model": "m-1", "provider": "p-1"})
    assert not getattr(child, "_inherited_cache_scope", None)


def test_fork_of_a_fork_inherits_without_a_memo():
    """A parent that itself inherited carries the scope directly; no memo needed."""
    parent = _Agent(session_id="parent-sid", model="m-1", provider="p-1",
                    _inherited_cache_scope="root-sid")
    child = _Agent(session_id="child-sid", model="m-1", provider="p-1")
    _apply_child_cache_parity(child, parent, {"model": "m-1", "provider": "p-1"})
    assert getattr(child, "_inherited_cache_scope", None) == "root-sid"


def test_missing_models_do_not_inherit():
    parent, child = _Agent(session_id="parent-sid"), _Agent(session_id="child-sid")
    _apply_child_cache_parity(child, parent, {})
    assert not getattr(child, "_inherited_cache_scope", None)


def test_resolution_failure_is_swallowed(monkeypatch):
    parent = _parent()
    child = _Agent(session_id="child-sid", model="m-1", provider="p-1")

    import agent.prompt_cache_scope as scope_mod

    def boom(_agent):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(scope_mod, "resolve_prompt_cache_scope_safe", boom)
    _apply_child_cache_parity(child, parent, {"model": "m-1", "provider": "p-1"})
    assert not getattr(child, "_inherited_cache_scope", None)
