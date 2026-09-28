"""Transactional lifecycle for PluginContext-backed provider registries.

Several optional capability families (image/video generation, web, browser,
TTS, STT, and dashboard auth) intentionally keep their own tiny runtime
registries.  This module gives those registries one shared plugin-lifecycle
contract without coupling the capability implementations together:

* exact owner/context provenance for every successful registration;
* per-``register(ctx)`` rollback;
* force-reload retirement with predecessor restoration;
* stale-context tombstones;
* atomic snapshots used by :class:`hermes_cli.plugins.PluginManager`.

Only registrations made through ``PluginContext`` are tracked.  The public
registry functions remain available for host code and tests; an out-of-band
replacement wins, and plugin retirement will not clobber that newer object.
"""

from __future__ import annotations

import sys
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, MutableMapping, Protocol


_MISSING = object()
_lock = threading.RLock()


class _LockLike(Protocol):
    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> object: ...
    def __exit__(self, exc_type, exc_value, traceback) -> object: ...


@dataclass(frozen=True)
class _RegistryBinding:
    name: str
    values: MutableMapping[str, object]
    lock: _LockLike


@dataclass(frozen=True)
class _ProviderRecord:
    registry_name: str
    key: str
    inserted: object
    owner: object
    predecessor_token: object | None
    predecessor_value: object


_bindings: Dict[str, _RegistryBinding] = {}
_records: Dict[object, _ProviderRecord] = {}
_owner_records: Dict[object, List[object]] = {}
_context_records: Dict[object, List[object]] = {}
_current: Dict[tuple[str, str], object] = {}
_retired_records: set[object] = set()
_retired_owners: weakref.WeakSet = weakref.WeakSet()
_retired_contexts: weakref.WeakSet = weakref.WeakSet()


def _caller_module() -> str:
    """Return the first caller outside this lifecycle module/contextlib."""
    try:
        frame = sys._getframe(2)
        while frame is not None:
            module = frame.f_globals.get("__name__", "") or ""
            if module not in {__name__, "contextlib"}:
                return module
            frame = frame.f_back
    except Exception:
        pass
    return ""


def _require_host(message: str) -> None:
    if _caller_module() != "hermes_cli.plugins":
        raise PermissionError(message)


def _bind_registry_locked(
    registry_name: str,
    values: MutableMapping[str, object],
    registry_lock: _LockLike,
) -> _RegistryBinding:
    existing = _bindings.get(registry_name)
    if existing is not None:
        if existing.values is not values or existing.lock is not registry_lock:
            raise RuntimeError(
                f"Provider registry binding changed for {registry_name!r}"
            )
        return existing
    binding = _RegistryBinding(registry_name, values, registry_lock)
    _bindings[registry_name] = binding
    return binding


def _sync_current_locked(binding: _RegistryBinding, key: str) -> object | None:
    current_key = (binding.name, key)
    token = _current.get(current_key)
    if token is None:
        return None
    record = _records.get(token)
    if record is None or binding.values.get(key, _MISSING) is not record.inserted:
        # Host code replaced or removed the plugin value out of band. Its value
        # is authoritative; future plugin layers treat it as a direct predecessor.
        _current.pop(current_key, None)
        return None
    return token


def register_plugin_provider(
    *,
    owner: object,
    context_owner: object,
    registry_name: str,
    values: MutableMapping[str, object],
    registry_lock: _LockLike,
    key: str,
    provider: object,
    replace_existing: bool = True,
    duplicate_label: str | None = None,
) -> bool:
    """Register one provider and retain exact rollback provenance.

    PluginContext validates provider type and evaluates plugin-controlled
    attributes before entering this function. The mutation below uses only the
    host-owned key and plain registry mapping, so no plugin callback runs while
    lifecycle and registry locks are held.
    """
    _require_host(
        "Plugin providers may only be registered through PluginContext"
    )
    if not isinstance(key, str):
        raise ValueError("Plugin provider key must be a non-empty string")
    # ``str`` subclasses may override hashing/equality. Canonicalize before
    # entering lifecycle/registry locks so dict operations use a host-owned key.
    key = str(key)
    if type(key) is not str:
        key = "{}".format(key)
    if not key.strip():
        raise ValueError("Plugin provider key must be a non-empty string")

    with _lock:
        binding = _bind_registry_locked(registry_name, values, registry_lock)
        with binding.lock:  # registry locks are RLocks; register_fn re-enters it
            if owner in _retired_owners:
                return False
            if context_owner in _retired_contexts:
                return False

            predecessor_token = _sync_current_locked(binding, key)
            predecessor_value = binding.values.get(key, _MISSING)
            try:
                if not replace_existing and predecessor_value is not _MISSING:
                    label = duplicate_label or registry_name
                    raise ValueError(
                        f"{label} provider already registered: {key!r}"
                    )
                binding.values[key] = provider
            except BaseException:
                if binding.values.get(key, _MISSING) is provider:
                    if predecessor_value is _MISSING:
                        binding.values.pop(key, None)
                    else:
                        binding.values[key] = predecessor_value
                raise

            if binding.values.get(key, _MISSING) is not provider:
                return False

            token = object()
            _records[token] = _ProviderRecord(
                registry_name=registry_name,
                key=key,
                inserted=provider,
                owner=owner,
                predecessor_token=predecessor_token,
                predecessor_value=predecessor_value,
            )
            _owner_records.setdefault(owner, []).append(token)
            _context_records.setdefault(context_owner, []).append(token)
            _current[(registry_name, key)] = token
            return True


def _resolve_predecessor_locked(
    record: _ProviderRecord,
) -> tuple[object, object | None]:
    token = record.predecessor_token
    seen: set[object] = set()
    while token is not None and token not in seen:
        seen.add(token)
        predecessor = _records.get(token)
        if predecessor is None:
            break
        if token not in _retired_records:
            return predecessor.inserted, token
        if predecessor.predecessor_token is None:
            return predecessor.predecessor_value, None
        token = predecessor.predecessor_token
    return record.predecessor_value, None


def _retire_tokens_locked(owner: object, tokens: List[object]) -> int:
    changes = 0
    # A context may re-register the same name repeatedly. Unwind newest first.
    for token in reversed(tokens):
        record = _records.get(token)
        if record is None:
            continue
        if record.owner is not owner:
            raise RuntimeError(
                "Plugin provider record belongs to a different discovery sweep"
            )
        _retired_records.add(token)
        binding = _bindings.get(record.registry_name)
        if binding is None:
            continue
        current_key = (record.registry_name, record.key)
        if _current.get(current_key) is token:
            if binding.values.get(record.key, _MISSING) is not record.inserted:
                # A newer direct host registration superseded this plugin layer.
                # It is authoritative and must survive plugin retirement.
                _current.pop(current_key, None)
                continue
            predecessor, predecessor_token = _resolve_predecessor_locked(record)
            if predecessor is _MISSING:
                binding.values.pop(record.key, None)
            else:
                binding.values[record.key] = predecessor
            if predecessor_token is None:
                _current.pop(current_key, None)
            else:
                _current[current_key] = predecessor_token
            changes += 1
    return changes


def _reachable_record_tokens_locked() -> set[object]:
    roots = set(_current.values())
    for tokens in _owner_records.values():
        roots.update(tokens)
    for tokens in _context_records.values():
        roots.update(tokens)

    reachable: set[object] = set()
    pending = list(roots)
    while pending:
        token = pending.pop()
        if token in reachable:
            continue
        record = _records.get(token)
        if record is None:
            continue
        reachable.add(token)
        if record.predecessor_token is not None:
            pending.append(record.predecessor_token)
    return reachable


def _prune_records_locked(deferred_releases: List[object]) -> None:
    reachable = _reachable_record_tokens_locked()
    for token in list(_records):
        if token not in reachable:
            record = _records.pop(token, None)
            if record is not None:
                deferred_releases.append(record)
            _retired_records.discard(token)


def _snapshot_state_locked() -> dict:
    return {
        "values": {
            name: dict(binding.values)
            for name, binding in _bindings.items()
        },
        "records": dict(_records),
        "owner_records": {
            owner: list(tokens) for owner, tokens in _owner_records.items()
        },
        "context_records": {
            owner: list(tokens) for owner, tokens in _context_records.items()
        },
        "current": dict(_current),
        "retired_records": set(_retired_records),
        "retired_owners": list(_retired_owners),
        "retired_contexts": list(_retired_contexts),
    }


def _resolve_snapshot_predecessor_locked(
    record: _ProviderRecord,
    snapshot_tokens: set[object],
) -> object:
    """Return the value below a post-snapshot provider layer.

    A registry can be bound lazily by the replacement generation itself.  In
    that case it has no entry in ``snapshot["values"]`` and the ordinary
    value-map replay below cannot remove the replacement.  Walk the exact
    predecessor chain until the first layer that existed at the savepoint,
    or until the concrete host value captured by the oldest new record.
    """
    current = record
    seen: set[object] = set()
    while True:
        token = current.predecessor_token
        if token is None:
            return current.predecessor_value
        if token in snapshot_tokens:
            predecessor = _records.get(token)
            if predecessor is not None:
                return predecessor.inserted
            return current.predecessor_value
        if token in seen:
            # A malformed/cyclic chain must not make rollback loop forever.
            return current.predecessor_value
        seen.add(token)
        predecessor = _records.get(token)
        if predecessor is None:
            return current.predecessor_value
        current = predecessor


def _restore_lazily_bound_registries_locked(snapshot: dict) -> None:
    """Remove exact replacement identities from registries born after a savepoint.

    ``_bindings`` is intentionally retained for the lifetime of a process,
    but a binding may first appear while a failed force-reload generation is
    executing.  Such a binding is absent from the snapshot, so replaying the
    snapshot's value dictionaries leaves its failed provider visible.  Only
    mutate a slot when the mapping still contains the exact provider object
    recorded by the lifecycle layer; a direct host replacement is newer and
    remains authoritative.
    """
    snapshot_values = snapshot["values"]
    snapshot_tokens = set(snapshot["records"])
    for registry_name, binding in _bindings.items():
        if registry_name in snapshot_values:
            continue
        for (current_registry, key), token in list(_current.items()):
            if current_registry != registry_name or token in snapshot_tokens:
                continue
            record = _records.get(token)
            if record is None or record.registry_name != registry_name:
                continue
            if binding.values.get(key, _MISSING) is not record.inserted:
                # A host (or another newer owner) replaced this exact layer.
                # Do not clobber that authoritative object during rollback.
                continue
            predecessor = _resolve_snapshot_predecessor_locked(
                record, snapshot_tokens
            )
            if predecessor is _MISSING:
                binding.values.pop(key, None)
            else:
                binding.values[key] = predecessor


def _restore_state_locked(
    snapshot: dict,
    deferred_releases: List[object],
) -> None:
    # Reconcile registries that were first bound by the failed generation
    # before replacing provenance.  Without this step, a provider can remain
    # in the live mapping after its record is discarded below, becoming an
    # unowned "ghost" provider that no later retirement can remove.
    _restore_lazily_bound_registries_locked(snapshot)
    for name, values in snapshot["values"].items():
        binding = _bindings.get(name)
        if binding is None:
            raise RuntimeError(
                f"Provider registry {name!r} disappeared during lifecycle rollback"
            )
        binding.values.clear()
        binding.values.update(values)
    deferred_releases.extend(
        record
        for token, record in _records.items()
        if snapshot["records"].get(token) is not record
    )
    _records.clear()
    _records.update(snapshot["records"])
    _owner_records.clear()
    _owner_records.update(
        {owner: list(tokens) for owner, tokens in snapshot["owner_records"].items()}
    )
    _context_records.clear()
    _context_records.update(
        {
            owner: list(tokens)
            for owner, tokens in snapshot["context_records"].items()
        }
    )
    _current.clear()
    _current.update(snapshot["current"])
    _retired_records.clear()
    _retired_records.update(snapshot["retired_records"])
    _retired_owners.clear()
    _retired_owners.update(snapshot["retired_owners"])
    _retired_contexts.clear()
    _retired_contexts.update(snapshot["retired_contexts"])


def _snapshot_current_token(
    snapshot: dict,
    registry_name: str,
    key: str,
) -> object | None:
    """Return a snapshot token only when its concrete value still matches."""
    token = snapshot["current"].get((registry_name, key))
    record = snapshot["records"].get(token) if token is not None else None
    value = snapshot["values"].get(registry_name, {}).get(key, _MISSING)
    if record is None or value is not record.inserted:
        return None
    return token


def _resolve_snapshot_layer(
    snapshot: dict,
    registry_name: str,
    key: str,
    *,
    excluded_tokens: set[object],
    excluded_owners: set[object],
) -> tuple[object, object | None]:
    """Resolve the first live snapshot layer after explicit exclusions."""
    value = snapshot["values"].get(registry_name, {}).get(key, _MISSING)
    token = _snapshot_current_token(snapshot, registry_name, key)
    if token is None:
        return value, None

    retired = set(snapshot["retired_records"])
    seen: set[object] = set()
    while token is not None and token not in seen:
        seen.add(token)
        record = snapshot["records"].get(token)
        if record is None:
            return value, None
        if (
            token not in retired
            and token not in excluded_tokens
            and record.owner not in excluded_owners
        ):
            return record.inserted, token
        token = record.predecessor_token
        if token is None:
            return record.predecessor_value, None
    return value, None


def _restore_failed_force_reload_locked(
    snapshot: dict,
    *,
    restored_owner: object,
    failed_owner: object | None,
    deferred_releases: List[object],
) -> None:
    """Restore one manager generation without reverting newer foreign state.

    A force-reload savepoint is released while replacement plugins execute.
    Another manager (or host code using a public capability registry) may
    therefore install a newer value before the replacement fails.  Exact
    snapshot replay would erase that value and its provenance.  Reconcile the
    three states instead: restore the committed owner, discard only the failed
    owner, and splice still-live post-savepoint layers over the baseline.
    """
    live_records = dict(_records)
    live_owner_records = {
        owner: list(tokens) for owner, tokens in _owner_records.items()
    }
    live_context_records = {
        context: list(tokens) for context, tokens in _context_records.items()
    }
    live_current = dict(_current)
    live_retired_records = set(_retired_records)
    live_retired_owners = set(_retired_owners)
    live_retired_contexts = set(_retired_contexts)
    live_values = {
        name: dict(binding.values) for name, binding in _bindings.items()
    }

    snapshot_tokens = set(snapshot["records"])
    snapshot_retired_owners = set(snapshot["retired_owners"])
    snapshot_retired_contexts = set(snapshot["retired_contexts"])
    newly_retired_owners = live_retired_owners - snapshot_retired_owners
    newly_retired_owners.discard(restored_owner)
    newly_retired_contexts = live_retired_contexts - snapshot_retired_contexts

    excluded_snapshot_tokens: set[object] = set()
    if failed_owner is not None:
        excluded_snapshot_tokens.update(
            snapshot["owner_records"].get(failed_owner, ())
        )
    for owner in newly_retired_owners:
        excluded_snapshot_tokens.update(snapshot["owner_records"].get(owner, ()))
    for context in newly_retired_contexts:
        excluded_snapshot_tokens.update(
            snapshot["context_records"].get(context, ())
        )

    merged_records = {
        token: record
        for token, record in snapshot["records"].items()
        if token not in excluded_snapshot_tokens
    }
    live_roots = set(live_current.values())
    for tokens in live_owner_records.values():
        live_roots.update(tokens)
    for tokens in live_context_records.values():
        live_roots.update(tokens)
    for token, record in live_records.items():
        if token in snapshot_tokens or token not in live_roots:
            continue
        if failed_owner is not None and record.owner is failed_owner:
            continue
        if record.owner in newly_retired_owners or token in live_retired_records:
            continue
        merged_records[token] = record

    all_keys: set[tuple[str, str]] = set(snapshot["current"])
    all_keys.update(live_current)
    for registry_name, values in snapshot["values"].items():
        all_keys.update((registry_name, key) for key in values)
    for registry_name, values in live_values.items():
        all_keys.update((registry_name, key) for key in values)
    all_keys.update(
        (record.registry_name, record.key)
        for record in snapshot["records"].values()
    )
    all_keys.update(
        (record.registry_name, record.key) for record in live_records.values()
    )

    final_values: dict[tuple[str, str], object] = {}
    final_current: dict[tuple[str, str], object] = {}
    for registry_name, key in all_keys:
        snapshot_value, snapshot_token = _resolve_snapshot_layer(
            snapshot,
            registry_name,
            key,
            excluded_tokens=excluded_snapshot_tokens,
            excluded_owners=set(),
        )
        exposed_value, _ = _resolve_snapshot_layer(
            snapshot,
            registry_name,
            key,
            excluded_tokens=excluded_snapshot_tokens,
            excluded_owners={restored_owner},
        )
        live_value = live_values.get(registry_name, {}).get(key, _MISSING)
        token = live_current.get((registry_name, key))
        live_record = live_records.get(token) if token is not None else None

        if live_record is None or live_value is not live_record.inserted:
            # No tracked layer owns the live slot.  The ordinary teardown value
            # is replaced by the committed savepoint; any other identity is a
            # newer direct host write and remains authoritative.
            if live_value is exposed_value:
                final_values[(registry_name, key)] = snapshot_value
                if snapshot_token is not None:
                    final_current[(registry_name, key)] = snapshot_token
            else:
                final_values[(registry_name, key)] = live_value
            continue

        chain: list[tuple[object, _ProviderRecord]] = []
        terminal_token: object | None = None
        terminal_value: object = _MISSING
        seen: set[object] = set()
        while token is not None and token not in seen:
            seen.add(token)
            record = live_records.get(token)
            if record is None:
                break
            if (
                (failed_owner is not None and record.owner is failed_owner)
                or record.owner in newly_retired_owners
                or token in live_retired_records
                or token in excluded_snapshot_tokens
            ):
                pass
            elif token in snapshot_tokens:
                terminal_token = token
                terminal_value = record.inserted
                break
            else:
                chain.append((token, record))

            if record.predecessor_token is None:
                terminal_value = record.predecessor_value
                break
            token = record.predecessor_token

        if terminal_token is not None:
            anchor_token = terminal_token
            anchor_value = terminal_value
        elif terminal_value is exposed_value:
            anchor_token = snapshot_token
            anchor_value = snapshot_value
        else:
            # A direct host replacement happened below the surviving plugin
            # layers.  Preserve it as their concrete predecessor.
            anchor_token = None
            anchor_value = terminal_value

        predecessor_token = anchor_token
        predecessor_value = anchor_value
        for current_token, record in reversed(chain):
            merged_records[current_token] = _ProviderRecord(
                registry_name=record.registry_name,
                key=record.key,
                inserted=record.inserted,
                owner=record.owner,
                predecessor_token=predecessor_token,
                predecessor_value=predecessor_value,
            )
            predecessor_token = current_token
            predecessor_value = record.inserted

        if chain:
            final_values[(registry_name, key)] = chain[0][1].inserted
            final_current[(registry_name, key)] = chain[0][0]
        else:
            final_values[(registry_name, key)] = anchor_value
            if anchor_token is not None:
                final_current[(registry_name, key)] = anchor_token

    merged_owner_records: Dict[object, List[object]] = {}
    for owner, tokens in snapshot["owner_records"].items():
        if (
            (failed_owner is not None and owner is failed_owner)
            or owner in newly_retired_owners
        ):
            continue
        kept = [token for token in tokens if token in merged_records]
        if kept:
            merged_owner_records[owner] = kept
    for owner, tokens in live_owner_records.items():
        if (
            (failed_owner is not None and owner is failed_owner)
            or owner in newly_retired_owners
        ):
            continue
        bucket = merged_owner_records.setdefault(owner, [])
        bucket.extend(
            token
            for token in tokens
            if token not in snapshot_tokens
            and token in merged_records
            and token not in bucket
        )

    merged_context_records: Dict[object, List[object]] = {}
    for context, tokens in snapshot["context_records"].items():
        if context in newly_retired_contexts:
            continue
        kept = [token for token in tokens if token in merged_records]
        if kept:
            merged_context_records[context] = kept
    for context, tokens in live_context_records.items():
        if context in newly_retired_contexts:
            continue
        bucket = merged_context_records.setdefault(context, [])
        bucket.extend(
            token
            for token in tokens
            if token not in snapshot_tokens
            and token in merged_records
            and token not in bucket
        )

    for (registry_name, key), value in final_values.items():
        binding = _bindings.get(registry_name)
        if binding is None:
            raise RuntimeError(
                f"Provider registry {registry_name!r} disappeared during lifecycle rollback"
            )
        if value is _MISSING:
            binding.values.pop(key, None)
        else:
            binding.values[key] = value

    deferred_releases.extend(
        record
        for token, record in _records.items()
        if merged_records.get(token) is not record
    )
    _records.clear()
    _records.update(merged_records)
    _owner_records.clear()
    _owner_records.update(merged_owner_records)
    _context_records.clear()
    _context_records.update(merged_context_records)
    _current.clear()
    _current.update(
        {
            current_key: token
            for current_key, token in final_current.items()
            if token in merged_records
        }
    )
    _retired_records.clear()
    _retired_records.update(
        token
        for token in (set(snapshot["retired_records"]) | live_retired_records)
        if token in merged_records and token not in _current.values()
    )
    _retired_owners.clear()
    _retired_owners.update(snapshot["retired_owners"])
    _retired_owners.update(live_retired_owners)
    _retired_owners.discard(restored_owner)
    if failed_owner is not None:
        _retired_owners.add(failed_owner)
    _retired_contexts.clear()
    _retired_contexts.update(snapshot["retired_contexts"])
    _retired_contexts.update(live_retired_contexts)
    _prune_records_locked(deferred_releases)


def retire_plugin_context(
    owner: object,
    context_owner: object,
    *,
    deferred_releases: List[object],
) -> int:
    """Roll back all provider registrations from one failed ``register(ctx)``."""
    _require_host("Plugin provider contexts may only be retired by the host")
    with _lock:
        tokens = list(_context_records.get(context_owner, ()))
        # Snapshots include every bound registry, so hold every matching lock;
        # otherwise a fault-path restore could replay stale unrelated values.
        bindings = sorted(_bindings)
        acquired: List[_LockLike] = []
        snapshot: dict | None = None
        try:
            for name in bindings:
                lock = _bindings[name].lock
                lock.acquire()
                acquired.append(lock)
            snapshot = _snapshot_state_locked()
            _retired_contexts.add(context_owner)
            changes = _retire_tokens_locked(owner, tokens)
            _context_records.pop(context_owner, None)
            if tokens:
                retired = set(tokens)
                remaining = [
                    token
                    for token in _owner_records.get(owner, ())
                    if token not in retired
                ]
                if remaining:
                    _owner_records[owner] = remaining
                else:
                    _owner_records.pop(owner, None)
            _prune_records_locked(deferred_releases)
            return changes
        except BaseException:
            if snapshot is not None:
                _restore_state_locked(snapshot, deferred_releases)
            raise
        finally:
            for lock in reversed(acquired):
                lock.release()


def retire_plugin_owner(
    owner: object,
    *,
    deferred_releases: List[object],
) -> int:
    """Retire all capability-provider registrations from one discovery sweep."""
    _require_host("Plugin provider sweeps may only be retired by the host")
    with _lock:
        tokens = list(_owner_records.get(owner, ()))
        # Snapshots include every bound registry, so hold every matching lock;
        # otherwise a fault-path restore could replay stale unrelated values.
        bindings = sorted(_bindings)
        acquired: List[_LockLike] = []
        snapshot: dict | None = None
        try:
            for name in bindings:
                lock = _bindings[name].lock
                lock.acquire()
                acquired.append(lock)
            snapshot = _snapshot_state_locked()
            _retired_owners.add(owner)
            changes = _retire_tokens_locked(owner, tokens)
            _owner_records.pop(owner, None)
            if tokens:
                retired = set(tokens)
                for context_owner, context_tokens in list(_context_records.items()):
                    remaining = [
                        token for token in context_tokens if token not in retired
                    ]
                    if remaining:
                        _context_records[context_owner] = remaining
                    else:
                        _context_records.pop(context_owner, None)
            _prune_records_locked(deferred_releases)
            return changes
        except BaseException:
            if snapshot is not None:
                _restore_state_locked(snapshot, deferred_releases)
            raise
        finally:
            for lock in reversed(acquired):
                lock.release()


@contextmanager
def lifecycle_transaction() -> Iterator[None]:
    """Hold every bound provider registry stable for a host transaction."""
    _require_host("Plugin provider lifecycle transactions are host-only")
    with _lock:
        acquired: List[_LockLike] = []
        try:
            for name in sorted(_bindings):
                lock = _bindings[name].lock
                lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()


def snapshot_plugin_lifecycle_state() -> dict:
    _require_host("Plugin provider lifecycle snapshots are host-only")
    with _lock:
        return _snapshot_state_locked()


def restore_plugin_lifecycle_state(
    snapshot: dict,
    *,
    restored_owner: object | None = None,
    failed_owner: object | None = None,
    deferred_releases: List[object],
) -> None:
    _require_host("Plugin provider lifecycle restoration is host-only")
    with _lock:
        if restored_owner is None:
            _restore_state_locked(snapshot, deferred_releases)
        else:
            _restore_failed_force_reload_locked(
                snapshot,
                restored_owner=restored_owner,
                failed_owner=failed_owner,
                deferred_releases=deferred_releases,
            )


def _reset_for_tests(*, clear_values: bool = False) -> None:
    """Reset provenance; optionally clear every registry bound so far."""
    deferred_releases: List[object] = []
    with _lock:
        acquired: List[_LockLike] = []
        try:
            for name in sorted(_bindings):
                lock = _bindings[name].lock
                lock.acquire()
                acquired.append(lock)
            if clear_values:
                for binding in _bindings.values():
                    deferred_releases.extend(binding.values.values())
                    binding.values.clear()
            deferred_releases.extend(_records.values())
            _records.clear()
            _owner_records.clear()
            _context_records.clear()
            _current.clear()
            _retired_records.clear()
            _retired_owners.clear()
            _retired_contexts.clear()
        finally:
            for lock in reversed(acquired):
                lock.release()
    deferred_releases.clear()
