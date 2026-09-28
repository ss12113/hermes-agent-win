"""Context engine plugin discovery: bundled ``plugins/context_engine/<name>/`` then user
``$HERMES_HOME/plugins/<name>/`` (bundled wins on collision) → ``ContextEngine``. Separate from the
general plugin system: ``context.engine`` in config.yaml names the active engine (default
``"compressor"``, the built-in ContextCompressor), so a user-installed engine needs no
``plugins.enabled`` entry to be selectable."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from plugins import plugin_loader as _loader

logger = logging.getLogger(__name__)

_CONTEXT_ENGINE_PLUGINS_DIR = Path(__file__).parent
# Synthetic parent package for user-installed engines (keeps them out of the bundled namespace).
_USER_NAMESPACE = "_hermes_user_context_engine"


def _is_context_engine_dir(path: Path) -> bool:
    """Cheap text heuristic: ``__init__.py`` mentions the context engine contract."""
    init_file = path / "__init__.py"
    try:
        source = init_file.read_text(errors="replace", encoding="utf-8-sig")[:8192]
    except OSError:
        return False
    return "register_context_engine" in source or "ContextEngine" in source


def _iter_engine_dirs() -> List[Tuple[str, Path]]:
    """``(name, path)`` for bundled then user engines; bundled wins on collisions."""
    dirs = [(child.name, child) for child in _loader.iter_plugin_dirs(_CONTEXT_ENGINE_PLUGINS_DIR)]
    seen = {name for name, _ in dirs}
    user_dir = _loader.user_plugins_dir()
    if user_dir:
        dirs.extend((child.name, child) for child in _loader.iter_plugin_dirs(user_dir)
                    if child.name not in seen and _is_context_engine_dir(child))
    return dirs


def discover_context_engines() -> List[Tuple[str, str, bool]]:
    """Return ``[(name, description, is_available), ...]`` for every bundled and user engine."""
    return [(name, _loader.read_plugin_description(child),
             _loader.probe_availability(lambda c=child: _load_engine_from_dir(c)))
            for name, child in _iter_engine_dirs()]


def find_engine_dir(name: str) -> Optional[Path]:
    """Resolve an engine name to its directory (bundled first, then user-installed)."""
    bundled = _CONTEXT_ENGINE_PLUGINS_DIR / name
    if bundled.is_dir():
        return bundled
    user_dir = _loader.user_plugins_dir()
    user = user_dir / name if user_dir else None
    return user if user and user.is_dir() and _is_context_engine_dir(user) else None


def load_context_engine(name: str) -> Optional["ContextEngine"]:  # noqa: F821
    """Load a ContextEngine instance by name; None if not found or it fails to load."""
    engine_dir = find_engine_dir(name)
    if engine_dir is None:
        logger.debug("Context engine '%s' not found in bundled or user plugins", name)
        return None
    return _loader.load_named(
        name, engine_dir, _load_engine_from_dir, kind="Context engine", noun="engine", logger=logger
    )


def _load_engine_from_dir(engine_dir: Path) -> Optional["ContextEngine"]:  # noqa: F821
    """Import an engine module and extract its ContextEngine (register(ctx) or subclass)."""
    from agent.context_engine import ContextEngine
    name = engine_dir.name
    is_bundled = engine_dir.parent == _CONTEXT_ENGINE_PLUGINS_DIR
    module_name = f"plugins.context_engine.{name}" if is_bundled else f"{_USER_NAMESPACE}.{name}"
    mod = _loader.load_plugin_module(
        module_name, engine_dir, parents=("plugins", "plugins.context_engine"), logger=logger,
        synthetic_namespace=None if is_bundled else _USER_NAMESPACE)
    return mod and _loader.instance_from_module(
        mod, collector=_EngineCollector(engine_name=name), collected_attr="engine",
        base_cls=ContextEngine, name=name, logger=logger)


class _EngineCollector(_loader.NoopPluginContext):
    """Captures register_context_engine; forwards register_command to the global plugin command
    registry so engine slash commands behave like plugin ones."""

    def __init__(self, engine_name: str = ""):
        self.engine = None
        self._engine_name = engine_name or "context_engine"

    def register_context_engine(self, engine):
        self.engine = engine

    def register_command(self, name: str, handler, description: str = "", args_hint: str = "") -> None:
        clean = (name or "").lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning("Context engine '%s' tried to register a command with an empty name.",
                           self._engine_name)
            return
        conflict = "Context engine '%s' tried to register command '/%s' which %s Skipping."
        try:
            from hermes_cli.commands import resolve_command
            if resolve_command(clean) is not None:
                logger.warning(conflict, self._engine_name, clean, "conflicts with a built-in command.")
                return
        except Exception:
            pass
        try:
            from hermes_cli.plugins import get_plugin_manager
            manager = get_plugin_manager()
            if clean in manager._plugin_commands:
                logger.warning(conflict, self._engine_name, clean, "is already registered by a plugin.")
                return
            manager._plugin_commands[clean] = {
                "handler": handler, "description": description or "Context engine command",
                "plugin": f"context-engine:{self._engine_name}", "args_hint": (args_hint or "").strip()}
            logger.debug("Context engine '%s' registered command: /%s", self._engine_name, clean)
        except Exception as exc:
            logger.debug("Context engine '%s' could not register /%s: %s", self._engine_name, clean, exc)


@dataclass(frozen=True)
class PreparedContextEngine:
    """One loaded engine whose process-global slash commands are not active yet."""

    engine: "ContextEngine"
    engine_name: str
    commands: tuple

    def activate_commands(self, *, selection_revision: Optional[int] = None) -> List[str]:
        """Publish the commands this candidate owns; returns their names."""
        return _replace_context_engine_commands(
            self.engine_name, self.commands, selection_revision=selection_revision
        )


def _replace_context_engine_commands(
    engine_name: str,
    commands: tuple,
    *,
    selection_revision: Optional[int] = None,
) -> List[str]:
    """Replace the slash commands owned by the active context engine.

    The registry keeps no cross-version ownership bookkeeping, so a retire pass
    drops exactly the dict objects a previous call published (tracked in
    ``_context_engine_commands`` when the manager exposes it) — never a plugin
    entry that legitimately replaced one; names a plugin already owns are
    skipped with a warning, matching the loader's own conflict policy.
    """
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    registered: List[str] = []
    published = getattr(manager, "_context_engine_commands", None)
    if isinstance(published, dict):
        for command_name in list(published):
            owned_entry = published.pop(command_name)
            if manager._plugin_commands.get(command_name) is owned_entry:
                del manager._plugin_commands[command_name]
    for command_name, entry in commands:
        if command_name in manager._plugin_commands:
            logger.warning(
                "Context engine '%s' tried to register command '/%s' which "
                "is already registered by a plugin. Skipping.", engine_name, command_name,
            )
            continue
        manager._plugin_commands[command_name] = entry
        registered.append(command_name)
        if isinstance(published, dict):
            published[command_name] = entry
    return registered


def _resolve_context_engine_dir(name: object) -> Optional[Path]:
    """Resolve one configured engine name inside the repository plugin root."""
    if (
        type(name) is not str
        or not name
        or "." in name
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        logger.warning("Invalid repository context engine name: %r", name)
        return None

    try:
        plugins_root = _CONTEXT_ENGINE_PLUGINS_DIR.resolve()
        engine_dir = (plugins_root / name).resolve()
    except (OSError, RuntimeError, ValueError):
        logger.warning("Could not resolve repository context engine name: %r", name)
        return None

    # A direct child that is a symlink outside the repository root is an escape
    # just like ``../name``. Resolve first, then require the exact parent.
    if engine_dir.parent != plugins_root:
        logger.warning(
            "Repository context engine %r resolves outside %s", name, _CONTEXT_ENGINE_PLUGINS_DIR,
        )
        return None
    return engine_dir


def prepare_context_engine(name: str) -> Optional[PreparedContextEngine]:
    """Load an engine candidate without publishing its slash commands."""
    engine_dir = _resolve_context_engine_dir(name)
    if engine_dir is None:
        return None
    if not engine_dir.is_dir():
        logger.debug("Context engine '%s' not found in %s", name, _CONTEXT_ENGINE_PLUGINS_DIR)
        return None
    engine = _load_engine_from_dir(engine_dir)
    if engine is None:
        logger.warning("Context engine '%s' loaded but no engine instance found", name)
        return None
    return PreparedContextEngine(engine=engine, engine_name=name, commands=())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import importlib.util  # noqa: F401,E402
import sys  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
