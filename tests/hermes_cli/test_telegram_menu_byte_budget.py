"""Telegram command menu must fit the Bot API's byte budget, not just the 100-command cap.

The Bot API rejects an over-long ``set_my_commands`` body with ``Bot_commands_too_much`` for
EVERY scope, so a menu that only respects ``max_commands`` publishes nothing at all. Measured on
this install: 100 commands / 6469 bytes → all three scopes failed; the same candidates under the
byte budget publish 40 commands / 3754 bytes.
"""

import asyncio

import pytest

from hermes_cli import commands_platforms as cp


@pytest.fixture
def menu(monkeypatch):
    """Call the real selector with a pinned (max_commands, byte budget) menu config."""
    def _menu(max_commands: int, budget: int):
        real = cp._telegram_command_menu_config
        monkeypatch.setattr(cp, "_telegram_command_menu_config",
                            lambda: {**real(), "max_commands": max_commands,
                                     "max_payload_bytes": budget})
        return asyncio.run(asyncio.to_thread(cp.telegram_menu_commands, max_commands=max_commands))

    return _menu


def test_menu_fits_the_byte_budget(menu):
    commands, _ = menu(100, 3800)
    assert commands, "a byte budget must not empty the menu"
    assert cp.telegram_menu_payload_bytes(commands) <= 3800


def test_smaller_budget_yields_a_smaller_menu(menu):
    wide, _ = menu(100, 100_000)
    narrow, _ = menu(100, 1200)
    assert len(narrow) < len(wide)
    assert cp.telegram_menu_payload_bytes(narrow) <= 1200


def test_byte_dropped_entries_are_reported_as_hidden(menu):
    _, hidden_wide = menu(100, 100_000)
    _, hidden_tight = menu(100, 1200)
    assert hidden_tight > hidden_wide


def test_trimming_keeps_earlier_candidates(menu):
    """Priority order survives trimming: the kept list is a subsequence of the full list."""
    wide, _ = menu(100, 100_000)
    narrow, _ = menu(100, 1200)
    kept = {name for name, _ in narrow}
    assert [name for name, _ in narrow] == [name for name, _ in wide if name in kept]


@pytest.mark.parametrize("entry,cost", [(("a", "b"), 2 + 32), (("help", "x" * 100), 104 + 32)])
def test_entry_cost_math(entry, cost):
    assert cp.telegram_menu_entry_bytes(entry) == cost
