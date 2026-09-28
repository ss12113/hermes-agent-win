"""Invariants for ``approvals.ask`` — user-defined globs that FORCE matching terminal
commands through the human prompt. Precedence: hardline/deny floors > ask > permanent
allowlist / session approvals / smart reviewer. Explicit yolo / mode=off still bypasses
the approval flow; unattended contexts keep their configured modes. Mirrors the
deny-rule test layout."""

import pytest

from tools import approval as mod
from tools import approval_context
import tools.approval_floors as approval_floors

DANGEROUS = "rm -rf /var/tmp/ask-rules-probe"


@pytest.fixture
def ask_config(monkeypatch):
    """Install an ask list into the approvals config and return a setter."""
    state = {"config": {"mode": "smart", "deny": [], "ask": []}}

    def set_ask(patterns, **extra):
        state["config"] = {"mode": "smart", "deny": [], "ask": list(patterns), **extra}

    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: state["config"])
    return set_ask


@pytest.fixture
def clean_env(monkeypatch):
    """Non-interactive, non-gateway, non-cron, non-yolo baseline; no transport."""
    for var in ("HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION",
                "HERMES_INTERACTIVE", "HERMES_EXEC_ASK", "HERMES_APPROVAL_TRANSPORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)


@pytest.fixture
def session(monkeypatch):
    """A fresh, cleared session key for this test."""
    key = "ask-rules-test-session"
    mod.clear_session(key)
    monkeypatch.setattr(mod, "get_current_session_key", lambda default="": key)
    return key


class FakePrompt:
    def __init__(self):
        self.answer = "once"
        self.calls = []

    def __call__(self, command, description, allow_permanent=True, smart_denied=False,
                 approval_callback=None):
        self.calls.append({"command": command, "description": description,
                           "allow_permanent": allow_permanent})
        return self.answer


class FakeSmart:
    def __init__(self):
        self.verdict = "approve"
        self.calls = []

    def __call__(self, spec, command, description, pattern_key, pattern_keys, session_key,
                 human_present=False):
        self.calls.append(command)
        if self.verdict == "approve":
            result = mod._approved()
            result["smart_approved"] = True
            return result, False
        return None, False


@pytest.fixture
def cli_console(monkeypatch):
    """CLI presence (a human is reachable) with the smart reviewer and the prompt seam stubbed."""
    monkeypatch.setattr(mod, "_presence", lambda cb=None: (cb, True, False, False))
    monkeypatch.setattr(mod, "_tirith_scan",
                        lambda command: {"action": "allow", "findings": [], "summary": ""})
    prompt, smart = FakePrompt(), FakeSmart()
    monkeypatch.setattr(mod, "prompt_dangerous_approval", prompt)
    monkeypatch.setattr(mod, "_smart_gate", smart)
    return prompt, smart


class TestMatchUserAskRule:
    def test_no_config_is_noop(self, ask_config):
        ask_config([])
        assert approval_floors._match_user_ask_rule("systemctl restart nginx") is None

    def test_missing_key_is_noop(self, monkeypatch):
        monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "smart"})
        assert approval_floors._match_user_ask_rule(DANGEROUS) is None

    def test_config_load_failure_fails_open(self, monkeypatch):
        def boom():
            raise RuntimeError("config unavailable")
        monkeypatch.setattr(approval_context, "_get_approval_config", boom)
        assert approval_floors._match_user_ask_rule(DANGEROUS) is None

    def test_quote_obfuscation_still_matches(self, ask_config):
        ask_config(["git push --force*"])
        assert approval_floors._match_user_ask_rule('git pu""sh --force origin main') is not None

    def test_pattern_is_returned(self, ask_config):
        ask_config(["rm -rf *"])
        assert approval_floors._match_user_ask_rule(DANGEROUS) == "rm -rf *"


def test_smart_auto_approves_without_ask_rule(ask_config, clean_env, session, cli_console):
    """Control: in smart mode a flagged command is auto-approved by the reviewer."""
    prompt, smart = cli_console
    ask_config([])
    result = mod.check_all_command_guards(DANGEROUS, "local")
    assert result["approved"] is True and result.get("smart_approved") is True
    assert len(smart.calls) == 1 and not prompt.calls


def test_ask_forces_prompt_and_skips_smart(ask_config, clean_env, session, cli_console):
    prompt, smart = cli_console
    ask_config(["rm -rf *"])
    result = mod.check_all_command_guards(DANGEROUS, "local")
    assert result["approved"] is True
    assert not smart.calls, "an ask rule must skip the smart reviewer"
    assert len(prompt.calls) == 1
    assert prompt.calls[0]["allow_permanent"] is False, \
        "ask-forced prompts offer once/[s]ession only — the rule lives in config"


def test_ask_prompts_even_when_nothing_flagged(ask_config, clean_env, session, cli_console):
    """The ask list is a standing instruction: matches always come to a human."""
    prompt, smart = cli_console
    ask_config(["echo *"])
    result = mod.check_all_command_guards("echo hello", "local")
    assert result["approved"] is True and len(prompt.calls) == 1 and not smart.calls


def test_ask_beats_permanent_allowlist(ask_config, clean_env, session, cli_console, monkeypatch):
    prompt, smart = cli_console
    monkeypatch.setattr(mod, "_permanent_set", lambda: {DANGEROUS, "recursive delete"})
    ask_config([])
    allowed = mod.check_all_command_guards(DANGEROUS, "local")
    assert allowed["approved"] is True and not prompt.calls and not smart.calls
    prompt.calls.clear()
    smart.calls.clear()
    ask_config(["rm -rf *"])
    forced = mod.check_all_command_guards(DANGEROUS, "local")
    assert forced["approved"] is True and len(prompt.calls) == 1 and not smart.calls, \
        "ask rules outrank the permanent allowlist"


def test_ask_session_choice_silences_rule_for_session(ask_config, clean_env, session, cli_console):
    prompt, smart = cli_console
    ask_config(["rm -rf *"])
    prompt.answer = "session"
    first = mod.check_all_command_guards(DANGEROUS, "local")
    assert first["approved"] is True and len(prompt.calls) == 1
    # Second call: the [s]ession choice silenced the ask rule → normal smart path.
    prompt.answer = "once"
    second = mod.check_all_command_guards(DANGEROUS, "local")
    assert second.get("smart_approved") is True
    assert len(prompt.calls) == 1 and len(smart.calls) == 1


def test_deny_beats_ask(ask_config, clean_env, session, cli_console):
    prompt, smart = cli_console
    ask_config(["rm -rf *"], deny=["rm -rf *"])
    result = mod.check_all_command_guards(DANGEROUS, "local")
    assert result.get("user_deny") is True and result["approved"] is False
    assert not prompt.calls and not smart.calls


def test_ask_ignored_under_yolo(ask_config, clean_env, session, cli_console):
    prompt, smart = cli_console
    ask_config(["rm -rf *"])
    mod._YOLO_MODE_FROZEN = True
    try:
        result = mod.check_all_command_guards(DANGEROUS, "local")
    finally:
        mod._YOLO_MODE_FROZEN = False
    assert result["approved"] is True and not prompt.calls and not smart.calls


def test_ask_does_not_prompt_without_human_surface(ask_config, clean_env, session, cli_console,
                                                   monkeypatch):
    """No CLI/gateway/ask surface → ask stays inert; the unattended contract applies."""
    prompt, smart = cli_console
    monkeypatch.setattr(mod, "_presence", lambda cb=None: (cb, False, False, False))
    ask_config(["rm -rf *"])
    result = mod.check_all_command_guards(DANGEROUS, "local")
    assert result["approved"] is True and not prompt.calls and not smart.calls
