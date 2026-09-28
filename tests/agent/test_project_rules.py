"""Tests for Claude-Code-style project rules in ``.hermes/rules``."""

from pathlib import Path

import pytest

from agent.project_rules import ProjectRuleSet


REPO_ROOT = Path(__file__).resolve().parents[2]


def _rule(root: Path, name: str, content: str) -> Path:
    path = root / ".hermes" / "rules" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_unconditional_and_scoped_rules_are_separated(tmp_path):
    (tmp_path / ".git").mkdir()
    _rule(tmp_path, "global.md", "Always-on rule")
    _rule(
        tmp_path,
        "scoped.md",
        "---\npaths:\n  - src/**\n---\nScoped rule",
    )

    rules = ProjectRuleSet.discover(tmp_path)

    startup = rules.startup_text()
    assert "Always-on rule" in startup
    assert "Scoped rule" not in startup
    assert "Scoped rule" in rules.claim_matches([tmp_path / "src" / "main.py"])


def test_rules_match_directory_targets_and_deduplicate(tmp_path):
    (tmp_path / "src").mkdir()
    _rule(
        tmp_path,
        "python.md",
        "---\npaths: src/**\n---\nPython rule",
    )
    rules = ProjectRuleSet.discover(tmp_path)

    first = rules.claim_matches([tmp_path / "src"])
    second = rules.claim_matches([tmp_path / "src" / "nested.py"])

    assert "Python rule" in first
    assert second == ""


def test_nested_rule_patterns_are_relative_to_the_rule_owning_project(tmp_path):
    nested = tmp_path / "nested-project"
    target = nested / "src" / "main.py"
    target.parent.mkdir(parents=True)
    rules_dir = nested / ".claude" / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "source.md").write_text(
        "---\npaths: src/**\n---\nNested source rule",
        encoding="utf-8",
    )
    rules = ProjectRuleSet.discover(tmp_path)

    result = rules.claim_matches([target])

    assert "Nested source rule" in result


def test_invalid_frontmatter_is_fail_closed(tmp_path):
    _rule(
        tmp_path,
        "broken.md",
        "---\npaths: [unterminated\n---\nShould not load",
    )
    rules = ProjectRuleSet.discover(tmp_path)

    assert rules.startup_text() == ""
    assert rules.claim_matches([tmp_path / "src" / "main.py"]) == ""


@pytest.mark.parametrize(
    "content",
    [
        "\n---\npaths: src/**\n---\nShould not become global",
        " ---\npaths: src/**\n---\nShould not become global",
        "--- # malformed opener\npaths: src/**\n---\nShould not become global",
    ],
)
def test_frontmatter_near_miss_is_fail_closed(tmp_path, content):
    _rule(tmp_path, "near-miss.md", content)
    rules = ProjectRuleSet.discover(tmp_path)

    assert "Should not become global" not in rules.startup_text()
    assert rules.claim_matches([tmp_path / "src" / "main.py"]) == ""


def test_startup_rule_budget_is_a_hard_character_limit(tmp_path):
    _rule(tmp_path, "a.md", "A" * 200)
    _rule(tmp_path, "b.md", "B" * 200)
    rules = ProjectRuleSet.discover(tmp_path, max_startup_chars=220)

    result = rules.startup_text()

    assert len(result) <= 220


def test_zero_startup_and_lazy_budget_loads_nothing(tmp_path):
    _rule(tmp_path, "global.md", "global")
    _rule(tmp_path, "scoped.md", "---\npaths: src/**\n---\nscoped")
    rules = ProjectRuleSet.discover(
        tmp_path,
        max_startup_chars=0,
        max_lazy_chars=0,
    )

    assert rules.startup_text() == ""
    assert rules.claim_matches([tmp_path / "src" / "main.py"]) == ""


def test_lazy_budget_emits_recovery_path_and_retries_deferred_rule(tmp_path):
    _rule(tmp_path, "one.md", "---\npaths: src/**\n---\n" + "one " * 25)
    _rule(tmp_path, "two.md", "---\npaths: src/**\n---\n" + "two " * 25)
    rules = ProjectRuleSet.discover(tmp_path, max_lazy_chars=230)

    first = rules.claim_matches([tmp_path / "src" / "main.py"])
    second = rules.claim_matches([tmp_path / "src" / "other.py"])

    assert len(first) <= 230
    assert ".hermes/rules/two.md" in first
    assert "one one" in first
    assert "two two" in second


def test_large_rule_keeps_full_snapshot_and_delivers_recovery_marker(tmp_path):
    body = "full-rule-body " * 100
    _rule(tmp_path, "large.md", "---\npaths: src/**\n---\n" + body)
    rules = ProjectRuleSet.discover(tmp_path, max_rule_chars=120, max_lazy_chars=200)

    result = rules.claim_matches([tmp_path / "src" / "main.py"])

    assert len(result) <= 200
    assert "read_file" in result
    assert ".hermes/rules/large.md" in result
    assert len(rules.rules[0].body) == len(body.strip())


def test_rule_count_limit_has_recovery_marker(tmp_path):
    for name in ("a.md", "b.md", "c.md"):
        _rule(tmp_path, name, f"rule {name}")
    rules = ProjectRuleSet.discover(tmp_path, max_rule_count=1)

    result = rules.startup_text()

    assert ".hermes/rules/b.md" in result
    assert ".hermes/rules/c.md" in result


def test_rule_symlink_cannot_escape_workspace(tmp_path):
    outside = tmp_path.parent / "outside-rule.md"
    outside.write_text("outside rule", encoding="utf-8")
    rules_dir = tmp_path / ".hermes" / "rules"
    rules_dir.mkdir(parents=True)
    link = rules_dir / "escape.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this filesystem")

    rules = ProjectRuleSet.discover(tmp_path)

    assert "outside rule" not in rules.startup_text()


def test_repository_root_stays_below_historical_default_cap():
    from agent.prompt_builder import CONTEXT_FILE_MAX_CHARS, build_context_files_prompt

    prompt = build_context_files_prompt(cwd=str(REPO_ROOT), skip_soul=True)

    assert len(prompt) <= CONTEXT_FILE_MAX_CHARS
    assert "TRUNCATED" not in prompt
    assert "## Contribution Rubric" not in prompt
    assert "## Known Pitfalls" not in prompt
    assert "## Scoped Subsystem Instructions" in prompt


@pytest.mark.parametrize(
    ("relative_path", "heading"),
    [
        ("run_agent.py", "## AIAgent Class"),
        ("cli.py", "## CLI Architecture"),
        ("ui-tui/src/app.tsx", "## TUI Architecture"),
        ("tools/registry.py", "## Adding New Tools"),
        ("plugins/example.py", "## Plugins"),
        ("skills/example/SKILL.md", "## Skills"),
        ("agent/delegation.py", "## Delegation"),
        ("cron/jobs.py", "## Cron"),
        ("plugins/kanban/worker.py", "## Kanban"),
        ("hermes_cli/kanban.py", "## Kanban"),
        ("agent/curator.py", "## Curator"),
        ("agent/memory_provider.py", "## Plugins"),
        ("agent/context_engine.py", "## Plugins"),
        ("gateway/slash_commands.py", "## CLI Architecture"),
        ("hermes_cli/skin_engine.py", "## Skin/Theme System"),
        ("hermes_cli/banner.py", "## Skin/Theme System"),
        ("agent/display.py", "## Skin/Theme System"),
        ("pyproject.toml", "## Contribution Rubric"),
        ("README.md", "## Contribution Rubric"),
        ("agent/prompt_builder.py", "## Contribution Rubric"),
        ("agent/prompt_builder.py", "## Known Pitfalls"),
    ],
)
def test_repository_scoped_rule_is_lazy_for_representative_path(relative_path, heading):
    from agent.subdirectory_hints import SubdirectoryHintTracker

    tracker = SubdirectoryHintTracker(working_dir=str(REPO_ROOT))
    result = tracker.check_tool_call(
        "read_file", {"path": str(REPO_ROOT / relative_path)}
    )

    assert result is not None
    assert heading in result


def test_large_migrated_rules_reach_their_tail_without_truncation():
    from agent.subdirectory_hints import SubdirectoryHintTracker

    tracker = SubdirectoryHintTracker(working_dir=str(REPO_ROOT))
    result = tracker.check_tool_call(
        "read_file", {"path": str(REPO_ROOT / "ui-tui" / "src" / "app.tsx")}
    )
    assert result is not None
    assert "Desktop Chat App" in result
    assert "Skill commands and `quick_commands` are extensions" in result

    tracker = SubdirectoryHintTracker(working_dir=str(REPO_ROOT))
    result = tracker.check_tool_call(
        "read_file", {"path": str(REPO_ROOT / "agent" / "prompt_builder.py")}
    )
    assert result is not None
    assert "The Footprint Ladder" in result
    assert "DO NOT hardcode `~/.hermes` paths" in result
