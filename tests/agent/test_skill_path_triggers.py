"""Path-triggered skill hints: ``paths:`` in SKILL.md frontmatter connects a touched
path to the skill that covers it (agent/skill_path_triggers.py + the tracker wiring)."""

from pathlib import Path

import pytest

from agent import skill_path_triggers
from agent.subdirectory_hints import SubdirectoryHintTracker


@pytest.fixture(autouse=True)
def _fresh_cache():
    skill_path_triggers.reset_cache()
    yield
    skill_path_triggers.reset_cache()


def _write_skill(skills_dir: Path, name: str, paths, *, category="cat", description="Covers project X."):
    skill_dir = skills_dir / category / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(paths, str):
        paths_yaml = repr(paths)
    elif isinstance(paths, list):
        paths_yaml = "[" + ", ".join(repr(p) for p in paths) + "]"
    else:
        paths_yaml = str(paths)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\npaths: {paths_yaml}\n---\n\nBody.\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr("hermes_constants.get_skills_dir", lambda: root)
    return root


class TestMatching:
    def test_absolute_pattern_matches_touched_path(self, tmp_path, skills_dir):
        project = tmp_path / "srv" / "app"
        project.mkdir(parents=True)
        _write_skill(skills_dir, "deploy-ops", [f"{project}/**"])
        touched = project / "main.py"
        touched.write_text("x", encoding="utf-8")
        out = skill_path_triggers.format_touched_skill_hints([touched], tmp_path)
        assert "deploy-ops" in out

    def test_relative_pattern_matches_under_working_dir(self, tmp_path, skills_dir):
        _write_skill(skills_dir, "sub2api-ops", ["sub2api/**"])
        touched = tmp_path / "sub2api" / "deploy" / "main.py"
        touched.parent.mkdir(parents=True)
        touched.write_text("x", encoding="utf-8")
        out = skill_path_triggers.format_touched_skill_hints([touched], tmp_path)
        assert "sub2api-ops" in out

    def test_no_match_returns_empty(self, tmp_path, skills_dir):
        _write_skill(skills_dir, "only-bak", ["**/*.bak"])
        touched = tmp_path / "notes.txt"
        touched.write_text("x", encoding="utf-8")
        assert skill_path_triggers.format_touched_skill_hints([touched], tmp_path) == ""

    def test_malformed_paths_never_promote_to_match_all(self, tmp_path, skills_dir):
        # A non-string/list ``paths:`` value must skip the skill entirely…
        _write_skill(skills_dir, "broken", "{a: b}")
        # …while a well-formed sibling keeps working.
        _write_skill(skills_dir, "good", ["**/*.bak"])
        touched = tmp_path / "x.bak"
        touched.write_text("x", encoding="utf-8")
        out = skill_path_triggers.format_touched_skill_hints([touched], tmp_path)
        assert "good" in out
        assert "broken" not in out

    def test_missing_skills_dir_returns_empty(self, tmp_path):
        out = skill_path_triggers.format_touched_skill_hints(
            [tmp_path / "x.py"], tmp_path, skills_dir=tmp_path / "nope")
        assert out == ""

    def test_cap_and_section_ceiling(self, tmp_path, skills_dir):
        for i in range(5):
            _write_skill(skills_dir, f"skill-{i}", ["**/*.py"])
        touched = tmp_path / "x.py"
        touched.write_text("x", encoding="utf-8")
        out = skill_path_triggers.format_touched_skill_hints([touched], tmp_path)
        assert len(out) <= skill_path_triggers.MAX_SECTION_CHARS
        assert out.count("\n- ") == skill_path_triggers.MAX_HINTED_SKILLS

    def test_starved_budget_skips_section(self, tmp_path, skills_dir):
        _write_skill(skills_dir, "any-py", ["**/*.py"])
        touched = tmp_path / "x.py"
        touched.write_text("x", encoding="utf-8")
        assert skill_path_triggers.format_touched_skill_hints(
            [touched], tmp_path, max_chars=10) == ""

    def test_claimed_set_gives_exactly_once_semantics(self, tmp_path, skills_dir):
        _write_skill(skills_dir, "any-py", ["**/*.py"])
        touched = tmp_path / "x.py"
        touched.write_text("x", encoding="utf-8")
        claimed: set = set()
        first = skill_path_triggers.format_touched_skill_hints(
            [touched], tmp_path, claimed=claimed)
        assert "any-py" in first and "any-py" in claimed
        second = skill_path_triggers.format_touched_skill_hints(
            [touched], tmp_path, claimed=claimed)
        assert second == ""


class TestTrackerWiring:
    def test_tracker_appends_skill_hint_on_touch(self, tmp_path, skills_dir):
        _write_skill(skills_dir, "bak-keeper", ["**/*.bak"])
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        touched = tmp_path / "data.bak"
        touched.write_text("x", encoding="utf-8")
        hints = tracker.check_tool_call("write_file", {"path": str(touched)})
        assert hints and "bak-keeper" in hints

    def test_tracker_silent_for_unmatched_paths(self, tmp_path, skills_dir):
        _write_skill(skills_dir, "bak-keeper", ["**/*.bak"])
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        touched = tmp_path / "data.txt"
        touched.write_text("x", encoding="utf-8")
        assert tracker.check_tool_call("write_file", {"path": str(touched)}) is None

    def test_tracker_hints_once_per_session(self, tmp_path, skills_dir):
        _write_skill(skills_dir, "bak-keeper", ["**/*.bak"])
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        touched = tmp_path / "data.bak"
        touched.write_text("x", encoding="utf-8")
        first = tracker.check_tool_call("write_file", {"path": str(touched)})
        assert first and "bak-keeper" in first
        assert tracker.check_tool_call("write_file", {"path": str(touched)}) is None

    def test_disabled_tracker_never_hints(self, tmp_path, skills_dir):
        _write_skill(skills_dir, "bak-keeper", ["**/*.bak"])
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path), enabled=False)
        touched = tmp_path / "data.bak"
        touched.write_text("x", encoding="utf-8")
        assert tracker.check_tool_call("write_file", {"path": str(touched)}) is None
