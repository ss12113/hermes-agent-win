from __future__ import annotations

import os
import subprocess

import agent.git_diff_summary as git_diff_summary
from agent.git_diff_summary import (
    collect_git_diff_summary,
    format_git_diff_summary,
    parse_numstat,
    parse_shortstat,
)


def _git(cwd, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Hermes Test",
            "GIT_AUTHOR_EMAIL": "hermes@example.test",
            "GIT_COMMITTER_NAME": "Hermes Test",
            "GIT_COMMITTER_EMAIL": "hermes@example.test",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_parse_shortstat_handles_optional_groups():
    assert parse_shortstat("1 file changed, 2 insertions(+)") == {
        "files": 1,
        "insertions": 2,
        "deletions": 0,
    }
    assert parse_shortstat("2 files changed, 1 insertion(+), 3 deletions(-)") == {
        "files": 2,
        "insertions": 1,
        "deletions": 3,
    }
    assert parse_shortstat("") == {"files": 0, "insertions": 0, "deletions": 0}


def test_parse_numstat_preserves_binary_and_tabs_in_filename():
    entries = parse_numstat("3\t1\tsrc/a file.py\n-\t-\tassets/logo.png\n4\t5\tweird\tname.txt\n")

    assert entries[0].path == "src/a file.py"
    assert entries[0].additions == 3
    assert entries[0].deletions == 1
    assert entries[1].binary is True
    assert entries[1].path == "assets/logo.png"
    assert entries[2].path == "weird\tname.txt"


def test_collect_git_diff_summary_includes_tracked_and_untracked(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "tracked.txt").write_text("one\ntwo\n", encoding="utf-8")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")

    summary = collect_git_diff_summary(repo)

    assert summary.error is None
    assert summary.repo_root == str(repo)
    assert summary.shortstat["files"] == 1
    assert summary.shortstat["insertions"] == 1
    assert [entry.path for entry in summary.files] == ["tracked.txt"]
    assert summary.untracked == ("new.txt",)

    rendered = format_git_diff_summary(repo)
    assert "Git diff summary for" in rendered
    assert "tracked.txt (+1/-0)" in rendered
    assert "?? new.txt" in rendered


def test_format_git_diff_summary_clean_repo(tmp_path):
    repo = _init_repo(tmp_path)

    assert format_git_diff_summary(repo) == "Working tree is clean."


def test_format_git_diff_summary_non_git_directory(tmp_path):
    assert format_git_diff_summary(tmp_path) == "Not in a git repository."
