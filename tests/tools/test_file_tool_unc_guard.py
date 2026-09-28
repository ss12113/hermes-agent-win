"""UNC/network path guards for file tools.

Claude Code v2.1.88 avoids filesystem probes on UNC paths because Windows can
initiate SMB/WebDAV authentication while merely checking path existence. Hermes
file tools should fail before resolving/probing those paths.
"""

import json
from unittest.mock import patch

import pytest

from tools import file_tools


UNC_BACKSLASH = r"\\server\share\secrets.txt"
UNC_SLASH = "//server/share/secrets.txt"
UNC_EXTENDED = "\\\\?\\UNC\\server\\share\\secrets.txt"


def _error_text(result: str) -> str:
    payload = json.loads(result)
    return payload.get("error", "")


class TestUncPathDetection:
    def test_detects_common_unc_forms(self):
        assert file_tools._looks_like_unc_path(UNC_BACKSLASH) is True
        assert file_tools._looks_like_unc_path(UNC_SLASH) is True
        assert file_tools._looks_like_unc_path(UNC_EXTENDED) is True

    def test_does_not_flag_normal_paths(self):
        assert file_tools._looks_like_unc_path("/tmp/work/file.txt") is False
        assert file_tools._looks_like_unc_path("relative/file.txt") is False
        assert file_tools._looks_like_unc_path("C:/Users/me/file.txt") is False
        assert file_tools._looks_like_unc_path("//server") is False

    def test_resolver_refuses_before_filesystem_probe(self):
        with pytest.raises(ValueError, match="UNC/network paths"):
            file_tools._resolve_path_for_task(UNC_BACKSLASH, task_id="unc_resolve")


class TestFileToolsRejectUncPaths:
    @patch("tools.file_tools._get_file_ops")
    def test_read_file_rejects_unc_before_file_ops(self, mock_ops):
        err = _error_text(file_tools.read_file_tool(UNC_BACKSLASH, task_id="unc_read"))

        assert "UNC/network paths" in err
        assert "SMB/WebDAV" in err
        mock_ops.assert_not_called()

    @patch("tools.file_tools._resolve_path_for_task")
    @patch("tools.file_tools._get_file_ops")
    def test_write_file_rejects_unc_before_resolve_or_file_ops(self, mock_ops, mock_resolve):
        err = _error_text(file_tools.write_file_tool(UNC_SLASH, "content", task_id="unc_write"))

        assert "UNC/network paths" in err
        mock_resolve.assert_not_called()
        mock_ops.assert_not_called()

    @patch("tools.file_tools._resolve_path_for_task")
    @patch("tools.file_tools._get_file_ops")
    def test_patch_replace_rejects_unc_before_resolve_or_file_ops(self, mock_ops, mock_resolve):
        err = _error_text(file_tools.patch_tool(
            mode="replace",
            path=UNC_BACKSLASH,
            old_string="old",
            new_string="new",
            task_id="unc_patch_replace",
        ))

        assert "UNC/network paths" in err
        mock_resolve.assert_not_called()
        mock_ops.assert_not_called()

    @patch("tools.file_tools._resolve_path_for_task")
    @patch("tools.file_tools._get_file_ops")
    def test_patch_v4a_rejects_unc_header_before_resolve_or_file_ops(self, mock_ops, mock_resolve):
        patch_text = "\n".join([
            "*** Begin Patch",
            "*** Update File: //server/share/secrets.txt",
            "@@",
            "-old",
            "+new",
            "*** End Patch",
        ])

        err = _error_text(file_tools.patch_tool(
            mode="patch",
            patch=patch_text,
            task_id="unc_patch_v4a",
        ))

        assert "UNC/network paths" in err
        mock_resolve.assert_not_called()
        mock_ops.assert_not_called()

    @patch("tools.file_tools._get_file_ops")
    def test_search_rejects_unc_before_file_ops(self, mock_ops):
        err = _error_text(file_tools.search_tool("needle", path=UNC_SLASH, task_id="unc_search"))

        assert "UNC/network paths" in err
        mock_ops.assert_not_called()
