"""Turn-end guard for ``hermes -z`` one-shot task runs.

A one-shot run exists to execute a task; it must not end before the agent
has executed at least one tool. Some models (observed: deepseek-v4-flash,
2026-08-16) open with a narration-only message and stop with
``finish_reason=stop`` and zero tool calls — Hermes treats that as a clean
exit and the whole run ends with the task unstarted.

This module is policy-only: when a one-shot conversation tries to finish
before any tool call, return a bounded synthetic nudge so the conversation
loop continues instead of exiting.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

_DEFAULT_MAX_ATTEMPTS = 3


def oneshot_start_nudge_enabled() -> bool:
    """Return whether the one-shot no-tool stop-guard is active.

    On by default; ``HERMES_ONESHOT_START_NUDGE=0/false/no/off`` disables it.
    """
    env = os.environ.get("HERMES_ONESHOT_START_NUDGE")
    if env is not None and env.strip().lower() in {"0", "false", "no", "off"}:
        return False
    return True


def has_executed_tool_call(messages: Iterable[dict] | None) -> bool:
    """True when any assistant message in this conversation issued a tool call."""
    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return True
    return False


def build_oneshot_start_nudge(
    *,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> Optional[str]:
    """Return a synthetic follow-up when a one-shot run stops before any tool call.

    Returns ``None`` when the guard is disabled or the nudge budget is
    exhausted.
    """
    if not oneshot_start_nudge_enabled():
        return None
    if attempts >= max_attempts:
        return None

    return (
        "[System: This run must not end yet — you have not called any tool. "
        "A text-only reply does not execute the assigned task. In this "
        "response, immediately call a tool to start the actual work: inspect "
        "the workspace, read the key files, run the diagnostics. Do not "
        "describe what you plan to do — act. (这个任务还没有开始执行:请立刻调用工具动手"
        "干活——查看工作区、读取关键文件、运行诊断;不要只输出文字说明或计划。)]"
    )


__all__ = [
    "build_oneshot_start_nudge",
    "has_executed_tool_call",
    "oneshot_start_nudge_enabled",
]
