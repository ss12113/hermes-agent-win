"""Anchored tool bootstrap for DeepSeek V4 Pro (port of dsh-anchored-standard).

Rationale (upstream): DeepSeek V4 Pro conditions strongly on the FIRST API
request — Minimal mode (2 tools) scores 99/96 while Standard (25 tools) scores
91/92 on the same task. The community plugin ``dsh-anchored-standard``
(https://github.com/xiaobright/dsh-anchored-standard, MIT) reproduces the
Minimal first request, then exposes the full Standard catalog once the session
records its first durable promotion signal (a tool call or the first assistant
message).

Three mechanisms (all request-local — the cached system prompt, tool registry
and token accounting are never mutated):

1. **Tool bootstrap** (``apply_tool_bootstrap``): request #1 exposes only the
   bootstrap subset — one shell + the edit tool (Minimal-aligned; upstream
   issue #11 proved the first-request tool *schema* is the deciding anchor, and
   ``read``-based surfaces failed 11/11 while ``bash+str_replace_editor``
   anchored 5/5). Request #2 onward sees the full catalog.

2. **Context strip** (``strip_bootstrap_context``): request #1 also strips the
   two auto-injected context blocks — the workspace instruction digest
   (AGENTS.md / CLAUDE.md / .cursorrules, upstream ``agent-instructions``) and
   the available-skills reminder (upstream ``skill-catalog``). Both return
   unchanged from request #2. Issue #6/#12: with the skill catalog present the
   anchor did not reproduce at all (0/9); without it ~81%.

3. **Output cap** (``bootstrap_max_output_tokens``): optional first-request
   ``max_tokens`` cap. Default 0 = disabled — issue #11 showed the cap is NOT
   the deciding variable (it was silently overridden by adapterDefaults on the
   official endpoint), and a real 1024 cap truncates first-round reasoning
   against Hermes' much-larger-than-Minimal core instructions.

4. **Auto-continuation** (``agent/conversation_loop.py``): the anchored round-1
   reply cannot execute anything (zero tools by construction), so the loop
   emits it as an interim assistant message, re-prompts with the synthetic
   continuation nudge and keeps going — the user's task completes in the same
   turn with the full catalog. ``apply_tool_bootstrap`` flags each anchored
   request via ``_AGENT_ACTIVE_REQUEST_ATTR``; the loop consumes the flag and
   fires exactly once (promotion is monotonic).

Subagents are exempt (issue #15): a delegated child runs in its own fresh
session, so bootstrapping its first request would wrongly re-anchor it and the
cap could truncate a long delegated output.

Config (``config.yaml`` → ``agent.anchored_tool_bootstrap``)::

    agent:
      anchored_tool_bootstrap:
        enabled: true
        # Empty list = every model. Substring match against the model name.
        models: [deepseek-v4-pro]
        # First-request tool subset. Must exist in the catalog or the
        # bootstrap degrades gracefully to the full catalog (no lockout).
        bootstrap_tools: [terminal, patch]
        # Full Minimal-condition port (upstream zero-anchored-standard):
        # expose ZERO tools on request #1 (default false keeps the 2-tool
        # surface). The first assistant message then promotes the session.
        bootstrap_zero_tools: false
        # Replace the whole request-#1 system prompt with this minimal
        # prompt (upstream "Keep the complete Minimal system prompt").
        # Empty = keep the stripped full prompt (default, no replacement).
        bootstrap_minimal_system_prompt: ""
        # Promotion signal: "tool-call" (default) or "either" (tool call OR
        # first assistant message; required for zero-tools anchoring, where
        # request #1 cannot produce a tool call by construction).
        promote_on: tool-call
        # First-request output cap; 0 = disabled (recommended).
        bootstrap_max_tokens: 0
        # Auto-injected context stripped from request #1; [] disables the strip.
        suppressed_context_sources: [agent-instructions, skill-catalog]

Disabled by default; absent config is a no-op returning the catalog intact.
Only models in ``models`` are affected — every other model is untouched.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_TOOLS = ("terminal", "patch")

# First-request output budget cap. Upstream issue #6 initially reported 1024 as
# the dominant trigger, but issue #11 later proved the maxTokens fix was silently
# overridden by adapterDefaults on the official endpoint AND that at the default
# 256000 budget the FIRST-REQUEST TOOL SCHEMA alone decides the anchor (minimal's
# bash+str_replace_editor anchored 5/5, every read-based surface failed). The cap
# is therefore NOT the deciding variable, and on Hermes a real 1024 output cap
# truncates the first-round reasoning against the (much larger than Minimal)
# core instructions. Default to 0 = no cap; keep the knob for experimenters.
DEFAULT_BOOTSTRAP_MAX_TOKENS = 0

# Context sources stripped from request #1 by default. Mirrors the upstream
# ``suppressedContextSources``: the two automatic injections Standard adds over
# Minimal — the workspace instruction digest (AGENTS.md / CLAUDE.md /
# .cursorrules → ``agent-instructions``) and the available-skills reminder
# (``skill-catalog``). An explicitly empty list disables the context filter
# while keeping the tool bootstrap and the output cap.
DEFAULT_SUPPRESSED_SOURCES = ("agent-instructions", "skill-catalog")

# Agent attribute the resolved config is cached on (loaded once per agent).
_AGENT_CFG_ATTR = "_anchored_tool_bootstrap_config"
# Once a tool call is observed, retain promotion on the cached agent even if
# context compression later removes the original tool-call messages from the
# API-visible transcript.  The upstream DSH plugin checks append-only session
# events; this latch gives Hermes the same monotonic per-session behaviour.
_AGENT_PROMOTED_ATTR = "_anchored_tool_bootstrap_promoted"
# Set on the agent for the lifetime of ONE request: True only while that
# request got the anchored treatment (zero tools / bootstrap subset).
# The conversation loop consumes it to auto-continue the turn after the
# anchored round-1 reply — the Minimal first round cannot execute anything
# by construction, so ending the turn there would strand the user's task.
_AGENT_ACTIVE_REQUEST_ATTR = "_anchored_tool_bootstrap_active_request"


def _resolve_config(agent: Any) -> Optional[Dict[str, Any]]:
    """Return the resolved bootstrap config for this agent, or None if off.

    The config is read from ``config.yaml`` on first use and cached on the
    agent instance so the hot request path never re-reads the file.
    """
    cfg = getattr(agent, _AGENT_CFG_ATTR, None)
    if cfg is None:
        cfg = {}
        try:
            from hermes_cli.config import load_config_readonly

            cfg = (
                (load_config_readonly() or {}).get("agent", {}).get("anchored_tool_bootstrap") or {}
            )
            if not isinstance(cfg, dict):
                cfg = {}
        except Exception as exc:  # pragma: no cover - defensive in hot path
            logger.debug("anchored_tool_bootstrap: config read failed: %s", exc)
            cfg = {}
        setattr(agent, _AGENT_CFG_ATTR, cfg)
    if not cfg.get("enabled", False):
        return None
    return cfg


def _model_matches(agent: Any, cfg: Dict[str, Any]) -> bool:
    """Model allow-list; empty list matches every model (substring match)."""
    models = cfg.get("models") or []
    if not models:
        return True
    model = (getattr(agent, "model", "") or "").strip().lower()
    return any(str(m).strip().lower() in model for m in models if str(m).strip())


def _tool_name(tool: Any) -> Optional[str]:
    """Extract the tool name from an API tool schema entry.

    Hermes registers tools in the OpenAI nested layout
    ``{"type": "function", "function": {"name": ...}}``; some paths
    (and the upstream dsh plugin) use the flat ``{"name": ...}``
    layout. Accept both so the filter matches the real catalog.
    """
    if not isinstance(tool, dict):
        return None
    name = tool.get("name")
    if isinstance(name, str) and name:
        return name
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if isinstance(name, str) and name:
            return name
    return None


def _history_promotion_signal(
    api_messages: Optional[List[Dict[str, Any]]],
) -> Optional[str]:
    """Return the first durable promotion signal in session history.

    ``"tool-call"``: an assistant message carrying ``tool_calls`` or a ``tool``
    result message (which only exists after one) — equivalence of the upstream
    ``session.events.some(type == 'tool/call')`` check.

    ``"assistant-message"``: any assistant message, with or without tool calls
    — equivalence of the upstream ``promoteOn: either`` second arm
    (``assistant/message``). Required for zero-tools anchoring, where request
    #1 cannot produce a tool call by construction.
    """
    if not isinstance(api_messages, list):
        return None
    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            if msg.get("tool_calls"):
                return "tool-call"
            return "assistant-message"
        if role == "tool":
            return "tool-call"
    return None


def _history_has_promotion(
    api_messages: Optional[List[Dict[str, Any]]],
    *,
    promote_on: str = "tool-call",
) -> bool:
    """True when history carries a signal that satisfies the promote_on rule."""
    signal = _history_promotion_signal(api_messages)
    if signal is None:
        return False
    if promote_on == "either":
        return True
    return signal == "tool-call"


def apply_tool_bootstrap(
    agent: Any,
    tools_for_api: Optional[List[Dict[str, Any]]],
    api_messages: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Filter the request-local tool catalog for a session's first request.

    Returns the original ``tools_for_api`` unchanged when the feature is
    disabled, the model is not targeted, the session already made a tool
    call, or the bootstrap subset is empty/unavailable (graceful fallback —
    never lock the agent out of its tools).
    """
    # Per-request marker consumed by the conversation loop's anchored
    # round-1 auto-continuation. Reset on every call so a stale True from
    # a previous request can never re-trigger a continuation.
    setattr(agent, _AGENT_ACTIVE_REQUEST_ATTR, False)
    cfg = _resolve_config(agent)
    if cfg is None:
        return tools_for_api
    # Subagents always see the full catalog (upstream issue #15): a delegated
    # child runs in its own fresh session, so the bootstrap would wrongly
    # re-anchor its first request (and a first-round cap could truncate a long
    # delegated output). Hermes flags delegated children via ``is_subagent``.
    if getattr(agent, "is_subagent", False):
        return tools_for_api
    # Promotion is monotonic for the lifetime of the cached session agent.
    # Scan before model matching so a session that used tools under another
    # model is not re-anchored if it later switches to a targeted V4 Pro.
    promote_on = str(cfg.get("promote_on") or "tool-call").strip()
    if _history_has_promotion(api_messages, promote_on=promote_on):
        setattr(agent, _AGENT_PROMOTED_ATTR, True)
    if getattr(agent, _AGENT_PROMOTED_ATTR, False):
        return tools_for_api
    if not _model_matches(agent, cfg):
        return tools_for_api
    # Some test/minimal-agent and no-tool runtime paths intentionally expose
    # None rather than an empty list.  The feature must be fail-open there.
    if not isinstance(tools_for_api, list) or not tools_for_api:
        return tools_for_api

    # Zero-tools anchoring (upstream zero-anchored-standard): request #1 sees
    # an EMPTY tool surface, so the model cannot act — it answers in prose and
    # that first assistant message promotes the session.  Bypasses the subset
    # filter below entirely (there is nothing to filter down to).
    if cfg.get("bootstrap_zero_tools"):
        setattr(agent, _AGENT_ACTIVE_REQUEST_ATTR, True)
        logger.info(
            "anchored_tool_bootstrap: first request anchored to zero tools "
            "(catalog %d -> 0 tools)",
            len(tools_for_api),
        )
        return []

    boot_cfg = cfg.get("bootstrap_tools")
    if boot_cfg is None:
        bootstrap = list(DEFAULT_BOOTSTRAP_TOOLS)
    else:
        # Explicit empty list means "do not anchor" — keep the full catalog.
        bootstrap = [
            str(t).strip()
            for t in boot_cfg
            if isinstance(t, str) and t.strip()
        ]
    if not bootstrap:
        return tools_for_api

    keep_names = set(bootstrap)
    kept = [
        t for t in tools_for_api
        if _tool_name(t) in keep_names
    ]
    if not kept:
        logger.warning(
            "anchored_tool_bootstrap: none of %s present in the tool catalog; "
            "keeping the full catalog",
            bootstrap,
        )
        return tools_for_api

    missing = [b for b in bootstrap if b not in {_tool_name(t) for t in tools_for_api}]
    if missing:
        logger.info(
            "anchored_tool_bootstrap: bootstrap tools not in catalog: %s",
            missing,
        )
    logger.info(
        "anchored_tool_bootstrap: first request anchored to %s "
        "(catalog %d -> %d tools)",
        sorted(n for n in (_tool_name(t) for t in kept) if n),
        len(tools_for_api),
        len(kept),
    )
    setattr(agent, _AGENT_ACTIVE_REQUEST_ATTR, True)
    return kept


def _remove_prompt_block(text: str, block: str) -> str:
    """Remove one system-prompt block (and its leading separator) from ``text``.

    The system prompt is ``"\\n\\n".join(parts)``, so each block sits between
    ``\\n\\n`` separators. Strip the block, find it, and cut it together with the
    separator *before* it so the neighbouring blocks stay joined by exactly one
    separator. No-op when the block is absent (fail-open — never lose context).
    """
    block = (block or "").strip()
    if not block or not text:
        return text
    idx = text.find(block)
    if idx < 0:
        return text
    start = idx
    if start >= 2 and text[start - 2:start] == "\n\n":
        start -= 2
    elif start >= 1 and text[start - 1:start] == "\n":
        start -= 1
    end = idx + len(block)
    return text[:start] + text[end:]


def _is_bootstrap_phase(agent: Any, cfg: Dict[str, Any]) -> bool:
    """True when this agent is still in the anchored (un-promoted) phase.

    Encapsulates the shared gate for the context strip and the output cap:
    feature on, model targeted, not a subagent, and no durable tool call yet.
    """
    if cfg is None:
        return False
    if getattr(agent, "is_subagent", False):
        return False
    if getattr(agent, _AGENT_PROMOTED_ATTR, False):
        return False
    if not _model_matches(agent, cfg):
        return False
    return True


def strip_bootstrap_context(agent: Any, system_prompt: Any) -> Any:
    """Strip auto-injected context from request #1, or replace the prompt.

    Mirrors the upstream ``agent/pre-step`` filter with
    ``suppressedContextSources`` (default ``['agent-instructions',
    'skill-catalog']``): while the session is un-promoted, remove the workspace
    instruction digest (AGENTS.md / CLAUDE.md / .cursorrules) and the
    available-skills reminder from the system prompt. Both return unchanged
    from request #2 on. The blocks are captured verbatim by
    ``system_prompt.build_system_prompt_parts`` on the agent, so the strip is an
    exact substring cut rather than a fragile marker regex.

    When ``bootstrap_minimal_system_prompt`` is non-empty, the WHOLE prompt is
    replaced by it on request #1 instead of being stripped — the upstream
    "Keep the complete Minimal system prompt" rule. Upstream issue #11 showed
    the first-request tool schema decides the anchor, but the dose-response
    data also shows a heavy agent identity layer (Hermes core instructions,
    MEMORY, USER PROFILE, …) keeps the trajectory standard-like even with the
    tools reduced; the minimal replacement is what makes the zero-tools
    surface reproduce the "We need" trajectory. Safety note: with zero tools
    the model cannot act on request #1, so the lean prompt cannot cause
    harmful actions — the full prompt and catalog return on promotion.
    """
    cfg = _resolve_config(agent)
    if cfg is None:
        return system_prompt
    if not _is_bootstrap_phase(agent, cfg):
        return system_prompt
    if not isinstance(system_prompt, str) or not system_prompt:
        return system_prompt

    minimal = str(cfg.get("bootstrap_minimal_system_prompt") or "").strip()
    if minimal:
        logger.info(
            "anchored_tool_bootstrap: first-request system prompt replaced "
            "by bootstrap_minimal_system_prompt (%d -> %d chars)",
            len(system_prompt),
            len(minimal),
        )
        return minimal

    suppressed = cfg.get("suppressed_context_sources")
    if suppressed is None:
        suppressed = list(DEFAULT_SUPPRESSED_SOURCES)
    else:
        suppressed = [str(s).strip() for s in suppressed if isinstance(s, str) and s.strip()]
    if not suppressed:
        return system_prompt

    result = system_prompt
    if "agent-instructions" in suppressed:
        block = getattr(agent, "_bootstrap_context_files_prompt", "") or ""
        if block:
            result = _remove_prompt_block(result, block)
    if "skill-catalog" in suppressed:
        block = getattr(agent, "_bootstrap_skills_prompt", "") or ""
        if block:
            result = _remove_prompt_block(result, block)
    if result != system_prompt:
        logger.info(
            "anchored_tool_bootstrap: first-request context stripped "
            "(sources=%s, %d -> %d chars)",
            suppressed,
            len(system_prompt),
            len(result),
        )
    return result


# User-message injections Hermes appends to the CURRENT turn (the recalled
# memory context block). They are runtime-context contributions, not user
# words — upstream's context-gate blanks the whole family during bootstrap.
_MEMORY_CONTEXT_TAIL_RE = re.compile(
    r"\n\n?<memory-context>.*?</memory-context>\s*$",
    re.DOTALL,
)


def strip_bootstrap_user_injections(
    agent: Any,
    api_messages: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Strip runtime-injected tails from user messages during bootstrap.

    The recalled-memory ``<memory-context>`` block rides on the last user
    message. It is Hermes runtime context, not user words — and it is the
    remaining lever that pulls the first-round trajectory back to the
    standard-like ``The user …`` style even with zero tools and the minimal
    system prompt (verified by live A/B: clean user message → "We need" 2/2,
    memory tail → "The user" 2/2). Gated by ``memory-context`` in
    ``suppressed_context_sources``; request-local, restored on promotion.
    """
    cfg = _resolve_config(agent)
    if cfg is None:
        return api_messages
    if not _is_bootstrap_phase(agent, cfg):
        return api_messages
    suppressed = cfg.get("suppressed_context_sources")
    if suppressed is None:
        suppressed = list(DEFAULT_SUPPRESSED_SOURCES)
    if "memory-context" not in suppressed:
        return api_messages
    if not isinstance(api_messages, list) or not api_messages:
        return api_messages

    changed = False
    out: List[Dict[str, Any]] = []
    for msg in api_messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        content = msg.get("content")
        if (
            msg.get("role") == "user"
            and isinstance(content, str)
            and "<memory-context>" in content
        ):
            stripped, n = _MEMORY_CONTEXT_TAIL_RE.subn("", content)
            if n and stripped != content:
                out.append({**msg, "content": stripped})
                changed = True
                continue
        out.append(msg)
    if changed:
        logger.info(
            "anchored_tool_bootstrap: first-request user-message runtime "
            "injections stripped (source=memory-context)",
        )
    return out


def bootstrap_max_output_tokens(agent: Any) -> Optional[int]:
    """Return the first-request output budget cap during bootstrap, else None.

    Mirrors the upstream ``agent/request`` cap listener. Default
    ``bootstrap_max_tokens`` is 0 (disabled): upstream issue #11 showed the cap
    is not the deciding variable (the first-request tool schema is), and a real
    1024 cap truncates the first-round reasoning on Hermes. Returns None once
    promoted so the reduced cap never leaks into later requests.
    """
    cfg = _resolve_config(agent)
    if cfg is None:
        return None
    if not _is_bootstrap_phase(agent, cfg):
        return None
    cap = cfg.get("bootstrap_max_tokens")
    if cap is None:
        cap = DEFAULT_BOOTSTRAP_MAX_TOKENS
    if not isinstance(cap, int) or cap <= 0:
        return None
    return cap
