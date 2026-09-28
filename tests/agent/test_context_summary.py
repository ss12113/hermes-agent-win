from types import SimpleNamespace

from agent.context_summary import format_context_usage_summary


def _agent(**kwargs):
    defaults = {
        "provider": "custom",
        "model": "gpt-test",
        "session_total_tokens": 0,
        "_session_messages": [{"role": "user", "content": "hello"}],
        "_cached_system_prompt": "system prompt",
        "tools": [{"function": {"name": "terminal", "description": "run shell commands"}}],
        "context_compressor": SimpleNamespace(
            last_prompt_tokens=0,
            context_length=200_000,
            compression_count=2,
        ),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_format_context_usage_summary_renders_markdown_table():
    text = format_context_usage_summary(_agent())

    assert text.startswith("## Context Usage")
    assert "**Model:** custom/gpt-test" in text
    assert "**Tokens:** ~" in text
    assert "**Source:** rough estimate from live request payload" in text
    assert "### Estimated usage by category" in text
    assert "| System prompt |" in text
    assert "| Conversation messages |" in text
    assert "| Tool schemas |" in text
    assert "| Free space |" in text
    assert "**Compressions:** 2" in text


def test_format_context_usage_summary_uses_reported_prompt_tokens_without_tilde():
    agent = _agent(
        context_compressor=SimpleNamespace(
            last_prompt_tokens=42_000,
            context_length=200_000,
            compression_count=0,
        )
    )

    text = format_context_usage_summary(agent)

    assert "**Tokens:** 42,000 / 200,000 (21.0%)" in text
    assert "**Source:** provider-reported last prompt" in text


def test_format_context_usage_summary_reports_pending_compaction_verdict():
    agent = _agent(
        context_compressor=SimpleNamespace(
            last_prompt_tokens=-1,
            context_length=200_000,
            compression_count=1,
            threshold_tokens=100_000,
            last_compression_rough_tokens=120_000,
            _last_compaction_verdict="pending",
            _last_compaction_prompt_tokens=0,
            _last_compaction_threshold_tokens=100_000,
            _ineffective_compression_count=0,
        )
    )

    text = format_context_usage_summary(agent)

    assert "### Last compaction" in text
    assert "**Verdict:** pending provider verification" in text
    assert "**Rough request after rewrite:** ~120,000" in text
    assert "**Threshold at verdict:** 100,000" in text
    assert "**Next-turn risk:** estimate remains at/above threshold; provider result decides" in text


def test_context_summary_does_not_label_native_precompact_estimate_as_postcompact():
    agent = _agent(
        context_compressor=SimpleNamespace(
            last_prompt_tokens=-1,
            context_length=200_000,
            compression_count=1,
            threshold_tokens=100_000,
            last_compression_rough_tokens=120_000,
            _last_compaction_verdict="pending",
            _last_compaction_prompt_tokens=0,
            _last_compaction_threshold_tokens=100_000,
            _last_compaction_rough_is_post_compaction=False,
            _ineffective_compression_count=0,
        )
    )

    text = format_context_usage_summary(agent)

    assert "~120,000" not in text
    assert "**Rough request after rewrite:** unavailable for native runtime" in text
    assert "**Next-turn risk:** native runtime estimate unavailable; provider result decides" in text


def test_context_summary_surfaces_open_summary_failure_breaker():
    agent = _agent()
    agent.context_compressor._consecutive_auto_summary_failures = 3

    output = format_context_usage_summary(agent)

    assert "### Summary route health" in output
    assert "Consecutive automatic summary failures:** 3" in output
    assert "paused until a successful manual `/compress` or `/new`" in output


def test_format_context_usage_summary_reports_cleared_compaction_verdict():
    agent = _agent(
        context_compressor=SimpleNamespace(
            last_prompt_tokens=42_000,
            context_length=200_000,
            compression_count=1,
            threshold_tokens=100_000,
            last_compression_rough_tokens=47_000,
            _last_compaction_verdict="cleared",
            _last_compaction_prompt_tokens=42_000,
            _last_compaction_threshold_tokens=100_000,
            _ineffective_compression_count=0,
        )
    )

    text = format_context_usage_summary(agent)

    assert "**Verdict:** cleared the auto-compaction threshold" in text
    assert "**Provider prompt at verdict:** 42,000" in text
    assert "**Next-turn risk:** automatic compaction not expected" in text


def test_format_context_usage_summary_reports_paused_compaction_verdict():
    agent = _agent(
        context_compressor=SimpleNamespace(
            last_prompt_tokens=120_000,
            context_length=200_000,
            compression_count=2,
            threshold_tokens=100_000,
            last_compression_rough_tokens=125_000,
            _last_compaction_verdict="paused",
            _last_compaction_prompt_tokens=120_000,
            _last_compaction_threshold_tokens=100_000,
            _last_compaction_ineffective_count=2,
            _ineffective_compression_count=0,
        )
    )

    text = format_context_usage_summary(agent)

    assert "**Verdict:** still above threshold; auto-compaction paused" in text
    assert "**Provider prompt at verdict:** 120,000" in text
    assert "**Next-turn risk:** paused after 2 ineffective compactions" in text
