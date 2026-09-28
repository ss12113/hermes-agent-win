"""Regression tests for bounded context-compression summary prompts."""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX


def _compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=272_000,
    ):
        return ContextCompressor(model="test-model", quiet_mode=True)


def test_summary_input_bound_is_identity_for_small_content():
    content = "hello world\n\n[USER]: keep this byte-identical"
    assert ContextCompressor._bound_summary_input(content) is content


def test_summary_input_bound_preserves_edges_and_marks_omission():
    cap = ContextCompressor._SUMMARY_INPUT_MAX_CHARS
    content = "HEAD_EDGE " + ("m" * (cap * 3)) + " TAIL_EDGE"

    bounded = ContextCompressor._bound_summary_input(content)

    assert len(bounded) <= cap
    assert bounded.startswith("HEAD_EDGE")
    assert bounded.endswith("TAIL_EDGE")
    assert "summary input truncated" in bounded


def test_generate_summary_caps_aggregate_serialized_turns():
    compressor = _compressor()
    messages = [
        {"role": "user", "content": f"turn-{i}-" + ("x" * 6_000)}
        for i in range(80)
    ]
    messages[0]["content"] = "FIRST_SENTINEL " + messages[0]["content"]
    messages[-1]["content"] = "LAST_SENTINEL " + messages[-1]["content"]

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "bounded summary"

    with patch(
        "agent.context_compressor.call_llm",
        return_value=response,
    ) as call:
        summary = compressor._generate_summary(messages)

    prompt = call.call_args.kwargs["messages"][0]["content"]
    assert summary is not None
    assert summary.startswith(SUMMARY_PREFIX)
    assert len(prompt) < 180_000
    # Lean mode samples head+tail slices and marks the gaps with an elision
    # marker; the legacy "summary input truncated" wording is emitted only in
    # non-lean modes, so assert the marker the active mode actually produces.
    assert "chars elided" in prompt
    assert "recover via session_search" in prompt
    assert "FIRST_SENTINEL" in prompt
    assert "LAST_SENTINEL" in prompt
