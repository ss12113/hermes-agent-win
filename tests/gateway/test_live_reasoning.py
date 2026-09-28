"""Tests for LiveReasoningDisplay — gateway live chain-of-thought box."""

from types import SimpleNamespace

import pytest

from gateway.live_reasoning import LiveReasoningDisplay


class _RecordingConsumer:
    """Minimal stand-in for GatewayStreamConsumer's queue API."""

    def __init__(self):
        self.items = []
        self.finished = 0
        self.final_content_delivered = False
        self.message_id = None
        self._already_sent = False

    def on_delta(self, text):
        self.items.append(text)

    def finish(self):
        self.finished += 1


def _display(**display_kwargs):
    consumer = _RecordingConsumer()
    return LiveReasoningDisplay(consumer, **display_kwargs), consumer


class TestDeltasAndClose:
    def test_first_delta_prepends_header(self):
        display, consumer = _display()
        display.on_delta("thinking")
        assert consumer.items[0].startswith("💭 **Reasoning:**")
        assert consumer.items[0].endswith("```\n")
        assert "thinking" in consumer.items[1]

    def test_no_header_without_deltas(self):
        display, consumer = _display()
        display.close()
        assert all("Reasoning" not in item for item in consumer.items)

    def test_close_appends_footer_and_finishes(self):
        display, consumer = _display()
        display.on_delta("a")
        display.close()
        assert consumer.items[-1] == "\n```"
        assert consumer.finished == 1
        assert display.closed

    def test_close_is_idempotent(self):
        display, consumer = _display()
        display.on_delta("a")
        display.close()
        display.close()
        assert consumer.finished == 1
        assert consumer.items.count("\n```") == 1

    def test_deltas_after_close_are_dropped(self):
        display, consumer = _display()
        display.on_delta("a")
        display.close()
        display.on_delta("b")
        assert "b" not in "".join(consumer.items)

    def test_budget_truncation_marker(self):
        display, consumer = _display(budget=4)
        display.on_delta("abcd")  # exactly consumes the budget
        display.on_delta("efgh")  # over budget — dropped
        display.close()
        joined = "".join(consumer.items)
        assert "abcd" in joined
        assert "efgh" not in joined
        assert "truncated" in joined

    def test_partial_delta_within_budget(self):
        display, consumer = _display(budget=4)
        display.on_delta("abcdef")
        display.close()
        joined = "".join(consumer.items)
        assert "abcd" in joined
        assert "abcdef" not in joined
        assert "truncated" in joined


class TestDelivered:
    def test_not_delivered_when_no_consumer(self):
        display = LiveReasoningDisplay(None)
        assert display.delivered() is False

    def test_delivered_via_final_content(self):
        display, consumer = _display()
        assert display.delivered() is False
        consumer.final_content_delivered = True
        assert display.delivered() is True

    def test_delivered_via_message_id(self):
        display, consumer = _display()
        consumer.message_id = "42"
        assert display.delivered() is True

    def test_delivered_via_already_sent(self):
        display, consumer = _display()
        consumer._already_sent = True
        assert display.delivered() is True

    def test_consumer_without_flags_is_not_delivered(self):
        display, consumer = _display()
        consumer = SimpleNamespace()  # no flags at all
        display._consumer = consumer
        assert display.delivered() is False


class TestFenceEscaping:
    """A ``` quoted in the reasoning must not close the box's outer fence.

    The box body is wrapped in ``` … ```. An unescaped inner fence ends the code
    block mid-thought and dumps the remaining reasoning outside it as ordinary
    prose — which reads exactly like the actual reply.
    """

    def test_inner_fence_escaped_keeps_single_block(self):
        display, consumer = _display()
        display.on_delta("quoted ```python\nx = 1\n``` backticks")
        display.close()
        joined = "".join(consumer.items)
        # Header open + footer close only — the inner fence was escaped.
        assert joined.count("```") == 2
        assert "\\`\\`\\`" in joined

    def test_fence_split_across_deltas_still_escaped(self):
        display, consumer = _display()
        display.on_delta("a `")
        display.on_delta("``")  # the run closes to a full fence across the boundary
        display.on_delta("b")
        display.close()
        joined = "".join(consumer.items)
        # Only the header open + footer close remain as raw fences.
        assert joined.count("```") == 2
        assert "\\`\\`\\`" in joined

    def test_short_backtick_tail_flushed_before_footer(self):
        display, consumer = _display()
        display.on_delta("done ``")
        display.close()
        joined = "".join(consumer.items)
        assert "done ``" in joined
        assert consumer.items[-1] == "\n```"
        assert joined.index("``", joined.index("done")) < joined.rindex("```")

    def test_regular_text_and_single_backticks_untouched(self):
        display, consumer = _display()
        display.on_delta("uses `code` spans")
        display.close()
        joined = "".join(consumer.items)
        assert "uses `code` spans" in joined
