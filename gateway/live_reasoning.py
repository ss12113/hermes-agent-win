"""Live chain-of-thought (CoT) display for chat platforms.

Mirrors the CLI reasoning box on gateway platforms: when ``display``
``show_reasoning`` is enabled AND token streaming is active, model
reasoning deltas stream into a dedicated progressively-edited message
while the agent thinks, and the box finalizes right before the visible
answer starts streaming.

The non-streaming path is untouched: the post-run prepend in
``gateway/run.py`` (``_handle_message_with_agent``) still renders the
``last_reasoning`` block and is skipped when a live box was displayed
(``reasoning_displayed_live`` on the agent result).

Reasoning deltas are fed from the agent worker thread; the consumer's
queue API is thread-safe, and a lock guards this feed's own state.
"""

from __future__ import annotations

import threading
from typing import Any

_HEADER = "💭 **Reasoning:**\n```\n"
_FOOTER = "\n```"
_TRUNCATION_MARKER = "\n… (truncated)"
DEFAULT_BUDGET = 12000


class LiveReasoningDisplay:
    """Thread-safe feed of reasoning deltas into a stream consumer.

    The consumer is a ``GatewayStreamConsumer`` configured with a slower
    edit cadence, no cursor, and no fresh-final re-delivery (the caller
    sets those up before constructing this display).
    """

    def __init__(
        self,
        consumer: Any,
        *,
        budget: int = DEFAULT_BUDGET,
        header: str = _HEADER,
    ) -> None:
        self._consumer = consumer
        self._budget = int(budget)
        self._header = header
        self._lock = threading.Lock()
        self._closed = False
        self._started = False
        # Trailing run of 1-2 backticks held back: a fence (```) may be split across deltas.
        self._tick_tail = ""

    def on_delta(self, text: str) -> None:
        """Queue a reasoning delta. Called from the agent worker thread."""
        if self._consumer is None or not text:
            return
        with self._lock:
            if self._closed:
                return
            if not self._started:
                self._started = True
                self._consumer.on_delta(self._header)
            if self._budget > 0:
                take = min(self._budget, len(text))
                self._budget -= len(text)
                if take:
                    visible = self._escape_fence_runs(text[:take])
                    if visible:
                        self._consumer.on_delta(visible)

    def _escape_fence_runs(self, text: str) -> str:
        """Escape runs of 3+ backticks so a fence quoted in the reasoning can't close the box.

        The box body is wrapped in an outer ``` fence; an unescaped inner fence would end the
        block mid-thought and dump the rest of the reasoning outside it as prose. Runs become
        ``\\`\\`\\``` (the same convention as the non-live reasoning prepend); a trailing run of
        1-2 backticks is held back until a later delta decides whether it grows into a fence.
        """
        buf = self._tick_tail + text
        self._tick_tail = ""
        out: list = []
        i, n = 0, len(buf)
        while i < n:
            if buf[i] != "`":
                out.append(buf[i])
                i += 1
                continue
            j = i
            while j < n and buf[j] == "`":
                j += 1
            run_len = j - i
            if run_len >= 3:
                out.append("\\`" * run_len)
            elif j == n:
                self._tick_tail = buf[i:j]  # may grow into a fence with the next delta
            else:
                out.append(buf[i:j])
            i = j
        return "".join(out)

    def close(self) -> None:
        """Finalize the reasoning message. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            started = self._started
            if self._consumer is None:
                return
            if started:
                if self._tick_tail:
                    # Short tail run is fence-safe on its own; flush before the footer.
                    self._consumer.on_delta(self._tick_tail)
                    self._tick_tail = ""
                if self._budget <= 0:
                    self._consumer.on_delta(_TRUNCATION_MARKER)
                self._consumer.on_delta(_FOOTER)
            self._consumer.finish()

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def delivered(self) -> bool:
        """True when the reasoning message reached the platform."""
        consumer = self._consumer
        if consumer is None:
            return False
        return bool(
            getattr(consumer, "final_content_delivered", False)
            or getattr(consumer, "message_id", None)
            or getattr(consumer, "_already_sent", False)
        )
