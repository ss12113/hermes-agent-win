"""Provider error codes embedded in an HTTP-200 body map to the right failure reason —
a rate-limit code must not surface as a slow/timeout classification (carried parity:
the old inline helper is now agent.turn_failure_copy.invalid_response_failure_reason)."""

from types import SimpleNamespace

from agent.error_classifier import FailoverReason
from agent.turn_failure_copy import invalid_response_failure_reason


def test_rate_limit_code_is_not_reported_as_timeout():
    resp = SimpleNamespace(error=SimpleNamespace(code=429))

    reason = invalid_response_failure_reason(resp)

    assert reason == FailoverReason.rate_limit.value
    assert "timeout" not in reason


def test_timeout_code_maps_to_timeout():
    resp = SimpleNamespace(error={"code": 504})

    assert invalid_response_failure_reason(resp) == FailoverReason.timeout.value


def test_unknown_or_missing_code_falls_back_to_invalid_response():
    assert invalid_response_failure_reason(SimpleNamespace(error=None)) == "invalid_response"
    assert invalid_response_failure_reason(SimpleNamespace(error=SimpleNamespace(code="weird"))) == "invalid_response"
