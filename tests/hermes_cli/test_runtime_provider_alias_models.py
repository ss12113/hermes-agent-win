"""Alias-model runtime routing (4399-style aliases resolve via their named custom provider).

See the alias-routing ladder rung in ``hermes_cli.runtime_provider``.
"""

import hermes_cli.runtime_provider as rp


def test_4399_models_force_chat_completions_and_preserve_named_provider(monkeypatch):
    calls = []

    def fake_get_named(provider):
        if provider == "custom:sub2api-example":
            return {"name": "sub2api.example"}
        return None

    def fake_resolve(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "custom",
            "api_mode": "codex_responses",
            "base_url": "https://sub2api.example/v1",
            "api_key": "test-key",
            "source": "custom-provider:sub2api.example",
            "extra_headers": {"X-Test": "kept"},
            "request_overrides": {"extra_body": {"reasoning": {"effort": "max"}}},
        }

    monkeypatch.setattr(rp, "_get_named_custom_provider", fake_get_named)
    monkeypatch.setattr(
        rp,
        "find_custom_provider_identity_by_model",
        lambda model: "custom:sub2api-example",
    )
    monkeypatch.setattr(rp, "_resolve_named_custom_runtime", fake_resolve)

    resolved = rp.resolve_runtime_provider(
        requested="custom",
        target_model="4399-gpt-5-6-thinking",
        explicit_api_key="test-key",
    )

    assert calls == [{
        "requested_provider": "custom:sub2api-example",
        "explicit_api_key": "test-key",
        "explicit_base_url": None,
    }]
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://sub2api.example/v1"
    assert resolved["model"] == "4399-gpt-5-6-thinking"
    assert resolved["requested_provider"] == "custom:sub2api-example"
    assert resolved["source"] == "custom-provider:sub2api.example"
    assert resolved["extra_headers"] == {"X-Test": "kept"}
    assert resolved["request_overrides"]["extra_body"]["reasoning"]["effort"] == "max"


def test_4399_named_provider_keeps_explicit_endpoint_override(monkeypatch):
    monkeypatch.setattr(
        rp,
        "_get_named_custom_provider",
        lambda provider: {"name": "sub2api"} if provider == "custom:sub2api" else None,
    )
    monkeypatch.setattr(
        rp,
        "_resolve_named_custom_runtime",
        lambda **kwargs: {
            "provider": "custom",
            "api_mode": "codex_responses",
            "base_url": kwargs["explicit_base_url"],
            "api_key": "test-key",
            "source": "custom-provider:sub2api",
        },
    )

    resolved = rp.resolve_runtime_provider(
        requested="custom:sub2api",
        target_model="4399-GPT-6-Astra",
        explicit_base_url="http://127.0.0.1:8080/v1",
        explicit_api_key="test-key",
    )
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "http://127.0.0.1:8080/v1"
    assert resolved["requested_provider"] == "custom:sub2api"
    assert resolved["model"] == "4399-GPT-6-Astra"
