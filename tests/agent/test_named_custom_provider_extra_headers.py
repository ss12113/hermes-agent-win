from types import SimpleNamespace

import agent.auxiliary_client as auxiliary_client
import hermes_cli.runtime_provider as runtime_provider


def test_named_custom_provider_applies_route_extra_headers(monkeypatch):
    """A named custom route must carry its headers into fallback/aux clients."""
    entry = {
        "name": "Api.longxiadev.store",
        "base_url": "https://api.longxiadev.store/v1",
        "api_key": "test-key",
        "api_mode": "codex_responses",
        "model": "gpt-5.6-sol",
        "extra_headers": {
            "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
            "originator": "codex_cli_rs",
        },
    }
    monkeypatch.setattr(
        runtime_provider,
        "_get_named_custom_provider",
        lambda provider: entry if provider == "custom:api.longxiadev.store" else None,
    )
    monkeypatch.setattr(auxiliary_client, "_apply_user_default_headers", lambda headers: headers)

    captured = {}
    created = {}

    def fake_create_openai_client(*, api_key, base_url, **kwargs):
        captured.update(kwargs)
        created["client"] = SimpleNamespace(api_key=api_key, base_url=base_url)
        return created["client"]

    monkeypatch.setattr(
        auxiliary_client,
        "_create_openai_client",
        fake_create_openai_client,
    )

    client, model = auxiliary_client.resolve_provider_client(
        "custom:api.longxiadev.store",
        model="gpt-5.6-sol",
        raw_codex=True,
    )

    assert client is created["client"]
    assert model == "gpt-5.6-sol"
    assert captured["default_headers"] == entry["extra_headers"]
