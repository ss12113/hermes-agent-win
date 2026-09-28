"""Tavily web search + content extraction (``/search``, ``/extract``; sync httpx).

Env: ``TAVILY_API_KEY`` (https://app.tavily.com/home, optional), ``TAVILY_BASE_URL``.
Keyed requests use ``Authorization``; without a key the request is keyless
(``X-Tavily-Access-Mode: keyless``). Tavily is NOT in the zero-config
keyless ring — keyless access is opt-in by selecting Tavily in ``hermes tools``.

``TAVILY_API_KEY`` may contain a comma-separated pool. Keyed requests start at
a round-robin key and rotate on Tavily rate-limit/auth/billing responses
(432/429/401/402) and 5xx responses. The pool is read dotenv-first on every
request so an edit to ``~/.hermes/.env`` can replace stale inherited keys
without restarting the process.
"""

from __future__ import annotations

import itertools
import logging
import threading
from typing import Any, Dict, List, Optional

import httpx

from plugins.web._common import (
    SEARCH_LIMIT_CAP, BaseWebSearchProvider, document, extract_fail, http_status_detail, provider_env, run_extract,
    run_search, search_fail, search_ok, setup_schema, title_hit, use_keyless,
)

logger = logging.getLogger(__name__)

_CLIENT_NAME = "hermes-agent"

_SEARCH_PAYLOAD = {"include_raw_content": False, "include_images": False}

_KEY_LOCK = threading.Lock()
_KEY_COUNTER = itertools.count()

# Statuses for which trying another credential is useful.  5xx is handled as
# a class below because Tavily can return several different server errors.
_RETRYABLE_STATUSES = frozenset({401, 402, 429, 432})


def _get_tavily_api_keys() -> List[str]:
    """Return the active Tavily key pool, preferring Hermes' dotenv layer.

    ``get_provider_env`` is retained as the fallback for stripped installs and
    test doubles where the config module is unavailable.  A deliberate value
    in ``~/.hermes/.env`` wins over a stale exported value, matching the
    carried provider's rotation contract.
    """
    from agent.web_search_provider import get_provider_env

    raw = get_provider_env("TAVILY_API_KEY")
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv

        dotenv_value = get_env_value_prefer_dotenv("TAVILY_API_KEY")
        if dotenv_value:
            raw = dotenv_value
    except Exception:  # noqa: BLE001 - config layer is optional here
        pass

    # Accept semicolons as a small compatibility courtesy for older local
    # configs, while the documented form remains comma-separated.
    values = [part.strip() for part in (raw or "").replace(";", ",").split(",")]
    return [value for value in values if value]


def _tavily_headers(api_key: str) -> Dict[str, str]:
    headers = {"X-Client-Name": _CLIENT_NAME}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["X-Tavily-Access-Mode"] = "keyless"
    return headers


def _tavily_request(
    endpoint: str,
    payload: Dict[str, Any],
    keys: Optional[List[str]] = None,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """POST to Tavily and return parsed JSON.

    ``api_key=None`` resolves the live dotenv-first pool; ``api_key=""``
    forces the upstream keyless header. ``keys`` is a test/embedding override
    and bypasses environment resolution.  A caller supplying one explicit
    non-empty ``api_key`` gets one keyed attempt, while normal provider calls
    pass ``None`` and therefore retain pool rotation.

    Final non-2xx responses remain ``ValueError`` with Tavily's response body,
    preserving the upstream provider error contract.
    """
    if keys:
        api_keys = [str(key).strip() for key in keys if str(key).strip()]
        force_keyless = False
    elif api_key == "":
        api_keys = []
        force_keyless = True
    elif api_key is not None:
        api_keys = [api_key]
        force_keyless = False
    else:
        api_keys = _get_tavily_api_keys()
        force_keyless = False

    base_url = provider_env("TAVILY_BASE_URL") or "https://api.tavily.com"
    url = f"{base_url}/{endpoint.lstrip('/')}"

    if force_keyless:
        order: List[Optional[str]] = [""]
    else:
        if not api_keys:
            # Keep the low-level helper useful on its own and preserve the
            # previous keyless behavior when the caller explicitly asks for
            # it through api_key="" only.
            order = [None]
        else:
            with _KEY_LOCK:
                start = next(_KEY_COUNTER) % len(api_keys)
            order = [*api_keys[start:], *api_keys[:start]]

    for index, selected_key in enumerate(order):
        last_key = index == len(order) - 1
        # ``None`` means the low-level legacy fallback; an empty string is the
        # deliberate keyless header value.
        request_key = selected_key if selected_key is not None else ""
        label = f"key-{index + 1}" if request_key else "keyless"
        logger.info(
            "Tavily %s request to %s (%s, %d/%d)",
            endpoint, url, label, index + 1, len(order),
        )
        try:
            response = httpx.post(url, json=payload, timeout=60, headers=_tavily_headers(request_key))
        except Exception as exc:  # noqa: BLE001 - rotate on transient transport failures too
            if not last_key and not force_keyless:
                logger.warning(
                    "Tavily %s failed on %s (%s), rotating to next key",
                    endpoint, label, exc,
                )
                continue
            raise

        if response.status_code >= 400:
            status = response.status_code
            if (not last_key and not force_keyless
                    and (status in _RETRYABLE_STATUSES or status >= 500)):
                logger.warning(
                    "Tavily %s failed with %s on %s, rotating to next key",
                    endpoint, status, label,
                )
                continue
            raise ValueError(http_status_detail(response))

        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001 - a bad response may be key-specific
            if not last_key and not force_keyless:
                logger.warning(
                    "Tavily %s returned invalid JSON on %s (%s), rotating to next key",
                    endpoint, label, exc,
                )
                continue
            raise

    # ``order`` is non-empty by construction; this protects future edits.
    raise RuntimeError("Tavily request failed: no keys left to try")


def _normalize_tavily_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    return search_ok([
        title_hit(r.get("title", ""), r.get("url", ""), r.get("content", ""), i + 1)
        for i, r in enumerate(response.get("results", []))
    ])


def _normalize_tavily_documents(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map ``/extract`` to documents without attributing missing URLs to a request."""
    documents = [
        document(r.get("url", ""), r.get("title", ""), r.get("raw_content", "") or r.get("content", ""))
        for r in response.get("results", [])
    ]
    documents += [_failed_document(f.get("url", ""), f.get("error", "extraction failed")) for f in response.get("failed_results", [])]
    documents += [_failed_document(str(u), "extraction failed") for u in response.get("failed_urls", [])]
    return documents


def _failed_document(url: str, error: str) -> Dict[str, Any]:
    return {"url": url, "title": "", "content": "", "raw_content": "", "error": error, "metadata": {"sourceURL": url}}


def _missing_key_error(action: str) -> str:
    return f"TAVILY_API_KEY is not set. Get a key at https://app.tavily.com/home or select Tavily in `hermes tools` for opt-in keyless {action}."


def _auth(action: str) -> tuple[Optional[str], Optional[str], str]:
    """Return ``(request_key, missing_error, log_prefix)`` for a provider call.

    Keyed calls return ``None`` deliberately: ``_tavily_request`` then reads
    the current dotenv-first pool and can rotate it.  ``""`` remains the
    explicit keyless sentinel used by the upstream tiering contract.
    """
    api_keys = _get_tavily_api_keys()
    force_keyless = use_keyless("tavily", api_keys[0] if api_keys else "")
    if not force_keyless and not api_keys:
        return None, _missing_key_error(action), ""
    return "" if force_keyless else None, None, "keyless " if force_keyless else ""


class TavilyWebSearchProvider(BaseWebSearchProvider):
    """Tavily search + extract provider (keyed, or opt-in keyless)."""

    NAME = "tavily"
    DISPLAY_NAME = "Tavily"
    KEY_ENV = "TAVILY_API_KEY"
    EXTRACT = True
    KEYLESS = True

    def is_available(self) -> bool:
        """Treat any non-empty key in the dotenv-first pool as availability."""
        return bool(_get_tavily_api_keys())

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        def _body() -> Dict[str, Any]:
            key, missing, prefix = _auth("search")
            if missing:
                return search_fail(missing)
            logger.info("Tavily %ssearch: '%s' (limit=%d)", prefix, query, limit)
            payload = {"query": query, "max_results": min(limit, SEARCH_LIMIT_CAP), **_SEARCH_PAYLOAD}
            return _normalize_tavily_search_results(_tavily_request("search", payload, api_key=key))

        return run_search("Tavily", logger, _body)

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        def _body() -> List[Dict[str, Any]]:
            key, missing, prefix = _auth("extract")
            if missing:
                return extract_fail(urls, missing)
            logger.info("Tavily %sextract: %d URL(s)", prefix, len(urls))
            raw = _tavily_request("extract", {"urls": urls, "include_images": False}, api_key=key)
            return _normalize_tavily_documents(raw)

        return run_extract("Tavily", logger, urls, _body)

    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "Tavily", "free · key optional", "Search + extract. Opt-in keyless; set TAVILY_API_KEY for higher limits.",
            "TAVILY_API_KEY", "Tavily API key (optional — keyless works when Tavily is selected)", "https://app.tavily.com/home",
        )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'WebSearchProvider': ('agent.web_search_provider', 'WebSearchProvider'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
