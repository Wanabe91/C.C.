from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import httpx


def _is_loopback_host(host: str) -> bool:
    normalized = (host or "").strip().lower()
    if not normalized:
        return False
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _trust_env_for_url(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return not _is_loopback_host(host)


def post(url: str, **kwargs) -> httpx.Response:
    kwargs.setdefault("trust_env", _trust_env_for_url(url))
    return httpx.post(url, **kwargs)
