"""The loopback allowlist is the offline guarantee — assert it actually holds.

CLAUDE.md calls conftest's socket guard "the whole guarantee": a test that
forgets to inject a fake falls back to the real Voyage/Anthropic path and must
*fail* rather than quietly call a paid API. The guard was relaxed from a blanket
``socket.socket`` block to a loopback-only allowlist (so the local Atlas
container stays reachable), which is a more selective and more fragile mechanism.
These tests pin that it still rejects a non-loopback connection while permitting
loopback — without them, a hole in the allowlist would read as green.
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest


def test_create_connection_to_a_non_loopback_host_is_blocked():
    # The httpcore/httpx path (Voyage, Anthropic) reaches the network here.
    with pytest.raises(RuntimeError, match="non-loopback socket"):
        socket.create_connection(("api.voyageai.com", 443), timeout=1)


def test_socket_connect_to_a_non_loopback_host_is_blocked():
    # The other path: a bare socket the allowlist patches at the method level.
    sock = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="non-loopback socket"):
            sock.connect(("api.anthropic.com", 443))
    finally:
        sock.close()


def test_loopback_is_still_allowed(atlas_uri):
    # The container URI is loopback; connecting to it must succeed, proving the
    # allowlist permits 127.0.0.1/localhost — otherwise the whole suite (every
    # store-touching test) could not run at all.
    parsed = urlparse(atlas_uri)
    sock = socket.create_connection((parsed.hostname, parsed.port), timeout=5)
    sock.close()
