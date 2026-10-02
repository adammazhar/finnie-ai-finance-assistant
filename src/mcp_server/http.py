"""Streamable HTTP transport on localhost, protected by a static bearer token.

Every HTTP request must carry ``Authorization: Bearer <MCP_API_TOKEN>`` (from ``.env``);
anything else gets ``401`` with ``WWW-Authenticate: Bearer`` before it reaches the MCP
server. The token is compared in constant time and never logged.

Defense in depth, from the MCP specification's transport security rules:
- binds to 127.0.0.1 only (config ``mcp.host``);
- the SDK's DNS-rebinding protection rejects a ``Host`` that isn't localhost (421) and an
  ``Origin`` that isn't a localhost page (403), so a web page in the user's browser can't
  reach the server even if it guessed the token;
- the server refuses to start without a token of at least ``mcp.min_token_length``
  characters.

A static token suits one person on one machine. The production path for a shared,
remote server is OAuth 2.1 with Auth0 (docs/DESIGN.md §9.4), not built.
"""

from __future__ import annotations

import hmac
import json
import logging
from typing import Any

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

from src.core.config import MCPConfig, Settings

logger = logging.getLogger(__name__)

LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


class TokenError(RuntimeError):
    """MCP_API_TOKEN is missing or too short, so the HTTP transport won't start."""


def api_token(settings: Settings) -> str:
    secret = settings.mcp_api_token.get_secret_value().strip() if settings.mcp_api_token else ""
    if not secret:
        raise TokenError(
            "MCP_API_TOKEN isn't set. Add it to .env; generate one with:\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    if len(secret) < settings.mcp.min_token_length:
        raise TokenError(
            f"MCP_API_TOKEN is too short ({len(secret)} characters); use at least "
            f"{settings.mcp.min_token_length}."
        )
    return secret


class BearerTokenMiddleware:
    """Pure ASGI middleware: 401 without the expected bearer token, 405 for anything
    but POST, otherwise the request goes on to the MCP app."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self._expected = token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan events pass through
            await self.app(scope, receive, send)
            return
        supplied = dict(scope.get("headers") or []).get(b"authorization", b"")
        scheme, _, credentials = supplied.partition(b" ")
        # the scheme name is case-insensitive (RFC 6750); the token is compared in constant time
        valid = scheme.lower() == b"bearer" and hmac.compare_digest(
            credentials.strip(), self._expected
        )
        if not valid:
            # RFC 6750 §3.1: no error code when the request had no credentials at all
            challenge = 'Bearer realm="finnie"' + (', error="invalid_token"' if supplied else "")
            error = "invalid_token" if supplied else "missing_token"
            logger.warning("Rejected MCP request: %s", error)
            await _respond(
                send,
                401,
                {
                    "error": error,
                    "error_description": "A valid bearer token (MCP_API_TOKEN) is required.",
                },
                [(b"www-authenticate", challenge.encode())],
            )
            return
        if scope["method"] != "POST":
            # Stateless server, no server-initiated messages: nothing to stream (GET) or
            # end (DELETE). The specification's answer is 405.
            await _respond(send, 405, {"error": "Only POST is supported."}, [(b"allow", b"POST")])
            return
        await self.app(scope, receive, send)


async def _respond(
    send: Send, status: int, payload: dict[str, str], headers: list[tuple[bytes, bytes]]
) -> None:
    body = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                *headers,
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def transport_security(config: MCPConfig) -> TransportSecuritySettings:
    hosts = sorted({config.host, *LOCAL_HOSTS})
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[f"{h}:*" for h in hosts],
        allowed_origins=[f"http://{h}:*" for h in hosts],
    )


def build_http_app(server: MCPServer, settings: Settings) -> Any:
    """The ASGI app: the SDK's Streamable HTTP app behind the bearer-token check.

    Stateless with JSON responses, which matches the 2026-07-28 specification and still
    serves clients that open with the older ``initialize`` handshake.
    """
    token = api_token(settings)
    app = server.streamable_http_app(
        streamable_http_path=settings.mcp.path,
        stateless_http=True,
        json_response=True,
        transport_security=transport_security(settings.mcp),
        host=settings.mcp.host,
    )
    return BearerTokenMiddleware(app, token)


def url(settings: Settings) -> str:
    return f"http://{settings.mcp.host}:{settings.mcp.port}{settings.mcp.path}"
