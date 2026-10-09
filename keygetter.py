#!/usr/bin/env python3
"""
Obtain Schwab OAuth access and refresh tokens via the browser login flow.

Usage:
    export SCHWAB_API_KEY="your_app_key"
    export SCHWAB_API_SECRET="your_app_secret"
    python keygetter.py

Refresh an existing token:
    python keygetter.py --refresh "your_refresh_token"
"""

from __future__ import annotations

import argparse
import base64
import os
import ssl
import subprocess
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

import config
from auth_tokens import save_token_file

SCHWAB_AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
DEFAULT_TIMEOUT_SECONDS = 300.0

class _CallbackState:
    def __init__(self) -> None:
        self.code: str | None = None
        self.error: str | None = None
        self.event = threading.Event()


def _require_credentials() -> tuple[str, str]:
    client_id = os.environ.get("SCHWAB_API_KEY", "").strip()
    client_secret = os.environ.get("SCHWAB_API_SECRET", "").strip()
    if not client_id or not client_secret:
        raise SystemExit(
            "Set SCHWAB_API_KEY and SCHWAB_API_SECRET in the environment, then run this script again."
        )
    return client_id, client_secret


def _build_authorize_url(client_id: str, redirect_uri: str) -> str:
    return (
        f"{SCHWAB_AUTH_URL}?response_type=code"
        f"&client_id={quote(client_id)}"
        f"&redirect_uri={quote(redirect_uri, safe='')}"
    )


def _extract_code_from_url(redirect_url: str) -> str:
    parsed = urlparse(redirect_url.strip())
    params = parse_qs(parsed.query)

    if "error" in params:
        description = params.get("error_description", params["error"])[0]
        raise RuntimeError(f"Authorization failed: {description}")

    if "code" not in params:
        raise RuntimeError("No authorization code found in the provided URL.")

    return unquote(params["code"][0])


def _create_self_signed_cert(cert_dir: Path) -> tuple[Path, Path]:
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-new",
                "-x509",
                "-keyout",
                str(key_file),
                "-out",
                str(cert_file),
                "-days",
                "1",
                "-nodes",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "openssl is required to run the local HTTPS callback server. "
            "Install OpenSSL or use --paste-url with the full redirect URL."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to generate self-signed certificate: {exc.stderr}") from exc

    return cert_file, key_file


def _make_callback_handler(state: _CallbackState) -> type[BaseHTTPRequestHandler]:
    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            params = parse_qs(urlparse(self.path).query)

            if "error" in params:
                state.error = params.get("error_description", params["error"])[0]
                self._send_html(
                    400,
                    f"Authorization failed: {state.error}",
                )
                state.event.set()
                return

            if "code" not in params:
                self._send_html(400, "Missing authorization code in callback URL.")
                state.event.set()
                return

            state.code = unquote(params["code"][0])
            self._send_html(
                200,
                "Authorization successful. Return to the terminal.",
            )
            state.event.set()

        def _send_html(self, status: int, message: str) -> None:
            body = (
                "<html><body style='font-family:sans-serif;padding:2rem'>"
                f"<h2>{message}</h2>"
                "</body></html>"
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return CallbackHandler


def _start_callback_server(callback_url: str) -> tuple[HTTPServer, _CallbackState]:
    parsed = urlparse(callback_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    if parsed.scheme != "https":
        raise ValueError(
            f"SCHWAB_CALLBACK_URL must use https (got {callback_url!r})."
        )

    state = _CallbackState()
    handler = _make_callback_handler(state)

    temp_dir = tempfile.TemporaryDirectory()
    cert_file, key_file = _create_self_signed_cert(Path(temp_dir.name))
    httpd = HTTPServer((host, port), handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    httpd._temp_dir = temp_dir  # type: ignore[attr-defined]  # keep certs alive

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, state


def _exchange_code_for_tokens(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> dict:
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    response = requests.post(
        SCHWAB_TOKEN_URL,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(
            f"Token exchange failed ({response.status_code}): {response.text}"
        )
    return response.json()


def _refresh_tokens(
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> dict:
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    response = requests.post(
        SCHWAB_TOKEN_URL,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(
            f"Token refresh failed ({response.status_code}): {response.text}"
        )
    return response.json()


def _print_tokens(token_payload: dict) -> None:
    access_token = token_payload.get("access_token", "")
    refresh_token = token_payload.get("refresh_token", "")
    expires_in = token_payload.get("expires_in", "?")

    print("\n" + "=" * 72)
    print("Schwab OAuth Tokens")
    print("=" * 72)
    print(f"\nAccess token (expires in {expires_in}s, ~30 minutes):\n")
    print(access_token)

    if refresh_token:
        print("\nRefresh token (~7 days):\n")
        print(refresh_token)

    print("\n" + "-" * 72)
    print("Shell export commands:\n")
    print(f'export SCHWAB_ACCESS_TOKEN="{access_token}"')
    if refresh_token:
        print(f'export SCHWAB_REFRESH_TOKEN="{refresh_token}"')
    print("=" * 72 + "\n")


def _save_tokens(token_payload: dict) -> None:
    path = save_token_file(token_payload)
    print(f"Tokens saved to {path}")
    print("main.py will load and auto-refresh from this file.\n")


def login_flow(timeout_seconds: float) -> dict:
    client_id, client_secret = _require_credentials()
    redirect_uri = config.SCHWAB_CALLBACK_URL

    print(f"Listening for OAuth callback at {redirect_uri}")
    httpd, state = _start_callback_server(redirect_uri)

    authorize_url = _build_authorize_url(client_id, redirect_uri)
    print("Opening Schwab login page in your browser...")
    print(f"\nIf it does not open automatically, visit:\n{authorize_url}\n")
    print(
        "Your browser may warn about the self-signed certificate on localhost. "
        "That is expected — proceed to complete login."
    )

    webbrowser.open(authorize_url)

    try:
        if not state.event.wait(timeout=timeout_seconds):
            raise TimeoutError(
                f"No callback received within {timeout_seconds:.0f} seconds."
            )

        if state.error:
            raise RuntimeError(f"Authorization error: {state.error}")
        if not state.code:
            raise RuntimeError("Authorization code was not captured.")

        print("Authorization code received. Exchanging for tokens...")
        return _exchange_code_for_tokens(
            client_id,
            client_secret,
            state.code,
            redirect_uri,
        )
    finally:
        httpd.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Request Schwab OAuth tokens and print them to the terminal.",
    )
    parser.add_argument(
        "--refresh",
        metavar="REFRESH_TOKEN",
        help="Use an existing refresh token instead of opening the browser flow.",
    )
    parser.add_argument(
        "--paste-url",
        metavar="REDIRECT_URL",
        help=(
            "Paste the full redirect URL from the browser address bar "
            "(useful if the local callback server cannot be reached)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Seconds to wait for browser callback (default: {DEFAULT_TIMEOUT_SECONDS:.0f}).",
    )
    args = parser.parse_args()

    client_id, client_secret = _require_credentials()

    if args.refresh:
        tokens = _refresh_tokens(client_id, client_secret, args.refresh)
    elif args.paste_url:
        code = _extract_code_from_url(args.paste_url)
        print("Authorization code parsed. Exchanging for tokens...")
        tokens = _exchange_code_for_tokens(
            client_id,
            client_secret,
            code,
            config.SCHWAB_CALLBACK_URL,
        )
    else:
        tokens = login_flow(timeout_seconds=args.timeout)

    _save_tokens(tokens)
    _print_tokens(tokens)


if __name__ == "__main__":
    try:
        main()
    except (TimeoutError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
