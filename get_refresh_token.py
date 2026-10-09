#!/usr/bin/env python3
"""
Mint a Schwab refresh token for the forward test.

Schwab shows the client ID (app key) and client secret in the developer
portal. The refresh token is not listed there. This script opens the login
page, catches the localhost callback, and prints the refresh token.

The callback URL must match the one registered on the Schwab app, including
https and the port. The default is https://127.0.0.1:8080.

Usage:
    python3 get_refresh_token.py
    python3 get_refresh_token.py --callback https://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import base64
import getpass
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

AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
DEFAULT_CALLBACK = "https://127.0.0.1:8080"


class _Callback:
    def __init__(self) -> None:
        self.code: str | None = None
        self.error: str | None = None
        self.ready = threading.Event()


def _ask_credentials() -> tuple[str, str]:
    app_key = os.environ.get("SCHWAB_APP_KEY", "").strip() or input("Schwab client ID (app key): ").strip()
    app_secret = os.environ.get("SCHWAB_APP_SECRET", "").strip() or getpass.getpass("Schwab client secret: ")
    if not app_key or not app_secret:
        raise SystemExit("Client ID and client secret are both required.")
    return app_key, app_secret


def _certificate(directory: Path) -> tuple[Path, Path]:
    cert_file = directory / "cert.pem"
    key_file = directory / "key.pem"
    subprocess.run(
        [
            "openssl", "req", "-new", "-x509", "-nodes",
            "-keyout", str(key_file),
            "-out", str(cert_file),
            "-days", "1",
            "-subj", "/CN=127.0.0.1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return cert_file, key_file


def _serve(callback_url: str) -> tuple[HTTPServer, _Callback]:
    parsed = urlparse(callback_url)
    if parsed.scheme != "https" or not parsed.hostname or not parsed.port:
        raise SystemExit("Callback URL must look like https://127.0.0.1:8080")

    state = _Callback()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            params = parse_qs(urlparse(self.path).query)
            if "error" in params:
                state.error = params.get("error_description", params["error"])[0]
                self._reply(400, "Schwab authorization was declined. Return to the terminal.")
            elif "code" not in params:
                self._reply(400, "The callback did not include an authorization code.")
            else:
                state.code = unquote(params["code"][0])
                self._reply(200, "Login captured. You can close this tab and return to the terminal.")
            state.ready.set()

        def _reply(self, status: int, message: str) -> None:
            body = f"<html><body style='font-family:sans-serif;padding:2rem'><p>{message}</p></body></html>".encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    temp_dir = tempfile.TemporaryDirectory()
    cert_file, key_file = _certificate(Path(temp_dir.name))
    server = HTTPServer((parsed.hostname, parsed.port), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.temp_dir = temp_dir  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def _exchange(app_key: str, app_secret: str, code: str, callback_url: str) -> str:
    basic = base64.b64encode(f"{app_key}:{app_secret}".encode()).decode()
    response = requests.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": callback_url,
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise SystemExit(f"Schwab token request failed ({response.status_code}). Check the app key, secret, and callback URL.")
    refresh_token = response.json().get("refresh_token")
    if not refresh_token:
        raise SystemExit("Schwab did not return a refresh token.")
    return str(refresh_token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Get a Schwab refresh token for the forward test.")
    parser.add_argument("--callback", default=os.environ.get("SCHWAB_CALLBACK_URL", DEFAULT_CALLBACK))
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    app_key, app_secret = _ask_credentials()
    print(f"\nListening on {args.callback}")
    print("If the browser warns about the certificate, continue to the page. That warning is from the local login catcher.\n")
    server, state = _serve(args.callback)
    login_url = (
        f"{AUTH_URL}?response_type=code"
        f"&client_id={quote(app_key)}"
        f"&redirect_uri={quote(args.callback, safe='')}"
    )
    print(login_url)
    webbrowser.open(login_url)
    try:
        if not state.ready.wait(args.timeout):
            raise SystemExit("No login callback arrived. Confirm the Schwab app callback URL matches exactly.")
        if state.error or not state.code:
            raise SystemExit(state.error or "Authorization code was missing.")
        refresh_token = _exchange(app_key, app_secret, state.code, args.callback)
    finally:
        server.shutdown()

    print("\nPut these four lines in /home/ubuntu/spy-forward/.env")
    print("The Discord line is created in Discord: Server Settings, Integrations, Webhooks, New Webhook, Copy Webhook URL.\n")
    print(f"SCHWAB_APP_KEY={app_key}")
    print(f"SCHWAB_APP_SECRET={app_secret}")
    print(f"SCHWAB_REFRESH_TOKEN={refresh_token}")
    print("DISCORD_WEBHOOK_URL=")
    print("\nSchwab refresh tokens last about 7 days. Run this script again when the server reports an auth failure.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or "openssl could not create the local certificate.", file=sys.stderr)
        sys.exit(1)
