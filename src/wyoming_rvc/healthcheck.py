"""Container healthcheck: ``python -m wyoming_rvc.healthcheck`` exits 0 when ready.

Uses HTTP ``/readyz``, or a Wyoming ``describe`` round trip when the HTTP API is
disabled (the Wyoming server only starts listening after models are warmed up).
Standard library only, so it stays cheap to run every 30 s.
"""

from __future__ import annotations

import os
import socket
import sys
import urllib.request


def _enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return default if not value else value not in ("0", "false", "no", "off")


def main() -> int:
    if _enabled("HTTP_ENABLED"):
        port = os.environ.get("HTTP_PORT") or "8080"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=4) as response:
                return 0 if response.status == 200 else 1
        except OSError:
            return 1
    port = int(os.environ.get("WYOMING_PORT") or 10200)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=4) as sock:
            sock.sendall(b'{"type": "describe"}\n')
            return 0 if sock.recv(1) else 1
    except OSError:
        return 1


if __name__ == "__main__":
    sys.exit(main())
