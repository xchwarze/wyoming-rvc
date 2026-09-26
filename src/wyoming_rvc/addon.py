"""Home Assistant add-on glue: add-on options -> env vars, and Supervisor discovery.

Both are no-ops outside the add-on (no /data/options.json, no SUPERVISOR_TOKEN).
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.request
from collections.abc import Callable, MutableMapping
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

OPTIONS_PATH = Path("/data/options.json")
DISCOVERY_URL = "http://supervisor/discovery"


def apply_options(environ: MutableMapping[str, str], path: Path = OPTIONS_PATH) -> bool:
    """Expose add-on options as the env vars the service reads (explicit env wins)."""
    if not path.is_file():
        return False
    options = json.loads(path.read_text(encoding="utf-8"))
    # The image sets MODELS_DIR=/models (an anonymous volume the Supervisor drops on
    # every update); /data is the add-on's persistent, backed-up folder.
    environ["MODELS_DIR"] = "/data"
    for key, value in options.items():
        if value is None or value == "":
            continue
        text = str(value).lower() if isinstance(value, bool) else str(value)
        environ.setdefault(key.upper(), text)
    return True


def announce_wyoming(
    port: int,
    environ: MutableMapping[str, str],
    urlopen: Callable[..., object] = urllib.request.urlopen,
) -> bool:
    """Register the Wyoming service with the Supervisor so Home Assistant discovers it."""
    token = environ.get("SUPERVISOR_TOKEN")
    if not token:
        return False
    uri = f"tcp://{socket.gethostname()}:{port}"
    request = urllib.request.Request(
        DISCOVERY_URL,
        data=json.dumps({"service": "wyoming", "config": {"uri": uri}}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:  # type: ignore[attr-defined]
            ok = 200 <= response.status < 300
    except OSError as err:
        _LOGGER.warning("Supervisor discovery failed (add it manually: port %s): %s", port, err)
        return False
    _LOGGER.info("Supervisor discovery %s: %s", "sent" if ok else "rejected", uri)
    return ok
