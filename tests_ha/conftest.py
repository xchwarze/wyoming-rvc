"""Home Assistant interoperability tests against a *running* wyoming-rvc server.

These use Home Assistant's own ``wyoming`` integration (config flow, TTS entity
and TTS manager), not a custom client. They need Python >= 3.14.2 and the
packages in ``tests_ha/requirements.txt``; see README "Home Assistant e2e test".
"""

from __future__ import annotations

import os

import pytest
import pytest_socket

HOST = os.environ.get("WYOMING_RVC_HOST", "127.0.0.1")
PORT = int(os.environ.get("WYOMING_RVC_PORT", "10200"))
OUT_DIR = os.environ.get("WYOMING_RVC_HA_OUT")


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    # pytest-homeassistant-custom-component only allows 127.0.0.1; also allow the server.
    pytest_socket.socket_allow_hosts([HOST, "127.0.0.1", "::1"], allow_unix_socket=True)


@pytest.fixture(autouse=True)
async def core_setup(hass, socket_enabled):
    """Like HA's own wyoming tests: the core component provides exposed_entities for conversation."""
    from homeassistant.setup import async_setup_component

    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def server() -> tuple[str, int]:
    return HOST, PORT


@pytest.fixture
def save_audio():
    def _save(name: str, data: bytes) -> None:
        if OUT_DIR:
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, name), "wb") as f:
                f.write(data)

    return _save
