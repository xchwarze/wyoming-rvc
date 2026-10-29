"""Idle sleep: unload the models after a quiet period and reload them on the next request.

``IdleSleeper`` knows nothing about models: it gets two async callbacks. Every request holds
``request()`` open and sleeping only happens with none in flight. One lock serializes both
transitions, so a request that arrives mid-sleep waits for it and then wakes the models once.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .metrics import now_ms

_LOGGER = logging.getLogger(__name__)


def release_memory() -> None:
    """Give freed memory back: Python garbage, the CUDA allocator cache and (glibc) the C heap."""
    gc.collect()
    with contextlib.suppress(Exception):  # torch is optional (unit tests run without it)
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with contextlib.suppress(Exception):  # not glibc (Windows, musl): nothing to trim
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)


class IdleSleeper:
    """Calls ``sleep`` after ``idle_s`` seconds without a request, and ``wake`` before the next one."""

    def __init__(
        self,
        idle_s: float,
        sleep: Callable[[], Awaitable[None]],
        wake: Callable[[], Awaitable[None]],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if idle_s <= 0:
            raise ValueError("idle_s must be > 0 (IDLE_SLEEP_S=0 means: do not create a sleeper)")
        self.idle_s = idle_s
        self._sleep_cb, self._wake_cb, self._clock = sleep, wake, clock
        self._lock = asyncio.Lock()
        self._active = 0
        self._last = clock()
        self._asleep = False
        self._task: asyncio.Task | None = None
        self.sleeps = self.wakes = 0
        self.last_wake_ms: float | None = None

    @property
    def asleep(self) -> bool:
        return self._asleep

    @property
    def active(self) -> int:
        return self._active

    def status(self) -> dict[str, Any]:
        return {
            "idle_sleep_s": self.idle_s,
            "asleep": self._asleep,
            "sleeps": self.sleeps,
            "wakes": self.wakes,
            "last_wake_ms": self.last_wake_ms,
        }

    # ------------------------------------------------------------------ requests

    @contextlib.asynccontextmanager
    async def request(self) -> AsyncIterator[None]:
        """Keep the models awake for one request; wakes them first if they are asleep."""
        async with self._lock:
            self._active += 1
            try:
                if self._asleep:
                    await self._wake()
            except BaseException:
                self._release()
                raise
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        self._active -= 1
        self._last = self._clock()  # the idle time starts when the last request ends

    async def _wake(self) -> None:
        # A cancelled request cannot stop the loading thread: let it finish, so the models
        # are not left half-loaded (the wake itself is idempotent either way).
        job = asyncio.ensure_future(self._wake_job())
        try:
            await asyncio.shield(job)
        finally:
            if not job.done():
                with contextlib.suppress(BaseException):
                    await job

    async def _wake_job(self) -> None:
        started = now_ms()
        _LOGGER.info("Waking up: reloading the models...")
        await self._wake_cb()
        self._asleep = False
        self.wakes += 1
        self.last_wake_ms = now_ms() - started
        _LOGGER.info("Models reloaded in %.0f ms", self.last_wake_ms)

    # ------------------------------------------------------------------ sleeping

    async def check(self) -> bool:
        """Go to sleep now if idle for long enough. Returns True when it did."""
        async with self._lock:
            if self._asleep or self._active or self._clock() - self._last < self.idle_s:
                return False
            started = now_ms()
            _LOGGER.info("No request for %.0f s: unloading the models", self.idle_s)
            self._asleep = True  # first: a failed or cancelled unload is repaired by the next wake
            self.sleeps += 1
            try:
                await self._sleep_cb()
            except Exception:
                _LOGGER.exception("Unloading the models failed; they are reloaded on the next request")
            else:
                _LOGGER.info("Models unloaded in %.0f ms", now_ms() - started)
            return True

    def start(self) -> None:
        self._last = self._clock()
        self._task = asyncio.create_task(self._monitor(), name="idle-sleep")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _monitor(self) -> None:
        interval = max(0.01, min(self.idle_s / 4, 5.0))
        while True:
            await asyncio.sleep(interval)
            await self.check()
