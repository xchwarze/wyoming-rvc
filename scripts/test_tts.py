#!/usr/bin/env python3
"""Synthesize one phrase through the running service and save a WAV.

  python scripts/test_tts.py "Hi, I am Teto." -o teto.wav            # HTTP
  python scripts/test_tts.py "Hi, I am Teto." -o teto.wav --wyoming  # Wyoming (like Home Assistant)

Only the standard library is needed for HTTP; --wyoming needs the ``wyoming`` package.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
import wave


def via_http(args: argparse.Namespace) -> None:
    payload = {"text": args.text}
    for key in ("pitch", "index_rate", "protect", "mode"):
        if getattr(args, key) is not None:
            payload[key] = getattr(args, key)
    if args.disable_rvc:
        payload["disable_rvc"] = True
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/tts",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
            headers = {k: v for k, v in response.headers.items() if k.lower().startswith("x-tts-")}
    except urllib.error.HTTPError as err:
        sys.exit(f"HTTP {err.code}: {err.read().decode(errors='replace')}")
    elapsed = (time.perf_counter() - start) * 1000
    with open(args.output, "wb") as f:
        f.write(data)
    print(f"saved {args.output} ({len(data)} bytes, client {elapsed:.0f} ms)")
    for key, value in headers.items():
        print(f"  {key[6:].lower()}: {value}")


async def via_wyoming(args: argparse.Namespace) -> None:
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.client import AsyncTcpClient
    from wyoming.error import Error
    from wyoming.tts import Synthesize, SynthesizeVoice

    start = time.perf_counter()
    first = None
    fmt = None
    frames = bytearray()
    async with AsyncTcpClient(args.host, args.port) as client:
        await client.write_event(Synthesize(text=args.text, voice=SynthesizeVoice(name="teto")).event())
        while True:
            event = await client.read_event()
            if event is None:
                sys.exit("connection closed")
            if Error.is_type(event.type):
                sys.exit(f"server error: {Error.from_event(event).text}")
            if AudioStart.is_type(event.type):
                fmt = AudioStart.from_event(event)
            elif AudioChunk.is_type(event.type):
                first = first or time.perf_counter()
                frames += AudioChunk.from_event(event).audio
            elif AudioStop.is_type(event.type):
                break
    total = (time.perf_counter() - start) * 1000
    assert fmt is not None
    with wave.open(args.output, "wb") as wav:
        wav.setnchannels(fmt.channels)
        wav.setsampwidth(fmt.width)
        wav.setframerate(fmt.rate)
        wav.writeframes(bytes(frames))
    seconds = len(frames) / (fmt.rate * fmt.width * fmt.channels)
    ttfa = (first - start) * 1000 if first else float("nan")
    print(f"saved {args.output}: {seconds:.2f} s @ {fmt.rate} Hz, ttfa {ttfa:.0f} ms, total {total:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", default="Hi, I am Teto. How can I help you?")
    parser.add_argument("-o", "--output", default="teto.wav")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--wyoming", action="store_true", help="use Wyoming instead of HTTP")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=10200)
    parser.add_argument("--pitch", type=int)
    parser.add_argument("--index-rate", type=float)
    parser.add_argument("--protect", type=float)
    parser.add_argument("--mode", choices=["whole", "sentence"])
    parser.add_argument("--disable-rvc", action="store_true")
    args = parser.parse_args()
    if args.wyoming:
        asyncio.run(via_wyoming(args))
    else:
        via_http(args)


if __name__ == "__main__":
    main()
