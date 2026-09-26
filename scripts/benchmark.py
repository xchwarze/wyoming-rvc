#!/usr/bin/env python3
"""Benchmark the running service (HTTP API; optional Wyoming TTFA).

  python scripts/benchmark.py                       # server default mode
  python scripts/benchmark.py --disable-rvc         # TTS source only (no RVC)
  python scripts/benchmark.py --compare             # TTS-only vs every RVC mode
  python scripts/benchmark.py --modes whole,sentence -n 10 --json bench.json
  python scripts/benchmark.py --wyoming             # also measure TTFA over Wyoming
  python scripts/benchmark.py --switch teto,miku    # voice switching: cold vs warm model

Server-side timings come from the X-TTS-* response headers; TTFA is measured
client-side on the chunked endpoint (/v1/tts/stream) and, with --wyoming, on
the Wyoming socket exactly like Home Assistant reads it. Standard library only
(plus ``wyoming`` for --wyoming).
"""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request

PHRASES = {
    "very_short": "Hi, I am Teto.",
    "short": "Good morning. The living room temperature is twenty two degrees.",
    "medium": (
        "Done, I turned on the kitchen and hallway lights. I also lowered the bedroom "
        "blinds and set the alarm for seven tomorrow morning."
    ),
    "long": (
        "Here is how the house is doing. The front door is locked and the alarm is armed. "
        "The living room is at twenty two degrees and the humidity is fifty percent. The "
        "washing machine finished ten minutes ago, so you can take the clothes out whenever "
        "you like. Rain is expected tomorrow afternoon, so I recommend taking an umbrella if "
        "you are going out after noon today."
    ),
}
WAV_HEADER = 44


def pct(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class Client:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        parsed = urllib.parse.urlparse(self.base)
        self.host, self.port = parsed.hostname or "localhost", parsed.port or 80

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return json.load(r)

    def post(self, path: str, body: dict | None = None) -> tuple[bytes, dict[str, str]]:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body or {}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.read(), {k.lower(): v for k, v in r.headers.items()}

    def ttfa(self, body: dict) -> tuple[float, float]:
        """(time to first audio byte, total) in ms over the chunked endpoint."""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=600)
        start = time.perf_counter()
        conn.request("POST", "/v1/tts/stream", json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"/v1/tts/stream -> {resp.status}: {resp.read()[:200]!r}")
        received, first = 0, None
        while chunk := resp.read1(65536):
            received += len(chunk)
            if first is None and received > WAV_HEADER:
                first = time.perf_counter()
        end = time.perf_counter()
        conn.close()
        return ((first or end) - start) * 1000, (end - start) * 1000


async def wyoming_ttfa(host: str, port: int, text: str) -> tuple[float, float]:
    from wyoming.audio import AudioChunk, AudioStop
    from wyoming.client import AsyncTcpClient
    from wyoming.error import Error
    from wyoming.tts import Synthesize

    start = time.perf_counter()
    first = None
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Synthesize(text=text).event())
        while True:
            event = await client.read_event()
            if event is None or Error.is_type(event.type):
                raise RuntimeError(f"wyoming error: {event}")
            if AudioChunk.is_type(event.type) and first is None:
                first = time.perf_counter()
            if AudioStop.is_type(event.type):
                break
    end = time.perf_counter()
    return ((first or end) - start) * 1000, (end - start) * 1000


def run_config(client: Client, args: argparse.Namespace, label: str, extra: dict) -> list[dict]:
    rows = []
    for name, text in PHRASES.items():
        if args.phrases and name not in args.phrases:
            continue
        body = {"text": text, **extra}
        for _ in range(args.warmup):
            client.post("/v1/tts", body)
        server: dict[str, list[float]] = {k: [] for k in ("source", "rvc", "encode", "total", "audio", "rtf", "queue")}
        client_ms: list[float] = []
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            _, h = client.post("/v1/tts", body)
            client_ms.append((time.perf_counter() - t0) * 1000)
            for key, header in (
                ("source", "x-tts-source-ms"),
                ("rvc", "x-tts-rvc-ms"),
                ("encode", "x-tts-encode-ms"),
                ("total", "x-tts-total-ms"),
                ("audio", "x-tts-audio-duration-ms"),
                ("rtf", "x-tts-rtf"),
                ("queue", "x-tts-queue-ms"),
            ):
                server[key].append(float(h.get(header, "nan")))
        ttfa = [client.ttfa(body)[0] for _ in range(args.iterations)]
        wy = []
        if args.wyoming and not extra.get("disable_rvc") and extra.get("mode") in (None, args.server_mode):
            wy = [
                asyncio.run(wyoming_ttfa(args.wyoming_host, args.wyoming_port, text))[0] for _ in range(args.iterations)
            ]
        rows.append(
            {
                "config": label,
                "phrase": name,
                "words": len(text.split()),
                "chars": len(text),
                "source_median_ms": statistics.median(server["source"]),
                "rvc_median_ms": statistics.median(server["rvc"]),
                "encode_median_ms": statistics.median(server["encode"]),
                "total_median_ms": statistics.median(server["total"]),
                "total_p95_ms": pct(server["total"], 95),
                "client_median_ms": statistics.median(client_ms),
                "ttfa_median_ms": statistics.median(ttfa),
                "ttfa_p95_ms": pct(ttfa, 95),
                "wyoming_ttfa_median_ms": statistics.median(wy) if wy else None,
                "audio_s": statistics.median(server["audio"]) / 1000,
                "rtf_median": statistics.median(server["rtf"]),
                "iterations": args.iterations,
            }
        )
        print(".", end="", flush=True)
    return rows


def print_table(rows: list[dict]) -> None:
    cols = [
        ("config", "config", "{}"),
        ("phrase", "phrase", "{}"),
        ("words", "words", "{}"),
        ("source_median_ms", "tts med", "{:.0f}"),
        ("rvc_median_ms", "rvc med", "{:.0f}"),
        ("total_median_ms", "total med", "{:.0f}"),
        ("total_p95_ms", "total p95", "{:.0f}"),
        ("ttfa_median_ms", "ttfa med", "{:.0f}"),
        ("wyoming_ttfa_median_ms", "wy ttfa", "{:.0f}"),
        ("audio_s", "audio s", "{:.2f}"),
        ("rtf_median", "rtf", "{:.3f}"),
    ]
    table = [[h for _, h, _ in cols]]
    for row in rows:
        table.append(["-" if row[k] is None else fmt.format(row[k]) for k, _, fmt in cols])
    widths = [max(len(r[i]) for r in table) for i in range(len(cols))]
    for i, r in enumerate(table):
        print("  ".join(c.rjust(w) if j > 1 else c.ljust(w) for j, (c, w) in enumerate(zip(r, widths, strict=True))))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def run_switch(client: Client, voices: list[str]) -> list[dict]:
    """A, A, B, B, A: shows model load cost on a switch vs warm requests."""
    a, b = voices
    text = PHRASES["short"]
    rows = []
    client.post("/v1/metrics/reset")
    for step, voice in enumerate([a, a, b, b, a], start=1):
        _, h = client.post("/v1/tts", {"text": text, "voice": voice})
        load_ms = float(h.get("x-tts-rvc-load-ms", "0"))
        rows.append(
            {
                "step": step,
                "voice": voice,
                "state": "cold (model loaded)" if load_ms > 0 else "warm",
                "source_ms": float(h.get("x-tts-source-ms", "nan")),
                "rvc_model_load_ms": load_ms,
                "rvc_inference_ms": float(h.get("x-tts-rvc-ms", "nan")),
                "total_ms": float(h.get("x-tts-total-ms", "nan")),
            }
        )
    memory = client.get("/info").get("gpu_memory", {})
    header = f"{'step':>4}  {'voice':<12} {'state':<20} {'tts ms':>7} {'load ms':>8} {'rvc ms':>7} {'total ms':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['step']:>4}  {r['voice']:<12} {r['state']:<20} {r['source_ms']:>7.0f} "
            f"{r['rvc_model_load_ms']:>8.0f} {r['rvc_inference_ms']:>7.0f} {r['total_ms']:>9.0f}"
        )
    if memory:
        print(
            f"peak CUDA VRAM: allocated {memory.get('peak_allocated_mb', 0):.0f} MB, "
            f"reserved {memory.get('peak_reserved_mb', 0):.0f} MB"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("-n", "--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1, help="unrecorded requests per phrase")
    parser.add_argument("--modes", help="comma list of whole,sentence (default: server default)")
    parser.add_argument("--disable-rvc", action="store_true", help="benchmark the TTS source without RVC")
    parser.add_argument("--compare", action="store_true", help="TTS-only plus every RVC mode")
    parser.add_argument("--phrases", help="comma list of " + ",".join(PHRASES))
    parser.add_argument("--wyoming", action="store_true", help="also measure TTFA over Wyoming")
    parser.add_argument("--wyoming-host", default="localhost")
    parser.add_argument("--wyoming-port", type=int, default=10200)
    parser.add_argument("--switch", help="two voice ids, e.g. teto,miku: benchmark voice switching")
    parser.add_argument("--json", help="write raw results to this file")
    args = parser.parse_args()
    args.phrases = args.phrases.split(",") if args.phrases else None

    client = Client(args.url)
    try:
        info = client.get("/info")
    except OSError as err:
        sys.exit(f"cannot reach {args.url}: {err}")
    if not info.get("ready"):
        sys.exit(f"service not ready (stage={info.get('stage')})")
    args.server_mode = (info.get("defaults") or {}).get("mode", "whole")
    platform = info.get("platform", {})
    rvc = info.get("rvc") or {}
    print(
        f"torch {platform.get('torch')} cuda={platform.get('cuda_runtime')} gpu={platform.get('gpu')} "
        f"default_voice={rvc.get('default_voice')} voices={[v['id'] for v in rvc.get('voices', []) if v['installed']]} "
        f"source={info['source'].get('name')} "
        f"server_mode={args.server_mode}"
    )

    if args.switch:
        voices = args.switch.split(",")
        if len(voices) != 2:
            sys.exit("--switch needs exactly two voice ids, e.g. teto,miku")
        rows = run_switch(client, voices)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump({"info": info, "switch": rows}, f, indent=2)
        return

    configs: list[tuple[str, dict]] = []
    if args.compare:
        configs = [("tts-only", {"disable_rvc": True})] + [(f"rvc-{m}", {"mode": m}) for m in ("whole", "sentence")]
    elif args.disable_rvc:
        configs = [("tts-only", {"disable_rvc": True})]
    elif args.modes:
        configs = [(f"rvc-{m}", {"mode": m}) for m in args.modes.split(",")]
    else:
        configs = [(f"rvc-{args.server_mode}", {})]

    client.post("/v1/metrics/reset")
    rows: list[dict] = []
    for label, extra in configs:
        rows += run_config(client, args, label, extra)
    print()
    memory = client.get("/info").get("gpu_memory", {})
    print_table(rows)
    if memory:
        print(
            f"\npeak CUDA VRAM during benchmark: allocated {memory.get('peak_allocated_mb', 0):.0f} MB, "
            f"reserved {memory.get('peak_reserved_mb', 0):.0f} MB (process total now {memory.get('reserved_mb', 0):.0f} MB)"
        )
    print("rtf = server processing time / generated audio duration (lower is better, <1 is faster than real time)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"info": info, "gpu_memory": memory, "results": rows}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
