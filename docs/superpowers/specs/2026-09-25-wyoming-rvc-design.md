# wyoming-rvc: design

Date: 2026-09-25. Status: approved in conversation ("sí, seguí de forma autónoma").

## Goal

A Wyoming TTS service for Home Assistant that gives any TTS an RVC voice. The
owner uses it with TetoTalk (`Slichi/KasaneTeto`); the published project must work
with any RVC v1/v2 ContentVec model on Hugging Face.

Success means **the voice conversion is imperceptible**: time to first audio (TTFA)
close to plain TTS, with every model resident in memory and no per-request process,
file, or model load.

Constraints: professional and polished, **not over-engineered**; every layer can be
turned on or off with environment variables.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Voices per instance | **One**, chosen by env (`RVC_REPO_ID`, `RVC_MODEL_FILE`, `RVC_INDEX_FILE`) and always resident | Zero switch latency, minimal code. Another voice means another container. |
| Audio source | `SOURCE=piper` (default: in-process, fastest) or `SOURCE=wyoming` (proxy to any Wyoming TTS via `WYOMING_UPSTREAM=tcp://host:port`) | The fast path stays in-process; the proxy makes this a drop-in layer for any TTS. |
| Piper device | `PIPER_DEVICE=auto` (CUDA if `DEVICE=cuda`), `cuda`, `cpu` | Measured about 3× faster on GPU for long texts. The JIT cache lives in `/models/cuda-cache`. |
| Default mode | `RVC_MODE=sentence` | Measured TTFA of about 200–270 ms, vs 0.4–2 s for `whole`. `whole` and `stream` (experimental) remain. |
| Toggles | `RVC_ENABLED`, `SOURCE`, `PIPER_DEVICE`, `RVC_MODE` | Env only; no config file. |

## Architecture

```
text ─► Source (per sentence) ─► RVC layer (optional) ─► PCM16 ─► Wyoming / HTTP
                               ├─ PiperSource   (in-process)
                               └─ WyomingSource (upstream TCP client)
```

* **Source protocol** (unchanged shape): `sample_rate: int` and
  `sentences(text) -> Iterator[np.ndarray]` (float32 mono at `sample_rate`).
  * `PiperEngine`: exists already.
  * `WyomingSource` (new): at `load()`, sends `describe` (fails fast if the
    upstream has no installed TTS) and one probe synthesis to learn the sample rate.
    Per request it opens one TCP connection, splits the text with
    `sentence_stream.SentenceBoundaryDetector`, sends one `synthesize` per sentence
    and yields each sentence's audio, resampled to the nominal rate if the upstream
    rate differs. `UPSTREAM_VOICE` / `UPSTREAM_SPEAKER` select the upstream voice.
* Everything else is kept: resident `RvcEngine`, zip-aware model loader, Wyoming
  server (one program, one voice), HTTP debug API, metrics, the concurrency semaphore.

## Naming

Repository and image: `wyoming-rvc` (`ghcr.io/xchwarze/wyoming-rvc`), Python package
`wyoming_rvc`. Env: `VOICE_NAME` (default `teto`), `VOICE_LANGUAGE` (default `es`),
`WYOMING_PROGRAM_NAME` (default `Wyoming RVC`).

## Error handling

Fail fast at startup when the upstream is unreachable, has no TTS, or the probe
returns no audio (the same fail-fast rules as for models and CUDA). Upstream errors
during a request become a Wyoming `error` event or an HTTP 500.

## Testing

* Unit: fake source/RVC (ordering, an aborted stream closes the source, cancellation
  mid-step), sentence splitting, and `WyomingSource` against an in-process Wyoming
  TTS server (probe, voices, unreachable upstream, end to end).
* Integration (real models), HA e2e (`tests_ha`, real HA integration), and
  `scripts/benchmark.py`: re-run and update the README numbers.

## Out of scope (YAGNI)

Multiple voices per instance, LRU model cache, fp16, `torch.compile`, extra audio
effects, extra pitch extractors, and a web UI.

## Measured during implementation (RTX 5080, 20-40 interleaved runs, medians)

| Idea | Result | Decision |
|---|---|---|
| Prefetch: TTS for sentence N+1 while RVC runs on sentence N | TTFA +55 ms (217 → 271 ms); total unchanged. HA's streaming path sends one sentence per request anyway. | **Rejected** |
| First-chunk split at the first clause | −25 ms TTFA on a 110-character first sentence, +200 ms total (an extra TTS call). GPU latency barely scales with length. | **Rejected** |
| RMVPE in a worker thread on a second CUDA stream, parallel to ContentVec | Short sentences 69 → 83 ms (GIL contention on kernel launches); only 6 s+ sentences gain. | **Rejected** |
| Retrieval as exact k-NN on the GPU over the resident index vectors (instead of CPU FAISS IVF with nprobe=1) | −14 ms on 2.5–6.6 s sentences, same on short ones. Mel correlation with Applio 0.994 (was 0.996; RVC noise makes runs differ by about this much). | **Adopted** |
| RMVPE fp16 autocast | 37 → 43 ms | Rejected |

Result: in `sentence` mode, RVC adds about **85 ms** to the time to first audio over
TTS-only (≈130 → ≈217 ms, Piper on GPU).
