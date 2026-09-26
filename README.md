# wyoming-rvc

A [Wyoming](https://github.com/OHF-Voice/wyoming) text-to-speech service that gives any
voice an **RVC timbre**, for Home Assistant Assist:

```
text ─► TTS source ─► RVC voice layer ─► PCM 16-bit ─► Wyoming ─► Home Assistant
        │                 │
        │                 └─ any RVC v1/v2 model from Hugging Face (default: TetoTalk, Kasane Teto)
        ├─ SOURCE=piper    Piper in-process (default, fastest)
        └─ SOURCE=wyoming  any existing Wyoming TTS (wyoming-piper, …) as upstream
```

* **Built for low latency.** Everything is resident in memory: the Piper voice, the RVC
  generator, ContentVec, RMVPE, and the retrieval index vectors on the GPU. No
  subprocesses, temp files, or model loads per request. With the defaults on an
  RTX 5080, RVC adds **about 80–110 ms** to the time to first audio.
* **Native Home Assistant integration.** Add it through the built-in *Wyoming Protocol*
  integration, including streaming text input from LLM agents. Interoperability is
  tested against Home Assistant's own integration code.
* **Every layer can be switched with environment variables:** the source, the voice
  model, `RVC_ENABLED`, the Piper device, and the conversion mode.
* No GUI, no Gradio, no training code. HTTP API for debugging and benchmarks.

> Status: young project, measured on one machine (numbers below). Benchmark your own
> hardware with `scripts/benchmark.py`.

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Home Assistant](#home-assistant) (app/add-on or Docker)
- [Using another voice or TTS](#using-another-voice-or-tts)
- [Performance](#performance)
- [Configuration](#configuration)
- [HTTP API](#http-api)
- [How it works](#how-it-works)
- [Development and tests](#development-and-tests)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Alternatives](#alternatives)
- [Licensing](#licensing)

## Requirements

* NVIDIA GPU, Turing or newer (sm_75 … sm_120, including RTX 50xx), about 3 GB free VRAM.
* NVIDIA driver **≥ 570** (the image uses the CUDA 12.8 build of PyTorch).
* Linux with Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  **or** Windows 10/11 with Docker Desktop on the WSL 2 backend (GPU support is built in).
* About 7 GB of disk for the image, plus about 1 GB for models (downloaded on first start into `./models`).

## Quick start

Prebuilt image (built by GitHub Actions on every push to `main`):

```bash
git clone https://github.com/xchwarze/wyoming-rvc && cd wyoming-rvc
docker compose pull          # or: docker compose build
docker compose up -d
docker compose logs -f       # wait for "Service ready"
```

The first start downloads about 1 GB (Piper voice, TetoTalk, ContentVec, RMVPE) into
`./models`. With Piper on the GPU (the default on a GPU host), the first warmup also
JIT-compiles CUDA kernels once (up to about 1 minute on RTX 50xx). Both results are cached;
later starts take seconds.

Without compose:

```bash
docker run -d --name wyoming-rvc --gpus all --restart unless-stopped \
  -p 127.0.0.1:8080:8080 -p 10200:10200 -v "$PWD/models:/models" \
  ghcr.io/xchwarze/wyoming-rvc:latest
```

Check that it works:

```bash
docker compose exec wyoming-rvc nvidia-smi                       # GPU visible in the container
curl http://localhost:8080/readyz                                # {"ready":true,...}
curl -X POST http://localhost:8080/v1/tts -H "Content-Type: application/json" \
     -d '{"text":"Hi, I am Teto."}' --output teto.wav
docker compose exec wyoming-rvc python scripts/benchmark.py --compare --wyoming
```

On Windows PowerShell, use `curl.exe` and escape the quotes:
`curl.exe -X POST http://localhost:8080/v1/tts -H "Content-Type: application/json" -d '{\"text\":\"Hi, I am Teto.\"}' --output teto.wav`.
Or avoid shell quoting completely:

```bash
docker compose exec wyoming-rvc python scripts/test_tts.py "Hi, I am Teto." -o /models/teto.wav
docker compose exec wyoming-rvc python scripts/test_tts.py "Hi, I am Teto." -o /models/teto.wav --wyoming
```

## Home Assistant

There are two ways to run it. With either one, Home Assistant talks to it through the
built-in **Wyoming Protocol** integration.

### Option A: Home Assistant app (add-on)

1. **Settings → Apps → App Store → ⋮ → Repositories**, then add
   `https://github.com/xchwarze/wyoming-rvc`.
2. Install **Wyoming RVC**, start it, and wait for the first start to download the
   models (about 1 GB).
3. Home Assistant discovers it automatically: accept the new **Wyoming Protocol** entry
   under **Settings → Devices & Services**.

The app is amd64 only. **Home Assistant OS does not give apps access to NVIDIA GPUs**, so
there it runs on the CPU (`device: auto`), which works but is noticeably slower. The
app's Documentation tab lists its options.

### Option B: Docker on a GPU machine (lowest latency)

Run the image as shown in [Quick start](#quick-start) on a machine with an NVIDIA GPU,
then add it to Home Assistant. No custom component, REST command, or shell command is
needed:

1. **Settings → Devices & Services → Add Integration → Wyoming Protocol**
2. **Host:** the IP of the machine running the container; **Port:** `10200`
3. The entry **Wyoming RVC** appears, with the TTS entity `tts.wyoming_rvc` and the voice `teto` (`en`).
4. **Settings → Voice assistants →** your assistant **→ Text-to-speech:** choose **Wyoming RVC**.

To test, use **Developer tools → Actions → `tts.speak`**, or the ▶ button next to the voice in the assistant settings.

* With Docker Desktop on Windows, use the Windows host's LAN IP and allow TCP 10200 in the firewall.
* Neither port has authentication. Wyoming (10200) must be reachable by Home Assistant; the
  compose file publishes the HTTP debug API (8080) on `127.0.0.1` only. Change it to
  `"8080:8080"` if you need to reach it from another machine on a trusted network.
* Streaming text input is advertised, so with LLM agents each sentence is spoken as soon as it is complete (`WYOMING_STREAMING=false` turns this off).
* Zeroconf discovery is optional (`WYOMING_ZEROCONF=true`) and needs `network_mode: host` on a Linux host.

**Verified interoperability.** `tests_ha/` drives the real **Home Assistant 2026.9.3**
`wyoming` integration (config flow, TTS entity, and TTS manager with ffmpeg, pinned to
`wyoming==1.10.0` as in that release) against a running server. All 5 tests pass in both
source modes:

| Test | What Home Assistant does |
|---|---|
| `test_config_flow_exposes_tts_entity` | Add Integration → host/port → entry **Wyoming RVC**, entity `tts.wyoming_rvc` (`es`, voice `teto`, streaming) |
| `test_entity_get_tts_audio` | `synthesize` → `audio-start/chunk/stop` → 32 kHz 16-bit mono WAV |
| `test_tts_manager_with_ffmpeg_conversion` | TTS manager + cache + ffmpeg to 16 kHz WAV (as for voice satellites) |
| `test_streaming_text_input` | streamed message → `synthesize-start/chunk…/stop` → `synthesize-stopped` |
| `test_default_playback_format_is_mp3` | the default output for media players (MP3 via ffmpeg) |

Playback on a physical speaker still needs a manual check on your installation.

## Using another voice or TTS

**Another RVC voice.** Point the service at any Hugging Face repo that contains an RVC
`.pth` (and optionally an `.index`), either directly or inside a `.zip`:

```yaml
    environment:
      - RVC_REPO_ID=someone/some-rvc-voice
      - RVC_MODEL_FILE=voice_300e.pth       # only if the repo has several .pth files
      - RVC_INDEX_FILE=added_voice.index    # only if it has several .index files
      - RVC_PITCH=0                         # semitones; tune by ear
      - VOICE_NAME=myvoice
      - VOICE_LANGUAGE=en
```

To use local files instead, mount them and set absolute paths:
`RVC_MODEL_FILE=/models/my/voice.pth`, `RVC_INDEX_FILE=/models/my/voice.index`.

**Several RVC voices.** Put a `voices.yaml` in `/config` (mount `./config:/config`, see
[examples/voices.yaml](examples/voices.yaml)). Every enabled voice whose files download
correctly is listed in Home Assistant, and each assistant picks one:

```yaml
voices:
  teto:
    name: "Kasane Teto"
    repo_id: "Slichi/KasaneTeto"
    preload: true
  miku:
    name: "Hatsune Miku"
    repo_id: "someone/miku-rvc"
    pitch: 4
```

ContentVec, RMVPE and Piper are loaded once and shared. Voices are downloaded at startup
but only enter VRAM when first used; `RVC_MAX_LOADED_MODELS` (default 1) keeps that many
resident and unloads the least recently used one. A switch to a voice that is not
resident costs one model load (see `scripts/benchmark.py --switch teto,miku`); raise the
limit if you have the VRAM and switch often. A voice in use is never unloaded. A request
for an unknown voice fails with an error instead of falling back to another voice.
Without `voices.yaml`, the `RVC_*` variables define a single voice as before.

**Another Piper voice or language.** The default source voice is `en_US-ljspeech-high`
(female, US English). Any voice from
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) works, or your own model
with `PIPER_MODEL=/models/x.onnx`. For another language, also set the language Home
Assistant should list and a warmup phrase in that language. For example, Spanish:

```yaml
      - PIPER_VOICE=es_AR-daniela-high   # or es_MX-claude-high, es_ES-davefx-medium, ...
      - VOICE_LANGUAGE=es
      - WARMUP_TEXT=Sistema iniciado.
```

**Another TTS (layer mode).** Put wyoming-rvc in front of an existing Wyoming TTS. Home
Assistant talks to wyoming-rvc, and wyoming-rvc talks to the upstream:

```yaml
services:
  piper:
    image: rhasspy/wyoming-piper
    command: --voice en_US-ljspeech-high
    volumes: [./piper-data:/data]
  wyoming-rvc:
    image: ghcr.io/xchwarze/wyoming-rvc:latest
    ports: ["10200:10200", "127.0.0.1:8080:8080"]
    volumes: [./models:/models]
    environment:
      - SOURCE=wyoming
      - WYOMING_UPSTREAM=tcp://piper:10200
      # - UPSTREAM_VOICE=en_US-ljspeech-high   # optional voice/speaker for the upstream
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
```

The service checks the upstream at startup (it must offer an installed TTS, and it probes
the sample rate) and fails fast if the upstream is unreachable. Each request uses one
connection to the upstream and sends one `synthesize` per sentence, so the first sentence
is converted and sent as soon as the upstream returns it.

## Performance

RTX 5080 16 GB, Ryzen 9 5900X (WSL 2: 6 cores / 12 threads), warm models,
`SOURCE=piper` with Piper on the GPU, `RVC_MODE=sentence`, TetoTalk 32 kHz, 10
iterations, medians. These numbers were measured with the Spanish voice `es_AR-daniela-high`
(same size and architecture as the English default) and Spanish phrases of the same lengths. Produced with `scripts/benchmark.py --compare --wyoming`.

* **TTS** is time spent in the source.
* **Total** is server processing time.
* **TTFA** is the client-measured time to the first audio byte (on the HTTP streaming endpoint; "Wyoming TTFA" is measured on the Wyoming socket the way Home Assistant reads it).
* **RTF** is processing time ÷ audio duration.

| config | phrase (words) | TTS ms | RVC ms | total ms | total p95 | TTFA ms | Wyoming TTFA | audio s | RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TTS only | very short (3) | 78 | – | 78 | 148 | 91 | – | 0.90 | 0.090 |
| TTS only | short (10) | 272 | – | 273 | 340 | 140 | – | 3.29 | 0.083 |
| TTS only | medium (25) | 253 | – | 254 | 307 | 128 | – | 6.84 | 0.037 |
| TTS only | long (66) | 672 | – | 674 | 801 | 121 | – | 17.24 | 0.039 |
| **`sentence`** (default) | very short | 80 | 85 | 177 | 278 | 171 | 188 | 0.88 | 0.199 |
| **`sentence`** | short | 264 | 181 | 488 | 552 | 231 | 239 | 3.26 | 0.150 |
| **`sentence`** | medium | 264 | 244 | 513 | 655 | 227 | 218 | 6.84 | 0.076 |
| **`sentence`** | long | 586 | 579 | 1178 | 1234 | 230 | 228 | 17.15 | 0.069 |
| `whole` | very short | 87 | 78 | 186 | 232 | 183 | – | 0.89 | 0.214 |
| `whole` | short | 267 | 151 | 440 | 521 | 372 | – | 3.32 | 0.133 |
| `whole` | medium | 248 | 149 | 402 | 470 | 375 | – | 6.82 | 0.059 |
| `whole` | long | 652 | 256 | 910 | 1091 | 924 | – | 17.25 | 0.053 |

Peak CUDA memory: about 1.8 GB allocated, 2.8 GB reserved (Piper and RVC together).

**Switching voices** (`scripts/benchmark.py --switch teto,miku`, short phrase, same
machine, English default Piper voice). Miku is
[binant/Hatsune_Miku__RVC_v2_](https://huggingface.co/binant/Hatsune_Miku__RVC_v2_)
(48 kHz, 223 MB index). A cold request includes loading the model and index into VRAM:

| step | voice | `RVC_MAX_LOADED_MODELS=1` | load ms | total ms | `=2` | load ms | total ms |
|---|---|---|---:|---:|---|---:|---:|
| 1 | teto | warm | 0 | 638 | warm | 0 | 729 |
| 2 | teto | warm | 0 | 454 | warm | 0 | 425 |
| 3 | miku | cold | 801 | 1446 | cold | 808 | 1259 |
| 4 | miku | warm | 0 | 462 | warm | 0 | 486 |
| 5 | teto | cold | 317 | 762 | **warm** | 0 | 394 |

Peak CUDA memory allocated: 1.34 GB with one resident voice, 1.52 GB with both. The load
time is mostly the index; a voice without an `.index` loads faster.

**What the numbers say**

* In `sentence` mode, the **first audio arrives about 80–110 ms later than with the TTS
  alone**, whatever the length of the text. The remaining audio is produced 7–14× faster
  than real time, so playback never waits.
* `whole` has the lowest total time, but nothing plays until the entire text is
  converted (TTFA ≈ total). Use it for batch or offline generation.
* With `SOURCE=wyoming` and the official `wyoming-piper` running Piper on the CPU as
  upstream, RVC adds the same amount. The total is dominated by the upstream TTS.
* Correctness: output matches Applio 3.6.5 for the same input and parameters (mel
  correlation 0.994–0.996, same RMS; RVC injects random noise, so runs are never
  bit-identical).

Several latency ideas were measured and **rejected** because they made the first audio
slower or added complexity for no audible gain: running the TTS for the next sentence
concurrently with RVC (+55 ms TTFA), splitting the first sentence at a comma
(−25 ms TTFA, +200 ms total), running pitch extraction on a second CUDA stream (slower for
short sentences), fp16 RMVPE (slower), and a `stream` mode that chunked inside each
sentence with crossfades (same TTFA as `sentence` at about 2× the GPU time). Details are in
`docs/superpowers/specs/2026-09-25-wyoming-rvc-design.md`.

## Configuration

Copy `.env.example` to `.env`; compose reads it. Empty values mean "use the default".

| Variable | Default | Meaning |
|---|---|---|
| `DEVICE` | `cuda` | Device for RVC: `cuda` (refuses to start without a usable GPU), `cpu`, or `auto` (CUDA when available, otherwise CPU; the add-on default) |
| `SOURCE` | `piper` | `piper` (in-process) or `wyoming` (upstream Wyoming TTS) |
| `WYOMING_UPSTREAM` | – | `tcp://host:port` of the upstream TTS (required when `SOURCE=wyoming`) |
| `UPSTREAM_VOICE` / `UPSTREAM_SPEAKER` | upstream default | Voice/speaker requested from the upstream |
| `UPSTREAM_TIMEOUT_S` | `30` | Connect/read timeout for the upstream |
| `PIPER_VOICE` | `en_US-ljspeech-high` | Any [Piper voice](https://huggingface.co/rhasspy/piper-voices), downloaded once to `/models/piper` |
| `PIPER_MODEL` / `PIPER_CONFIG` | – | Explicit `.onnx` (and `.onnx.json`); overrides `PIPER_VOICE` |
| `PIPER_DEVICE` | `auto` | `auto` (GPU when `DEVICE=cuda`), `cuda`, `cpu` |
| `PIPER_LENGTH_SCALE`, `PIPER_NOISE_SCALE`, `PIPER_NOISE_W_SCALE`, `PIPER_SPEAKER_ID` | voice defaults | Piper synthesis options |
| `SENTENCE_SILENCE_MS` | `0` | Silence inserted between sentences (any source) |
| `RVC_ENABLED` | `true` | `false` passes the source audio through unchanged |
| `RVC_REPO_ID` | `Slichi/KasaneTeto` | Hugging Face repo with the RVC model (`.pth`, `.index`, or `.zip` archives) |
| `RVC_REVISION` | `main` | Pin a repo commit |
| `RVC_MODEL_FILE` / `RVC_INDEX_FILE` | auto | File name, relative path or absolute path. Required when the repo has several candidates. |
| `RVC_PITCH` | `0` | Semitones, −24…24 |
| `RVC_INDEX_RATE` | `0.6` | Retrieval strength, 0…1 (0 turns it off) |
| `RVC_PROTECT` | `0.33` | Protect unvoiced consonants and breaths, 0…0.5 (0.5 = off) |
| `RVC_F0_METHOD` | `rmvpe` | Pitch extractor (only `rmvpe`) |
| `RVC_MODE` | `sentence` | `sentence` (lowest time to first audio) or `whole` (lowest total time) |
| `RVC_CONCURRENCY` | `1` | Simultaneous conversions on the same loaded models. At 1, extra requests wait in a queue; each extra slot adds peak activation memory. |
| `RVC_ASSETS_REPO_ID` / `RVC_ASSETS_REVISION` | `IAHispano/Applio` / `main` | Source of ContentVec and RMVPE |
| `VOICES_FILE` | `/config/voices.yaml` | Voice list; if the default path is missing, the `RVC_*` variables define one voice |
| `DEFAULT_VOICE` | first enabled voice | Voice used when a request names none |
| `RVC_MAX_LOADED_MODELS` | `1` | RVC voices kept in VRAM (LRU). Each one costs the model plus its index, typically 100–400 MB |
| `VOICE_NAME` / `VOICE_LANGUAGE` | `teto` / `en` | Voice advertised to Home Assistant (set the language to match `PIPER_VOICE`) |
| `WYOMING_PROGRAM_NAME` | `Wyoming RVC` | Name shown in Home Assistant |
| `WYOMING_HOST` / `WYOMING_PORT` | `0.0.0.0` / `10200` | Wyoming server |
| `WYOMING_STREAMING` | `true` | Advertise streaming text input |
| `WYOMING_ZEROCONF` / `WYOMING_ZEROCONF_NAME` | `false` / MAC | mDNS discovery |
| `WYOMING_SAMPLES_PER_CHUNK` | `1024` | Samples per `audio-chunk` event |
| `HTTP_ENABLED` / `HTTP_HOST` / `HTTP_PORT` | `true` / `0.0.0.0` / `8080` | Debug API |
| `MODELS_DIR` | `/models` | Root for downloads (`piper/`, `rvc/`, `huggingface/`, `cuda-cache/`) |
| `HF_HOME` / `HF_TOKEN` | `$MODELS_DIR/huggingface` / – | Hugging Face cache and optional token (the CUDA JIT cache goes to `$MODELS_DIR/cuda-cache`) |
| `WARMUP_TEXT` | `System ready.` | Synthesized and discarded at startup (use your voice's language) |
| `MAX_TEXT_CHARS` | `5000` | HTTP request limit |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `text` | `LOG_FORMAT=json` gives structured logs |

**Model resolution.** The RVC repo is snapshot-downloaded (only `*.pth`, `*.index`,
`*.zip`) into `HF_HOME`. Zips are extracted once to `/models/rvc/<repo>/<archive>/`,
keeping only `.pth`/`.index` members, flattened (no path traversal). The service expects
exactly one `.pth` and at most one `.index` unless `RVC_MODEL_FILE`/`RVC_INDEX_FILE` pick
one; when the choice is ambiguous it fails and lists the candidates. Cached files resolve
without network access. Any RVC v1/v2 model trained with ContentVec and a HiFi-GAN,
HiFi-GAN-NSF or MRF-HiFi-GAN vocoder is supported. `Slichi/KasaneTeto` ships a zip with
`KasaneTetotalkV1_240e_7680s.pth` (v2, 32 kHz, f0, 240 epochs) and a 25,572 × 768 index.

## HTTP API

For debugging and benchmarks only; Home Assistant uses Wyoming.

| Method | Path | |
|---|---|---|
| `GET` | `/healthz` | Process alive |
| `GET` | `/readyz` | 200 only once every model is loaded and warmed up; 503 before |
| `GET` | `/info` | Versions, GPU, source, model metadata, defaults, CUDA memory |
| `GET` | `/v1/voices` | Configured voices: installed, loaded, default, sample rate, error |
| `POST` | `/v1/tts` | JSON → `audio/wav`, with timings in `X-TTS-*` headers |
| `POST` | `/v1/tts/stream` | Same body; chunked WAV (0-frame header, then PCM as produced) |
| `POST` | `/v1/metrics/reset` | Reset the CUDA peak-memory counters |

```json
{"text": "Hi, I am Teto.", "voice": "teto", "pitch": 0, "index_rate": 0.6, "protect": 0.33,
 "f0_method": "rmvpe", "mode": "sentence", "disable_rvc": false}
```

Only `text` is required; `voice` defaults to `DEFAULT_VOICE`, and the RVC parameters to
that voice's settings. Errors are returned as `{"error": "..."}` with status 422
(invalid), 413 (too long), 503 (not ready), or 500 (synthesis failed). Bodies that are
not valid UTF-8 are decoded as cp1252, because Windows shells re-encode curl arguments.
Every request logs one line:

```
synthesis source=wyoming text_chars=66 sentences=3 mode=sentence rvc=True source_ms=235.1 rvc_ms=259.3 encode_ms=0.4 queue_ms=0.0 total_ms=512.0 ttfa_ms=231.2 audio_duration_ms=6810.0 sample_rate=32000 rtf=0.0752
```

## How it works

```
startup:  config ─► CUDA check ─► resolve/download ─► load source (Piper, or probe upstream)
          ─► load RVC (generator, ContentVec, RMVPE, index vectors → GPU) ─► warmup (all modes)
          ─► Wyoming up ─► ready
request:  text ─► source, one sentence at a time (worker thread) ─► one resample to 16 kHz
          ─► 48 Hz high-pass ─► RMVPE f0 + ContentVec ─► exact 8-NN retrieval on GPU
          ─► RVC generator ─► 32 kHz ─► clip guard ─► PCM16 ─► audio-start/chunk*/stop
```

| File | Role |
|---|---|
| `src/wyoming_rvc/main.py` | Startup and shutdown, device checks, signals |
| `config.py` | Environment parsing and validation |
| `piper_engine.py` | Resident Piper voice (source `piper`) |
| `wyoming_source.py` | Upstream Wyoming TTS client (source `wyoming`) |
| `model_loader.py` | Resolves or downloads Piper, RVC (zip-aware) and assets; cache first |
| `rvc_engine.py` | Resident RVC: `load()`, `convert()` |
| `pipeline.py` | Source → RVC per mode, concurrency queue, metrics |
| `wyoming_server.py` / `http_server.py` | Front-ends |
| `audio.py`, `text.py`, `metrics.py` | PCM/resample/WAV, sentence splitting, timings |
| `vendor/applio/` | Minimal RVC network and RMVPE code from Applio (MIT) |

Design notes:

* Applio re-creates RMVPE and re-reads the FAISS index on every call. Here everything
  loads once. Weight-norm is folded into plain weights at load time, and retrieval is an
  exact k-NN matmul against the index vectors kept on the GPU.
* Audio is resampled once on the way in (source rate → 16 kHz) and sent at the model's
  native rate. There is no normalization beyond Applio's input peak guard and a
  clip-only output guard.
* Blocking work runs in worker threads. Models are never duplicated: RVC calls go through
  a semaphore (`RVC_CONCURRENCY`, default 1), so concurrent requests queue. If a client
  disconnects mid-sentence, the in-flight conversion still finishes (and keeps its slot)
  and the source is closed (tested).
* SIGTERM closes Wyoming and HTTP and exits cleanly.

## Development and tests

```bash
pip install -r requirements-test.txt && pip install --no-deps -e .
pytest                                   # 110 unit tests, no torch or downloads, ~3 s
```

With real models (downloads about 1 GB on the first run; uses the GPU):

```bash
uv pip install --index-strategy unsafe-best-match --require-hashes -r requirements.lock
uv pip install --no-deps -e . && uv pip install pytest pytest-asyncio
WYOMING_RVC_INTEGRATION=1 MODELS_DIR=$HOME/models pytest tests/test_integration.py
```

<a id="home-assistant-e2e-test"></a>Home Assistant e2e (Python ≥ 3.14.2, a running server, `ffmpeg` on `PATH`):

```bash
python3.14 -m venv .venv-ha && . .venv-ha/bin/activate
pip install -r tests_ha/requirements.txt
WYOMING_RVC_HOST=127.0.0.1 WYOMING_RVC_PORT=10200 WYOMING_RVC_HA_OUT=./ha-out pytest tests_ha -v
```

`WYOMING_RVC_HA_OUT` keeps the audio that Home Assistant produced, so you can listen to it.
The command to regenerate `requirements.lock` is in the header of `requirements.in`.
CI (`.github/workflows/docker.yml`) runs the unit tests, builds the image, pushes it to
GHCR (not for pull requests), and smoke-tests the pushed image.

## Troubleshooting

**NVIDIA Container Toolkit (Linux).** `docker run --rm --gpus all ubuntu nvidia-smi` must
work first. If it doesn't: install the toolkit, run
`sudo nvidia-ctk runtime configure --runtime=docker`, then `sudo systemctl restart docker`.

**WSL 2 / Docker Desktop.** Use the WSL 2 backend and a current Windows NVIDIA driver
(≥ 570). Never install a Linux NVIDIA driver inside WSL. Run `wsl --update`.

**`DEVICE=cuda but torch.cuda.is_available() is False`.** The container has no GPU.
Check the `deploy.resources.reservations.devices` block (or `--gpus all`), the toolkit,
and the driver. A warning `No native kernels for … (sm_XX)` means torch has to JIT
its kernels for that GPU; if the CUDA check right after it fails, the GPU is not supported.

**Out of VRAM.** The service needs about 1 GB resident and about 2.5 GB peak. Keep
`RVC_CONCURRENCY=1`, set `PIPER_DEVICE=cpu`, and close other GPU applications.

**Running as non-root.** The image runs as root so that a bind-mounted `./models` works
without extra steps. To run unprivileged, run `sudo chown -R 1000:1000 models` and add
`user: "1000:1000"` to the service in `docker-compose.yml`.

**Hugging Face download fails.** Check connectivity, set `HF_TOKEN` if you are rate-limited,
or pre-populate `./models` and use absolute `PIPER_MODEL`/`RVC_MODEL_FILE`/`RVC_INDEX_FILE`.
Cached starts need no network.

**Model/index mismatch.** "`index … has dimension 256, an RVC v2 model needs 768`" means
the `.index` belongs to another model. Fix `RVC_INDEX_FILE`, or set `RVC_INDEX_RATE=0`.

**Upstream errors (`SOURCE=wyoming`).** "Cannot connect to upstream" means the
host/port is wrong or the upstream is not up yet (with compose, start it first).
"not offered by upstream" means `UPSTREAM_VOICE` isn't one of the upstream's voices;
the error lists them. The upstream must send 16-bit PCM.

**Bad pitch.** Adjust `RVC_PITCH` in steps of 2–3 semitones (per request: `"pitch": 3`)
and listen. No single value is right for every source voice. For a male source voice
and a female target, start around +6…+9.

**Metallic or robotic output.** Lower `RVC_INDEX_RATE` (0.3–0.5), raise `RVC_PROTECT`
toward 0.5, and keep the pitch shift small.

**Artifacts between sentences.** Each sentence is converted on its own in `sentence` mode.
If joins sound uneven, try `SENTENCE_SILENCE_MS=80`, or `RVC_MODE=whole` (one pass with
full context, at the cost of a later first audio).

**CPU fallback.** `DEVICE=cpu` works for debugging, but RVC on the CPU runs at roughly
real time or slower.

**Slow first start with Piper on the GPU.** onnxruntime-gpu 1.26 has no native kernels
for the newest GPUs (RTX 50xx / sm_120), so the driver JIT-compiles them once (about 1
minute). The result is cached in `/models/cuda-cache`, and later starts take seconds.
`PIPER_DEVICE=cpu` avoids it; for very short sentences CPU Piper is about as fast.

**Windows curl returns 422 or garbled accents.** Use `curl.exe` with the `\"` escapes
shown above, or `scripts/test_tts.py`.

## Known limitations

* Only the `rmvpe` pitch extractor and ContentVec-based models; the RefineGAN vocoder is not supported.
* Upstream sources must send 16-bit PCM (every known Wyoming TTS does).
* x86-64 Linux image with NVIDIA only; the image is about 7 GB because the PyTorch wheels bundle the CUDA runtime.
* Home Assistant interoperability is verified with HA's integration code in a test harness.
  Playback on a real speaker or satellite still needs a manual check.

## Alternatives

* [TextyMcSpeechy](https://github.com/domesticatedviking/TextyMcSpeechy) uses RVC once to
  build a dataset and trains a native Piper voice from it. That adds no runtime cost and
  runs on a Raspberry Pi, but each voice takes a training run. wyoming-rvc is the opposite
  trade-off: any RVC model instantly, at the cost of a GPU.
* [wyoming_openai](https://github.com/roryeckel/wyoming_openai) proxies Wyoming to
  OpenAI-compatible TTS servers (for example AllTalk, which also has RVC). It is more
  general but heavier, and has no resident RVC layer of its own.

## Licensing

This repository is **GPL-3.0-or-later**, because it links Piper (GPL-3.0) in-process.
It vendors minimal inference code from [Applio](https://github.com/IAHispano/Applio)
(MIT) and a mel filterbank adapted from librosa (ISC); see `THIRD_PARTY_LICENSES.md`.

**No model weights are included** in the repository or the image; they are downloaded
at runtime from their original locations. TetoTalk (`Slichi/KasaneTeto`) declares
`openrail`. Kasane Teto is a character/voicebank by TWINDRILL with its own terms. The
default Piper voice is trained on LJ Speech (public domain). Check the license of any voice you use, and do not
use voice conversion to impersonate real people.
