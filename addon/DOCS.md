# Wyoming RVC (NVIDIA)

Speaks with an [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion)
voice: Piper synthesizes the text, then a resident RVC model changes its timbre.
Everything stays loaded in memory. It comes with two voices, **Kasane Teto** (default) and
**Hatsune Miku**: pick one per assistant in **Settings → Voice assistants → your assistant →
Text-to-speech → Voice**. The voices are listed under the language of `piper_voice`, so pick
a Piper voice in your assistant's language.

## Setup

1. Start the add-on. The first start downloads about 1.3 GB of models into the add-on's
   data folder; later starts reuse them.
2. Home Assistant discovers it automatically: **Settings → Devices & Services** shows a
   new **Wyoming Protocol** entry. Click **Configure**.
3. In **Settings → Voice assistants**, pick **Wyoming RVC** as the text-to-speech engine.

## Hardware

- **amd64 only.**
- This profile uses the NVIDIA CUDA image. `device: auto` uses an NVIDIA GPU if the
  host exposes the required runtime, driver libraries and devices to the container;
  otherwise it uses the CPU. `device: cuda` refuses to start without a usable GPU.
- **Official Home Assistant OS does not provide CUDA/ROCm compute access to add-ons.**
  Choose **Wyoming RVC (CPU)** there: it has the same voices without GPU libraries.
  Installing a GPU profile does not install a host driver or enable GPU passthrough.
  Proposed [HAOS GPU access](https://github.com/home-assistant/architecture/discussions/1568)
  is not a released feature.
- **Wyoming RVC (AMD ROCm)** is a separate, experimental profile for a compatible
  Linux host with AMD compute support; it cannot use the NVIDIA image.
- For the lowest latency (about 200 ms to first audio), run the Docker image on a machine
  with an NVIDIA GPU (or the ROCm image on a Linux machine with an AMD GPU) and add it to
  Home Assistant with **Wyoming Protocol → host:10200**.
  See the [project README](https://github.com/xchwarze/wyoming-rvc).

## Profiles, data and migration

The three store entries use different images, not a runtime image selector. Changing
`device` never changes the image or its download size. The existing add-on is now
named **Wyoming RVC (NVIDIA)**, with the same `wyoming_rvc` slug, image, options,
`/data` model cache and `/addon_configs/<id>_wyoming_rvc/` configuration folder.

CPU and AMD have separate slugs, configuration folders and private `/data` caches.
To change profile:

1. Back up the existing add-on and keep it installed while setting up the new one.
2. Install the desired profile and copy the voice options you changed. Use the new
   profile's documented `device` setting instead of copying a GPU setting to CPU.
3. If present, copy `voices.yaml` and any local files it references to the new public
   config folder. Adjust absolute paths; private `/data` downloads are not shared
   and will download again. Paths outside the new add-on's mounts are not accessible.
4. Start the new profile, accept its Wyoming integration and select its TTS engine
   and voice in each assistant. Test speech before removing the old installation.

Public config folders are `/addon_configs/<id>_wyoming_rvc_cpu/` for CPU and
`/addon_configs/<id>_wyoming_rvc_rocm/` for AMD. `<id>` is the repository identifier
assigned by Supervisor. Inside each container its own public folder is `/config`.

Host port 10200 is disabled by default; Supervisor discovery does not need it. If
you expose Wyoming for an external client, use different host ports when running
profiles together. Wyoming has no authentication: expose it only on a trusted
network. The HTTP debug API (8080) is not published by these add-ons.

## Options

This is the shared option reference for all three profiles. Only `device` differs:
NVIDIA defaults to `auto`, CPU to `cpu` (`auto` or `cpu` only), and AMD to `cuda`
(PyTorch's device name for HIP). All other defaults below are shared.

| Option | Default | Meaning |
|---|---|---|
| `device` | `auto` | `auto`, `cuda` or `cpu` |
| `piper_voice` | `en_US-ljspeech-high` | Any [Piper voice](https://huggingface.co/rhasspy/piper-voices) |
| `voice_language` | from `piper_voice` | Language Home Assistant lists the voices under (`es_MX-claude-high` → `es`); set it only to override |
| `warmup_text` | `System ready.` | Phrase synthesized at startup, in the voice's language |
| `probe_text` | Borges line (Spanish) | Phrase used to validate upstream Wyoming TTS sources from `voices.yaml`; set it in a language those sources support. Empty uses the default |
| `rvc_repo_id` | `Slichi/KasaneTeto` | Set to another Hugging Face RVC repo (`.pth`, `.index`, or a `.zip`) to serve that single voice instead of the built-in ones |
| `rvc_model_file` / `rvc_index_file` | – | Pick a file when the repo contains several |
| `rvc_pitch` | `0` | Pitch shift in semitones (−24…24) |
| `rvc_index_rate` | `0.6` | How strongly to match the model's timbre (0…1) |
| `rvc_protect` | `0.33` | Protects consonants and breaths (0…0.5) |
| `rvc_mode` | `sentence` | `sentence` (fastest first audio) or `whole` |
| `voice_name` | `teto` | Voice id of your own single voice (changing it also replaces the built-in voices) |
| `default_voice` | – | Voice used when Home Assistant names none (with `voices.yaml`) |
| `rvc_max_loaded_models` | `1` | RVC voices kept in VRAM; the least recently used is unloaded |
| `idle_sleep_s` | `0` | Seconds without a request before the models are unloaded to free VRAM (reloaded on the next request, a few seconds slower); `0` never sleeps, `300` is a good start |
| `log_level` | `INFO` | Log verbosity |

## Several voices

Create `voices.yaml` in this app's config folder (`/addon_configs/<id>_wyoming_rvc/`,
reachable with the File editor or Samba apps) and restart the app. Each voice in it is
listed in Home Assistant instead of the built-in voices; `rvc_pitch`, `rvc_index_rate`,
`rvc_protect` and `voice_language` become the defaults for values a voice leaves out. Example:

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

Voices download at startup and load into VRAM on first use. After changing the voices,
reload the Wyoming integration (**Settings → Devices & Services → Wyoming → ⋮ → Reload**). Switching to a voice that is
not loaded adds one model load to that reply.

A voice can also have its own TTS source instead of `piper_voice`: add
`piper_voice: en_GB-northern_english_male-medium` for its own Piper voice, or
`upstream: "tcp://host:10200"` for any Wyoming TTS. See the README for every key.

Other settings from the README can be passed as environment variables when running
the Docker image directly.

## License notes

The add-on downloads model weights at runtime from their original locations; check
the license of any voice you use. See the project's `THIRD_PARTY_LICENSES.md`.
