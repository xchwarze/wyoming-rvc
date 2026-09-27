# Wyoming RVC

Speaks with an [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion)
voice: Piper synthesizes the text, then a resident RVC model changes its timbre
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
- `device: auto` uses an NVIDIA GPU if the add-on can see one, otherwise the CPU.
  **Home Assistant OS does not give add-ons access to NVIDIA GPUs**, so on HA OS the
  add-on runs on the CPU. That works, but with noticeably higher latency than on a GPU.
- For the lowest latency (about 200 ms to first audio), run the Docker image on a machine
  with an NVIDIA GPU and add it to Home Assistant with **Wyoming Protocol → host:10200**.
  See the [project README](https://github.com/xchwarze/wyoming-rvc).

## Options

| Option | Default | Meaning |
|---|---|---|
| `device` | `auto` | `auto`, `cuda` or `cpu` |
| `piper_voice` | `en_US-ljspeech-high` | Any [Piper voice](https://huggingface.co/rhasspy/piper-voices) |
| `voice_language` | from `piper_voice` | Language Home Assistant lists the voices under (`es_MX-claude-high` → `es`); set it only to override |
| `warmup_text` | `System ready.` | Phrase synthesized at startup, in the voice's language |
| `rvc_repo_id` | `Slichi/KasaneTeto` | Set to another Hugging Face RVC repo (`.pth`, `.index`, or a `.zip`) to serve that single voice instead of the built-in ones |
| `rvc_model_file` / `rvc_index_file` | – | Pick a file when the repo contains several |
| `rvc_pitch` | `0` | Pitch shift in semitones (−24…24) |
| `rvc_index_rate` | `0.6` | How strongly to match the model's timbre (0…1) |
| `rvc_protect` | `0.33` | Protects consonants and breaths (0…0.5) |
| `rvc_mode` | `sentence` | `sentence` (fastest first audio) or `whole` |
| `voice_name` | `teto` | Voice id of your own single voice (changing it also replaces the built-in voices) |
| `default_voice` | – | Voice used when Home Assistant names none (with `voices.yaml`) |
| `rvc_max_loaded_models` | `1` | RVC voices kept in VRAM; the least recently used is unloaded |
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

Other settings from the README can be passed as environment variables when running
the Docker image directly.

## License notes

The add-on downloads model weights at runtime from their original locations; check
the license of any voice you use. See the project's `THIRD_PARTY_LICENSES.md`.
