# Wyoming RVC

Speaks with an [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion)
voice: Piper synthesizes the text, then a resident RVC model changes its timbre
(default: Kasane Teto, TetoTalk). Everything stays loaded in memory.

## Setup

1. Start the add-on. The first start downloads about 1 GB of models into the add-on's
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
| `voice_language` | `en` | Language Home Assistant lists for the voice; match `piper_voice` |
| `warmup_text` | `System ready.` | Phrase synthesized at startup, in the voice's language |
| `rvc_repo_id` | `Slichi/KasaneTeto` | Hugging Face repo with the RVC model (`.pth`, `.index`, or a `.zip`) |
| `rvc_model_file` / `rvc_index_file` | – | Pick a file when the repo contains several |
| `rvc_pitch` | `0` | Pitch shift in semitones (−24…24) |
| `rvc_index_rate` | `0.6` | How strongly to match the model's timbre (0…1) |
| `rvc_protect` | `0.33` | Protects consonants and breaths (0…0.5) |
| `rvc_mode` | `sentence` | `sentence` (fastest first audio) or `whole` |
| `voice_name` | `teto` | Voice name shown in Home Assistant |
| `log_level` | `INFO` | Log verbosity |

Other settings from the README can be passed as environment variables when running
the Docker image directly.

## License notes

The add-on downloads model weights at runtime from their original locations; check
the license of any voice you use. See the project's `THIRD_PARTY_LICENSES.md`.
