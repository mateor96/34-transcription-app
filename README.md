# Nota.ai

A fully local, on-device audio transcription and speaker diarization web app for Apple Silicon Macs. No cloud, no subscriptions, no audio leaving your machine.

Built on [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for transcription and [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker separation.

---

## Why I built this

I tried a lot of transcription tools and never found one I was fully happy with. Notion's transcription came close — transcripts uploaded directly to Notion, no bot joining the call — but it lacks speaker recognition and the free tier runs out fast. Google Meet's Gemini note-taking has the same story. There are free tools out there, but every single one came with at least one trade-off: no speaker labels, cloud uploads, a bot sitting in your meeting, or a subscription wall.

The final straw was transcribing a regular phone call. Suddenly none of the meeting-focused tools applied and I had to go looking for something completely different. That inconsistency is why I built my own tool — one that handles any audio, keeps everything on your machine, and actually tells you who said what.

---

## Features

- **Transcription** via Whisper Large v3 Turbo — runs on the M-series GPU through Apple MLX
- **Speaker diarization** via pyannote community-1 — separates and labels each speaker
- **Live recording** — record directly from any audio input (mic, BlackHole, etc.)
- **Archive** — all past transcriptions saved locally in SQLite, accessible anytime
- **Speaker renaming** — rename Speaker 00 / Speaker 01 to real names; persists per transcript
- **Export** — download as TXT, Markdown, or PDF
- **Click-to-seek** — click any line in the transcript to jump to that moment in the audio

Everything runs at `localhost:8765`. No internet required after first setup.

---

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13+
- [uv](https://github.com/astral-sh/uv) — `brew install uv`
- [ffmpeg](https://ffmpeg.org) — `brew install ffmpeg`
- A [Hugging Face](https://huggingface.co) account (free)

---

## Setup

**1. Clone and install**

```bash
git clone https://github.com/mateor96/34-transcription-app.git
cd 34-transcription-app
uv sync
```

**2. Authenticate with Hugging Face**

```bash
uv run huggingface-cli login
```

Then visit these two pages while logged in and click **Agree** on each:

- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

> Your token needs **"Read access to public gated repositories"** enabled under token settings.

**3. Run**

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

The first transcription will download the Whisper model (~1.5 GB) and pyannote model (~30 MB). After that, everything runs offline.

---

## Recording calls (optional)

To record both sides of a call with headphones, install [BlackHole](https://github.com/ExistentialAudio/BlackHole) — a free virtual audio device for macOS.

**One-time setup (~10 min):**

1. Install BlackHole 2ch from [existential.audio](https://existential.audio/blackhole/)
2. Open **Audio MIDI Setup** (in `/Applications/Utilities/`)
3. Click **+** → **Create Multi-Output Device**
4. Check both your headphones and **BlackHole 2ch**
5. In Google Meet (or any call tool): set the speaker output to this Multi-Output Device

After that, select **BlackHole 2ch** as the audio source in the app's Record tab and hit Start. You'll hear the call normally through your headphones; the app captures everything in the background.

---

## Tech stack

| Layer | Choice |
|---|---|
| Transcription | `mlx-whisper` + `mlx-community/whisper-large-v3-turbo` |
| Diarization | `pyannote/speaker-diarization-community-1` |
| Backend | FastAPI + uvicorn + SSE |
| Storage | SQLite via `aiosqlite` (`~/.transcribe/archive.db`) |
| Frontend | Vanilla HTML/JS |
| Audio normalisation | ffmpeg → 16 kHz mono WAV |

---

## Notes

- Models are cached in `~/.cache/huggingface/hub`
- Transcripts are stored in `~/.transcribe/archive.db`
- Neither folder is tracked by git
