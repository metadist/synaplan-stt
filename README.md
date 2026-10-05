# synaplan-stt image

whisper.cpp `whisper-server` with the German Whisper large-v3-turbo fine-tune
baked in. Deployed by the [`stt` chart](../../charts/stt/); Synaplan reaches it
through `speech.whisperServerUrl`.

| Tag | Runs on |
| --- | ------- |
| `ghcr.io/metadist/synaplan-stt:<version>-cuda` | NVIDIA GPU (driver on the node, device plugin or GPU operator) |
| `ghcr.io/metadist/synaplan-stt:<version>-cpu` | Any x86-64 node |

## Contents

| Path | What | Source and licence |
| ---- | ---- | ------------------ |
| `/app/build/bin/whisper-server` | whisper.cpp v1.9.4 | [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp), MIT |
| `/models/ggml-large-v3-turbo-german-q8_0.bin` | Whisper large-v3-turbo, German fine-tune, q8_0 (874 MB) | [primeline/whisper-large-v3-turbo-german](https://huggingface.co/primeline/whisper-large-v3-turbo-german), Apache-2.0; base model OpenAI Whisper, MIT |
| `/models/ggml-silero-v5.1.2.bin` | Silero VAD | [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad), MIT |
| `/usr/share/doc/synaplan-stt/` | Licence texts and model cards | — |

Every source is pinned (commit, revision, digest or sha256) in the
`Dockerfile`. The build converts the Transformers checkpoint to ggml and
quantizes it, so the image does not depend on a third-party ggml upload.

## API

OpenAI-compatible `POST /v1/audio/transcriptions` (multipart `file`, optional
`language`, `prompt`, `temperature`, `response_format=json|verbose_json|text`).
`verbose_json` carries per segment `start`, `end`, `avg_logprob`,
`no_speech_prob` and `words`. `GET /health` answers `{"status":"ok"}` once the
model is loaded. Any audio format ffmpeg reads is accepted.

One server process decodes one request at a time.

## Configuration

| Variable | Default |
| -------- | ------- |
| `STT_LANGUAGE` | `de` (a request's `language` wins; `auto` detects) |
| `STT_THREADS` | `4` (set to the CPU limit for the cpu variant) |
| `STT_MODEL` | `/models/ggml-large-v3-turbo-german-q8_0.bin` |
| `STT_VAD` | `false` |
| `STT_PORT` | `8080` |

Arguments after the image name are passed to `whisper-server`.

## Measured (2026-10-04)

FLEURS de_de dev, 363 read utterances, average 12.5 s:

| Variant | WER | Time per utterance |
| ------- | --- | ------------------ |
| cuda, NVIDIA RTX PRO 6000 Blackwell | 5.30 % | 0.17 s (about 70× real time), 1.6 GB GPU memory |
| cpu, 8 threads | 3.65 % on the first 30 | 5.8 s (about 2× real time) |

Same set, same server: stock large-v3-turbo f16 5.88 %, German f16 5.36 %,
German q5_0 5.41 %.

## Build

```bash
docker build --build-arg VARIANT=cuda -t synaplan-stt:dev-cuda images/stt
docker run --rm --gpus all -p 8080:8080 synaplan-stt:dev-cuda
curl -s localhost:8080/v1/audio/transcriptions -F file=@sample.wav -F language=de
```

CI (`.github/workflows/stt-image.yaml`) builds both variants on pull requests,
publishes `0.0.0-dev.<commit>-<variant>` from `main` and `<version>-<variant>`
from a `stt-image-v<version>` tag.
