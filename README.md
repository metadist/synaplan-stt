# Synaplan STT

> **Optional companion** to [Synaplan](https://github.com/metadist/synaplan): a speech-to-text server for installations that want German transcription on their own hardware.
>
> **Image**: [`ghcr.io/metadist/synaplan-stt`](https://github.com/metadist/synaplan-stt/pkgs/container/synaplan-stt) &nbsp;|&nbsp; **Helm chart**: `stt` in [metadist/synaplan-charts](https://github.com/metadist/synaplan-charts) &nbsp;|&nbsp; **Main app**: [github.com/metadist/synaplan](https://github.com/metadist/synaplan)

[whisper.cpp](https://github.com/ggml-org/whisper.cpp) `whisper-server` with a German fine-tune of Whisper large-v3-turbo and the Silero voice activity detection model baked in. It exposes the OpenAI-compatible `POST /v1/audio/transcriptions` endpoint.

Nothing is downloaded when the container starts. The model is converted and quantized while the image is built, and every input of that build is pinned, so a mirrored image runs air-gapped.

## Variants

The image is `linux/amd64` and comes in two variants. There is no plain `latest` tag; pick one:

| Tag | Runs on | Size |
| --- | ------- | ---- |
| `ghcr.io/metadist/synaplan-stt:<version>-cuda` | NVIDIA GPU, about 1.6 GB of GPU memory. Needs the NVIDIA driver on the host and the GPU exposed to the container (NVIDIA Container Toolkit, device plugin or GPU operator). | 4.3 GB |
| `ghcr.io/metadist/synaplan-stt:<version>-cpu` | Any x86-64 host | 1.9 GB |

`<version>` is a release such as `0.1.0`, the moving `0.1`, or `latest` for the current `main`.

The cuda variant does not start without a GPU: `libcuda.so.1` comes from the host driver. Use the cpu variant there.

## Why a separate service?

Synaplan ships a whisper.cpp binary in its own image, which is enough for short voice notes on CPU. A dedicated server pays off when transcription should run on a GPU:

- A GPU workload has its own lifecycle: node pool, taints, driver, rollouts. It should not be tied to the web application's pods.
- One server can serve several Synaplan instances.
- The model (874 MB) stays out of the main image and is upgraded on its own.

## Quick Start

CPU, on any x86-64 machine with Docker:

```bash
docker run -d --name synaplan-stt -p 127.0.0.1:8080:8080 \
  -e STT_THREADS=8 \
  ghcr.io/metadist/synaplan-stt:latest-cpu
```

GPU:

```bash
docker run -d --name synaplan-stt --gpus all -p 127.0.0.1:8080:8080 \
  ghcr.io/metadist/synaplan-stt:latest-cuda
```

Or with the compose file from this repository:

```bash
git clone https://github.com/metadist/synaplan-stt.git
cd synaplan-stt
docker compose --profile cpu up -d   # or: --profile gpu
```

Verify. The server answers once the model is loaded, which takes a few seconds:

```bash
curl http://127.0.0.1:8080/health
# {"status":"ok"}
```

Transcribe a file. Any audio format ffmpeg reads is accepted:

```bash
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@sample.wav -F language=de
```

The container runs as UID 65532 and writes only to `/tmp`, so it also works with a read-only root filesystem (`--read-only --tmpfs /tmp`).

## Kubernetes

The `stt` chart in [metadist/synaplan-charts](https://github.com/metadist/synaplan-charts) deploys this image: Deployment and Service, the GPU request for the cuda variant, non-root with a read-only root filesystem, an optional NetworkPolicy and a `helm test`. The chart's `appVersion` is the image version; it appends `-cuda` or `-cpu`.

## Use with Synaplan

Synaplan is meant to reach the server through `WHISPER_SERVER_URL` (in the Helm chart: `speech.whisperServerUrl`), for example `http://stt:8080`. That server mode is not part of a Synaplan release yet (checked against 5.4.0), which until then keeps using the whisper.cpp binary in its own pod.

Because the endpoint follows the OpenAI transcription API, any client for that API works against it today.

## API Reference

### `GET /health`

Returns `{"status":"ok"}` once the model is loaded.

### `POST /v1/audio/transcriptions`

Multipart form upload, compatible with the OpenAI transcription API.

| Field | Required | Description |
| ----- | -------- | ----------- |
| `file` | yes | The audio file. Converted with ffmpeg, so wav, mp3, ogg, flac, m4a, webm and others work. |
| `language` | no | Language code. Defaults to `STT_LANGUAGE`; `auto` detects it. |
| `prompt` | no | Text that primes the decoder, for example names and terms that occur in the audio. |
| `temperature` | no | Sampling temperature. |
| `response_format` | no | `json` (default), `verbose_json` or `text`. |

`verbose_json` carries `start`, `end`, `avg_logprob`, `no_speech_prob` and `words` for each segment.

One server process decodes one request at a time. Requests that arrive while it is busy wait. For parallel load run more replicas.

## Configuration

Environment variables:

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `STT_LANGUAGE` | `de` | Language when a request does not name one. `auto` detects it. |
| `STT_THREADS` | `4` | CPU threads. For the cpu variant set it to the number of cores the container may use. |
| `STT_VAD` | `true` | Voice activity detection: skips silence before decoding. |
| `STT_PORT` | `8080` | Listen port. |
| `STT_HOST` | `0.0.0.0` | Listen address. |
| `STT_MODEL` | `/models/ggml-large-v3-turbo-german-q8_0.bin` | Model file. |
| `STT_VAD_MODEL` | `/models/ggml-silero-v5.1.2.bin` | VAD model file. |
| `STT_INFERENCE_PATH` | `/v1/audio/transcriptions` | Path of the transcription endpoint. |
| `STT_TMP_DIR` | `/tmp` | Scratch directory for the ffmpeg conversion. |

Arguments after the image name are appended to the `whisper-server` command line, so any of its flags can be set, for example a beam size:

```bash
docker run -d -p 127.0.0.1:8080:8080 ghcr.io/metadist/synaplan-stt:latest-cpu --beam-size 5
```

`--help` after the image name lists them all.

## What is in the image

| Path | What | Source and licence |
| ---- | ---- | ------------------ |
| `/app/build/bin/whisper-server` | whisper.cpp v1.9.4 | [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp), MIT |
| `/models/ggml-large-v3-turbo-german-q8_0.bin` | Whisper large-v3-turbo, German fine-tune, quantized to q8_0 (874 MB) | [primeline/whisper-large-v3-turbo-german](https://huggingface.co/primeline/whisper-large-v3-turbo-german), Apache-2.0; base model OpenAI Whisper, MIT |
| `/models/ggml-silero-v5.1.2.bin` | Silero VAD | [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad), MIT |
| `/usr/share/doc/synaplan-stt/` | Licence texts and model cards of the above | |

The build takes the Transformers checkpoint from Hugging Face, converts it to ggml and quantizes it with whisper.cpp's own tools. The image therefore does not depend on a third-party ggml upload.

### Measured

FLEURS `de_de` dev set, 363 read utterances, 12.5 s on average, measured on 2026-10-04:

| Model | Variant | WER | Time per utterance | GPU memory |
| ----- | ------- | --- | ------------------ | ---------- |
| stock large-v3-turbo, f16 | cuda | 5.88 % | 0.17 s | 2.5 GB |
| German fine-tune, f16 | cuda | 5.36 % | 0.17 s | 2.5 GB |
| German fine-tune, q5_0 | cuda | 5.41 % | | |
| **German fine-tune, q8_0 (this image)** | **cuda** | **5.30 %** | **0.18 s** (about 70 times real time) | **1.6 GB** |
| German fine-tune, q8_0 (this image) | cpu, 8 threads | 3.65 % on the first 30 utterances | 5.8 s (about 2 times real time) | |

GPU: NVIDIA RTX PRO 6000 Blackwell. q8_0 is as accurate as f16 at half the size, which is why both variants ship the same file.

## Building

```bash
docker build --build-arg VARIANT=cpu  -t synaplan-stt:dev-cpu .
docker build --build-arg VARIANT=cuda -t synaplan-stt:dev-cuda .
```

The build downloads the 1.6 GB checkpoint and the PyTorch CPU wheel for the conversion stage, and the cuda base image alone is 3.7 GB. It needs a lot of free disk space: the CI job clears unused toolchains from the runner before it starts.

### Pinned inputs

Every input is pinned in the `Dockerfile`:

| Input | Pinned by |
| ----- | --------- |
| whisper.cpp base images (cuda and cpu), Python image, Dockerfile frontend | image digest |
| Model and VAD model on Hugging Face | repository revision; the VAD file also by sha256 |
| Conversion script, mel filters, licence texts | commit in the URL plus `ADD --checksum` |
| Python packages of the conversion stage, including transitive ones | `requirements.txt` with hashes, installed with `--require-hashes` |

To change a Python dependency, edit `requirements.in` and regenerate `requirements.txt` with the command in its header.

To move to another whisper.cpp release, change `WHISPER_CPP_REF` together with both base image digests and the checksums of the files fetched from that commit. Then compare the sha256 of the model that CI prints with the previous one, and measure again if it changed.

## Releases

GitHub Actions (`.github/workflows/ci.yml`) builds both variants:

| Trigger | Result |
| ------- | ------ |
| Pull request | Builds both variants. Starts the cpu image the way the Helm chart runs it and waits for `/health`. Pushes nothing. |
| Push to `main` | `latest-cuda`, `latest-cpu` |
| Tag `v1.2.3` | `1.2.3-cuda`, `1.2.3-cpu`, and the moving `1.2-cuda`, `1.2-cpu` |

To release, tag `main` with `v<version>`. Then set `appVersion` of the `stt` chart in [metadist/synaplan-charts](https://github.com/metadist/synaplan-charts) to that version.

## Related

- [Main app: synaplan](https://github.com/metadist/synaplan)
- [Helm charts: synaplan-charts](https://github.com/metadist/synaplan-charts)
- [Text-to-speech companion: synaplan-tts](https://github.com/metadist/synaplan-tts)

## License

Apache License 2.0 for the contents of this repository. The image additionally contains whisper.cpp (MIT) and the models listed above under their own licences; the texts are in `/usr/share/doc/synaplan-stt/`.
