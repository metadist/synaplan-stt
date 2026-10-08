# syntax=docker/dockerfile:1
#
# synaplan-stt: whisper.cpp `whisper-server` with the German Whisper
# large-v3-turbo fine-tune (q8_0) and the Silero VAD model baked in.
#
#   docker build --build-arg VARIANT=cuda -t synaplan-stt:dev-cuda images/stt
#   docker build --build-arg VARIANT=cpu  -t synaplan-stt:dev-cpu  images/stt
#
# The cuda variant links libcuda.so.1 from the host driver: it starts only on a
# node with an NVIDIA GPU exposed to the container. Use the cpu variant elsewhere.

ARG VARIANT=cuda

# whisper.cpp v1.9.4 (commit 927cfce3). Upstream builds one image per commit;
# pinned by digest.
ARG WHISPER_CPP_REF=927cfce34f31707e17f2bff35c349632fb9e2c3a
ARG BASE_CUDA=ghcr.io/ggml-org/whisper.cpp:main-cuda-927cfce34f31707e17f2bff35c349632fb9e2c3a@sha256:e78e2a3cba6013df9b016ba68645adbf47e1a3f24d356cc308b6ab757f6c1088
ARG BASE_CPU=ghcr.io/ggml-org/whisper.cpp:main-927cfce34f31707e17f2bff35c349632fb9e2c3a@sha256:09f8f3bace5bcd98b9c7406f78c36bd98bdcc084438ce36f660ead1b776541d9
ARG PYTHON_IMAGE=python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d

ARG MODEL_REPO=primeline/whisper-large-v3-turbo-german
ARG MODEL_REVISION=9e7012dad9247f07553aac8a99c4d50ad7c3d36d
ARG OPENAI_WHISPER_REF=86098128c0b4f24f0e2aa2994de830614b474227
ARG VAD_REPO=ggml-org/whisper-vad
ARG VAD_REVISION=9ffd54a1e1ee413ddf265af9913beaf518d1639b
ARG VAD_SHA256=29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf

# ---------------------------------------------------------------------------
# Hugging Face (Transformers) checkpoint -> ggml f16
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS convert
ARG WHISPER_CPP_REF
ARG OPENAI_WHISPER_REF
ARG MODEL_REPO
ARG MODEL_REVISION
ARG VAD_REPO
ARG VAD_REVISION
ARG VAD_SHA256
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HUB_DISABLE_TELEMETRY=1

# Fully pinned, including transitive wheels. linux/amd64, Python 3.12.
# See requirements.in for how to regenerate requirements.txt.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

WORKDIR /work

ADD --checksum=sha256:9cc282dfcd9a24da03ffa1b4123b5508ee935e19e369c33520b1a73f89440094 \
    https://raw.githubusercontent.com/ggml-org/whisper.cpp/${WHISPER_CPP_REF}/models/convert-h5-to-ggml.py \
    convert-h5-to-ggml.py
ADD --checksum=sha256:7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c \
    https://raw.githubusercontent.com/openai/whisper/${OPENAI_WHISPER_REF}/whisper/assets/mel_filters.npz \
    openai-whisper/whisper/assets/mel_filters.npz
ADD --chmod=0644 --checksum=sha256:94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d \
    https://raw.githubusercontent.com/ggml-org/whisper.cpp/${WHISPER_CPP_REF}/LICENSE \
    out/licenses/whisper.cpp-LICENSE
ADD --chmod=0644 --checksum=sha256:b5d65a59060e68c4ff940e1eddfa6f94b2d68fdf58ed7f4dd57721c997e35e9d \
    https://raw.githubusercontent.com/openai/whisper/${OPENAI_WHISPER_REF}/LICENSE \
    out/licenses/openai-whisper-LICENSE

# The fine-tune ships bfloat16 weights, which numpy cannot hold: cast to
# float32 first. Weight matrices are still written as f16.
RUN grep -q 'list_vars\[src\]\.squeeze()\.numpy()' convert-h5-to-ggml.py \
 && sed -i 's/list_vars\[src\]\.squeeze()\.numpy()/list_vars[src].float().squeeze().numpy()/' convert-h5-to-ggml.py

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_REPO}', revision='${MODEL_REVISION}', local_dir='/work/hf', allow_patterns=['*.json', '*.safetensors', '*.txt', 'README.md'])" \
 && python convert-h5-to-ggml.py /work/hf /work/openai-whisper /work/out \
 && cp /work/hf/README.md out/licenses/whisper-large-v3-turbo-german-README.md \
 && rm -rf /work/hf

RUN python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('${VAD_REPO}', 'ggml-silero-v5.1.2.bin', revision='${VAD_REVISION}'), '/work/out/ggml-silero-v5.1.2.bin'); shutil.copy(hf_hub_download('${VAD_REPO}', 'README.md', revision='${VAD_REVISION}'), '/work/out/licenses/whisper-vad-README.md')" \
 && echo "${VAD_SHA256}  /work/out/ggml-silero-v5.1.2.bin" | sha256sum -c -

# ---------------------------------------------------------------------------
# ggml f16 -> q8_0 (same accuracy as f16 on German, half the size)
# ---------------------------------------------------------------------------
FROM ${BASE_CPU} AS quantize
COPY --from=convert /work/out/ggml-model.bin /work/ggml-model.bin
RUN /app/build/bin/whisper-quantize /work/ggml-model.bin /work/ggml-large-v3-turbo-german-q8_0.bin q8_0 \
 && rm /work/ggml-model.bin

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM ${BASE_CUDA} AS runtime-cuda
FROM ${BASE_CPU} AS runtime-cpu

FROM runtime-${VARIANT}
ARG VARIANT
ARG VERSION=0.0.0-dev
ARG WHISPER_CPP_REF
ARG MODEL_REPO
ARG MODEL_REVISION

LABEL org.opencontainers.image.title="synaplan-stt" \
      org.opencontainers.image.description="whisper.cpp whisper-server with the German Whisper large-v3-turbo model (q8_0) for Synaplan" \
      org.opencontainers.image.source="https://github.com/metadist/synaplan-charts" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      com.metadist.stt.variant="${VARIANT}" \
      com.metadist.stt.whisper-cpp="${WHISPER_CPP_REF}" \
      com.metadist.stt.model="${MODEL_REPO}@${MODEL_REVISION} q8_0"

COPY --from=quantize /work/ggml-large-v3-turbo-german-q8_0.bin /models/ggml-large-v3-turbo-german-q8_0.bin
COPY --from=convert /work/out/ggml-silero-v5.1.2.bin /models/ggml-silero-v5.1.2.bin
COPY --from=convert /work/out/licenses/ /usr/share/doc/synaplan-stt/
COPY --chmod=0755 entrypoint.sh /usr/local/bin/synaplan-stt

ENV HOME=/tmp \
    CUDA_CACHE_PATH=/tmp/.nv \
    STT_HOST=0.0.0.0 \
    STT_PORT=8080 \
    STT_MODEL=/models/ggml-large-v3-turbo-german-q8_0.bin \
    STT_LANGUAGE=de \
    STT_THREADS=4 \
    STT_INFERENCE_PATH=/v1/audio/transcriptions \
    STT_TMP_DIR=/tmp \
    STT_VAD=false \
    STT_VAD_MODEL=/models/ggml-silero-v5.1.2.bin

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/synaplan-stt"]
CMD []
