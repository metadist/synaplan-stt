#!/bin/bash
# Starts whisper-server from STT_* environment variables. Arguments passed to
# the container are appended, so any whisper-server flag can still be set.
set -euo pipefail

args=(
  --host "${STT_HOST}"
  --port "${STT_PORT}"
  --model "${STT_MODEL}"
  --language "${STT_LANGUAGE}"
  --threads "${STT_THREADS}"
  --inference-path "${STT_INFERENCE_PATH}"
  --convert
  --tmp-dir "${STT_TMP_DIR}"
)

if [ "${STT_VAD}" = "true" ]; then
  args+=(--vad --vad-model "${STT_VAD_MODEL}")
fi

exec /app/build/bin/whisper-server "${args[@]}" "$@"
