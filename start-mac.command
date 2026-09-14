#!/bin/bash
set -Eeuo pipefail

INSTALL_DIR="${SPEAKER_DIARIZATION_HOME:-$HOME/SpeakerDiarization}"
URL="http://127.0.0.1:7860"
APP_PID=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" >/dev/null 2>&1; then
    kill "$APP_PID" >/dev/null 2>&1 || true
  fi
  if [[ $status -ne 0 ]]; then
    echo
    echo "Speaker Diarization stopped because something failed."
    echo "Copy the error above and send it to Deepak."
    echo
    read -r -p "Press Return to close this window..." _ || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher is for macOS only."
  exit 1
fi
if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  echo "Speaker Diarization is not installed at $INSTALL_DIR"
  echo "Run 'Update Speaker Diarization.command' first."
  exit 1
fi

if [[ -x /opt/homebrew/bin/brew ]]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
if ! command -v brew >/dev/null 2>&1 || ! command -v uv >/dev/null 2>&1; then
  echo "The setup is incomplete. Run 'Update Speaker Diarization.command' first."
  exit 1
fi

if brew info ffmpeg@7 >/dev/null 2>&1 && brew list --versions ffmpeg@7 >/dev/null 2>&1; then
  FFMPEG_PREFIX="$(brew --prefix ffmpeg@7)"
elif brew list --versions ffmpeg >/dev/null 2>&1; then
  FFMPEG_PREFIX="$(brew --prefix ffmpeg)"
else
  echo "FFmpeg is missing. Run 'Update Speaker Diarization.command' first."
  exit 1
fi
export PATH="$FFMPEG_PREFIX/bin:$PATH"
export DYLD_LIBRARY_PATH="$FFMPEG_PREFIX/lib:${DYLD_LIBRARY_PATH:-}"

cd "$INSTALL_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo "Finishing the Python setup..."
  uv sync --locked
fi

if curl -fsS "$URL" >/dev/null 2>&1; then
  open "$URL"
  trap - EXIT INT TERM
  exit 0
fi

echo "Starting Speaker Diarization..."
echo "Keep this Terminal window open while you use the app."
uv run python app.py &
APP_PID=$!

opened=0
for _ in {1..90}; do
  if ! kill -0 "$APP_PID" >/dev/null 2>&1; then
    wait "$APP_PID"
    exit $?
  fi
  if curl -fsS "$URL" >/dev/null 2>&1; then
    open "$URL"
    opened=1
    break
  fi
  sleep 1
done

if [[ $opened -ne 1 ]]; then
  echo "The app did not become ready at $URL."
  exit 1
fi

wait "$APP_PID"
