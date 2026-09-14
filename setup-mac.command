#!/bin/bash
set -Eeuo pipefail
REPO_URL="https://github.com/DevDeepakBhattarai/speaker-diaziation.git"
REPO_REF="${SPEAKER_DIARIZATION_REF:-feat/mac-launchers}"
INSTALL_DIR="${SPEAKER_DIARIZATION_HOME:-$HOME/SpeakerDiarization}"

finish_setup() {
  status=$?
  trap - EXIT
  if [[ $status -ne 0 ]]; then
    echo
    echo "Setup stopped because something failed."
    echo "Copy the error above and send it to Deepak."
    echo
    read -r -p "Press Return to close this window..." _ || true
  fi
  exit "$status"
}
trap finish_setup EXIT

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup file is for macOS only."
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This project currently requires an Apple Silicon Mac (M1 or newer)."
  exit 1
fi

if ! command -v brew >/dev/null 2>&1 && [[ ! -x /opt/homebrew/bin/brew ]]; then
  echo "Homebrew is required once. The official Homebrew page is opening now."
  open "https://brew.sh/"
  echo "Install Homebrew using the command shown on that page, then double-click this setup file again."
  read -r -p "Press Return to close this window..." _
  exit 0
fi
if [[ -x /opt/homebrew/bin/brew ]]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

packages=()
command -v git >/dev/null 2>&1 || packages+=(git)
command -v uv >/dev/null 2>&1 || packages+=(uv)
if ((${#packages[@]})); then
  brew install "${packages[@]}"
fi
if brew info ffmpeg@7 >/dev/null 2>&1; then
  brew list --versions ffmpeg@7 >/dev/null 2>&1 || brew install ffmpeg@7
  FFMPEG_PREFIX="$(brew --prefix ffmpeg@7)"
else
  brew list --versions ffmpeg >/dev/null 2>&1 || brew install ffmpeg
  FFMPEG_PREFIX="$(brew --prefix ffmpeg)"
fi
export PATH="$FFMPEG_PREFIX/bin:$PATH"
export DYLD_LIBRARY_PATH="$FFMPEG_PREFIX/lib:${DYLD_LIBRARY_PATH:-}"

if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_videotoolbox; then
  echo "FFmpeg does not expose the h264_videotoolbox hardware encoder."
  exit 1
fi
if ! ffmpeg -hide_banner -hwaccels 2>/dev/null | grep -q videotoolbox; then
  echo "FFmpeg does not expose VideoToolbox hardware decoding."
  exit 1
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
  if [[ -n "$(git -C "$INSTALL_DIR" status --porcelain)" ]]; then
    echo "The existing install has local source-code changes, so setup will not overwrite it."
    exit 1
  fi
  git -C "$INSTALL_DIR" fetch origin "$REPO_REF"
  if git -C "$INSTALL_DIR" show-ref --verify --quiet "refs/heads/$REPO_REF"; then
    git -C "$INSTALL_DIR" switch "$REPO_REF"
  else
    git -C "$INSTALL_DIR" switch -c "$REPO_REF" --track "origin/$REPO_REF"
  fi
  git -C "$INSTALL_DIR" pull --ff-only origin "$REPO_REF"
elif [[ -e "$INSTALL_DIR" ]]; then
  echo "Install path exists and is not a Git checkout: $INSTALL_DIR"
  exit 1
else
  git clone --branch "$REPO_REF" --single-branch "$REPO_URL" "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"
uv sync --locked


echo
if [[ -f .env ]] && grep -qE '^HF_TOKEN=hf_' .env; then
  echo "Existing Hugging Face token found."
else
  echo "Hugging Face access is required once for the diarization model."
  open "https://huggingface.co/pyannote/speaker-diarization-community-1"
  open "https://huggingface.co/settings/tokens"
  echo "Accept the model terms, create a read token, then return here."
  read -r -p "Press Return when those steps are done..." _
  read -r -s -p "Paste the Hugging Face token: " HF_TOKEN
  echo
  if [[ "$HF_TOKEN" != hf_* ]]; then
    echo "That token does not look valid. Hugging Face tokens start with hf_."
    exit 1
  fi
  tmp_env="$(mktemp)"
  if [[ -f .env ]]; then
    grep -vE '^HF_TOKEN=' .env > "$tmp_env" || true
  fi
  printf 'HF_TOKEN=%s
' "$HF_TOKEN" >> "$tmp_env"
  mv "$tmp_env" .env
fi

echo
echo "Checking Python and model access..."
uv run python - <<'PY'
import os
from dotenv import load_dotenv
from pyannote.audio import Pipeline
import torch
import torchcodec

load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN is missing from .env")
print(f"PyTorch {torch.__version__} is ready. Diarization will use CPU on this Mac.")
Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
print("Pyannote model access is ready.")
PY

echo
echo "Creating desktop launchers..."
mkdir -p "$HOME/Desktop"
cp "$INSTALL_DIR/start-mac.command" "$HOME/Desktop/Start Speaker Diarization.command"
cp "$INSTALL_DIR/setup-mac.command" "$HOME/Desktop/Update Speaker Diarization.command"
chmod +x "$INSTALL_DIR/setup-mac.command" "$INSTALL_DIR/start-mac.command"
chmod +x "$HOME/Desktop/Start Speaker Diarization.command" "$HOME/Desktop/Update Speaker Diarization.command"

echo
echo "Setup is complete."
echo "From now on, double-click 'Start Speaker Diarization.command' on the Desktop."
echo "Use 'Update Speaker Diarization.command' when you want to update or repair the install."
echo
read -r -p "Press Return to close this window..." _
trap - EXIT
