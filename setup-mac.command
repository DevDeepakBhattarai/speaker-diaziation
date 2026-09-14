#!/bin/bash
set -Eeuo pipefail
REPO_URL="https://github.com/DevDeepakBhattarai/speaker-diaziation.git"
STABLE_REF="${SPEAKER_DIARIZATION_REF:-main}"
BOOTSTRAP_REF="feat/mac-launchers"
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
elif [[ -e "$INSTALL_DIR" ]]; then
  echo "Install path exists and is not a Git checkout: $INSTALL_DIR"
  exit 1
else
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

fetch_ref() {
  git -C "$INSTALL_DIR" fetch origin "refs/heads/$1:refs/remotes/origin/$1"
}

ref_has_mac_launchers() {
  git -C "$INSTALL_DIR" cat-file -e "refs/remotes/origin/$1:setup-mac.command" 2>/dev/null &&
    git -C "$INSTALL_DIR" cat-file -e "refs/remotes/origin/$1:start-mac.command" 2>/dev/null
}

TARGET_REF="$STABLE_REF"
if ! fetch_ref "$TARGET_REF"; then
  if [[ -n "${SPEAKER_DIARIZATION_REF:-}" ]]; then
    echo "Could not fetch the requested branch: $TARGET_REF"
    exit 1
  fi
  TARGET_REF="$BOOTSTRAP_REF"
  fetch_ref "$TARGET_REF"
elif [[ -z "${SPEAKER_DIARIZATION_REF:-}" ]] && ! ref_has_mac_launchers "$TARGET_REF"; then
  # Before the Mac launcher change lands on main, bootstrap from the review branch.
  # Once main contains these files, the next setup run automatically migrates to main.
  TARGET_REF="$BOOTSTRAP_REF"
  fetch_ref "$TARGET_REF"
fi

if git -C "$INSTALL_DIR" show-ref --verify --quiet "refs/heads/$TARGET_REF"; then
  git -C "$INSTALL_DIR" switch "$TARGET_REF"
  git -C "$INSTALL_DIR" merge --ff-only "origin/$TARGET_REF"
else
  git -C "$INSTALL_DIR" switch -c "$TARGET_REF" "refs/remotes/origin/$TARGET_REF"
fi

cd "$INSTALL_DIR"
uv sync --locked

echo
echo "Checking the local Python/audio stack..."
uv run python - <<'PY'
from pyannote.audio import Pipeline
import torch
import torchcodec

print(f"PyTorch {torch.__version__} is ready. Diarization will use CPU on this Mac.")
PY

write_hf_token() {
  token="$1"
  tmp_env="$(mktemp)"
  if [[ -f .env ]]; then
    grep -vE '^HF_TOKEN=' .env > "$tmp_env" || true
  fi
  printf 'HF_TOKEN=%s\n' "$token" >> "$tmp_env"
  mv "$tmp_env" .env
}

validate_hf_access() {
  uv run python - <<'PY'
import os
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv(override=True)
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN is missing from .env")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
if pipeline is None:
    raise SystemExit(
        "Hugging Face could not load the Pyannote model with this token. "
        "Check the token and accept the model terms."
    )
print("Pyannote model access is ready.")
PY
}

echo
echo "Checking Hugging Face model access..."
if [[ -f .env ]] && grep -qE '^HF_TOKEN=hf_' .env && validate_hf_access; then
  echo "Existing Hugging Face access is valid."
else
  echo
  echo "A valid Hugging Face token with Pyannote model access is required."
  open "https://huggingface.co/pyannote/speaker-diarization-community-1"
  open "https://huggingface.co/settings/tokens"
  echo "Accept the model terms and create a read token, then return here."
  while true; do
    read -r -s -p "Paste the Hugging Face token: " HF_TOKEN
    echo
    if [[ "$HF_TOKEN" != hf_* ]]; then
      echo "That token does not look valid. Hugging Face tokens start with hf_."
      continue
    fi
    write_hf_token "$HF_TOKEN"
    if validate_hf_access; then
      break
    fi
    echo
    echo "That token could not access the model. Check the account/model terms and try again."
  done
fi

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
