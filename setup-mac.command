#!/bin/bash
set -Eeuo pipefail

REPO_URL="https://github.com/DevDeepakBhattarai/speaker-diaziation.git"
INSTALL_DIR="${SPEAKER_DIARIZATION_HOME:-$HOME/SpeakerDiarization}"
PYTHON_VERSION="3.12"

finish_setup() {
  status=$?
  trap - EXIT
  if [[ $status -ne 0 ]]; then
    echo
    echo "Setup stopped because something failed."
    echo "Copy the error above and send it to Deepak."
    echo
    if [[ -t 0 ]]; then
      read -r -p "Press Return to close this window..." _ || true
    fi
  fi
  exit "$status"
}
trap finish_setup EXIT

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup is for macOS only."
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This project requires an Apple Silicon Mac (M1 or newer)."
  exit 1
fi

clear_quarantine() {
  command -v xattr >/dev/null 2>&1 || return 0
  for path in "$@"; do
    [[ -e "$path" ]] || continue
    xattr -dr com.apple.quarantine "$path" 2>/dev/null || true
  done
}

load_homebrew() {
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
}

load_homebrew
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is not installed. Installing it now..."
  echo "macOS may ask for your login password. This is normal for the Homebrew installer."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  load_homebrew
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew installation finished, but brew is still not available."
  exit 1
fi

packages=(git uv)
if brew info ffmpeg@7 >/dev/null 2>&1; then
  packages+=(ffmpeg@7)
else
  packages+=(ffmpeg)
fi

echo
echo "Installing required tools: ${packages[*]}"
brew install "${packages[@]}"

if brew list --versions ffmpeg@7 >/dev/null 2>&1; then
  FFMPEG_PREFIX="$(brew --prefix ffmpeg@7)"
else
  FFMPEG_PREFIX="$(brew --prefix ffmpeg)"
fi
export PATH="$FFMPEG_PREFIX/bin:$PATH"
export DYLD_LIBRARY_PATH="$FFMPEG_PREFIX/lib:${DYLD_LIBRARY_PATH:-}"

encoders="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
if [[ "$encoders" != *"h264_videotoolbox"* ]]; then
  echo "FFmpeg does not expose the h264_videotoolbox hardware encoder."
  exit 1
fi
hwaccels="$(ffmpeg -hide_banner -hwaccels 2>/dev/null)"
if [[ "$hwaccels" != *"videotoolbox"* ]]; then
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
  echo
  echo "Downloading Speaker Diarization..."
  git clone --branch main --single-branch "$REPO_URL" "$INSTALL_DIR"
fi

git -C "$INSTALL_DIR" fetch origin refs/heads/main:refs/remotes/origin/main
if git -C "$INSTALL_DIR" show-ref --verify --quiet refs/heads/main; then
  git -C "$INSTALL_DIR" switch main
  git -C "$INSTALL_DIR" merge --ff-only origin/main
else
  git -C "$INSTALL_DIR" switch -c main refs/remotes/origin/main
fi

cd "$INSTALL_DIR"
clear_quarantine "$INSTALL_DIR/setup-mac.command" "$INSTALL_DIR/start-mac.command"
chmod +x setup-mac.command start-mac.command

echo
echo "Installing Python $PYTHON_VERSION and project dependencies..."
uv python install "$PYTHON_VERSION"
uv sync --locked --python "$PYTHON_VERSION"

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
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

MODEL_ID = "pyannote/speaker-diarization-community-1"
load_dotenv(dotenv_path=".env", override=True)
token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    print("Hugging Face token is missing.", file=sys.stderr)
    raise SystemExit(10)

api = HfApi()
try:
    account = api.whoami(token=token)
except HfHubHTTPError as exc:
    status = getattr(exc.response, "status_code", None)
    if status == 401:
        print("Hugging Face rejected this token. Create a new read token and try again.", file=sys.stderr)
        raise SystemExit(11)
    print(f"Could not validate the Hugging Face token: {exc}", file=sys.stderr)
    raise SystemExit(12)

username = account.get("name") or account.get("fullname") or "unknown account"
print(f"Hugging Face token is valid for account: {username}")

try:
    hf_hub_download(
        repo_id=MODEL_ID,
        filename="config.yaml",
        token=token,
        force_download=True,
    )
except GatedRepoError:
    print(
        f"The token is valid, but its Hugging Face account does not have access to {MODEL_ID}.\n"
        f"Open https://huggingface.co/{MODEL_ID}, accept/request access while signed into the same account, "
        "then try this token again.",
        file=sys.stderr,
    )
    raise SystemExit(13)
except HfHubHTTPError as exc:
    status = getattr(exc.response, "status_code", None)
    print(
        f"The token is valid, but Hugging Face returned HTTP {status or 'error'} while downloading "
        f"{MODEL_ID}/config.yaml: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(14)

print("Authenticated Pyannote model download is ready.")
PY
}

echo
echo "Checking Hugging Face model access..."
if [[ -f .env ]] && grep -qE '^HF_TOKEN=hf_' .env && validate_hf_access; then
  echo "Existing Hugging Face access is valid."
else
  echo
  echo "A Hugging Face read token with Pyannote model access is required."
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
    echo "The validation message above explains what Hugging Face rejected. Fix that issue and try again."
  done
fi

echo
echo "Creating Desktop launchers..."
mkdir -p "$HOME/Desktop"
cp "$INSTALL_DIR/start-mac.command" "$HOME/Desktop/Start Speaker Diarization.command"
cp "$INSTALL_DIR/setup-mac.command" "$HOME/Desktop/Update Speaker Diarization.command"
chmod +x "$HOME/Desktop/Start Speaker Diarization.command" "$HOME/Desktop/Update Speaker Diarization.command"
clear_quarantine \
  "$HOME/Desktop/Start Speaker Diarization.command" \
  "$HOME/Desktop/Update Speaker Diarization.command"

echo
echo "Setup is complete."
echo "From now on, double-click 'Start Speaker Diarization.command' on the Desktop."
echo "Use 'Update Speaker Diarization.command' to update or repair the install."
echo
if [[ -t 0 ]]; then
  read -r -p "Press Return to close this window..." _
fi
trap - EXIT
