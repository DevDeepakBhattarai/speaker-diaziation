# Speaker Diarization Video Switcher

Creates a speaker-focused final video from either embedded video audio or a separate soundtrack and:

- two independent speaker camera videos, or
- one combined video containing the left and right speakers.

The script runs Pyannote directly against the selected local media/audio source,
without creating a separate diarization copy or applying custom speaker cleanup.
The default `community-1` model produces both regular and exclusive diarization.
Those model timestamps are stored untouched. A separate camera-only policy then
uses the exclusive timeline to debounce tiny cuts and apply long-silence
look-ahead before generating the FFmpeg switch timeline.

Set the Hugging Face token before running either mode:

```powershell
$env:HF_TOKEN = "hf_..."
```

## Input Modes

### Two separate speaker videos

This is the original mode. The first unique speaker detected in the soundtrack
maps to the first speaker video, and the second unique speaker maps to the
second speaker video.

```powershell
python main.py .\conversation.wav .\speaker_0.mp4 .\speaker_1.mp4 -o .\out.mp4
```

### One combined left/right video

This mode accepts one video containing both people. Its embedded audio can be
used directly, or a separate soundtrack can be selected in the Gradio app. The
video is divided vertically into two equal views:

- left edge to center = left speaker camera
- center to right edge = right speaker camera

Only the active speaker's half is rendered, at that half's own native pixel
density. A 3840x2160 combined recording produces a 1920x2160 output: every
source pixel of the visible speaker is kept, and nothing is scaled, stretched,
or padded with black bars. Choose whether the first unique speaker detected in
the soundtrack is the person on the left or the person on the right.

First detected speaker is on the left:

```powershell
python main.py .\conversation.wav .\both_speakers.mp4 `
  --mode split-video `
  --first-speaker-side left `
  -o .\out.mp4
```

First detected speaker is on the right:

```powershell
python main.py .\conversation.wav .\both_speakers.mp4 `
  --mode split-video `
  --first-speaker-side right `
  -o .\out.mp4
```

The original command remains unchanged because `separate-videos` is still the
default mode.

### One pipeline for both modes

Both modes reduce to the same thing before anything is rendered: **two camera
angles drawn on one shared canvas**. A combined recording contributes two crops
of a single input; two camera files contribute one uncropped angle each. From
that point on the camera timeline, the switched render, the Resolve camera
media, and the OTIO document all read the same render plan, so neither mode has
a code path of its own.

The canvas is chosen so nothing is ever scaled:

- **split-video** - the canvas is one native half of the source.
- **separate-videos** - the canvas is the per-axis maximum of the two cameras,
  so the larger camera stays native and a smaller one is padded onto it rather
  than being stretched or shrunk to fit.

## Requirements

- FFmpeg and FFprobe on `PATH`
- Python 3.10, 3.11, or 3.12
- A Hugging Face token with access to `pyannote/speaker-diarization-community-1`
- Windows/Linux NVIDIA: CUDA-enabled PyTorch and an FFmpeg build with `h264_nvenc`
- Apple Silicon macOS: regular PyTorch wheels and FFmpeg with Apple VideoToolbox
- On Windows, an FFmpeg 4-7 **shared** build for TorchCodec DLL loading

Install/update Python dependencies with uv:

```powershell
uv sync
```

The project pins Pyannote 4.0.7 with PyTorch 2.8 / CUDA 12.6-compatible wheels.
The diarization inference batch is kept at 1 so the full recording can be handled
on an 8 GiB GPU without requiring the entire media file to fit in VRAM.

On Windows, TorchCodec 0.7 requires FFmpeg 4-7 shared libraries. Install the
supported FFmpeg 7.1 shared build once with:

```powershell
winget install --id BtbN.FFmpeg.GPL.Shared.7.1 -e
```

The app automatically registers that WinGet DLL directory before importing
Pyannote, so it can coexist with a newer FFmpeg CLI used for rendering.

### Apple Silicon Mac setup

For a fresh Mac, open **Terminal** and paste this one command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/DevDeepakBhattarai/speaker-diaziation/main/setup-mac.command)"
```

This first-run command avoids macOS Gatekeeper quarantine on a browser-downloaded `.command` file. The setup script installs Homebrew when needed, then installs Git, `uv`, FFmpeg, Python 3.12, and the locked project dependencies. It clones or updates `main` in `~/SpeakerDiarization`, validates VideoToolbox, asks for a Hugging Face token, verifies the token account with the Hugging Face API, forces an authenticated download of the gated Pyannote `config.yaml`, and creates executable Desktop launchers.

After setup, double-click `Start Speaker Diarization.command` on the Desktop whenever the app is needed. Use `Update Speaker Diarization.command` to update or repair the installation. Setup removes the quarantine attribute from both Desktop launchers.

A `.command` file downloaded through Safari, Messages, AirDrop, or another quarantine-aware app can still be blocked before its code runs. macOS only treats a downloaded executable as fully verified when it has an Apple Developer ID signature and notarization. The Terminal bootstrap above avoids that first-run Gatekeeper path without requiring Git, Python, Homebrew, or executable permissions beforehand.

The pinned `torchcodec==0.7.0` package publishes a macOS wheel for Apple Silicon, not Intel Macs, so this setup intentionally requires an M1 or newer Mac. Diarization runs on CPU. FFmpeg auto-selects `h264_videotoolbox` for GPU-backed video encoding and `videotoolbox` for hardware video decoding. The app maps its CRF/CQ quality control to VideoToolbox's quality scale. MLX is not required.

## Gradio App

Start the local web app:

```powershell
.venv\Scripts\python.exe app.py
```

Then open the local URL printed by Gradio:

1. Select **Two separate speaker videos** or **One combined left/right video**.
2. Paste each file's full local Windows path. The browser does not upload the
   media and the app does not copy the source files into the output job.
3. Choose **Use audio from the video** or **Use a separate local audio file**.
4. In combined-video mode, choose whether the first detected speaker is on the
   left or right.
5. Adjust the **Camera switching** controls if needed. They affect only the
   rendered camera timeline. The raw Pyannote diarization is never merged,
   shortened, extended, or relabeled.

For two separate videos, embedded-audio mode uses the audio track from the first
speaker video. For a combined video, it uses that video's own audio track.

The app also serves completed outputs directly from the `outputs` directory
instead of making another Gradio cache copy. The output video is shown as a file
result rather than a browser preview so Gradio cannot silently transcode or
duplicate a very large render. Diarization reads the chosen source directly and
does not create a second full soundtrack copy.

### DaVinci Resolve project export

Enable **Also create a portable DaVinci Resolve project (.otioz)** before running
the job. The app then returns an additional downloadable project bundle with:

- an editable **V1 - Active Speaker** track containing every diarization cut,
- a continuous **A1 - Master Audio** track,
- two full-length, time-aligned, native-density camera angles,
- `speaker_segments.json`, `davinci_manifest.json`, and import instructions.

The same export is available from the CLI:

```powershell
python main.py .\conversation.wav .\both_speakers.mp4 `
  --mode split-video `
  --davinci-project .\speaker_edit.otioz `
  -o .\out.mp4
```

Import the downloaded file in DaVinci Resolve using **File > Import > Timeline**
and select `speaker_edit.otioz`. Keep the imported timeline frame rate when
Resolve asks. V1 remains editable for trimming or replacing camera cuts, while
A1 remains a single continuous soundtrack.

Enabling the export adds no extra decoding and no extra reads of the source
files. The camera angles come out of the *same* FFmpeg invocation as the
switched video, so they are frame-identical to what the render used. It does add
two more encodes and the bundle itself is large, so leave the checkbox disabled
when only the rendered video and JSON timeline are needed.

## One FFmpeg Pass

A job that produces everything - the switched video, both Resolve camera angles,
and the master audio - runs as a single FFmpeg invocation. Each source is
decoded once, split inside the filter graph, and fed to every output at the same
time:

```text
[0:v]setpts=PTS-STARTPTS,split=2[src0][src1];
[src0]crop=1920:2160:0:0,setsar=1,split=2[camera0][mix0];
[src1]crop=1920:2160:1920:0,setsar=1,split=2[camera1][mix1];
[mix0][mix1]overlay=enable='...'[switched]
```

Renders and Resolve camera exports keep the selected CQ/CRF quality setting
instead of forcing lossless QP=0, which avoids unusable long-form 4K
intermediates.

Consumer GPUs cap how many hardware encoder sessions can be open at once. The
job probes that limit on a tiny clip before starting, so a driver that refuses
three concurrent sessions falls back to the fewest extra passes that fit rather
than failing an hour into a 4K render. The status output reports how many passes
were actually used.

If you already have `speaker_segments.json`, enable timeline reuse to skip
diarization and only rerender the video. In split-video mode, the selected
left/right first-speaker option is applied even when reusing a timeline.

The script writes `speaker_segments.json` with three useful views: the regular
Pyannote diarization, the model-native exclusive diarization, and the final
full-duration camera timeline. The first two are raw model outputs. Only the
camera timeline applies minimum-visible-turn and long-silence look-ahead rules.
Legacy timeline files can still be reused as stored, but rerun diarization once
to get the new Community-1 exclusive-speaker data.

Camera videos do not loop by default. Pass `--loop-speaker-videos` only when a
camera file is shorter than the soundtrack.

## GPU Acceleration

The default hardware choices are now automatic:

- NVIDIA systems select CUDA for Pyannote when available, `h264_nvenc` for FFmpeg encoding, and CUDA hardware decoding.
- Apple Silicon Macs run Pyannote on CPU and select Apple VideoToolbox for FFmpeg encoding and decoding.
- segmentation and speaker-embedding inference use a memory-safe batch size of 1.
- FFmpeg decodes and encodes frames as a stream; the full video is never loaded into GPU memory.

A 24 GB source file does not need 24 GB of VRAM. VRAM usage is controlled by
the model and current inference/frame batches, not by the source file size.
Pyannote/TorchCodec decodes the source through FFmpeg while the model processes
the recording; the source file itself is never loaded wholesale into VRAM.

Useful speed options:

```powershell
python main.py .\conversation.wav .\speaker_0.mp4 .\speaker_1.mp4 --preset p1 --crf 26
```

Use `--hwaccel none` if your FFmpeg build has hardware decode issues.

## Model Access

The default local diarization model is `pyannote/speaker-diarization-community-1`,
the current highest-quality open Pyannote pipeline. It is gated by Hugging Face,
so the token in `.env` must come from an account that accepted its conditions:

```text
https://hf.co/pyannote/speaker-diarization-community-1
```

Pyannote's `precision-2` benchmarks higher, but it is a premium cloud service and
does not run on the local 8 GiB GPU. This project intentionally uses Community-1
for local diarization. After accepting the model terms, rerun once to download
and cache it locally.
