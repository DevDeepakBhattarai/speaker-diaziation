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

Only the active speaker's half is rendered. The output keeps the cropped half's
native aspect ratio and dimensions instead of horizontally stretching it.
Choose whether the first unique speaker detected in the soundtrack is the
person on the left or the person on the right.

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

## Requirements

- FFmpeg and FFprobe on `PATH`
- Python 3.10, 3.11, or 3.12
- A Hugging Face token with access to `pyannote/speaker-diarization-community-1`
- A CUDA-enabled PyTorch build and an FFmpeg build with `h264_nvenc`
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
- two full-length, time-aligned camera-angle proxies,
- `speaker_segments.json`, `davinci_manifest.json`, and import instructions.

For a combined left/right recording, the renderer crops the active half at
native pixel density and never scales it. The cropped pixels are placed on an
output canvas matching the source dimensions, so a 3840x2160 source remains
3840x2160 instead of becoming a 1920-wide render. The unused canvas area is
padded rather than stretching the crop. Split-video renders and Resolve camera
exports use lossless video encoding to avoid adding generation loss. For two
separate camera files, each source keeps its original resolution.

Import the downloaded file in DaVinci Resolve using **File > Import > Timeline**
and select `speaker_edit.otioz`. Keep the imported timeline frame rate when
Resolve asks. V1 remains editable for trimming or replacing camera cuts, while
A1 remains a single continuous soundtrack.

The bundle can be large and takes longer to create because it performs two
additional full-length camera proxy encodes. Leave the checkbox disabled when
only the rendered video and JSON timeline are needed.

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

By default, the script uses the NVIDIA CUDA path:

- pyannote diarization runs with `--device cuda`
- segmentation and speaker-embedding inference use a memory-safe batch size of 1
- FFmpeg decodes and encodes frames as a stream; the full video is never loaded
  into VRAM
- FFmpeg encoding uses `h264_nvenc`

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
