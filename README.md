# Speaker Diarization Video Switcher

Creates a speaker-focused final video from either embedded video audio or a separate soundtrack and:

- two independent speaker camera videos, or
- one combined video containing the left and right speakers.

The script converts the selected audio to mono 16 kHz WAV for pyannote, runs
speaker diarization, writes a stable speaker timeline, and uses FFmpeg to show
the active speaker. Overlapping pyannote tracks are resolved correctly, and
short detections are ignored so the camera does not flicker.

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
- A Hugging Face token with access to `pyannote/speaker-diarization-3.1`
- A CUDA-enabled PyTorch build and an FFmpeg build with `h264_nvenc`

Install Python dependencies:

```powershell
pip install -e .
```

For NVIDIA GPU diarization, install PyTorch from the CUDA wheel index that
matches your machine before installing the project dependencies.

## Gradio App

Start the local web app:

```powershell
.venv\Scripts\python.exe app.py
```

Then open the local URL printed by Gradio:

1. Select **Two separate speaker videos** or **One combined left/right video**.
2. Choose **Use audio from the video** or **Upload a separate audio file**.
3. In combined-video mode, choose whether the first detected speaker is on the
   left or right.
4. Adjust **Minimum speaker turn before switching** when you want a longer or
   shorter camera hold. The default is 1.0 seconds.

For two separate videos, embedded-audio mode uses the audio track from the first
speaker video. For a combined video, it uses that video's own audio track.

If you already have `speaker_segments.json`, enable timeline reuse to skip
diarization and only rerender the video. In split-video mode, the selected
left/right first-speaker option is applied even when reusing a timeline.

The script writes `speaker_segments.json`, which contains the detected speaker
labels, camera mapping, and stabilized timestamp timeline used for the FFmpeg
render. Nested speaker turns inside a longer overlapping turn are retained, but
turns shorter than the configured camera-switch duration are absorbed into the
current shot.

Camera videos do not loop by default. Pass `--loop-speaker-videos` only when a
camera file is shorter than the soundtrack.

## GPU Acceleration

By default, the script uses the NVIDIA CUDA path:

- pyannote diarization runs with `--device cuda`
- FFmpeg decoding uses CUDA when requested
- FFmpeg encoding uses `h264_nvenc`

Useful speed options:

```powershell
python main.py .\conversation.wav .\speaker_0.mp4 .\speaker_1.mp4 --preset p1 --crf 26
```

Use `--hwaccel none` if your FFmpeg build has hardware decode issues.

## Model Access

The default diarization model is gated by Hugging Face. The token in `.env` must
come from an account that has accepted the conditions at:

```text
https://hf.co/pyannote/speaker-diarization-3.1
https://hf.co/pyannote/segmentation-3.0
```

After accepting, rerun the script once to download and cache the model locally.
