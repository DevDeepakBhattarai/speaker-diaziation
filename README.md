# Speaker Diarization Video Switcher

Creates a final video from exactly three input files:

- one audio file used for diarization and final output audio
- one video for the first detected speaker
- one video for the second detected speaker

The script converts the audio to mono 16 kHz WAV for pyannote, runs speaker
diarization, writes speaker timestamps, then uses FFmpeg to switch between the
two speaker videos. The first unique speaker detected in the audio maps to the
first speaker video. The second unique speaker detected maps to the second
speaker video.

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

## Usage

```powershell
$env:HF_TOKEN = "hf_..."
python main.py .\conversation.wav .\speaker_0.mp4 .\speaker_1.mp4 -o .\out.mp4
```

## Gradio App

Start the local web app:

```powershell
.venv\Scripts\python.exe app.py
```

Then open the local URL printed by Gradio. Upload the audio file, speaker 0
video, and speaker 1 video. If you already have `speaker_segments.json`, enable
timeline reuse to skip diarization and only rerender the video.

The output video uses:

- `conversation.wav` as the audio track
- `speaker_0.mp4` whenever the first detected speaker is talking
- `speaker_1.mp4` whenever the second detected speaker is talking

The script writes `speaker_segments.json`, which contains the detected speaker
labels, camera mapping, and timestamp timeline used for the FFmpeg render.

Speaker videos do not loop by default. Pass `--loop-speaker-videos` only when
one of the camera files is shorter than the audio.

## GPU Acceleration

By default, the script uses the NVIDIA CUDA path:

- pyannote diarization runs with `--device cuda`
- FFmpeg decodes with `--hwaccel cuda`
- FFmpeg encodes with `--video-encoder h264_nvenc`

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
