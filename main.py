from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str


def run(command: list[str], *, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print("$ " + " ".join(shlex.quote(part) for part in command))
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.PIPE if quiet else None,
    )


def require_tool(name: str) -> None:
    try:
        run([name, "-version"], quiet=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Missing required tool: {name}") from exc


def ffprobe_json(path: Path) -> dict:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        quiet=True,
    )
    return json.loads(result.stdout)


def media_duration(path: Path) -> float:
    info = ffprobe_json(path)
    duration = info.get("format", {}).get("duration")
    if duration is None:
        raise SystemExit(f"Could not read duration from {path}")
    return float(duration)


def source_video_size(path: Path) -> tuple[int, int]:
    for stream in ffprobe_json(path).get("streams", []):
        if stream.get("codec_type") == "video":
            return int(stream["width"]), int(stream["height"])
    raise SystemExit(f"No video stream found in {path}")


def available_ffmpeg_encoders() -> str:
    try:
        return run(["ffmpeg", "-hide_banner", "-encoders"], quiet=True).stdout
    except subprocess.CalledProcessError:
        return ""


def choose_video_encoder(requested: str) -> str:
    if requested != "auto":
        return requested

    encoders = available_ffmpeg_encoders()
    for encoder in ("h264_nvenc", "h264_amf", "h264_qsv"):
        if encoder in encoders:
            return encoder
    return "libx264"


def prepare_audio(audio_file: Path, output_wav: Path, sample_rate: int) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            str(audio_file),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(output_wav),
        ]
    )


def diarize_audio(
    audio_path: Path,
    *,
    model: str,
    hf_token: str | None,
    device: str,
    min_speakers: int | None,
    max_speakers: int | None,
    num_speakers: int | None,
) -> list[Segment]:
    try:
        from dotenv import load_dotenv
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing Python dependencies. Install pyannote.audio, python-dotenv, and a CUDA-enabled torch build."
        ) from exc

    load_dotenv()
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "Set HF_TOKEN or pass --hf-token. pyannote diarization models require Hugging Face access."
        )

    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading diarization model on {selected_device}: {model}")
    try:
        pipeline = Pipeline.from_pretrained(model, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
    except Exception as exc:
        raise SystemExit(
            "Could not load the pyannote diarization model. If this is a 403 error, "
            f"accept the Hugging Face model conditions for https://hf.co/{model} "
            "using the same account that created your HF_TOKEN."
        ) from exc
    pipeline.to(torch.device(selected_device))

    options: dict[str, int] = {}
    if num_speakers is not None:
        options["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            options["min_speakers"] = min_speakers
        if max_speakers is not None:
            options["max_speakers"] = max_speakers

    diarization = pipeline(str(audio_path), **options)
    segments = [
        Segment(float(turn.start), float(turn.end), speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    return sorted(segments, key=lambda item: (item.start, item.end, item.speaker))


def merge_segments(segments: list[Segment], *, gap: float, min_duration: float) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if segment.end - segment.start < min_duration:
            continue
        if (
            merged
            and merged[-1].speaker == segment.speaker
            and segment.start - merged[-1].end <= gap
        ):
            previous = merged[-1]
            merged[-1] = Segment(previous.start, max(previous.end, segment.end), previous.speaker)
        else:
            merged.append(segment)
    return merged


def fill_timeline(
    segments: list[Segment],
    *,
    duration: float,
    fallback_speaker: str,
    gap_padding: float,
) -> list[Segment]:
    timeline: list[Segment] = []
    cursor = 0.0
    current_speaker = segments[0].speaker if segments else fallback_speaker

    for segment in segments:
        start = max(0.0, min(duration, segment.start - gap_padding))
        end = max(start, min(duration, segment.end + gap_padding))
        if start > cursor:
            timeline.append(Segment(cursor, start, current_speaker))
        if end > cursor:
            timeline.append(Segment(max(cursor, start), end, segment.speaker))
            cursor = end
            current_speaker = segment.speaker

    if cursor < duration:
        timeline.append(Segment(cursor, duration, current_speaker))

    compacted: list[Segment] = []
    for item in timeline:
        if item.end - item.start <= 0.01:
            continue
        if compacted and compacted[-1].speaker == item.speaker:
            previous = compacted[-1]
            compacted[-1] = Segment(previous.start, item.end, previous.speaker)
        else:
            compacted.append(item)
    return compacted


def speaker_indexes_by_detection_order(segments: list[Segment]) -> dict[str, int]:
    labels: list[str] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        if segment.speaker not in labels:
            labels.append(segment.speaker)

    if len(labels) != 2:
        raise SystemExit(
            f"Expected exactly 2 speakers in the audio, but diarization found {len(labels)}: "
            f"{', '.join(labels) or 'none'}"
        )

    return {labels[0]: 0, labels[1]: 1}


def ffmpeg_filter_for_segments(
    timeline: list[Segment],
    mapping: dict[str, int],
    *,
    width: int,
    height: int,
    use_cuda_overlay: bool,
) -> tuple[str, str]:
    output_label = "outv"
    speaker1_intervals = [
        segment for segment in timeline if mapping[segment.speaker] == 1
    ]
    conditions = [
        f"between(t\\,{segment.start:.3f}\\,{segment.end:.3f})"
        for segment in speaker1_intervals
    ]
    active_expression = "+".join(conditions) if conditions else "0"

    if use_cuda_overlay:
        filter_text = (
            "[0:v]setpts=PTS-STARTPTS[base];\n"
            "[1:v]setpts=PTS-STARTPTS[fg];\n"
            f"[base][fg]overlay_cuda=x='if(gt({active_expression}\\,0)\\,0\\,{width})':"
            f"y=0:eof_action=repeat:repeatlast=1[{output_label}]"
        )
        return filter_text, output_label

    filter_text = (
        f"[0:v]setpts=PTS-STARTPTS,scale={width}:{height}:"
        f"force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[base];\n"
        f"[1:v]setpts=PTS-STARTPTS,scale={width}:{height}:"
        f"force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[fg];\n"
        f"[base][fg]overlay=enable='gt({active_expression}\\,0)':"
        f"x=0:y=0:eof_action=repeat:repeatlast=1[{output_label}]"
    )
    return filter_text, output_label


def choose_hwaccel(requested: str, encoder: str) -> str | None:
    if requested == "none":
        return None
    if requested != "auto":
        return requested
    if encoder in {"h264_nvenc", "hevc_nvenc"}:
        return "cuda"
    if encoder.endswith("_qsv"):
        return "qsv"
    if encoder.endswith("_amf"):
        return "dxva2"
    return None


def assemble_video(
    *,
    audio_file: Path,
    camera_videos: list[Path],
    output_video: Path,
    timeline: list[Segment],
    mapping: dict[str, int],
    encoder: str,
    audio_codec: str,
    preset: str,
    crf: int,
    hwaccel: str | None,
    loop_cameras: bool,
    render_duration: float | None,
) -> None:
    width, height = source_video_size(camera_videos[0])
    filter_text, output_label = ffmpeg_filter_for_segments(
        timeline,
        mapping,
        width=width,
        height=height,
        use_cuda_overlay=hwaccel == "cuda" and encoder in {"h264_nvenc", "hevc_nvenc"},
    )

    with tempfile.NamedTemporaryFile("w", suffix=".ffmpeg", delete=False, encoding="utf-8") as file:
        filter_path = Path(file.name)
        file.write(filter_text)

    command = ["ffmpeg", "-y", "-hide_banner"]
    for index, video in enumerate(camera_videos):
        if loop_cameras:
            command.extend(["-stream_loop", "-1"])
        if hwaccel:
            command.extend(["-hwaccel", hwaccel, "-hwaccel_output_format", hwaccel])
        command.extend(["-i", str(video)])
    command.extend(["-i", str(audio_file)])

    command.extend(
        [
            "-filter_complex_script",
            str(filter_path),
            "-map",
            f"[{output_label}]",
            "-map",
            f"{len(camera_videos)}:a:0",
            "-c:v",
            encoder,
            "-aspect",
            f"{width}:{height}",
        ]
    )

    if encoder in {"h264_nvenc", "hevc_nvenc"}:
        command.extend(["-preset", preset, "-cq", str(crf)])
    elif encoder == "libx264":
        x264_preset = (
            preset
            if preset
            in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}
            else "veryfast"
        )
        command.extend(["-preset", x264_preset, "-crf", str(crf)])

    command.extend(
        [
            "-c:a",
            audio_codec,
            "-max_muxing_queue_size",
            "4096",
            "-t",
            f"{(render_duration or media_duration(audio_file)):.3f}",
            str(output_video),
        ]
    )

    try:
        run(command)
    finally:
        filter_path.unlink(missing_ok=True)


def write_segments_json(path: Path, segments: list[Segment], mapping: dict[str, int]) -> None:
    payload = {
        "speaker_to_camera": mapping,
        "segments": [
            {"start": item.start, "end": item.end, "speaker": item.speaker}
            for item in segments
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_segments_json(path: Path) -> tuple[list[Segment], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(key): int(value) for key, value in payload["speaker_to_camera"].items()}
    segments = [
        Segment(float(item["start"]), float(item["end"]), str(item["speaker"]))
        for item in payload["segments"]
    ]
    return segments, mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diarize an audio file, then create a switched video from speaker 0 and "
            "speaker 1 camera videos."
        )
    )
    parser.add_argument("audio_file", type=Path, help="Audio file used for diarization and output.")
    parser.add_argument("speaker0_video", type=Path, help="Video to show for the first detected speaker.")
    parser.add_argument("speaker1_video", type=Path, help="Video to show for the second detected speaker.")
    parser.add_argument("-o", "--output", type=Path, default=Path("speaker_switched.mp4"))
    parser.add_argument("--model", default=DIARIZATION_MODEL, help="Hugging Face diarization model.")
    parser.add_argument("--hf-token", help="Hugging Face token. Defaults to HF_TOKEN.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--num-speakers", type=int, default=2, help="Exact speaker count for diarization.")
    parser.add_argument("--min-speakers", type=int, help="Minimum speaker count for diarization.")
    parser.add_argument("--max-speakers", type=int, help="Maximum speaker count for diarization.")
    parser.add_argument("--merge-gap", type=float, default=0.30)
    parser.add_argument("--min-segment", type=float, default=0.25)
    parser.add_argument("--gap-padding", type=float, default=0.05)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--video-encoder", default="h264_nvenc")
    parser.add_argument("--audio-codec", default="aac")
    parser.add_argument("--preset", default="p4", help="Encoder preset. For NVENC, p1 is fastest.")
    parser.add_argument("--crf", type=int, default=23, help="CRF/CQ quality value.")
    parser.add_argument(
        "--hwaccel",
        default="cuda",
        help="FFmpeg hardware decoder: cuda, auto, qsv, dxva2, d3d11va, or none.",
    )
    parser.add_argument(
        "--loop-speaker-videos",
        action="store_true",
        help="Loop speaker videos when the audio is longer than the videos.",
    )
    parser.add_argument("--segments-json", type=Path, default=Path("speaker_segments.json"))
    parser.add_argument(
        "--reuse-segments",
        action="store_true",
        help="Skip diarization and reuse --segments-json from a previous run.",
    )
    parser.add_argument(
        "--render-duration",
        type=float,
        help="Render only the first N seconds. Useful for performance tests.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    require_tool("ffmpeg")
    require_tool("ffprobe")

    audio_file = args.audio_file.resolve()
    speaker0_video = args.speaker0_video.resolve()
    speaker1_video = args.speaker1_video.resolve()
    output_video = args.output.resolve()
    camera_videos = [speaker0_video, speaker1_video]

    for path in [audio_file, *camera_videos]:
        if not path.exists():
            raise SystemExit(f"Input file does not exist: {path}")

    encoder = choose_video_encoder(args.video_encoder)
    hwaccel = choose_hwaccel(args.hwaccel, encoder)
    print(f"Using video encoder: {encoder}")
    print(f"Using FFmpeg hwaccel: {hwaccel or 'none'}")

    duration = media_duration(audio_file)
    segments_json = args.segments_json.resolve()
    if args.reuse_segments:
        timeline, mapping = read_segments_json(segments_json)
        print(f"Reused diarization timeline: {segments_json}")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "source_audio.wav"
            prepare_audio(audio_file, audio_path, args.sample_rate)
            raw_segments = diarize_audio(
                audio_path,
                model=args.model,
                hf_token=args.hf_token,
                device=args.device,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                num_speakers=args.num_speakers,
            )

        if not raw_segments:
            raise SystemExit("Diarization produced no speaker segments.")

        merged = merge_segments(
            raw_segments,
            gap=args.merge_gap,
            min_duration=args.min_segment,
        )
        mapping = speaker_indexes_by_detection_order(merged)
        fallback_speaker = next(iter(mapping))
        timeline = fill_timeline(
            merged,
            duration=duration,
            fallback_speaker=fallback_speaker,
            gap_padding=args.gap_padding,
        )

        segments_json.parent.mkdir(parents=True, exist_ok=True)
        write_segments_json(segments_json, timeline, mapping)
        print(f"Wrote diarization timeline: {segments_json}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    assemble_video(
        audio_file=audio_file,
        camera_videos=camera_videos,
        output_video=output_video,
        timeline=timeline,
        mapping=mapping,
        encoder=encoder,
        audio_codec=args.audio_codec,
        preset=args.preset,
        crf=args.crf,
        hwaccel=hwaccel,
        loop_cameras=args.loop_speaker_videos,
        render_duration=args.render_duration,
    )
    print(f"Wrote switched video: {output_video}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
