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


DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
MODE_SEPARATE_VIDEOS = "separate-videos"
MODE_SPLIT_VIDEO = "split-video"
_WINDOWS_DLL_HANDLES: list[object] = []


class DiarizationError(RuntimeError):
    """Raised when local speaker diarization cannot be initialized or run."""


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


def media_has_audio(path: Path) -> bool:
    return any(
        stream.get("codec_type") == "audio"
        for stream in ffprobe_json(path).get("streams", [])
    )


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


def _ensure_torchcodec_ffmpeg_dlls() -> None:
    """Expose an FFmpeg 4-7 shared build to TorchCodec on Windows.

    TorchCodec 0.7 uses Windows DLL loading rather than the ffmpeg.exe CLI.
    The regular renderer may use any FFmpeg executable, but diarization needs
    compatible shared avcodec/avformat DLLs available to the Python process.
    """
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if _WINDOWS_DLL_HANDLES:
        return

    supported_avcodec = (
        "avcodec-61.dll",
        "avcodec-60.dll",
        "avcodec-59.dll",
        "avcodec-58.dll",
    )
    candidates: list[Path] = []

    configured = os.getenv("FFMPEG_SHARED_BIN")
    if configured:
        candidates.append(Path(configured).expanduser())

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry.strip():
            candidates.append(Path(entry.strip().strip('"')))

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if packages.is_dir():
            for package in packages.glob("*FFmpeg*Shared*"):
                candidates.extend(package.glob("*/bin"))
                candidates.extend(package.glob("bin"))

    for candidate in candidates:
        try:
            if not candidate.is_dir():
                continue
            if not any((candidate / name).is_file() for name in supported_avcodec):
                continue
            handle = os.add_dll_directory(str(candidate))
            _WINDOWS_DLL_HANDLES.append(handle)
            os.environ["PATH"] = f"{candidate}{os.pathsep}{os.environ.get('PATH', '')}"
            return
        except OSError:
            continue

    raise DiarizationError(
        "Pyannote 4/TorchCodec requires an FFmpeg 4-7 shared build on Windows. "
        "Install it with: winget install --id BtbN.FFmpeg.GPL.Shared.7.1 -e"
    )


def _configure_diarization_batch_size(pipeline: object, batch_size: int) -> int:
    safe_batch_size = max(1, int(batch_size))
    if hasattr(pipeline, "segmentation_batch_size"):
        pipeline.segmentation_batch_size = safe_batch_size
    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = safe_batch_size
    return safe_batch_size


def _segments_from_annotation(annotation: object) -> list[Segment]:
    return sorted(
        [
            Segment(float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ],
        key=lambda item: (item.start, item.end, item.speaker),
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
    batch_size: int = 1,
) -> tuple[list[Segment], list[Segment]]:
    _ensure_torchcodec_ffmpeg_dlls()
    try:
        from dotenv import load_dotenv
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationError(
            "Missing Python dependencies. Run `uv sync` to install the Pyannote/CUDA stack."
        ) from exc

    load_dotenv()
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise DiarizationError(
            "Set HF_TOKEN or pass --hf-token. Pyannote diarization models require Hugging Face access."
        )

    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading diarization model on {selected_device}: {model}")
    try:
        pipeline = Pipeline.from_pretrained(model, token=token)
    except Exception as exc:
        raise DiarizationError(
            "Could not load the Pyannote diarization model. If this is a 403 error, "
            f"accept the Hugging Face model conditions for https://hf.co/{model} "
            "using the same account that created your HF_TOKEN."
        ) from exc
    pipeline.to(torch.device(selected_device))

    # Batch size changes only inference scheduling, not diarization semantics.
    # Keep it conservative for an 8 GiB GPU while processing the whole source.
    safe_batch_size = _configure_diarization_batch_size(pipeline, batch_size)
    print(f"Using diarization GPU batch size: {safe_batch_size}")

    options: dict[str, int] = {}
    if num_speakers is not None:
        options["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            options["min_speakers"] = min_speakers
        if max_speakers is not None:
            options["max_speakers"] = max_speakers

    output = pipeline(str(audio_path), **options)
    regular = getattr(output, "speaker_diarization", output)
    exclusive = getattr(output, "exclusive_speaker_diarization", None)

    regular_segments = _segments_from_annotation(regular)
    exclusive_segments = (
        _segments_from_annotation(exclusive)
        if exclusive is not None
        else regular_segments
    )
    return regular_segments, exclusive_segments


def _compact_timeline(timeline: list[Segment]) -> list[Segment]:
    compacted: list[Segment] = []
    for item in sorted(timeline, key=lambda segment: (segment.start, segment.end)):
        if item.end <= item.start:
            continue
        if compacted and compacted[-1].speaker == item.speaker:
            previous = compacted[-1]
            compacted[-1] = Segment(
                previous.start,
                max(previous.end, item.end),
                previous.speaker,
            )
        else:
            compacted.append(item)
    return compacted


def _speaker_evidence_duration(
    segments: list[Segment],
    *,
    speaker: str,
    start: float,
    end: float,
) -> float:
    intervals: list[tuple[float, float]] = []
    for segment in segments:
        if segment.speaker != speaker or segment.end <= start or segment.start >= end:
            continue
        intervals.append((max(start, segment.start), min(end, segment.end)))

    total = 0.0
    merged_start: float | None = None
    merged_end: float | None = None
    for interval_start, interval_end in sorted(intervals):
        if merged_start is None:
            merged_start = interval_start
            merged_end = interval_end
        elif interval_start <= merged_end:
            merged_end = max(merged_end, interval_end)
        else:
            total += merged_end - merged_start
            merged_start = interval_start
            merged_end = interval_end

    if merged_start is not None and merged_end is not None:
        total += merged_end - merged_start
    return total


def build_camera_timeline(
    segments: list[Segment],
    *,
    duration: float,
    fallback_speaker: str,
    min_switch_duration: float = 1.0,
    silence_threshold: float = 2.5,
    silence_lookahead: float = 5.0,
    gap_padding: float = 0.05,
) -> list[Segment]:
    """Convert untouched diarization timestamps into an FFmpeg camera policy.

    This function never changes diarization labels or writes back into raw model
    output. It only decides which camera should be visible. Silence keeps the
    current camera. Short speaker turns are ignored unless a long preceding
    silence plus bounded look-ahead provides enough evidence for the new speaker.
    """
    if duration <= 0:
        return []

    normalized = sorted(
        [
            Segment(
                max(0.0, min(duration, segment.start)),
                max(0.0, min(duration, segment.end)),
                segment.speaker,
            )
            for segment in segments
            if segment.end > segment.start
        ],
        key=lambda segment: (segment.start, segment.end, segment.speaker),
    )
    normalized = [segment for segment in normalized if segment.end > segment.start]
    if not normalized:
        return [Segment(0.0, duration, fallback_speaker)]

    timeline: list[Segment] = []
    current_speaker = fallback_speaker
    camera_segment_start = 0.0
    activity_end = 0.0
    index = 0

    while index < len(normalized):
        event_start = normalized[index].start
        event_segments: list[Segment] = []
        while index < len(normalized) and abs(normalized[index].start - event_start) <= 1e-9:
            event_segments.append(normalized[index])
            index += 1

        silence_before = max(0.0, event_start - activity_end)
        candidate = max(
            event_segments,
            key=lambda segment: (segment.end - segment.start, segment.end, segment.speaker),
        )

        if candidate.speaker != current_speaker:
            qualifies = candidate.end - candidate.start >= min_switch_duration

            if (
                not qualifies
                and silence_before >= silence_threshold
                and silence_lookahead > 0
            ):
                lookahead_end = min(duration, candidate.start + silence_lookahead)
                for future in normalized[index:]:
                    if future.start >= lookahead_end:
                        break
                    if future.speaker != candidate.speaker:
                        lookahead_end = min(lookahead_end, future.start)
                        break
                qualifies = _speaker_evidence_duration(
                    normalized,
                    speaker=candidate.speaker,
                    start=candidate.start,
                    end=lookahead_end,
                ) >= min_switch_duration

            if qualifies:
                if event_start > camera_segment_start:
                    timeline.append(
                        Segment(camera_segment_start, event_start, current_speaker)
                    )
                current_speaker = candidate.speaker
                camera_segment_start = event_start

        activity_end = max(
            activity_end,
            max(segment.end + max(0.0, gap_padding) for segment in event_segments),
        )

    if camera_segment_start < duration:
        timeline.append(Segment(camera_segment_start, duration, current_speaker))

    return _compact_timeline(timeline)


def speaker_indexes_by_detection_order(
    segments: list[Segment],
    *,
    first_camera_index: int = 0,
) -> dict[str, int]:
    if first_camera_index not in {0, 1}:
        raise ValueError("first_camera_index must be 0 or 1")

    labels: list[str] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        if segment.speaker not in labels:
            labels.append(segment.speaker)

    if len(labels) != 2:
        raise SystemExit(
            f"Expected exactly 2 speakers in the audio, but diarization found {len(labels)}: "
            f"{', '.join(labels) or 'none'}"
        )

    second_camera_index = 1 - first_camera_index
    return {labels[0]: first_camera_index, labels[1]: second_camera_index}


def speaker1_active_expression(
    timeline: list[Segment],
    mapping: dict[str, int],
) -> str:
    speaker1_intervals = [
        segment for segment in timeline if mapping[segment.speaker] == 1
    ]
    conditions = [
        f"between(t\\,{segment.start:.3f}\\,{segment.end:.3f})"
        for segment in speaker1_intervals
    ]
    return "+".join(conditions) if conditions else "0"


def ffmpeg_filter_for_segments(
    timeline: list[Segment],
    mapping: dict[str, int],
    *,
    width: int,
    height: int,
    use_cuda_overlay: bool,
) -> tuple[str, str]:
    output_label = "outv"
    active_expression = speaker1_active_expression(timeline, mapping)

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


def ffmpeg_filter_for_split_video_segments(
    timeline: list[Segment],
    mapping: dict[str, int],
    *,
    source_width: int,
    source_height: int,
) -> tuple[str, str, int, int]:
    output_label = "outv"
    output_width = source_width // 2
    output_height = source_height

    # H.264 encoders commonly require even output dimensions. Dropping one edge
    # pixel is preferable to stretching either person's half of the source frame.
    output_width -= output_width % 2
    output_height -= output_height % 2
    if output_width < 2 or output_height < 2:
        raise SystemExit(
            f"Combined video is too small to split: {source_width}x{source_height}"
        )

    right_x = source_width - output_width
    active_expression = speaker1_active_expression(timeline, mapping)
    crop_x = f"if(gt({active_expression}\\,0)\\,{right_x}\\,0)"
    filter_text = (
        "[0:v]setpts=PTS-STARTPTS,"
        f"crop={output_width}:{output_height}:x='{crop_x}':y=0,"
        f"setsar=1[{output_label}]"
    )
    return filter_text, output_label, output_width, output_height


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


def append_video_encoding_options(
    command: list[str],
    *,
    encoder: str,
    preset: str,
    crf: int,
) -> None:
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
    audio_input_index: int | None = None
    resolved_audio = audio_file.resolve()
    for index, video in enumerate(camera_videos):
        if loop_cameras:
            command.extend(["-stream_loop", "-1"])
        if hwaccel:
            command.extend(["-hwaccel", hwaccel, "-hwaccel_output_format", hwaccel])
        command.extend(["-i", str(video)])
        if video.resolve() == resolved_audio:
            audio_input_index = index

    if audio_input_index is None:
        audio_input_index = len(camera_videos)
        command.extend(["-i", str(audio_file)])

    command.extend(
        [
            "-filter_complex_script",
            str(filter_path),
            "-map",
            f"[{output_label}]",
            "-map",
            f"{audio_input_index}:a:0",
            "-c:v",
            encoder,
            "-aspect",
            f"{width}:{height}",
        ]
    )
    append_video_encoding_options(command, encoder=encoder, preset=preset, crf=crf)

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


def assemble_split_video(
    *,
    audio_file: Path,
    combined_video: Path,
    output_video: Path,
    timeline: list[Segment],
    mapping: dict[str, int],
    encoder: str,
    audio_codec: str,
    preset: str,
    crf: int,
    hwaccel: str | None,
    loop_video: bool,
    render_duration: float | None,
) -> None:
    source_width, source_height = source_video_size(combined_video)
    filter_text, output_label, output_width, output_height = (
        ffmpeg_filter_for_split_video_segments(
            timeline,
            mapping,
            source_width=source_width,
            source_height=source_height,
        )
    )

    with tempfile.NamedTemporaryFile("w", suffix=".ffmpeg", delete=False, encoding="utf-8") as file:
        filter_path = Path(file.name)
        file.write(filter_text)

    command = ["ffmpeg", "-y", "-hide_banner"]
    if loop_video:
        command.extend(["-stream_loop", "-1"])
    if hwaccel:
        # Keep decoded frames in system memory because crop/overlay are software
        # filters. Encoding can still use NVENC/AMF/QSV.
        command.extend(["-hwaccel", hwaccel])
    command.extend(["-i", str(combined_video)])
    audio_input_index = 0
    if combined_video.resolve() != audio_file.resolve():
        audio_input_index = 1
        command.extend(["-i", str(audio_file)])
    command.extend(
        [
            "-filter_complex_script",
            str(filter_path),
            "-map",
            f"[{output_label}]",
            "-map",
            f"{audio_input_index}:a:0",
            "-c:v",
            encoder,
            "-aspect",
            f"{output_width}:{output_height}",
        ]
    )
    append_video_encoding_options(command, encoder=encoder, preset=preset, crf=crf)
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


def _segments_payload(segments: list[Segment]) -> list[dict[str, float | str]]:
    return [
        {"start": item.start, "end": item.end, "speaker": item.speaker}
        for item in segments
    ]


def write_segments_json(
    path: Path,
    segments: list[Segment],
    mapping: dict[str, int],
    *,
    speech_segments: list[Segment] | None = None,
    exclusive_speech_segments: list[Segment] | None = None,
) -> None:
    payload: dict[str, object] = {
        "speaker_to_camera": mapping,
        "segments": _segments_payload(segments),
    }
    if speech_segments is not None:
        payload["speech_segments"] = _segments_payload(speech_segments)
    if exclusive_speech_segments is not None:
        payload["exclusive_speech_segments"] = _segments_payload(exclusive_speech_segments)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_segments(payload: object) -> list[Segment]:
    if not isinstance(payload, list):
        return []
    return [
        Segment(float(item["start"]), float(item["end"]), str(item["speaker"]))
        for item in payload
        if isinstance(item, dict)
    ]


def read_segments_json(path: Path) -> tuple[list[Segment], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    mapping = {str(key): int(value) for key, value in payload["speaker_to_camera"].items()}
    return _parse_segments(payload["segments"]), mapping


def read_speech_segments_json(path: Path) -> list[Segment] | None:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if "speech_segments" not in payload:
        return None
    return _parse_segments(payload["speech_segments"])


def read_exclusive_speech_segments_json(path: Path) -> list[Segment] | None:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if "exclusive_speech_segments" not in payload:
        return None
    return _parse_segments(payload["exclusive_speech_segments"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diarize an audio file, then switch between either two camera videos or "
            "the left/right halves of one combined video."
        )
    )
    parser.add_argument("audio_file", type=Path, help="Audio file used for diarization and output.")
    parser.add_argument(
        "speaker0_video",
        type=Path,
        nargs="?",
        help=(
            "First detected speaker video in separate mode, or the combined left/right "
            "video in split-video mode."
        ),
    )
    parser.add_argument(
        "speaker1_video",
        type=Path,
        nargs="?",
        help="Second detected speaker video. Required only in separate-videos mode.",
    )
    parser.add_argument(
        "--mode",
        choices=(MODE_SEPARATE_VIDEOS, MODE_SPLIT_VIDEO),
        default=MODE_SEPARATE_VIDEOS,
        help="Use two camera files or split one combined left/right video.",
    )
    parser.add_argument(
        "--first-speaker-side",
        choices=("left", "right"),
        default="left",
        help="In split-video mode, side occupied by the first detected speaker.",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("speaker_switched.mp4"))
    parser.add_argument("--model", default=DIARIZATION_MODEL, help="Hugging Face diarization model.")
    parser.add_argument("--hf-token", help="Hugging Face token. Defaults to HF_TOKEN.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--num-speakers", type=int, default=2, help="Exact speaker count for diarization.")
    parser.add_argument("--min-speakers", type=int, help="Minimum speaker count for diarization.")
    parser.add_argument("--max-speakers", type=int, help="Maximum speaker count for diarization.")
    parser.add_argument(
        "--min-switch-duration",
        type=float,
        default=1.0,
        help="Camera-only debounce: minimum spoken evidence before switching views.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=2.5,
        help="Camera-only silence duration that enables speaker look-ahead.",
    )
    parser.add_argument(
        "--silence-lookahead",
        type=float,
        default=5.0,
        help="Camera-only seconds to inspect after long silence before switching.",
    )
    parser.add_argument(
        "--gap-padding",
        type=float,
        default=0.05,
        help="Camera-only speech-end padding when measuring silence gaps.",
    )
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
        help="Loop the camera video(s) when the audio is longer.",
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

    if args.speaker0_video is None:
        parser.error("A video file is required.")
    if args.mode == MODE_SEPARATE_VIDEOS and args.speaker1_video is None:
        parser.error("separate-videos mode requires both speaker video files.")
    if args.mode == MODE_SPLIT_VIDEO and args.speaker1_video is not None:
        parser.error("split-video mode accepts exactly one combined video file.")

    audio_file = args.audio_file.resolve()
    first_video = args.speaker0_video.resolve()
    second_video = args.speaker1_video.resolve() if args.speaker1_video else None
    output_video = args.output.resolve()

    input_paths = [audio_file, first_video]
    if second_video is not None:
        input_paths.append(second_video)
    for path in input_paths:
        if not path.exists():
            raise SystemExit(f"Input file does not exist: {path}")

    encoder = choose_video_encoder(args.video_encoder)
    hwaccel = choose_hwaccel(args.hwaccel, encoder)
    print(f"Using mode: {args.mode}")
    print(f"Using video encoder: {encoder}")
    print(f"Using FFmpeg hwaccel: {hwaccel or 'none'}")

    duration = media_duration(audio_file)
    segments_json = args.segments_json.resolve()
    first_camera_index = 0
    if args.mode == MODE_SPLIT_VIDEO and args.first_speaker_side == "right":
        first_camera_index = 1

    if args.reuse_segments:
        timeline, mapping = read_segments_json(segments_json)
        speech_segments = read_speech_segments_json(segments_json)
        exclusive_segments = read_exclusive_speech_segments_json(segments_json)
        mapping_source = exclusive_segments or speech_segments or timeline
        if args.mode == MODE_SPLIT_VIDEO:
            mapping = speaker_indexes_by_detection_order(
                mapping_source,
                first_camera_index=first_camera_index,
            )
        if exclusive_segments:
            fallback_speaker = min(
                exclusive_segments,
                key=lambda segment: (segment.start, segment.end),
            ).speaker
            timeline = build_camera_timeline(
                exclusive_segments,
                duration=duration,
                fallback_speaker=fallback_speaker,
                min_switch_duration=args.min_switch_duration,
                silence_threshold=args.silence_threshold,
                silence_lookahead=args.silence_lookahead,
                gap_padding=args.gap_padding,
            )
        print(f"Reused diarization timeline: {segments_json}")
    else:
        raw_segments, exclusive_segments = diarize_audio(
            audio_file,
            model=args.model,
            hf_token=args.hf_token,
            device=args.device,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            num_speakers=args.num_speakers,
        )

        if not raw_segments or not exclusive_segments:
            raise SystemExit("Diarization produced no speaker segments.")

        mapping = speaker_indexes_by_detection_order(
            exclusive_segments,
            first_camera_index=first_camera_index,
        )
        fallback_speaker = min(
            exclusive_segments,
            key=lambda segment: (segment.start, segment.end),
        ).speaker
        timeline = build_camera_timeline(
            exclusive_segments,
            duration=duration,
            fallback_speaker=fallback_speaker,
            min_switch_duration=args.min_switch_duration,
            silence_threshold=args.silence_threshold,
            silence_lookahead=args.silence_lookahead,
            gap_padding=args.gap_padding,
        )

        segments_json.parent.mkdir(parents=True, exist_ok=True)
        write_segments_json(
            segments_json,
            timeline,
            mapping,
            speech_segments=raw_segments,
            exclusive_speech_segments=exclusive_segments,
        )
        print(f"Wrote diarization timeline: {segments_json}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_SPLIT_VIDEO:
        assemble_split_video(
            audio_file=audio_file,
            combined_video=first_video,
            output_video=output_video,
            timeline=timeline,
            mapping=mapping,
            encoder=encoder,
            audio_codec=args.audio_codec,
            preset=args.preset,
            crf=args.crf,
            hwaccel=hwaccel,
            loop_video=args.loop_speaker_videos,
            render_duration=args.render_duration,
        )
    else:
        assert second_video is not None
        assemble_video(
            audio_file=audio_file,
            camera_videos=[first_video, second_video],
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
    except DiarizationError as exc:
        raise SystemExit(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
