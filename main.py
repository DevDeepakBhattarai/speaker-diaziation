from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
MODE_SEPARATE_VIDEOS = "separate-videos"
MODE_SPLIT_VIDEO = "split-video"

# Filter-graph labels produced by the unified render pass.
SWITCHED_LABEL = "switched"
CAMERA_LABELS = ("camera0", "camera1")

_WINDOWS_DLL_HANDLES: list[object] = []
_ENCODER_SESSION_LIMITS: dict[str, int] = {}


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
    for encoder in ("h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox"):
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
            "Missing Python dependencies. Run `uv sync` to install the project dependencies."
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
    print(f"Using diarization batch size: {safe_batch_size}")

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


def split_video_geometry(source_width: int, source_height: int) -> tuple[int, int]:
    """Return the native half-crop size of a combined left/right recording.

    The crop is the output: each speaker's own pixels are kept at 1:1 density
    and delivered as-is. Padding them back onto the full source canvas would
    add nothing but black bars for the encoder to chew through.
    """
    crop_width = source_width // 2
    crop_width -= crop_width % 2
    crop_height = source_height - (source_height % 2)
    if crop_width < 2 or crop_height < 2:
        raise SystemExit(
            f"Combined video is too small to split: {source_width}x{source_height}"
        )
    return crop_width, crop_height


@dataclass(frozen=True)
class CameraAngle:
    """One camera view: an FFmpeg input plus the native crop that isolates it."""

    input_index: int
    source_size: tuple[int, int]
    crop: tuple[int, int, int, int] | None = None

    @property
    def size(self) -> tuple[int, int]:
        if self.crop is None:
            return self.source_size
        return self.crop[2], self.crop[3]


@dataclass(frozen=True)
class RenderPlan:
    """Mode-independent description of the two camera angles and their canvas.

    Both input modes reduce to the same thing: two camera angles drawn on one
    shared canvas. A combined recording contributes two crops of a single input;
    two camera files contribute one uncropped angle each. Everything downstream
    - the switched render, the Resolve camera media, and the OTIO document -
    reads this plan instead of branching on the mode.
    """

    mode: str
    video_inputs: tuple[Path, ...]
    cameras: tuple[CameraAngle, CameraAngle]
    canvas: tuple[int, int]
    camera_names: tuple[str, str]

    @property
    def canvas_width(self) -> int:
        return self.canvas[0]

    @property
    def canvas_height(self) -> int:
        return self.canvas[1]


def build_render_plan(
    mode: str,
    videos: Sequence[Path],
    *,
    sizes: Sequence[tuple[int, int]] | None = None,
) -> RenderPlan:
    """Reduce either input mode to the same two-angle render plan."""
    videos = tuple(videos)
    if sizes is None:
        sizes = [source_video_size(path) for path in videos]
    if len(sizes) != len(videos):
        raise SystemExit("Each video input needs exactly one source size")

    if mode == MODE_SPLIT_VIDEO:
        if len(videos) != 1:
            raise SystemExit("split-video mode needs exactly one combined video")
        source_width, source_height = sizes[0]
        crop_width, crop_height = split_video_geometry(source_width, source_height)
        right_x = source_width - crop_width
        return RenderPlan(
            mode=mode,
            video_inputs=videos,
            cameras=(
                CameraAngle(0, sizes[0], (0, 0, crop_width, crop_height)),
                CameraAngle(0, sizes[0], (right_x, 0, crop_width, crop_height)),
            ),
            canvas=(crop_width, crop_height),
            camera_names=("Left Speaker", "Right Speaker"),
        )

    if mode == MODE_SEPARATE_VIDEOS:
        if len(videos) != 2:
            raise SystemExit("separate-videos mode needs exactly two camera videos")
        cameras: list[CameraAngle] = []
        for input_index, size in enumerate(sizes):
            width, height = size
            even = (width - (width % 2), height - (height % 2))
            if even[0] < 2 or even[1] < 2:
                raise SystemExit(f"Camera video is too small: {width}x{height}")
            crop = None if even == size else (0, 0, even[0], even[1])
            cameras.append(CameraAngle(input_index, size, crop))
        # The canvas is the per-axis maximum so neither angle is ever downscaled;
        # a smaller angle is padded onto it instead of being stretched to fit.
        canvas = (
            max(camera.size[0] for camera in cameras),
            max(camera.size[1] for camera in cameras),
        )
        return RenderPlan(
            mode=mode,
            video_inputs=videos,
            cameras=(cameras[0], cameras[1]),
            canvas=canvas,
            camera_names=("Camera 1", "Camera 2"),
        )

    raise SystemExit(f"Unsupported mode: {mode}")


def camera_filter_chain(camera: CameraAngle, canvas: tuple[int, int]) -> list[str]:
    """Isolate one angle and centre it on the shared canvas at native density.

    Nothing is ever scaled. Cropping selects the angle's own pixels and padding
    restores the canvas, so a 3840x2160 source stays 3840x2160 without inventing
    or stretching a single pixel.
    """
    canvas_width, canvas_height = canvas
    width, height = camera.size
    if width > canvas_width or height > canvas_height:
        raise SystemExit(
            f"Camera angle {width}x{height} does not fit the "
            f"{canvas_width}x{canvas_height} canvas"
        )

    filters: list[str] = []
    if camera.crop is not None:
        crop_x, crop_y, crop_width, crop_height = camera.crop
        filters.append(f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}")
    if (width, height) != canvas:
        filters.append(
            f"pad={canvas_width}:{canvas_height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    filters.append("setsar=1")
    return filters


def build_switch_filter_graph(
    plan: RenderPlan,
    timeline: Sequence[Segment],
    mapping: dict[str, int],
    *,
    wanted: Sequence[str],
    hold_duration: float | None,
) -> str:
    """Build one filter graph that feeds every requested output from one decode.

    `wanted` names the labels the caller will map: the switched programme feed
    and/or the two full-length camera angles. Each source is decoded once and
    split only as many ways as the requested outputs actually need.
    """
    requested = list(dict.fromkeys(wanted))
    unknown = set(requested) - {SWITCHED_LABEL, *CAMERA_LABELS}
    if unknown:
        raise ValueError(f"unknown render outputs: {sorted(unknown)}")
    if not requested:
        raise ValueError("at least one render output is required")

    needed_cameras = {
        index for index, label in enumerate(CAMERA_LABELS) if label in requested
    }
    if SWITCHED_LABEL in requested:
        needed_cameras |= {0, 1}

    lines: list[str] = []
    source_label: dict[int, str] = {}
    consumers: dict[int, list[int]] = {}
    for index in sorted(needed_cameras):
        consumers.setdefault(plan.cameras[index].input_index, []).append(index)

    for input_index, camera_indexes in sorted(consumers.items()):
        labels = [f"src{index}" for index in camera_indexes]
        head = f"[{input_index}:v]setpts=PTS-STARTPTS"
        if len(labels) == 1:
            lines.append(f"{head}[{labels[0]}]")
        else:
            # One combined recording feeds both angles, so it is decoded once
            # and split rather than opened a second time.
            joined = "".join(f"[{label}]" for label in labels)
            lines.append(f"{head},split={len(labels)}{joined}")
        source_label.update(zip(camera_indexes, labels))

    for index in sorted(needed_cameras):
        chain = camera_filter_chain(plan.cameras[index], plan.canvas)
        if hold_duration is not None:
            # Hold the final frame when a camera file is shorter than the
            # soundtrack, so every output stays aligned to the master audio.
            chain.append(f"tpad=stop_mode=clone:stop_duration={hold_duration:.3f}")

        sinks: list[str] = []
        if CAMERA_LABELS[index] in requested:
            sinks.append(CAMERA_LABELS[index])
        if SWITCHED_LABEL in requested:
            sinks.append(f"mix{index}")
        body = f"[{source_label[index]}]" + ",".join(chain)
        if len(sinks) > 1:
            body += f",split={len(sinks)}"
        lines.append(body + "".join(f"[{sink}]" for sink in sinks))

    if SWITCHED_LABEL in requested:
        active_expression = speaker1_active_expression(timeline, mapping)
        lines.append(
            f"[mix0][mix1]overlay=enable='gt({active_expression}\\,0)':"
            f"x=0:y=0:eof_action=repeat:repeatlast=1[{SWITCHED_LABEL}]"
        )
    return ";\n".join(lines)


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
    if encoder.endswith("_videotoolbox"):
        return "videotoolbox"
    return None


def append_video_encoding_options(
    command: list[str],
    *,
    encoder: str,
    preset: str,
    crf: int,
    lossless: bool = False,
) -> None:
    if encoder in {"h264_nvenc", "hevc_nvenc"}:
        if lossless:
            command.extend(
                ["-preset", preset, "-tune", "lossless", "-rc", "constqp", "-qp", "0"]
            )
        else:
            command.extend(["-preset", preset, "-cq", str(crf)])
    elif encoder == "libx264":
        x264_preset = (
            preset
            if preset
            in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}
            else "veryfast"
        )
        command.extend(["-preset", x264_preset, "-crf", "0" if lossless else str(crf)])


@dataclass(frozen=True)
class RenderOutputs:
    """Every artifact the render pass should emit from one decode."""

    switched_video: Path
    camera_videos: tuple[Path, Path] | None = None
    master_audio: Path | None = None

    def labels(self) -> list[str]:
        labels = [SWITCHED_LABEL]
        if self.camera_videos is not None:
            labels.extend(CAMERA_LABELS)
        return labels


def _probe_encoder_sessions(encoder: str, count: int) -> bool:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        # Comfortably above NVENC's minimum frame size, still trivial to encode.
        "-i",
        "color=c=black:s=256x256:r=25:d=0.2",
    ]
    for _ in range(count):
        command.extend(
            ["-map", "0:v", "-c:v", encoder, "-frames:v", "1", "-f", "null", os.devnull]
        )
    try:
        run(command, quiet=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def max_parallel_encoder_sessions(encoder: str, desired: int) -> int:
    """Return how many simultaneous sessions this encoder will actually open.

    Consumer NVIDIA/Intel/AMD drivers cap concurrent hardware encode sessions.
    Probing a 64x64 clip up front costs about a second and avoids discovering
    the limit an hour into a 4K render.
    """
    if desired <= 1:
        return 1
    if not any(
        encoder.endswith(suffix)
        for suffix in ("_nvenc", "_qsv", "_amf", "_vaapi", "_videotoolbox")
    ):
        return desired

    cached = _ENCODER_SESSION_LIMITS.get(encoder)
    if cached is None:
        cached = 1
        for count in range(desired, 1, -1):
            if _probe_encoder_sessions(encoder, count):
                cached = count
                break
        _ENCODER_SESSION_LIMITS[encoder] = cached
    return min(cached, desired)


def _append_video_output(
    command: list[str],
    *,
    label: str,
    path: Path,
    plan: RenderPlan,
    encoder: str,
    preset: str,
    crf: int,
    duration: float,
    audio_input_index: int | None,
    audio_codec: str,
) -> None:
    command.extend(["-map", f"[{label}]"])
    if audio_input_index is None:
        command.append("-an")
    else:
        command.extend(["-map", f"{audio_input_index}:a:0", "-c:a", audio_codec])
    command.extend(
        ["-c:v", encoder, "-aspect", f"{plan.canvas_width}:{plan.canvas_height}"]
    )
    append_video_encoding_options(command, encoder=encoder, preset=preset, crf=crf)
    command.extend(
        ["-max_muxing_queue_size", "4096", "-t", f"{duration:.3f}", str(path)]
    )


def render_pipeline(
    *,
    plan: RenderPlan,
    audio_file: Path,
    timeline: Sequence[Segment],
    mapping: dict[str, int],
    outputs: RenderOutputs,
    encoder: str,
    audio_codec: str,
    preset: str,
    crf: int,
    hwaccel: str | None,
    loop_videos: bool,
    duration: float,
    max_parallel_encodes: int | None = None,
) -> int:
    """Render every requested artifact, decoding each source only once.

    The switched programme feed, the two full-length camera angles, and the
    master audio all come out of a single FFmpeg invocation. Only when the
    hardware encoder refuses that many concurrent sessions is the work split
    into the fewest additional passes that fit. Returns the number of passes.
    """
    video_labels = outputs.labels()
    limit = max_parallel_encodes or max_parallel_encoder_sessions(
        encoder, len(video_labels)
    )
    batches = [
        video_labels[start : start + limit]
        for start in range(0, len(video_labels), limit)
    ]

    resolved_audio = audio_file.resolve()
    audio_input_index: int | None = None
    for index, video in enumerate(plan.video_inputs):
        if video.resolve() == resolved_audio:
            audio_input_index = index
            break
    extra_audio_input = audio_input_index is None
    if audio_input_index is None:
        audio_input_index = len(plan.video_inputs)

    hold_duration = None if loop_videos else duration
    for batch_number, batch in enumerate(batches):
        filter_text = build_switch_filter_graph(
            plan,
            timeline,
            mapping,
            wanted=batch,
            hold_duration=hold_duration,
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".ffmpeg", delete=False, encoding="utf-8"
        ) as file:
            filter_path = Path(file.name)
            file.write(filter_text)

        command = ["ffmpeg", "-y", "-hide_banner"]
        for video in plan.video_inputs:
            if loop_videos:
                command.extend(["-stream_loop", "-1"])
            if hwaccel:
                # Decode on the GPU but hand frames back to system memory: crop,
                # pad, and overlay are software filters, and the encoders upload
                # again on their own.
                command.extend(["-hwaccel", hwaccel])
            command.extend(["-i", str(video)])
        if extra_audio_input:
            command.extend(["-i", str(audio_file)])
        command.extend(["-filter_complex_script", str(filter_path)])

        for label in batch:
            if label == SWITCHED_LABEL:
                _append_video_output(
                    command,
                    label=label,
                    path=outputs.switched_video,
                    plan=plan,
                    encoder=encoder,
                    preset=preset,
                    crf=crf,
                    duration=duration,
                    audio_input_index=audio_input_index,
                    audio_codec=audio_codec,
                )
                continue
            assert outputs.camera_videos is not None
            _append_video_output(
                command,
                label=label,
                path=outputs.camera_videos[CAMERA_LABELS.index(label)],
                plan=plan,
                encoder=encoder,
                preset=preset,
                crf=crf,
                duration=duration,
                audio_input_index=None,
                audio_codec=audio_codec,
            )

        if outputs.master_audio is not None and batch_number == 0:
            command.extend(
                [
                    "-map",
                    f"{audio_input_index}:a:0",
                    "-vn",
                    "-af",
                    "apad",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "48000",
                    "-t",
                    f"{duration:.3f}",
                    str(outputs.master_audio),
                ]
            )

        try:
            run(command)
        finally:
            filter_path.unlink(missing_ok=True)

    return len(batches)


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
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
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
    parser.add_argument("--video-encoder", default="auto")
    parser.add_argument("--audio-codec", default="aac")
    parser.add_argument("--preset", default="p4", help="Encoder preset. For NVENC, p1 is fastest.")
    parser.add_argument("--crf", type=int, default=23, help="CRF/CQ quality value.")
    parser.add_argument(
        "--hwaccel",
        default="auto",
        help=(
            "FFmpeg hardware decoder: auto, cuda, videotoolbox, qsv, dxva2, "
            "d3d11va, or none."
        ),
    )
    parser.add_argument(
        "--loop-speaker-videos",
        action="store_true",
        help="Loop the camera video(s) when the audio is longer.",
    )
    parser.add_argument("--segments-json", type=Path, default=Path("speaker_segments.json"))
    parser.add_argument(
        "--davinci-project",
        type=Path,
        help=(
            "Also write a portable DaVinci Resolve .otioz project. Its camera "
            "angles come out of the same FFmpeg pass as the switched video."
        ),
    )
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
    # Imported here because the job orchestrator imports this module back.
    import job

    parser = build_parser()
    args = parser.parse_args()

    if args.speaker0_video is None:
        parser.error("A video file is required.")
    if args.mode == MODE_SEPARATE_VIDEOS and args.speaker1_video is None:
        parser.error("separate-videos mode requires both speaker video files.")
    if args.mode == MODE_SPLIT_VIDEO and args.speaker1_video is not None:
        parser.error("split-video mode accepts exactly one combined video file.")

    videos = [args.speaker0_video.resolve()]
    if args.speaker1_video is not None:
        videos.append(args.speaker1_video.resolve())
    segments_json = args.segments_json.resolve()

    request = job.JobRequest(
        mode=args.mode,
        videos=tuple(videos),
        audio_file=args.audio_file.resolve(),
        output_video=args.output.resolve(),
        segments_json=segments_json,
        davinci_bundle=(
            args.davinci_project.resolve() if args.davinci_project else None
        ),
        first_camera_index=(
            1
            if args.mode == MODE_SPLIT_VIDEO and args.first_speaker_side == "right"
            else 0
        ),
        reuse_segments_from=segments_json if args.reuse_segments else None,
        model=args.model,
        hf_token=args.hf_token,
        device=args.device,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        min_switch_duration=args.min_switch_duration,
        silence_threshold=args.silence_threshold,
        silence_lookahead=args.silence_lookahead,
        gap_padding=args.gap_padding,
        video_encoder=args.video_encoder,
        audio_codec=args.audio_codec,
        preset=args.preset,
        crf=args.crf,
        hwaccel=args.hwaccel,
        loop_videos=args.loop_speaker_videos,
        render_duration=args.render_duration,
    )

    print(f"Using mode: {args.mode}")
    result = job.run_job(request, progress=lambda _, message: print(f"-> {message}"))

    print(f"Using video encoder: {result.encoder}")
    print(f"Using FFmpeg hwaccel: {result.hwaccel or 'none'}")
    print(f"Wrote diarization timeline: {result.segments_json}")
    print(f"Wrote switched video: {result.output_video}")
    for warning in result.warnings:
        print(f"Note: {warning}")
    if result.davinci_bundle is not None:
        print(f"Wrote DaVinci Resolve project: {result.davinci_bundle}")
    elif result.davinci_error is not None:
        print(
            "DaVinci Resolve project failed after a successful render: "
            f"{result.davinci_error}",
            file=sys.stderr,
        )


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
