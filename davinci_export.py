from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any, Sequence

import main as pipeline


OTIOZ_VERSION = "1.0.0"


def _round4(value: float) -> float:
    return round(float(value), 4)


def _frames(seconds: float, fps: float) -> int:
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    return max(0, int(round(float(seconds) * fps)))


def _rational_time(frames: int, fps: float) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "RationalTime.1",
        "value": max(0, int(frames)),
        "rate": float(fps),
    }


def _time_range(start_seconds: float, duration_seconds: float, fps: float) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "TimeRange.1",
        "start_time": _rational_time(_frames(start_seconds, fps), fps),
        "duration": _rational_time(max(1, _frames(duration_seconds, fps)), fps),
    }


def trim_timeline(
    timeline: Sequence[pipeline.Segment],
    *,
    duration: float,
) -> list[pipeline.Segment]:
    """Clamp a camera timeline to the exported record duration."""
    if duration <= 0:
        return []

    trimmed: list[pipeline.Segment] = []
    for item in timeline:
        start = max(0.0, min(duration, float(item.start)))
        end = max(0.0, min(duration, float(item.end)))
        if end <= start:
            continue
        if trimmed and trimmed[-1].speaker == item.speaker and math.isclose(
            trimmed[-1].end,
            start,
            abs_tol=1e-6,
        ):
            previous = trimmed[-1]
            trimmed[-1] = pipeline.Segment(previous.start, end, previous.speaker)
        else:
            trimmed.append(pipeline.Segment(start, end, item.speaker))
    return trimmed


def video_fps(path: Path) -> float:
    """Read a usable timeline frame rate from the first video stream."""
    for stream in pipeline.ffprobe_json(path).get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        for key in ("avg_frame_rate", "r_frame_rate"):
            value = str(stream.get(key) or "")
            if not value or value == "0/0":
                continue
            if "/" in value:
                numerator, denominator = value.split("/", 1)
                rate = float(numerator) / float(denominator)
            else:
                rate = float(value)
            if rate > 0:
                return rate
    return 30.0


def _external_reference(
    *,
    media_name: str,
    duration: float,
    fps: float,
    source_kind: str,
) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "ExternalReference.1",
        "target_url": f"media/{media_name}",
        "available_range": _time_range(0.0, duration, fps),
        "available_image_bounds": None,
        "name": "",
        "metadata": {
            "speaker_diarization": {
                "source_kind": source_kind,
                "bundled_media": True,
            }
        },
    }


def _video_clip(
    item: pipeline.Segment,
    *,
    camera_index: int,
    camera_name: str,
    media_name: str,
    timeline_duration: float,
    fps: float,
    event_number: int,
) -> dict[str, Any]:
    clip_duration = item.end - item.start
    return {
        "OTIO_SCHEMA": "Clip.2",
        "name": f"{camera_name} - {item.speaker}",
        "media_references": {
            "DEFAULT_MEDIA": _external_reference(
                media_name=media_name,
                duration=timeline_duration,
                fps=fps,
                source_kind=f"camera_{camera_index}",
            )
        },
        "active_media_reference_key": "DEFAULT_MEDIA",
        "source_range": _time_range(item.start, clip_duration, fps),
        "effects": [],
        "markers": [],
        "enabled": True,
        "color": None,
        "metadata": {
            "speaker_diarization": {
                "event_number": event_number,
                "speaker": item.speaker,
                "camera_index": camera_index,
                "record_start": _round4(item.start),
                "record_end": _round4(item.end),
                "source_start": _round4(item.start),
                "source_end": _round4(item.end),
            }
        },
    }


def _audio_clip(
    *,
    media_name: str,
    timeline_duration: float,
    fps: float,
) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "Clip.2",
        "name": "Diarization master audio",
        "media_references": {
            "DEFAULT_MEDIA": _external_reference(
                media_name=media_name,
                duration=timeline_duration,
                fps=fps,
                source_kind="master_audio",
            )
        },
        "active_media_reference_key": "DEFAULT_MEDIA",
        "source_range": _time_range(0.0, timeline_duration, fps),
        "effects": [],
        "markers": [],
        "enabled": True,
        "color": None,
        "metadata": {
            "speaker_diarization": {
                "record_start": 0.0,
                "record_end": _round4(timeline_duration),
                "master_audio": True,
            }
        },
    }


def build_otio_document(
    timeline: Sequence[pipeline.Segment],
    mapping: dict[str, int],
    *,
    title: str,
    fps: float,
    width: int,
    height: int,
    camera_media_names: Sequence[str],
    camera_names: Sequence[str],
    camera_resolutions: Sequence[tuple[int, int]],
    audio_media_name: str,
) -> dict[str, Any]:
    if not timeline:
        raise ValueError("the camera timeline is empty")
    if (
        len(camera_media_names) != 2
        or len(camera_names) != 2
        or len(camera_resolutions) != 2
    ):
        raise ValueError(
            "exactly two camera media names, labels, and resolutions are required"
        )
    if fps <= 0:
        raise ValueError("fps must be greater than zero")

    duration = float(timeline[-1].end)
    video_children: list[dict[str, Any]] = []
    for event_number, item in enumerate(timeline, start=1):
        if item.speaker not in mapping:
            raise ValueError(f"speaker has no camera mapping: {item.speaker}")
        camera_index = int(mapping[item.speaker])
        if camera_index not in {0, 1}:
            raise ValueError(f"invalid camera index for {item.speaker}: {camera_index}")
        video_children.append(
            _video_clip(
                item,
                camera_index=camera_index,
                camera_name=camera_names[camera_index],
                media_name=camera_media_names[camera_index],
                timeline_duration=duration,
                fps=fps,
                event_number=event_number,
            )
        )

    video_track = {
        "OTIO_SCHEMA": "Track.1",
        "name": "V1 - Active Speaker",
        "kind": "Video",
        "children": video_children,
        "metadata": {
            "speaker_diarization": {
                "contiguous_record_timeline": True,
                "editable_camera_cuts": True,
            }
        },
        "effects": [],
        "markers": [],
        "enabled": True,
        "source_range": None,
    }
    audio_track = {
        "OTIO_SCHEMA": "Track.1",
        "name": "A1 - Master Audio",
        "kind": "Audio",
        "children": [
            _audio_clip(
                media_name=audio_media_name,
                timeline_duration=duration,
                fps=fps,
            )
        ],
        "metadata": {
            "speaker_diarization": {
                "continuous_master_audio": True,
            }
        },
        "effects": [],
        "markers": [],
        "enabled": True,
        "source_range": None,
    }

    return {
        "OTIO_SCHEMA": "Timeline.1",
        "name": title.strip() or "Speaker Diarization Edit",
        "global_start_time": _rational_time(0, fps),
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "name": "tracks",
            "children": [video_track, audio_track],
            "effects": [],
            "markers": [],
            "enabled": True,
            "metadata": {
                "speaker_diarization": {
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "duration_seconds": _round4(duration),
                    "camera_resolutions": [
                        {"width": size[0], "height": size[1]}
                        for size in camera_resolutions
                    ],
                }
            },
            "source_range": None,
        },
        "metadata": {
            "speaker_diarization": {
                "format": "portable_otioz_active_speaker_edit",
                "event_count": len(video_children),
                "duration_seconds": _round4(duration),
                "speaker_to_camera": mapping,
                "notes": [
                    "V1 contains the editable active-speaker cuts.",
                    "A1 contains a continuous 48 kHz AAC master audio track.",
                    "Camera media keeps native source resolution; split-screen crops keep the native pixels of each half.",
                ],
            }
        },
    }


def build_manifest(
    timeline: Sequence[pipeline.Segment],
    mapping: dict[str, int],
    *,
    mode: str,
    fps: float,
    width: int,
    height: int,
    camera_names: Sequence[str],
    camera_resolutions: Sequence[tuple[int, int]],
    original_inputs: Sequence[Path],
) -> dict[str, Any]:
    duration = float(timeline[-1].end if timeline else 0.0)
    return {
        "kind": "speaker_diarization_davinci_handoff",
        "format": "opentimelineio_bundle",
        "mode": mode,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": _round4(duration),
        "speaker_to_camera": mapping,
        "camera_names": list(camera_names),
        "camera_resolutions": [
            {"width": size[0], "height": size[1]}
            for size in camera_resolutions
        ],
        "original_inputs": [str(path.resolve()) for path in original_inputs],
        "events": [
            {
                "number": index,
                "speaker": item.speaker,
                "camera_index": mapping[item.speaker],
                "record_start": _round4(item.start),
                "record_end": _round4(item.end),
                "duration": _round4(item.end - item.start),
            }
            for index, item in enumerate(timeline, start=1)
        ],
        "notes": [
            "The bundled camera angles are full-length, time-aligned, native-density sources for trim freedom in Resolve.",
            "The OTIO record timeline preserves the exact camera switches produced by the diarization app.",
            "The camera angles and the rendered MP4 come from the same FFmpeg pass, so they are frame-identical.",
        ],
    }


def write_otioz_bundle(
    output: Path,
    *,
    document: dict[str, Any],
    media_files: Sequence[Path],
    manifest: dict[str, Any],
    segments_json: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    basenames = [path.name for path in media_files]
    if len(set(basenames)) != len(basenames):
        raise ValueError("OTIOZ media files must have unique basenames")

    partial_output = output.with_name(f".{output.name}.partial")
    partial_output.unlink(missing_ok=True)

    readme = (
        "DaVinci Resolve import\n"
        "======================\n\n"
        "1. Open DaVinci Resolve and create/open a project.\n"
        "2. Use File > Import > Timeline and select this .otioz file.\n"
        "3. Keep the imported frame rate when Resolve asks.\n"
        "4. V1 contains active-speaker cuts; A1 is the continuous master audio.\n\n"
        "The bundle also contains davinci_manifest.json and speaker_segments.json "
        "for auditing or regenerating the edit.\n"
    )

    try:
        with zipfile.ZipFile(partial_output, "w", allowZip64=True) as archive:
            archive.writestr(
                "content.otio",
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                compress_type=zipfile.ZIP_DEFLATED,
            )
            archive.writestr(
                "version.txt",
                OTIOZ_VERSION,
                compress_type=zipfile.ZIP_DEFLATED,
            )
            archive.writestr(
                "davinci_manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                compress_type=zipfile.ZIP_DEFLATED,
            )
            archive.writestr(
                "README.txt",
                readme,
                compress_type=zipfile.ZIP_DEFLATED,
            )
            archive.write(
                segments_json,
                arcname="speaker_segments.json",
                compress_type=zipfile.ZIP_DEFLATED,
            )
            for media_file in media_files:
                # OTIOZ media is conventionally stored without ZIP recompression.
                archive.write(
                    media_file,
                    arcname=f"media/{media_file.name}",
                    compress_type=zipfile.ZIP_STORED,
                )
        partial_output.replace(output)
    finally:
        partial_output.unlink(missing_ok=True)


CAMERA_MEDIA_NAMES = ("camera_1.mp4", "camera_2.mp4")
AUDIO_MEDIA_NAME = "master_audio.m4a"


def bundle_media_paths(directory: Path) -> tuple[tuple[Path, Path], Path]:
    """Return the camera and audio media paths the render pass should write."""
    camera_videos = (
        directory / CAMERA_MEDIA_NAMES[0],
        directory / CAMERA_MEDIA_NAMES[1],
    )
    return camera_videos, directory / AUDIO_MEDIA_NAME


def create_davinci_bundle(
    *,
    plan: pipeline.RenderPlan,
    audio_file: Path,
    segments_json: Path,
    timeline: Sequence[pipeline.Segment],
    mapping: dict[str, int],
    output_bundle: Path,
    camera_videos: Sequence[Path],
    master_audio: Path,
    duration: float,
) -> Path:
    """Package already-rendered camera media into a portable OTIOZ project.

    The camera angles and master audio come straight out of the unified render
    pass, so this step never re-encodes anything: it only reads the frame rate
    off the rendered media and writes the OTIO document, manifest, and archive.
    """
    if len(camera_videos) != 2:
        raise ValueError("exactly two rendered camera videos are required")

    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    export_timeline = trim_timeline(timeline, duration=duration)
    if not export_timeline:
        raise ValueError("no camera events remain after trimming the timeline")

    width, height = plan.canvas
    camera_resolutions = [plan.canvas, plan.canvas]
    fps = video_fps(camera_videos[0])
    title = output_bundle.stem.replace("_", " ").strip().title()

    document = build_otio_document(
        export_timeline,
        mapping,
        title=title,
        fps=fps,
        width=width,
        height=height,
        camera_media_names=[path.name for path in camera_videos],
        camera_names=plan.camera_names,
        camera_resolutions=camera_resolutions,
        audio_media_name=master_audio.name,
    )
    manifest = build_manifest(
        export_timeline,
        mapping,
        mode=plan.mode,
        fps=fps,
        width=width,
        height=height,
        camera_names=plan.camera_names,
        camera_resolutions=camera_resolutions,
        original_inputs=[audio_file, *plan.video_inputs],
    )
    write_otioz_bundle(
        output_bundle,
        document=document,
        media_files=[*camera_videos, master_audio],
        manifest=manifest,
        segments_json=segments_json,
    )
    return output_bundle
