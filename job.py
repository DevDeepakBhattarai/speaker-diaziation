"""One unified job: resolve a timeline, render every artifact, bundle the result.

Both entry points - the CLI in `main.py` and the Gradio app - describe the work
as a `JobRequest` and hand it to `run_job`. Whether the source is one combined
recording or two camera files stops mattering at `build_render_plan`: everything
after that point sees the same two camera angles on one canvas.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import davinci_export
import main as pipeline


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class JobRequest:
    mode: str
    videos: tuple[Path, ...]
    audio_file: Path
    output_video: Path
    segments_json: Path
    davinci_bundle: Path | None = None
    first_camera_index: int = 0
    reuse_segments_from: Path | None = None
    model: str = pipeline.DIARIZATION_MODEL
    hf_token: str | None = None
    device: str = "auto"
    num_speakers: int | None = 2
    min_speakers: int | None = None
    max_speakers: int | None = None
    min_switch_duration: float = 1.0
    silence_threshold: float = 2.5
    silence_lookahead: float = 5.0
    gap_padding: float = 0.05
    video_encoder: str = "auto"
    audio_codec: str = "aac"
    preset: str = "p4"
    crf: int = 23
    hwaccel: str = "auto"
    loop_videos: bool = False
    render_duration: float | None = None


@dataclass(frozen=True)
class JobResult:
    output_video: Path
    segments_json: Path
    plan: pipeline.RenderPlan
    timeline: list[pipeline.Segment]
    mapping: dict[str, int]
    duration: float
    encoder: str
    hwaccel: str | None
    ffmpeg_passes: int
    timeline_details: str
    davinci_bundle: Path | None = None
    davinci_error: str | None = None
    warnings: list[str] = field(default_factory=list)


def _report(progress: ProgressCallback | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(fraction, message)


def resolve_timeline(
    request: JobRequest,
    *,
    duration: float,
    progress: ProgressCallback | None = None,
) -> tuple[list[pipeline.Segment], dict[str, int], str]:
    """Produce the camera timeline and speaker-to-camera mapping for a job.

    Raw Pyannote output is never rewritten. Reused timelines rebuild the camera
    policy from the stored exclusive diarization so the switching controls still
    apply, and the mapping is recomputed so the first-speaker side is honoured.
    """
    if request.reuse_segments_from is not None:
        _report(progress, 0.12, "Reading the existing timeline")
        source = request.reuse_segments_from
        timeline, mapping = pipeline.read_segments_json(source)
        speech_segments = pipeline.read_speech_segments_json(source)
        exclusive_segments = pipeline.read_exclusive_speech_segments_json(source)
        try:
            mapping = pipeline.speaker_indexes_by_detection_order(
                exclusive_segments or speech_segments or timeline,
                first_camera_index=request.first_camera_index,
            )
        except SystemExit:
            # Legacy files may not carry two labelled speakers; keep their mapping.
            pass
        if exclusive_segments:
            timeline = _camera_timeline(request, exclusive_segments, duration=duration)
            details = (
                "Camera policy rebuilt from stored untouched Pyannote exclusive diarization"
            )
        else:
            details = "Legacy stored camera timeline reused unchanged"
        pipeline.write_segments_json(
            request.segments_json,
            timeline,
            mapping,
            speech_segments=speech_segments,
            exclusive_speech_segments=exclusive_segments,
        )
        return timeline, mapping, details

    _report(progress, 0.20, "Running raw-source GPU speaker diarization")
    raw_segments, exclusive_segments = pipeline.diarize_audio(
        request.audio_file,
        model=request.model,
        hf_token=request.hf_token,
        device=request.device,
        min_speakers=request.min_speakers,
        max_speakers=request.max_speakers,
        num_speakers=request.num_speakers,
    )
    if not raw_segments or not exclusive_segments:
        raise SystemExit("Diarization produced no speaker segments.")

    _report(progress, 0.62, "Building the model-native speaker timeline")
    mapping = pipeline.speaker_indexes_by_detection_order(
        exclusive_segments,
        first_camera_index=request.first_camera_index,
    )
    timeline = _camera_timeline(request, exclusive_segments, duration=duration)
    pipeline.write_segments_json(
        request.segments_json,
        timeline,
        mapping,
        speech_segments=raw_segments,
        exclusive_speech_segments=exclusive_segments,
    )
    return (
        timeline,
        mapping,
        "Camera policy generated from untouched Pyannote exclusive diarization",
    )


def _camera_timeline(
    request: JobRequest,
    exclusive_segments: Sequence[pipeline.Segment],
    *,
    duration: float,
) -> list[pipeline.Segment]:
    fallback_speaker = min(
        exclusive_segments,
        key=lambda segment: (segment.start, segment.end),
    ).speaker
    return pipeline.build_camera_timeline(
        list(exclusive_segments),
        duration=duration,
        fallback_speaker=fallback_speaker,
        min_switch_duration=request.min_switch_duration,
        silence_threshold=request.silence_threshold,
        silence_lookahead=request.silence_lookahead,
        gap_padding=request.gap_padding,
    )


def run_job(
    request: JobRequest,
    *,
    progress: ProgressCallback | None = None,
) -> JobResult:
    """Run the whole pipeline: diarize, render, and optionally bundle for Resolve."""
    pipeline.require_tool("ffmpeg")
    pipeline.require_tool("ffprobe")

    _report(progress, 0.02, "Validating local paths")
    for path in (request.audio_file, *request.videos):
        if not path.exists():
            raise SystemExit(f"Input file does not exist: {path}")

    plan = pipeline.build_render_plan(request.mode, request.videos)
    encoder = pipeline.choose_video_encoder(request.video_encoder)
    hwaccel = pipeline.choose_hwaccel(request.hwaccel, encoder)

    full_duration = pipeline.media_duration(request.audio_file)
    duration = (
        min(full_duration, request.render_duration)
        if request.render_duration
        else full_duration
    )

    timeline, mapping, timeline_details = resolve_timeline(
        request,
        duration=duration,
        progress=progress,
    )

    request.output_video.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    davinci_bundle: Path | None = None
    davinci_error: str | None = None

    if request.davinci_bundle is None:
        _report(progress, 0.68, "Rendering the switched video with FFmpeg")
        passes = pipeline.render_pipeline(
            plan=plan,
            audio_file=request.audio_file,
            timeline=timeline,
            mapping=mapping,
            outputs=pipeline.RenderOutputs(switched_video=request.output_video),
            encoder=encoder,
            audio_codec=request.audio_codec,
            preset=request.preset,
            crf=request.crf,
            hwaccel=hwaccel,
            loop_videos=request.loop_videos,
            duration=duration,
        )
    else:
        request.davinci_bundle.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="davinci-media-",
            dir=request.davinci_bundle.parent,
        ) as tmpdir:
            camera_videos, master_audio = davinci_export.bundle_media_paths(Path(tmpdir))
            _report(
                progress,
                0.68,
                "Rendering the switched video and Resolve camera angles together",
            )
            passes = pipeline.render_pipeline(
                plan=plan,
                audio_file=request.audio_file,
                timeline=timeline,
                mapping=mapping,
                outputs=pipeline.RenderOutputs(
                    switched_video=request.output_video,
                    camera_videos=camera_videos,
                    master_audio=master_audio,
                ),
                encoder=encoder,
                audio_codec=request.audio_codec,
                preset=request.preset,
                crf=request.crf,
                hwaccel=hwaccel,
                loop_videos=request.loop_videos,
                duration=duration,
            )
            if passes > 1:
                warnings.append(
                    f"{encoder} refused three concurrent encode sessions, so the "
                    f"render needed {passes} FFmpeg passes instead of one."
                )

            _report(progress, 0.90, "Packaging the portable DaVinci Resolve project")
            try:
                davinci_bundle = davinci_export.create_davinci_bundle(
                    plan=plan,
                    audio_file=request.audio_file,
                    segments_json=request.segments_json,
                    timeline=timeline,
                    mapping=mapping,
                    output_bundle=request.davinci_bundle,
                    camera_videos=camera_videos,
                    master_audio=master_audio,
                    duration=duration,
                )
            except Exception as exc:  # the render itself is already complete
                request.davinci_bundle.unlink(missing_ok=True)
                davinci_error = f"{type(exc).__name__}: {exc}"

    _report(progress, 1.0, "Done")
    return JobResult(
        output_video=request.output_video,
        segments_json=request.segments_json,
        plan=plan,
        timeline=timeline,
        mapping=mapping,
        duration=duration,
        encoder=encoder,
        hwaccel=hwaccel,
        ffmpeg_passes=passes,
        timeline_details=timeline_details,
        davinci_bundle=davinci_bundle,
        davinci_error=davinci_error,
        warnings=warnings,
    )
