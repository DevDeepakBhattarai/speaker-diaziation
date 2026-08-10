from __future__ import annotations

import time
import traceback
from pathlib import Path

import gradio as gr

import davinci_export
import main as pipeline


WORK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORK_DIR / "outputs"
UI_MODE_SEPARATE = "Two separate speaker videos"
UI_MODE_SPLIT = "One combined left/right video"
UI_AUDIO_EMBEDDED = "Use audio from the video"
UI_AUDIO_SEPARATE = "Use a separate local audio file"


def _resolve_local_file(path: str | None, label: str) -> Path:
    if not path or not path.strip():
        raise gr.Error(f"Enter the full local path for {label}.")

    # Windows Explorer's "Copy as path" includes surrounding quotes.
    normalized = path.strip().strip('"').strip("'")
    source = Path(normalized).expanduser()
    if not source.is_absolute():
        source = WORK_DIR / source
    source = source.resolve()

    if not source.is_file():
        raise gr.Error(f"{label} does not exist or is not a file: {source}")
    return source


def _serve_outputs_in_place(*paths: Path | None) -> None:
    existing = [path for path in paths if path is not None and path.exists()]
    if existing:
        # Prevent Gradio from copying multi-gigabyte outputs into its cache.
        gr.set_static_paths(paths=existing)


def _validate_full_render(output_video: Path, *, expected_duration: float) -> float:
    rendered_duration = pipeline.media_duration(output_video)
    duration_tolerance = max(1.0, min(5.0, expected_duration * 0.001))
    if rendered_duration + duration_tolerance < expected_duration:
        raise gr.Error(
            "The rendered video is incomplete: "
            f"expected approximately {expected_duration:.3f} seconds but FFmpeg produced "
            f"only {rendered_duration:.3f} seconds. The partial output was left at "
            f"{output_video} for debugging."
        )
    return rendered_duration


def _update_input_fields(mode: str, audio_source: str) -> tuple[dict, dict, dict, dict, dict]:
    split_mode = mode == UI_MODE_SPLIT
    separate_audio = audio_source == UI_AUDIO_SEPARATE
    return (
        gr.update(visible=separate_audio),
        gr.update(visible=not split_mode),
        gr.update(visible=not split_mode),
        gr.update(visible=split_mode),
        gr.update(visible=split_mode),
    )


def _run_switcher(
    mode: str,
    audio_source: str,
    audio_file: str | None,
    speaker0_video: str | None,
    speaker1_video: str | None,
    combined_video: str | None,
    first_speaker_side: str,
    segments_json_file: str | None,
    reuse_segments: bool,
    model: str,
    device: str,
    hwaccel: str,
    video_encoder: str,
    preset: str,
    crf: int,
    min_switch_duration: float,
    silence_threshold: float,
    silence_lookahead: float,
    loop_speaker_videos: bool,
    create_davinci_project: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str | None, str | None, str | None, str]:
    split_mode = mode == UI_MODE_SPLIT
    separate_audio = audio_source == UI_AUDIO_SEPARATE
    if separate_audio and not audio_file:
        raise gr.Error("Enter the separate soundtrack path, or choose audio from the video.")
    if split_mode and not combined_video:
        raise gr.Error("Enter the path to one combined left/right video.")
    if not split_mode and (not speaker0_video or not speaker1_video):
        raise gr.Error("Enter both local speaker-video paths.")

    if reuse_segments and not segments_json_file:
        raise gr.Error("Enter an existing speaker_segments.json path or disable reuse.")

    started = time.perf_counter()
    job_dir = OUTPUT_DIR / time.strftime("%Y%m%d-%H%M%S")
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress(0.02, desc="Validating local paths")
        speaker0_path: Path | None = None
        speaker1_path: Path | None = None
        combined_path: Path | None = None
        if split_mode:
            combined_path = _resolve_local_file(combined_video, "the combined video")
            output_video = job_dir / "speaker_split_switched.mp4"
            primary_video_path = combined_path
        else:
            speaker0_path = _resolve_local_file(speaker0_video, "the first speaker video")
            speaker1_path = _resolve_local_file(speaker1_video, "the second speaker video")
            output_video = job_dir / "speaker_switched.mp4"
            primary_video_path = speaker0_path
        segments_json = job_dir / "speaker_segments.json"

        pipeline.require_tool("ffmpeg")
        pipeline.require_tool("ffprobe")
        if separate_audio:
            audio_path = _resolve_local_file(audio_file, "the separate soundtrack")
            audio_details = "Separate local audio path"
        else:
            audio_path = primary_video_path
            if not pipeline.media_has_audio(audio_path):
                raise gr.Error(
                    "The selected video has no audio track. Enter a separate audio-file path instead."
                )
            audio_details = "Audio embedded in the local video"
        encoder = pipeline.choose_video_encoder(video_encoder)
        selected_hwaccel = pipeline.choose_hwaccel(hwaccel, encoder)
        duration = pipeline.media_duration(audio_path)
        first_camera_index = 1 if split_mode and first_speaker_side == "Right" else 0

        if reuse_segments:
            progress(0.12, desc="Reading existing timeline")
            source_segments = _resolve_local_file(
                segments_json_file, "the existing speaker_segments.json"
            )
            timeline, mapping = pipeline.read_segments_json(source_segments)
            speech_segments = pipeline.read_speech_segments_json(source_segments)
            exclusive_segments = pipeline.read_exclusive_speech_segments_json(source_segments)
            mapping_source = exclusive_segments or speech_segments or timeline
            if split_mode:
                mapping = pipeline.speaker_indexes_by_detection_order(
                    mapping_source,
                    first_camera_index=first_camera_index,
                )
            if exclusive_segments:
                fallback_speaker = min(
                    exclusive_segments,
                    key=lambda segment: (segment.start, segment.end),
                ).speaker
                timeline = pipeline.build_camera_timeline(
                    exclusive_segments,
                    duration=duration,
                    fallback_speaker=fallback_speaker,
                    min_switch_duration=min_switch_duration,
                    silence_threshold=silence_threshold,
                    silence_lookahead=silence_lookahead,
                )
            pipeline.write_segments_json(
                segments_json,
                timeline,
                mapping,
                speech_segments=speech_segments,
                exclusive_speech_segments=exclusive_segments,
            )
            timeline_details = (
                "Camera policy rebuilt from stored untouched Pyannote exclusive diarization"
                if exclusive_segments
                else "Legacy stored camera timeline reused unchanged"
            )
        else:
            progress(0.20, desc="Running raw-source GPU speaker diarization")
            raw_segments, exclusive_segments = pipeline.diarize_audio(
                audio_path,
                model=model,
                hf_token=None,
                device=device,
                min_speakers=None,
                max_speakers=None,
                num_speakers=2,
            )

            if not raw_segments or not exclusive_segments:
                raise gr.Error("Diarization produced no speaker segments.")

            progress(0.62, desc="Building model-native speaker timeline")
            mapping = pipeline.speaker_indexes_by_detection_order(
                exclusive_segments,
                first_camera_index=first_camera_index,
            )
            fallback_speaker = min(
                exclusive_segments,
                key=lambda segment: (segment.start, segment.end),
            ).speaker
            timeline = pipeline.build_camera_timeline(
                exclusive_segments,
                duration=duration,
                fallback_speaker=fallback_speaker,
                min_switch_duration=min_switch_duration,
                silence_threshold=silence_threshold,
                silence_lookahead=silence_lookahead,
            )
            pipeline.write_segments_json(
                segments_json,
                timeline,
                mapping,
                speech_segments=raw_segments,
                exclusive_speech_segments=exclusive_segments,
            )
            timeline_details = "Camera policy generated from untouched Pyannote exclusive diarization"

        progress(0.68, desc="Rendering switched video with FFmpeg")
        if split_mode:
            assert combined_path is not None
            pipeline.assemble_split_video(
                audio_file=audio_path,
                combined_video=combined_path,
                output_video=output_video,
                timeline=timeline,
                mapping=mapping,
                encoder=encoder,
                audio_codec="aac",
                preset=preset,
                crf=crf,
                hwaccel=selected_hwaccel,
                loop_video=loop_speaker_videos,
                render_duration=None,
            )
            mode_details = f"Split-video mode; first detected speaker: {first_speaker_side.lower()}"
        else:
            assert speaker0_path is not None and speaker1_path is not None
            pipeline.assemble_video(
                audio_file=audio_path,
                camera_videos=[speaker0_path, speaker1_path],
                output_video=output_video,
                timeline=timeline,
                mapping=mapping,
                encoder=encoder,
                audio_codec="aac",
                preset=preset,
                crf=crf,
                hwaccel=selected_hwaccel,
                loop_cameras=loop_speaker_videos,
                render_duration=None,
            )
            mode_details = "Two separate speaker videos"

        rendered_duration = _validate_full_render(
            output_video,
            expected_duration=duration,
        )

        davinci_bundle: Path | None = None
        davinci_error: str | None = None
        if create_davinci_project:
            progress(0.86, desc="Creating portable DaVinci Resolve project")
            davinci_bundle = job_dir / "speaker_edit.otioz"
            if split_mode:
                assert combined_path is not None
                export_mode = pipeline.MODE_SPLIT_VIDEO
                export_cameras = [combined_path]
            else:
                assert speaker0_path is not None and speaker1_path is not None
                export_mode = pipeline.MODE_SEPARATE_VIDEOS
                export_cameras = [speaker0_path, speaker1_path]

            try:
                davinci_export.create_davinci_otioz(
                    mode=export_mode,
                    audio_file=audio_path,
                    camera_videos=export_cameras,
                    segments_json=segments_json,
                    timeline=timeline,
                    mapping=mapping,
                    output_bundle=davinci_bundle,
                    encoder=encoder,
                    preset=preset,
                    crf=crf,
                    hwaccel=selected_hwaccel,
                    loop_cameras=loop_speaker_videos,
                    render_duration=None,
                )
            except Exception as exc:
                davinci_bundle.unlink(missing_ok=True)
                davinci_error = f"{type(exc).__name__}: {exc}"
                print(f"DaVinci export failed after the video render completed: {davinci_error}")
                davinci_bundle = None

        elapsed = time.perf_counter() - started
        if davinci_bundle:
            davinci_details = f"DaVinci project: {davinci_bundle}\n"
        elif davinci_error:
            davinci_details = (
                "DaVinci project: export failed, but the rendered video and timeline are valid. "
                f"{davinci_error}\n"
            )
        else:
            davinci_details = "DaVinci project: not requested\n"
        status = (
            f"Done in {elapsed / 60:.1f} minutes.\n"
            f"Mode: {mode_details}\n"
            f"Audio: {audio_details}\n"
            f"Camera debounce: {min_switch_duration:.2f}s; long silence: "
            f"{silence_threshold:.2f}s; look-ahead: {silence_lookahead:.2f}s\n"
            f"Rendered duration: {rendered_duration:.3f} seconds (full source)\n"
            f"Timeline behavior: {timeline_details}\n"
            f"Output: {output_video}\n"
            f"Timeline: {segments_json}\n"
            f"{davinci_details}"
            f"Encoder: {encoder}; hwaccel: {selected_hwaccel or 'none'}"
        )
        _serve_outputs_in_place(output_video, segments_json, davinci_bundle)
        progress(1.0, desc="Done")
        return (
            str(output_video),
            str(segments_json),
            str(davinci_bundle) if davinci_bundle else None,
            status,
        )
    except gr.Error:
        raise
    except Exception as exc:
        details = traceback.format_exc(limit=8)
        raise gr.Error(f"{exc}\n\n{details}") from exc


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Speaker Diarization Video Switcher") as demo:
        gr.Markdown("# Speaker Diarization Video Switcher")
        gr.Markdown(
            "Create a stable active-speaker video from two camera files, or crop and "
            "switch between the left and right halves of one combined recording."
        )
        gr.Markdown(
            "**Large-file mode:** paste full local file paths below. The app reads the "
            "original files directly, so Gradio does not upload them and the job does not "
            "make duplicate input copies."
        )

        with gr.Group():
            gr.Markdown("### 1. Choose the video layout")
            mode = gr.Radio(
                [UI_MODE_SEPARATE, UI_MODE_SPLIT],
                value=UI_MODE_SEPARATE,
                label="Video input mode",
            )

            with gr.Row():
                speaker0_video = gr.Textbox(
                    label="First detected speaker video - full local path",
                    placeholder=r"D:\Podcast\episode\camera_left.mp4",
                )
                speaker1_video = gr.Textbox(
                    label="Second detected speaker video - full local path",
                    placeholder=r"D:\Podcast\episode\camera_right.mp4",
                )

            with gr.Row():
                combined_video = gr.Textbox(
                    label="Combined video (left person | right person) - full local path",
                    placeholder=r"D:\Podcast\episode\combined.mp4",
                    visible=False,
                )
                first_speaker_side = gr.Radio(
                    ["Left", "Right"],
                    value="Left",
                    label="Where is the first detected speaker?",
                    info="This maps pyannote's first speaker label to the correct cropped half.",
                    visible=False,
                )

        with gr.Group():
            gr.Markdown("### 2. Choose the soundtrack")
            audio_source = gr.Radio(
                [UI_AUDIO_EMBEDDED, UI_AUDIO_SEPARATE],
                value=UI_AUDIO_EMBEDDED,
                label="Audio source",
                info=(
                    "Embedded mode uses the combined video audio, or the first video audio "
                    "when two separate videos are selected."
                ),
            )
            audio_file = gr.Textbox(
                label="Separate soundtrack / audio file - full local path",
                placeholder=r"D:\Podcast\episode\master_audio.wav",
                visible=False,
            )

        input_outputs = [
            audio_file,
            speaker0_video,
            speaker1_video,
            combined_video,
            first_speaker_side,
        ]
        mode.change(
            _update_input_fields,
            inputs=[mode, audio_source],
            outputs=input_outputs,
        )
        audio_source.change(
            _update_input_fields,
            inputs=[mode, audio_source],
            outputs=input_outputs,
        )

        with gr.Accordion("Existing timeline", open=False):
            reuse_segments = gr.Checkbox(
                label="Reuse an existing speaker_segments.json",
                value=False,
            )
            segments_json_file = gr.Textbox(
                label="Existing speaker_segments.json - full local path",
                placeholder=r"D:\Podcast\episode\speaker_segments.json",
            )

        with gr.Accordion("Camera switching", open=True):
            gr.Markdown(
                "Pyannote diarization remains untouched. These controls affect only the "
                "camera timeline sent to FFmpeg: silence holds the current view, and a "
                "bounded look-ahead can confirm a short turn after a long silence."
            )
            with gr.Row():
                min_switch_duration = gr.Slider(
                    0.0,
                    3.0,
                    value=1.0,
                    step=0.05,
                    label="Minimum visible speaker turn (seconds)",
                    info="Camera-only debounce. Raw diarization is never filtered.",
                )
                silence_threshold = gr.Slider(
                    0.0,
                    10.0,
                    value=2.5,
                    step=0.1,
                    label="Long silence threshold (seconds)",
                )
                silence_lookahead = gr.Slider(
                    0.0,
                    10.0,
                    value=5.0,
                    step=0.25,
                    label="Post-silence look-ahead (seconds)",
                )

        with gr.Accordion("Speed and quality", open=False):
            with gr.Row():
                preset = gr.Dropdown(
                    ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                    value="p1",
                    label="NVENC preset",
                )
                crf = gr.Slider(18, 35, value=26, step=1, label="CQ quality")
            with gr.Row():
                device = gr.Dropdown(
                    ["cuda", "auto", "cpu"],
                    value="cuda",
                    label="Diarization device",
                )
                hwaccel = gr.Dropdown(
                    ["cuda", "auto", "none"],
                    value="cuda",
                    label="FFmpeg hardware acceleration",
                )
                video_encoder = gr.Dropdown(
                    ["h264_nvenc", "hevc_nvenc", "auto", "libx264"],
                    value="h264_nvenc",
                    label="Video encoder",
                )
            loop_speaker_videos = gr.Checkbox(
                label="Loop video input(s) if shorter than the soundtrack",
                value=False,
            )

        with gr.Accordion("DaVinci Resolve export", open=True):
            create_davinci_project = gr.Checkbox(
                label="Also create a portable DaVinci Resolve project (.otioz)",
                value=False,
                info=(
                    "The bundle contains editable active-speaker cuts, two native-resolution "
                    "camera-angle exports, master audio, and the timeline manifest. "
                    "The proxy encodes use the selected CQ quality instead of lossless QP=0 "
                    "to keep long 4K projects to a practical size."
                ),
            )

        with gr.Accordion("Advanced diarization", open=False):
            model = gr.Textbox(
                value=pipeline.DIARIZATION_MODEL,
                label="Pyannote model",
                info="Community-1 is the current highest-quality open local Pyannote diarization model.",
            )

        run_button = gr.Button("Create switched video", variant="primary", size="lg")

        with gr.Row():
            output_video = gr.File(
                label="Output video (served directly from outputs; no Gradio cache copy)"
            )
            output_segments = gr.File(label="Stable timeline JSON")
            output_davinci = gr.File(label="DaVinci Resolve project (.otioz)")
        status = gr.Textbox(label="Status", lines=9)

        run_button.click(
            _run_switcher,
            inputs=[
                mode,
                audio_source,
                audio_file,
                speaker0_video,
                speaker1_video,
                combined_video,
                first_speaker_side,
                segments_json_file,
                reuse_segments,
                model,
                device,
                hwaccel,
                video_encoder,
                preset,
                crf,
                min_switch_duration,
                silence_threshold,
                silence_lookahead,
                loop_speaker_videos,
                create_davinci_project,
            ],
            outputs=[output_video, output_segments, output_davinci, status],
        )

    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
    )
