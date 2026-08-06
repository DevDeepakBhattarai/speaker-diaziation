from __future__ import annotations

import shutil
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr

import main as pipeline


WORK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORK_DIR / "outputs"
UI_MODE_SEPARATE = "Two separate speaker videos"
UI_MODE_SPLIT = "One combined left/right video"
UI_AUDIO_EMBEDDED = "Use audio from the video"
UI_AUDIO_SEPARATE = "Upload a separate audio file"


def _copy_input(path: str, target_dir: Path, name: str) -> Path:
    source = Path(path)
    suffix = source.suffix or Path(name).suffix
    target = target_dir / f"{Path(name).stem}{suffix}"
    shutil.copy2(source, target)
    return target


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
    merge_gap: float,
    min_segment: float,
    gap_padding: float,
    min_switch_duration: float,
    loop_speaker_videos: bool,
    render_duration: float | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str | None, str | None, str]:
    split_mode = mode == UI_MODE_SPLIT
    separate_audio = audio_source == UI_AUDIO_SEPARATE
    if separate_audio and not audio_file:
        raise gr.Error("Upload the separate soundtrack, or choose audio from the video.")
    if split_mode and not combined_video:
        raise gr.Error("Upload one combined video with the left and right speakers visible.")
    if not split_mode and (not speaker0_video or not speaker1_video):
        raise gr.Error("Upload both speaker video files.")

    if reuse_segments and not segments_json_file:
        raise gr.Error("Upload an existing speaker_segments.json file or disable reuse.")

    started = time.perf_counter()
    job_dir = OUTPUT_DIR / time.strftime("%Y%m%d-%H%M%S")
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress(0.02, desc="Copying inputs")
        speaker0_path: Path | None = None
        speaker1_path: Path | None = None
        combined_path: Path | None = None
        if split_mode:
            assert combined_video is not None
            combined_path = _copy_input(combined_video, job_dir, "combined_video")
            output_video = job_dir / "speaker_split_switched.mp4"
            primary_video_path = combined_path
        else:
            assert speaker0_video is not None and speaker1_video is not None
            speaker0_path = _copy_input(speaker0_video, job_dir, "speaker_0")
            speaker1_path = _copy_input(speaker1_video, job_dir, "speaker_1")
            output_video = job_dir / "speaker_switched.mp4"
            primary_video_path = speaker0_path
        segments_json = job_dir / "speaker_segments.json"

        pipeline.require_tool("ffmpeg")
        pipeline.require_tool("ffprobe")
        if separate_audio:
            assert audio_file is not None
            audio_path = _copy_input(audio_file, job_dir, "audio")
            audio_details = "Separate uploaded audio"
        else:
            audio_path = primary_video_path
            if not pipeline.media_has_audio(audio_path):
                raise gr.Error(
                    "The selected video has no audio track. Upload a separate audio file instead."
                )
            audio_details = "Audio embedded in the video"
        encoder = pipeline.choose_video_encoder(video_encoder)
        selected_hwaccel = pipeline.choose_hwaccel(hwaccel, encoder)
        duration = pipeline.media_duration(audio_path)
        first_camera_index = 1 if split_mode and first_speaker_side == "Right" else 0

        if reuse_segments:
            progress(0.12, desc="Reading existing timeline")
            copied_segments = _copy_input(segments_json_file, job_dir, "speaker_segments.json")
            timeline, mapping = pipeline.read_segments_json(copied_segments)
            segments_json = copied_segments
            if split_mode:
                mapping = pipeline.speaker_indexes_by_detection_order(
                    timeline,
                    first_camera_index=first_camera_index,
                )
            timeline = pipeline.stabilize_timeline(
                timeline,
                min_switch_duration=min_switch_duration,
            )
            pipeline.write_segments_json(segments_json, timeline, mapping)
        else:
            progress(0.12, desc="Preparing audio")
            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = Path(tmpdir) / "source_audio.wav"
                pipeline.prepare_audio(audio_path, wav_path, 16000)

                progress(0.28, desc="Running speaker diarization")
                raw_segments = pipeline.diarize_audio(
                    wav_path,
                    model=model,
                    hf_token=None,
                    device=device,
                    min_speakers=None,
                    max_speakers=None,
                    num_speakers=2,
                )

            if not raw_segments:
                raise gr.Error("Diarization produced no speaker segments.")

            progress(0.62, desc="Building persistent speaker timeline")
            mapping = pipeline.speaker_indexes_by_detection_order(
                raw_segments,
                first_camera_index=first_camera_index,
            )
            merged = pipeline.merge_segments(
                raw_segments,
                gap=merge_gap,
                min_duration=min_segment,
            )
            timeline = pipeline.fill_timeline(
                merged,
                duration=duration,
                fallback_speaker=next(iter(mapping)),
                gap_padding=gap_padding,
            )
            timeline = pipeline.stabilize_timeline(
                timeline,
                min_switch_duration=min_switch_duration,
            )
            pipeline.write_segments_json(segments_json, timeline, mapping)

        progress(0.72, desc="Rendering switched video with FFmpeg")
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
                render_duration=render_duration or None,
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
                render_duration=render_duration or None,
            )
            mode_details = "Two separate speaker videos"

        elapsed = time.perf_counter() - started
        status = (
            f"Done in {elapsed / 60:.1f} minutes.\n"
            f"Mode: {mode_details}\n"
            f"Audio: {audio_details}\n"
            f"Minimum camera hold: {min_switch_duration:.2f} seconds\n"
            f"Output: {output_video}\n"
            f"Timeline: {segments_json}\n"
            f"Encoder: {encoder}; hwaccel: {selected_hwaccel or 'none'}"
        )
        progress(1.0, desc="Done")
        return str(output_video), str(segments_json), status
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

        with gr.Group():
            gr.Markdown("### 1. Choose the video layout")
            mode = gr.Radio(
                [UI_MODE_SEPARATE, UI_MODE_SPLIT],
                value=UI_MODE_SEPARATE,
                label="Video input mode",
            )

            with gr.Row():
                speaker0_video = gr.File(
                    label="First detected speaker video",
                    file_types=["video"],
                    type="filepath",
                )
                speaker1_video = gr.File(
                    label="Second detected speaker video",
                    file_types=["video"],
                    type="filepath",
                )

            with gr.Row():
                combined_video = gr.File(
                    label="Combined video (left person | right person)",
                    file_types=["video"],
                    type="filepath",
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
            audio_file = gr.File(
                label="Separate soundtrack / audio file",
                file_types=["audio"],
                type="filepath",
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
            segments_json_file = gr.File(
                label="Existing speaker_segments.json",
                file_types=[".json"],
                type="filepath",
            )

        with gr.Accordion("Camera switching", open=True):
            gr.Markdown(
                "Short detections are ignored so the output does not flicker between people."
            )
            with gr.Row():
                min_switch_duration = gr.Slider(
                    0.0,
                    3.0,
                    value=1.0,
                    step=0.05,
                    label="Minimum speaker turn before switching (seconds)",
                )
                merge_gap = gr.Slider(
                    0.0,
                    2.0,
                    value=0.30,
                    step=0.05,
                    label="Merge nearby detections (seconds)",
                )

        with gr.Accordion("Speed and quality", open=False):
            with gr.Row():
                preset = gr.Dropdown(
                    ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                    value="p1",
                    label="NVENC preset",
                )
                crf = gr.Slider(18, 35, value=26, step=1, label="CQ quality")
                render_duration = gr.Number(
                    value=None,
                    label="Render only first N seconds",
                    precision=1,
                )
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

        with gr.Accordion("Advanced diarization", open=False):
            model = gr.Textbox(
                value=pipeline.DIARIZATION_MODEL,
                label="Pyannote model",
            )
            with gr.Row():
                min_segment = gr.Slider(
                    0.0,
                    2.0,
                    value=0.20,
                    step=0.05,
                    label="Minimum raw detection (seconds)",
                )
                gap_padding = gr.Slider(
                    0.0,
                    1.0,
                    value=0.05,
                    step=0.01,
                    label="Speaker boundary padding",
                )

        run_button = gr.Button("Create switched video", variant="primary", size="lg")

        with gr.Row():
            output_video = gr.Video(label="Output video")
            output_segments = gr.File(label="Stable timeline JSON")
        status = gr.Textbox(label="Status", lines=8)

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
                merge_gap,
                min_segment,
                gap_padding,
                min_switch_duration,
                loop_speaker_videos,
                render_duration,
            ],
            outputs=[output_video, output_segments, status],
        )

    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
