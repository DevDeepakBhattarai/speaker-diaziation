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


def _copy_input(path: str, target_dir: Path, name: str) -> Path:
    source = Path(path)
    suffix = source.suffix or Path(name).suffix
    target = target_dir / f"{Path(name).stem}{suffix}"
    shutil.copy2(source, target)
    return target


def _run_switcher(
    audio_file: str | None,
    speaker0_video: str | None,
    speaker1_video: str | None,
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
    loop_speaker_videos: bool,
    render_duration: float | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str | None, str | None, str]:
    if not audio_file or not speaker0_video or not speaker1_video:
        raise gr.Error("Upload an audio file, speaker 0 video, and speaker 1 video.")

    if reuse_segments and not segments_json_file:
        raise gr.Error("Upload an existing speaker_segments.json file or disable reuse.")

    started = time.perf_counter()
    job_dir = OUTPUT_DIR / time.strftime("%Y%m%d-%H%M%S")
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress(0.02, desc="Copying inputs")
        audio_path = _copy_input(audio_file, job_dir, "audio")
        speaker0_path = _copy_input(speaker0_video, job_dir, "speaker_0")
        speaker1_path = _copy_input(speaker1_video, job_dir, "speaker_1")
        output_video = job_dir / "speaker_switched.mp4"
        segments_json = job_dir / "speaker_segments.json"

        pipeline.require_tool("ffmpeg")
        pipeline.require_tool("ffprobe")
        encoder = pipeline.choose_video_encoder(video_encoder)
        selected_hwaccel = pipeline.choose_hwaccel(hwaccel, encoder)
        duration = pipeline.media_duration(audio_path)
        camera_videos = [speaker0_path, speaker1_path]

        if reuse_segments:
            progress(0.12, desc="Reading existing timeline")
            copied_segments = _copy_input(segments_json_file, job_dir, "speaker_segments.json")
            timeline, mapping = pipeline.read_segments_json(copied_segments)
            segments_json = copied_segments
        else:
            progress(0.12, desc="Preparing audio")
            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = Path(tmpdir) / "source_audio.wav"
                pipeline.prepare_audio(audio_path, wav_path, 16000)

                progress(0.28, desc="Running speaker diarization on CUDA")
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
            merged = pipeline.merge_segments(
                raw_segments,
                gap=merge_gap,
                min_duration=min_segment,
            )
            mapping = pipeline.speaker_indexes_by_detection_order(merged)
            timeline = pipeline.fill_timeline(
                merged,
                duration=duration,
                fallback_speaker=next(iter(mapping)),
                gap_padding=gap_padding,
            )
            pipeline.write_segments_json(segments_json, timeline, mapping)

        progress(0.72, desc="Rendering switched video with FFmpeg/NVENC")
        pipeline.assemble_video(
            audio_file=audio_path,
            camera_videos=camera_videos,
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

        elapsed = time.perf_counter() - started
        status = (
            f"Done in {elapsed / 60:.1f} minutes.\n"
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
            "Upload one audio file and two speaker videos. The app diarizes the audio, "
            "keeps the last active speaker visible through pauses, and renders with CUDA/NVENC."
        )

        with gr.Row():
            audio_file = gr.File(label="Audio file", file_types=["audio"], type="filepath")
            speaker0_video = gr.File(label="Speaker 0 video", file_types=["video"], type="filepath")
            speaker1_video = gr.File(label="Speaker 1 video", file_types=["video"], type="filepath")

        with gr.Accordion("Timeline reuse", open=True):
            reuse_segments = gr.Checkbox(
                label="Reuse existing speaker_segments.json",
                value=False,
            )
            segments_json_file = gr.File(
                label="Existing speaker_segments.json",
                file_types=[".json"],
                type="filepath",
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
                device = gr.Dropdown(["cuda", "auto", "cpu"], value="cuda", label="Diarization device")
                hwaccel = gr.Dropdown(["cuda", "auto", "none"], value="cuda", label="FFmpeg hwaccel")
                video_encoder = gr.Dropdown(
                    ["h264_nvenc", "hevc_nvenc", "auto", "libx264"],
                    value="h264_nvenc",
                    label="Video encoder",
                )
            loop_speaker_videos = gr.Checkbox(
                label="Loop speaker videos if shorter than audio",
                value=False,
            )

        with gr.Accordion("Diarization tuning", open=False):
            model = gr.Textbox(
                value=pipeline.DIARIZATION_MODEL,
                label="Pyannote model",
            )
            with gr.Row():
                merge_gap = gr.Slider(0.0, 2.0, value=0.30, step=0.05, label="Merge gap")
                min_segment = gr.Slider(0.0, 2.0, value=0.25, step=0.05, label="Minimum segment")
                gap_padding = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Gap padding")

        run_button = gr.Button("Create switched video", variant="primary")

        with gr.Row():
            output_video = gr.Video(label="Output video")
            output_segments = gr.File(label="Timeline JSON")
        status = gr.Textbox(label="Status", lines=5)

        run_button.click(
            _run_switcher,
            inputs=[
                audio_file,
                speaker0_video,
                speaker1_video,
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
                loop_speaker_videos,
                render_duration,
            ],
            outputs=[output_video, output_segments, status],
        )

    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch()
