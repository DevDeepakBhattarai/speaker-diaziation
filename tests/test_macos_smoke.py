"""Real Apple VideoToolbox encode checks, run on a macOS ARM GitHub Actions runner."""
from __future__ import annotations

import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

import main


@unittest.skipUnless(
    platform.system() == "Darwin" and platform.machine() == "arm64",
    "VideoToolbox hardware smoke tests require Apple Silicon macOS",
)
class AppleSiliconVideoToolboxTests(unittest.TestCase):
    @staticmethod
    def _ffmpeg(*args: str) -> None:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)

    def _source(self, root: Path, *, high_bit_depth: bool) -> Path:
        output = root / ("source10.mp4" if high_bit_depth else "source8.mp4")
        command = [
            "-f", "lavfi", "-i", "testsrc2=size=960x540:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "0.5", "-c:a", "aac", "-b:a", "128k",
        ]
        if high_bit_depth:
            command.extend([
                "-vf", "format=yuv420p10le", "-c:v", "libx265",
                "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",
                "-x265-params", "log-level=error:frame-threads=1:pools=none",
            ])
        else:
            command.extend(["-c:v", "libx264", "-preset", "ultrafast"])
        self._ffmpeg(*command, str(output))
        return output

    def _render(self, source: Path, output: Path, encoder: str) -> dict:
        plan = main.build_render_plan(main.MODE_SPLIT_VIDEO, [source])
        main.render_pipeline(
            plan=plan,
            audio_file=source,
            timeline=[main.Segment(0, 0.25, "A"), main.Segment(0.25, 0.5, "B")],
            mapping={"A": 0, "B": 1},
            outputs=main.RenderOutputs(switched_video=output),
            encoder=encoder,
            audio_codec="copy",
            preset="p6",
            crf=16,
            hwaccel=main.choose_hwaccel("auto", encoder),
            loop_videos=False,
            duration=0.5,
            max_parallel_encodes=1,
        )
        self.assertTrue(output.is_file() and output.stat().st_size > 1000)
        return main.source_video_stream(output)

    def test_h264_videotoolbox_native_split_and_audio_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root, high_bit_depth=False)
            output = root / "switched_h264.mp4"
            video = self._render(source, output, "h264_videotoolbox")
            self.assertEqual((video["width"], video["height"]), (480, 540))
            self.assertEqual(video["codec_name"], "h264")
            self.assertEqual(main.source_audio_codec(output), "aac")

    def test_hevc_videotoolbox_preserves_10bit_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root, high_bit_depth=True)
            self.assertTrue(main.is_high_bit_depth(main.source_video_stream(source)["pix_fmt"]))
            output = root / "switched_hevc.mp4"
            video = self._render(source, output, "hevc_videotoolbox")
            self.assertEqual((video["width"], video["height"]), (480, 540))
            self.assertEqual(video["codec_name"], "hevc")
            self.assertTrue(main.is_high_bit_depth(video["pix_fmt"]))


if __name__ == "__main__":
    unittest.main()
