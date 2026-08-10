from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import davinci_export
import main


class DavinciTimelineTests(unittest.TestCase):
    def test_trim_timeline_clamps_and_compacts(self) -> None:
        timeline = [
            main.Segment(0.0, 2.0, "SPEAKER_00"),
            main.Segment(2.0, 4.0, "SPEAKER_00"),
            main.Segment(4.0, 8.0, "SPEAKER_01"),
        ]

        self.assertEqual(
            davinci_export.trim_timeline(timeline, duration=6.0),
            [
                main.Segment(0.0, 4.0, "SPEAKER_00"),
                main.Segment(4.0, 6.0, "SPEAKER_01"),
            ],
        )

    def test_build_otio_document_contains_editable_v1_and_master_a1(self) -> None:
        timeline = [
            main.Segment(0.0, 2.0, "SPEAKER_00"),
            main.Segment(2.0, 5.0, "SPEAKER_01"),
        ]
        mapping = {"SPEAKER_00": 1, "SPEAKER_01": 0}

        document = davinci_export.build_otio_document(
            timeline,
            mapping,
            title="Podcast Edit",
            fps=30.0,
            width=960,
            height=1080,
            camera_media_names=["camera_1.mp4", "camera_2.mp4"],
            camera_names=["Left Speaker", "Right Speaker"],
            camera_resolutions=[(960, 1080), (960, 1080)],
            audio_media_name="master_audio.m4a",
        )

        self.assertEqual(document["OTIO_SCHEMA"], "Timeline.1")
        tracks = document["tracks"]["children"]
        self.assertEqual([track["kind"] for track in tracks], ["Video", "Audio"])
        self.assertEqual(tracks[0]["name"], "V1 - Active Speaker")
        self.assertEqual(tracks[1]["name"], "A1 - Master Audio")

        video_clips = tracks[0]["children"]
        self.assertEqual(len(video_clips), 2)
        self.assertEqual(
            video_clips[0]["media_references"]["DEFAULT_MEDIA"]["target_url"],
            "media/camera_2.mp4",
        )
        self.assertEqual(
            video_clips[1]["media_references"]["DEFAULT_MEDIA"]["target_url"],
            "media/camera_1.mp4",
        )
        self.assertEqual(
            video_clips[0]["source_range"]["start_time"]["value"],
            0,
        )
        self.assertEqual(
            video_clips[1]["source_range"]["start_time"]["value"],
            60,
        )
        self.assertEqual(
            tracks[1]["children"][0]["media_references"]["DEFAULT_MEDIA"]["target_url"],
            "media/master_audio.m4a",
        )
        self.assertEqual(
            tracks[1]["children"][0]["source_range"]["duration"]["value"],
            150,
        )


class OtiozBundleTests(unittest.TestCase):
    def test_bundle_contains_standard_otioz_layout_and_audit_files(self) -> None:
        timeline = [
            main.Segment(0.0, 1.0, "SPEAKER_00"),
            main.Segment(1.0, 2.0, "SPEAKER_01"),
        ]
        mapping = {"SPEAKER_00": 0, "SPEAKER_01": 1}
        document = davinci_export.build_otio_document(
            timeline,
            mapping,
            title="Portable Edit",
            fps=24.0,
            width=1280,
            height=720,
            camera_media_names=["camera_1.mp4", "camera_2.mp4"],
            camera_names=["Camera 1", "Camera 2"],
            camera_resolutions=[(1280, 720), (1280, 720)],
            audio_media_name="master_audio.m4a",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            media_files = [
                root / "camera_1.mp4",
                root / "camera_2.mp4",
                root / "master_audio.m4a",
            ]
            for index, path in enumerate(media_files):
                path.write_bytes(f"media-{index}".encode("utf-8"))
            segments_json = root / "speaker_segments.json"
            segments_json.write_text(
                json.dumps({"speaker_to_camera": mapping}),
                encoding="utf-8",
            )
            output = root / "speaker_edit.otioz"

            davinci_export.write_otioz_bundle(
                output,
                document=document,
                media_files=media_files,
                manifest={"kind": "test_manifest"},
                segments_json=segments_json,
            )

            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertTrue(
                    {
                        "content.otio",
                        "version.txt",
                        "media/camera_1.mp4",
                        "media/camera_2.mp4",
                        "media/master_audio.m4a",
                        "davinci_manifest.json",
                        "speaker_segments.json",
                        "README.txt",
                    }.issubset(names)
                )
                self.assertEqual(
                    archive.read("version.txt").decode("utf-8"),
                    "1.0.0",
                )
                bundled_document = json.loads(
                    archive.read("content.otio").decode("utf-8")
                )
                self.assertEqual(bundled_document["OTIO_SCHEMA"], "Timeline.1")
                for media_path in (
                    "media/camera_1.mp4",
                    "media/camera_2.mp4",
                    "media/master_audio.m4a",
                ):
                    self.assertEqual(
                        archive.getinfo(media_path).compress_type,
                        zipfile.ZIP_STORED,
                    )


class DavinciBundleTests(unittest.TestCase):
    """The bundler consumes rendered media; it must never encode anything itself."""

    def _bundle(self, plan: main.RenderPlan, root: Path) -> Path:
        camera_videos, master_audio = davinci_export.bundle_media_paths(root)
        for path in (*camera_videos, master_audio):
            path.write_bytes(b"rendered")
        segments = root / "speaker_segments.json"
        segments.write_text("{}", encoding="utf-8")
        output = root / "speaker_edit.otioz"

        with mock.patch("davinci_export.video_fps", return_value=30.0):
            return davinci_export.create_davinci_bundle(
                plan=plan,
                audio_file=root / "audio.wav",
                segments_json=segments,
                timeline=[
                    main.Segment(0.0, 1.0, "SPEAKER_00"),
                    main.Segment(1.0, 2.0, "SPEAKER_01"),
                ],
                mapping={"SPEAKER_00": 0, "SPEAKER_01": 1},
                output_bundle=output,
                camera_videos=camera_videos,
                master_audio=master_audio,
                duration=2.0,
            )

    def test_bundling_rendered_media_runs_no_ffmpeg_at_all(self) -> None:
        plan = main.build_render_plan(
            main.MODE_SPLIT_VIDEO,
            [Path("combined.mp4")],
            sizes=[(3840, 2160)],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("davinci_export.pipeline.run") as run:
                output = self._bundle(plan, Path(tmpdir))

            run.assert_not_called()
            self.assertTrue(output.exists())

    def test_split_bundle_uses_the_native_half_canvas_and_side_labels(self) -> None:
        plan = main.build_render_plan(
            main.MODE_SPLIT_VIDEO,
            [Path("combined.mp4")],
            sizes=[(3840, 2160)],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = self._bundle(plan, Path(tmpdir))
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("davinci_manifest.json"))

        self.assertEqual((manifest["width"], manifest["height"]), (1920, 2160))
        self.assertEqual(manifest["camera_names"], ["Left Speaker", "Right Speaker"])
        self.assertEqual(
            manifest["camera_resolutions"],
            [{"width": 1920, "height": 2160}, {"width": 1920, "height": 2160}],
        )

    def test_separate_bundle_uses_the_shared_canvas_of_both_cameras(self) -> None:
        plan = main.build_render_plan(
            main.MODE_SEPARATE_VIDEOS,
            [Path("camera_1.mp4"), Path("camera_2.mp4")],
            sizes=[(3840, 2160), (4096, 2160)],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = self._bundle(plan, Path(tmpdir))
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("davinci_manifest.json"))

        self.assertEqual((manifest["width"], manifest["height"]), (4096, 2160))
        self.assertEqual(manifest["camera_names"], ["Camera 1", "Camera 2"])


if __name__ == "__main__":
    unittest.main()
