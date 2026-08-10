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

    def test_separate_camera_layout_preserves_each_native_4k_resolution(self) -> None:
        timeline_size, proxy_sizes, crops = (
            davinci_export.resolve_camera_export_layout(
                main.MODE_SEPARATE_VIDEOS,
                [(3840, 2160), (4096, 2160)],
            )
        )

        self.assertEqual(timeline_size, (3840, 2160))
        self.assertEqual(proxy_sizes, [(3840, 2160), (4096, 2160)])
        self.assertEqual(crops, [None, None])

    def test_split_camera_layout_preserves_source_canvas_and_native_crop_pixels(self) -> None:
        timeline_size, proxy_sizes, crops = (
            davinci_export.resolve_camera_export_layout(
                main.MODE_SPLIT_VIDEO,
                [(3840, 2160)],
            )
        )

        self.assertEqual(timeline_size, (3840, 2160))
        self.assertEqual(proxy_sizes, [(3840, 2160), (3840, 2160)])
        self.assertEqual(
            crops,
            [(0, 0, 1920, 2160), (1920, 0, 1920, 2160)],
        )

    def test_proxy_filter_pads_crop_without_scaling_native_media(self) -> None:
        filter_text = davinci_export._proxy_filter(
            duration=10.0,
            crop=(0, 0, 1920, 2160),
            output_size=(3840, 2160),
            loop=False,
        )

        self.assertIn("crop=1920:2160:0:0", filter_text)
        self.assertIn("pad=3840:2160", filter_text)
        self.assertNotIn("scale=", filter_text)

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


class DavinciProxyEncodingTests(unittest.TestCase):
    def test_split_davinci_proxies_use_selected_quality_not_forced_lossless(self) -> None:
        timeline = [
            main.Segment(0.0, 1.0, "SPEAKER_00"),
            main.Segment(1.0, 2.0, "SPEAKER_01"),
        ]
        mapping = {"SPEAKER_00": 0, "SPEAKER_01": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            combined = root / "combined.mp4"
            audio = root / "audio.wav"
            segments = root / "speaker_segments.json"
            output = root / "speaker_edit.otioz"
            combined.write_bytes(b"video")
            audio.write_bytes(b"audio")
            segments.write_text("{}", encoding="utf-8")

            with (
                mock.patch("davinci_export.pipeline.media_duration", return_value=2.0),
                mock.patch("davinci_export.pipeline.source_video_size", return_value=(3840, 2160)),
                mock.patch("davinci_export.video_fps", return_value=30.0),
                mock.patch("davinci_export.generate_video_proxy") as generate_video_proxy,
                mock.patch("davinci_export.generate_audio_proxy"),
                mock.patch("davinci_export.write_otioz_bundle"),
            ):
                davinci_export.create_davinci_otioz(
                    mode=main.MODE_SPLIT_VIDEO,
                    audio_file=audio,
                    camera_videos=[combined],
                    segments_json=segments,
                    timeline=timeline,
                    mapping=mapping,
                    output_bundle=output,
                    encoder="h264_nvenc",
                    preset="p1",
                    crf=26,
                    hwaccel="cuda",
                    loop_cameras=False,
                    render_duration=None,
                )

            self.assertEqual(generate_video_proxy.call_count, 2)
            for call in generate_video_proxy.call_args_list:
                self.assertFalse(call.kwargs.get("lossless", False))
                self.assertEqual(call.kwargs["crf"], 26)


if __name__ == "__main__":
    unittest.main()
