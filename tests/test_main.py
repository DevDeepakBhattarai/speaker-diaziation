from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import main


class SpeakerMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = [
            main.Segment(0.0, 1.0, "SPEAKER_00"),
            main.Segment(1.0, 2.0, "SPEAKER_01"),
        ]

    def test_default_mapping_preserves_original_behavior(self) -> None:
        self.assertEqual(
            main.speaker_indexes_by_detection_order(self.segments),
            {"SPEAKER_00": 0, "SPEAKER_01": 1},
        )

    def test_first_detected_speaker_can_map_to_right_side(self) -> None:
        self.assertEqual(
            main.speaker_indexes_by_detection_order(
                self.segments,
                first_camera_index=1,
            ),
            {"SPEAKER_00": 1, "SPEAKER_01": 0},
        )


class CameraTimelineTests(unittest.TestCase):
    def test_substantial_exclusive_turn_switches_and_returns(self) -> None:
        exclusive = [
            main.Segment(0.0, 4.0, "SPEAKER_00"),
            main.Segment(4.0, 6.0, "SPEAKER_01"),
            main.Segment(6.0, 10.0, "SPEAKER_00"),
        ]

        self.assertEqual(
            main.build_camera_timeline(
                exclusive,
                duration=10.0,
                fallback_speaker="SPEAKER_00",
                min_switch_duration=1.0,
            ),
            exclusive,
        )

    def test_short_turn_is_debounced_only_for_camera(self) -> None:
        exclusive = [
            main.Segment(0.0, 4.0, "SPEAKER_00"),
            main.Segment(4.0, 4.2, "SPEAKER_01"),
            main.Segment(4.2, 10.0, "SPEAKER_00"),
        ]
        original = list(exclusive)

        timeline = main.build_camera_timeline(
            exclusive,
            duration=10.0,
            fallback_speaker="SPEAKER_00",
            min_switch_duration=1.0,
        )

        self.assertEqual(timeline, [main.Segment(0.0, 10.0, "SPEAKER_00")])
        self.assertEqual(exclusive, original)

    def test_long_silence_keeps_last_camera_visible(self) -> None:
        exclusive = [
            main.Segment(0.0, 2.0, "SPEAKER_00"),
            main.Segment(5.0, 7.0, "SPEAKER_00"),
        ]

        self.assertEqual(
            main.build_camera_timeline(
                exclusive,
                duration=9.0,
                fallback_speaker="SPEAKER_00",
                min_switch_duration=1.0,
                silence_threshold=2.5,
                silence_lookahead=5.0,
                gap_padding=0.0,
            ),
            [main.Segment(0.0, 9.0, "SPEAKER_00")],
        )

    def test_long_silence_lookahead_can_confirm_fragmented_new_turn(self) -> None:
        exclusive = [
            main.Segment(0.0, 2.0, "SPEAKER_00"),
            main.Segment(5.0, 5.6, "SPEAKER_01"),
            main.Segment(5.7, 6.3, "SPEAKER_01"),
            main.Segment(7.0, 9.0, "SPEAKER_00"),
        ]

        self.assertEqual(
            main.build_camera_timeline(
                exclusive,
                duration=10.0,
                fallback_speaker="SPEAKER_00",
                min_switch_duration=1.0,
                silence_threshold=2.5,
                silence_lookahead=5.0,
                gap_padding=0.0,
            ),
            [
                main.Segment(0.0, 5.0, "SPEAKER_00"),
                main.Segment(5.0, 7.0, "SPEAKER_01"),
                main.Segment(7.0, 10.0, "SPEAKER_00"),
            ],
        )

    def test_lookahead_stops_when_other_speaker_returns(self) -> None:
        exclusive = [
            main.Segment(0.0, 2.0, "SPEAKER_00"),
            main.Segment(5.0, 5.6, "SPEAKER_01"),
            main.Segment(5.7, 6.1, "SPEAKER_00"),
            main.Segment(6.2, 7.0, "SPEAKER_01"),
        ]

        self.assertEqual(
            main.build_camera_timeline(
                exclusive,
                duration=8.0,
                fallback_speaker="SPEAKER_00",
                min_switch_duration=1.0,
                silence_threshold=2.5,
                silence_lookahead=5.0,
                gap_padding=0.0,
            ),
            [main.Segment(0.0, 8.0, "SPEAKER_00")],
        )

    def test_lookahead_is_not_used_without_long_silence(self) -> None:
        exclusive = [
            main.Segment(0.0, 4.8, "SPEAKER_00"),
            main.Segment(5.0, 5.6, "SPEAKER_01"),
            main.Segment(5.7, 6.3, "SPEAKER_01"),
            main.Segment(6.4, 8.0, "SPEAKER_00"),
        ]

        self.assertEqual(
            main.build_camera_timeline(
                exclusive,
                duration=8.0,
                fallback_speaker="SPEAKER_00",
                min_switch_duration=1.0,
                silence_threshold=2.5,
                silence_lookahead=5.0,
                gap_padding=0.0,
            ),
            [main.Segment(0.0, 8.0, "SPEAKER_00")],
        )

    def test_other_speaker_switches_immediately_after_silence_when_turn_is_long(self) -> None:
        exclusive = [
            main.Segment(0.0, 2.0, "SPEAKER_00"),
            main.Segment(5.0, 7.0, "SPEAKER_01"),
        ]

        self.assertEqual(
            main.build_camera_timeline(
                exclusive,
                duration=9.0,
                fallback_speaker="SPEAKER_00",
                min_switch_duration=1.0,
            ),
            [
                main.Segment(0.0, 5.0, "SPEAKER_00"),
                main.Segment(5.0, 9.0, "SPEAKER_01"),
            ],
        )


class TimelineJsonTests(unittest.TestCase):
    def test_bom_timeline_preserves_raw_and_exclusive_diarization(self) -> None:
        payload = {
            "speaker_to_camera": {"SPEAKER_00": 0, "SPEAKER_01": 1},
            "segments": [
                {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
                {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"},
            ],
            "speech_segments": [
                {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
                {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01"},
            ],
            "exclusive_speech_segments": [
                {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
                {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "speaker_segments.json"
            path.write_text(json.dumps(payload), encoding="utf-8-sig")

            timeline, mapping = main.read_segments_json(path)
            raw = main.read_speech_segments_json(path)
            exclusive = main.read_exclusive_speech_segments_json(path)

        self.assertEqual(mapping, {"SPEAKER_00": 0, "SPEAKER_01": 1})
        self.assertEqual(
            timeline,
            [
                main.Segment(0.0, 3.0, "SPEAKER_00"),
                main.Segment(3.0, 6.0, "SPEAKER_01"),
            ],
        )
        self.assertEqual(
            raw,
            [
                main.Segment(0.0, 4.0, "SPEAKER_00"),
                main.Segment(3.0, 4.0, "SPEAKER_01"),
            ],
        )
        self.assertEqual(
            exclusive,
            [
                main.Segment(0.0, 3.0, "SPEAKER_00"),
                main.Segment(3.0, 4.0, "SPEAKER_01"),
            ],
        )


class SplitVideoEncodingTests(unittest.TestCase):
    def test_nvenc_lossless_encoding_uses_constant_qp_zero(self) -> None:
        command: list[str] = []
        main.append_video_encoding_options(
            command,
            encoder="h264_nvenc",
            preset="p4",
            crf=26,
            lossless=True,
        )

        self.assertEqual(
            command,
            ["-preset", "p4", "-tune", "lossless", "-rc", "constqp", "-qp", "0"],
        )

    def test_x264_lossless_encoding_uses_crf_zero(self) -> None:
        command: list[str] = []
        main.append_video_encoding_options(
            command,
            encoder="libx264",
            preset="medium",
            crf=26,
            lossless=True,
        )

        self.assertEqual(command, ["-preset", "medium", "-crf", "0"])


class SplitVideoFilterTests(unittest.TestCase):
    def test_combined_video_is_cropped_into_native_halves(self) -> None:
        timeline = [
            main.Segment(0.0, 1.0, "SPEAKER_00"),
            main.Segment(1.0, 2.0, "SPEAKER_01"),
        ]
        filter_text, output_label, width, height = (
            main.ffmpeg_filter_for_split_video_segments(
                timeline,
                {"SPEAKER_00": 0, "SPEAKER_01": 1},
                source_width=1920,
                source_height=1080,
            )
        )

        self.assertEqual(output_label, "outv")
        self.assertEqual((width, height), (1920, 1080))
        self.assertIn("crop=960:1080", filter_text)
        self.assertIn("pad=1920:1080", filter_text)
        self.assertNotIn("scale=", filter_text)
        self.assertIn("between(t\\,1.000\\,2.000)", filter_text)
        self.assertIn("960", filter_text)
        self.assertNotIn("overlay", filter_text)

    def test_split_renderer_uses_one_dynamic_crop_instead_of_overlay(self) -> None:
        timeline = [
            main.Segment(0.0, 1.0, "SPEAKER_00"),
            main.Segment(1.0, 2.0, "SPEAKER_01"),
            main.Segment(2.0, 3.0, "SPEAKER_00"),
        ]
        mapping = {"SPEAKER_00": 0, "SPEAKER_01": 1}

        filter_text, output_label, width, height = (
            main.ffmpeg_filter_for_split_video_segments(
                timeline,
                mapping,
                source_width=3840,
                source_height=2160,
            )
        )

        self.assertEqual(output_label, "outv")
        self.assertEqual((width, height), (3840, 2160))
        self.assertNotIn("overlay", filter_text)
        self.assertNotIn("split=", filter_text)
        self.assertIn("crop=1920:2160", filter_text)
        self.assertIn("pad=3840:2160", filter_text)
        self.assertNotIn("scale=", filter_text)
        self.assertIn("between(t\\,1.000\\,2.000)", filter_text)
        self.assertIn("1920", filter_text)

    def test_split_and_separate_renderers_share_camera_intervals(self) -> None:
        timeline = [
            main.Segment(0.0, 4.0, "SPEAKER_00"),
            main.Segment(4.0, 6.0, "SPEAKER_01"),
            main.Segment(6.0, 10.0, "SPEAKER_00"),
        ]
        mapping = {"SPEAKER_00": 0, "SPEAKER_01": 1}
        active_expression = main.speaker1_active_expression(timeline, mapping)

        separate_filter, _ = main.ffmpeg_filter_for_segments(
            timeline,
            mapping,
            width=1920,
            height=1080,
            use_cuda_overlay=False,
        )
        split_filter, _, _, _ = main.ffmpeg_filter_for_split_video_segments(
            timeline,
            mapping,
            source_width=3840,
            source_height=2160,
        )

        self.assertIn(active_expression, separate_filter)
        self.assertIn(active_expression, split_filter)


class DiarizationConfigurationTests(unittest.TestCase):
    def test_default_model_is_current_local_community_pipeline(self) -> None:
        self.assertEqual(
            main.DIARIZATION_MODEL,
            "pyannote/speaker-diarization-community-1",
        )

    def test_no_custom_clustering_override_remains(self) -> None:
        self.assertFalse(hasattr(main, "_configure_diarization_clustering"))

    def test_annotation_conversion_preserves_model_timestamps(self) -> None:
        class Turn:
            def __init__(self, start: float, end: float) -> None:
                self.start = start
                self.end = end

        class Annotation:
            def itertracks(self, *, yield_label: bool):
                self.yield_label = yield_label
                yield Turn(3.25, 4.75), None, "SPEAKER_01"
                yield Turn(0.1, 2.2), None, "SPEAKER_00"

        annotation = Annotation()
        self.assertEqual(
            main._segments_from_annotation(annotation),
            [
                main.Segment(0.1, 2.2, "SPEAKER_00"),
                main.Segment(3.25, 4.75, "SPEAKER_01"),
            ],
        )
        self.assertTrue(annotation.yield_label)


class LargeFileGpuTests(unittest.TestCase):
    def test_diarization_batches_are_clamped_to_memory_safe_minimum(self) -> None:
        class FakePipeline:
            segmentation_batch_size = 32
            embedding_batch_size = 32

        pipeline = FakePipeline()
        selected = main._configure_diarization_batch_size(pipeline, 0)

        self.assertEqual(selected, 1)
        self.assertEqual(pipeline.segmentation_batch_size, 1)
        self.assertEqual(pipeline.embedding_batch_size, 1)


if __name__ == "__main__":
    unittest.main()
