from __future__ import annotations

import unittest

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


class TimelineStabilityTests(unittest.TestCase):
    def test_adjacent_fragments_are_merged_before_minimum_duration_filter(self) -> None:
        segments = [
            main.Segment(0.0, 0.18, "SPEAKER_00"),
            main.Segment(0.22, 0.42, "SPEAKER_00"),
            main.Segment(0.50, 1.20, "SPEAKER_01"),
        ]

        self.assertEqual(
            main.merge_segments(segments, gap=0.10, min_duration=0.30),
            [
                main.Segment(0.0, 0.42, "SPEAKER_00"),
                main.Segment(0.50, 1.20, "SPEAKER_01"),
            ],
        )

    def test_nested_speaker_turn_is_not_discarded(self) -> None:
        segments = [
            main.Segment(0.0, 10.0, "SPEAKER_00"),
            main.Segment(4.0, 6.0, "SPEAKER_01"),
        ]

        self.assertEqual(
            main.fill_timeline(
                segments,
                duration=10.0,
                fallback_speaker="SPEAKER_00",
                gap_padding=0.0,
            ),
            [
                main.Segment(0.0, 4.0, "SPEAKER_00"),
                main.Segment(4.0, 6.0, "SPEAKER_01"),
                main.Segment(6.0, 10.0, "SPEAKER_00"),
            ],
        )

    def test_brief_middle_detection_does_not_switch_camera(self) -> None:
        timeline = [
            main.Segment(0.0, 5.0, "SPEAKER_00"),
            main.Segment(5.0, 5.20, "SPEAKER_01"),
            main.Segment(5.20, 10.0, "SPEAKER_00"),
        ]

        self.assertEqual(
            main.stabilize_timeline(timeline, min_switch_duration=1.0),
            [main.Segment(0.0, 10.0, "SPEAKER_00")],
        )

    def test_substantial_turn_switches_camera(self) -> None:
        timeline = [
            main.Segment(0.0, 5.0, "SPEAKER_00"),
            main.Segment(5.0, 7.0, "SPEAKER_01"),
            main.Segment(7.0, 10.0, "SPEAKER_00"),
        ]

        self.assertEqual(
            main.stabilize_timeline(timeline, min_switch_duration=1.0),
            timeline,
        )

    def test_brief_detection_at_start_uses_first_substantial_speaker(self) -> None:
        timeline = [
            main.Segment(0.0, 0.10, "SPEAKER_00"),
            main.Segment(0.10, 4.0, "SPEAKER_01"),
        ]

        self.assertEqual(
            main.stabilize_timeline(timeline, min_switch_duration=1.0),
            [main.Segment(0.0, 4.0, "SPEAKER_01")],
        )


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
        self.assertEqual((width, height), (960, 1080))
        self.assertIn("crop=960:1080:0:0", filter_text)
        self.assertIn("crop=960:1080:960:0", filter_text)
        self.assertIn("between(t\\,1.000\\,2.000)", filter_text)


if __name__ == "__main__":
    unittest.main()
