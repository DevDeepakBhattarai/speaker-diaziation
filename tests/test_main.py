from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main
import job


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


class HardwareSelectionTests(unittest.TestCase):
    def test_auto_encoder_uses_apple_videotoolbox_when_available(self) -> None:
        with mock.patch(
            "main.available_ffmpeg_encoders",
            return_value=" V..... h264_videotoolbox VideoToolbox H.264 Encoder",
        ):
            self.assertEqual(main.choose_video_encoder("auto"), "h264_videotoolbox")

    def test_auto_hwaccel_matches_videotoolbox_encoder(self) -> None:
        self.assertEqual(
            main.choose_hwaccel("auto", "h264_videotoolbox"),
            "videotoolbox",
        )

    def test_videotoolbox_session_limit_is_probed(self) -> None:
        self.addCleanup(main._ENCODER_SESSION_LIMITS.clear)
        with mock.patch("main._probe_encoder_sessions", return_value=True) as probe:
            self.assertEqual(
                main.max_parallel_encoder_sessions("h264_videotoolbox", 3),
                3,
            )
        probe.assert_called_once_with("h264_videotoolbox", 3)

    def test_job_defaults_are_portable(self) -> None:
        request = job.JobRequest(
            mode=main.MODE_SPLIT_VIDEO,
            videos=(Path("combined.mp4"),),
            audio_file=Path("combined.mp4"),
            output_video=Path("out.mp4"),
            segments_json=Path("segments.json"),
        )
        self.assertEqual(request.device, "auto")
        self.assertEqual(request.video_encoder, "auto")
        self.assertEqual(request.hwaccel, "auto")


TIMELINE = [
    main.Segment(0.0, 1.0, "SPEAKER_00"),
    main.Segment(1.0, 2.0, "SPEAKER_01"),
    main.Segment(2.0, 3.0, "SPEAKER_00"),
]
MAPPING = {"SPEAKER_00": 0, "SPEAKER_01": 1}


def split_plan(size: tuple[int, int] = (3840, 2160)) -> main.RenderPlan:
    return main.build_render_plan(
        main.MODE_SPLIT_VIDEO,
        [Path("combined.mp4")],
        sizes=[size],
    )


def separate_plan(
    sizes: list[tuple[int, int]] | None = None,
) -> main.RenderPlan:
    return main.build_render_plan(
        main.MODE_SEPARATE_VIDEOS,
        [Path("camera_1.mp4"), Path("camera_2.mp4")],
        sizes=sizes or [(3840, 2160), (3840, 2160)],
    )


class RenderPlanTests(unittest.TestCase):
    def test_combined_video_becomes_two_native_half_crops(self) -> None:
        plan = split_plan()

        # Each half is delivered at its own native size, not padded back to 4K.
        self.assertEqual(plan.canvas, (1920, 2160))
        self.assertEqual(
            [camera.crop for camera in plan.cameras],
            [(0, 0, 1920, 2160), (1920, 0, 1920, 2160)],
        )
        self.assertEqual([camera.input_index for camera in plan.cameras], [0, 0])
        self.assertEqual([camera.size for camera in plan.cameras], [(1920, 2160), (1920, 2160)])

    def test_an_odd_width_combined_source_still_splits_on_an_even_grid(self) -> None:
        plan = split_plan((3841, 2161))

        self.assertEqual(plan.canvas, (1920, 2160))
        self.assertEqual(
            [camera.crop for camera in plan.cameras],
            [(0, 0, 1920, 2160), (1921, 0, 1920, 2160)],
        )

    def test_separate_videos_become_two_uncropped_angles_on_one_canvas(self) -> None:
        plan = separate_plan()

        self.assertEqual(plan.canvas, (3840, 2160))
        self.assertEqual([camera.crop for camera in plan.cameras], [None, None])
        self.assertEqual([camera.input_index for camera in plan.cameras], [0, 1])

    def test_mismatched_cameras_share_a_canvas_that_never_downscales_either(self) -> None:
        plan = separate_plan([(3840, 2160), (4096, 2160)])

        self.assertEqual(plan.canvas, (4096, 2160))
        self.assertEqual([camera.size for camera in plan.cameras], [(3840, 2160), (4096, 2160)])

    def test_odd_source_dimensions_are_cropped_to_an_even_grid(self) -> None:
        plan = separate_plan([(1921, 1081), (1920, 1080)])

        self.assertEqual(plan.cameras[0].crop, (0, 0, 1920, 1080))
        self.assertEqual(plan.cameras[1].crop, None)
        self.assertEqual(plan.canvas, (1920, 1080))


class CameraFilterChainTests(unittest.TestCase):
    def test_a_half_crop_needs_no_padding_and_no_scaling(self) -> None:
        plan = split_plan()

        self.assertEqual(
            main.camera_filter_chain(plan.cameras[1], plan.canvas),
            ["crop=1920:2160:1920:0", "setsar=1"],
        )

    def test_a_full_canvas_angle_is_passed_through_untouched(self) -> None:
        plan = separate_plan()

        self.assertEqual(main.camera_filter_chain(plan.cameras[0], plan.canvas), ["setsar=1"])

    def test_a_smaller_angle_is_padded_onto_the_shared_canvas(self) -> None:
        plan = separate_plan([(3840, 2160), (4096, 2160)])

        self.assertEqual(
            main.camera_filter_chain(plan.cameras[0], plan.canvas),
            ["pad=4096:2160:(ow-iw)/2:(oh-ih)/2:color=black", "setsar=1"],
        )

    def test_no_input_mode_ever_scales_pixels(self) -> None:
        for plan in (split_plan(), separate_plan([(3840, 2160), (4096, 2160)])):
            for camera in plan.cameras:
                chain = main.camera_filter_chain(camera, plan.canvas)
                self.assertFalse(
                    any(step.startswith("scale") for step in chain),
                    msg=f"{plan.mode} scaled an angle: {chain}",
                )

    def test_an_angle_larger_than_the_canvas_is_rejected_rather_than_shrunk(self) -> None:
        oversized = main.CameraAngle(0, (4096, 2160))

        with self.assertRaises(SystemExit):
            main.camera_filter_chain(oversized, (3840, 2160))


class FilterGraphTests(unittest.TestCase):
    def test_one_decode_feeds_the_switched_feed_and_both_camera_angles(self) -> None:
        graph = main.build_switch_filter_graph(
            split_plan(),
            TIMELINE,
            MAPPING,
            wanted=[main.SWITCHED_LABEL, *main.CAMERA_LABELS],
            hold_duration=None,
        )

        # The combined source is opened once and split, never decoded twice.
        self.assertEqual(graph.count("[0:v]"), 1)
        self.assertIn("split=2[src0][src1]", graph)
        self.assertIn("[camera0][mix0]", graph)
        self.assertIn("[camera1][mix1]", graph)
        self.assertIn("[mix0][mix1]overlay=enable=", graph)
        self.assertIn(f"[{main.SWITCHED_LABEL}]", graph)

    def test_switched_only_graph_skips_the_camera_branches(self) -> None:
        graph = main.build_switch_filter_graph(
            separate_plan(),
            TIMELINE,
            MAPPING,
            wanted=[main.SWITCHED_LABEL],
            hold_duration=None,
        )

        self.assertNotIn("camera0", graph)
        self.assertNotIn("camera1", graph)
        self.assertNotIn("split=", graph)
        self.assertIn("[0:v]setpts=PTS-STARTPTS[src0]", graph)
        self.assertIn("[1:v]setpts=PTS-STARTPTS[src1]", graph)

    def test_camera_only_graph_omits_the_overlay(self) -> None:
        graph = main.build_switch_filter_graph(
            split_plan(),
            TIMELINE,
            MAPPING,
            wanted=[main.CAMERA_LABELS[1]],
            hold_duration=None,
        )

        self.assertNotIn("overlay", graph)
        self.assertNotIn("camera0", graph)
        self.assertIn("crop=1920:2160:1920:0", graph)

    def test_both_modes_switch_on_the_same_camera_intervals(self) -> None:
        active_expression = main.speaker1_active_expression(TIMELINE, MAPPING)

        for plan in (split_plan(), separate_plan()):
            graph = main.build_switch_filter_graph(
                plan,
                TIMELINE,
                MAPPING,
                wanted=[main.SWITCHED_LABEL],
                hold_duration=None,
            )
            self.assertIn(active_expression, graph)
            self.assertIn("between(t\\,1.000\\,2.000)", graph)

    def test_holding_the_last_frame_keeps_every_output_audio_aligned(self) -> None:
        graph = main.build_switch_filter_graph(
            split_plan(),
            TIMELINE,
            MAPPING,
            wanted=[main.SWITCHED_LABEL, *main.CAMERA_LABELS],
            hold_duration=12.5,
        )

        self.assertEqual(graph.count("tpad=stop_mode=clone:stop_duration=12.500"), 2)

    def test_unknown_output_labels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            main.build_switch_filter_graph(
                split_plan(),
                TIMELINE,
                MAPPING,
                wanted=["nope"],
                hold_duration=None,
            )


class SinglePassRenderTests(unittest.TestCase):
    def _render(self, outputs: main.RenderOutputs, **overrides: object) -> list[list[str]]:
        with mock.patch("main.run") as run:
            main.render_pipeline(
                plan=overrides.pop("plan", split_plan()),
                audio_file=Path("combined.mp4"),
                timeline=TIMELINE,
                mapping=MAPPING,
                outputs=outputs,
                encoder="h264_nvenc",
                audio_codec="aac",
                preset="p1",
                crf=26,
                hwaccel="cuda",
                loop_videos=False,
                duration=3.0,
                max_parallel_encodes=overrides.pop("max_parallel_encodes", 3),
                **overrides,
            )
        return [call.args[0] for call in run.call_args_list]

    def test_every_artifact_is_produced_by_a_single_ffmpeg_invocation(self) -> None:
        commands = self._render(
            main.RenderOutputs(
                switched_video=Path("out.mp4"),
                camera_videos=(Path("camera_1.mp4"), Path("camera_2.mp4")),
                master_audio=Path("master_audio.m4a"),
            )
        )

        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command.count("-i"), 1)
        for path in ("out.mp4", "camera_1.mp4", "camera_2.mp4", "master_audio.m4a"):
            self.assertIn(path, command)

    def test_the_switched_output_keeps_the_configured_cq_quality(self) -> None:
        command = self._render(main.RenderOutputs(switched_video=Path("out.mp4")))[0]

        self.assertEqual(command[command.index("-cq") + 1], "26")
        self.assertNotIn("-tune", command)
        self.assertNotIn("constqp", command)

    def test_separate_audio_is_added_as_one_extra_input_and_shared(self) -> None:
        command = self._render(
            main.RenderOutputs(
                switched_video=Path("out.mp4"),
                master_audio=Path("master_audio.m4a"),
            ),
            plan=separate_plan(),
        )[0]

        self.assertEqual(command.count("-i"), 3)
        # Both the switched mux and the master audio read that one extra input.
        self.assertEqual(command.count("2:a:0"), 2)

    def test_a_session_limited_encoder_falls_back_to_the_fewest_extra_passes(self) -> None:
        commands = self._render(
            main.RenderOutputs(
                switched_video=Path("out.mp4"),
                camera_videos=(Path("camera_1.mp4"), Path("camera_2.mp4")),
                master_audio=Path("master_audio.m4a"),
            ),
            max_parallel_encodes=2,
        )

        self.assertEqual(len(commands), 2)
        self.assertIn("out.mp4", commands[0])
        self.assertIn("camera_1.mp4", commands[0])
        self.assertIn("camera_2.mp4", commands[1])
        # The master audio is muxed once, alongside the first pass.
        self.assertIn("master_audio.m4a", commands[0])
        self.assertNotIn("master_audio.m4a", commands[1])

    def test_hardware_session_limits_are_probed_once_and_cached(self) -> None:
        main._ENCODER_SESSION_LIMITS.clear()
        self.addCleanup(main._ENCODER_SESSION_LIMITS.clear)

        with mock.patch("main._probe_encoder_sessions", return_value=False) as probe:
            self.assertEqual(main.max_parallel_encoder_sessions("h264_nvenc", 3), 1)
            self.assertEqual(main.max_parallel_encoder_sessions("h264_nvenc", 3), 1)

        self.assertEqual(probe.call_count, 2)

    def test_software_encoders_are_never_probed_for_session_limits(self) -> None:
        with mock.patch("main._probe_encoder_sessions") as probe:
            self.assertEqual(main.max_parallel_encoder_sessions("libx264", 3), 3)

        probe.assert_not_called()


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
