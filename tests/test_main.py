from __future__ import annotations

import json
import sys
import tempfile
import types
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

    def test_videotoolbox_quality_changes_with_requested_crf(self) -> None:
        high_quality: list[str] = []
        low_quality: list[str] = []

        main.append_video_encoding_options(
            high_quality,
            encoder="h264_videotoolbox",
            preset="p1",
            crf=18,
        )
        main.append_video_encoding_options(
            low_quality,
            encoder="h264_videotoolbox",
            preset="p1",
            crf=35,
        )

        self.assertEqual(high_quality[:1], ["-q:v"])
        self.assertEqual(low_quality[:1], ["-q:v"])
        self.assertNotEqual(high_quality, low_quality)
        self.assertGreater(int(high_quality[1]), int(low_quality[1]))


class HardwareSelectionTests(unittest.TestCase):
    def test_auto_encoder_uses_apple_videotoolbox_when_available(self) -> None:
        with mock.patch(
            "main.available_ffmpeg_encoders",
            return_value=" V..... h264_videotoolbox VideoToolbox H.264 Encoder",
        ):
            self.assertEqual(main.choose_video_encoder("auto"), "h264_videotoolbox")

    def test_auto_encoder_prefers_hevc_videotoolbox_for_10bit_source(self) -> None:
        with mock.patch(
            "main.available_ffmpeg_encoders",
            return_value=" V..... h264_videotoolbox\n V..... hevc_videotoolbox",
        ):
            self.assertEqual(
                main.choose_video_encoder("auto", prefer_10bit=True),
                "hevc_videotoolbox",
            )
            self.assertEqual(
                main.choose_video_encoder("auto"), "h264_videotoolbox"
            )

    def test_videotoolbox_refuses_to_pretend_hardware_is_lossless(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot guarantee lossless"):
            main.append_video_encoding_options(
                [], encoder="h264_videotoolbox", preset="p6", crf=16, lossless=True
            )

    def test_software_x265_lossless_preserves_10bit(self) -> None:
        command: list[str] = []
        main.append_video_encoding_options(
            command, encoder="libx265", preset="p6", crf=16, lossless=True
        )
        self.assertEqual(command, ["-preset", "slow", "-x265-params", "lossless=1"])

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



    def test_nvenc_quality_uses_uncapped_vbr_and_detail_preserving_tuning(self) -> None:
        command: list[str] = []
        main.append_video_encoding_options(command, encoder="h264_nvenc", preset="p6", crf=16)
        for key, value in {
            "-preset": "p6", "-tune": "hq", "-rc": "vbr", "-cq": "16",
            "-b:v": "0", "-multipass": "fullres", "-spatial-aq": "1",
            "-temporal-aq": "1", "-rc-lookahead": "20",
        }.items():
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertNotIn("-qp", command)

    def test_hevc_nvenc_uses_the_same_quality_controls(self) -> None:
        command: list[str] = []
        main.append_video_encoding_options(command, encoder="hevc_nvenc", preset="p6", crf=16)
        self.assertEqual(command[command.index("-b:v") + 1], "0")
        self.assertEqual(command[command.index("-cq") + 1], "16")

    def test_software_encoder_defaults_to_slow_when_nvenc_preset_is_selected(self) -> None:
        command: list[str] = []
        main.append_video_encoding_options(command, encoder="libx264", preset="p6", crf=16)
        self.assertEqual(command, ["-preset", "slow", "-crf", "16"])

    def test_encoding_rejects_invalid_quality(self) -> None:
        with self.assertRaisesRegex(ValueError, "CRF/CQ"):
            main.append_video_encoding_options([], encoder="h264_nvenc", preset="p6", crf=-1)

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

    def test_overlay_does_not_force_all_cameras_to_eight_bit(self) -> None:
        graph = main.build_switch_filter_graph(
            split_plan(), TIMELINE, MAPPING,
            wanted=[main.SWITCHED_LABEL], hold_duration=None,
        )
        self.assertIn("format=yuv420[switched]", graph)
        self.assertNotIn("format=auto", graph)

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

    def test_hevc_with_10bit_source_preserves_10bit_filter_and_encoder_format(self) -> None:
        plan = main.build_render_plan(
            main.MODE_SPLIT_VIDEO, [Path("source.mp4")], sizes=[(3840, 2160)]
        )
        from dataclasses import replace
        plan = replace(plan, source_pixel_formats=("yuv420p10le",))
        graph = main.build_switch_filter_graph(
            plan, TIMELINE, MAPPING, wanted=[main.SWITCHED_LABEL],
            hold_duration=None, overlay_format="yuv420p10",
        )
        self.assertIn("format=yuv420p10[switched]", graph)
        with mock.patch("main.run") as run:
            main.render_pipeline(
                plan=plan, audio_file=Path("source.mp4"), timeline=TIMELINE,
                mapping=MAPPING, outputs=main.RenderOutputs(switched_video=Path("out.mp4")),
                encoder="hevc_nvenc", audio_codec="aac", preset="p6", crf=16,
                hwaccel=None, loop_videos=False, duration=3.0, max_parallel_encodes=1,
            )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-pix_fmt") + 1], "p010le")

    def test_the_switched_output_keeps_the_configured_cq_quality(self) -> None:
        command = self._render(main.RenderOutputs(switched_video=Path("out.mp4")))[0]

        self.assertEqual(command[command.index("-cq") + 1], "26")
        self.assertEqual(command[command.index("-rc") + 1], "vbr")
        self.assertEqual(command[command.index("-b:v") + 1], "0")
        self.assertEqual(command[command.index("-tune") + 1], "hq")
        self.assertEqual(command[command.index("-b:a") + 1], "256k")
        self.assertNotIn("constqp", command)

    def test_lossless_flag_reaches_the_actual_ffmpeg_output(self) -> None:
        with mock.patch("main.run") as run:
            main.render_pipeline(
                plan=split_plan(), audio_file=Path("combined.mp4"),
                timeline=TIMELINE, mapping=MAPPING,
                outputs=main.RenderOutputs(switched_video=Path("out.mp4")),
                encoder="h264_nvenc", audio_codec="aac", preset="p6", crf=16,
                hwaccel=None, loop_videos=False, duration=3.0,
                max_parallel_encodes=1, lossless=True,
            )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-qp") + 1], "0")
        self.assertNotIn("-cq", command)

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

class ProgressReportingTests(unittest.TestCase):
    def test_diarization_reports_progress_before_timeline_building(self) -> None:
        request = job.JobRequest(
            mode=main.MODE_SPLIT_VIDEO,
            videos=(Path("combined.mp4"),),
            audio_file=Path("combined.mp4"),
            output_video=Path("out.mp4"),
            segments_json=Path("segments.json"),
        )
        segments = [
            main.Segment(0.0, 2.0, "SPEAKER_00"),
            main.Segment(2.0, 4.0, "SPEAKER_01"),
        ]
        updates: list[tuple[float, str]] = []

        def fake_diarize(*args, progress, **kwargs):
            progress(0.5, "Segmenting speakers")
            return segments, segments

        with (
            mock.patch("job.pipeline.diarize_audio", side_effect=fake_diarize),
            mock.patch("job.pipeline.write_segments_json"),
        ):
            job.resolve_timeline(
                request,
                duration=4.0,
                progress=lambda fraction, message: updates.append((fraction, message)),
            )

        self.assertTrue(
            any(0.20 < fraction < 0.62 for fraction, _ in updates),
            updates,
        )

    def test_render_pipeline_forwards_ffmpeg_progress(self) -> None:
        updates: list[float] = []

        def fake_run(command, **kwargs):
            progress = kwargs.get("progress")
            self.assertIsNotNone(progress)
            progress(0.5)

        with mock.patch("main.run", side_effect=fake_run):
            main.render_pipeline(
                plan=split_plan(),
                audio_file=Path("combined.mp4"),
                timeline=TIMELINE,
                mapping=MAPPING,
                outputs=main.RenderOutputs(switched_video=Path("out.mp4")),
                encoder="h264_nvenc",
                audio_codec="aac",
                preset="p1",
                crf=26,
                hwaccel="cuda",
                loop_videos=False,
                duration=3.0,
                max_parallel_encodes=1,
                progress=updates.append,
            )

        self.assertEqual(updates, [0.5])

class DiarizationConfigurationTests(unittest.TestCase):
    def test_model_loader_returning_none_is_reported_as_access_error(self) -> None:
        class FakePipeline:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return None

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.device = lambda value: value
        fake_pyannote = types.ModuleType("pyannote")
        fake_pyannote_audio = types.ModuleType("pyannote.audio")
        fake_pyannote_audio.Pipeline = FakePipeline

        with (
            mock.patch("main._ensure_torchcodec_ffmpeg_dlls"),
            mock.patch.dict(
                sys.modules,
                {
                    "dotenv": fake_dotenv,
                    "torch": fake_torch,
                    "pyannote": fake_pyannote,
                    "pyannote.audio": fake_pyannote_audio,
                },
            ),
        ):
            with self.assertRaisesRegex(main.DiarizationError, "HF_TOKEN is valid"):
                main.diarize_audio(
                    Path("conversation.wav"),
                    model=main.DIARIZATION_MODEL,
                    hf_token="hf_test",
                    device="cpu",
                    min_speakers=None,
                    max_speakers=None,
                    num_speakers=2,
                )

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

    def test_diarization_memory_is_returned_before_rendering(self) -> None:
        events: list[str] = []

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def empty_cache() -> None:
                events.append("cuda")

        class FakeTorch:
            cuda = FakeCuda()

        with (
            mock.patch.dict(sys.modules, {"torch": FakeTorch()}),
            mock.patch("gc.collect", side_effect=lambda: events.append("gc")),
        ):
            main.release_diarization_resources()

        self.assertEqual(events, ["gc", "cuda"])

    def test_aac_soundtrack_is_copied_instead_of_reencoded(self) -> None:
        with mock.patch("main.ffprobe_json", return_value={"streams": [
            {"codec_type": "audio", "codec_name": "aac"}
        ]}):
            self.assertEqual(main.source_audio_codec(Path("input.mp4")), "aac")
            self.assertTrue(main.media_has_audio(Path("input.mp4")))
        command: list[str] = []
        main._append_video_output(
            command, label="switched", path=Path("out.mp4"), plan=split_plan(),
            encoder="h264_nvenc", preset="p6", crf=16, duration=1.0,
            audio_input_index=0, audio_codec="copy",
        )
        self.assertEqual(command[command.index("-c:a") + 1], "copy")
        self.assertNotIn("-b:a", command)

    def test_videotoolbox_lossless_job_selects_software_encoder(self) -> None:
        events: list[tuple[str, object]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "combined.mp4"
            source.touch()
            request = job.JobRequest(
                mode=main.MODE_SPLIT_VIDEO, videos=(source,), audio_file=source,
                output_video=Path(tmpdir) / "output.mp4",
                segments_json=Path(tmpdir) / "segments.json", lossless=True,
            )
            with (
                mock.patch("job.pipeline.require_tool"),
                mock.patch("job.pipeline.build_render_plan", return_value=split_plan()),
                mock.patch("job.pipeline.choose_video_encoder", return_value="h264_videotoolbox"),
                mock.patch("job.pipeline.media_duration", return_value=1.0),
                mock.patch("job.pipeline.source_audio_codec", return_value="aac"),
                mock.patch("job.resolve_timeline", return_value=(TIMELINE, MAPPING, "test")),
                mock.patch("job.pipeline.release_diarization_resources"),
                mock.patch("job.pipeline.render_pipeline", side_effect=lambda **kw: events.append((kw["encoder"],kw["lossless"])) or 1),
            ):
                result=job.run_job(request)
        self.assertEqual(events, [("libx264", True)])
        self.assertTrue(any("VideoToolbox" in note for note in result.warnings))

    def test_job_releases_diarization_memory_before_ffmpeg(self) -> None:
        events: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "combined.mp4"
            video.touch()
            request = job.JobRequest(
                mode=main.MODE_SPLIT_VIDEO,
                videos=(video,),
                audio_file=video,
                output_video=root / "out.mp4",
                segments_json=root / "segments.json",
            )

            with (
                mock.patch("job.pipeline.require_tool"),
                mock.patch("job.pipeline.build_render_plan", return_value=split_plan()),
                mock.patch("job.pipeline.choose_video_encoder", return_value="h264_nvenc"),
                mock.patch("job.pipeline.choose_hwaccel", return_value="cuda"),
                mock.patch("job.pipeline.media_duration", return_value=3.0),
                mock.patch("job.pipeline.source_audio_codec", return_value="aac"),
                mock.patch("job.resolve_timeline", return_value=(TIMELINE, MAPPING, "test")),
                mock.patch(
                    "job.pipeline.release_diarization_resources",
                    create=True,
                    side_effect=lambda: events.append("release"),
                ),
                mock.patch(
                    "job.pipeline.render_pipeline",
                    side_effect=lambda **_: events.append("render") or 1,
                ),
            ):
                job.run_job(request)

        self.assertEqual(events, ["release", "render"])


if __name__ == "__main__":
    unittest.main()
