from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

import app


class LocalPathInputTests(unittest.TestCase):
    def test_quoted_windows_style_path_is_used_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "large podcast.mp4"
            source.write_bytes(b"not copied")

            resolved = app._resolve_local_file(f'"{source}"', "the video")

            self.assertEqual(resolved, source.resolve())
            self.assertEqual(source.read_bytes(), b"not copied")

    def test_missing_local_path_is_rejected(self) -> None:
        with self.assertRaises(gr.Error):
            app._resolve_local_file(r"Z:\missing\podcast.mp4", "the video")

    def test_gradio_switcher_has_no_render_duration_override(self) -> None:
        parameters = inspect.signature(app._run_switcher).parameters

        self.assertNotIn("render_duration", parameters)

    def test_full_render_duration_is_accepted(self) -> None:
        output = Path("full.mp4")
        with mock.patch("app.pipeline.media_duration", return_value=2496.5):
            rendered = app._validate_full_render(
                output,
                expected_duration=2496.995,
            )

        self.assertEqual(rendered, 2496.5)

    def test_truncated_render_is_rejected(self) -> None:
        output = Path("partial.mp4")
        with mock.patch("app.pipeline.media_duration", return_value=2.5):
            with self.assertRaisesRegex(gr.Error, "incomplete"):
                app._validate_full_render(
                    output,
                    expected_duration=2496.995,
                )

    def test_completed_outputs_are_served_without_cache_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.mp4"
            output.write_bytes(b"video")
            missing = Path(tmpdir) / "missing.json"

            with mock.patch("app.gr.set_static_paths") as set_static_paths:
                app._serve_outputs_in_place(output, missing, None)

            set_static_paths.assert_called_once_with(paths=[output])


if __name__ == "__main__":
    unittest.main()
