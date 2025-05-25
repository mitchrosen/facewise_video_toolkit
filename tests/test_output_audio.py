import unittest
from unittest.mock import patch
from facekit.output.audio_tools import restore_audio_from_source
import subprocess

class TestRestoreAudioFromSource(unittest.TestCase):

    @patch("facekit.output.audio_tools.os.replace")
    @patch("facekit.output.audio_tools.subprocess.run")
    def test_restore_audio_command_contains_expected_parts(self, mock_run, mock_replace):
        input_path = "input_with_audio.mp4"
        output_path = "output_no_audio.mp4"
        expected_temp = "temp_output.mp4"

        restore_audio_from_source(input_path, output_path)

        # Ensure subprocess.run was called
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args

        # Extract command list
        cmd = args[0]

        # Assert key elements are present in any order
        required_parts = {
            "ffmpeg", "-i", input_path, output_path,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", "-y", "-hide_banner", "-loglevel", "quiet", expected_temp
        }

        self.assertTrue(required_parts.issubset(set(cmd)))

        # Ensure os.replace was called
        mock_replace.assert_called_once_with(expected_temp, output_path)

    @patch("facekit.output.audio_tools.subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg"))
    def test_raises_if_ffmpeg_fails(self, mock_run):
        """Ensure ffmpeg failure is propagated via CalledProcessError."""
        with self.assertRaises(subprocess.CalledProcessError, msg="Expected ffmpeg failure to be propagated"):
            restore_audio_from_source("input.mp4", "output.mp4")

    @patch("facekit.output.audio_tools.os.replace", side_effect=FileNotFoundError("temp file missing"))
    @patch("facekit.output.audio_tools.subprocess.run")
    def test_raises_if_os_replace_fails(self, mock_run, mock_replace):
        """Ensure os.replace failure is propagated (e.g., missing temp output)."""
        with self.assertRaises(FileNotFoundError, msg="Expected os.replace failure to be propagated"):
            restore_audio_from_source("input.mp4", "output.mp4")
