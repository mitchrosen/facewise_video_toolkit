import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import json
from unittest.mock import patch
from facekit.postprocessing import crop_portrait_video


class TestCropPortraitVideo(unittest.TestCase):

    def setUp(self):
        patcher = patch("facekit.postprocessing.crop_portrait_video.restore_audio_from_source")
        self.mock_restore = patcher.start()
        self.addCleanup(patcher.stop)

    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    @patch("builtins.open", new_callable=mock_open)
    @patch("facekit.postprocessing.crop_portrait_video.json.load")
    def test_empty_json_does_not_crash(self, mock_json_load, mock_file, mock_capture, mock_writer):
        mock_json_load.return_value = []
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.return_value = (False, None)
        mock_cap_instance.get.side_effect = [1920, 1080, 30.0]
        mock_capture.return_value = mock_cap_instance

        crop_portrait_video.crop_video_from_json("fake.json", "video.mp4", "out.mp4")
        mock_file.assert_called_once_with("fake.json", "r")

    @patch("facekit.postprocessing.crop_portrait_video.crop_for_single_face")
    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    @patch("builtins.open", new_callable=mock_open)
    @patch("facekit.postprocessing.crop_portrait_video.json.load")
    def test_calls_single_face_crop(self, mock_json_load, mock_file, mock_capture, mock_writer, mock_crop_single):
        mock_json_load.return_value = [{"faces": [{"bbox": [100, 100, 200, 200]}]}]

        frame = np.ones((1080, 1920, 3), dtype=np.uint8)
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, frame), (False, None)]
        mock_cap_instance.get.side_effect = [1920, 1080, 30.0]
        mock_capture.return_value = mock_cap_instance

        mock_crop_single.return_value = frame

        crop_portrait_video.crop_video_from_json("fake.json", "video.mp4", "out.mp4")

        mock_crop_single.assert_called_once()

    @patch("facekit.postprocessing.crop_portrait_video.crop_for_two_faces")
    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    @patch("builtins.open", new_callable=mock_open)
    @patch("facekit.postprocessing.crop_portrait_video.json.load")
    def test_calls_two_face_crop(self, mock_json_load, mock_file, mock_capture, mock_writer, mock_crop_two):
        mock_json_load.return_value = [{"faces": [
            {"bbox": [100, 100, 200, 200]}, {"bbox": [400, 100, 500, 200]}
        ]}]

        frame = np.ones((1080, 1920, 3), dtype=np.uint8)
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, frame), (False, None)]
        mock_cap_instance.get.side_effect = [1920, 1080, 30.0]
        mock_capture.return_value = mock_cap_instance

        mock_crop_two.return_value = frame

        crop_portrait_video.crop_video_from_json("fake.json", "video.mp4", "out.mp4")

        mock_crop_two.assert_called_once()

    @patch("facekit.postprocessing.crop_portrait_video.crop_for_three_faces")
    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    @patch("builtins.open", new_callable=mock_open)
    @patch("facekit.postprocessing.crop_portrait_video.json.load")
    def test_calls_three_face_crop(self, mock_json_load, mock_file, mock_capture, mock_writer, mock_crop_three):
        mock_json_load.return_value = [{"faces": [
            {"bbox": [100, 100, 200, 200]},
            {"bbox": [400, 100, 500, 200]},
            {"bbox": [200, 100, 400, 200]}
        ]}]

        frame = np.ones((1080, 1920, 3), dtype=np.uint8)
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, frame), (False, None)]
        mock_cap_instance.get.side_effect = [1920, 1080, 30.0]
        mock_capture.return_value = mock_cap_instance

        mock_crop_three.return_value = frame

        crop_portrait_video.crop_video_from_json("fake.json", "video.mp4", "out.mp4")

        mock_crop_three.assert_called_once()

    @patch("facekit.postprocessing.crop_portrait_video.letterbox_frame")
    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    @patch("builtins.open", new_callable=mock_open)
    @patch("facekit.postprocessing.crop_portrait_video.json.load")
    def test_calls_letterbox_frame_zeroFaces(self, mock_json_load, mock_file, mock_capture, mock_writer, mock_letterbox):
        mock_json_load.return_value = [{"faces": []}]

        frame = np.ones((1080, 1920, 3), dtype=np.uint8)
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, frame), (False, None)]
        mock_cap_instance.get.side_effect = [1920, 1080, 30.0]
        mock_capture.return_value = mock_cap_instance

        mock_letterbox.return_value = frame

        crop_portrait_video.crop_video_from_json("fake.json", "video.mp4", "out.mp4")

        mock_letterbox.assert_called_once()

    @patch("facekit.postprocessing.crop_portrait_video.letterbox_frame")
    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    @patch("builtins.open", new_callable=mock_open)
    @patch("facekit.postprocessing.crop_portrait_video.json.load")
    def test_calls_letterbox_frame_fourFaces(self, mock_json_load, mock_file, mock_capture, mock_writer, mock_letterbox):
        mock_json_load.return_value = [{"faces": [
            {"bbox": [100, 100, 200, 200]},
            {"bbox": [400, 100, 500, 200]},
            {"bbox": [200, 100, 400, 200]},
            {"bbox": [300, 100, 600, 200]}
        ]}]

        frame = np.ones((1080, 1920, 3), dtype=np.uint8)
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, frame), (False, None)]
        mock_cap_instance.get.side_effect = [1920, 1080, 30.0]
        mock_capture.return_value = mock_cap_instance

        mock_letterbox.return_value = frame

        crop_portrait_video.crop_video_from_json("fake.json", "video.mp4", "out.mp4")

        mock_letterbox.assert_called_once()


if __name__ == "__main__":
    unittest.main()
