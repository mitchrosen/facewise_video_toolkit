import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

class TestMouthtrackPipeline(unittest.TestCase):
    @patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.load_yolo5face_model")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.restore_audio_from_source")
    @patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.detect_faces_in_frame")
    @patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.draw_faces_and_mouths")
    @patch("cv2.VideoWriter")
    @patch("cv2.VideoCapture")
    def test_pipeline_calls_detection_and_drawing(
        self, mock_capture, mock_writer, mock_draw, mock_detect, mock_restore, mock_cuda, mock_load_model
    ):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.side_effect = [True, True, True, True, False]
        mock_cap_instance.read.side_effect = [
            (True, dummy_frame.copy()),
            (True, dummy_frame.copy()),
            (True, dummy_frame.copy()),
            (False, None),
        ]
        mock_cap_instance.get.side_effect = lambda x: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480
        }[x]
        mock_capture.return_value = mock_cap_instance

        mock_detect.return_value = (
            [[10, 10, 100, 100]],
            [[[15, 15], [90, 15], [50, 30], [30, 70], [70, 70]]],
            [0.95]
        )
        mock_draw.return_value = 1
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        mock_load_model.return_value = MagicMock()  # Fake model

        from mouthtracker.pipeline.mouthtrack_frame_by_frame import level3_multiface_yolo5face_mouthtrack_v2
        level3_multiface_yolo5face_mouthtrack_v2(
            input_path="dummy.mp4",
            output_path="out.mp4",
            model_path="dummy_model.pt",
            config_path="dummy.yaml",
            require_gpu=False
        )

        self.assertEqual(mock_detect.call_count, 3)
        self.assertEqual(mock_draw.call_count, 3)
        self.assertEqual(mock_writer_instance.write.call_count, 3)
        mock_capture.assert_called_once_with("dummy.mp4")
        mock_writer.assert_called_once()
        mock_restore.assert_called_once_with("dummy.mp4", "out.mp4")
