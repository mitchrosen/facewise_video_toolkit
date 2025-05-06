from unittest.mock import patch
import unittest
import inspect
from mouthtracker.detection.yolo5face_model import load_yolo5face_model

class TestYolo5faceModel(unittest.TestCase):

    @patch("mouthtracker.yolo5faceInference.yolo5face.yoloface.face_detector.YoloDetector")
    def test_load_yolo5face_model(self, mock_detector):
        dummy_model_path = "model_path"
        dummy_config_path = "config_path"
        dummy_device = "device"
        dummy_min_face = "min_face"

        return_value = "return from detector"
        mock_detector.return_value = return_value

        result = load_yolo5face_model(
            model_path=dummy_model_path,
            config_path=dummy_config_path,
            device=dummy_device,
            min_face=dummy_min_face
        )

        mock_detector.assert_called_once_with(
            dummy_model_path,
            config_name=dummy_config_path,
            device=dummy_device,
            min_face=dummy_min_face
        )
        self.assertEqual(result, return_value)

    @patch("mouthtracker.yolo5faceInference.yolo5face.yoloface.face_detector.YoloDetector")
    def test_load_yolo5face_model_with_min_face_default(self, mock_detector):
        dummy_model_path = "model_path"
        dummy_config_path = "config_path"
        dummy_device = "device"

        defaults = inspect.signature(load_yolo5face_model).parameters
        expected_min_face = defaults["min_face"].default

        expected_return = "mock_detector_instance"
        mock_detector.return_value = expected_return

        result = load_yolo5face_model(
            model_path=dummy_model_path,
            config_path=dummy_config_path,
            device=dummy_device
        )

        mock_detector.assert_called_once_with(
            dummy_model_path,
            config_name=dummy_config_path,
            device=dummy_device,
            min_face=expected_min_face
        )
        self.assertEqual(result, expected_return)
