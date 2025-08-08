import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
import pytest
from facekit.pipeline import generate_shot_features
from facekit.postprocessing.validate_shot_features_json import validate_shot_features_json

SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "shot_features.schema.json"


@pytest.mark.timeout(10)
@patch("facekit.pipeline.generate_shot_features.FaceDetector")
@patch("facekit.pipeline.generate_shot_features.load_yolo5face_model")
@patch("torch.cuda.is_available", return_value=True)
@patch("facekit.pipeline.generate_shot_features.detect_scenes")
@patch("cv2.VideoCapture")
def test_generate_shot_features_runs(
    mock_capture,
    mock_detect_scenes,
    mock_cuda,
    mock_load_detector_model,
    mock_face_detector_class,
    tmp_path
):
    # Simulate 2 shots
    s0, e0 = MagicMock(), MagicMock()
    s1, e1 = MagicMock(), MagicMock()
    s0.get_frames.return_value = 0
    e0.get_frames.return_value = 2
    s1.get_frames.return_value = 3
    e1.get_frames.return_value = 5
    mock_detect_scenes.return_value = [(s0, e0), (s1, e1)]

    # Mock FaceDetector instance and detection output
    mock_detector_instance = MagicMock()
    mock_detector_instance.detect_faces_in_frame.side_effect = [
        ([[10, 10, 100, 100]], [[(50, 50)] * 5], [0.9]),
        ([[30, 30, 130, 130]], [[(70, 70)] * 5], [0.95])
    ]
    mock_face_detector_class.return_value = mock_detector_instance

    # Dummy detector model
    mock_load_detector_model.return_value = MagicMock()

    # Dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Simulated cv2.VideoCapture
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda x: {
        7: 6, 4: 480, 3: 640, 5: 30
    }.get(x, 0)
    mock_cap.set.return_value = None
    mock_cap.release.return_value = None
    mock_cap.read.side_effect = [(True, dummy_frame)] * 6 + [(False, None)]
    mock_capture.return_value = mock_cap

    # Run
    input_video = "tests/data/short_video.mp4"
    output_json_path = tmp_path / "shot_features.json"
    generate_shot_features.generate_shot_features_json(str(input_video), str(output_json_path))

    # Assert JSON structure
    parsed = json.loads(output_json_path.read_text())
    assert isinstance(parsed, dict)
    assert "shots" in parsed
    assert len(parsed["shots"]) == 2

    for shot in parsed["shots"]:
        assert "first_frame" in shot
        assert "last_frame" in shot
        assert "detected_faces" in shot
        assert shot["detected_faces"]["face_count"] == 1
        assert len(shot["detected_faces"]["face_details"]) == 1


@pytest.mark.timeout(10)
@patch("facekit.pipeline.generate_shot_features.FaceDetector")
@patch("facekit.pipeline.generate_shot_features.load_yolo5face_model")
@patch("torch.cuda.is_available", return_value=True)
@patch("facekit.pipeline.generate_shot_features.detect_scenes")
@patch("cv2.VideoCapture")
def test_generate_shot_features_schema_compliance(
    mock_capture,
    mock_detect_scenes,
    mock_cuda,
    mock_load_detector_model,
    mock_face_detector_class,
    tmp_path
):
    # Simulate 2 shots
    s0, e0 = MagicMock(), MagicMock()
    s1, e1 = MagicMock(), MagicMock()
    s0.get_frames.return_value = 0
    e0.get_frames.return_value = 2
    s1.get_frames.return_value = 2  # contiguous
    e1.get_frames.return_value = 6
    mock_detect_scenes.return_value = [(s0, e0), (s1, e1)]

    # Mock FaceDetector instance and detection output
    mock_detector_instance = MagicMock()
    mock_detector_instance.detect_faces_in_frame.side_effect = [
        ([[10, 10, 100, 100]], [[(50, 50)] * 5], [0.9]),   # frame 1
        ([[30, 30, 130, 130]], [[(70, 70)] * 5], [0.94])   # frame 4
    ]
    mock_face_detector_class.return_value = mock_detector_instance

    # Dummy detector model
    mock_load_detector_model.return_value = MagicMock()

    # Dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Simulated cv2.VideoCapture
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda x: {
        7: 6, 4: 480, 3: 640, 5: 30
    }.get(x, 0)
    mock_cap.set.return_value = None
    mock_cap.release.return_value = None
    mock_cap.read.side_effect = [(True, dummy_frame)] * 6 + [(False, None)]
    mock_capture.return_value = mock_cap

    # Run
    input_video = "tests/data/short_video.mp4"
    output_path = tmp_path / "output.json"
    generate_shot_features.generate_shot_features_json(str(input_video), str(output_path))

    # Validate schema
    result = validate_shot_features_json(str(output_path), SCHEMA_PATH, total_frame_count=6)
    assert result == []
