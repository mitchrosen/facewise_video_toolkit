import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
import os

import sys
sys.modules.pop("facekit.pipeline.mouthtrack_frame_by_frame", None)
sys.modules.pop("facekit.pipeline", None)

@patch("facekit.pipeline.mouthtrack_frame_by_frame.load_yolo5face_model")
@patch("torch.cuda.is_available", return_value=True)
@patch("facekit.pipeline.mouthtrack_frame_by_frame.restore_audio_from_source")
@patch("facekit.pipeline.mouthtrack_frame_by_frame.FaceDetector.detect_faces_in_frame")
@patch("facekit.pipeline.mouthtrack_frame_by_frame.draw_faces_and_mouths")
@patch("cv2.VideoWriter")
@patch("cv2.VideoCapture")
def test_pipeline_calls_detection_and_drawing(
    mock_capture,
    mock_writer,
    mock_draw,
    mock_detect_method,
    mock_restore,
    mock_cuda,
    mock_load_detector_model
):
    import numpy as np
    from unittest.mock import MagicMock
    from facekit.pipeline.mouthtrack_frame_by_frame import multiface_mouthtrack

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

    mock_detect_method.return_value = (
        [[10, 10, 100, 100]],
        [[(15, 15), (90, 15), (50, 30), (30, 70), (70, 70)]],
        [0.95]
    )
    mock_draw.return_value = 1
    mock_writer_instance = MagicMock()
    mock_writer.return_value = mock_writer_instance
    mock_load_detector_model.return_value = MagicMock()

    multiface_mouthtrack(
        input_path="dummy.mp4",
        output_path="out.mp4",
        detector_model_path="dummy_detector_model.pt",
        config_path="dummy.yaml",
        require_gpu=False,
        track_interval=1
    )

    assert mock_detect_method.call_count == 3
    assert mock_draw.call_count == 3
    assert mock_writer_instance.write.call_count == 3
    mock_capture.assert_called_once_with("dummy.mp4")
    mock_writer.assert_called_once()
    mock_restore.assert_called_once_with("dummy.mp4", "out.mp4")

@patch("torch.cuda.is_available", return_value=False)
def test_pipeline_requires_gpu_raises_error(mock_cuda):
    from facekit.pipeline.mouthtrack_frame_by_frame import multiface_mouthtrack

    with pytest.raises(RuntimeError, match="GPU required but CUDA is not available"):
        multiface_mouthtrack(
            input_path="dummy.mp4",
            output_path="out.mp4",
            detector_model_path="dummy_detector_model.pt",
            config_path="dummy.yaml",
            require_gpu=True
        )

def find_tracking_dot(frame, color=(0, 255, 255)):
    """
    Finds the first yellow dot in the frame by searching for approximate color.
    """
    # Allow for slight variation in color
    lower = np.array([0, 250, 250], dtype=np.uint8)
    upper = np.array([10, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(frame, lower, upper)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        return tuple(coords[0][0])  # (x, y)
    return None

from unittest.mock import patch, MagicMock
import numpy as np
import cv2

@patch("facekit.pipeline.mouthtrack_frame_by_frame.load_yolo5face_model")
@patch("torch.cuda.is_available", return_value=True)
@patch("facekit.pipeline.mouthtrack_frame_by_frame.restore_audio_from_source")
@patch("facekit.pipeline.mouthtrack_frame_by_frame.draw_faces_and_mouths")
@patch("facekit.pipeline.mouthtrack_frame_by_frame.FaceDetector")
@patch("facekit.pipeline.mouthtrack_frame_by_frame.FaceTracker")
@patch("cv2.VideoWriter")
@patch("cv2.VideoCapture")
def test_detection_and_tracking_call_sequence(
    mock_capture,
    mock_writer,
    mock_mouth_tracker_class,
    mock_face_detector_class,
    mock_draw,
    mock_restore,
    mock_cuda,
    mock_load_detector_model,
):
    from facekit.pipeline.mouthtrack_frame_by_frame import multiface_mouthtrack

    # === Prepare video ===
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap_instance = MagicMock()
    mock_cap_instance.isOpened.side_effect = [True] * 5 + [False]
    mock_cap_instance.read.side_effect = [(True, dummy_frame.copy()) for _ in range(5)]
    mock_cap_instance.get.side_effect = lambda x: {
        cv2.CAP_PROP_FPS: 30,
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
    }[x]
    mock_capture.return_value = mock_cap_instance

    # === Prepare mocked FaceDetector instance ===
    mock_detector_instance = MagicMock()
    mock_detector_instance.detect_faces_in_frame.return_value = (
        [[10, 10, 100, 100]],        # boxes
        [[(50, 50)] * 5],            # landmarks
        [0.95],                      # confidences
    )
    mock_face_detector_class.return_value = mock_detector_instance

    # === Mock draw and writer ===
    mock_draw.return_value = 1
    mock_writer.return_value = MagicMock()
    mock_load_detector_model.return_value = MagicMock()

    # === Mock tracker ===
    mock_tracker_instance = MagicMock()
    mock_tracker_instance.init_trackers = MagicMock()
    mock_tracker_instance.update_trackers.return_value = [(50, 50, 30, 30)]
    mock_mouth_tracker_class.return_value = mock_tracker_instance

    # === Run pipeline ===
    multiface_mouthtrack(
        input_path="dummy.mp4",
        output_path="out.mp4",
        detector_model_path="dummy_detector_model.pt",
        config_path="dummy.yaml",
        require_gpu=False,
        track_interval=2
    )

    # === Assertions ===
    assert mock_detector_instance.detect_faces_in_frame.call_count == 2
    assert mock_tracker_instance.init_trackers.call_count == 2
    assert mock_tracker_instance.update_trackers.call_count == 4
    assert mock_draw.call_count == 2

    # Optionally inspect call order
    expected_calls = [0, 2]
    actual_calls = [call[0][0] for call in mock_detector_instance.detect_faces_in_frame.call_args_list]
    assert len(actual_calls) == len(expected_calls)

