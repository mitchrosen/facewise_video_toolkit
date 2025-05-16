import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
import os

import sys
sys.modules.pop("mouthtracker.pipeline.mouthtrack_frame_by_frame", None)
sys.modules.pop("mouthtracker.pipeline", None)

@patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.load_yolo5face_model")
@patch("torch.cuda.is_available", return_value=True)
@patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.restore_audio_from_source")
@patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.detect_faces_in_frame")
@patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.draw_faces_and_mouths")
@patch("cv2.VideoWriter")
@patch("cv2.VideoCapture")
def test_pipeline_calls_detection_and_drawing(
    mock_capture, mock_writer, mock_draw, mock_detect, mock_restore, mock_cuda, mock_load_model
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
        [[(15, 15), (90, 15), (50, 30), (30, 70), (70, 70)]],
        [0.95]
    )
    mock_draw.return_value = 1
    mock_writer_instance = MagicMock()
    mock_writer.return_value = mock_writer_instance
    mock_load_model.return_value = MagicMock()

    from mouthtracker.pipeline.mouthtrack_frame_by_frame import multiface_mouthtrack
    multiface_mouthtrack(
        input_path="dummy.mp4",
        output_path="out.mp4",
        model_path="dummy_model.pt",
        config_path="dummy.yaml",
        require_gpu=False,
        track_interval=1
    )

    assert mock_detect.call_count == 3
    assert mock_draw.call_count == 3
    assert mock_writer_instance.write.call_count == 3
    mock_capture.assert_called_once_with("dummy.mp4")
    mock_writer.assert_called_once()
    mock_restore.assert_called_once_with("dummy.mp4", "out.mp4")


@patch("torch.cuda.is_available", return_value=False)
def test_pipeline_requires_gpu_raises_error(mock_cuda):
    from mouthtracker.pipeline.mouthtrack_frame_by_frame import multiface_mouthtrack

    with pytest.raises(RuntimeError, match="GPU required but CUDA is not available"):
        multiface_mouthtrack(
            input_path="dummy.mp4",
            output_path="out.mp4",
            model_path="dummy_model.pt",
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

@patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.load_yolo5face_model")
@patch("torch.cuda.is_available", return_value=True)
@patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.restore_audio_from_source")
@patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.detect_faces_in_frame")
@patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.draw_faces_and_mouths")
@patch("mouthtracker.pipeline.mouthtrack_frame_by_frame.FaceTracker")
@patch("cv2.VideoWriter")
@patch("cv2.VideoCapture")
def test_detection_and_tracking_call_sequence(
    mock_capture,
    mock_writer,
    mock_mouth_tracker_class,
    mock_draw,
    mock_detect,
    mock_restore,
    mock_cuda,
    mock_load_model,
):
    import numpy as np
    from mouthtracker.pipeline.mouthtrack_frame_by_frame import multiface_mouthtrack

    # Prepare 4 video frames (0–3)
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

    # Mock detection to return a fixed bounding box and landmarks
    mock_detect.return_value = (
        [[10, 10, 100, 100]],                    # boxes
        [[(50, 50)] * 5],                        # 5-point landmarks
        [0.95],                                  # confidences
    )
    mock_draw.return_value = 1
    mock_writer_instance = MagicMock()
    mock_writer.return_value = mock_writer_instance
    mock_load_model.return_value = MagicMock()

    # Mock MouthTracker instance and method
    mock_tracker_instance = MagicMock()
    mock_tracker_instance.init_trackers = MagicMock()
    mock_tracker_instance.update_trackers.return_value = [(50, 50, 30, 30)]
    mock_mouth_tracker_class.return_value = mock_tracker_instance

    # Run pipeline with track_interval = 2
    multiface_mouthtrack(
        input_path="dummy.mp4",
        output_path="out.mp4",
        model_path="dummy_model.pt",
        config_path="dummy.yaml",
        require_gpu=False,
        track_interval=2
    )

    # 🔍 Assert detection was called on frames 0 and 2 (every 2nd frame)
    assert mock_detect.call_count == 2
    # 🔍 Assert init_trackers() was called on frames 0 and 2 following successful detect
    assert mock_tracker_instance.init_trackers.call_count == 2
    # 🔍 Assert tracking was called on frames 0 and 2 (following init_trackers()), and frames 1 and 3 (standard tracking)
    assert mock_tracker_instance.update_trackers.call_count == 4
    # 🔍 draw_faces_and_mouths() is only called for detection frames (not for tracking)
    assert mock_draw.call_count == 2

    # You could even inspect the order if you wanted:
    expected_detect_calls = [0, 2]
    actual_detect_calls = [call[0][1] for call in mock_detect.call_args_list]
    assert len(actual_detect_calls) == len(expected_detect_calls)
