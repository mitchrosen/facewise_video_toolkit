import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open
from facekit.output.json_writer import multiface_tracking_to_json


# --- Helpers for expected structure ---
def to_det_result(box, conf):
    return [list(box)], [[(0, 0)] * 5], [conf]

def to_json_face(box, conf=None, source="tracked"):
    if len(box) == 4 and isinstance(conf, float):  # detection box
        x1, y1, x2, y2 = box
        return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "conf": round(conf, 3), "source": source}
    elif len(box) == 4:  # tracked box
        x, y, w, h = box
        return {"x": x, "y": y, "w": w, "h": h, "source": source}


@patch("facekit.tracking.face_tracker.FaceTracker.update_trackers")
@patch("facekit.detection.face_detector.FaceDetector.detect_faces_in_frame")
@patch("facekit.output.json_writer.FaceDetector")
@patch("facekit.output.json_writer.load_yolo5face_model")
@patch("torch.cuda.is_available", return_value=True)
@patch("cv2.VideoCapture")
@patch("facekit.output.json_writer.open", new_callable=mock_open)
def test_multiface_tracking_to_json_precise_boxes(
    mock_file, mock_capture, mock_cuda, mock_load_detector_model,
    mock_detector_class, mock_detect, mock_track
):
    FRAME_BOX_CONFIDENCE = {
        0: ((10, 20, 110, 220), 0.95),  # Detect
        1: (12, 22, 110, 220),          # Track
        2: ((15, 25, 115, 225), 0.93),  # Detect
        3: ((20, 30, 120, 230), 0.91),  # Fallback
    }

    # --- Dummy video capture ---
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_cap = MagicMock()
    dummy_cap.isOpened.side_effect = [True] * 5 + [False]
    dummy_cap.read.side_effect = [(True, dummy_frame.copy()) for _ in range(4)]
    dummy_cap.get.side_effect = lambda x: {
        5: 30,  # FPS
        3: 640, 4: 480
    }[x]
    mock_capture.return_value = dummy_cap

    # --- Mock model loading ---
    mock_load_detector_model.return_value = MagicMock()

    # --- Mock FaceDetector instance and method ---
    mock_detector_instance = MagicMock()
    mock_detector_instance.detect_faces_in_frame.side_effect = [
        to_det_result(*FRAME_BOX_CONFIDENCE[0]),
        to_det_result(*FRAME_BOX_CONFIDENCE[2]),
        to_det_result(*FRAME_BOX_CONFIDENCE[3]),
    ]
    mock_detector_class.return_value = mock_detector_instance

    # --- Track update behavior across frames ---
    mock_track.side_effect = [
        [None],                       # frame 0 — post-detect
        [FRAME_BOX_CONFIDENCE[1]],   # frame 1 — tracked
        [None],                       # frame 2 — post-detect
        [None],                       # frame 3 — track fails
        [None],                       # frame 3 — post-fallback detect
    ]

    # --- Execute ---
    multiface_tracking_to_json(
        input_path="dummy.mp4",
        output_json_path="fake_output.json",
        detector_model_path="dummy.pt",
        config_path="dummy.yaml",
        require_gpu=False,
        track_interval=2,
        tracker_type="KCF"
    )

    # --- Read captured output ---
    handle = mock_file()
    written = "".join(call.args[0] for call in handle.write.call_args_list)
    data = json.loads(written)

    expected = [
        {"frame": 0, "faces": [to_json_face(FRAME_BOX_CONFIDENCE[0][0], FRAME_BOX_CONFIDENCE[0][1], "detected")]},
        {"frame": 1, "faces": [to_json_face(FRAME_BOX_CONFIDENCE[1])]},
        {"frame": 2, "faces": [to_json_face(FRAME_BOX_CONFIDENCE[2][0], FRAME_BOX_CONFIDENCE[2][1], "detected")]},
        {"frame": 3, "faces": [to_json_face(FRAME_BOX_CONFIDENCE[3][0], FRAME_BOX_CONFIDENCE[3][1], "fallback")]},
    ]

    assert len(data) == 4
    for i, (expected_frame, actual_frame) in enumerate(zip(expected, data)):
        assert expected_frame == actual_frame, f"Mismatch at frame {i}:\nExpected: {expected_frame}\nGot: {actual_frame}"
