import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open
from facekit.pipeline.json_writer import multiface_tracking_to_json

# --- Calculate derived detection results and expected JSON ---
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
@patch("facekit.pipeline.json_writer.load_yolo5face_model")
@patch("torch.cuda.is_available", return_value=True)
@patch("facekit.pipeline.json_writer.detect_faces_in_frame")
@patch("cv2.VideoCapture")
@patch("facekit.pipeline.json_writer.open", new_callable=mock_open)
def test_multiface_tracking_to_json_precise_boxes(
    mock_file, mock_capture, mock_detect, mock_cuda, mock_load_model, mock_track
):
    # --- Define box data constants for 4 frames ---
    FRAME_BOX_CONFIDENCE = {
        0: ((10, 20, 110, 220), 0.95),   # Detect
        1: (12, 22, 110, 220),         # Track
        2: ((15, 25, 115, 225), 0.93),   # Detect
        3: ((20, 30, 120, 230), 0.91),        # Fallback detect
    }

    # --- Setup dummy frame stream (4 frames total) ---
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_cap = MagicMock()
    dummy_cap.isOpened.side_effect = [True] * 5 + [False]  # 4 frames then stop
    dummy_cap.read.side_effect = [(True, dummy_frame.copy()) for _ in range(4)]
    dummy_cap.get.side_effect = lambda x: {
        5: 30,   # CAP_PROP_FPS
        3: 640,  # CAP_PROP_FRAME_WIDTH
        4: 480   # CAP_PROP_FRAME_HEIGHT
    }[x]
    mock_capture.return_value = dummy_cap

    # --- Setup mock model ---
    mock_load_model.return_value = MagicMock()  # Fake model object, never uses torch.load

    # --- Detection results with varying bounding boxes and confidences ---
    mock_detect.side_effect = [
        to_det_result(FRAME_BOX_CONFIDENCE[0][0], FRAME_BOX_CONFIDENCE[0][1]),  # frame 0, detect
        to_det_result(FRAME_BOX_CONFIDENCE[2][0], FRAME_BOX_CONFIDENCE[2][1]),  # frame 2, detect
        to_det_result(FRAME_BOX_CONFIDENCE[3][0], FRAME_BOX_CONFIDENCE[3][1]),  # frame 3, (fallback)
    ]

    # Frame-by-frame expected update_trackers() calls:
    # 0: after detect → ignore result
    # 1: track → should return one box
    # 2: after detect → ignore result
    # 3: track → return None → fallback
    mock_track.side_effect = [
        [None],                     # frame 0 – post-detect
        [FRAME_BOX_CONFIDENCE[1]], # frame 1 – successful track
        [None],                     # frame 2 – post-detect
        [None],                     # frame 3 – failed track → fallback
        [None],                      # frame 3 – post-fallback detect
    ]

    # --- Step 4: Run the function with track_interval=2 and where open is being mocked ---
    multiface_tracking_to_json(
        input_path="dummy.mp4",
        output_json_path="fake_output.json",
        model_path="dummy.pt",
        config_path="dummy.yaml",
        require_gpu=False,
        track_interval=2,
        tracker_type="KCF"  # Fast-failing so frame 3 falls back
    )

    # --- Step 5: Extract and parse JSON written to mock file ---
    handle = mock_file()
    written = "".join(call.args[0] for call in handle.write.call_args_list)
    data = json.loads(written)


    # --- Step 6: Validate structure and content of frames ---
    assert len(data) == 4, f"Expected 4 frames in JSON, got {len(data)}"

    expected = [
        {"frame": 0, "faces": [to_json_face(FRAME_BOX_CONFIDENCE[0][0], FRAME_BOX_CONFIDENCE[0][1], "detected")]},
        {"frame": 1, "faces": [to_json_face(FRAME_BOX_CONFIDENCE[1])]},
        {"frame": 2, "faces": [to_json_face(FRAME_BOX_CONFIDENCE[2][0], FRAME_BOX_CONFIDENCE[2][1], "detected")]},
        {"frame": 3, "faces": [to_json_face(FRAME_BOX_CONFIDENCE[3][0], FRAME_BOX_CONFIDENCE[3][1], "fallback")]},
    ]

    for i, (expected_frame, actual_frame) in enumerate(zip(expected, data)):
        assert expected_frame == actual_frame, f"Mismatch at frame {i}:\nExpected: {expected_frame}\nGot: {actual_frame}"