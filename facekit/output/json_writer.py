import cv2
import torch
import json
from typing import Dict, List
from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.detection.face_detector import FaceDetector
from facekit.tracking.face_tracker import FaceTracker
from facekit.tracking.face_structures import FaceTrack

def multiface_tracking_to_json(
    input_path: str,
    output_json_path: str,
    detector_model_path: str,
    config_path: str,
    require_gpu: bool = False,
    min_face: int = 10,
    track_interval: int = 30,
    tracker_type: str = "CSRT"
) -> None:
    """
    Performs face detection and tracking on a video and writes results to a JSON file.

    For each frame, records the coordinates of all detected or tracked faces.
    Each JSON entry includes the frame number and a list of face bounding boxes,
    with the bounding box source and (if detected) the confidence score.

    Args:
        input_path (str): Path to the input video file.
        output_json_path (str): Path where the JSON output will be saved.
        detector_model_path (str): Path to the YOLOv5 face detector weights (.pt file).
        config_path (str): Path to the YOLOv5 model configuration YAML file.
        require_gpu (bool): If True, raises an error if CUDA is not available.
        min_face (int): Minimum face size in pixels.
        track_interval (int): Number of frames to skip between detections.
        tracker_type (str): Tracker to use: "CSRT" or "KCF".

    Returns:
        None. Writes tracking output to the specified JSON file.
    """
    import time
    start_time = time.time()
 
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU required but CUDA is not available.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector_model = load_yolo5face_model(
        detector_model_path=detector_model_path,
        config_path=config_path,
        device=device,
        min_face=min_face
    )
    detector = FaceDetector(detector_model)

    elapsed = time.time() - start_time
    print(f"⏱️ startup time: {elapsed:.2f} seconds")

    start_time = time.time()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Could not open video: {input_path}")
        return

    tracker = FaceTracker(tracker_type=tracker_type)
    results = []
    frame_num = 0
    prev_face_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        should_detect = not prev_face_count or (frame_num % track_interval == 0)
        do_fallback = False
        frame_faces = []

        if not should_detect:
            tracked_boxes = tracker.update_trackers(frame)
            for box in tracked_boxes:
                if box is not None:
                    x, y, w, h = map(int, box)
                    frame_faces.append({
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "source": "tracked"
                    })
                else:
                    do_fallback = True

        if should_detect or do_fallback:
            result = detector.detect_faces_in_frame(frame, target_size=640)
            if result is not None:
                boxes, landmarks, confidences = result
                if boxes:
                    tracker.init_trackers(frame, [
                        (x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes
                    ])
                    _ = tracker.update_trackers(frame)  # <- Ensures tracker is initialized fully

                    for (x1, y1, x2, y2), conf in zip(boxes, confidences):
                        frame_faces.append({
                            "x": int(x1),
                            "y": int(y1),
                            "w": int(x2 - x1),
                            "h": int(y2 - y1),
                            "conf": round(float(conf), 3),
                            "source": "detected" if should_detect else "fallback"
                        })

        results.append({
            "frame": frame_num,
            "faces": frame_faces
        })

        prev_face_count = len(frame_faces)
        frame_num += 1

    cap.release()
    elapsed = time.time() - start_time
    print(f"⏱️ detection/tracking time: {elapsed:.2f} seconds")

    start_time = time.time()
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Write json time: {elapsed:.2f} seconds")

    print(f"JSON tracking output saved to: {output_json_path}")
    print(f"Total frames processed: {frame_num}")


def export_vchunk_id_map(
    shot_to_tracks: Dict[str, List[FaceTrack]],
    output_path: str
):
    """
    Saves a mapping of shot → track → vchunk ID to a JSON file.

    Args:
        shot_to_tracks: Dict mapping shot_id to list of FaceTrack objects
        output_path: Path to output JSON
    """
    result = {}
    for shot_id, track_list in shot_to_tracks.items():
        result[shot_id] = {
            track.track_id: f"face_{track.vchunk_id}"
            for track in track_list
            if track.vchunk_id is not None
        }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
