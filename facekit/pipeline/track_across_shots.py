import json
import cv2
from pathlib import Path
from typing import List

from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.detection.detection_helpers import detect_faces_in_frame
from facekit.tracking.face_tracks import FaceObservation, FaceTrack
from facekit.tracking.aggregator import ShotFaceTrackAggregator

def track_across_shots(
    video_path: str,
    shot_json_path: str,
    model_path: str = "models/yolov5n_state_dict.pt",
    config_path: str = "models/yolov5n.yaml",
    iou_thresh: float = 0.5,
    embedding_thresh: float = 0.6
) -> List[FaceTrack]:
    """
    Track faces across the entire video, shot by shot, returning intra-shot tracks.
    
    Args:
        video_path (str): Path to the input video.
        shot_json_path (str): Path to the shot_features.json file.
        model_path (str): Path to the YOLOv5 face model weights.
        config_path (str): Path to the YOLOv5 model config.
        iou_thresh (float): IoU threshold for track matching.
        embedding_thresh (float): Embedding similarity threshold for merging.

    Returns:
        List[FaceTrack]: A flat list of all face tracks from all shots.
    """
    video_path = Path(video_path)
    shot_json_path = Path(shot_json_path)
    
    # Load shot boundaries
    with open(shot_json_path) as f:
        shot_data = json.load(f)
    shots = shot_data["shots"]

    # Prepare model and video
    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    model = load_yolo5face_model(model_path=model_path, config_path=config_path, device=device)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    all_tracks = []
    next_global_id = 0

    for shot in shots:
        shot_number = shot["shot_number"]
        first = shot["first_frame"]
        last = shot["last_frame"]
        aggregator = ShotFaceTrackAggregator(
            shot_number=shot_number,
            iou_threshold=iou_thresh,
            embedding_threshold=embedding_thresh
        )

        for frame_idx in range(first, last + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                print(f"⚠️ Failed to read frame {frame_idx}")
                continue

            results = detect_faces_in_frame(model, frame)
            if results is None:
                continue
            boxes, _, confs = results
            observations = [
                FaceObservation(frame_idx=frame_idx, bbox=tuple(map(int, box)), confidence=conf)
                for box, conf in zip(boxes, confs)
            ]
            aggregator.add_frame_observations(frame_idx, observations)

        shot_tracks = aggregator.finalize_tracks()

        # 💡 assign globally unique IDs to each track
        for track in shot_tracks:
            track.track_id = next_global_id
            next_global_id += 1

        all_tracks.extend(shot_tracks)

    cap.release()
    return all_tracks
