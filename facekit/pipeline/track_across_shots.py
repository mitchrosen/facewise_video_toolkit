import json
import cv2
from pathlib import Path
from typing import List
import numpy as np

from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.detection.detection_helpers import detect_faces_in_frame
from facekit.tracking.face_structures import FaceObservation, FaceTrack
from facekit.tracking.aggregator import ShotFaceTrackAggregator

def track_across_shots(
    video_path: str,
    shot_json_path: str,
    embedder=None ,
    model_path: str = "models/yolov5n_state_dict.pt",
    config_path: str = "models/yolov5n.yaml",
    iou_thresh: float = 0.5,
    embedding_thresh: float = 0.6,
) -> List[FaceTrack]:
    """
    Track faces across the entire video, shot by shot, returning intra-shot tracks.
    
    Args:
        video_path (str): Path to the input video.
        shot_json_path (str): Path to the shot_features.json file.
        embedder (Optional[object]): Optional face embedding extractor with a
            `get_embedding(frame, bbox)` method. Used to generate embeddings
            for identity resolution across shots.
        model_path (str): Path to the YOLOv5 face model weights.
        config_path (str): Path to the YOLOv5 model config.
        iou_thresh (float): IoU threshold for track matching.
        embedding_thresh (float): Embedding distance threshold for merging.

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
    next_vchunk_id = 0

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

            observations = []
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                embedding = embedder.get_embedding(frame, box) if embedder else None

                obs = FaceObservation(
                    frame_idx=frame_idx,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    embedding=embedding
                )
                observations.append(obs)
                
            aggregator.add_frame_observations(frame_idx, observations)

        aggregator.finalize_tracks()

        # ✅ Assign vchunk IDs using match logic within chunk
        next_vchunk_id = aggregator.resolve_vchunk_ids(
            vchunk_id_counter=next_vchunk_id,
            embedding_threshold=embedding_thresh
        )

        all_tracks.extend(aggregator.tracks)

    cap.release()
    return all_tracks
