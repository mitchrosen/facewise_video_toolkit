import json
import cv2
from pathlib import Path
from typing import List

from facekit.detection.detection_helpers import detect_faces_and_embeddings
from facekit.tracking.face_structures import FaceTrack
from facekit.tracking.aggregator import ShotFaceTrackAggregator

def track_across_shots(
    video_path: str,
    shot_json_path: str,
    embedder=None,
    iou_thresh: float = 0.5,
    embedding_thresh: float = 0.7,
) -> List[FaceTrack]:
    """
    Track faces across the entire video, shot by shot, returning intra-shot tracks.
    
    Args:
        video_path (str): Path to the input video.
        shot_json_path (str): Path to the shot_features.json file.
        embedder (Optional[object]): Optional face embedding extractor with a
            `get_embedding(frame, bbox)` method.
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

    # Prepare video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    all_tracks = []

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

            # ✅ Use detect_faces_and_embeddings (with expanded crops + embeddings)
            observations = detect_faces_and_embeddings(frame, frame_idx, embedder=embedder)
            aggregator.update_tracks_with_frame(frame_idx, observations)

        # After processing all frames in the shot
        aggregator.finalize_tracks()

        # Assign vchunk IDs using embedding logic
        aggregator.resolve_vchunk_ids(
            vchunk_id_counter=0,
            embedding_threshold=embedding_thresh
        )

        # Debug print for each track
        print("[DEBUG] After tracking shot {}: {} tracks".format(shot_number, len(aggregator.tracks)))
        for t in aggregator.tracks:
            print(f"[DEBUG] Track {t.track_id}@Shot{t.shot_id} → vchunk_id={t.vchunk_id}, embeddings={len(t.embeddings)}")

        all_tracks.extend(aggregator.tracks)

    cap.release()
    return all_tracks
