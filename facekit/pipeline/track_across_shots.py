from pathlib import Path
import json
from typing import List

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceTrack
from facekit.embedding.embedder import FaceEmbedder
from facekit.detection.face_detector import FaceDetector
from facekit.utils.video_reader import VideoReader
from facekit.embedding.alignment import align_face_for_arcface


def track_across_shots(
    video_path: str,
    shot_json_path: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    iou_thresh: float = 0.5,
    embedding_thresh: float = 0.7,
    detect_interval: int = 10,
    embedding_batch_size_max: int = 32
) -> List[FaceTrack]:
    """
    Track faces across shots using YOLO detection, hybrid tracking, and batch embeddings.

    Args:
        video_path (str): Path to input video file.
        shot_json_path (str): Path to shot segmentation JSON.
        detector (FaceDetector): Wrapper for YOLO-based detector.
        embedder (FaceEmbedder): Wrapper for ArcFace embedding model.
        iou_thresh (float): IoU threshold for merging detections.
        embedding_thresh (float): Embedding similarity threshold for merging.
        detect_interval (int): Frequency of running detection (vs tracking).
        embedding_batch_size_max (int): Max faces per embedding batch.

    Returns:
        List[FaceTrack]: All face tracks across shots.
    """
    video_path = Path(video_path)
    shot_json_path = Path(shot_json_path)

    # Load shot boundaries
    with open(shot_json_path) as f:
        shot_data = json.load(f)
    shots = shot_data["shots"]

    reader = VideoReader(str(video_path))
    all_tracks = []

    for shot in shots:
        shot_number = shot["shot_number"]
        first, last = shot["first_frame"], shot["last_frame"]

        aggregator = ShotFaceTrackAggregator(
            shot_number=shot_number,
            iou_threshold=iou_thresh,
            embedding_threshold=embedding_thresh
        )

        frames = reader.get_frames(first, last)
        track_crops = {}  # track_id -> list of aligned face crops for embeddings

        for i, frame in enumerate(frames):
            frame_idx = first + i
            should_detect = (i % detect_interval == 0) or (len(aggregator.tracks) == 0)
            observations = []

            if should_detect:
                detections = detector.detect_faces_in_frame(frame)
                if detections:
                    boxes, landmarks, confidences = detections
                    for box, lm in zip(boxes, landmarks):
                        aligned_face = align_face_for_arcface(frame, lm)
                        observations.append((box, lm, aligned_face))

            # Add observations to aggregator
            aggregator.add_frame_observations(frame_idx, observations)

            # Collect crops for embedding extraction
            for track in aggregator.tracks:
                if track.is_active and track.track_id not in track_crops:
                    track_crops[track.track_id] = []
                if track.is_active and track.track_id in track_crops:
                    # Use last aligned face
                    if observations:
                        track_crops[track.track_id].append(observations[-1][2])

        for track_id, crops in track_crops.items():
            if crops:
                embeddings = embedder.get_embedding_batch(crops, batch_size=embedding_batch_size_max)
                aggregator.attach_embeddings(track_id, embeddings)

        aggregator.finalize_tracks()

        all_tracks.extend(aggregator.tracks)

    return all_tracks
