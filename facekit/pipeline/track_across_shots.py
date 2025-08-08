from pathlib import Path
import json
from typing import List, Tuple
import numpy as np

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.embedding.embedder import FaceEmbedder
from facekit.detection.face_detector import FaceDetector
from facekit.tracking.face_tracker import FaceTracker
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
    embedding_batch_size_max: int = 32,
) -> List[FaceTrack]:
    video_path = Path(video_path)
    shot_json_path = Path(shot_json_path)

    with open(shot_json_path) as f:
        shot_data = json.load(f)
    shots = shot_data["shots"]

    reader = VideoReader(str(video_path))
    all_tracks: List[FaceTrack] = []

    for shot in shots:
        shot_number = shot["shot_number"]
        first, last = shot["first_frame"], shot["last_frame"]

        aggregator = ShotFaceTrackAggregator(
            shot_number=shot_number,
            iou_threshold=iou_thresh,
            embedding_threshold=embedding_thresh,
        )

        face_tracker = FaceTracker(tracker_type="CSRT")
        tracker_active = False

        frames = reader.get_frames(first, last)

        for i, frame in enumerate(frames):
            frame_idx = first + i
            observations: List[FaceObservation] = []

            # Determine if this is a scheduled detection frame
            scheduled_detect = (i % detect_interval == 0) or (len(aggregator.tracks) == 0)
            need_detect = False  # ensure it's defined

            if scheduled_detect or not tracker_active:
                need_detect = True
                tracker_active = False  # reset tracker state when detection is scheduled
            else:
                # Try tracking all existing tracks
                tracked_boxes_xywh = face_tracker.update_trackers(frame)
                any_tracker_fails = any(b is None for b in tracked_boxes_xywh)

                if any_tracker_fails:
                    # Close all current tracks, and fall back to detection-only this frame
                    for t in aggregator.tracks:
                        if not t.is_closed():
                            t.mark_closed()
                    tracker_active = False
                    need_detect = True
                else:
                    # Track-only observations will lack aligned_face until optical flow landmarks are added
                    for tb in tracked_boxes_xywh:
                        if tb is None:
                            continue
                        x, y, w, h = tb
                        bbox = (int(x), int(y), int(x + w), int(y + h))
                        observations.append(FaceObservation(
                            frame_idx=frame_idx,
                            bbox=bbox,
                            embedding=None,
                            confidence=None,
                            aligned_face=None  # To be estimated via optical flow in future pass
                        ))

            # Run detection if needed
            if need_detect:
                boxes_xyxy = []
                detections = detector.detect_faces_in_frame(frame)
                if detections:
                    boxes, landmark_lists, confidences = detections
                    for box, landmarks in zip(boxes, landmark_lists):
                        bbox = tuple(int(v) for v in box[:4])
                        aligned_face = align_face_for_arcface(frame, landmarks)
                        observations.append(FaceObservation(
                            frame_idx=frame_idx,
                            bbox=bbox,
                            embedding=None,
                            confidence=None,
                            aligned_face=aligned_face
                        ))
                        boxes_xyxy.append(bbox)

                    if boxes_xyxy:
                        # Tracker gets (re)initialized after detection
                        boxes_xywh = [(b[0], b[1], b[2]-b[0], b[3]-b[1]) for b in boxes_xyxy]
                        face_tracker.init_trackers(frame, boxes_xywh)
                        tracker_active = True
                    else:
                        tracker_active = False

            # Add current frame observations to aggregator
            aggregator.update_tracks_with_frame(frame_idx, observations)

        # End of shot: batch embed tracks with aligned faces
        for track in aggregator.tracks:
            crops = [obs.aligned_face for obs in track.observations if obs.aligned_face is not None]
            if not crops:
                continue
            embs = embedder.get_embedding_batch(crops, batch_size=embedding_batch_size_max)
            if not isinstance(embs, np.ndarray):
                raise TypeError(f"Embedder must return np.ndarray, got {type(embs)}")
            if embs.ndim != 2 or embs.shape[1] != 512:
                raise ValueError(f"Embedder returned invalid array shape {embs.shape}; expected (K,512)")
            if embs.dtype != np.float32:
                embs = np.asarray(embs, dtype=np.float32, order="C")
            aggregator.attach_embeddings(track.track_id, embs)

        aggregator.finalize_tracks()

        # Assign vchunk_id per shot
        _ = aggregator.resolve_vchunk_ids(vchunk_id_counter=0, embedding_threshold=embedding_thresh)
        all_tracks.extend(aggregator.tracks)

    return all_tracks