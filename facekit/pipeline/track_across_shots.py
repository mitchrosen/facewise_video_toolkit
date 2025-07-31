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
from facekit.utils.geometry import compute_iou

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
    """
    For each shot:
      - run detection on a cadence, track by IoU between detects,
      - store aligned faces in observations,
      - batch-compute embeddings per track, attach them,
      - resolve per-shot persistent IDs (vchunk_id, counter resets per shot).

    Global IDs are resolved by the caller after this returns.
    """
    video_path = Path(video_path)
    shot_json_path = Path(shot_json_path)

    def _xywh_to_xyxy(box_xywh):
        x, y, w, h = box_xywh
        return (int(x), int(y), int(x + w), int(y + h))

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

        # Per‑shot tracker instance
        face_tracker = FaceTracker(tracker_type="CSRT")
        tracker_active = False

        frames = reader.get_frames(first, last)

        for i, frame in enumerate(frames):
            frame_idx = first + i

            scheduled_detect = (i % detect_interval == 0) or (len(aggregator.tracks) == 0)
            observations: List[Tuple[Tuple[int,int,int,int], List[Tuple[int,int]], np.ndarray]] = []

            # Track first, if active
            tracker_active = tracker_active  # keep prior state
            tracked_boxes_xyxy = []
            if tracker_active:
                got_any = False
                tracked_boxes_xywh = face_tracker.update_trackers(frame)
                for tb in tracked_boxes_xywh:
                    if tb is None:
                        continue
                    x, y, w, h = tb
                    bbox_xyxy = (int(x), int(y), int(x + w), int(y + h))
                    tracked_boxes_xyxy.append(bbox_xyxy)
                    # provisional obs from tracker (no landmarks/aligned_face)
                    observations.append((bbox_xyxy, [], None))
                    got_any = True
                tracker_active = got_any

            # Decide if detection is needed
            need_detect = scheduled_detect or not tracker_active

            if not need_detect and tracked_boxes_xyxy:
                any_unmatched = False
                open_tracks = [t for t in aggregator.tracks if not t.is_closed()]
                for tbx in tracked_boxes_xyxy:
                    if not open_tracks:  # nothing to match against → treat as unmatched
                        any_unmatched = True
                        break
                    # if tbx fails IoU vs ALL open tracks' last bbox, mark unmatched
                    if all(
                        (t.get_last_bbox() is None) or (compute_iou(t.get_last_bbox(), tbx) < iou_thresh)
                        for t in open_tracks
                    ):
                        any_unmatched = True
                        break
                if any_unmatched:
                    need_detect = True

            boxes_xyxy = []  # will hold detection boxes for dup filtering & tracker (re)init

            # Detect, if needed
            if need_detect and detector is not None:
                detections = detector.detect_faces_in_frame(frame)
                det_obs = []
                if detections:
                    boxes, landmark_lists, confidences = detections
                    for box, landmarks in zip(boxes, landmark_lists):
                        bbox = tuple(int(v) for v in box[:4])
                        boxes_xyxy.append(bbox)
                        aligned_face = align_face_for_arcface(frame, landmarks)
                        det_obs.append((bbox, landmarks, aligned_face))

                # De‑dup: prefer detections over tracked when they overlap strongly
                if det_obs:
                    if observations:  # we had tracker outputs
                        kept_tracked = []
                        for trk_bbox, _, _ in observations:
                            if all(compute_iou(trk_bbox, dbx) < iou_thresh for dbx in boxes_xyxy):
                                kept_tracked.append((trk_bbox, [], None))
                        observations = kept_tracked + det_obs
                    else:
                        observations = det_obs

                # (Re)initialize tracker from detections (if any)
                if boxes_xyxy:
                    boxes_xywh = [(b[0], b[1], b[2]-b[0], b[3]-b[1]) for b in boxes_xyxy]
                    face_tracker.init_trackers(frame, boxes_xywh)
                    tracker_active = True
                else:
                    tracker_active = False

            # Send frame observations to aggregator
            face_observations = [
                FaceObservation(
                    frame_idx=frame_idx,
                    bbox=bbox,
                    embedding=None,
                    confidence=None,
                    aligned_face=aligned_face,
                )
                for (bbox, landmarks, aligned_face) in observations
            ]

            aggregator.update_tracks_with_frame(frame_idx, face_observations)
      
        # ---- End of shot: batch embeddings per track using stored aligned faces
        for track in aggregator.tracks:
            crops = [obs.aligned_face for obs in track.observations if obs.aligned_face is not None]
            if not crops:
                continue

            embs = embedder.get_embedding_batch(crops, batch_size=embedding_batch_size_max)

            # Explicit validation (fail fast if embedder misbehaves)
            if not isinstance(embs, np.ndarray):
                raise TypeError(f"Embedder must return np.ndarray, got {type(embs)}")
            if embs.ndim != 2 or embs.shape[1] != 512:
                raise ValueError(f"Embedder returned invalid array shape {embs.shape}; expected (K,512)")
            if embs.dtype != np.float32:
                embs = np.asarray(embs, dtype=np.float32, order="C")

            aggregator.attach_embeddings(track.track_id, embs)

        aggregator.finalize_tracks()

        # Per-shot vchunk assignment (counter resets in each shot loop)
        _ = aggregator.resolve_vchunk_ids(vchunk_id_counter=0, embedding_threshold=embedding_thresh)

        all_tracks.extend(aggregator.tracks)

    return all_tracks
