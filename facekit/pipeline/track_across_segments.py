from pathlib import Path
import json
from typing import List, Tuple, Dict
import numpy as np
import cv2
from dataclasses import dataclass

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.embedding.embedder import FaceEmbedder
from facekit.detection.face_detector import FaceDetector
from facekit.tracking.face_tracker import FaceTracker
from facekit.utils.video_reader import VideoReader
from facekit.embedding.alignment import align_face_for_arcface
from facekit.tracking.tracker_validator import TrackerValidator, ValidatorParams

def track_across_segments(
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
        first_frame = int(shot["first_frame"])
        last_frame = int(shot["last_frame"])

        aggregator = ShotFaceTrackAggregator(
            shot_number=shot_number,
            iou_threshold=iou_thresh,
            embedding_threshold=embedding_thresh,
        )

        face_tracker = FaceTracker(tracker_type="CSRT")
        tracker_active = False

        # --- FRAME LOADING: prefer sequential decode across adjacent shots ---
        nframes = last_frame - first_frame + 1
        if nframes <= 0:
            continue

        use_seq = False
        if shot_number > 1:
            prev_last_frame = int(shots[shot_number - 2]["last_frame"])
            use_seq = (prev_last_frame + 1 == first_frame)

        if use_seq:
            frames = reader.read_next_n(nframes)
        else:
            frames = reader.get_frames(first_frame, last_frame)

        validator = TrackerValidator(frames=frames, 
                                     first_frame_idx=first_frame, 
                                     params=ValidatorParams(iou_thresh=iou_thresh))

        detection_enabled = detector is not None
        
        for i, frame in enumerate(frames):

            frame_idx = first_frame + i

            observations: List[FaceObservation] = []

            # Determine if this is a scheduled detection frame
            scheduled_detect = (
                detection_enabled and ((i % detect_interval == 0) or (len(aggregator.tracks) == 0))
            )
            
            need_detect = False  # ensure it's defined
            bootstrap_boxes_xyxy: List[Tuple[int, int, int, int]] = []

            if scheduled_detect or not tracker_active:
                need_detect = detection_enabled  # only detect when we actually have a detector
                tracker_active = False  # reset tracker state when detection is scheduled
            else:
                tracked_boxes = face_tracker.update_trackers(frame)  # {track_id: (x,y,w,h) or None}

                # Basic failure if nothing returned or any None
                basic_fail = (not tracked_boxes) or any(b is None for b in tracked_boxes.values())

                if not basic_fail and validator.validate(tracked_boxes, frame_idx, verbose = True):
                    for track_id, tb in tracked_boxes.items():
                        if tb is None:
                            continue
                        x, y, w, h = tb
                        bbox = (int(x), int(y), int(x + w), int(y + h))
                        observations.append(FaceObservation(
                            frame_idx=frame_idx,
                            track_id=track_id,
                            bbox=bbox,
                            embedding=None,
                            confidence=None,
                            aligned_face=None,  # will be estimated by Optical Flow pass later
                            source='tracking'
                        ))
                else:
                    for t in aggregator.tracks:
                        if not t.is_closed():
                            t.mark_closed()
                    tracker_active = False
                    need_detect = detection_enabled

            # Run detection if needed and available
            if need_detect and detection_enabled:

                detections = detector.detect_faces_in_frame(frame)
                
                if detections:
                    boxes, landmark_lists, confidences = detections

                    for box, landmarks in zip(boxes, landmark_lists):
                        bbox = tuple(int(v) for v in box[:4])  # detector returns XYXY
                        aligned_face = align_face_for_arcface(frame, landmarks, frame_idx, source="detect")
                        observations.append(FaceObservation(
                            frame_idx=frame_idx,
                            bbox=bbox,
                            embedding=None,
                            confidence=None,
                            aligned_face=aligned_face,
                            source='detection'
                        ))
                        bootstrap_boxes_xyxy.append(bbox)

            # Add current frame observations to aggregator
            aggregator.update_tracks_with_frame(frame_idx, observations)

            # Initialize (or re-initialize) the tracker from detections that landed this frame.
            if need_detect:
                boxes_xywh, track_ids = [], []
                for t in aggregator.tracks:
                    if t.is_closed() or not t.observations:
                        continue
                    last_obs = t.observations[-1]
                    # only use detections that belong to *this* frame
                    if last_obs.frame_idx != frame_idx or last_obs.source != "detection":
                        continue
                    x0, y0, x1, y1 = last_obs.bbox
                    boxes_xywh.append((x0, y0, x1 - x0, y1 - y0))
                    track_ids.append(t.track_id)

                if boxes_xywh:
                    face_tracker.init_trackers(frame, boxes_xywh, track_ids)
                    tracker_active = True
                    seed_map = {tid: box for tid, box in zip(track_ids, boxes_xywh)}
                    validator.set_baseline(seed_map, frame_idx)
                else:
                    tracker_active = False

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

        # Assign segment_id per shot
        _ = aggregator.resolve_segment_ids(segment_id_counter=0, embedding_threshold=embedding_thresh)
        all_tracks.extend(aggregator.tracks)

    return all_tracks
