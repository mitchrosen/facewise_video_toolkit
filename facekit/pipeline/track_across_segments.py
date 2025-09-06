"""
Track faces across shot segments with periodic detection and in-between tracking.

We keep your original control flow and debug style. The only functional change is:
- Initialize the FaceTracker **after** the aggregator has assigned IDs to the
  detection observations for the current frame. This guarantees the tracker
  uses the same IDs as the aggregator.
"""

from pathlib import Path
import json
from typing import List, Tuple, Dict
import numpy as np
import cv2

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.embedding.embedder import FaceEmbedder
from facekit.detection.face_detector import FaceDetector
from facekit.tracking.face_tracker import FaceTracker
from facekit.utils.video_reader import VideoReader
from facekit.embedding.alignment import align_face_for_arcface

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
            # DEBUG
            if frame_idx in {14863,14864}:
                print("[DEBUG] in track_across_segments(), hit {frame_idx}")

                import os
                os.makedirs("debug_output", exist_ok=True)
                print(f"[DEBUG] Frame {frame_idx}: Debug images to be saved to debug_output")

                # Save raw frame
                cv2.imwrite(f"debug_output/frame_{frame_idx}_raw.jpg", frame)

            observations: List[FaceObservation] = []

            # Determine if this is a scheduled detection frame
            scheduled_detect = (i % detect_interval == 0) or (len(aggregator.tracks) == 0)
            need_detect = False  # ensure it's defined

            # DEBUG
            if frame_idx in {14863, 14864}:
                print(f"[DEBUG] in track_across_segments(), hit {frame_idx}, sheduled_detect = {scheduled_detect}, tracker_active = {tracker_active}")

            # we may need these after aggregator assigns IDs
            bootstrap_boxes_xyxy: List[Tuple[int,int,int,int]] = []

            if scheduled_detect or not tracker_active:
                need_detect = True
                tracker_active = False  # reset tracker state when detection is scheduled
            else:
                # Try tracking all existing tracks
                tracked_boxes: Dict[int, Tuple[float, float, float, float]] = face_tracker.update_trackers(frame)

                # DEBUG
                if frame_idx in {14863, 14864}:
                    print(f"[DEBUG] Frame {frame_idx}: {len(face_tracker.trackers)} surviving trackers")
                    print(f"[DEBUG] Frame {frame_idx}: {len(tracked_boxes)} boxes returned")
                    for j, (tid, tb) in enumerate(tracked_boxes.items()):
                        print(f"    box {j}: id={tid}, {tb}")

                any_tracker_fails = (not tracked_boxes) or any(b is None for b in tracked_boxes.values())

                # DEBUG
                if frame_idx in {14863, 14864}:
                    print(f"[DEBUG] in track_across_segments(), hit {frame_idx}, any_tracker_fails = {any_tracker_fails}")
                    print(f"[DEBUG] update_trackers() @ {frame_idx}: received {len(tracked_boxes)} boxes:")
                    for j, (tid, b) in enumerate(tracked_boxes.items()):
                        print(f"    box {j}: id={tid}, {b}")

                if any_tracker_fails:
                    # Close all current tracks, and fall back to detection-only this frame
                    for t in aggregator.tracks:
                        if not t.is_closed():
                            t.mark_closed()
                    tracker_active = False
                    need_detect = True
                else:
                    # Track-only observations will lack aligned_face until optical flow landmarks are added
                    for track_id, tb in tracked_boxes.items():

                        # DEBUG
                        if frame_idx in {14863, 14864}:
                            print(f"[DEBUG] in track_across_segments(), hit {frame_idx}, looping on tracked_boxes, tb = {tb}")

                        if tb is None:
                            continue
                        x, y, w, h = tb
                        bbox = (int(x), int(y), int(x + w), int(y + h))
                        fo = FaceObservation(
                            frame_idx=frame_idx,
                            track_id = track_id,
                            bbox=bbox,
                            embedding=None,
                            confidence=None,
                            aligned_face=None,  # To be estimated via optical flow in future pass
                            source='tracking'
                        )

                        # DEBUG
                        if frame_idx in {14863, 14864}:
                            print(f"[DEBUG] track_across_segments(): frame {frame_idx}, appended tracking observation: {fo}")
                                            
                        observations.append(fo)

            # Run detection if needed
            if need_detect:
 
                # DEBUG
                if frame_idx in {14863, 14864}:
                    print(f"[DEBUG] in track_across_segments(), hit {frame_idx}, detect needed")

                detections = detector.detect_faces_in_frame(frame)

                # DEBUG
                if frame_idx in {14863, 14864}:
                    print(f"[DEBUG] in track_across_segments(), hit {frame_idx}, detections = {detections}")

                if detections:
                    boxes, landmark_lists, confidences = detections

                    # DEBUG
                    if frame_idx in {14863, 14864}:
                        print(f"[DEBUG] in track_across_segments(), hit {frame_idx}, about to loop on detections")

                    for box, landmarks in zip(boxes, landmark_lists):

                        # DEBUG
                        if frame_idx in {14863, 14864}:
                            print(f"[DEBUG] in track_across_segments(), hit {frame_idx}, about to call align_face_for_arcface()")

                        bbox = tuple(int(v) for v in box[:4])  # detector returns XYXY
                        aligned_face = align_face_for_arcface(frame, landmarks, frame_idx, source="detect")

                        if frame_idx in {14863, 14864}:
                            print(f"[DEBUG] in track_across_segments(), hit {frame_idx}, after call align_face_for_arcface(), aligned_face = {aligned_face}")

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

            # If we just detected, aggregator has now assigned IDs to those detection obs.
            # Initialize the tracker *now* so it uses the aggregator's IDs.
            if need_detect and bootstrap_boxes_xyxy:
                det_obs = [obs for obs in observations if obs.source == 'detection' and obs.track_id is not None]
                if det_obs:
                    boxes_xywh = [
                        (b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in (obs.bbox for obs in det_obs)
                    ]
                    track_ids = [obs.track_id for obs in det_obs]
                    face_tracker.init_trackers(frame, boxes_xywh, track_ids)
                    tracker_active = True
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
