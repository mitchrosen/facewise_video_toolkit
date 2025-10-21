from pathlib import Path
import json
from typing import List, Tuple, Dict, Union
import numpy as np
from contextlib import ExitStack
from bisect import bisect_left
import logging

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.embedding.embedder import FaceEmbedder
from facekit.detection.face_detector import FaceDetector
from facekit.tracking.face_tracker import FaceTracker
from facekit.embedding.alignment import align_face_for_arcface
from facekit.io.frame_provider import FrameProvider, ReaderCoordinator
from facekit.tracking.tracker_validator import TrackerValidator, ValidatorParams
from facekit.pipeline.checkpoint import TrackingCheckpoint
from facekit.common.obs_consts import Source


def _shot_idx_by_shotnum(shots, target_shot_num: int) -> int:
    """
    Return the index of the shot whose `shot_number` == target shot number.
    If `shot_number` isn't present, returns the insertion point (may be len(shots)).
    Assumes shots are sorted by ascending consecutive `shot_number`.
    """
    # Assumes ascending, consecutive shot_number
    shot_nums = [s["shot_number"] for s in shots]
    return bisect_left(shot_nums, target_shot_num)

def _shot_idx_by_abs_frame(shots, abs_frame_idx: int) -> int:
    """
    Return the index of the first shot whose `last_frame` >= abs_frame_idx.
    If abs_frame_idx is beyond all shots, returns len(shots).
    If `frame_idx` falls before the shot's `first_frame`, the first frame index (0) is returned.
    Assumes shots are sorted, non-overlapping, ascending by frame range.
    """
    # First shot whose last_frame >= resume_abs_frame
    last_frames = [s["last_frame"] for s in shots]
    return bisect_left(last_frames, abs_frame_idx)

def track_across_segments(
    frame_source: Union[str, Path, FrameProvider],
    shot_json_path: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    iou_thresh: float = 0.5,
    embedding_thresh: float = 0.7,
    detect_interval: int = 10,
    embedding_batch_size_max: int = 32,
    *,
    checkpoint: TrackingCheckpoint | None = None,
) -> List[FaceTrack]:
    """
    Track faces across precomputed shot segments, attach embeddings, and return per-shot tracks.

    This function is the main, **shot-aware** tracker. For each shot [first_frame..last_frame]:
      1) **Streams frames sequentially** via a `FrameProvider` (`ReaderCoordinator` by default).
      2) Performs **periodic face detection** every `detect_interval` frames (and always on the first frame
         of the shot or after tracker resets). Detection observations are handed to the per-shot
         `ShotFaceTrackAggregator`, which assigns/continues **shot-local segment IDs**.
      3) Between detection frames, it uses a `FaceTracker` (CSRT by default) to propagate existing boxes
         and emits **tracking** observations for currently active tracks.
      4) If any tracker update fails, all open tracks are marked closed, and the pipeline **falls back to detection**
         on that frame to reboot tracking cleanly.
      5) At shot end, batches all aligned detection crops through the `embedder` and attaches 512-D embeddings
         to each `FaceTrack`. The aggregator is then finalized, and per-shot **segment IDs** are resolved.

    Parameters
    ----------
    frame_source : Union[str, Path, FrameProvider]
        Either:
          - a filesystem path to a video file (str or Path), in which case a `ReaderCoordinator`
            will be constructed and managed internally, OR
          - a preconstructed `FrameProvider` instance, in which case this function will
            use it but **not close it** (caller retains ownership).
    shot_json_path : str
        Path to a JSON produced by shot segmentation, containing
        `{"shots": [{"shot_number": int, "first_frame": int, "last_frame": int}, ...]}`.
    detector : FaceDetector
        Face detector used on scheduled detection frames. Its `detect_faces_in_frame(frame)` is expected
        to return `(boxes_xyxy, landmarks, confidences)` or `None` when no faces are present.
    embedder : FaceEmbedder
        Embeds aligned detection crops at the end of each shot; must return an `np.ndarray` of shape (K, 512), float32.
    iou_thresh : float, default 0.5
        IOU threshold inside the per-shot `ShotFaceTrackAggregator` for associating detections to tracks.
    embedding_thresh : float, default 0.7
        Cosine similarity threshold used **within the shot** when resolving/merging segment IDs.
        (Global identity resolution happens later, outside this function.)
    detect_interval : int, default 10
        Run the detector every `detect_interval` frames (and on the first frame of a shot or after tracker reset).
    embedding_batch_size_max : int, default 32
        Maximum batch size for `embedder.get_embedding_batch`.
    checkpoint: provides callbacks for checkpointing

    Returns
    -------
    List[FaceTrack]
        All finalized `FaceTrack` objects across all shots, each with:
          - `shot_id`, `track_id` (per-shot), `segment_id` (shot-local face ID),
          - `observations` (detection + tracking),
          - attached 512-D embeddings for detection observations (if any crops existed).

    Notes
    -----
    - **Detector scheduling:** Frames with detections produce aligned crops; tracking-only frames intentionally **do not**
      create embeddings to keep the hot path fast. (A later second-pass can compute extra embeddings if needed.)
    - **Lifecycle:** When `frame_provider` is supplied, this function does not close it. When omitted, the internally
      constructed provider is closed automatically via `ExitStack()`.
    - **Performance knobs:** `detect_interval` trades accuracy for speed; increasing it reduces detector calls.
      `embedding_batch_size_max` controls memory/throughput on the embedder.
    """

    shot_json_path = Path(shot_json_path)
    with open(shot_json_path, "r") as f:
        _shot_data = json.load(f)
    shots = _shot_data["shots"]

    with ExitStack() as stack:
        if isinstance(frame_source, (str, Path)):
            frame_provider = stack.enter_context(ReaderCoordinator(str(frame_source)))
        else:
            # Accept real FrameProvider or duck-typed providers (fps/size/total_frames/get)
            _ok = isinstance(frame_source, FrameProvider) or all(
                hasattr(frame_source, m) for m in ("fps", "size", "total_frames", "get")
            )
            if not _ok:
                raise TypeError(
                    f"frame_source must be a str/Path (video path) or a FrameProvider-like object; "
                    f"got {type(frame_source)!r}"
                )
            frame_provider = frame_source

        all_tracks: List[FaceTrack] = []

        resume = False
        if checkpoint:
            resume_anchor = checkpoint.get_resume_anchor()
            if resume_anchor:
                resume_abs_frame, resume_shot, resume_shot_first_frame = resume_anchor        
                start_shot_idx = _shot_idx_by_abs_frame(shots, resume_abs_frame)
                shots = shots[start_shot_idx:]
                logging.info(
                    "resume: anchor frame=%d shot=%d (shot_first=%s) -> starting at shot index %d",
                    resume_abs_frame, resume_shot, resume_shot_first_frame, start_shot_idx
                )
                resume = True

        for shot in shots:
            shot_number = shot["shot_number"]
            first, last = shot["first_frame"], shot["last_frame"]

            # If resuming and this is the anchor shot, 
            # it is possible that the checkpoint was captured in the middle of the shot - start from that frame
            if resume and (shot_number == resume_shot):
                start_at = max(first, resume_abs_frame) 
            else:
                start_at = first

            if start_at > last:
                continue  # nothing to do in this shot

            aggregator = ShotFaceTrackAggregator(
                shot_number=shot_number,
                iou_threshold=iou_thresh,
                embedding_threshold=embedding_thresh,
            )

            face_tracker = FaceTracker(tracker_type="CSRT")
            tracker_active = False

            shot_frames: List[np.ndarray] = []
            validator = TrackerValidator(
                frames=shot_frames, 
                first_frame_idx=start_at, 
                params=ValidatorParams(iou_thresh=iou_thresh))

            frame_provider.reset_to_frame(start_at)

            for frame_idx in range(start_at, last + 1):
                frame = frame_provider.next()
                if frame is None:
                    break

                shot_frames.append(frame)
                observations: List[FaceObservation] = []

                detect_interval_hit = (frame_idx % detect_interval == 0)

                no_tracker   = (not tracker_active)
                open_tracks = [t for t in aggregator.tracks if not t.is_closed()]
                no_open_tracks = (len(open_tracks) == 0)
   
                # Determine if this is a scheduled detection frame
                need_detect = detect_interval_hit or no_tracker or no_open_tracks

                logging.debug(
                    "frame=%d need_detect=%s [det_interval_hit=%s no_tracker=%s no_open_tracks=%s] open_tracks=%d total_tracks=%d",
                    frame_idx, need_detect, detect_interval_hit, no_tracker, no_open_tracks, len(open_tracks),
                    len(aggregator.tracks),
                )

                if not need_detect:
                    # Try tracking all existing tracks
                    tracked_boxes: Dict[int, Tuple[float, float, float, float]] = face_tracker.update_trackers(frame)
                    basic_fail = (not tracked_boxes) or any(b is None for b in tracked_boxes.values())

                    logging.debug("TRACK: frame=%d (tracker_active=%s, basic_fail=%s)", frame_idx, tracker_active, basic_fail)

                    if (not basic_fail) and validator.validate(tracked_boxes, frame_idx, verbose=True):
                        # Track-only observations will lack aligned_face until optical flow landmarks are added
                        for track_id, tb in tracked_boxes.items():
                            if tb is None:
                                continue
                            x, y, w, h = tb
                            bbox = (int(x), int(y), int(x + w), int(y + h))
                            observations.append(FaceObservation(
                                frame_idx=frame_idx,
                                track_id = track_id,
                                bbox=bbox,
                                embedding=None,
                                confidence=None,
                                aligned_face=None,
                                source=Source.TRACKED
                            ))
                    else:
                        # validator rejected or tracker failed → close & force detection reboot
                        logging.debug("TRACK - validator reject or tracker fail: frame=%d (basic_fail=%s)", frame_idx, basic_fail)
                        aggregator.finalize_tracks()
                        tracker_active = False
                        need_detect = True                       

                if need_detect:  # Run detection if needed
                    logging.info(
                        "shot=%d first=%d frame=%d (mod=%d) tracker_active=%s, detect_interval=%d",
                        shot_number, first, frame_idx, (frame_idx % detect_interval), tracker_active, detect_interval
                    )
                    if checkpoint:
                        checkpoint.checkpoint_now(
                            frame_idx=frame_idx, 
                            shot_number=shot_number,
                            shot_first_frame=first)
                    
                    detections = detector.detect_faces_in_frame(frame)
                    
                    if detections:
                        boxes, landmark_lists, confidences = detections
                        for box, landmarks, confidence in zip(boxes, landmark_lists, confidences):
                            bbox = tuple(int(v) for v in box[:4])  # detector returns XYXY
                            aligned_face = align_face_for_arcface(frame, landmarks, frame_idx, source="detect")
                            observations.append(FaceObservation(
                                frame_idx=frame_idx,
                                bbox=bbox,
                                embedding=None,
                                confidence=float(confidence),
                                aligned_face=aligned_face,
                                source=Source.DETECTED
                            ))
                    else:
                        # no faces this frame; close shot-local tracks
                        aggregator.finalize_tracks()
                    

                # Add current frame observations to aggregator
                created_count = aggregator.update_tracks_with_frame(frame_idx, observations)

                agg_det = aggregator.observations_at(frame_idx, source=Source.DETECTED, require_track_id=True)
                agg_trk = aggregator.observations_at(frame_idx, source=Source.TRACKED,  require_track_id=True)
                logging.debug(
                    "POST-ASSIGN: frame=%d det_assigned=%d trk_assigned=%d created=%d open_tracks=%d",
                    frame_idx, len(agg_det), len(agg_trk), created_count,
                    sum(1 for t in aggregator.tracks if not t.is_closed()),
                )

                # Checkpoint observations
                if checkpoint:
                    frame_obs = aggregator.observations_at(frame_idx, require_track_id=True)
                    if frame_obs:
                        checkpoint.add_observations(shot_number, frame_idx, frame_obs)
                    if created_count:
                        checkpoint.on_new_tracks(created_count)

                # If faces detected, init tracker with aggregator IDs and seed validator
                if need_detect:
                    det_obs = aggregator.observations_at(frame_idx, source=Source.DETECTED, require_track_id=True)
                    if det_obs:
                        boxes_xywh = [
                            (b[0], b[1], b[2] - b[0], b[3] - b[1]) 
                            for b in (obs.bbox for obs in det_obs)
                        ]
                        track_ids = [obs.track_id for obs in det_obs]
                        face_tracker.init_trackers(frame, boxes_xywh, track_ids)

                        # --- seed validator baseline with the just-initialized tracker state ---
                        boxes_map = dict(zip(track_ids, boxes_xywh))
                        validator.set_baseline(boxes_map, frame_idx)
            
                        tracker_active = True
                    else:
                        tracker_active = False

                if checkpoint:
                    checkpoint.on_frame(frame_idx)

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

                if checkpoint and embs.size:
                    last_idx = max(o.frame_idx for o in track.observations if o.aligned_face is not None)
                    checkpoint.add_embeddings(shot_number, track.track_id, last_idx, embs)

            aggregator.finalize_tracks()

            # Assign segment_id per shot
            _ = aggregator.resolve_segment_ids(segment_id_counter=0, embedding_threshold=embedding_thresh)
            all_tracks.extend(aggregator.tracks)

            if checkpoint:
                checkpoint.on_shot_done()

    return all_tracks
