from pathlib import Path
import json
from typing import List, Tuple, Dict, Union
import numpy as np
from contextlib import ExitStack

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.embedding.embedder import FaceEmbedder
from facekit.detection.face_detector import FaceDetector
from facekit.tracking.face_tracker import FaceTracker
from facekit.embedding.alignment import align_face_for_arcface
from facekit.io.frame_provider import FrameProvider, ReaderCoordinator
from facekit.tracking.tracker_validator import TrackerValidator, ValidatorParams

def track_across_segments(
    frame_source: Union[str, Path, FrameProvider],
    shot_json_path: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    iou_thresh: float = 0.5,
    embedding_thresh: float = 0.7,
    detect_interval: int = 10,
    embedding_batch_size_max: int = 32,
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

    with open(shot_json_path) as f:
        shot_data = json.load(f)
    shots = shot_data["shots"]

    with ExitStack() as stack:
        if isinstance(frame_source, (str, Path)):
            frame_provider = stack.enter_context(ReaderCoordinator(str(frame_source)))
        elif isinstance(frame_source, FrameProvider):
            frame_provider = frame_source
        else:
            raise TypeError(
                f"frame_source must be a str/Path (video path) or a FrameProvider; "
                f"got {type(frame_source)!r}"
        )

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

            shot_frames: List[np.ndarray] = []
            validator = TrackerValidator(frames=shot_frames, 
                                first_frame_idx=first, 
                                params=ValidatorParams(iou_thresh=iou_thresh))

            frame_provider.reset_to_frame(first)

            for i, frame_idx in enumerate(range(first, last + 1)):
                frame = frame_provider.next()
                if frame is None:
                    break

                # keep the validator’s frame buffer in lock-step with absolute indices
                # (local idx == frame_idx - first)
                shot_frames.append(frame)
   
                observations: List[FaceObservation] = []

                # Determine if this is a scheduled detection frame
                need_detect = (i % detect_interval == 0) or not tracker_active or (len(aggregator.tracks) == 0)

                # we may need these after aggregator assigns IDs
                bootstrap_boxes_xyxy: List[Tuple[int,int,int,int]] = []

                if not need_detect:
                    # Try tracking all existing tracks
                    tracked_boxes: Dict[int, Tuple[float, float, float, float]] = face_tracker.update_trackers(frame)
                    basic_fail = (not tracked_boxes) or any(b is None for b in tracked_boxes.values())

                    if (not basic_fail) and validator.validate(tracked_boxes, frame_idx, verbose=False):
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
                                source='tracking'
                            ))
                    else:
                        # validator rejected or tracker failed → close & force detection reboot
                        aggregator.finalize_tracks()
                        tracker_active = False
                        need_detect = True                       

                if need_detect:
                # Run detection if needed    
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
                    else:
                        # no faces this frame; close shot-local tracks
                        aggregator.finalize_tracks()

                # Add current frame observations to aggregator
                aggregator.update_tracks_with_frame(frame_idx, observations)

                # If we successfully detected faces this frame, init tracker with aggregator IDs and seed validator
                if need_detect:
                    if bootstrap_boxes_xyxy:
                        det_obs = [obs for obs in observations 
                                   if obs.source == 'detection' and obs.track_id is not None]
                        if det_obs:
                            boxes_xywh = [
                                (b[0], b[1], b[2] - b[0], b[3] - b[1]) 
                                for b in (obs.bbox for obs in det_obs)
                            ]
                            track_ids = [obs.track_id for obs in det_obs]
                            face_tracker.init_trackers(frame, boxes_xywh, track_ids)

                            # --- seed validator baseline with the just-initialized tracker state ---
                            boxes_map = {tid: b for tid, b in zip(track_ids, boxes_xywh)}
                            validator.set_baseline(boxes_map, frame_idx)
                            
                            tracker_active = True
                        else:
                            tracker_active = False
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
