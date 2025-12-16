import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Union, Callable
import copy
from contextlib import ExitStack
from PIL import Image
import logging

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.tracking.face_tracker import FaceTracker
from facekit.tracking.tracker_validator import TrackerValidator, ValidatorParams
from facekit.embedding.embedder import FaceEmbedder
from facekit.embedding.alignment import align_face_for_arcface
from facekit.detection.face_detector import FaceDetector
from facekit.io.frame_provider import FrameProvider, ReaderCoordinator
from facekit.pipeline.checkpoint import TrackingCheckpoint
from facekit.common.obs_consts import Source
from facekit.pipeline.resume_rehydrate import (
    ResumePlan,
    _build_resume_plan,
    _validate_shots_are_absolute_and_increasing,
)
from facekit.pipeline.checkpoint_io import (
    _checkpoint_root_dir,
    _save_crops_for_frame, 
    do_checkpoint, 
    _checkpoint_observations_and_snapshot,
    _persist_embeddings_for_track,
    _finalize_checkpoint_run
)
from facekit.errors import ResumeSafetyError

logger = logging.getLogger(__name__)

def _init_shot_aggregator(
    *,
    shot_idx: int,
    shot_number: int,
    first: int,
    last: int,
    detect_interval: int,
    resume_plan: ResumePlan,
    iou_thresh: float,
    embedding_thresh: float,
    checkpoint: TrackingCheckpoint | None,
) -> tuple[int, ShotFaceTrackAggregator, int]:
    """
    Initialize a ShotFaceTrackAggregator for a single shot, applying resume rules.

    Responsibilities
    ----------------
    * Decide the starting frame for this shot:
        - First processed shot starts at max(first_frame, anchor_frame).
        - Subsequent shots start at their first_frame.
    * For the anchor-containing shot, seed the aggregator with pre-anchor tracks
      so that labels and embeddings continue seamlessly.
    * Apply track_id seeding and (optional) forced tid reuse at the resume boundary.
    * Compute the per-shot segment-id seed returned as seg_seed.

    Parameters
    ----------
    shot_idx :
        0-based index of this shot within the trimmed shot list.
    shot_number :
        Logical shot identifier from the shot JSON.
    first, last :
        Absolute first and last frame indices for this shot.
    detect_interval :
        Detection interval used for logging the resume fence.
    resume_plan :
        ResumePlan containing anchor frame and per-shot seeds.
    iou_thresh, embedding_thresh :
        Aggregator matching and within-shot identity thresholds.
    checkpoint :
        Optional checkpoint used to hydrate open tracks for seeded shots.

    Returns
    -------
    start_at, aggregator, seg_seed :
        start_at : int
            Absolute frame index at which this shot-loop should start.
        aggregator : ShotFaceTrackAggregator
            Fully initialized aggregator for this shot.
        seg_seed : int
            Initial segment_id_counter seed for this shot (used at resolve_segment_ids()).
    """
    # Decide starting frame
    if shot_idx == 0:
        start_at = max(first, resume_plan.anchor_frame)
    else:
        start_at = first

    if shot_idx == 0:
        logging.info(
            "resume: first_new_frame=%d (shot=[%d..%d]) detect_interval=%d mod=%d",
            start_at,
            first,
            last,
            detect_interval,
            ((start_at - first) % max(1, detect_interval)),
        )
        if start_at > last:
            raise ResumeSafetyError(
                f"empty work-range at resume: start_at={start_at} > shot_last={last} for shot={shot_number}"
            )

    # Seeded tracks only for anchor-containing shot
    seeded_tracks = None
    if (
        shot_idx == 0
        and resume_plan.is_resume
        and resume_plan.prior_tracks_anchor
    ):
        seeded_tracks = [
            copy.deepcopy(t)
            for t in resume_plan.prior_tracks_anchor
            if int(getattr(t, "shot_id", -1)) == int(shot_number)
        ]

    # Build aggregator
    if seeded_tracks:
        aggregator = ShotFaceTrackAggregator(
            shot_number=shot_number,
            iou_threshold=iou_thresh,
            embedding_threshold=embedding_thresh,
            prior_tracks=seeded_tracks,
            resume_abs_frame=start_at,
            next_tid_seed=int(resume_plan.trackid_seed_by_shot.get(int(shot_number), 0)),
        )
        if checkpoint:
            checkpoint.hydrate_open_tracks_into(aggregator)
    else:
        aggregator = ShotFaceTrackAggregator(
            shot_number=shot_number,
            iou_threshold=iou_thresh,
            embedding_threshold=embedding_thresh,
        )

    # Apply forced tid reuse on very first detection frame of anchor shot
    if (
        shot_idx == 0
        and resume_plan.is_resume
        and resume_plan.reuse_tid_for_first_shot is not None
        and int(shot_number) == int(resume_plan.first_processed_shot_number)
    ):
        try:
            aggregator.set_resume_force_tid(int(resume_plan.reuse_tid_for_first_shot))
            logging.info(
                "resume: aggregator will force tid=%d on the next created track in shot=%d",
                int(resume_plan.reuse_tid_for_first_shot),
                int(shot_number),
            )
        except Exception:
            logging.exception("resume: failed to set forced tid on aggregator")

    if seeded_tracks:
        logging.info("RESUME: aggregator seeded with %d tracks", len(aggregator.tracks))
        for track in aggregator.tracks:
            last_bbox = getattr(track, "get_last_bbox", lambda: None)()
            logging.info(
                "RESUME TRACK: tid=%s closed=%s last_frame=%s last_bbox=%s",
                track.track_id,
                track.is_closed() if hasattr(track, "is_closed") else None,
                track.last_frame() if hasattr(track, "last_frame") else None,
                last_bbox,
            )

    seed_tid = int(resume_plan.trackid_seed_by_shot.get(int(shot_number), 0))
    seg_seed = int(
        resume_plan.segment_id_seed_by_shot.get(int(shot_number), 0)
    ) if resume_plan.segment_id_seed_by_shot else 0

    logging.info(
        "shot=%d init: aggregator.next_track_id=%d, seed_tid=%d, seg_seed=%d",
        int(shot_number),
        int(getattr(aggregator, "next_track_id", -1)),
        seed_tid,
        seg_seed,
    )

    # Seed track_id if we didn't create the aggregator with a next_tid_seed
    if not seeded_tracks and seed_tid > 0:
        if hasattr(aggregator, "set_track_id_seed") and callable(
            getattr(aggregator, "set_track_id_seed")
        ):
            aggregator.set_track_id_seed(seed_tid)
        elif hasattr(aggregator, "next_track_id"):
            setattr(aggregator, "next_track_id", seed_tid)
        elif hasattr(aggregator, "_next_track_id"):
            setattr(aggregator, "_next_track_id", seed_tid)

    return start_at, aggregator, seg_seed


def _guard_seek_failure(
    *,
    frame: np.ndarray | None,
    frame_idx: int,
    start_at: int,
    shot_number: int,
    first: int,
    last: int,
    resume_plan: ResumePlan,
) -> np.ndarray | None:
    """
    Handle frame_provider.next() returning None at the start of shot processing.

    If the failure occurs exactly on the anchor frame for a resume run, this is
    considered unsafe and a ResumeSafetyError is raised. Otherwise a warning is
    logged and the caller is allowed to skip the frame.

    Parameters
    ----------
    frame :
        Frame returned by frame_provider.next(), or None.
    frame_idx :
        Absolute index of the frame we attempted to read.
    start_at :
        First frame this shot-loop is supposed to process.
    shot_number :
        Logical shot identifier, for logging.
    first, last :
        Absolute first and last frame indices for the shot.
    resume_plan :
        ResumePlan indicating whether this run is a resume and the anchor frame.

    Returns
    -------
    frame or None :
        The same frame if non-None; otherwise None (caller should skip work).
    """
    if frame is not None:
        return frame

    if resume_plan.is_resume and frame_idx == start_at:
        raise ResumeSafetyError(
            f"frame provider failed to seek to start_at={start_at} for shot={shot_number} "
            f"(first={first}, last={last})."
        )

    logging.warning(
        "frame_provider returned None at frame %d (shot=%d); skipping frame.",
        frame_idx,
        shot_number,
    )
    return None

def _guard_no_rewind(frame_idx: int, resume_plan: ResumePlan) -> None:
    """
    Guardrail to prevent emitting pre-anchor frames after a resume.

    Any attempt to produce observations for a frame strictly before
    resume_plan.anchor_frame is treated as a bug and raises ResumeSafetyError.
    """
    if resume_plan.anchor_frame and frame_idx < resume_plan.anchor_frame:
        raise ResumeSafetyError(
            f"Illegal rewind: got frame_idx={frame_idx} < anchor={resume_plan.anchor_frame}"
        )
    
def _default_embedding_frame_policy(track, obs, frame_idx: int) -> bool:
    """
    Decide whether this observation/frame should be used for embedding.

    For now this always returns True, meaning:
      - if we can materialize an aligned_face or crop for this obs,
        we will include it in the embed batch.

    Later, we can replace or override this with:
      - stride-based sampling (every Nth frame),
      - max-embeddings-per-track caps,
      - quality heuristics, etc.
    """
    return True


def _collect_crops_for_embedding(
    track,
    *,
    checkpoint,
    run_root: Path | None,
    embedding_frame_policy: Callable[[object, object, int], bool] | None = None,
):
    """
    Collect (crops, frame_indices) for a single track, using:

      - obs.aligned_face if present
      - otherwise, a crop loaded from obs.crop_ref (if checkpoint + file exists)

    The optional embedding_frame_policy(track, obs, frame_idx) can be used
    to subsample frames (e.g., stride rules) without changing the call site.
    """
    if embedding_frame_policy is None:
        embedding_frame_policy = _default_embedding_frame_policy

    crops: list[np.ndarray] = []
    frames_for_embed: list[int] = []

    for obs in getattr(track, "observations", []) or []:
        frame_idx = int(getattr(obs, "frame_idx", -1))

        # Let the policy decide if this frame is even eligible
        try:
            if not embedding_frame_policy(track, obs, frame_idx):
                continue
        except Exception:
            # Be conservative; if the policy explodes, skip this obs but keep going.
            logging.exception(
                "embed-policy: error applying embedding_frame_policy for "
                "track_id=%s frame=%s",
                getattr(track, "track_id", None),
                frame_idx,
            )
            continue

        # Case 1: aligned_face already in memory
        if getattr(obs, "aligned_face", None) is not None:
            try:
                crops.append(obs.aligned_face)
                frames_for_embed.append(frame_idx)
            except Exception:
                logging.exception(
                    "embed-collect: failed to use aligned_face for track_id=%s frame=%s",
                    getattr(track, "track_id", None),
                    frame_idx,
                )
            continue

        # Case 2: try to load crop from disk via crop_ref
        cr = getattr(obs, "crop_ref", None)
        if checkpoint and cr and run_root is not None:
            try:
                abs_path = Path(run_root, cr)
                if abs_path.exists():
                    img = Image.open(abs_path).convert("RGB")
                    crops.append(np.asarray(img))
                    frames_for_embed.append(frame_idx)
            except Exception:
                logging.exception("embed-load: failed to load crop %s", cr)

    return crops, frames_for_embed

from facekit.utils.geometry import compute_iou  # add if not already imported


def extend_prev_track_for_overlapping_detection(
    *,
    aggregator: ShotFaceTrackAggregator,
    detections: List[FaceObservation],
    iou_threshold: float,
) -> int:
    """
    Pair-driven greedy IoU matching:
      - Build all (open_track, detection_obs) pairs with IoU >= threshold
      - Sort by IoU desc, tie-break deterministically
      - Assign 1-to-1 matches by descending IoU
      - Mutates detection FaceObservations by setting obs.track_id for matched obs

    Returns
    -------
    int : number of matches made
    """
    if not detections:
        return 0

    # Only consider detections that are not already assigned (should all be None here)
    det_indices = [i for i, ob in enumerate(detections) if getattr(ob, "track_id", None) is None]
    if not det_indices:
        return 0

    # Collect open tracks with a last bbox
    open_tracks = []
    for t in getattr(aggregator, "tracks", []) or []:
        if t.is_closed():
            continue
        last_bbox = t.get_last_bbox()
        if last_bbox is None:
            continue
        open_tracks.append((int(t.track_id), t, last_bbox))

    if not open_tracks:
        return 0

    # Build candidate pairs
    # pair = (iou, track_id, det_idx)
    pairs: list[tuple[float, int, int]] = []
    for det_idx in det_indices:
        ob = detections[det_idx]
        for track_id, _t, last_bbox in open_tracks:
            iou = float(compute_iou(last_bbox, ob.bbox))
            if iou >= float(iou_threshold):
                pairs.append((iou, track_id, det_idx))

    if not pairs:
        return 0

    # Sort: IoU desc, then track_id asc, then det_idx asc (det order already deterministic)
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    matched = 0

    for _iou, track_id, det_idx in pairs:
        if track_id in used_tracks:
            continue
        if det_idx in used_dets:
            continue

        detections[det_idx].track_id = int(track_id)
        used_tracks.add(track_id)
        used_dets.add(det_idx)
        matched += 1

    return matched



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
    resume_enabled: bool = True,
) -> List[FaceTrack]:
    """
    Track faces across precomputed shot segments, attach embeddings, and return per-shot tracks.

    This function is the main, **shot-aware** tracker. For each shot [first_frame..last_frame]:
      1) **Streams frames sequentially** via a `FrameProvider` (`ReaderCoordinator` by default).
         We jump directly to the desired starting frame with `reset_to_frame(start_at)` and
         then consume frames with `next()`.
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
    checkpoint : TrackingCheckpoint | None, optional
        Checkpoint interface used for sidecar persistence (observations/embeddings),
        status.json updates, and rehydration of prior observations/tracks.
    resume_enabled : bool, default True
        When True, this function resolves a resume anchor from the `checkpoint`
        (preferring `get_resume_anchor()`, then `last_detection_frame` from status,
        then the maximum observed frame in the observations collector). The tracker
        will start at `max(anchor, shot_first)` within the containing shot and will
        rehydrate prior observations strictly *before* the anchor. When False,
        the function ignores any resume state and starts from the first shot frame.

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
    - **Lifecycle:** When a `FrameProvider` is supplied, this function does not close it. When a path is supplied,
      the internally constructed provider is closed automatically via `ExitStack()`.
    - **Performance knobs:** `detect_interval` trades accuracy for speed; increasing it reduces detector calls.
      `embedding_batch_size_max` controls memory/throughput on the embedder.

    Resume semantics & label continuity
    -----------------------------------
    When resuming, we rehydrate prior tracks (up to the resume anchor) from the persisted
    observations sidecar. We then assign stable per-shot `segment_id`s **immediately** to the
    rehydrated tracks using the checkpoint's `track_order` mapping, so labels match the
    pre-interruption numbering. For the active shot (and any subsequent shots), we seed the
    segment-id counter with the number of already-assigned tracks for that shot so new tracks
    continue numbering where the previous run left off (no gaps or renumbering).
    """
    logging.info(
        "ckpt.paths: status_path=%s; _ckpt_run_root(checkpoint)=%s",
        getattr(checkpoint, "status_path", None),
        (_checkpoint_root_dir(checkpoint) if checkpoint else None),
    )

    shot_json_path = Path(shot_json_path)
    with open(shot_json_path, "r") as f:
        _shot_data = json.load(f)
    shots = _shot_data["shots"]

    _validate_shots_are_absolute_and_increasing(shots)

    with ExitStack() as stack:
        if isinstance(frame_source, (str, Path)):
            frame_provider = stack.enter_context(ReaderCoordinator(str(frame_source)))
        else:
            has_core = all(hasattr(frame_source, m) for m in ("fps", "size", "total_frames"))
            has_iter = hasattr(frame_source, "reset_to_frame") and hasattr(frame_source, "next")
            _ok = isinstance(frame_source, FrameProvider) or (has_core and has_iter)
            if not _ok:
                raise TypeError(
                    "frame_source must be a str/Path or a FrameProvider-like object with "
                    "fps, size, total_frames, reset_to_frame(i), and next(). "
                    f"Got {type(frame_source)!r}"
                )
            frame_provider = frame_source

        all_tracks: List[FaceTrack] = []

        # Centralized resume/rehydrate logic
        resume_plan, shots = _build_resume_plan(
            shots,
            checkpoint=checkpoint,
            resume_enabled=resume_enabled,
            all_tracks=all_tracks,
        )

        for shot_idx, shot_num in enumerate(shots):
            shot_number = shot_num["shot_number"]
            first = shot_num["first_frame"]
            last = shot_num["last_frame"]

            # Initialize shot-level aggregator and seeds (resume aware)
            start_at, aggregator, seg_seed = _init_shot_aggregator(
                shot_idx=shot_idx,
                shot_number=shot_number,
                first=first,
                last=last,
                detect_interval=detect_interval,
                resume_plan=resume_plan,
                iou_thresh=iou_thresh,
                embedding_thresh=embedding_thresh,
                checkpoint=checkpoint,
            )

            face_tracker = FaceTracker(tracker_type="CSRT")
            tracker_active = False

            shot_frames: List[np.ndarray] = []
            validator = TrackerValidator(
                frames=shot_frames,
                first_frame_idx=start_at,
                params=ValidatorParams(iou_thresh=iou_thresh),
            )

            frame_provider.reset_to_frame(int(start_at))

            for frame_idx in range(start_at, last + 1):
                frame = frame_provider.next()
                frame = _guard_seek_failure(
                    frame=frame,
                    frame_idx=frame_idx,
                    start_at=start_at,
                    shot_number=shot_number,
                    first=first,
                    last=last,
                    resume_plan=resume_plan,
                )
                if frame is None:
                    continue

                _guard_no_rewind(frame_idx, resume_plan)

                shot_frames.append(frame)
                observations: List[FaceObservation] = []

                is_scheduled_detect_frame = (frame_idx % detect_interval == 0)

                no_tracker = (not tracker_active)
                no_open_tracks = not any(not t.is_closed() for t in aggregator.tracks)

                # Determine if this is a detection frame
                do_detection = is_scheduled_detect_frame or no_tracker or no_open_tracks

                if not do_detection:
                    # Try tracking all existing tracks
                    tracked_boxes: Dict[int, Tuple[float, float, float, float]] = face_tracker.update_trackers(frame)
                    basic_fail = (not tracked_boxes) or any(b is None for b in tracked_boxes.values())

                    if (not basic_fail) and validator.validate(tracked_boxes, frame_idx, verbose=True):
                        for track_id, tb in tracked_boxes.items():
                            if tb is None:
                                continue
                            x, y, w, h = tb
                            bbox = (int(x), int(y), int(x + w), int(y + h))
                            observations.append(
                                FaceObservation(
                                    frame_idx=frame_idx,
                                    track_id=track_id,
                                    bbox=bbox,
                                    embedding=None,
                                    confidence=None,
                                    aligned_face=None,
                                    source=Source.TRACKED,
                                )
                            )
                    else:
                        aggregator.finalize_tracks()
                        tracker_active = False
                        do_detection = True

                if do_detection:  # Run detection 

                    do_checkpoint(
                        checkpoint,
                        frame_idx=frame_idx,
                        shot_number=shot_number,
                        aggregator=aggregator,
                        shot_first_frame=first,
                    )

                    detections = detector.detect_faces_in_frame(frame)

                    if detections:
                        boxes, landmark_lists, confidences = detections
                        aligned_ok = 0

                        # Deterministically order detections
                        order = sorted(
                            range(len(boxes)),
                            key=lambda i: (
                                int(boxes[i][0]),
                                int(boxes[i][1]),
                                int(boxes[i][2]),
                                int(boxes[i][3]),
                                -float(confidences[i]),
                            ),
                        )
                        boxes = [boxes[i] for i in order]
                        landmark_lists = [landmark_lists[i] for i in order]
                        confidences = [confidences[i] for i in order]

                        for box, landmarks, confidence in zip(boxes, landmark_lists, confidences):
                            bbox = tuple(int(v) for v in box[:4])
                            aligned_face = align_face_for_arcface(
                                frame, landmarks, frame_idx, source="detect"
                            )

                            if aligned_face is not None:
                                aligned_ok += 1

                                observations.append(
                                    FaceObservation(
                                        frame_idx=frame_idx,
                                        bbox=bbox,
                                        embedding=None,
                                        confidence=float(confidence),
                                        aligned_face=aligned_face,
                                        source=Source.DETECTED,
                                    )
                                )
                    else:
                        # no faces this frame; close shot-local tracks
                        aggregator.finalize_tracks()

                if observations and observations[0].source == Source.DETECTED:
                    dets = []
                    for ob in observations:
                        x1, y1, x2, y2 = [int(v) for v in ob.bbox[:4]]
                        dets.append(
                            {
                                "bbox": (x1, y1, x2, y2),
                                "conf": float(ob.confidence)
                                if ob.confidence is not None
                                else None,
                            }
                        )

                if observations and observations[0].source == Source.DETECTED:
                    _ = extend_prev_track_for_overlapping_detection(
                        aggregator=aggregator,
                        detections=observations,
                        iou_threshold=iou_thresh,
                    )

                _ = aggregator.update_tracks_with_frame(
                    frame_idx, observations
                    )

                _save_crops_for_frame(
                    checkpoint,
                    shot_number=shot_number,
                    frame_idx=frame_idx,
                    aggregator=aggregator,
                )

                _checkpoint_observations_and_snapshot(
                    checkpoint,
                    shot_number=shot_number,
                    frame_idx=frame_idx,
                    aggregator=aggregator,
                    resume_plan=resume_plan,
                )

                agg_det = aggregator.observations_at(
                    frame_idx, source=Source.DETECTED, require_track_id=True
                )
                agg_trk = aggregator.observations_at(
                    frame_idx, source=Source.TRACKED, require_track_id=True
                )

                # If faces detected, init tracker with aggregator IDs and seed validator
                if do_detection:
                    det_obs = aggregator.observations_at(
                        frame_idx, source=Source.DETECTED, require_track_id=True
                    )
                    if det_obs:
                        boxes_xywh = [
                            (b[0], b[1], b[2] - b[0], b[3] - b[1])
                            for b in (obs.bbox for obs in det_obs)
                        ]
                        track_ids = [obs.track_id for obs in det_obs]
                        face_tracker.init_trackers(frame, boxes_xywh, track_ids)

                        boxes_map = dict(zip(track_ids, boxes_xywh))
                        validator.set_baseline(boxes_map, frame_idx)

                        tracker_active = True
                    else:
                        tracker_active = False

                if checkpoint:
                    checkpoint.on_frame(frame_idx)

            # ---- end-of-shot embedding pass ----------
            run_root = _checkpoint_root_dir(checkpoint) if checkpoint else None

            for track in aggregator.tracks:
                crops, frames_for_embed = _collect_crops_for_embedding(
                    track,
                    checkpoint=checkpoint,
                    run_root=run_root,
                    # embedding_frame_policy=None  # uses default for now
                )

                if not crops:
                    continue

                embs = embedder.get_embedding_batch(
                    crops,
                    batch_size=embedding_batch_size_max,
                )


                if embs is None:
                    logging.error(
                        "embed: embedder returned None for shot=%d tid=%s",
                        int(shot_number),
                        str(getattr(track, "track_id", "NA")),
                    )
                    continue


                if not isinstance(embs, np.ndarray):
                    raise TypeError(f"Embedder must return np.ndarray, got {type(embs)}")
            
                if embs.ndim != 2 or embs.shape[1] != 512:
                    raise ValueError(
                        f"Embedder returned invalid array shape {embs.shape}; expected (K,512)"
                    )
                if embs.dtype != np.float32:
                    embs = np.asarray(embs, dtype=np.float32, order="C")

                
                if len(embs) != len(frames_for_embed):
                    raise RuntimeError(
                        f"Embed count/frame mismatch: embs={len(embs)} frames={len(frames_for_embed)} "
                        f"(shot={shot_number}, tid={track.track_id})"
                    )

                aggregator.attach_embeddings(track.track_id, embs)

                logging.info(
                    f"end of shot {shot_number} and {len(embs)} embeddings attached to aggregator"
                )

                _persist_embeddings_for_track(
                    checkpoint,
                    shot_number=shot_number,
                    track=track,
                    frames_for_embed=frames_for_embed,
                    embs=embs,
                )

            aggregator.finalize_tracks()

            # Assign segment_id per shot, seeded to continue numbering after any rehydrated tracks
            seed = int(seg_seed)
            try:
                _ = aggregator.resolve_segment_ids(
                    segment_id_counter=seed,
                    embedding_threshold=embedding_thresh,
                )
            except RuntimeError as e:
                logging.error(
                    "segment-id resolution skipped for shot=%d due to missing embeddings: %s",
                    int(shot_number),
                    e,
                )

            all_tracks.extend(aggregator.tracks)

            if checkpoint:
                checkpoint.on_shot_done()

        _finalize_checkpoint_run(checkpoint)

        return all_tracks

