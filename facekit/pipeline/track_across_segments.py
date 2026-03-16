import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Union, Callable
import copy
from contextlib import ExitStack
import logging

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.tracking.face_tracker import FaceTracker
from facekit.tracking.tracker_validator import TrackerValidator, ValidatorParams
from facekit.embedding.embedder import FaceEmbedder
from facekit.embedding.alignment import align_face_for_arcface
from facekit.embedding.embedding_queue import AlignedFaceEmbeddingQueue
from facekit.detection.face_detector import FaceDetector
from facekit.io.frame_provider import FrameProvider, ReaderCoordinator
from facekit.pipeline.checkpoint import TrackingCheckpoint
from facekit.common.obs_consts import Source
from facekit.utils.geometry import compute_iou 
from facekit.pipeline.resume_rehydrate import (
    ResumePlan,
    _build_resume_plan,
    _validate_shots_are_absolute_and_increasing,
    bootstrap_runtime_trackers_for_resume_frame,
)
from facekit.pipeline.checkpoint_io import (
    _checkpoint_root_dir,
    _checkpoint_observations_and_snapshot,
    _persist_embeddings_for_track,
    _finalize_checkpoint_run
)
from facekit.errors import ResumeSafetyError
from facekit.tracking.landmark_propagation import propagate_landmarks_by_bbox_transform
from facekit.embedding.track_embedding_queueing import maybe_enqueue_track_observation_for_embedding
from facekit.embedding.embedding_selection import TrackEmbeddingSample

logger = logging.getLogger(__name__)

def _snapshot_open_tracks_for_status(aggregator, shot_number: int) -> list[dict]:
    rows: list[dict] = []
    for t in getattr(aggregator, "tracks", []):
        try:
            is_closed = t.is_closed() if callable(getattr(t, "is_closed", None)) else bool(getattr(t, "is_closed", False))
        except Exception:
            is_closed = False
        if is_closed:
            continue
        rows.append(
            {
                "shot": int(shot_number),
                "track_id": int(getattr(t, "track_id")),
            }
        )
    rows.sort(key=lambda r: (int(r["shot"]), int(r["track_id"])))
    return rows

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
    # Under the embedding-safe-frame contract, resume begins at the frame
    # *following* the embedding-safe frame.
    if shot_idx == 0 and resume_plan.is_resume:
        start_at = max(first, int(resume_plan.anchor_frame) + 1)
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
        
    is_resume_shot = (
        shot_idx == 0
        and resume_plan.is_resume
        and int(shot_number) == int(resume_plan.first_processed_shot_number)
    )

    # Seeded tracks only for the anchor-containing resume shot.
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

        # Normalize seeded track lifecycle from checkpoint-declared open ids.
        # Having pre-resume observations does not, by itself, mean the track was
        # still open at the embedding-safe boundary.
        open_tids = set(getattr(resume_plan, "open_track_ids_anchor", frozenset()))
        for t in seeded_tracks:
            tid = int(getattr(t, "track_id", -1))
            if tid in open_tids:
                if hasattr(t, "mark_open") and callable(getattr(t, "mark_open")):
                    t.mark_open()
            else:
                if hasattr(t, "mark_closed") and callable(getattr(t, "mark_closed")):
                    t.mark_closed()
            t.is_active = False

        # Resume-shot seeded tracks should enter resolve_segment_ids() in the same
        # shot-local state as a cold run: track continuity preserved, but no
        # preassigned shot-local segment labels.
        if is_resume_shot:
            for t in seeded_tracks:
                setattr(t, "segment_id", None)

        # Preserve only the checkpoint-declared OPEN tracks as open for resume.
        # A rehydrated track having pre-resume observations does NOT imply it was
        # still live at the embedding-safe boundary.
        open_tids = set(getattr(resume_plan, "open_track_ids_anchor", frozenset()))
        for t in seeded_tracks:
            tid = int(getattr(t, "track_id", -1))
            if tid in open_tids:
                if hasattr(t, "mark_open") and callable(getattr(t, "mark_open")):
                    t.mark_open()
            else:
                if hasattr(t, "mark_closed") and callable(getattr(t, "mark_closed")):
                    t.mark_closed()
            t.is_active = False

        # For the resume shot, preserve the warm-start track state but clear any
        # preassigned segment_id so resolve_segment_ids() sees the same shot-local
        # input state as a cold run.
        if is_resume_shot:
            for t in seeded_tracks:
                setattr(t, "segment_id", None)

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
        logging.info(
            "resume: suppressing forced tid reuse=%d in shot=%d because seeded tracks already exist",
            int(resume_plan.reuse_tid_for_first_shot),
            int(shot_number),
        )

    if seeded_tracks:
        logging.info("RESUME: aggregator seeded with %d tracks", len(aggregator.tracks))
        for track in aggregator.tracks:
            last_bbox = getattr(track, "get_last_bbox", lambda: None)()
            logging.info(
                "RESUME TRACK: shot=%s tid=%s seg=%s first=%s last=%s closed=%s last_bbox=%s",
                getattr(track, "shot_id", None),
                getattr(track, "track_id", None),
                getattr(track, "segment_id", None),
                (track.first_frame() if hasattr(track, "first_frame") else None),
                (track.last_frame() if hasattr(track, "last_frame") else None),
                track.is_closed() if hasattr(track, "is_closed") else None,
                last_bbox,
            )

    seed_tid = int(resume_plan.trackid_seed_by_shot.get(int(shot_number), 0))
    seg_seed = int(
        resume_plan.segment_id_seed_by_shot.get(int(shot_number), 0)
    ) if resume_plan.segment_id_seed_by_shot else 0

    # For the resume shot, segment-id resolution should begin from the same
    # shot-local seed as a cold run.
    if is_resume_shot:
        seg_seed = 0

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


def _collect_aligned_faces_for_embedding(
    track,
    *,
    frame_provider: FrameProvider,
    embedding_frame_policy: Callable[[object, object, int], bool] | None = None,
):
    """
    Builds and collects (aligned_faces, frame_indices) for a single track.
    The optional embedding_frame_policy(track, obs, frame_idx) can be used
    to subsample frames (e.g., stride rules) without changing the call site.
    """
    if embedding_frame_policy is None:
        embedding_frame_policy = _default_embedding_frame_policy

    aligned_faces: list[np.ndarray] = []
    frames_for_embed: list[int] = []

    for obs in getattr(track, "observations", []) or []:
        frame_idx = int(getattr(obs, "frame_idx", -1))

        # Let the policy decide if this frame is even eligible
        try:
            if not embedding_frame_policy(track, obs, frame_idx):
                continue
        except Exception:
            # If policy explodes, skip this obs but keep going.
            logging.exception(
                "embed-policy: error applying embedding_frame_policy for "
                "track_id=%s frame=%s",
                getattr(track, "track_id", None),
                frame_idx,
            )
            continue

        # # Case 1: aligned_face already in memory
        # if getattr(obs, "aligned_face", None) is not None:
        #     try:
        #         crops.append(obs.aligned_face)
        #         frames_for_embed.append(frame_idx)
        #     except Exception:
        #         logging.exception(
        #             "embed-collect: failed to use aligned_face for track_id=%s frame=%s",
        #             getattr(track, "track_id", None),
        #             frame_idx,
        #         )
        #     continue

        # # Case 2: try to load crop from disk via crop_ref
        # cr = getattr(obs, "crop_ref", None)
        # if checkpoint and cr and run_root is not None:
        #     try:
        #         abs_path = Path(run_root, cr)
        #         if abs_path.exists():
        #             img = Image.open(abs_path).convert("RGB")
        #             crops.append(np.asarray(img))
        #             frames_for_embed.append(frame_idx)
        #     except Exception:
        #         logging.exception("embed-load: failed to load crop %s", cr)

        # Only embed from DETECTED observations
        if getattr(obs, "source", None) != Source.DETECTED:
            continue

        # Skip if already embedded
        if getattr(obs, "embedding", None) is not None:
            continue

        # Must already have aligned_face cached
        aligned = getattr(obs, "aligned_face", None)
        if aligned is None:
            continue

        aligned_faces.append(aligned)
        frames_for_embed.append(frame_idx)

    return aligned_faces, frames_for_embed

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

def _normalize_embedding_for_sample(embedding: np.ndarray) -> np.ndarray:
    arr = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm

def _attach_and_persist_embedded_obs(
    *,
    embedded_obs: list[FaceObservation],
    aggregator: ShotFaceTrackAggregator,
    checkpoint: TrackingCheckpoint | None,
    shot_number: int,
    shot_first_frame: int | None,
    safe_frame_idx: int | None = None,
) -> None:
    """
    For a batch of observations that have just been embedded:
      - attach embeddings to the aggregator per track_id
      - persist embeddings per track via _persist_embeddings_for_track
      - record an embedding-safe anchor after successful persistence (checkpointing only)

    Notes
    -----
    The "embedding payload frame" and the "embedding-safe frame" are not always the same.
    - payload frame: newest frame that actually contributed embeddings in this batch
    - safe frame: frame whose processing completed the flush/drain boundary

    If `safe_frame_idx` is provided, it is the authoritative embedding-safe frame.
    Otherwise we fall back to the newest persisted payload frame.
    """
    if not embedded_obs:
        return

    # Group by track_id
    by_tid: dict[int, list[FaceObservation]] = {}
    for ob in embedded_obs:
        tid = getattr(ob, "track_id", None)
        emb = getattr(ob, "embedding", None)
        if tid is None or emb is None:
            continue
        by_tid.setdefault(int(tid), []).append(ob)

    # Track the newest frame whose embeddings were actually persisted in this batch.
    max_persisted_frame: int | None = None

    for tid, obs_list in by_tid.items():
        # Ensure stable ordering by frame index
        obs_list.sort(key=lambda o: int(o.frame_idx))

        frames_for_embed = [int(o.frame_idx) for o in obs_list]
        embs = np.stack([np.asarray(o.embedding, dtype=np.float32) for o in obs_list], axis=0)

        aggregator.attach_embeddings(int(tid), embs)

        # Find the actual FaceTrack object for this tid.
        track = next(
            (t for t in aggregator.tracks if int(getattr(t, "track_id", -1)) == int(tid)),
            None,
        )

        # Register per-sample embedding metadata on the track.
        if track is not None and hasattr(track, "add_embedding_sample"):
            for ob in obs_list:
                try:
                    track_local_index = next(
                        i
                        for i, candidate in enumerate(track.observations)
                        if candidate is ob
                    )
                except StopIteration:
                    logging.warning(
                        "track-embed: embedded observation missing from track history "
                        "shot=%d track_id=%d frame=%d source=%s",
                        int(shot_number),
                        int(tid),
                        int(getattr(ob, "frame_idx", -1)),
                        str(getattr(ob, "source", None)),
                    )
                    continue

                track.add_embedding_sample(
                    TrackEmbeddingSample(
                        frame_idx=int(ob.frame_idx),
                        track_local_index=int(track_local_index),
                        source=getattr(ob, "source", None),
                        embedding=_normalize_embedding_for_sample(ob.embedding),
                        quality_score=None,
                    )
                )

        # Persist per track (if checkpoint enabled)
        if checkpoint is not None and track is not None:
            _persist_embeddings_for_track(
                checkpoint,
                shot_number=shot_number,
                track=track,
                frames_for_embed=frames_for_embed,
                embs=embs,
            )

            # Track the newest frame whose embeddings were persisted in this batch.
            if frames_for_embed:
                last_f = int(frames_for_embed[-1])
                if max_persisted_frame is None or last_f > max_persisted_frame:
                    max_persisted_frame = last_f
                    
    # Advance the embedding-safe resume anchor once per batch (after successful persistence).
    # Prefer the explicit boundary-completion frame when provided; otherwise use the newest
    # payload frame that contributed embeddings.
    anchor_frame = (
        int(safe_frame_idx)
        if safe_frame_idx is not None
        else (int(max_persisted_frame) if max_persisted_frame is not None else None)
    )
    if checkpoint is not None and anchor_frame is not None:
        try:
            open_tracks = _snapshot_open_tracks_for_status(aggregator, int(shot_number))
            checkpoint.mark_embedding_safe(
                frame_idx=int(anchor_frame),
                shot_number=int(shot_number),
                shot_first_frame=(int(shot_first_frame) if shot_first_frame is not None else None),
                open_tracks=open_tracks,
                note="embeddings persisted",
            )
        except Exception:
            # Non-fatal: embeddings were already persisted; this only affects resume metadata.
            logging.exception("track: failed to mark embedding-safe (non-fatal)")

def _maybe_enqueue_track_embedding_observations_for_frame(
    *,
    aggregator,
    frame_idx: int,
    frame,
    track_sample_interval: int,
    embedding_queue,
) -> None:
    """
    For the current frame's authoritative observations (after track assignment),
    decide which observations should be sampled for embedding, attempt alignment
    online, and enqueue the observation itself when alignment succeeds.

    Notes:
    - DETECTED observations are always eligible by policy.
    - TRACKED observations are eligible only when they land on the track-local
      sampling interval.
    - Observations without landmarks are skipped.
    - This helper mutates observation.aligned_face only for items that are
      actually enqueued.
    """
    obs_for_frame = aggregator.observations_at(
        frame_idx,
        require_track_id=True,
    )

    for obs in obs_for_frame:
        tid = getattr(obs, "track_id", None)
        if tid is None:
            continue

        track = next(
            (
                t
                for t in aggregator.tracks
                if int(getattr(t, "track_id", -1)) == int(tid)
            ),
            None,
        )
        if track is None:
            continue

        # update_tracks_with_frame(...) has already appended this frame's observation
        # to the authoritative track history, so the track-local index for this obs
        # is its current position in the per-track observation list.
        try:
            track_local_index = next(
                i
                for i, candidate in enumerate(track.observations)
                if candidate is obs
            )
        except StopIteration:
            continue

        if getattr(obs, "aligned_face", None) is not None:
            continue

        try:
            maybe_enqueue_track_observation_for_embedding(
                observation=obs,
                track_local_index=track_local_index,
                track_sample_interval=track_sample_interval,
                frame=frame,
                align_face_fn=align_face_for_arcface,
                embedding_queue=embedding_queue,
            )
        except Exception:
            logging.exception(
                "track-embed: failed enqueue attempt shot=%d frame=%d track_id=%s source=%s",
                int(getattr(track, "shot_id", -1)),
                int(frame_idx),
                str(tid),
                str(getattr(obs, "source", None)),
            )

def append_detection_observation(observations,
                                shot_number, 
                                frame_idx, 
                                frame, 
                                detected_box, 
                                landmarks, 
                                confidence):
    bbox = tuple(int(v) for v in detected_box[:4])

    aligned_face = None
    try:
        aligned_face = align_face_for_arcface(frame, landmarks)
    except Exception:
        logging.exception(
            "align: failed to compute aligned_face shot=%d frame=%d bbox=%s",
            int(shot_number),
            int(frame_idx),
            str(bbox),
        )

    observations.append(FaceObservation(
                frame_idx=frame_idx,
                bbox=bbox,
                track_id=None,  # aggregator will set
                embedding=None,
                confidence=float(confidence) if confidence is not None else None,
                aligned_face=aligned_face,
                landmarks=landmarks,
                source=Source.DETECTED,
            )
        )

def append_tracking_observation(
    observations,
    frame_idx,
    track_id,
    tracked_box,
    aggregator,
):
    x, y, w, h = tracked_box
    bbox = (int(x), int(y), int(x + w), int(y + h))

    previous_observation = None
    track = next(
        (t for t in aggregator.tracks if int(getattr(t, "track_id", -1)) == int(track_id)),
        None,
    )
    if track is not None and getattr(track, "observations", None):
        if track.observations:
            previous_observation = track.observations[-1]

    landmarks = None
    if previous_observation is not None:
        prev_landmarks = getattr(previous_observation, "landmarks", None)
        prev_bbox = getattr(previous_observation, "bbox", None)
        if prev_landmarks is not None and prev_bbox is not None:
            landmarks = propagate_landmarks_by_bbox_transform(
                prev_landmarks=prev_landmarks,
                prev_bbox=prev_bbox,
                curr_bbox=bbox,
            )

    observations.append(
        FaceObservation(
            frame_idx=frame_idx,
            track_id=track_id,
            bbox=bbox,
            embedding=None,
            confidence=None,
            aligned_face=None,
            landmarks=landmarks,
            source=Source.TRACKED,
        )
    )

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
    embedding_queue_max_pending: int = 1024,
    track_sample_interval: int = 10,
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
      5) At shot end, batches all aligned faces through the `embedder` and attaches 512-D embeddings
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
        Embeds aligned faces at the end of each shot; must return an `np.ndarray` of shape (K, 512), float32.
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
          - attached 512-D embeddings for detection observations (if landmarks existed).

    Notes
    -----
    - **Detector scheduling:** Frames with detections produce aligned faces; tracking-only frames intentionally **do not**
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

    # # DEBUG
    # shots = [s for s in shots if int(s["shot_number"]) in [63,64]]

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

            logger.info(f"shot num: {shot_number}")

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

            validator = TrackerValidator(
                params=ValidatorParams(iou_thresh=iou_thresh),
            )

            embed_q = AlignedFaceEmbeddingQueue(
                max_batch_size=int(embedding_batch_size_max),
                max_pending=int(embedding_queue_max_pending)
            )

            logging.info(
                "embed_q created: max_batch_size=%d max_pending=%d shot=%d",
                int(embed_q.max_batch_size),
                int(embed_q.max_pending),
                int(shot_number),
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

                # Resume-specific bootstrap lives in resume_rehydrate.py:
                # on the first resumed frame of the anchor shot, recreate live
                # tracker objects from rehydrated OPEN tracks so this frame can
                # follow the normal tracking-first path.
                if bootstrap_runtime_trackers_for_resume_frame(
                    resume_plan=resume_plan,
                    shot_number=shot_number,
                    frame_idx=frame_idx,
                    start_at=start_at,
                    aggregator=aggregator,
                    face_tracker=face_tracker,
                    validator=validator,
                    frame=frame,
                ):
                    tracker_active = True

                observations: List[FaceObservation] = []

                is_scheduled_detect_frame = (frame_idx % detect_interval == 0)
                open_tracks = [t for t in aggregator.tracks if not t.is_closed()]
                no_open_tracks = (len(open_tracks) == 0)
                no_tracker = (not tracker_active)

                can_try_tracking = (not no_open_tracks) and tracker_active

                # Resume/cold parity rule:
                # - scheduled detection frames always detect
                # - otherwise, try tracking first if tracking is possible
                # - only fall back to detection if tracking is not eligible or fails
                try_tracking = (not is_scheduled_detect_frame) and can_try_tracking
                do_detection = not try_tracking

                if try_tracking:
                        # Try tracking all existing tracks
                        tracked_boxes: Dict[int, Tuple[float, float, float, float]] = face_tracker.update_trackers(frame)
                        basic_fail = (not tracked_boxes) or any(b is None for b in tracked_boxes.values())

                        if (not basic_fail) and validator.validate(tracked_boxes, frame, frame_idx):
                            for track_id, tracked_box in tracked_boxes.items():
                                if tracked_box is None:
                                    continue

                                append_tracking_observation(
                                    observations,
                                    frame_idx,
                                    track_id,
                                    tracked_box,
                                    aggregator=aggregator,
                                )
                        else:
                            aggregator.finalize_tracks()
                            tracker_active = False
                            do_detection = True
                            observations = []

                if do_detection:  # Run detection 

                    detections = detector.detect_faces_in_frame(frame)

                    if detections:
                        boxes, landmark_lists, confidences = detections

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
                            if landmarks is None:
                                continue  # treat as "not a detected face"

                            append_detection_observation(observations,
                                                        shot_number, 
                                                        frame_idx, 
                                                        frame, 
                                                        box, 
                                                        landmarks, 
                                                        confidence)
                            
                        # print(f"Detection frame number: {frame_idx}, num faces: {len(observations)}")
                    
                    if not detections or not observations:
                        # no faces this frame; close shot-local tracks
                        aggregator.finalize_tracks()

                    if observations:
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

                        _ = extend_prev_track_for_overlapping_detection(
                            aggregator=aggregator,
                            detections=observations,
                            iou_threshold=iou_thresh,
                            )
      
                _ = aggregator.update_tracks_with_frame(frame_idx, observations)

                _maybe_enqueue_track_embedding_observations_for_frame(
                    aggregator=aggregator,
                    frame_idx=frame_idx,
                    frame=frame,
                    track_sample_interval=track_sample_interval,
                    embedding_queue=embed_q,
                )

                # Persist observations for EVERY processed frame.
                #
                # This includes TRACKED observations, which are still needed by the
                # current resume design to reconstruct pre-resume track history.
                #
                # Embedding persistence remains detection-only below.
                _checkpoint_observations_and_snapshot(
                    checkpoint,
                    shot_number=shot_number,
                    frame_idx=frame_idx,
                    aggregator=aggregator,
                    resume_plan=resume_plan,
                )

                if do_detection:
                    # Enqueue DETECTED observations for embedding now that track_id has
                    # been assigned and the DET rows for this frame have already been
                    # checkpointed above.
                    det_obs = aggregator.observations_at(
                        frame_idx, source=Source.DETECTED, require_track_id=True
                    )

                    # Embed with bounded memory, while preserving across-frame batching:
                    # - Opportunistically flush if we exceed max_pending.
                    # - ONLY do an end-of-frame flush if we already flushed at least once
                    #   during this frame (meaning we crossed the pending fence mid-frame).
                    embedded_obs = []
                    flushed_during_frame = False
                    for ob in det_obs:
                        embed_q.enqueue(ob)
                        flushed_now = embed_q.maybe_flush(embedder)
                        if flushed_now:
                            flushed_during_frame = True
                            embedded_obs.extend(flushed_now)

                    if flushed_during_frame:
                        tail = embed_q.flush(embedder)
                        embedded_obs.extend(tail)

                    _attach_and_persist_embedded_obs(
                        embedded_obs=embedded_obs,
                        aggregator=aggregator,
                        checkpoint=checkpoint,
                        shot_number=shot_number,
                        shot_first_frame=int(first),
                        safe_frame_idx=int(frame_idx),
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
                        validator.seed_validator(boxes_map, frame_idx, frame)

                        tracker_active = True
                    else:
                        tracker_active = False

                if checkpoint:
                    checkpoint.on_frame(frame_idx)

            # ---- end-of-shot embedding drain (queue) ----------

            # Flush any remaining queued aligned faces.
            embedded_obs = embed_q.flush(embedder)
            _attach_and_persist_embedded_obs(
                embedded_obs=embedded_obs,
                aggregator=aggregator,
                checkpoint=checkpoint,
                shot_number=shot_number,
                shot_first_frame=int(first),
                safe_frame_idx=int(last),
            )

            aggregator.finalize_tracks()

            # Assign segment_id per shot, seeded to continue numbering after any rehydrated tracks
            seed = int(seg_seed)

            try:
                _ = aggregator.resolve_segment_ids(
                    segment_id_counter=seed,
                    embedding_threshold=embedding_thresh,
                )
                logging.info(
                    "shot=%d seg_seed=%d segment_ids_by_range=%s",
                    int(shot_number),
                    int(seed),
                    sorted(
                        [
                            (
                                int(t.first_frame()),
                                int(t.last_frame()),
                                int(getattr(t, "track_id", -1)),
                                getattr(t, "segment_id", None),
                            )
                            for t in aggregator.tracks
                        ],
                        key=lambda x: (x[0], x[1], x[2]),
                    ),
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

        logger.info("Done processing video")

        return all_tracks
