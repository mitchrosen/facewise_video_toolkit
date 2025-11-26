import numpy as np
from pathlib import Path
import logging

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.pipeline.checkpoint import TrackingCheckpoint
from facekit.common.obs_consts import Source
from facekit.pipeline.resume_rehydrate import ResumePlan
from facekit.tracking.face_structures import FaceTrack, FaceObservation

logger = logging.getLogger(__name__)

def _checkpoint_root_dir(checkpoint) -> Path | None:
    """
    Return the run root directory for the active checkpoint.
    Prefers .root (run_dir), falls back to .run_dir, else parent of .ckpt_dir.
    """
    candidate = getattr(checkpoint, "root", None)
    if candidate:
        return Path(candidate)
    candidate = getattr(checkpoint, "run_dir", None)
    if candidate:
        return Path(candidate)
    ckpt_dir = getattr(checkpoint, "ckpt_dir", None)
    if ckpt_dir:
        return Path(ckpt_dir).parent
    # For in-memory / test stubs that don't persist to disk, we can proceed
    # without a run root. This only affects logging / debug snapshots.
    logger.warning(
        "Cannot determine checkpoint run root (need one of .root, .run_dir, or .ckpt_dir); "
        "proceeding with None (disk-backed debug snapshots will be disabled)."
    )
    return None

# def _save_crops_for_frame(
#     checkpoint: TrackingCheckpoint | None,
#     *,
#     shot_number: int,
#     frame_idx: int,
#     aggregator: ShotFaceTrackAggregator,
# ) -> None:
#     """
#     Persist aligned 112×112 crops associated with detection observations on a frame.

#     For each DET observation with an in-memory aligned_face and no crop_ref yet,
#     this function:
#       * Writes a PNG into the checkpoint's run-root under ckpt/crops/shot-####.
#       * Sets obs.crop_ref to the path relative to the run-root so that
#         checkpoint.add_observations can persist the reference.

#     Parameters
#     ----------
#     checkpoint :
#         Optional TrackingCheckpoint providing the run-root. If None, this is a no-op.
#     shot_number :
#         Logical shot identifier.
#     frame_idx :
#         Absolute frame index just processed.
#     aggregator :
#         ShotFaceTrackAggregator from which DET observations are read.
#     """
#     if not checkpoint:
#         return
#     det_obs = aggregator.observations_at(
#         frame_idx, source=Source.DETECTED, require_track_id=True
#     )
#     if not det_obs:
#         return

#     crops_root = _checkpoint_root_dir(checkpoint)
#     _saved = 0
#     for ob in det_obs:
#         if getattr(ob, "aligned_face", None) is None:
#             continue
#         if getattr(ob, "crop_ref", None):
#             continue
#         try:
#             rel = _save_crop_for_obs(
#                 crops_root, shot_number, ob.frame_idx, ob.track_id, ob.aligned_face
#             )
#             setattr(ob, "crop_ref", rel)
#             _saved += 1
#         except Exception:
#             logger.exception(
#                 "crop-archive: failed at frame=%d tid=%s", ob.frame_idx, ob.track_id
#             )
#     logger.info(
#         "CROP-SAVED frame=%d shot=%d saved=%d",
#         frame_idx,
#         int(shot_number),
#         int(_saved),
#     )

def do_checkpoint(
    checkpoint: TrackingCheckpoint | None,
    *,
    frame_idx: int,
    shot_number: int,
    aggregator: ShotFaceTrackAggregator,
    shot_first_frame: int,
) -> None:
    """
    Issue a checkpoint event, if a checkpoint is configured.

    This is called just before running the detector on a given frame so that
    status.json and sidecars reflect the current state even if detection or
    embedding work crashes.

    Parameters
    ----------
    checkpoint :
        Optional TrackingCheckpoint; if None, this is a no-op.
    frame_idx :
        Absolute frame index about to be processed by the detector.
    shot_number :
        Logical shot identifier.
    aggregator :
        Current ShotFaceTrackAggregator for this shot.
    shot_first_frame :
        Absolute first frame of this shot (for status/debug).
    """
    if not checkpoint:
        return
    try:
        checkpoint.checkpoint_now(
            frame_idx=frame_idx,
            shot_number=shot_number,
            aggregator=aggregator,
            shot_first_frame=shot_first_frame,
            note=f"detect@{frame_idx}",
        )
    except Exception:
        logger.exception("checkpoint: failed to persist at detect frame %s", frame_idx)


def _checkpoint_observations_and_snapshot(
    checkpoint: TrackingCheckpoint | None,
    *,
    shot_number: int,
    frame_idx: int,
    aggregator: ShotFaceTrackAggregator,
    resume_plan: ResumePlan,
) -> None:
    """
    Persist observations for a frame and (optionally) write a lightweight snapshot.

    This function:
      * Collects FaceObservation objects for the frame.
      * At the anchor frame, optionally retries without require_track_id to avoid
        losing DET rows at the resume boundary.
      * Validates that all observations have FaceObservation type and Source enums.
      * Calls checkpoint.add_observations().
      * If snapshots are enabled, writes a small payload describing the detect frame.

    Parameters
    ----------
    checkpoint :
        Optional TrackingCheckpoint; if None, this is a no-op.
    shot_number :
        Logical shot identifier.
    frame_idx :
        Absolute frame index just processed.
    aggregator :
        ShotFaceTrackAggregator from which we read the observations.
    resume_plan :
        ResumePlan with anchor_frame for the anchor special case.
    """
    if not checkpoint:
        return

    frame_obs_objs = aggregator.observations_at(frame_idx, require_track_id=True)
    if (not frame_obs_objs) and (frame_idx == resume_plan.anchor_frame):
        logger.info(
            "resume: no require_track_id observations at anchor; retrying without strictness."
        )
        frame_obs_objs = aggregator.observations_at(frame_idx, require_track_id=False)

    if frame_obs_objs:
        if not all(isinstance(o, FaceObservation) for o in frame_obs_objs):
            bad = [
                type(o).__name__
                for o in frame_obs_objs
                if not isinstance(o, FaceObservation)
            ]
            logger.error(
                "seg: Non-FaceObservation in frame_obs_objs at frame=%s shot=%s types=%s sample=%r",
                frame_idx,
                shot_number,
                bad,
                frame_obs_objs[:1],
            )
            raise TypeError(
                f"frame_obs_objs must be FaceObservation objects, got {bad}"
            )
        if not all(isinstance(o.source, Source) for o in frame_obs_objs):
            bad = [
                getattr(o, "source", None)
                for o in frame_obs_objs
                if not isinstance(getattr(o, "source", None), Source)
            ]
            logger.error(
                "seg: Observation with non-enum source at frame=%s shot=%s bad=%r",
                frame_idx,
                shot_number,
                bad[:3],
            )
            raise TypeError(
                f"Observation.source must be Source enum; bad={bad[:3]}"
            )

        # Enforce: DETECTED observations must have 5-point landmarks.
        for obs in frame_obs_objs:
            if obs.source == Source.DETECTED:
                lm = getattr(obs, "landmarks", None)
                if lm is None:
                    raise ValueError(
                        f"checkpoint: DETECTED obs missing landmarks "
                        f"(shot={shot_number}, frame={frame_idx}, tid={getattr(obs,'track_id',None)})"
                    )
                arr = np.asarray(lm, dtype=np.float32)
                if arr.shape != (5, 2):
                    raise ValueError(
                        f"checkpoint: landmarks must be (5,2) for DETECTED obs "
                        f"(shot={shot_number}, frame={frame_idx}, tid={getattr(obs,'track_id',None)}), got {arr.shape}"
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"checkpoint: landmarks contain NaN/Inf for DETECTED obs "
                        f"(shot={shot_number}, frame={frame_idx}, tid={getattr(obs,'track_id',None)})"
                    )

        checkpoint.add_observations(shot_number, frame_idx, frame_obs_objs)

    try:
        if getattr(checkpoint, "snapshots_ready", False):
            checkpoint.write_checkpoint_snapshot(
                name=f"detect-{shot_number}-{frame_idx}",
                payload={
                    "shot": int(shot_number),
                    "frame": int(frame_idx),
                    "note": f"detect@{frame_idx}",
                },
            )
    except Exception:
        logger.exception("checkpoint: snapshot write failed at detect frame %s", frame_idx)

def _persist_embeddings_for_track(
    checkpoint: TrackingCheckpoint | None,
    *,
    shot_number: int,
    track: FaceTrack,
    frames_for_embed: list[int],
    embs: np.ndarray,
) -> None:
    """
    Persist per-frame embeddings for a single track into the checkpoint sidecar.

    B2 contract enforced here:
      - embeddings may ONLY be written for DETECTED frames
      - DETECTED frames must have valid landmarks
      - (optional) refuse to overwrite an existing embedding for the same frame
    """
    if checkpoint is None or embs is None or int(getattr(embs, "shape", (0,))[0]) == 0:
        logging.info(f"end of shot {shot_number} and NO embeddings added to checkpoint")
        return

    if len(frames_for_embed) != int(embs.shape[0]):
        raise ValueError(
            f"frames/embs mismatch: frames={len(frames_for_embed)} embs={embs.shape}"
        )

    shot_i = int(shot_number)
    tid_i = int(track.track_id)

    # ------------------------------------------------------------------
    # Embeddings may ONLY be written for DETECTED
    # frames AND those DETECTED frames must have landmarks.
    # ------------------------------------------------------------------
    max_f = max(int(f) for f in frames_for_embed)

    # 1) DETECTED frames up to max_f
    det_frames = set(
        checkpoint.get_det_frames_with_landmarks_for_track(
            shot_i,
            tid_i,
            frame_max=max_f,
        )
    )

    # 2) DETECTED+LANDMARKS frames up to max_f
    #    (This assumes your checkpoint can answer this. If it can't yet,
    #     see "Minimal supporting API" below.)
    det_landmark_frames = set(
        int(f)
        for f in checkpoint.get_det_frames_with_landmarks_for_track(
            shot_i,
            tid_i,
            frame_max=max_f,
        )
    )

    # 3) Every requested frame must be in det_frames
    bad_not_detected = [int(f) for f in frames_for_embed if int(f) not in det_frames]
    if bad_not_detected:
        raise ValueError(
            "[ckpt] refusing to persist embeddings for non-DETECTED frames: "
            f"shot={shot_i} track={tid_i} bad_frames={bad_not_detected} "
            f"det_frames_up_to_{max_f}={sorted(det_frames)}"
        )

    # 4) Every requested frame must be in det_landmark_frames
    bad_missing_landmarks = [
        int(f) for f in frames_for_embed if int(f) not in det_landmark_frames
    ]
    if bad_missing_landmarks:
        raise ValueError(
            "[ckpt] refusing to persist embeddings for DETECTED frames without landmarks: "
            f"shot={shot_i} track={tid_i} bad_frames={bad_missing_landmarks} "
            f"detected_with_landmarks_up_to_{max_f}={sorted(det_landmark_frames)}"
        )

    # ------------------------------------------------------------------
    # Refuse to overwrite if we already have an embedding stored
    # for that frame. This prevents silent double-writes on resume logic.
    # ------------------------------------------------------------------
    already = set(
        int(f)
        for f in checkpoint.get_emb_frames_for_track(
            shot_i,
            tid_i,
            frame_max=max_f,
        )
    )
    overwrite_frames = [int(f) for f in frames_for_embed if int(f) in already]
    if overwrite_frames:
        raise ValueError(
            "[ckpt] refusing to overwrite existing per-frame embeddings: "
            f"shot={shot_i} track={tid_i} frames={overwrite_frames}"
        )

    for f_idx, vec in zip(frames_for_embed, embs):
        checkpoint.add_embeddings(
            shot_i,
            tid_i,
            int(f_idx),
            np.asarray(vec, dtype=np.float32).reshape(1, -1),
        )

    logging.info(
        f"end of shot {shot_number} and {len(embs)} per-frame embeddings added to checkpoint"
    )

def _finalize_checkpoint_run(checkpoint: TrackingCheckpoint | None) -> None:
    """
    Finalize checkpoint sidecars at end-of-video and mark the run as completed.

    This flushes in-memory collectors to disk (obs_ckpt.npz, emb_ckpt.npz) and
    clears the "resume anchor" semantics so downstream tools can safely read
    the full obs/emb datasets without trimming to the last detection frame.

    Parameters
    ----------
    checkpoint :
        Optional TrackingCheckpoint; if None, this is a no-op.
    """
    if not checkpoint:
        return
    checkpoint.finalize(note="final video flush")
    checkpoint.mark_completed()

