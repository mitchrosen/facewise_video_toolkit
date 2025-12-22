import numpy as np
from pathlib import Path
import logging

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.pipeline.checkpoint import TrackingCheckpoint
from facekit.common.obs_consts import Source
import facekit.pipeline.resume_rehydrate as _resume_rehydrate
from facekit.pipeline.resume_rehydrate import ResumePlan
from facekit.tracking.face_structures import FaceTrack, FaceObservation

logger = logging.getLogger(__name__)

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

    Embeddings are stored with *frame-level* indices so that rehydrate logic
    can enforce parity between:
      - DET observations that have landmarks, and
      - stored embeddings
    and attach embeddings back to observations strictly before the resume anchor.

    Parameters
    ----------
    checkpoint :
        Optional TrackingCheckpoint; if None, only logging is performed.
    shot_number :
        Logical shot identifier.
    track :
        FaceTrack whose embeddings are being persisted.
    frames_for_embed :
        List of frame indices corresponding 1:1 with rows in `embs`.
    embs :
        (K, 512) float32 embedding array returned by the embedder.
    """
    if not (checkpoint and embs.size):
        logging.info(
            f"end of shot {shot_number} and NO embeddings added to checkpoint"
        )
        return

    for f_idx, vec in zip(frames_for_embed, embs):
        checkpoint.add_embeddings(
            int(shot_number),
            int(track.track_id),
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

