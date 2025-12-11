import numpy as np
from pathlib import Path
from facekit.utils.io import fsync_parent_dir
import tempfile
import os
from PIL import Image
import logging

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.pipeline.checkpoint import TrackingCheckpoint
from facekit.common.obs_consts import Source
import facekit.pipeline.resume_rehydrate as _resume_rehydrate
from facekit.pipeline.resume_rehydrate import ResumePlan
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.pipeline.checkpoint import TrackingCheckpoint


logger = logging.getLogger(__name__)

def _ckpt_run_root(checkpoint) -> Path | None:
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

def _shot_crops_dir(ckpt_root: Path, shot_number: int) -> Path:
    p = ckpt_root / "ckpt" / "crops" / f"shot-{int(shot_number):04d}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _atomic_write_png(dst: Path, img_np) -> None:
    # img_np expected 112x112x3, RGB or BGR depending on aligner
    # ArcFace aligner you’re using returns RGB — if yours is BGR, swap here.
    im = Image.fromarray(img_np)  # assumes RGB
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dst.parent, suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        im.save(tmp_path, format="PNG", optimize=True)
        tmp.flush(); os.fsync(tmp.fileno())
    os.replace(tmp_path, dst)
    # Ensure directory entry is durable on crash
    try:
        fsync_parent_dir(dst)
    except Exception as e:
        logger.info(f"_atomic_write_png: fsync of parent dir failed: {e}")
        pass

def _save_crop_for_obs(ckpt_root: Path, shot_number: int, frame_idx: int, tid: int, aligned_face) -> str:
    crops_dir = _shot_crops_dir(ckpt_root, shot_number)
    rel_name  = f"f{int(frame_idx):06d}_tid{int(tid):03d}.png"
    abs_path  = crops_dir / rel_name
    if not abs_path.exists():
        _atomic_write_png(abs_path, aligned_face)
    # return path relative to run root to keep status portable
    # run root == checkpoint.root
    rel_path = abs_path.relative_to(ckpt_root)
    return str(rel_path)

def _save_crops_for_frame(
    checkpoint: TrackingCheckpoint | None,
    *,
    shot_number: int,
    frame_idx: int,
    aggregator: ShotFaceTrackAggregator,
) -> None:
    """
    Persist aligned 112×112 crops associated with detection observations on a frame.

    For each DET observation with an in-memory aligned_face and no crop_ref yet,
    this function:
      * Writes a PNG into the checkpoint's run-root under ckpt/crops/shot-####.
      * Sets obs.crop_ref to the path relative to the run-root so that
        checkpoint.add_observations can persist the reference.

    Parameters
    ----------
    checkpoint :
        Optional TrackingCheckpoint providing the run-root. If None, this is a no-op.
    shot_number :
        Logical shot identifier.
    frame_idx :
        Absolute frame index just processed.
    aggregator :
        ShotFaceTrackAggregator from which DET observations are read.
    """
    if not checkpoint:
        return
    det_obs = aggregator.observations_at(
        frame_idx, source=Source.DETECTED, require_track_id=True
    )
    if not det_obs:
        return

    crops_root = _ckpt_run_root(checkpoint)
    _saved = 0
    for ob in det_obs:
        if getattr(ob, "aligned_face", None) is None:
            continue
        if getattr(ob, "crop_ref", None):
            continue
        try:
            rel = _save_crop_for_obs(
                crops_root, shot_number, ob.frame_idx, ob.track_id, ob.aligned_face
            )
            setattr(ob, "crop_ref", rel)
            _saved += 1
        except Exception:
            logger.exception(
                "crop-archive: failed at frame=%d tid=%s", ob.frame_idx, ob.track_id
            )
    logger.info(
        "CROP-SAVED frame=%d shot=%d saved=%d",
        frame_idx,
        int(shot_number),
        int(_saved),
    )

def _checkpoint_pre_detect(
    checkpoint: TrackingCheckpoint | None,
    *,
    frame_idx: int,
    shot_number: int,
    aggregator: ShotFaceTrackAggregator,
    shot_first_frame: int,
) -> None:
    """
    Issue a checkpoint_before-detection event, if a checkpoint is configured.

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

def _log_detect_persist(
    checkpoint: TrackingCheckpoint | None,
    *,
    shot_number: int,
    frame_idx: int,
    aggregator: ShotFaceTrackAggregator,
) -> None:
    """
    Emit a diagnostic log summarizing which DET rows were persisted for a frame.

    This is purely for debugging/forensics. It inspects either:
      - checkpoint.obs_collector.rows_for_frame(shot, frame), if available, or
      - aggregator.observations_at(frame, Source.DETECTED), as a fallback.

    Parameters
    ----------
    checkpoint :
        Optional TrackingCheckpoint; if None, this is a no-op.
    shot_number :
        Logical shot identifier.
    frame_idx :
        Absolute frame index just processed by the detector.
    aggregator :
        ShotFaceTrackAggregator from which we can derive DET observations
        when the collector API is not available.
    """
    if not checkpoint:
        return
    try:
        det_persisted = []
        if (
            hasattr(checkpoint, "obs_collector")
            and hasattr(checkpoint.obs_collector, "rows_for_frame")
        ):
            for r in checkpoint.obs_collector.rows_for_frame(shot_number, frame_idx):
                if int(r.get("src", -1)) == int(Source.DETECTED.value):
                    det_persisted.append(
                        {
                            "tid": int(r.get("tid", -1)),
                            "emb_idx": int(r.get("emb_idx", -1)),
                            "has_crop": 1 if r.get("crop_ref") else 0,
                        }
                    )
        else:
            for ob in aggregator.observations_at(
                frame_idx, source=Source.DETECTED, require_track_id=True
            ):
                det_persisted.append(
                    {
                        "tid": int(getattr(ob, "track_id", -1)),
                        "emb_idx": -1,
                        "has_crop": 1
                        if (
                            getattr(ob, "aligned_face", None) is not None
                            or getattr(ob, "crop_ref", None)
                        )
                        else 0,
                    }
                )
        if det_persisted:
            logging.info(
                "DETECT-PERSIST shot=%d frame=%d rows=%s",
                int(shot_number),
                int(frame_idx),
                det_persisted,
            )
    except Exception:
        logging.exception("resume-log: failed DETECT-PERSIST probe")

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
    can enforce DET↔EMB parity and attach embeddings back to observations
    strictly before the resume anchor.

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

