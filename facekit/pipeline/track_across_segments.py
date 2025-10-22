import os
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional, Iterable
import copy
import tempfile
from contextlib import ExitStack
from bisect import bisect_left
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
from facekit.pipeline.resume_rehydrate import rehydrate_tracks
from facekit.utils.geometry import compute_iou
from facekit.errors import ResumeSafetyError
from facekit.utils.io import fsync_parent_dir
from facekit.utils.debug_logging import _dump_agg_state

logger = logging.getLogger(__name__)
PARANOID = bool(os.environ.get("FACEKIT_PARANOID"))

# --- Logging helpers (diagnostics only) ---------------------------------------

def _last_det_bbox_and_frame(tr: FaceTrack) -> tuple[Optional[tuple[int,int,int,int]], int]:
    last_det_frame = -1
    last_det_bbox = None
    for o in getattr(tr, "observations", []) or []:
        src = getattr(o, "source", None)
        assert isinstance(src, Source), f"Non-enum source in FaceObservation: {src!r}"
 
        if src is Source.DETECTED and o.bbox is not None:
            if o.frame_idx > last_det_frame:
                last_det_frame = int(o.frame_idx)
                last_det_bbox = tuple(int(v) for v in o.bbox[:4])
    return last_det_bbox, last_det_frame

def _shot_idx_by_abs_frame(shots, abs_frame_idx: int) -> int:
    """
    Return the **shot index** i such that shots[i]["last_frame"] >= abs_frame_idx,
    choosing the leftmost such i. If no such shot exists (abs_frame_idx is after
    all shots), return len(shots).

    Notes:
    - Shots must be sorted, non-overlapping, ascending by frame range.
    - If abs_frame_idx < shots[0]["first_frame"], this returns 0. The caller
      could clamp the starting frame i.e:
          start_at = max(shots[i]["first_frame"], abs_frame_idx)
    """
    # First shot whose last_frame >= resume_abs_frame
    last_frames = [s["last_frame"] for s in shots]
    return bisect_left(last_frames, abs_frame_idx)

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
    logging.getLogger(__name__).warning(
        "Cannot determine checkpoint run root (need one of .root, .run_dir, or .ckpt_dir); "
        "proceeding with None (disk-backed debug snapshots will be disabled)."
    )
    return None

def _audit_preanchor_embedding_parity(checkpoint, *, shots: list, anchor_frame: int, anchor_shot: Optional[int]):
    """
    Resume-only audit: verify that every DET row with frame <= anchor-1 has an embedding.
    Uses rows_for_frame(shot, frame) if available. Falls back to iter_track_frames(...) to
    at least count DETs and flag likely gaps. Logs which frames are missing and whether a
    crop exists (when we can check it).
    """
    if not checkpoint or not hasattr(checkpoint, "obs_collector"):
        logging.info("RESUME-AUDIT: no checkpoint/collector; skipping.")
        return

    oc = checkpoint.obs_collector
    has_rows_for_frame = hasattr(oc, "rows_for_frame") and callable(getattr(oc, "rows_for_frame"))
    has_iter_track_frames = hasattr(oc, "iter_track_frames") and callable(getattr(oc, "iter_track_frames"))
    run_root = None
    try:
        run_root = _ckpt_run_root(checkpoint)
    except Exception:
        run_root = None

    if not has_rows_for_frame and not has_iter_track_frames:
        logging.info("RESUME-AUDIT: collector lacks rows_for_frame/iter_track_frames; skipping.")
        return

    # Helper: record a missing embedding at (shot, tid, frame) with crop presence if we can see it.
    def _record_missing(shot: int, tid: int, frame: int, crop_ref: Optional[str], det_by_key: Dict[tuple[int,int], list]):
        has_crop = False
        if crop_ref and run_root is not None:
            try:
                has_crop = (run_root / crop_ref).exists()
            except Exception:
                has_crop = False
        det_by_key.setdefault((shot, tid), []).append({"f": int(frame), "has_crop": int(bool(has_crop))})

    det_by_key: Dict[tuple[int,int], list] = {}
    total_missing = 0

    # Iterate only up to anchor-1 and (optionally) only up to anchor_shot.
    for s in shots:
        shot_num = int(s["shot_number"])
        if anchor_shot is not None and shot_num > int(anchor_shot):
            break  # nothing pre-anchor beyond the anchor shot
        first_f = int(s["first_frame"])
        last_f  = int(s["last_frame"])
        limit_last = min(last_f, (anchor_frame - 1)) if anchor_frame > 0 else last_f
        if limit_last < first_f:
            continue

        if has_rows_for_frame:
            # Strong path: examine persisted rows per frame.
            for f in range(first_f, limit_last + 1):
                try:
                    rows = list(oc.rows_for_frame(shot_num, f))  # iterable of dict-ish
                except Exception:
                    logging.exception("RESUME-AUDIT: rows_for_frame failed at shot=%d frame=%d", shot_num, f)
                    continue
                for r in rows:
                    try:
                        if int(r.get("src", -1)) != int(Source.DETECTED.value):
                            continue
                        tid = int(r.get("tid", -1))
                        emb_idx = int(r.get("emb_idx", -1))
                        if emb_idx < 0:
                            _record_missing(shot_num, tid, f, r.get("crop_ref"), det_by_key)
                            total_missing += 1
                    except Exception:
                        # tolerate malformed rows
                        continue
        elif has_iter_track_frames:
            # We can at least detect DET frames per (shot, tid) and infer that missing embs likely exist.
            # We won’t know crop_ref on this path unless rows_for_frame exists, so mark has_crop=0.
            # This is a coarse safety net.
            # Iterate known tids by scanning iter_track_frames for small tid range until it yields nothing.
            # If your collector exposes a way to list tids per shot, prefer that.
            seen_any = False
            for tid_guess in range(0, 2048):  # generous upper bound; adjust if you know a tighter cap
                try:
                    frames_src = list(oc.iter_track_frames(shot_num, tid_guess))
                except Exception:
                    frames_src = []
                if not frames_src:
                    # Heuristic break: after a few consecutive empty tids, stop probing.
                    # (Assumes dense, small tid space per shot.)
                    if tid_guess > 32:
                        break
                    continue
                seen_any = True
                for f, src_code in frames_src:
                    try:
                        f = int(f)
                        if f > limit_last:
                            continue
                        if int(src_code) != int(Source.DETECTED.value):
                            continue
                        # We don't know emb_idx here, but strict parity expects one vector per DET.
                        # Flag as "missing/unknown"; later rows_for_frame (if implemented) will be authoritative.
                        _record_missing(shot_num, int(tid_guess), f, None, det_by_key)
                        total_missing += 1
                    except Exception:
                        continue
            if not seen_any:
                logging.debug("RESUME-AUDIT: no tracks observed via iter_track_frames for shot=%d", shot_num)

    # Summarize
    if not det_by_key:
        logging.info("RESUME-AUDIT: no pre-anchor DET rows to audit (or none missing).")
        return

    for (shot, tid), misses in sorted(det_by_key.items()):
        misses.sort(key=lambda r: int(r["f"]))
        frames_str = ",".join(f'{m["f"]}({m["has_crop"]})' for m in misses)
        logging.error(
            "RESUME-AUDIT shot=%d tid=%d missing_preanchor=%d frames=[%s]",
            shot, tid, len(misses), frames_str
        )

    if total_missing == 0:
        logging.info("RESUME-AUDIT OK: all pre-anchor DET rows have embeddings.")
    else:
        logging.error("RESUME-AUDIT FAIL: total missing pre-anchor DET embeddings=%d", total_missing)

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
        logging.info(f"_atomic_write_png: fsync of parent dir failed: {e}")
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

def _assign_segment_ids_for_rehydrated(prior_tracks, track_order: dict[tuple[int,int], int]) -> dict[int, int]:
    """
    Deterministically assign segment_id to already rehydrated tracks per shot using persisted track_order.
    Returns: {shot_number: next_segment_seed} where the seed equals the count of assigned segments in that shot.
    """
    by_shot: dict[int, list] = {}
    for t in prior_tracks or []:
        s = int(getattr(t, "shot_id", 0))
        by_shot.setdefault(s, []).append(t)

    seeds: dict[int, int] = {}
    for s, tracks in by_shot.items():
        # Sort tracks by the persisted first-seen order
        tracks.sort(key=lambda tr: track_order.get((s, int(getattr(tr, "track_id", -1))), 1 << 30))
        # Assign deterministic segment ids
        for idx, tr in enumerate(tracks):
            setattr(tr, "segment_id", idx)
        seeds[s] = len(tracks)
    return seeds

def _validate_shots_are_absolute_and_increasing(shots):
    if not shots:
        return
    # Absolute: shot 0 must start at 0 (or ≥ 0) and later shots must NOT restart at 0
    later_all_zero = len(shots) > 1 and all(s.get("first_frame", None) == 0 for s in shots[1:])
    # Strictly increasing windows
    strictly_increasing = all(
        int(shots[i]["first_frame"]) > int(shots[i-1]["last_frame"])
        for i in range(1, len(shots))
    )

    if later_all_zero or not strictly_increasing:
        # Build a short diagnostic of the first few shots
        diag = ", ".join(
            f'#{i}[{int(s["first_frame"])},{int(s["last_frame"])}]'
            for i, s in enumerate(shots[:5])
        )
        raise ResumeSafetyError(
            "Shots must be in ABSOLUTE frame space and strictly increasing. "
            "Detected per-shot relative indexing and/or non-monotone ranges. "
            f"Example windows: {diag}. "
            "Ensure the shot detector writes global first_frame/last_frame (no reset to 0 between shots)."
        )
    
def _backfill_preanchor_crops_for_seeded_tracks(
    seeded_tracks: List[FaceTrack],
    *,
    frame_provider: FrameProvider,
    detector: FaceDetector,
    align_fn,  # e.g., align_face_for_arcface
    crops_root: Path,
    shot_number: int,
    shot_first: int,
    anchor_abs: int,
    iou_thresh: float = 0.5,
) -> int:
    """
    For each DET observation with frame_idx <= anchor_abs and missing aligned_face,
    (1) try to re-detect on that frame and match by IOU; use landmarks to align;
    (2) else fall back to simple bbox crop+resize to 112x112 (last resort).
    Archive PNG and set obs.aligned_face and obs.crop_ref.
    Returns number of crops created.
    """
    made = 0
    # collect unique frames to process
    frames_needed = set()
    obs_by_frame: Dict[int, List[tuple[FaceObservation, int]]] = {}
    for tr in seeded_tracks or []:
        for o in getattr(tr, "observations", []) or []:
            if getattr(o, "source", None) != Source.DETECTED:
                continue
            if int(o.frame_idx) > int(anchor_abs):
                continue
            if (getattr(o, "aligned_face", None) is not None) or (getattr(o, "crop_ref", None) is not None):
                continue
            frames_needed.add(int(o.frame_idx))
            obs_by_frame.setdefault(int(o.frame_idx), []).append((o, int(getattr(tr, "track_id", -1))))

    for f in sorted(frames_needed):
        # fetch exact frame f (absolute index)
        frame_provider.reset_to_frame(f)
        frame = frame_provider.next()
        if frame is None:
            continue

        # run detection to get landmarks; match to each obs by IOU
        dets = detector.detect_faces_in_frame(frame) or None
        boxes, lmks, confs = (dets if dets else ([], [], []))

        for o, tid in obs_by_frame.get(f, []):
            # best IOU match box->o.bbox
            best_iou, best_j = 0.0, None
            for j, b in enumerate(boxes):
                bb = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                iou = compute_iou(bb, tuple(int(v) for v in o.bbox[:4]))
                if iou > best_iou:
                    best_iou, best_j = iou, j

            aligned = None
            if best_j is not None and best_iou >= iou_thresh:
                try:
                    aligned = align_fn(frame, lmks[best_j], f, source="resume-backfill")
                except Exception:
                    aligned = None

            # last-resort: bbox crop+resize (still beats having no embedding)
            if aligned is None:
                try:
                    x1,y1,x2,y2 = [int(v) for v in o.bbox[:4]]
                    x1,y1 = max(x1,0), max(y1,0)
                    crop = frame[y1:y2, x1:x2]
                    if crop is not None and crop.size:
                        # resize to 112x112
                        import cv2
                        aligned = cv2.resize(crop, (112,112), interpolation=cv2.INTER_LINEAR)
                        # convert BGR->RGB if needed for the embedder
                        aligned = aligned[:, :, ::-1]
                except Exception:
                    aligned = None

            if aligned is None:
                continue

            try:
                rel = _save_crop_for_obs(crops_root, int(shot_number), f, tid, aligned)
                setattr(o, "aligned_face", aligned)
                setattr(o, "crop_ref", rel)
                made += 1
            except Exception:
                logging.exception("backfill: failed to archive crop for frame=%d tid=%d", f, tid)

    return made

def _debug_det_embed_event(
    log,
    *,
    stage: str,            # 'detect' | 'attach' | 'checkpoint'
    shot: int,
    frame: int | None,
    tid: int | None,
    det_idx: int | None,   # index within the detections on that frame (for 'detect' stage), else None
    has_crop: bool | None, # True if aligned_face/crop_ref present
    aligned_ok: bool | None,
    emb_ok: bool | None,   # True if we actually have a vector (post-attach), else False/None
    emb_idx: int | None,   # final sidecar index; only known inside checkpoint.add_embeddings
    reason: str | None = None
):
    """
    Single-line, grep-friendly log for any DET row’s lifecycle.
    Examples:
      stage=detect:    EMB-EVENT stage=detect shot=1 frame=180 tid=- det=0 has_crop=1 aligned_ok=1 emb_ok=0 emb_idx=-1 reason=ok
      stage=attach:    EMB-EVENT stage=attach shot=1 frame=180 tid=3  det=- has_crop=1 aligned_ok=1 emb_ok=1 emb_idx=-1 reason=ok
      stage=checkpoint:EMB-EVENT stage=checkpoint shot=1 frame=180 tid=3  det=- has_crop=1 aligned_ok=1 emb_ok=1 emb_idx=42 reason=ok
    """
    log.info(
        "EMB-EVENT stage=%s shot=%s frame=%s tid=%s det=%s has_crop=%s aligned_ok=%s emb_ok=%s emb_idx=%s reason=%s",
        stage,
        (str(shot) if shot is not None else "-"),
        (str(frame) if frame is not None else "-"),
        (str(tid) if tid is not None else "-"),
        (str(det_idx) if det_idx is not None else "-"),
        int(bool(has_crop)) if has_crop is not None else -1,
        int(bool(aligned_ok)) if aligned_ok is not None else -1,
        int(bool(emb_ok)) if emb_ok is not None else -1,
        (-1 if emb_idx is None else int(emb_idx)),
        (reason or "ok"),
    )

def _build_emb_lookups_for_checkpoint(
    checkpoint: TrackingCheckpoint | None,
    *,
    anchor_frame: int,
):
    """
    Build emb_lookup/emb_array_lookup callables for resume_rehydrate.rehydrate_tracks.

    emb_lookup(shot, tid) returns (frames, embs) for DET observations strictly
    before the anchor frame, using the checkpoint's sidecar accessors:
      - get_det_frames_for_track(shot, tid, frame_max=anchor-1)
      - get_embeddings_by_frames(shot, frames)
    """
    if checkpoint is None or anchor_frame <= 0:
        return None, None

    def emb_lookup(shot: int, tid: int):
        try:
            frames = checkpoint.get_det_frames_for_track(
                int(shot), int(tid), frame_max=int(anchor_frame - 1)
            )
        except Exception:
            return None

        if not frames:
            return None

        try:
            embs = checkpoint.get_embeddings_by_frames(int(shot), frames)
        except Exception:
            return None

        if embs is None or len(embs) != len(frames):
            # Let resume_rehydrate.attach_embeddings enforce strict parity.
            return None

        return frames, embs

    # We intentionally disable emb_array_lookup to avoid any "count only" matching.
    return emb_lookup, None


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
        (_ckpt_run_root(checkpoint) if checkpoint else None),
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
            # Require the canonical FrameProvider interface:
            # props: fps, size, total_frames; methods: reset_to_frame(i), next()
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
        segment_id_seed_by_shot: Dict[int, int] = {}
        trackid_seed_by_shot: Dict[int, int] = {}
        prior_tracks: List[FaceTrack] = []
        # For exact resume parity: reuse the last pre-anchor track id on the
        # first detection inside the anchor-containing shot.
        reuse_tid_for_first_shot: Optional[int] = None
        reuse_binding_used: bool = False

        # ---------- Single, explicit resume arbiter ----------
        def _resolve_anchor() -> int:
            if not (resume_enabled and checkpoint):
                logging.info("resume: disabled or no checkpoint -> anchor=0")
                return 0
            # (1) explicit tuple from get_resume_anchor()
            try:
                anchors = checkpoint.get_resume_anchor()
                if anchors is not None and len(anchors) >= 1:
                    logging.info("resume: using get_resume_anchor() -> %r", anchors)
                    return int(anchors[0])
            except Exception:
                pass
            # (2) prefer read_status() → status.json
            try:
                rs = getattr(checkpoint, "read_status", None)
                if callable(rs):
                    status = rs() or {}
                    val = status.get("last_detection_frame")
                    if val is not None:
                        logging.info("resume: using read_status()['last_detection_frame'] -> %r", val)
                        return int(val)
            except Exception:
                pass
            # (3) direct status.json path if exposed
            try:
                status_path = getattr(checkpoint, "status_path", None)
                if status_path:
                    import os, json
                    if os.path.exists(status_path):
                        with open(status_path, "r") as f:
                            status = json.load(f) or {}
                        val = status.get("last_detection_frame")
                        if val is not None:
                            logging.info("resume: using status.json file -> %r", val)
                            return int(val)
            except Exception:
                pass
            # (4) legacy callable/attr last_detection_frame
            try:
                candidate = getattr(checkpoint, "last_detection_frame", None)
                val = candidate() if callable(candidate) else candidate
                if val is not None:
                    logging.info("resume: using legacy last_detection_frame -> %r", val)
                    return int(val)
            except Exception:
                pass
            # (5) max observed frame from the obs collector
            try:
                collector = getattr(checkpoint, "obs_collector", None)
                if collector is not None and hasattr(collector, "get_all_frame_indices"):
                    frames_seen = collector.get_all_frame_indices()
                    if frames_seen is not None and len(frames_seen) > 0:
                        max_frame = int(np.max(frames_seen))
                        logging.info("resume: using obs_collector max frame -> %d", max_frame)
                        return max_frame
            except Exception:
                pass
            logging.info("resume: no anchor inputs -> anchor=0")
            return 0

        resume_abs_frame: int = _resolve_anchor()

        # === AUDIT FENCE #0: record collector rows BEFORE any new work ===
        rows_before = None
        try:
            if checkpoint and hasattr(checkpoint, "obs_collector"):
                rows_before = int(checkpoint.obs_collector.count())
                logging.info("resume: obs_collector rows BEFORE processing=%d", rows_before)
        except Exception:
            logging.exception("resume: failed to read obs_collector.count() before")
        #############----------------##################

        # Identify anchor-containing shot BEFORE any trimming (used for logging)
        anchor_shot_idx = _shot_idx_by_abs_frame(shots, resume_abs_frame) if resume_abs_frame > 0 else None
        anchor_shot_num = (
            shots[anchor_shot_idx]["shot_number"]
            if (anchor_shot_idx is not None and anchor_shot_idx < len(shots))
            else None
        )

        # --- AUDIT: verify pre-anchor DET rows have embeddings in sidecar ---
        _audit_preanchor_embedding_parity(
            checkpoint,
            shots=shots,
            anchor_frame=resume_abs_frame,
            anchor_shot=anchor_shot_num
        )

        # Compute completed shots (fully before anchor) BEFORE trimming
        completed_shot_nums = {
            s["shot_number"] for s in shots
        } if resume_abs_frame == 0 else {
            s["shot_number"] for s in shots if s["last_frame"] < resume_abs_frame
        }

        # Rehydrate BEFORE the anchor (if any)
        obs_for_rehydrate = getattr(checkpoint, "obs_collector", None)
        if obs_for_rehydrate is not None and resume_abs_frame > 0:
            # Build embedding lookups from the checkpoint sidecars:
            #   - (shot, tid) -> list of DET frames <= anchor-1
            #   - (shot, frames) -> (K,512) embedding array
            emb_lookup, emb_array_lookup = _build_emb_lookups_for_checkpoint(
                checkpoint,
                anchor_frame=resume_abs_frame,
            )
            prior_tracks = rehydrate_tracks(
                obs_for_rehydrate,
                frame_max=resume_abs_frame - 1,
                track_order=(checkpoint.get_track_order() or {}),
                emb_lookup=emb_lookup,
                emb_array_lookup=emb_array_lookup,  # keep None if you only want frame-aware lookup
                anchor_shot_id=anchor_shot_num,
                strict=True,                         # fail fast if any pre-anchor track lacks embs
            )
        elif obs_for_rehydrate is not None:
            # Anchor==0 → this is effectively a cold start; we skip pre-anchor rehydrate.
            logging.info("resume: anchor=0 -> cold start; skipping pre-anchor rehydration")
            prior_tracks = []
        else:
            logging.info("resume: no obs_collector on checkpoint; skipping pre-anchor rehydration")
            prior_tracks = []

        # Hard guard:
        #   - For **completed shots** (< anchor shot): every pre-anchor DET must have an embedding.
        #   - For both completed shots and the **anchor shot**: every pre-anchor DET must have a crop_ref.
        #     (We rely on crops in the anchor shot to recompute embeddings during the resume run.)
        for t in prior_tracks or []:
            shot_id = int(getattr(t, "shot_id", -1))
            is_completed_shot = shot_id in completed_shot_nums
            is_anchor_shot = (anchor_shot_num is not None and shot_id == int(anchor_shot_num))

            # Ignore any unexpected "future" shots defensively.
            if not (is_completed_shot or is_anchor_shot):
                continue

            det_cnt = sum(
                1
                for o in (getattr(t, "observations", []) or [])
                if getattr(o, "source", None) is Source.DETECTED
                and int(o.frame_idx) <= int(resume_abs_frame - 1)
            )
            emb_cnt = len(getattr(t, "embeddings", []) or [])

            # Only completed shots are required to have strict DET↔EMB parity.
            if is_completed_shot and det_cnt > 0 and emb_cnt != det_cnt:
                raise ResumeSafetyError(
                    f"rehydrate: pre-anchor embedding parity failed for (shot={shot_id}, "
                    f"tid={int(getattr(t,'track_id',-1))}): DET={det_cnt} vs EMB={emb_cnt}"
                )

            # For both completed-shots and anchor-shot, every pre-anchor DET must have a crop_ref.
            for o in getattr(t, "observations", []) or []:
                if (
                    getattr(o, "source", None) is Source.DETECTED
                    and int(o.frame_idx) <= int(resume_abs_frame - 1)
                ):
                    if getattr(o, "crop_ref", None) in (None, "", 0):
                        raise ResumeSafetyError(
                            "rehydrate: missing crop_ref for pre-anchor DET "
                            f"(shot={shot_id}, tid={int(getattr(t,'track_id',-1))}, frame={int(o.frame_idx)})"
                        )

        if checkpoint and hasattr(checkpoint, "compare_rehydrate_to_snapshot"):
            checkpoint.compare_rehydrate_to_snapshot(
                prior_tracks=prior_tracks,
                anchor_frame=resume_abs_frame
            )

        # -------------------------------
        # Split rehydrated tracks:
        #   - pre-anchor shots -> KEEP in outputs now
        #   - anchor-containing shot -> ONLY seed aggregator (do NOT append to outputs here)
        # -------------------------------
        prior_tracks_completed: List[FaceTrack] = []
        prior_tracks_anchor: List[FaceTrack] = []
        for t in prior_tracks or []:
            s = int(getattr(t, "shot_id", -1))
            if s in completed_shot_nums:
                prior_tracks_completed.append(t)
            elif s == (anchor_shot_num if anchor_shot_num is not None else -1):
                prior_tracks_anchor.append(t)
            else:
                # Future shots should not exist pre-anchor; be defensive and ignore.
                logging.debug("rehydrate: ignoring pre-anchor track from unexpected shot=%s", s)

        # Append only completed-shot tracks now; anchor-shot tracks will be continued by the aggregator.
        if prior_tracks_completed:
            logging.info("resume: adding %d completed pre-anchor tracks to outputs", len(prior_tracks_completed))
            all_tracks.extend(prior_tracks_completed)



        # Log by-shot counts & seeds we computed
        try:
            by_shot_counts = {}
            for track in prior_tracks:
                shot = int(getattr(track, "shot_id", -1))
                by_shot_counts[shot] = by_shot_counts.get(shot, 0) + 1
            logging.info("resume: rehydrated counts by shot: %r", by_shot_counts)
            logging.info("resume: segment_id_seed_by_shot: %r", {int(k): int(v) for k,v in (segment_id_seed_by_shot or {}).items()})
            logging.info("resume: trackid_seed_by_shot: %r", {int(k): int(v) for k,v in (trackid_seed_by_shot or {}).items()})
        except Exception:
            logging.exception("resume: failed summarizing seeds")

        # Assign stable segment_ids to the rehydrated tracks and compute per-shot seeds
        try:
            track_order_map = checkpoint.get_track_order() if hasattr(checkpoint, "get_track_order") else {}
            segment_id_seed_by_shot = _assign_segment_ids_for_rehydrated(prior_tracks, track_order_map or {})
            # Compute next track_id seed per shot = max(track_id)+1 seen in that shot
            tmp: Dict[int, int] = {}
            # Also compute, per shot, the *last* observed track (by last frame) so we
            # can reuse that exact track_id on the first post-resume detection.
            last_tid_by_shot: Dict[int, int] = {}
            last_frame_by_shot_tid: Dict[Tuple[int, int], int] = {}

            for track in prior_tracks:
                shot = int(getattr(track, "shot_id", 0))
                tid = int(getattr(track, "track_id", -1))
                if tid >= 0:
                    tmp[shot] = max(tmp.get(shot, -1), tid)
                    # Track the last frame seen for (shot, tid) and pick the tid with the max last frame.
                    if getattr(track, "observations", None):
                        last_frame = max(o.frame_idx for o in track.observations)
                        key = (shot, tid)
                        if key not in last_frame_by_shot_tid or last_frame > last_frame_by_shot_tid[key]:
                            last_frame_by_shot_tid[key] = last_frame
            trackid_seed_by_shot = {shot: (max_tid + 1) for shot, max_tid in tmp.items()}
            # Reduce last_frame_by_shot_tid -> last_tid_by_shot (argmax over tid per shot)
            for (shot, tid), last_frame in last_frame_by_shot_tid.items():
                if (shot not in last_tid_by_shot) or last_frame > last_frame_by_shot_tid.get((shot, last_tid_by_shot[shot]), -1):
                    last_tid_by_shot[shot] = tid                
            logging.info("resume: assigned segment_ids to %d rehydrated tracks; seeds=%s",
                            len(prior_tracks), {k: int(v) for k, v in segment_id_seed_by_shot.items()})
        except Exception:
            logging.exception("resume: failed to assign segment_ids to rehydrated tracks; continuing with empty seeds")
            segment_id_seed_by_shot = {}
            trackid_seed_by_shot = {}

        if checkpoint and hasattr(checkpoint, "_validate_resume_embeddings"):
            checkpoint._validate_resume_embeddings(anchor_shot=anchor_shot_num)
            
        logging.info(
            "resume: prior_tracks strictness OK (anchor_shot=%s); rehydrated=%d (completed=%d, anchor-shot=%d)",
            anchor_shot_num, len(prior_tracks or []),
            len(prior_tracks_completed), len(prior_tracks_anchor)
        )

        # Trim shots so the first processed shot is the one containing the anchor (always)
        start_shot_idx = _shot_idx_by_abs_frame(shots, resume_abs_frame)
        if start_shot_idx >= len(shots):
            raise ResumeSafetyError("resume anchor beyond last shot; aborting for safety.")
        shots = shots[start_shot_idx:]

        # Track whether this invocation is a true resume (anchor > shot_first)
        _is_resume = bool(resume_abs_frame > 0)

        # Determine which shot contains the anchor and pre-compute its reuse tid.
        first_processed_shot_number = shots[0]["shot_number"]
        if checkpoint and resume_abs_frame > 0:
            try:
                # Prefer the tid whose last observation is closest to the anchor (already computed above).
                reuse_tid_for_first_shot = last_tid_by_shot.get(int(first_processed_shot_number))  # type: ignore[name-defined]
                if reuse_tid_for_first_shot is not None:
                    logging.info("resume: will reuse tid=%d for first detection in shot=%d",
                                 int(reuse_tid_for_first_shot), int(first_processed_shot_number))
            except Exception:
                reuse_tid_for_first_shot = None

            logging.info(
                "resume: anchor=%d anchor_shot_num=%s first_processed_shot_number=%s reuse_tid_for_first_shot=%s",
                resume_abs_frame, anchor_shot_num, first_processed_shot_number, reuse_tid_for_first_shot
            )

        for shot_idx, shot_num in enumerate(shots):
            shot_number = shot_num["shot_number"]
            first = shot_num["first_frame"]
            last = shot_num["last_frame"]
            # If resuming, start from the anchor frame inside the first shot; otherwise from shot start.
 
            if shot_idx == 0:
                start_at = max(first, resume_abs_frame)
            else:
                start_at = first
            if shot_idx == 0:
                # === AUDIT FENCE #1: make the loop range explicit & fail fast on empty range ===
                logging.info(
                    "resume: first_new_frame=%d (shot=[%d..%d]) detect_interval=%d mod=%d",
                    start_at, first, last, detect_interval, ((start_at - first) % max(1, detect_interval)),
                )
                if start_at > last:
                    raise ResumeSafetyError(
                        f"empty work-range at resume: start_at={start_at} > shot_last={last} for shot={shot_number}"
                    )

            # Determine seeded prior tracks only when resuming and 
            # for the first processed shot (anchor-containing shot)
            seeded_tracks = None
            if (shot_idx == 0) and (resume_abs_frame > 0) and prior_tracks:
                # Seed only the anchor-containing shot with its pre-anchor rehydrated tracks
                seeded_tracks = [
                    copy.deepcopy(t) for t in (prior_tracks_anchor if 'prior_tracks_anchor' in locals() else [])
                        if int(getattr(t, "shot_id", -1)) == int(shot_number)
                ]

            # --- shot-level diagnostics ---
            shot_det_raw_total = 0          # number of raw det boxes this shot
            shot_det_aligned_total = 0      # number of dets that produced aligned_face this shot

            if seeded_tracks:
                aggregator = ShotFaceTrackAggregator(
                    shot_number=shot_number,
                    iou_threshold=iou_thresh,
                    embedding_threshold=embedding_thresh,
                    prior_tracks=seeded_tracks,
                    resume_abs_frame=start_at,
                    next_tid_seed=int(trackid_seed_by_shot.get(int(shot_number), 0)),
                )
                checkpoint.hydrate_open_tracks_into(aggregator)

                _dump_agg_state("RESUME-AFTER-REHYDRATE", aggregator)
            else:
                aggregator = ShotFaceTrackAggregator(
                    shot_number=shot_number,
                    iou_threshold=iou_thresh,
                    embedding_threshold=embedding_thresh,
                )

            if (shot_idx == 0) and (resume_abs_frame > 0) and (reuse_tid_for_first_shot is not None):
                try:
                    aggregator.set_resume_force_tid(int(reuse_tid_for_first_shot))
                    logging.info("resume: aggregator will force tid=%d on the next created track in shot=%d",
                                int(reuse_tid_for_first_shot), int(shot_number))
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
                        last_bbox
                    )        

            seed_tid = int(trackid_seed_by_shot.get(int(shot_number), 0))
            seg_seed = int(segment_id_seed_by_shot.get(int(shot_number), 0)) if segment_id_seed_by_shot else 0

            logging.info(
                "shot=%d init: aggregator.next_track_id=%d, seed_tid=%d, seg_seed=%d",
                int(shot_number),
                int(getattr(aggregator, "next_track_id", -1)),
                seed_tid,
                seg_seed,
            )

            # Seed the aggregator’s next track_id only if we did not already pass next_tid_seed.
            if not seeded_tracks and seed_tid > 0:
                if hasattr(aggregator, "set_track_id_seed") and callable(getattr(aggregator, "set_track_id_seed")):
                    aggregator.set_track_id_seed(seed_tid)
                elif hasattr(aggregator, "next_track_id"):
                    setattr(aggregator, "next_track_id", seed_tid)
                elif hasattr(aggregator, "_next_track_id"):
                    setattr(aggregator, "_next_track_id", seed_tid)

            face_tracker = FaceTracker(tracker_type="CSRT")
            tracker_active = False

            shot_frames: List[np.ndarray] = []
            validator = TrackerValidator(
                frames=shot_frames, 
                first_frame_idx=start_at, 
                params=ValidatorParams(iou_thresh=iou_thresh))

            frame_provider.reset_to_frame(int(start_at))
    
            for frame_idx in range(start_at, last + 1):
                frame = frame_provider.next()
                if frame is None:
                    # If we're resuming, failing to land exactly at the anchor is unsafe.
                    if _is_resume and frame_idx == start_at:
                        raise ResumeSafetyError(
                            f"frame provider failed to seek to start_at={start_at} for shot={shot_number} "
                            f"(first={first}, last={last})."
                        )
                    # Cold start or transient hiccup: quick retry, then skip this frame.
                    frame = frame_provider.next()
                    if frame is None:
                        logging.warning(
                            "frame_provider returned None at frame %d (shot=%d); skipping frame.",
                            frame_idx, shot_number
                        )
                        continue

                # Log once to prove we actually enter the processing loop post-anchor
                if frame_idx == start_at:
                    logging.info("ENTER processing loop at frame=%d (anchor=%d)", frame_idx, resume_abs_frame)

                # Guardrail: never emit pre-anchor frames
                if resume_abs_frame and frame_idx < resume_abs_frame:
                    raise ResumeSafetyError(f"Illegal rewind: got frame_idx={frame_idx} < anchor={resume_abs_frame}")

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

                    try:
                        if checkpoint:
                            checkpoint.checkpoint_now(
                                frame_idx=frame_idx, 
                                shot_number=shot_number,
                                aggregator=aggregator,
                                shot_first_frame=first,
                                note=f"detect@{frame_idx}",
                            )

                    except Exception:
                        logging.exception("checkpoint: failed to persist at detect frame %s", frame_idx)                       
    
                    detections = detector.detect_faces_in_frame(frame)
                    
                    if detections:
                        boxes, landmark_lists, confidences = detections
                        raw_count = len(boxes)
                        aligned_ok = 0

                        # --- deterministically order detections to stabilize which track is "created first" ---
                        order = sorted(
                            range(len(boxes)),
                            key=lambda i: (
                                int(boxes[i][0]), int(boxes[i][1]),
                                int(boxes[i][2]), int(boxes[i][3]),
                                -float(confidences[i])
                            )
                        )
                        boxes          = [boxes[i] for i in order]
                        landmark_lists = [landmark_lists[i] for i in order]
                        confidences    = [confidences[i] for i in order]

                        for det_idx, (box, landmarks, confidence) in enumerate(zip(boxes, landmark_lists, confidences)):
                            bbox = tuple(int(v) for v in box[:4])  # detector returns XYXY
                            aligned_face = align_face_for_arcface(frame, landmarks, frame_idx, source="detect")

                            if aligned_face is not None:
                                aligned_ok += 1
                                # ---- DEBUG: detection produced a crop; no embedding yet at this stage ----
                                _debug_det_embed_event(
                                    logger, stage="detect",
                                    shot=int(shot_number), frame=int(frame_idx), tid=None, det_idx=int(det_idx),
                                    has_crop=True, aligned_ok=True, emb_ok=False, emb_idx=None, reason="ok"
                                )

                                observations.append(FaceObservation(
                                    frame_idx=frame_idx,
                                    bbox=bbox,
                                    embedding=None,
                                    confidence=float(confidence),
                                    aligned_face=aligned_face,
                                    source=Source.DETECTED
                                ))
                            else:
                                # ---- DEBUG: alignment failed -> this DET row must NOT be persisted or used to start a track ----
                                _debug_det_embed_event(
                                    logger, stage="detect",
                                    shot=int(shot_number), frame=int(frame_idx), tid=None, det_idx=int(det_idx),
                                    has_crop=False, aligned_ok=False, emb_ok=False, emb_idx=None, reason="align_fail"
                                )
                            
                        shot_det_raw_total     += int(raw_count)
                        shot_det_aligned_total += int(aligned_ok)  

                    else:
                        # no faces this frame; close shot-local tracks
                        aggregator.finalize_tracks()    
    

                # If this is the anchor-containing shot, and we haven't yet bound the
                # first post-resume detection, temporarily override the allocator so the
                # very first created track reuses the pre-anchor tid (exact parity).
                if (
                    need_detect
                    and not reuse_binding_used
                    and reuse_tid_for_first_shot is not None
                    and int(shot_number) == int(first_processed_shot_number)
                    and observations  # only if we actually detected faces
                ):
                    try:
                        # Force the next allocated id to be the reused tid.
                        if hasattr(aggregator, "set_track_id_seed") and callable(getattr(aggregator, "set_track_id_seed")):
                            aggregator.set_track_id_seed(int(reuse_tid_for_first_shot))
                        elif hasattr(aggregator, "next_track_id"):
                            setattr(aggregator, "next_track_id", int(reuse_tid_for_first_shot))
                        elif hasattr(aggregator, "_next_track_id"):
                            setattr(aggregator, "_next_track_id", int(reuse_tid_for_first_shot))
                        logging.info("resume: temporarily overriding next_track_id -> %d (shot=%d)",
                                     int(reuse_tid_for_first_shot), int(shot_number))
                    except Exception:
                        logging.exception("resume: failed to set temporary allocator for reuse tid; continuing without binding")

                if observations and observations[0].source == Source.DETECTED:
                    # Expand detection list for logging
                    dets = []
                    for ob in observations:
                        x1, y1, x2, y2 = [int(v) for v in ob.bbox[:4]]
                        dets.append({
                            "bbox": (x1, y1, x2, y2),
                            "conf": float(ob.confidence) if ob.confidence is not None else None
                        })

                det_count = sum(1 for o in observations if getattr(o, "source", None) == Source.DETECTED)
                logging.info("DETECTION frame=%d shot=%d raw-detections=%d", frame_idx, int(shot_number), det_count)

                if observations:
                    det_cnt = sum(1 for o in observations if getattr(o, "source", None) == Source.DETECTED)
                    if det_cnt:
                        # Before assignment: track_id may be None; we just log bbox/intention
                        bboxes = [
                            tuple(int(v) for v in o.bbox[:4])
                            for o in observations if getattr(o, "source", None) == Source.DETECTED
                        ]
                        logging.info(
                            "DETECT-EMIT shot=%d frame=%d det_n=%d boxes=%s",
                            int(shot_number), int(frame_idx), int(det_cnt), bboxes[:6]
                        )

                # Add current frame observations to aggregator
                created_count = aggregator.update_tracks_with_frame(frame_idx, observations)

                if checkpoint:
                    try:
                        # Pull what we just persisted (by frame) in the obs collector if supported.
                        # Fallback: use aggregator view and log crop presence.
                        det_persisted = []
                        if hasattr(checkpoint, "obs_collector") and hasattr(checkpoint.obs_collector, "rows_for_frame"):
                            # rows_for_frame(shot, frame) -> iterable of row dicts
                            for r in checkpoint.obs_collector.rows_for_frame(shot_number, frame_idx):
                                if int(r.get("src", -1)) == int(Source.DETECTED.value):
                                    det_persisted.append({
                                        "tid": int(r.get("tid", -1)),
                                        "emb_idx": int(r.get("emb_idx", -1)),
                                        "has_crop": 1 if r.get("crop_ref") else 0,
                                    })
                        else:
                            # aggregator fallback: we won't have emb_idx, but we can see crops
                            for ob in aggregator.observations_at(frame_idx, source=Source.DETECTED, require_track_id=True):
                                det_persisted.append({
                                    "tid": int(getattr(ob, "track_id", -1)),
                                    "emb_idx": -1,  # unknown here
                                    "has_crop": 1 if (getattr(ob, "aligned_face", None) is not None or getattr(ob, "crop_ref", None)) else 0,
                                })
                        if det_persisted:
                            logging.info("DETECT-PERSIST shot=%d frame=%d rows=%s", int(shot_number), int(frame_idx), det_persisted)
                    except Exception:
                        logging.exception("resume-log: failed DETECT-PERSIST probe")

                _dump_agg_state(f"AFTER FRAME {frame_idx}", aggregator, frame=frame_idx)

                # Mark binding as used exactly once on the first detection frame where it applied.
                if (
                    need_detect
                    and not reuse_binding_used
                    and reuse_tid_for_first_shot is not None
                    and int(shot_number) == int(first_processed_shot_number)
                    and created_count > 0
                ):
                    reuse_binding_used = True
                    logging.info("resume: reuse binding applied (shot=%d, reused tid=%d).",
                                 int(shot_number), int(reuse_tid_for_first_shot))

                agg_det = aggregator.observations_at(frame_idx, source=Source.DETECTED, require_track_id=True)
                agg_trk = aggregator.observations_at(frame_idx, source=Source.TRACKED,  require_track_id=True)
                logging.debug(
                    "POST-ASSIGN: frame=%d det_assigned=%d trk_assigned=%d created=%d open_tracks=%d",
                    frame_idx, len(agg_det), len(agg_trk), created_count,
                    sum(1 for t in aggregator.tracks if not t.is_closed()),
                )

                # Persist 112×112 crops to disk for DET observations on this frame
                if checkpoint:
                    det_obs = aggregator.observations_at(frame_idx, source=Source.DETECTED, require_track_id=True)
                    if det_obs:
                        crops_root = _ckpt_run_root(checkpoint)
                        _saved = 0
                        for ob in det_obs:
                            # Only save if aligned_face present (detection path) & no crop_ref yet
                            if getattr(ob, "aligned_face", None) is None:
                                continue
                            if getattr(ob, "crop_ref", None):
                                continue
                            try:
                                rel = _save_crop_for_obs(
                                    crops_root, shot_number, ob.frame_idx, ob.track_id, ob.aligned_face
                                )
                                setattr(ob, "crop_ref", rel) # so checkpoint.add_observations can persist the reference
                                _saved += 1
                            except Exception:
                                logging.exception("crop-archive: failed at frame=%d tid=%s", ob.frame_idx, ob.track_id)
                        logging.info(
                            "CROP-SAVED frame=%d shot=%d saved=%d",
                            frame_idx, int(shot_number), int(_saved)
                        )

                # Checkpoint observations (PERSISTENCE BOUNDARY)
                if checkpoint:
                    frame_obs_objs = aggregator.observations_at(frame_idx, require_track_id=True)
                    # At the anchor frame, some pipelines can momentarily lack track_ids.
                    # Retry *only at the anchor* without strictness so we at least persist those detections.
                    if (not frame_obs_objs) and (frame_idx == resume_abs_frame):
                        logging.info("resume: no require_track_id observations at anchor; retrying without strictness.")
                        frame_obs_objs = aggregator.observations_at(frame_idx, require_track_id=False)
                    if frame_obs_objs:
                        if PARANOID:
                            if frame_obs_objs:
                                logger.debug("seg: frame=%s shot=%s n_obs=%s first=%r",
                                             frame_idx, shot_number, len(frame_obs_objs),
                                             {"tid": frame_obs_objs[0].track_id,
                                              "src": getattr(frame_obs_objs[0], "source", None),
                                              "bbox": getattr(frame_obs_objs[0], "bbox", None)})
                        if not all(isinstance(o, FaceObservation) for o in frame_obs_objs):
                            bad = [type(o).__name__ for o in frame_obs_objs if not isinstance(o, FaceObservation)]
                            logger.error("seg: Non-FaceObservation in frame_obs_objs at frame=%s shot=%s types=%s sample=%r",
                                         frame_idx, shot_number, bad, frame_obs_objs[:1])
                            raise TypeError(f"frame_obs_objs must be FaceObservation objects, got {bad}")
                        if not all(isinstance(o.source, Source) for o in frame_obs_objs):
                            bad = [getattr(o, "source", None) for o in frame_obs_objs if not isinstance(getattr(o, "source", None), Source)]
                            logger.error("seg: Observation with non-enum source at frame=%s shot=%s bad=%r", frame_idx, shot_number, bad[:3])
                            raise TypeError(f"Observation.source must be Source enum; bad={bad[:3]}")


                        checkpoint.add_observations(shot_number, frame_idx, frame_obs_objs)                        # After observations are persisted, emit a human-readable snapshot.
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
                        logging.exception("checkpoint: snapshot write failed at detect frame %s", frame_idx)

                if created_count and checkpoint and hasattr(checkpoint, "on_new_tracks"):
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
            logging.info(
                "shot=%d summary: det_raw=%d det_aligned(crop|face)=%d",
                int(shot_number), int(shot_det_raw_total), int(shot_det_aligned_total)
            )

            # --- embedding diagnostics (shot scope) ---
            shot_embed_intended = 0   # total crops/aligned faces we intend to embed for this shot
            shot_embed_vectors  = 0   # total vectors returned by embedder for this shot

            for track in aggregator.tracks:
                crops = []
                frames_for_embed: list[int] = []

                # Prefer in-memory aligned_face; otherwise lazy-load archived crop
                for obs in track.observations:
                    if obs.aligned_face is not None:
                        crops.append(obs.aligned_face)
                        frames_for_embed.append(int(obs.frame_idx))

                        logging.info(f"shot: {shot_number}, track:{track}, obs.aligned_face not None and appended")
                        continue
                    cr = getattr(obs, "crop_ref", None)
                    if checkpoint and cr:
                        try:
                            abs_path = Path(_ckpt_run_root(checkpoint), cr)
                            if abs_path.exists():
                                img = Image.open(abs_path).convert("RGB")
                                crops.append(np.asarray(img))
                                frames_for_embed.append(int(obs.frame_idx))

                            logging.info(f"shot: {shot_number}, track:{track}, no obs.aligned_face but abs_path found and crop appended")
                        except Exception:
                            logging.exception("embed-load: failed to load crop %s", cr)
                if not crops:
                    logging.info(f"end of shot {shot_number} and no aligned faces found")
                    continue

                # Count the crops we plan to embed for this track
                track_intended = len(crops)
                shot_embed_intended += track_intended
                logging.info(
                    "embed-intent: shot=%d tid=%s crops_for_embed=%d",
                    int(shot_number), str(track.track_id), int(track_intended)
                )

                embs = embedder.get_embedding_batch(crops, batch_size=embedding_batch_size_max)
                if len(embs) != len(frames_for_embed):
                    raise RuntimeError(
                        f"Embed count/frame mismatch: embs={len(embs)} frames={len(frames_for_embed)} "
                        f"(shot={shot_number}, tid={track.track_id})"
                    )

                logging.info(
                    "embed-return: shot=%d tid=%s vectors=%d shape=%s dtype=%s",
                    int(shot_number), str(track.track_id),
                    0 if embs is None else int(len(embs)),
                    getattr(embs, "shape", None),
                    getattr(getattr(embs, "dtype", None), "name", None)
                )
                shot_embed_vectors += 0 if embs is None else int(len(embs))

                logging.info(
                    "embed-shot-summary: shot=%d intended=%d returned=%d",
                    int(shot_number), int(shot_embed_intended), int(shot_embed_vectors)
                )

                if not isinstance(embs, np.ndarray):
                    raise TypeError(f"Embedder must return np.ndarray, got {type(embs)}")
                if embs.ndim != 2 or embs.shape[1] != 512:
                    raise ValueError(f"Embedder returned invalid array shape {embs.shape}; expected (K,512)")
                if embs.dtype != np.float32:
                    embs = np.asarray(embs, dtype=np.float32, order="C")
                aggregator.attach_embeddings(track.track_id, embs)

                # Log outcome per DET observation in this track (emb_idx still unknown here).
                for ob in getattr(track, "observations", []) or []:
                    if getattr(ob, "source", None) is Source.DETECTED:
                        has_crop = (ob.aligned_face is not None) or bool(getattr(ob, "crop_ref", None))
                        # If attach_embeddings filled FaceObservation.embedding (your aggregator typically does),
                        # emb_ok reflects that; emb_idx is still unknown at this stage.
                        emb_ok = (getattr(ob, "embedding", None) is not None)
                        _debug_det_embed_event(
                            logger, stage="attach",
                            shot=int(shot_number), frame=int(ob.frame_idx), tid=int(getattr(ob, "track_id", -1)),
                            det_idx=None, has_crop=has_crop, aligned_ok=has_crop, emb_ok=emb_ok, emb_idx=None, reason=("ok" if emb_ok else "attach_miss")
                        )

                logging.info(f"end of shot {shot_number} and {len(embs)} embeddings attached to aggregator")

                if checkpoint and embs.size:
                    # Persist with the correct frame index for EACH vector to satisfy strict rehydrate.
                    for f_idx, vec in zip(frames_for_embed, embs):
                        checkpoint.add_embeddings(
                            int(shot_number),
                            int(track.track_id),
                            int(f_idx),
                            np.asarray(vec, dtype=np.float32).reshape(1, -1),
                        )
                    logging.info(f"end of shot {shot_number} and {len(embs)} per-frame embeddings added to checkpoint")
                else:
                    logging.info(f"end of shot {shot_number} and NO embeddings added to checkpoint")

            logging.info(
                "EMB-RECONCILE shot=%d intended=%d returned=%d note=post-embed-loop",
                int(shot_number), int(shot_embed_intended), int(shot_embed_vectors)
            )

            aggregator.finalize_tracks()

            # Assign segment_id per shot, seeded to continue numbering after any rehydrated tracks
            seed = int(segment_id_seed_by_shot.get(int(shot_number), 0))
            try:
                _ = aggregator.resolve_segment_ids(
                    segment_id_counter=seed,
                    embedding_threshold=embedding_thresh
                )
            except RuntimeError as e:
                logging.error("segment-id resolution skipped for shot=%d due to missing embeddings: %s",
                              int(shot_number), e)

            all_tracks.extend(aggregator.tracks)

            if checkpoint:
                checkpoint.on_shot_done()

        #end of video
        if checkpoint:
            # Persist live collectors to ckpt/obs_ckpt.npz and ckpt/emb_ckpt.npz
            checkpoint.finalize(note="final video flush")

            # Let downstream tools know this run is complete so they can read all rows
            # without trimming to the last detection anchor.
            checkpoint.mark_completed()

        return all_tracks
