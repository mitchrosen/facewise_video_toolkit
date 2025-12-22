from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import time
import tempfile
import typing as _t
import hashlib
from typing import Protocol, List, Optional, Any, Dict, Tuple
import numpy as np
from numpy.lib import recfunctions as rfn
import shutil
import logging

from facekit.errors import ResumeSafetyError
from facekit.tracking.face_structures import FaceObservation
from facekit.pipeline.track_order import (
    track_order_dict_to_list,
    track_order_list_to_dict,
    track_order_add,
    track_order_summary,
    TrackOrderError,
)
from facekit.utils.io import fsync_parent_dir
from facekit.utils.io import atomic_write_npz
from facekit.tracking.aggregator import ShotFaceTrackAggregatorProtocol
from facekit.utils.debug_snapshots import (
    load_latest_snapshot, 
    diff_snapshot_vs_rehydrate,
    write_snapshot_atomic,
)
from facekit.common.obs_consts import Source, SRC_TO_CODE, CODE_TO_SRC, src_to_code, code_to_src
from facekit.tracking.face_structures import FaceObservation

REQUIRED_SCHEMA_VERSION = "2.3"

class TrackingCheckpoint(Protocol):
    # lifecycle/progress
    def on_frame(self, frame_idx: int) -> None: ...
    def on_shot_done(self) -> None: ...
    def on_new_tracks(self, n: int) -> None: ...
    
    # Stable, persisted order for (shot_number, track_id) -> order_index.
    # Used to deterministically assign segment_ids to rehydrated tracks.
    def get_track_order(self) -> dict[tuple[int, int], int]: ...

    # Pre-detect checkpoint: anchor==this frame; resume rehydrates <anchor and starts at anchor
    def checkpoint_now(
            self, 
            *,
            frame_idx: int, 
            shot_number: int,
            aggregator: ShotFaceTrackAggregatorProtocol,
            shot_first_frame: int | None,
            note: str = "checkpoint") -> None: ...
    
    # persistence
    def add_observations(self, shot_number: int, frame_idx: int, observations: List["FaceObservation"]) -> int: ...
    def add_embeddings(self, shot_number: int, track_id: int, frame_idx_last: int, embs: np.ndarray) -> int: ...

    # resuming
    def get_resume_anchor(self) -> tuple[int, int, int] | None: ...     # Return (frame_idx, shot_number, shot_first_frame) or None for fresh run.

_PROTECTED_KEYS = (
    "video_path",
    "detector_model_path",
    "embedding_model_path",
    "yolo_config_path",
    "shot_segmentation_path",
    "detect_interval",
)

def _utcstamp_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def _paramhash8(snapshot: dict) -> str:
    s = "|".join(f"{k}={snapshot.get(k)}" for k in _PROTECTED_KEYS).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:8]

def _video_parent_dir(default_root: Path, video_path: Path) -> Path:
    # stable parent per *absolute* video path
    h = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return default_root / f"{video_path.stem}__{h}"

def _stat_for_log(p: Path) -> dict:
    try:
        s = p.stat()
        return {"exists": True, "size": s.st_size, "inode": getattr(s, "st_ino", None), "path": str(p)}
    except FileNotFoundError:
        return {"exists": False, "path": str(p)}

def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    logging.debug("ckpt._atomic_write_bytes: begin write → %s", dst)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dst.parent, suffix=".tmp", delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dst)
    try:
        fsync_parent_dir(dst)
    except Exception as e:
        logging.debug("ckpt._atomic_write_bytes: fsync_parent_dir failed (non-fatal): %s", e)

    # Read-back & log proof
    try:
        with open(dst, "rb") as f:
            _ = f.read(1)  # minimal read is enough to prove visibility
    except Exception:
        logging.exception("ckpt._atomic_write_bytes: post-write open/read failed")

    logging.info("ckpt.atomic_write post-write %s", _stat_for_log(dst))

def _atomic_write_text(dst: Path, text: str) -> None:
     _atomic_write_bytes(dst, text.encode("utf-8"))

def _dump_npz_atomic(collector, final_path: Path) -> None:
    """
    Ask a collector to dump to a temporary NPZ file and atomically swap it in.
    We fsync the temp file and the parent directory to make the checkpoint durable.
    NOTE: numpy.savez appends '.npz' if the provided filename does not end with '.npz'.
    Therefore our temp path MUST itself end with '.npz', or we'll miss the file we just wrote.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure tmp name ends with '.npz' to avoid NumPy appending another '.npz'
    # Example: 'obs_ckpt.npz' -> 'obs_ckpt.tmp.npz'
    tmp = final_path.with_name(final_path.stem + ".tmp.npz")

    # Let the collector write its NPZ to the temp path.
    collector.dump_npz(tmp)

    # Robustness: if the collector (or numpy) appended '.npz' again for some reason,
    # fall back to that file name. (Expected path is '...tmp.npz'; fallback is '...tmp.npz.npz')
    actual_tmp = Path(tmp)
    if not actual_tmp.exists():
        fallback = Path(str(tmp) + ".npz")
        if fallback.exists():
            actual_tmp = fallback
        else:
            raise FileNotFoundError(f"Checkpoint temp not written by collector: expected '{tmp}'")

    # Best-effort fsync of the tmp file (collector may have closed it).
    try:
        with open(actual_tmp, "rb") as fh:
            os.fsync(fh.fileno())
    except Exception:
        pass

    os.replace(actual_tmp, final_path)
    fsync_parent_dir(final_path)

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()
    except Exception:
        return None

def _assert_npz_keys(npz_path: Path, required_keys: tuple[str, ...], *, strict: bool = True) -> None:
    """
    Ensure an NPZ exists and contains specific top-level keys.
    In strict mode, raise ResumeSafetyError if violated; else log and continue.
    """
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            files = set(npz.files)
        missing = [k for k in required_keys if k not in files]
        if missing:
            msg = f"[ckpt] {npz_path.name} missing required arrays: {missing}; present={list(files)}"
            if strict:
                raise ResumeSafetyError(msg)
            logging.info("ckpt:non-strict: %s", msg)
    except FileNotFoundError:
        # The caller should already have checked existence; treat as structural failure in strict mode.
        msg = f"[ckpt] {npz_path} not found"
        if strict:
            raise ResumeSafetyError(msg)
        logging.info("ckpt:non-strict: %s", msg)
    except Exception as e:
        msg = f"[ckpt] failed to inspect {npz_path}: {e}"
        if strict:
            raise ResumeSafetyError(msg)
        logging.info("ckpt:non-strict: %s", msg)
        

@dataclass
class CheckpointStatus:
    """Small, human-readable status snapshot you can tail while jobs run."""
    last_saved_utc: str
    video_path: str
    # progress
    frames_done: int
    shots_done: int
    tracks_seen: int
    obs_rows: int
    emb_rows: int
    # restart anchor (detection boundary)
    last_detection_frame: int | None
    last_detection_shot: int | None
    last_detection_shot_first_frame: int | None
    obs_rows_at_last_detection: int
    emb_rows_at_last_detection: int
    # configuration snapshot (for safe resume)
    schema_version: str
    detect_interval: int
    embedding_batch_size_max: int
    device: str
    emb_store: str  # "inline" | "sidecar" | "none"
    emb_sidecar_path: str | None
    obs_sidecar_path: str | None
    detector_model_path: str
    embedding_model_path: str
    yolo_config_path: str
    shot_segmentation_path: str | None
    checkpoint_dir: str
    log_level: str
    log_file: str | None
    track_order: list[dict] | None = None
    open_tracks: list | None = None

    # misc
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class CheckpointManager(TrackingCheckpoint):
    """
    Checkpointing with safe resume:

    - At every point we call checkpoint_now(), we:
        * snapshot obs/emb collectors to NPZ
        * record exact row counts as the resume anchor
        * write status.json with all CLI params we must protect on resume
    - On shot boundaries, we update status.json (helpful for monitoring).
    - On finalize, we also write one last status.

    Resume:
      - Load status.json + NPZs.
      - Trim live collectors back to the point of checkpoint so re-running is deterministic.
    """
    @property
    def run_id(self) -> str:
        return self.root.name

    def __init__(self,
                 root_dir: _t.Union[str, Path],
                 *,
                 video_path: _t.Union[str, Path],
                 resume: bool = True) -> None:
        
        # Paths
        self.root = Path(root_dir)
        self.ckpt_dir  = Path(self.root, "ckpt")
        self.video_path = str(Path(video_path).resolve())
        self.status_path = Path(self.root, "status.json")

        logging.info("ckpt.paths: root=%s ckpt_dir=%s status_path=%s", self.root, self.ckpt_dir, self.status_path)


        self.obs_path    = Path(self.ckpt_dir, "obs_ckpt.npz")
        self.emb_path    = Path(self.ckpt_dir, "emb_ckpt.npz")

        # Create dirs when missing
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        self.resume_enabled = bool(resume)
        self._shot_track_to_order: dict[tuple[int,int], int] = {}
        self._next_track_order: int = 0
        self._anchor_frame: int | None = None

        # Counters
        self._frames_done = 0
        self._shots_done = 0
        self._tracks_seen = 0

        # Pre-detection cursor (resume starts *on* this frame)
        self._pending_det_shot: int | None = None
        self._pending_det_frame: int | None = None
        self._pending_det_shot_first: int | None = None
        self._pending_det_reason: str | None = None

        # Detection anchors
        self._last_det_frame: int | None = None
        self._last_det_shot: int | None = None
        self._obs_rows_at_det: int = 0
        self._emb_rows_at_det: int = 0
        self._last_det_shot_first_frame: int | None = None

        # Collector pointers (set via `start`)
        self._obs = None  # ObservationsCollector
        self._emb = None  # EmbeddingCollector

        # Inline, JSON representation of open tracks at the anchor.
        self._open_tracks_inline: list[dict] | None = None

        # Snapshot of CLI/options for safe resume (populated in start())
        self._cfg: dict[str, _t.Any] = {}

        self.logger = getattr(self, "logger", None) or logging.getLogger("facekit.checkpoint")

    # ---------- small helpers ----------
    def _is_pre_anchor(self, frame_idx: int) -> bool:
        a = self._anchor_frame
        return self.resume_enabled and a is not None and int(frame_idx) < int(a)

    # ---------- public API ----------

    def start(self,
            obs_collector,
            emb_collector,
            *,
            tracks_seen: int = 0,
            shots_done: int = 0,
            frames_done: int = 0,
            options_snapshot: dict[str, _t.Any] | None = None) -> None:
        """
        Bind collectors & counters and prepare track-order state.

        Rules:
        - If resuming and track_order is already in memory (e.g., restored by
            load_and_anchor_collectors), preserve it and advance _next_track_order.
        - Else, if resuming, rehydrate track_order from status.json (if present).
        - Else (fresh run), start with an empty track_order.
        """
        # bind collectors (private) and expose public aliases used by pipeline
        self._obs = obs_collector
        self._emb = emb_collector
        self.obs_collector = obs_collector
        self.emb_collector = emb_collector
        self._tracks_seen = int(tracks_seen)
        self._shots_done = int(shots_done)
        self._frames_done = int(frames_done)
        self._cfg = dict(options_snapshot or {})

        if self.resume_enabled:
            status = self.read_status() or {}

            # ---- Pre-hydrate detection anchors BEFORE the first write ----
            try:
                if self._last_det_frame is None:
                    last_det_frame = status.get("last_detection_frame")
                    if last_det_frame is not None:
                        self._last_det_frame = int(last_det_frame)
                if self._last_det_shot is None:
                    last_det_shot = status.get("last_detection_shot")
                    if last_det_shot is not None:
                        self._last_det_shot = int(last_det_shot)
                if self._last_det_shot_first_frame is None:
                    last_det_shot_first_frame = status.get("last_detection_shot_first_frame")
                    if last_det_shot_first_frame is not None:
                        self._last_det_shot_first_frame = int(last_det_shot_first_frame)
                if not self._obs_rows_at_det:
                    last_det_obs_rows = status.get("obs_rows_at_last_detection")
                    if last_det_obs_rows is not None:
                        self._obs_rows_at_det = int(last_det_obs_rows)
                if not self._emb_rows_at_det:
                    last_det_emb_rows = status.get("emb_rows_at_last_detection")
                    if last_det_emb_rows is not None:
                        self._emb_rows_at_det = int(last_det_emb_rows)
            except Exception:
                # Non-fatal: resume can still proceed without these hints.
                pass

            try:
                prev_open_tracks = status.get("open_tracks")
                if isinstance(prev_open_tracks, list) and self._open_tracks_inline is None:
                    self._open_tracks_inline = prev_open_tracks
            except Exception:
                pass

            if self._shot_track_to_order:
                # Already restored upstream (e.g., load_and_anchor_collectors) -> keep it.
                self._next_track_order = max(self._shot_track_to_order.values(), default=-1) + 1
                logging.debug(
                    "ckpt.start: preserving pre-populated track_order (entries=%d) next=%d",
                    len(self._shot_track_to_order), self._next_track_order
                )
            else:
                # Resume requested but nothing in memory yet -> rehydrate from status.json.
                shot_track_order_list = status.get("track_order") or []
                try:
                    self._shot_track_to_order, self._next_track_order = track_order_list_to_dict(
                        shot_track_order_list, strict=True
                    )
                    logging.info(
                        "ckpt.start: rehydrated track_order: %s",
                        track_order_summary(self._shot_track_to_order),
                    )
                except TrackOrderError as e:
                    logging.warning("ckpt.start: corrupt/legacy track_order; trying sidecar fallback (%s)", e)
                    self._shot_track_to_order = {}
                    self._next_track_order = 0

                # --- sidecar fallback (if collector exposes a stable order) ---
                if not self._shot_track_to_order:
                    try:
                        omap = getattr(self._obs, "_order_map", None)
                        if isinstance(omap, dict) and omap:
                            # install in order of appearance (value is order)
                            self._shot_track_to_order = {
                                (int(s), int(t)): int(o)
                                for (s, t), o in sorted(omap.items(), key=lambda kv: kv[1])
                            }
                            self._next_track_order = max(self._shot_track_to_order.values(), default=-1) + 1
                            logging.info("ckpt.start: installed track_order from sidecar (entries=%d)",
                                         len(self._shot_track_to_order))
                            # persist immediately so future resumes don't rely on fallback
                            self._write_status("track_order from sidecar")
                    except Exception:
                        logging.exception("ckpt.start: sidecar fallback for track_order failed")
        else:
            # Fresh run.
            self._shot_track_to_order = {}
            self._next_track_order = 0
            logging.debug("ckpt.start: initializing empty track_order (fresh run).")

        # --- Always write an initial status snapshot ---
        try:
            logging.debug("ckpt.start: writing initial status.json")
            self._write_status("checkpointing started")
        except Exception as e:
            logging.exception("ckpt.start: initial status write failed: %s", e)

    def on_new_tracks(self, n: int) -> None:
        self._tracks_seen += int(n)

    def on_shot_done(self) -> None:
        self._shots_done += 1
        # Ensure the anchor’s row counters include end-of-shot backfill (embeddings/landmarks links).
        try:
            self.refresh_anchor_row_counts_post_backfill()
        except Exception:
            logging.exception("ckpt:on_shot_done: refresh post-backfill failed (non-fatal)")
        # Flush collectors so emb_idx links & new embeddings are on disk for resume.
        try:
            if self._obs is not None:
                _dump_npz_atomic(self._obs, self.obs_path)
            if self._emb is not None:
                _dump_npz_atomic(self._emb, self.emb_path)
        except Exception:
            logging.exception("ckpt:on_shot_done: sidecar flush failed (non-fatal)")
        # Status snapshot now reflects refreshed counters and flushed sidecars.
        self._write_status("shot boundary (post-backfill + flush)")

    def refresh_anchor_row_counts_post_backfill(self) -> None:
        if self._last_det_frame is None or self._obs is None or self._emb is None:
            return
        try:
            if hasattr(self._obs, "to_array"):
                arr = self._obs.to_array()
                pre_anchor_obs = int((arr["f"] < int(self._last_det_frame)).sum())
                self._obs_rows_at_det = max(int(self._obs_rows_at_det or 0), pre_anchor_obs)
            else:
                self._obs_rows_at_det = max(int(self._obs_rows_at_det or 0), int(self._obs.count()))
        except Exception:
            self._obs_rows_at_det = max(int(self._obs_rows_at_det or 0), int(self._obs.count()))
        try:
            # end-of-shot embeddings are pre-anchor; safe to include
            self._emb_rows_at_det = max(int(self._emb_rows_at_det or 0), int(self._emb.count()))
        except Exception:
            pass
        self._write_status("post-shot backfill")
        
    def get_track_order(self) -> dict[tuple[int,int], int]:
        """
        Return the persisted (shot, track_id) -> first-seen order mapping used to stabilize segment labels.
        """
        return dict(self._shot_track_to_order) if hasattr(self, "_shot_track_to_order") else {}

    def on_frame(self, frame_idx: int) -> None:
        # Called for every processed frame (0-based)
        self._frames_done = int(frame_idx) + 1

    @staticmethod
    def _src_to_name(src_obj) -> str:
        if isinstance(src_obj, Source):
            return src_obj.value
        if isinstance(src_obj, str):
            return Source(src_obj.lower()).value
        raise TypeError(f"invalid observation source for in-RAM row: {src_obj!r}")
    

    @staticmethod
    def _obs_to_row_dict(shot: int, ob) -> dict:
        # Require a real FaceObservation with a Source
        from facekit.tracking.face_structures import FaceObservation
        if not isinstance(ob, FaceObservation):
            raise TypeError(f"_obs_to_row_dict requires FaceObservation, got {type(ob).__name__}")
        if not isinstance(ob.source, Source):
            raise TypeError(f"FaceObservation.source must be Source enum, got {ob.source!r}")
                            
        # bbox normalized to xyxy list
        if getattr(ob, "bbox", None) is None:
            raise ValueError("ob.bbox required")
        x1,y1,x2,y2 = [float(v) for v in ob.bbox[:4]]

       # Canonicalize and take the numeric code we persist to the sidecar.
        src_name = CheckpointManager._src_to_name(ob.source)  # 'detected' / 'tracked' / 'flow'
        src_code = int(src_to_code(src_name))  # <- persist numeric code, not string

        landmarks = getattr(ob, "landmarks", None)
        # Persist as a simple nested list (or empty)
        landmarks_val = landmarks.tolist() if isinstance(landmarks, np.ndarray) else (landmarks if landmarks is not None else [])

        # Confidence can legitimately be None; in that case use NaN (collector already handles NaN).
        conf_val = getattr(ob, "confidence", None)
        d = {
            "shot":       int(shot),
            "track_id":   int(getattr(ob, "track_id")),
            "f":          int(getattr(ob, "frame_idx")),
            "bbox_xyxy":  [x1,y1,x2,y2],
            "src":        src_code,          # *** numeric code persisted to sidecar ***
            "conf":       (float(conf_val) if conf_val is not None else float("nan")),
            "landmarks": landmarks_val,
        }
        return d
    
    def _snapshot_open_tracks_list(self, aggregator) -> list[dict]:
        open_list: list[dict] = []

        for track in getattr(aggregator, "tracks", []):
            if track.is_closed():
                continue

            last_any = track.last_frame()
            last_det = track.last_det_frame()

            bbox = None
            if last_det is not None:
                for o in reversed(track.observations):
                    if o.frame_idx == last_det and o.bbox is not None:
                        bbox = [int(v) for v in o.bbox[:4]]
                        break

            if bbox is None:
                lb = track.get_last_bbox()
                bbox = [int(v) for v in lb[:4]] if lb else [0, 0, 0, 0]

            open_list.append({
                "shot": int(getattr(track, "shot_id", -1)),
                "track_id": int(getattr(track, "track_id", -1)),
                "last_frame": int(last_any) if last_any is not None else -1,
                "last_det_frame": int(last_det) if last_det is not None else -1,
                "closed": False,
                "bbox": bbox,
            })

        return open_list

    def add_observations(self, shot_number: int, frame_idx: int, observations: list[FaceObservation]) -> int:
        """
        Append observations for THIS FRAME into the observations collector.
        STRICT: only accepts list[FaceObservation].
        """
        if not observations:
            return 0
        
        if not isinstance(observations, list):
            raise TypeError("add_observations expects list[FaceObservation]")
        # Fail fast on anything that isn't a FaceObservation
        offenders = [type(x).__name__ for x in observations if not isinstance(x, FaceObservation)]
        if offenders:
            sample = observations[:3]
            raise TypeError(
                f"add_observations requires FaceObservation objects, got {offenders}. "
                f"Sample (truncated)={sample!r}"
            )
        # Convert FaceObservation -> dict rows exactly once (src already numeric)
        observations = [self._obs_to_row_dict(shot_number, ob) for ob in observations]

        # Skip anything strictly before the resume anchor.
        if self._is_pre_anchor(frame_idx):
            logging.debug("ckpt:add_obs SKIP (pre-anchor) frame=%s anchor=%s",
                          frame_idx, self._anchor_frame)
            return 0

        if self._obs is None:
            return 0

        # Group rows by track_id (collector requires one track at a time).
        by_tid: dict[int, list[dict]] = {}
        for r in observations:
            tid = int(r.get("track_id", -1))
            if tid < 0:
                continue
            by_tid.setdefault(tid, []).append(r)

        track_order_changed = False
        total_rows_added = 0

        for tid, rows_in in by_tid.items():
            key = (int(shot_number), int(tid))
            if key not in self._shot_track_to_order:
                self._next_track_order = track_order_add(
                    self._shot_track_to_order, shot=int(shot_number), track_id=int(tid),
                    next_order=self._next_track_order
                )
                track_order_changed = True

            out_rows = []
            for r in rows_in:
                bbox = r.get("bbox_xyxy")
                if not bbox or len(bbox) != 4:
                    raise ValueError(f"Invalid bbox in row for tid={tid}: {bbox!r}")

                # Rows here are produced by _obs_to_row_dict; 'src' is already an int code.
                src_val = r.get("src", None)
                if not isinstance(src_val, int):
                    self.logger.error("BUG: ckpt row missing/invalid 'src' code "
                                 "(shot=%s frame=%s tid=%s) row=%r",
                                 shot_number, frame_idx, tid, r)
                    raise TypeError(f"BUG: row without valid 'src' int code: {r!r}")

                out = {
                    "shot": int(shot_number),
                    "track_id": int(tid),
                    "f": int(r.get("f", frame_idx)),
                    "bbox_xyxy": [float(v) for v in bbox],
                    "src": int(src_val),
                }
                if r.get("conf") is not None:
                    out["conf"] = float(r["conf"])

                landmarks = r.get("landmarks", None)
                has_landmarks = int(landmarks is not None and landmarks != [])
                out["has_landmarks"] = has_landmarks
                out["landmarks"] = (landmarks if has_landmarks else [])
                
                out_rows.append(out)

            # sanity: every row must have an int 'src'
            bad = [rr for rr in out_rows if not isinstance(rr.get("src"), int)]
            if bad:
                self.logger.error("BUG: rows to collector missing 'src' int code: count=%s "
                             "(shot=%s frame=%s) first=%r",
                             len(bad), shot_number, frame_idx, bad[0])
                raise TypeError(f"BUG: row passed to collector without int 'src': {bad[0]!r}")
            # emb_idx unknown here: set -1 via emb_idx_fn

            _, added = self._obs.append_track_obs(out_rows, emb_idx_fn=lambda _o: -1)
            total_rows_added += int(added)

        logging.debug(
            "ckpt:add_obs shot=%s frame=%s tracks=%d rows_added=%d",
            int(shot_number), int(frame_idx), len(by_tid), total_rows_added
        )

        if track_order_changed:
            logging.info(
                "ckpt:track_order persisted: entries=%d (shot=%s, frame=%s)",
                len(self._shot_track_to_order), int(shot_number), int(frame_idx)
            )
            self._write_status("track_order updated")

        return total_rows_added

    def add_embeddings(self, shot_number: int, track_id: int, frame_idx_last: int, embs: np.ndarray) -> int:
        anchor = self._anchor_frame
        self.logger.info(
            "add_embeddings: shot=%d tid=%d frame_last=%d anchor=%s embs_shape=%s",
            int(shot_number), int(track_id), int(frame_idx_last),
            (str(anchor) if anchor is not None else "None"),
            tuple(embs.shape) if hasattr(embs, "shape") else "?"
        )

        # ---------- basic guards / shape checks ----------
        if embs is None:
            return 0
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim != 2:
            raise ResumeSafetyError(
                f"[ckpt] add_embeddings expected 2D array (N, D), got shape={embs.shape!r}"
            )
        if embs.shape[0] == 0:
            return 0

        # If you have pre-anchor rules, keep them here (not shown in your snippet):
        # e.g.:
        # if anchor is not None and frame_idx_last < anchor and shot_number < self.anchor_shot():
        #     self.logger.info("add_embeddings: SKIP pre-anchor shot=%d frame_last=%d", shot_number, frame_idx_last)
        #     return 0

        if self._emb is None or self._obs is None:
            return 0

        # ---------- assign vectors into the embedding sidecar ----------
        assigned: list[int] = []
        for row_vec in embs:
            emb_idx = self._emb.assign(row_vec)
            if not isinstance(emb_idx, int):
                raise ResumeSafetyError(
                    "[ckpt] EmbeddingCollector.assign(vec) must return int index, "
                    f"got {type(emb_idx).__name__}: {emb_idx!r}"
                )
            assigned.append(int(emb_idx))

        cnt = len(assigned)
        self.logger.debug(
            "add_embeddings: assigned %d vectors; emb_rows_total=%d",
            cnt, (self._emb.count() if hasattr(self._emb, "count") else -1)
        )

        if cnt == 0:
            return 0

        find_rows = getattr(self._obs, "find_rows", None)
        update_emb_idx = getattr(self._obs, "update_emb_idx", None)
        if not callable(find_rows) or not callable(update_emb_idx):
            raise ResumeSafetyError("[ckpt] ObservationsCollector needs find_rows/update_emb_idx.")

        # ---------- find candidate obs rows for linking ----------
        candidate_rows = find_rows(
            shot=int(shot_number),
            track_id=int(track_id),
            frame_last=int(frame_idx_last),
            only_unassigned=True,
            only_with_landmarks=True,
            source=SRC_TO_CODE[Source.DETECTED],
        )
        self.logger.info(
            "link: candidates=%d (DET+landmarks, unassigned, ≤%d) for shot=%d tid=%d",
            len(candidate_rows), int(frame_idx_last), int(shot_number), int(track_id)
        )

        # --- Normalize candidate positions to (block_idx, row_idx) tuples ---
        norm_candidates: list[tuple[int, int]] = []
        for pos in candidate_rows:
            # Canonical: already a tuple
            if isinstance(pos, tuple) and len(pos) == 2:
                b, r = pos
                norm_candidates.append((int(b), int(r)))
                continue

            # Legacy: flat integer row index
            if isinstance(pos, (int, np.integer)):
                # Treat legacy flat index as (0, pos)
                norm_candidates.append((0, int(pos)))
                continue

            raise ResumeSafetyError(
                "[ckpt] ObservationsCollector.find_rows must return (block,row) tuples; "
                f"got {type(pos)!r} value={pos!r}"
            )

        candidate_rows = norm_candidates

        if not candidate_rows:
            raise ResumeSafetyError(
                f"[ckpt] no candidate obs rows for shot={shot_number} track={track_id} "
                f"≤ frame={frame_idx_last}"
            )

        if len(candidate_rows) < cnt:
            self.logger.error(
                "link:mismatch fewer candidates than embeddings: candidates=%d < embs=%d "
                "(shot=%d tid=%d ≤%d). Likely causes: landmarks missing, wrong src, or wrong frame_last.",
                len(candidate_rows), cnt, int(shot_number), int(track_id), int(frame_idx_last)
            )
            raise ResumeSafetyError(
                f"[ckpt] fewer DET+landmarks rows ({len(candidate_rows)}) than embeddings ({cnt}) "
                f"for shot={shot_number} track={track_id}."
            )

        # Choose newest K rows (collector defines meaning; we treat as flat row indices)
        target_positions = candidate_rows[-cnt:]

        # ---------- log mapping (pos -> frame, emb_idx) for debugging ----------
        frames_for_targets: list[int] = []
        if self.logger.isEnabledFor(logging.INFO):
            try:
                if hasattr(self._obs, "frame_at_pos"):
                    # frame_at_pos(pos: int) -> frame index
                    frames_for_targets = [
                        int(self._obs.frame_at_pos(pos)) for pos in target_positions
                    ]
                else:
                    # Fallback via to_array(): interpret positions as flat indices
                    arr = self._obs.to_array()
                    frames_for_targets = []
                    for pos in target_positions:
                        flat_idx = int(pos)
                        if 0 <= flat_idx < arr.shape[0]:
                            frames_for_targets.append(int(arr["f"][flat_idx]))
                        else:
                            frames_for_targets.append(-1)
            except Exception:
                frames_for_targets = [-1] * len(target_positions)

        linked_emb_indices = assigned[-len(target_positions):]

        for (pos, f, emb_idx) in zip(target_positions, frames_for_targets, linked_emb_indices):
            self.logger.info(
                "EMB-EVENT stage=checkpoint shot=%d tid=%d frame=%d pos=%s emb_idx=%d",
                int(shot_number), int(track_id),
                int(f if f is not None else -1),
                f"{pos}",
                int(emb_idx),
            )

        # ---------- do the actual linking ----------
        if len(target_positions) != len(linked_emb_indices):
            raise ResumeSafetyError(
                f"[ckpt] internal error: positions={len(target_positions)} "
                f"emb_indices={len(linked_emb_indices)}"
            )

        update_emb_idx(
            positions=target_positions,
            emb_indices=linked_emb_indices,
        )

        self.logger.info(
            "link:done shot=%d tid=%d linked=%d_of_%d (candidates=%d)",
            int(shot_number), int(track_id),
            len(target_positions), cnt, len(candidate_rows)
        )
        return cnt

    def checkpoint_now(
            self, 
            *, 
            frame_idx: int, 
            shot_number: int,
            aggregator: ShotFaceTrackAggregatorProtocol,
            shot_first_frame: int | None = None,
            note: str = "checkpoint") -> None:
        """
        Persist a point-in-time snapshot that allows a safe resume:
        - Write obs_ckpt-<frame>.npz under <run_root>/ckpt atomically.
        - (Optionally) write emb_ckpt-<frame>.npz if an API is available.
        - Update status.json with last_detection_frame/shot metadata.
        """
        if self._obs is None or self._emb is None:
            return

        obs_count = self._obs.count()
        emb_count = self._emb.count()
 
        self._last_det_frame = int(frame_idx)
        self._last_det_shot = int(shot_number)
        self._last_det_shot_first_frame = (int(shot_first_frame) if shot_first_frame is not None else None)
        self._obs_rows_at_det = max(0, obs_count)
        self._emb_rows_at_det = max(0, emb_count)

        logging.info(
            "ckpt:anchor set @ frame=%d shot=%d shot_first=%s "
            "obs_rows_at_det=%d emb_rows_at_det=%d (pre-write)",
            self._last_det_frame, self._last_det_shot, self._last_det_shot_first_frame,
            self._obs_rows_at_det, self._emb_rows_at_det
        )

        _dump_npz_atomic(self._obs, self.obs_path)
        _dump_npz_atomic(self._emb, self.emb_path)
        # Capture a compact JSON list of the open tracks at this anchor.
        self._open_tracks_inline = self._snapshot_open_tracks_list(aggregator)
        self._write_status(note or "checkpoint")

    def matches_video(self, video_path: _t.Union[str, Path]) -> bool:
        """Return True if the stored checkpoint was created for this video path."""
        status = self.read_status()
        return bool(status and str(status.get("video_path")) == str(video_path))

    def finalize(self, note: str = "final") -> None:
        # Ensure sidecars exist even if no detection checkpoint happened.
        if self._obs is not None:
            _dump_npz_atomic(self._obs, self.obs_path)
        if self._emb is not None:
             _dump_npz_atomic(self._emb, self.emb_path)
        self._write_status(note)

    def build_manifest_v2_3_from_obs_sidecar(
        self,
        *,
        obs_sidecar_path: str,
        emb_sidecar_path: str | None,
        manifest_path: str,
    ) -> None:
        """
        Deterministic manifest builder that relies ONLY on persisted sidecars.
        Sorts obs rows by (shot, track_id, f) so each (shot,track) is contiguous,
        then emits per-track (offset,count) slices into the obs NPZ.
        """
        obs_npz_path = Path(obs_sidecar_path)
        with np.load(obs_npz_path, allow_pickle=False) as z:
            obs = z["observations"]

        order = np.lexsort((obs["f"], obs["track_id"], obs["shot"]))
        obs = obs[order]

        # Rewrite the obs sidecar in sorted order so offsets match file contents
        atomic_write_npz(obs_npz_path, observations=obs)

        # Build slices
        shots: dict[int, list[dict]] = {}
        n = obs.shape[0]
        i = 0
        while i < n:
            shot = int(obs["shot"][i])
            tid  = int(obs["track_id"][i])
            start = i
            i += 1
            while i < n and int(obs["shot"][i]) == shot and int(obs["track_id"][i]) == tid:
                i += 1
            count = i - start
            first_f = int(obs["f"][start])
            last_f  = int(obs["f"][start + count - 1])

            shots.setdefault(shot, []).append({
                "track_id": tid,
                "first_frame": first_f,
                "last_frame": last_f,
                "obs_offset": start,
                "obs_count": count,
            })

        # stable ordering
        shot_items = []
        for shot in sorted(shots.keys()):
            tracks = sorted(shots[shot], key=lambda t: (t["first_frame"], t["track_id"]))
            shot_items.append({"shot_id": shot, "face_tracks": tracks})

        manifest = {
            "schema_version": REQUIRED_SCHEMA_VERSION,
            "video_path": self.video_path,
            "obs_sidecar": Path(obs_sidecar_path).name,
            "emb_sidecar": (Path(emb_sidecar_path).name if emb_sidecar_path else None),
            "shots": shot_items,
            "generated_at_utc": _utc_now(),
        }

        _atomic_write_text(Path(manifest_path), json.dumps(manifest, indent=2))


    def copy_ckpt_sidecars_to_final(
        self,
        obs_sidecar_path: str | None = None,
        emb_sidecar_path: str | None = None,
    ) -> None:
        """
        Export sidecars from the live checkpoint directory to the final locations.
        Copy the observations NPZ verbatim so the final file contains a single 
        structured array under the key `'observations'`
        with fields: f, shot, track_id, bbox_xyxy, src, conf, emb_idx.
        Embeddings are also copied verbatim.
        Safe no-ops if a given source file doesn't exist or a target path is None.
        """
        try:
            if obs_sidecar_path:
                src = self.ckpt_dir / "obs_ckpt.npz"
                if src.exists():
                    shutil.copy2(src, obs_sidecar_path)
                    logging.info("ckpt:export wrote legacy structured obs sidecar -> %s", obs_sidecar_path)
                else:
                    logging.info("ckpt:export obs sidecar missing at %s", src)

            if emb_sidecar_path:
                src = self.ckpt_dir / "emb_ckpt.npz"
                if src.exists():
                    shutil.copy2(src, emb_sidecar_path)
                    logging.info("ckpt:export copied emb sidecar -> %s", emb_sidecar_path)
                else:
                    logging.info("ckpt:export emb sidecar missing at %s", src)

        except Exception as e:
            # Do not crash pipeline completion if export copy fails; just log.
            logging.error("ckpt:export failed: %s", e)

    def _write_status_dict(self, status: dict) -> None:
        _atomic_write_text(self.status_path, json.dumps(status, indent=2))

    def mark_completed(self) -> None:
        """
        Mark the run as completed in status.json. This enables 'resume-for-outputs'
        semantics without adding any special CLI switches.
        """
        status = self.read_status()
        status["completed"] = True
        status["completed_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._write_status_dict(status)
        logging.info("ckpt:run marked completed")

    @staticmethod
    def compute_parent_dir(default_root: Path, video_path: Path) -> Path:
        return _video_parent_dir(default_root=default_root, video_path=video_path)

    # ---------- resume helpers ----------
    def get_resume_anchor(self):
        """
        Return (last_detection_frame, last_detection_shot, last_detection_shot_first_frame)
        when available. If status.json exists, prefer it. Otherwise, fall back
        to in-memory attributes that may be set during tests.
        """
        # 1) Prefer status.json (single source of truth)
        try:
            if self.status_path and self.status_path.exists():
                import json
                st = json.loads(self.status_path.read_text() or "{}")
                frame = st.get("last_detection_frame")
                shot = st.get("last_detection_shot")
                shot_first_frame = st.get("last_detection_shot_first_frame")
                if frame is not None:
                    return int(frame), (int(shot) if shot is not None else None), (int(shot_first_frame) if shot_first_frame is not None else None)
        except Exception:
            pass

        # 2) Fall back to in-memory fields (used by tests)
        frame = getattr(self, "_last_det_frame", None)
        shot = getattr(self, "_last_det_shot", None)
        shot_first_frame = getattr(self, "_last_det_shot_first_frame", None)
        if frame is not None:
            # Ensure ints; return triple (shot_first may be None)
            return int(frame), (int(shot) if shot is not None else None), (int(shot_first_frame) if shot_first_frame is not None else None)

        return None
    
    def _obs_np(self):
        """
        Load obs sidecar and tolerate either 'track_id' (canonical) or legacy 'tid'.
        If only one exists, add the other as an alias field so downstream code can use either.
        """
        p = self.obs_path
        arr = np.load(p, allow_pickle=False)["observations"]
        names = arr.dtype.names or ()
        if ("track_id" not in names) and ("tid" in names):
            # add canonical alias
            arr = rfn.append_fields(arr, "track_id", arr["tid"].astype(np.int32, copy=False), usemask=False)
        elif ("tid" not in names) and ("track_id" in names):
            # add legacy alias to satisfy older consumers
            arr = rfn.append_fields(arr, "tid", arr["track_id"].astype(np.int32, copy=False), usemask=False)
        return arr


    def _emb_np(self):
        """
        Return a numpy structured array of embedding rows from ckpt/emb_ckpt.npz.
        Expected fields: ['shot','frame','vec', ...]
        Read-only, used for resume/audit; does nothing on hot path.
        """
        p = self.emb_path  # wherever you already save the emb sidecar
        arr = np.load(p, allow_pickle=False)['embeddings']
        return arr

    # def get_embeddings_by_frames(self, shot: int, frames: list[int]) -> np.ndarray | None:
    #     """
    #     Return embeddings stacked in the same order as `frames` for a given shot.

    #     Semantics:
    #     - Looks only at the embedding sidecar (ckpt/emb_ckpt.npz).
    #     - Does *not* distinguish tracks; it just ensures that for each requested
    #         frame there is at least one embedding row for this shot.
    #     - If ANY requested frame has no embedding row → return None.

    #     This is suitable for resume/debug sanity checks, not for per-track analysis.
    #     """
    #     if not frames:
    #         return None

    #     # Structured array with fields like: ['shot', 'frame', 'vec', ...]
    #     embedding_arr = self._emb_np()

    #     shot_int = int(shot)

    #     # Normalize columns
    #     shots = np.asarray(embedding_arr["shot"], dtype=np.int32)
    #     frs   = np.asarray(embedding_arr["frame"], dtype=np.int32)
    #     vecs  = np.asarray(embedding_arr["vec"], dtype=np.float32)

    #     # Filter rows for this shot only
    #     mask_shot = (shots == shot_int)
    #     frames_for_shot = frs[mask_shot]
    #     vecs_for_shot   = vecs[mask_shot]

    #     # Build map: frame_idx -> "some" embedding for that frame in this shot.
    #     # If multiple tracks have embeddings at the same frame, the last one wins;
    #     # for completeness checks we only care that *at least one* exists.
    #     by_frame: dict[int, np.ndarray] = {}
    #     for f, v in zip(frames_for_shot, vecs_for_shot):
    #         by_frame[int(f)] = v

    #     # Assemble result in the order of `frames`, fail if any missing.
    #     out: list[np.ndarray] = []
    #     for f in frames:
    #         v = by_frame.get(int(f))
    #         if v is None:
    #             # Missing embedding for a requested frame → signal failure
    #             return None
    #         out.append(v)

    #     return np.stack(out, axis=0)

    def get_det_frames_for_track(self, shot: int, track_id: int, frame_max: int | None = None) -> list[int]:
        arr = self._obs_np()
        det_code = int(SRC_TO_CODE[Source.DETECTED])

        mask = (
            (arr["shot"] == int(shot)) &
            (arr["track_id"] == int(track_id)) &
            (arr["src"] == det_code)
        )
        if frame_max is not None:
            mask &= (arr["f"] <= int(frame_max))

        frames = [int(f) for f in np.asarray(arr["f"][mask]).tolist()]
        frames.sort()
        return frames

    def get_embeddings_by_frames(self, shot: int, frames: list[int]) -> np.ndarray | None:
        """
        Return embeddings stacked in the same order as `frames` for a given shot.

        Semantics:
        - Uses the obs sidecar to map (shot, frame) -> emb_idx.
        - The emb sidecar is a flat 2D array 'embeddings' where emb_idx is the row index.
        - Does *not* distinguish tracks; it just ensures that for each requested
          frame there is at least one observation with an attached embedding.
        - If ANY requested frame has no embedding row → return None.

        This is mainly for resume/debug sanity checks, not per-track analysis.
        """
        if not frames:
            return None

        # Load obs sidecar as structured array ('observations').
        obs_arr = self._obs_np()
        names = obs_arr.dtype.names or ()
        if "emb_idx" not in names:
            # Without emb_idx, we can't map frames to embeddings.
            return None

        shot_int = int(shot)
        requested_frames = [int(f) for f in frames]
        requested_set = set(requested_frames)

        # Filter obs rows for this shot that have an assigned embedding.
        mask_shot = (obs_arr["shot"] == shot_int)
        mask_emb  = (obs_arr["emb_idx"] >= 0)
        obs_for_shot = obs_arr[mask_shot & mask_emb]

        # Build frame -> list[emb_idx] (in append order) for this shot.
        frame_to_idxs: dict[int, list[int]] = {}
        for row in obs_for_shot:
            f = int(row["f"])
            if f not in requested_set:
                continue
            ei = int(row["emb_idx"])
            frame_to_idxs.setdefault(f, []).append(ei)

        # Load flat embeddings array (shape (N, D)).
        emb_arr = self._emb_np()
        if emb_arr is None or getattr(emb_arr, "ndim", 0) != 2:
            return None

        n_rows = emb_arr.shape[0]
        out_vecs: list[np.ndarray] = []

        # For each requested frame, grab "some" embedding (last one if multiple).
        for f in requested_frames:
            idxs = frame_to_idxs.get(int(f))
            if not idxs:
                # No embedding recorded for this frame in this shot.
                return None
            emb_idx = int(idxs[-1])  # newest / last one wins
            if emb_idx < 0 or emb_idx >= n_rows:
                # Defensive: corrupted emb_idx.
                return None
            out_vecs.append(np.asarray(emb_arr[emb_idx], dtype=np.float32))

        return np.stack(out_vecs, axis=0)
    
    # --- helper: current resume anchor frame (if any) -------------------------
    def snapshot_open_tracks(self, aggregator) -> None:
        """
        Store open tracks with:
          - shot
          - track_id
          - last_frame (ANY)
          - last_det_frame (authoritative)
          - bbox at last_det_frame (if known), else last bbox
        """
        open_list = []

        for track in getattr(aggregator, "tracks", []):
            if track.is_closed():
                continue
            last_any = track.last_frame()
            last_det = track.last_det_frame()
            # Prefer bbox at last DET; fall back to last ANY bbox
            bbox = None
            if last_det is not None:
                # find obs at last_det
                for o in reversed(track.observations):
                    if o.frame_idx == last_det and o.bbox is not None:
                        bbox = tuple(int(v) for v in o.bbox[:4])
                        break
            if bbox is None:
                lb = track.get_last_bbox()
                bbox = tuple(int(v) for v in lb[:4]) if lb else (0, 0, 0, 0)
            open_list.append({
                "shot": int(getattr(track, "shot_id", -1)),
                "track_id": int(getattr(track, "track_id", -1)),
                "last_frame": (int(last_any) if last_any is not None else -1),
                "last_det_frame": (int(last_det) if last_det is not None else -1),
                "bbox": bbox,
            })
        status = self.read_status() or {}
        status["open_tracks"] = open_list
        # Minimal safe persist: stash into the manager then use _write_status
        self._open_tracks_inline = open_list
        self._write_status("open_tracks updated")
    
    def resume_available(self) -> bool:
        return self.status_path.exists() and self.obs_path.exists() and self.emb_path.exists()

    def read_status(self) -> dict | None:
        if not self.status_path.exists():
            return None
        try:
            return json.loads(self.status_path.read_text())
        except Exception as e:
            return None
        
    def _validate_collectors_schema(self, obs_collector, emb_collector) -> None:
        """
        Lightweight sanity checks for collectors when resuming.

        We *do not* support any legacy NPZ layouts. We assume:
          - ckpt/obs_ckpt.npz has an 'observations' array.
          - ckpt/emb_ckpt.npz, if present, contains an 'embeddings' array and
            (optionally) the embedding collector exposes (N, 512) float32 vectors.

        Fresh runs (no_resume=True) skip this entirely.
        """
        # If this is a cold run, there is nothing to validate.
        if not self.resume_enabled or not self.resume_available():
            return

        # ---- observations sidecar structure (ckpt/obs_ckpt.npz) ----
        # We always require the checkpoint obs NPZ to have an 'observations' array.
        if self.obs_path.exists():
            _assert_npz_keys(self.obs_path, ("observations",))
        else:
            # If resume was requested but the obs sidecar is missing, this is a hard error.
            raise ResumeSafetyError(
                f"[ckpt] expected observations sidecar at {self.obs_path} when resuming, but it is missing."
            )

        # ---- embeddings sidecar structure (ckpt/emb_ckpt.npz) ----
        # Embeddings are optional, but when the file exists it must expose 'embeddings'.
        if self.emb_path.exists():
            _assert_npz_keys(self.emb_path, ("embeddings",))

        # ---- in-memory embedding collector sanity checks ----
        # If the embedding collector can expose its in-RAM array, make sure it looks sane.
        try:
            if hasattr(emb_collector, "to_array"):
                arr = emb_collector.to_array()
                if arr is not None and getattr(arr, "size", 0) > 0:
                    if arr.ndim != 2 or arr.shape[1] != 512:
                        raise ResumeSafetyError(
                            f"[ckpt] emb collector has invalid shape {arr.shape}, "
                            "expected (N, 512)"
                        )
                    if arr.dtype != np.float32:
                        raise ResumeSafetyError(
                            f"[ckpt] emb collector dtype {arr.dtype} is not float32"
                        )
        except AttributeError:
            # Older / simpler collectors without to_array() are allowed.
            pass


    def _resolve_obs_columns(self, obs_collector) -> set[str]:
        """
        Return a set of column names for the observations collector, handling:
        - attribute 'columns' / 'COLUMNS' / 'required_columns' as set/list/tuple
        - callable variants of the above
        - optional 'schema' dict with 'columns' or 'fields'
        If nothing is advertised, return an empty set (caller decides whether to enforce).
        """
        # Common attribute names
        for name in ("columns", "COLUMNS", "required_columns"):
            val = getattr(obs_collector, name, None)
            if val is not None:
                if callable(val):
                    val = val()
                if isinstance(val, (set, list, tuple)):
                    return set(val)

        # Optional schema dict
        schema = getattr(obs_collector, "schema", None)
        if isinstance(schema, dict):
            for key in ("columns", "fields"):
                vals = schema.get(key)
                if isinstance(vals, (set, list, tuple)):
                    return set(vals)

        return set()

    def load_into_collectors(self, obs_collector, emb_collector) -> tuple[int, int]:
        """
        Load NPZs into the provided collectors. Returns (obs_rows, emb_rows).
        You should then *trim* to status['*_at_last_detection'] to anchor resume.
        """
        obs_rows = emb_rows = 0
        if self.resume_enabled and self.resume_available():
            try:
                obs_rows = obs_collector.load_npz(self.obs_path)
            except Exception:
                logging.info("ckpt:load obs_collector.load_npz() raised exception")
                pass
            try:
                emb_rows = emb_collector.load_npz(self.emb_path)
            except Exception:
                logging.info("ckpt:load emb_collector.load_npz() raised exception (non-strict)")


        logging.info("ckpt:load obs_rows=%d emb_rows=%d from (%s, %s)",
             obs_rows, emb_rows, self.obs_path, self.emb_path)
        return obs_rows, emb_rows
    
    def anchor_frame(self) -> int | None:
        return self._last_det_frame
    
    def anchor_shot(self) -> int | None:
        return self._last_det_shot
    
    def anchor_shot_first_frame(self) -> int | None:
        return self._last_det_shot_first_frame
    
    # ---------- run-dir factory ----------
    @classmethod
    def open(
        cls,
        *,
        parent_dir: Path,
        video_path: Path,
        options_snapshot: dict,
        no_resume: bool,
        run_id: str | None = None,
        resume_latest: bool = False,
        force_new_run: bool = False,
    ) -> "CheckpointManager":
        """
        Create or select a checkpoint run directory and return a configured CheckpointManager.

        Run directory selection:
        - If `run_id` is provided, use that exact subdirectory under `parent_dir`.
        - Else if `resume_latest` is True, select the newest existing `run-*` subdirectory.
        - Else create a fresh `run-<timestamp>-<hash>` subdirectory.

        Modes:
        - no_resume
            When true, forbid the loading of existing checkpoint artifacts (status.json, ckpt/*.npz).

        - force_new_run=True:
            Start clean in the selected directory by purging checkpoint artifacts
            (deletes `ckpt/` and `status.json`).

        Side effects:
        - Ensures `parent_dir` exists.
        - When `force_new_run=True` and the selected directory exists, removes
        `run_dir/ckpt` and `run_dir/status.json`, then recreates `run_dir/ckpt`.
        - Writes/updates `status.json` with note "opened" (only for new runs).

        Returns:
        CheckpointManager: Instance bound to the chosen `run_dir`. Its `resume_enabled`
        flag reflects the (possibly downgraded) `resume` value.

        Raises:
        FileNotFoundError: If `run_id` is specified but the directory does not exist.
        ValueError:
            - If both `run_id` and `resume_latest` are provided.
            - If an existing run is being targeted (`run_id` or `resume_latest`) but
            `no_resume` is True.
        """
        # Ensure parent exists before globbing
        parent_dir = Path(parent_dir)
        parent_dir.mkdir(parents=True, exist_ok=True)

        logging.debug(
            "ckpt.open: args run_id=%r resume_latest=%r force_new_run=%r no_resume=%r parent=%s video=%s",
            run_id, resume_latest, force_new_run, no_resume, str(parent_dir), str(video_path)
        )

        # Guard contradictions about selected dir
        if run_id and resume_latest:
            raise ValueError("run_id and resume_latest both select particular previous runs to resume; they are mutually exclusive.")

        # If targeting an existing dir, caller must not choose no_resume
        if (run_id or resume_latest) and no_resume:
            raise ValueError(
                "When selecting an existing run (run_id or resume_latest), no_resume must not also be set."
            )

        # Select run directory
        do_resume = not (no_resume or force_new_run)
        selected_dir_exists = False

        if run_id:
            run_dir = parent_dir / run_id
            if not run_dir.exists():
                raise FileNotFoundError(f"checkpoint run not found: {run_dir}")
            selected_dir_exists = True
        elif resume_latest:
            prev_runs = sorted([d for d in parent_dir.glob("run-*") if d.is_dir()], key=lambda p: p.name)
            if prev_runs:
                run_dir = prev_runs[-1]
                selected_dir_exists = True
            else:
                # no existing; fall back to new
                run_dir = cls._create_new_run_dir(parent_dir, options_snapshot)
                do_resume = False
        else:
            run_dir = cls._create_new_run_dir(parent_dir, options_snapshot)
            do_resume = False

        logging.debug(
            "ckpt.open: selected run_dir=%s selected_dir_exists=%s do_resume=%s",
            str(run_dir), selected_dir_exists, do_resume
        )

        # Purge directory if overwrite set and directory exists
        if selected_dir_exists and force_new_run:
            # purge ckpt artifacts so we truly start fresh
            shutil.rmtree(run_dir / "ckpt", ignore_errors=True)
            (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
            (run_dir / "status.json").unlink(missing_ok=True)

        mgr = cls(run_dir, video_path=video_path, resume=do_resume)
        mgr._cfg = dict(options_snapshot or {})

        # Helpful debug logging for presence/sizes of checkpoint sidecars
        obs_sidecar_path = mgr.ckpt_dir / "obs_ckpt.npz"
        emb_sidecar_path = mgr.ckpt_dir / "emb_ckpt.npz"
        def _sz(p: Path) -> str:
            try:
                return f"{p.stat().st_size}B"
            except Exception:
                return "NA"
        logging.debug(
            "ckpt.open: ckpt files: obs=%s(%s) exists=%s | emb=%s(%s) exists=%s",
            str(obs_sidecar_path), _sz(obs_sidecar_path), obs_sidecar_path.exists(),
            str(emb_sidecar_path), _sz(emb_sidecar_path), emb_sidecar_path.exists(),
        )

        # Inspect status
        if do_resume:
            status = {}
            if mgr.status_path.exists():
                try:
                    status = json.loads(mgr.status_path.read_text() or "{}") or {}
                except Exception as e:
                    raise ValueError(f"Failed to read status.json for resume: {mgr.status_path}") from e
            else:
                # If user asked to resume, but there's no status.json, that's not resumable.
                # Either downgrade to cold start or raise; your tests expect safety checks,
                # so raising is typically better.
                raise FileNotFoundError(f"Cannot resume: missing status.json at {mgr.status_path}")

            # Hard stop: schema mismatch
            sv = str(status.get("schema_version", ""))
            if sv != REQUIRED_SCHEMA_VERSION:
                raise ResumeSafetyError(
                    f"[resume safety] unsupported checkpoint schema: found {sv!r}, "
                    f"required {REQUIRED_SCHEMA_VERSION!r}. Please re-run to regenerate checkpoints."
                )

            status_path = mgr.status_path
            obs_sidecar_path = mgr.ckpt_dir / "obs_ckpt.npz"
            emb_sidecar_path = mgr.ckpt_dir / "emb_ckpt.npz"

            if not status_path.exists():
                raise ResumeSafetyError(
                    "ckpt.open: resume requested but status.json is missing. "
                    "Cannot verify anchors or resume-safety. "
                    "Use --force-new-run or --no-resume to start fresh."
                )

            status = mgr.read_status() or {}
            # Extract minimal signals of progress/anchors
            frames_done = int(status.get("frames_done", 0) or 0)
            shots_done  = int(status.get("shots_done", 0) or 0)
            obs_anchor  = int(status.get("obs_rows_at_last_detection", 0) or 0)
            emb_anchor  = int(status.get("emb_rows_at_last_detection", 0) or 0)
            track_order = status.get("track_order") if isinstance(status.get("track_order"), list) else []
            had_progress = any([
                frames_done > 0, shots_done > 0,
                obs_anchor > 0, emb_anchor > 0,
                len(track_order) > 0
            ])

            mgr._had_prior_progress = bool(had_progress)

            have_obs = obs_sidecar_path.exists()
            have_emb = emb_sidecar_path.exists()

            emb_store = str(status.get("emb_store") or "inline").lower()
            expect_emb = emb_store in ("inline", "sidecar")
            require_emb_file = expect_emb and int(emb_anchor) > 0

            # Embeddings in ckpt are required if we have evidence of prior progress that
            # could include completed shots, i.e. obs rows exist before anchor.
            obs_anchor  = int(status.get("obs_rows_at_last_detection", 0) or 0)
            emb_anchor  = int(status.get("emb_rows_at_last_detection", 0) or 0)

            have_obs = obs_sidecar_path.exists()
            have_emb = emb_sidecar_path.exists()

            if had_progress:
                if not have_obs:
                    raise ResumeSafetyError(...)

                # If checkpoint claims embeddings rows, emb sidecar must exist.
                if emb_anchor > 0 and not have_emb:
                    raise ResumeSafetyError(
                        f"ckpt.open: inconsistent checkpoint: emb_rows_at_last_detection={emb_anchor} "
                        f"but {emb_sidecar_path} is missing."
                    )

                # Structural sanity
                _assert_npz_keys(obs_sidecar_path, ("observations",))
                if have_emb:
                    _assert_npz_keys(emb_sidecar_path, ("embeddings",))

            else:
                # No progress - missing sidecars normal at the very beginning.
                logging.debug(
                    "ckpt.open: status present but no prior progress; sidecars present? obs=%s emb=%s",
                    have_obs, have_emb
                )


        return mgr

    @classmethod
    def _create_new_run_dir(cls, parent_dir: Path, snapshot: dict) -> Path:
        rid = f"run-{_utcstamp_compact()}-{_paramhash8(snapshot)}"
        run_dir = parent_dir / rid
        run_dir.mkdir(parents=True, exist_ok=False)
        # Best-effort convenience symlink
        try:
            cur = parent_dir / "current"
            if cur.exists() or cur.is_symlink():
                cur.unlink()
            cur.symlink_to(run_dir.name)
        except Exception:
            pass
        return run_dir

    # ---------- resume safety ----------

    def protected_keys(self) -> tuple[str, ...]:
        return _PROTECTED_KEYS

    def diff_snapshot(self, current: dict) -> list[tuple[str, object, object]]:
        """
        Compare current runtime options with what’s in status.json (if present).
        Returns a list of (key, was, now) for protected keys that differ.
        Missing status.json yields [].
        """
        status = self.read_status() or {}
        diffs = []
        for k in _PROTECTED_KEYS:
            if str(status.get(k)) != str(current.get(k)):
                diffs.append((k, status.get(k), current.get(k)))
        return diffs

    def validate_resume_or_raise(self, current: dict, *, force: bool) -> bool:
        """
        If status/sidecars exist:
          - Enforce same video_path (hard stop; force does NOT override).
          - Enforce protected keys unless force=True.
        Returns True if resume should proceed (files present & allowed), else False.
        """
        if not self.resume_available():
            return False

        status = self.read_status() or {}

        # hard stop: different video (we’re inside a run for a specific video)
        if str(status.get("video_path")) != str(current.get("video_path")):
            raise ResumeSafetyError(
                f"[resume safety] video_path mismatch: was {status.get('video_path')!r}, now {current.get('video_path')!r}"
            )
        
        # Hard stop: schema mismatch
        sv = str(status.get("schema_version", ""))
        if sv != REQUIRED_SCHEMA_VERSION:
            raise ResumeSafetyError(
                f"[resume safety] unsupported checkpoint schema: found {sv!r}, "
                f"required {REQUIRED_SCHEMA_VERSION!r}. Please re-run to regenerate checkpoints."
            )

        diffs = self.diff_snapshot(current)
        if diffs and not force:
            msg = "[resume safety] The following parameters differ from the checkpoint:\n" + \
                  "\n".join(f" - {k}: was {old!r}, now {new!r}" for k, old, new in diffs) + \
                  "\nRefusing to resume. Re-run with --force to override (results may be non-deterministic)."
            raise ResumeSafetyError(msg)

        return True  # OK to resume

    def _log_track_order_summary(self, where: str) -> None:
        ko = self._shot_track_to_order
        shots = sorted({s for (s, _) in ko})
        total = len(ko)
        # sample the first few keys deterministically
        sample = sorted(ko.items(), key=lambda kv: kv[1])[:10]
        logging.info(
            "ckpt:track_order %s: entries=%d shots=%s next=%d sample(first10 by order)=%s",
            where, total, shots, self._next_track_order,
            [((s,t), ko[(s,t)]) for (s,t), _ in sample]
        )
    
    def load_and_anchor_collectors(
        self,
        obs_collector,
        emb_collector,
        *,
        trim_to_anchor: bool = True,
    ) -> tuple[int, int]:
        """
        Load NPZs (if available) into collectors. Optionally trim them back to
        the last detection checkpoint anchor so re-runs are deterministic.
        Returns (loaded_obs_rows, loaded_emb_rows).
        """
        loaded_obs, loaded_emb = self.load_into_collectors(obs_collector, emb_collector)
        self._validate_collectors_schema(obs_collector, emb_collector)

        # Make these the live collectors for this manager. This is required so that:
        #  - later calls like add_embeddings()/add_observations() have a collector
        #  - sidecar fallback below (which consults self._obs) can run
        self._obs = obs_collector
        self._emb = emb_collector

        # Counts BEFORE any trimming (for logging)
        pre_obs = getattr(obs_collector, "count", lambda: None)()
        pre_emb = getattr(emb_collector, "count", lambda: None)()

        # Read status ONCE
        status = self.read_status() or {}

        shot_track_order_list = status.get("track_order") or []

        # Restore track-order (once, early)
        try:
            self._shot_track_to_order, self._next_track_order = track_order_list_to_dict(
                shot_track_order_list, strict=True
            )
        except TrackOrderError as e:
            logging.warning("ckpt.start: corrupt/legacy track_order; trying sidecar fallback (%s)", e)
            self._shot_track_to_order = {}
            self._next_track_order = 0

        # --- sidecar fallback (if collector exposes a stable order) ---
        if not self._shot_track_to_order:
            try:
                omap = getattr(self._obs, "_order_map", None)
                if isinstance(omap, dict) and omap:
                    # install in order of appearance (value is order)
                    self._shot_track_to_order = {
                        (int(s), int(t)): int(o)
                        for (s, t), o in sorted(omap.items(), key=lambda kv: kv[1])
                    }
                    self._next_track_order = max(self._shot_track_to_order.values(), default=-1) + 1
                    logging.info("ckpt.start: installed track_order from sidecar (entries=%d)",
                                    len(self._shot_track_to_order))
                    # persist immediately so future resumes don't rely on fallback
                    self._write_status("track_order from sidecar")
            except Exception:
                logging.exception("ckpt.start: sidecar fallback for track_order failed")

        # Summarize what we restored
        try:
            shots_present = sorted({s for (s, _) in self._shot_track_to_order})
        except Exception:
            shots_present = []
        logging.info(
            "ckpt:track_order resumed: %s",
            track_order_summary(self._shot_track_to_order),
        )

        if not trim_to_anchor:
            logging.debug(
                "ckpt:anchor trimming disabled; loaded_obs=%s loaded_emb=%s",
                loaded_obs, loaded_emb
            )
            return loaded_obs, loaded_emb

        # ---- Trim to anchor (use the same status dict) ----
        orig_od = int(status.get("obs_rows_at_last_detection", 0) or 0)
        orig_ed = int(status.get("emb_rows_at_last_detection", 0) or 0)
        lf = status.get("last_detection_frame")   # may be None
        ls = status.get("last_detection_shot")    # may be None
        lfirst = status.get("last_detection_shot_first_frame")

        self._last_det_frame = (int(lf) if lf is not None else None)
        self._last_det_shot  = (int(ls) if ls is not None else None)
        self._last_det_shot_first_frame = (int(lfirst) if lfirst is not None else None)
        self._obs_rows_at_det = int(orig_od)
        self._emb_rows_at_det = int(orig_ed)

        if orig_od == 0 and orig_ed == 0:
            logging.info(
                "ckpt:anchor no anchor present; pre_obs=%s pre_emb=%s loaded_obs=%s loaded_emb=%s",
                pre_obs, pre_emb, loaded_obs, loaded_emb
            )
            return loaded_obs, loaded_emb

        # Clamp if status claims more than we loaded (corrupt/incomplete NPZs)
        od = min(orig_od, pre_obs or orig_od) if pre_obs is not None else orig_od
        ed = min(orig_ed, pre_emb or orig_ed) if pre_emb is not None else orig_ed
        if od != orig_od or ed != orig_ed:
            logging.warning(
                "ckpt:anchor requested beyond loaded rows; "
                "requested(obs=%d, emb=%d) clamped_to(obs=%d, emb=%d) pre_obs=%s pre_emb=%s frame=%s shot=%s",
                orig_od, orig_ed, od, ed, pre_obs, pre_emb, lf, ls
            )

        # Do the trimming (each independently)
        try:
            if od:
                obs_collector.trim_to(int(od))
        except Exception as e:
            logging.exception("ckpt:anchor obs trim_to(%d) failed: %s", od, e)

        try:
            if ed:
                emb_collector.trim_to(int(ed))
        except Exception as e:
            logging.exception("ckpt:anchor emb trim_to(%d) failed: %s", ed, e)

        logging.info(
            "ckpt:anchor post-trim obs.count=%s emb.count=%s (clamped from obs=%d emb=%d)",
            getattr(obs_collector, "count", lambda: None)(),
            getattr(emb_collector, "count", lambda: None)(), orig_od, orig_ed
        )

        # Post-trim counts
        post_obs = getattr(obs_collector, "count", lambda: None)()
        post_emb = getattr(emb_collector, "count", lambda: None)()

        # Final summary
        logging.info(
            "ckpt:anchor trimmed obs %s→%s (anchor %d→%d) emb %s→%s (anchor %d→%d) @ frame=%s shot=%s",
            pre_obs, post_obs, orig_od, od,
            pre_emb, post_emb, orig_ed, ed,
            lf, ls
        )

        # --- Trim BY FRAME so nothing at/after the anchor remains ---
        # If we know the last detection frame, drop any rows with f >= anchor_frame.
        try:
            if lf is not None:
                anchor_frame_int = int(lf)

                # Prefer collector-native frame trimming if available.
                trimmed_by_frame = False
                if hasattr(obs_collector, "trim_to_frame") and callable(obs_collector.trim_to_frame):
                    obs_before = getattr(obs_collector, "count", lambda: None)()
                    obs_collector.trim_to_frame(anchor_frame_int - 1)   # keep strictly < anchor
                    obs_after = getattr(obs_collector, "count", lambda: None)()
                    logging.info(
                        "ckpt:frame-trim obs %s→%s @ frame=%d (strictly < anchor)",
                        obs_before, obs_after, anchor_frame_int
                    )
                    trimmed_by_frame = True

                if hasattr(emb_collector, "trim_to_frame") and callable(emb_collector.trim_to_frame):
                    emb_before = getattr(emb_collector, "count", lambda: None)()
                    emb_collector.trim_to_frame(anchor_frame_int - 1)
                    emb_after = getattr(emb_collector, "count", lambda: None)()
                    logging.info(
                        "ckpt:frame-trim emb %s→%s @ frame=%d (strictly < anchor)",
                        emb_before, emb_after, anchor_frame_int
                    )
                    trimmed_by_frame = True

                # Fallback: if no trim_to_frame(), derive cut rows from the array & call trim_to(row_count)
                if not trimmed_by_frame and hasattr(obs_collector, "to_array"):
                    try:
                        arr = obs_collector.to_array()
                        # find the last index whose frame ('f') is < anchor_frame
                        if getattr(arr, "size", 0) > 0:
                            # We assume the collector preserves append order;
                            # find the *position* of the last < anchor frame row.
                            valid_mask = (arr["f"] < anchor_frame_int)
                            keep_rows = int(valid_mask.sum())
                            if keep_rows < getattr(obs_collector, "count", lambda: 0)():
                                obs_before = getattr(obs_collector, "count", lambda: None)()
                                if hasattr(obs_collector, "trim_to") and callable(obs_collector.trim_to):
                                    obs_collector.trim_to(keep_rows)
                                    obs_after = getattr(obs_collector, "count", lambda: None)()
                                    logging.info(
                                        "ckpt:frame-trim(obs) by row-count %s→%s using f<%d",
                                        obs_before, obs_after, anchor_frame_int
                                    )
                    except Exception:
                        logging.exception("ckpt:frame-trim fallback failed; continuing with row-anchors only.")

                # Cache for downstream logs
                self._last_det_frame = anchor_frame_int
            else:
                logging.info("ckpt:frame-trim skipped (no last_detection_frame in status).")
        except Exception as e:
            logging.exception("ckpt:frame-trim encountered an error: %s", e)

        return loaded_obs, loaded_emb
    
    def audit_missing_embeddings_before_anchor_shot(self) -> dict[int, tuple[int, int]]:
        """
        Enforce invariant:
        - Shots strictly BEFORE the anchor shot:
            * OK if det_rows == 0 (no faces detected).
            * NOT OK if 0 < have < det (some DET rows lack embeddings).
        - Current anchor shot may have have < det (we batch at end-of-shot).
        Returns a map of offending shots -> (det_rows, have_rows).
        """
        out: dict[int, tuple[int,int]] = {}
        if self._last_det_shot is None:
            return out

        for shot in range(1, int(self._last_det_shot)):
            det, have = self._count_det_rows_for_shot(shot)
            # Only flag shots that actually had detections.
            if det > 0 and have < det:
                out[int(shot)] = (det, have)
        return out
        
    def rehydrate_runtime(
        self,
        obs_collector,
        emb_collector,
        *,
        trim_to_anchor: bool = True,
    ) -> dict:
        """
        One-shot rehydration used by the pipeline:
        - Load obs/emb NPZ sidecars (if present).
        - Validate collector schemas.
        - Restore stable track_order.
        - Restore detection anchors and trim collectors back to the last detect frame (optional).
        - Set in-memory pointers so downstream code (e.g. resume-time rehydration of tracks)
            can read from `self.obs_collector` / `self.emb_collector`.
        - Update simple counters to reflect the checkpointed position (frames/shots/tracks).
        Returns a dict summary with anchor & row counts for logging/tests.
        """
        loaded_obs, loaded_emb = self.load_and_anchor_collectors(
            obs_collector, emb_collector, trim_to_anchor=trim_to_anchor
        )

        # Expose collectors for downstream rehydration helpers
        self.obs_collector = obs_collector
        self.emb_collector = emb_collector

        status = self.read_status() or {}

        #set anchor frame
        ldf = status.get("last_detection_frame")
        self._anchor_frame = int(ldf) if ldf else None

        # Restore minimal counters (best-effort; these are for status/telemetry, not logic)
        try:
            self._frames_done = int(status.get("frames_done", 0) or 0)
            self._shots_done  = int(status.get("shots_done", 0) or 0)
            self._tracks_seen = int(status.get("tracks_seen", 0) or 0)
        except Exception:
            pass

        # If counters are behind the anchor, bump frames_done to at least the anchor frame.
        if self._anchor_frame is not None:
            self._frames_done = max(int(self._frames_done or 0), int(self._last_det_frame) + 1)

        # Sanity: ensure the collectors reflect the anchor row counts (already trimmed)
        cur_obs = getattr(obs_collector, "count", lambda: None)() or 0
        cur_emb = getattr(emb_collector, "count", lambda: None)() or 0
        if (self._obs_rows_at_det and cur_obs < self._obs_rows_at_det) or \
        (self._emb_rows_at_det and cur_emb < self._emb_rows_at_det):
            raise ResumeSafetyError(
                "rehydrate: collectors row counts are below anchor after trimming: "
                f"obs={cur_obs} (anchor={self._obs_rows_at_det}) emb={cur_emb} (anchor={self._emb_rows_at_det})"
            )

        # Log a compact track_order summary (determinism of segment IDs)
        try:
            self._log_track_order_summary("rehydrated")
        except Exception:
            pass

        summary = {
            "anchor_frame": self._last_det_frame,
            "anchor_shot": self._last_det_shot,
            "anchor_shot_first_frame": self._last_det_shot_first_frame,
            "obs_rows": cur_obs,
            "emb_rows": cur_emb,
            "track_order_entries": len(self._shot_track_to_order or {}),
        }
        logging.info(
            "rehydrate: anchor=%r shot=%r shot_first=%r rows(obs=%d,emb=%d) track_order=%d",
            summary["anchor_frame"], summary["anchor_shot"], summary["anchor_shot_first_frame"],
            summary["obs_rows"], summary["emb_rows"], summary["track_order_entries"]
        )

        # Guardrail: show any discrepancy if some helper computed a different heuristic.
        try:
            heur = getattr(self, "compute_anchor_from_collectors", None)
            if callable(heur):
                h = heur()
                if isinstance(h, tuple):
                    h = h[0]
                if h is not None and self._last_det_frame is not None and int(h) != int(self._last_det_frame):
                    logging.warning(
                        "ckpt:collector-heuristic anchor=%s disagrees with status.json anchor=%s; ignoring heuristic",
                        h, self._last_det_frame
                    )
        except Exception:
            # purely diagnostic; never fatal
            pass

        # --- Resume-time safety check for embeddings completeness in completed shots ---
        try:
            self._validate_resume_embeddings(
                anchor_shot=self._last_det_shot,
            )
        except ResumeSafetyError:
            # Bubble up as fatal (global-ID resolution requires completeness on past shots)
            raise
        except Exception as e:
            # Non-schema exceptions should not crash resume; log loudly.
            logging.exception("rehydrate: embeddings validation encountered a non-fatal error: %s", e)

        return summary

    # ---------- embeddings validation helpers ----------
    def _det_stats_for_shot(self, shot_num: int) -> tuple[int, int]:
        """Returns (det_with_landmarks, det_with_landmarks_and_emb) for a given shot."""
        if not getattr(self, "obs_collector", None):
            return (0, 0)

        try:
            arr = self.obs_collector.to_array()
        except Exception:
            return (0, 0)

        if getattr(arr, "size", 0) == 0:
            return (0, 0)

        try:
            det_code = int(SRC_TO_CODE[Source.DETECTED])

            # Required columns
            if "shot" not in arr.dtype.names or "src" not in arr.dtype.names:
                return (0, 0)

            mask_shot = (arr["shot"] == int(shot_num))
            mask_det  = (arr["src"] == det_code)

            # Landmark presence mask
            if "landmarks" in (arr.dtype.names or ()):
                landmarks = arr["landmarks"]

                # Case A: fixed-shape numeric array field, e.g. dtype float32 with shape (..., 68, 2)
                # landmarks.ndim >= 2 means there is at least one extra dimension beyond row.
                if isinstance(landmarks, np.ndarray) and landmarks.ndim >= 2 and np.issubdtype(landmarks.dtype, np.number):
                    # Mark as present if ANY value is non-zero for that row.
                    # Collapse all non-row dimensions.
                    nonzero = np.any(landmarks != 0, axis=tuple(range(1, landmarks.ndim)))
                    mask_landmarks = nonzero

                # Case B: object array (per-row python objects: None, list, np.ndarray, etc.)
                elif isinstance(landmarks, np.ndarray) and landmarks.dtype == object:
                    def _present(x) -> bool:
                        if x is None:
                            return False
                        # numpy array
                        if isinstance(x, np.ndarray):
                            return x.size > 0 and np.any(x != 0)
                        # list/tuple
                        if isinstance(x, (list, tuple)):
                            return len(x) > 0
                        # string/bytes
                        if isinstance(x, (str, bytes)):
                            return len(x) > 0 and str(x).strip() not in ("", "[]", "null", "None")
                        # fallback: truthiness
                        return True

                    mask_landmarks = np.fromiter((_present(x) for x in landmarks), dtype=bool, count=landmarks.shape[0])

                # Case C: string/bytes array (rare): treat non-empty/non-"[]"/non-"null" as present
                elif isinstance(landmarks, np.ndarray) and landmarks.dtype.kind in ("U", "S"):
                    s = landmarks.astype(str)
                    mask_landmarks = (np.char.strip(s) != "") & (np.char.strip(s) != "[]") & (np.char.strip(s) != "null")

                else:
                    # Unknown representation: be conservative (treat as not present)
                    mask_landmarks = np.zeros(arr.shape[0], dtype=bool)
            else:
                # If landmarks doesn't exist, you cannot enforce the invariant reliably.
                raise ResumeSafetyError("[ckpt] observations sidecar missing required 'landmarks' field")

            base = mask_shot & mask_det & mask_landmarks
            det_with_landmarks = int(base.sum())
            if det_with_landmarks == 0:
                return (0, 0)

            if "emb_idx" not in (arr.dtype.names or ()):
                return (det_with_landmarks, 0)

            with_emb = int(((arr["emb_idx"] >= 0) & base).sum())
            return (det_with_landmarks, with_emb)

        except Exception:
            # Be cautious; caller enforces policy.
            return (0, 0)
        
    def _shots_with_any_rows(self) -> list[int]:
        try:
            arr = self.obs_collector.to_array()
            if getattr(arr, "size", 0) == 0:
                return []
            return sorted({int(s) for s in np.asarray(arr["shot"]).tolist()})
        except Exception:
            return []

    def _validate_resume_embeddings(self, *, anchor_shot: int | None) -> None:
        """
        Policy:
        - For shots < anchor_shot:
            * If det_with_landmarks == 0: OK (no embedding-eligible DET rows).
            * Else require with_embeddings == det_with_landmarks; otherwise fatal.
        - For shot == anchor_shot: allow partial/zero (mid-shot resume).
        """
        try:
            shots_present = sorted({int(s) for (s, _) in (self._shot_track_to_order or {}).keys()})
        except Exception:
            shots_present = []

        if not shots_present:
            logging.info("resume: no shots present in track_order; skipping embeddings validation.")
            return

        a_shot = int(anchor_shot) if anchor_shot is not None else None

        for s in shots_present:
            if a_shot is None:
                continue
            if s >= a_shot:
                continue  # do NOT enforce on anchor shot or future

            det_rows, with_emb = self._det_stats_for_shot(s)
            if det_rows == 0:
                continue

            if with_emb < det_rows:
                raise ResumeSafetyError(
                    f"resume: embeddings incomplete for completed shot {s}: "
                    f"det_with_landmarks={det_rows}, with_embeddings={with_emb}. "
                    "This would break deterministic global-ID resolution. "
                    "Re-run prior segment or regenerate checkpoints."
                )

            logging.info(
                "resume: shot=%d embeddings coverage OK (det_with_landmarks=%d, with_emb=%d).",
                s, det_rows, with_emb
            )
           
    def hydrate_open_tracks_into(self, aggregator) -> int:
        """
        Read status.json['open_tracks'] and rehydrate them into the provided aggregator.
        Safe no-op if nothing is present.
        """
        try:
            status = self.read_status() or {}
            tracks = status.get("open_tracks") or []
            return int(aggregator.rehydrate_open_tracks(tracks))
        except Exception:
            logging.exception("checkpoint: hydrate_open_tracks_into failed")
            return 0
     
    # ---------- internals ----------
    def _write_status(self, note: str) -> None:
        # derive a stable, ordered list from the dict
        shot_track_order_list = track_order_dict_to_list(self._shot_track_to_order)

        snap = {
            "schema_version": REQUIRED_SCHEMA_VERSION,
            "detect_interval": int(self._cfg.get("detect_interval", 30)),
            "embedding_batch_size_max": int(self._cfg.get("embedding_batch_size_max", 32)),
            "device": self._cfg.get("device", "auto"),
            "emb_store": self._cfg.get("emb_store", "inline"),
            "emb_sidecar_path": self._cfg.get("emb_sidecar_path"),
            "obs_sidecar_path": self._cfg.get("obs_sidecar_path"),
            "detector_model_path": self._cfg.get("detector_model_path", ""),
            "embedding_model_path": self._cfg.get("embedding_model_path", ""),
            "yolo_config_path": self._cfg.get("yolo_config_path", ""),
            "shot_segmentation_path": self._cfg.get("shot_segmentation_path"),
            "checkpoint_dir": str(self.root),
            "track_order": shot_track_order_list,
            "open_tracks": self._open_tracks_inline,
            "log_level": self._cfg.get("log_level", "INFO"),
            "log_file": self._cfg.get("log_file"),
        }

        status = CheckpointStatus(
            last_saved_utc=_utc_now(),
            video_path=self.video_path,
            frames_done=self._frames_done,
            shots_done=self._shots_done,
            tracks_seen=self._tracks_seen,
            obs_rows=(self._obs.count() if self._obs else 0),
            emb_rows=(self._emb.count() if self._emb else 0),
            last_detection_frame=self._last_det_frame,
            last_detection_shot=self._last_det_shot,
            last_detection_shot_first_frame=self._last_det_shot_first_frame,
            obs_rows_at_last_detection=self._obs_rows_at_det,
            emb_rows_at_last_detection=self._emb_rows_at_det,

            **snap,
            note=note,
        )
        _atomic_write_text(self.status_path, status.to_json())
        logging.debug("in checkpoint._write_status - just called _atomic_write_text()")

    def _rewrite_obs_npz_to_flat(self, src_path, dst_path) -> bool:
        """
        Load an observations NPZ from `src_path` and write a 'flat' NPZ to `dst_path`
        with one named array per field (e.g., 'frame', 'shot_id', ...).
        Returns True if a flat file was written, False if we fell back to a raw copy.
        """
        try:
            with np.load(src_path, allow_pickle=True) as npz:
                # Case 1: already flat (has explicit 'frame' key) -> direct copy
                if "frame" in npz.files:
                    shutil.copy2(src_path, dst_path)
                    return True

                # Case 2: single structured array (common for checkpoint collectors)
                # e.g., npz.files == ['arr_0'] or ['obs']
                if len(npz.files) == 1:
                    key = npz.files[0]
                    arr = npz[key]
                    if hasattr(arr, "dtype") and arr.dtype.fields:
                        # Build flat dict: one named array per field
                        flat = {name: arr[name] for name in arr.dtype.names}
                        # Guarantee we at least include 'frame' if present
                        if "frame" in flat:
                            np.savez(dst_path, **flat)
                            return True
                        # If no 'frame' field exists, still write out all fields
                        # (the test only requires that 'frame' be present; if it's
                        # absent in the source, we won't synthesize it.)
                        np.savez(dst_path, **flat)
                        return True

                # Case 3: multiple arrays but none named 'frame' -> try to flatten if any is structured
                made_flat = False
                out = {}
                for k in npz.files:
                    a = npz[k]
                    if hasattr(a, "dtype") and a.dtype.fields:
                        for fname in a.dtype.names:
                            out[fname] = a[fname]
                        made_flat = True
                    else:
                        # keep raw arrays with their existing names
                        out[k] = a
                if made_flat:
                    np.savez(dst_path, **out)
                    return True

        except Exception as e:
            logging.error("ckpt:export rewrite failed: %s", e)

        # Fallback to raw copy (may not satisfy callers that expect 'frame')
        try:
            shutil.copy2(src_path, dst_path)
            return False
        except Exception as e:
            logging.error("ckpt:export fallback copy failed: %s", e)
            return False

    @property
    def snapshots_dir(self) -> Path:
        # self.root is the run directory; self.run_dir is not defined on this class.
        return Path(self.root, "ckpt", "snapshots")
    
    @property
    def snapshots_ready(self) -> bool:
        """True iff snapshotting can be performed (run directory exists)."""
        try:
            return self.root is not None and Path(self.root).exists()
        except Exception:
            return False

    def write_checkpoint_snapshot(self, name: str, payload: dict) -> Path:
        """
        Persist a compact snapshot at detect frames.
        """
        if not self.snapshots_ready:
            raise RuntimeError(
                "ckpt:snapshot attempted before run_dir was initialized. "
                "Call start_new_run() (or equivalent) to establish run_dir."
            )
        try:
            sd = self.snapshots_dir
            sd.mkdir(parents=True, exist_ok=True)
            return write_snapshot_atomic(sd, name, payload)
        except Exception:
            self.logger.exception("ckpt:snapshot failed (name=%s)", name)
            raise

    def compare_rehydrate_to_snapshot(self, *, prior_tracks: list, anchor_frame: int | None) -> None:
        """
        Load the latest snapshot at/before the anchor frame and print a concise diff
        against rehydrated tracks. Safe if nothing to compare.
        """
        try:
            snap = load_latest_snapshot(self.snapshots_dir, up_to_frame=(anchor_frame if anchor_frame is not None else None))
            if not snap:
                self.logger.info("ckpt:snapshot none found for diff (up_to_frame=%s)", anchor_frame)
                return
            lines = diff_snapshot_vs_rehydrate(snap, prior_tracks)
            for line in lines:
                self.logger.info(line)
        except Exception:
            self.logger.exception("ckpt:snapshot diff failed")