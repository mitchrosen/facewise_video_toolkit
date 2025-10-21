from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import tempfile
import typing as _t
import hashlib
from typing import Protocol, List
import numpy as np
import shutil
import logging
from facekit.errors import ResumeSafetyError
from facekit.utils.io import fsync_parent_dir

from facekit.common.obs_consts import Source, SRC_TO_CODE

REQUIRED_SCHEMA_VERSION = "2.1"

class TrackingCheckpoint(Protocol):
    # lifecycle/progress
    def on_frame(self, frame_idx: int) -> None: ...
    def on_shot_done(self) -> None: ...
    def on_new_tracks(self, n: int) -> None: ...

    # checkpointing
    def checkpoint_now(
            self, 
            *,
            frame_idx: int, 
            shot_number: int,
            shot_first_frame: int | None,
            note: str = "checkpoint") -> None: ...

    # persistence
    def add_observations(self, shot_number: int, frame_idx: int, observations: List["FaceObservation"]) -> int: ...
    def add_embeddings(self, shot_number: int, track_id: int, frame_idx_last: int, embs: np.ndarray) -> int: ...

    # resuming
    # Return (frame_idx, shot_number, shot_first_frame) or None for fresh run.
    def get_resume_anchor(self) -> tuple[int, int, int] | None: ...

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

def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dst)
    fsync_parent_dir(dst)

def _atomic_write_text(dst: Path, text: str) -> None:
    _atomic_write_bytes(dst, text.encode("utf-8"))

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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
    # misc
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class CheckpointManager (TrackingCheckpoint):
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
        self.ckpt_dir  = self.root / "ckpt"
        self.video_path = str(Path(video_path).resolve())
        self.status_path = self.root / "status.json"
        self.obs_path    = self.ckpt_dir / "obs_ckpt.npz"
        self.emb_path    = self.ckpt_dir / "emb_ckpt.npz"

        # Create checkpoint dir, if needed
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        self.resume_enabled = bool(resume)

        # Counters
        self._frames_done = 0
        self._shots_done = 0
        self._tracks_seen = 0

        # Detection anchors
        self._last_det_frame: int | None = None
        self._last_det_shot: int | None = None
        self._obs_rows_at_det: int = 0
        self._emb_rows_at_det: int = 0
        self._last_det_shot_first_frame: int | None = None

        # Collector pointers (set via `start`)
        self._obs = None  # ObservationsCollector
        self._emb = None  # EmbeddingCollector

        # Snapshot of CLI/options for safe resume (populated in start())
        self._cfg: dict[str, _t.Any] = {}

    # ---------- public API ----------

    def start(self,
              obs_collector,
              emb_collector,
              *,
              tracks_seen: int = 0,
              shots_done: int = 0,
              frames_done: int = 0,
              options_snapshot: dict[str, _t.Any] | None = None) -> None:
        self._obs = obs_collector
        self._emb = emb_collector
        self._tracks_seen = int(tracks_seen)
        self._shots_done = int(shots_done)
        self._frames_done = int(frames_done)
        self._cfg = dict(options_snapshot or {})
        self._write_status("checkpointing started")

    def on_new_tracks(self, n: int) -> None:
        self._tracks_seen += int(n)

    def on_shot_done(self) -> None:
        self._shots_done += 1
        self._write_status("shot boundary")

    def on_frame(self, frame_idx: int) -> None:
        # Called for every processed frame (0-based)
        self._frames_done = int(frame_idx) + 1
    
    def add_observations(self, shot_number: int, frame_idx: int, observations) -> int:
        """
        Append observations for THIS FRAME into the observations collector.
        ObservationsCollector.append_track_obs expects one track at a time.
        """
        if self._obs is None or not observations:
            return 0

        # Group by track_id (required by ObservationsCollector)
        by_tid = {}
        for obs in observations:
            if obs.track_id is None:
                # Skip unassigned (shouldn't happen here, but be defensive)
                continue
            by_tid.setdefault(int(obs.track_id), []).append(obs)

        total = 0
        for tid, obs_list in by_tid.items():
            rows = []
            for obs in obs_list:
                # Use the source from the observation if present; default to DETECTED
                src_val = getattr(obs, "source", None)
                src_val = src_val if src_val is not None else Source.DETECTED

                rows.append({
                    "shot": int(shot_number),
                    "track_id": int(tid),
                    "f": int(obs.frame_idx),
                    "bbox_xyxy": [float(v) for v in obs.bbox],
                    "src": src_val,
                    **({"conf": float(obs.confidence)} if getattr(obs, "confidence", None) is not None else {}),
                })


            # emb_idx is unknown here; set to -1 via emb_idx_fn
            _, k = self._obs.append_track_obs(rows, emb_idx_fn=lambda _o: -1)
            total += int(k)

        logging.debug(
            "ckpt:add_obs shot=%s frame=%s tracks=%d rows_added=%d",
            shot_number, frame_idx, len(by_tid), total
        )

        return total

    def add_embeddings(self, shot_number: int, track_id: int, frame_idx_last: int, embs: np.ndarray) -> int:
        if self._emb is None or embs is None or embs.size == 0:
            return 0

        embs = np.asarray(embs, dtype=np.float32, order="C")
        if embs.ndim != 2 or embs.shape[1] != 512:
            raise ResumeSafetyError(f"[ckpt] embeddings must be (N,512) float32; got {embs.shape} {embs.dtype}")

        # Require assign() -> int
        assigned = []
        for row in embs:
            emb_idx = self._emb.assign(row)
            if not isinstance(emb_idx, int):
                raise ResumeSafetyError("[ckpt] EmbeddingCollector.assign(vec) must return assigned row index (int).")
            assigned.append(emb_idx)

        cnt = len(assigned)
        logging.debug("ckpt:add_emb shot=%s track_id=%s frame_last=%s vecs=%d total_emb_rows=%d",
                    shot_number, track_id, frame_idx_last, cnt, self._emb.count() if self._emb else -1)

        if self._obs is None or cnt == 0:
            return cnt

        # Require observations collector helpers
        find_rows = getattr(self._obs, "find_rows", None)
        update_emb_idx = getattr(self._obs, "update_emb_idx", None)
        if not callable(find_rows) or not callable(update_emb_idx):
            raise ResumeSafetyError(
                "[ckpt] ObservationsCollector must implement find_rows(...) and update_emb_idx(...)."
            )

        # Find candidate obs rows to link (ascending by frame)
        cand_rows = find_rows(
            shot=int(shot_number),
            track_id=int(track_id),
            frame_last=int(frame_idx_last),
            only_unassigned=True,
            only_with_crop=True,
            source=SRC_TO_CODE[Source.DETECTED],
        )
        if not cand_rows:
            raise ResumeSafetyError(
                f"[ckpt] no candidate observation rows to receive embeddings for shot={shot_number} track={track_id}."
            )

        # Map newest K crops (keep order)
        if len(cand_rows) >= cnt:
            target_rows = cand_rows[-cnt:]
        else:
            raise ResumeSafetyError(
                f"[ckpt] fewer observation rows with crops ({len(cand_rows)}) than embeddings ({cnt}) "
                f"for shot={shot_number} track={track_id}."
            )

        update_emb_idx(positions=target_rows, emb_indices=assigned[-len(target_rows):])

        return cnt

    def checkpoint_now(
            self, 
            *, 
            frame_idx: int, 
            shot_number: int, 
            shot_first_frame: int | None = None,
            note: str = "checkpoint") -> None:
        """
        Persist a restartable checkpoint at the current processing boundary.
        Call this right before you perform a potentially state-changing step,
        so a resume can re-execute that step deterministically.
        The provided frame/shot are recorded as the anchor point.
        """
        if self._obs is None or self._emb is None:
            return

        # Save collectors as they existed BEFORE this detection.
        # Always produce files (even if empty) so resume/findability is stable.
        self._obs.dump_npz(self.obs_path)
        obs_count = self._obs.count()
        self._emb.dump_npz(self.emb_path)
        emb_count = self._emb.count()

        self._last_det_frame = int(frame_idx)
        self._last_det_shot = int(shot_number)
        self._last_det_shot_first_frame = (int(shot_first_frame) if shot_first_frame is not None else None)

        self._obs_rows_at_det = max(0, obs_count)
        self._emb_rows_at_det = max(0, emb_count)

        logging.info(
            "ckpt:snapshot frame=%s shot=%s obs_rows=%d emb_rows=%d files=(%s, %s)",
            frame_idx, shot_number, obs_count, emb_count, self.obs_path, self.emb_path
        )

        self._write_status(note or "checkpoint")

    def matches_video(self, video_path: _t.Union[str, Path]) -> bool:
        """Return True if the stored checkpoint was created for this video path."""
        st = self.read_status()
        return bool(st and str(st.get("video_path")) == str(video_path))

    def finalize(self, note: str = "final") -> None:
        # Ensure sidecars exist even if no detection checkpoint happened.
        if self._obs is not None:
            self._obs.dump_npz(self.obs_path)
        if self._emb is not None:
            self._emb.dump_npz(self.emb_path)
        self._write_status(note)

    @staticmethod
    def compute_parent_dir(default_root: Path, video_path: Path) -> Path:
        return _video_parent_dir(default_root=default_root, video_path=video_path)

    # ---------- resume helpers ----------

    def get_resume_anchor(self):
        if self._last_det_frame is None:
            return None
        return (self._last_det_frame, self._last_det_shot, self._last_det_shot_first_frame)
    
    def resume_available(self) -> bool:
        return self.status_path.exists() and self.obs_path.exists() and self.emb_path.exists()

    def read_status(self) -> dict | None:
        if not self.status_path.exists():
            return None
        try:
            return json.loads(self.status_path.read_text())
        except Exception:
            return None
        
    def _validate_collectors_schema(self, obs_collector, emb_collector) -> None:
        # Helper to get row counts even for test doubles
        def _count_rows(obj) -> int:
            try:
                if hasattr(obj, "count") and callable(obj.count):
                    return int(obj.count() or 0)
            except Exception:
                pass
            # Fallbacks used by simple test doubles
            return int(getattr(obj, "rows", 0) or 0)

        obs_n = _count_rows(obs_collector)
        emb_n = _count_rows(emb_collector)

        # --- Embeddings: only validate if we actually have rows loaded/appended ---
        if emb_n > 0:
            e_shape = getattr(emb_collector, "shape", None)
            e_dtype = getattr(emb_collector, "dtype", None)
            if callable(e_shape): e_shape = e_shape()
            if callable(e_dtype): e_dtype = e_dtype()
            if e_shape is not None:
                if len(e_shape) != 2 or e_shape[1] != 512:
                    raise ResumeSafetyError(f"[ckpt] emb_ckpt schema invalid: shape={e_shape}, expected (N,512)")
            if e_dtype is not None and np.dtype(e_dtype) != np.float32:
                raise ResumeSafetyError(f"[ckpt] emb_ckpt dtype invalid: {e_dtype}, expected float32")

        # --- Observations: only validate if there are rows ---
        if obs_n > 0:
            required = {"shot", "track_id", "f", "bbox_xyxy", "src", "has_crop", "emb_idx"}
            cols = self._resolve_obs_columns(obs_collector)
            if cols:
                missing = required - cols
                if missing:
                    raise ResumeSafetyError(f"[ckpt] obs_ckpt schema missing columns: {sorted(missing)}")
            # If collector doesn’t advertise columns at all (cols == set()), skip strict check here.

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
                logging.info("ckpt:load emb_collector.load_npz() raised exception")
                pass

        logging.info("ckpt:load obs_rows=%d emb_rows=%d from (%s, %s)",
             obs_rows, emb_rows, self.obs_path, self.emb_path)
        return obs_rows, emb_rows
    
        # ---------- run-dir factory ----------

    @classmethod
    def open(
        cls,
        *,
        parent_dir: Path,
        video_path: Path,
        options_snapshot: dict,
        resume: bool,                     # load & trim collectors from NPZs in the chosen dir
        run_id: str | None = None,        # choose this exact run dir
        resume_latest: bool = False,      # otherwise choose newest existing run dir
        force_new_run: bool = False,
    ) -> "CheckpointManager":
        """
        Create or select a checkpoint run directory and return a configured CheckpointManager.

        Run directory selection:
        - If `run_id` is provided, use that exact subdirectory under `parent_dir`.
        - Else if `resume_latest` is True, select the newest existing `run-*` subdirectory.
        - Else create a fresh `run-<timestamp>-<hash>` subdirectory.

        Modes:
        - resume=True:
            Load existing checkpoint artifacts (status.json, ckpt/*.npz) when the selected
            run directory exists. If the selected directory does not exist (e.g. no
            previous runs and `resume_latest=True`), this method will *downgrade* to
            resume=False and log a warning, then proceed with a new run directory.
        - force_new_run=True:
            Start clean in the selected directory by purging checkpoint artifacts
            (deletes `ckpt/` and `status.json`). Mutually exclusive with `resume=True`.

        Side effects:
        - Ensures `parent_dir` exists.
        - When `force_new_run=True` and the selected directory exists, removes
            `run_dir/ckpt` and `run_dir/status.json`, then recreates `run_dir/ckpt`.
        - Writes/updates `status.json` with note "opened".

        Returns:
        CheckpointManager: Instance bound to the chosen `run_dir`. Its `resume_enabled`
        flag reflects the (possibly downgraded) `resume` value.

        Raises:
        FileNotFoundError: If `run_id` is specified but the directory does not exist.
        ValueError:
            - If both `run_id` and `resume_latest` are provided.
            - If both `resume` and `force_new_run` are True.
            - If an existing run is being targeted (`run_id` or `resume_latest`) but neither
            `resume` nor `force_new_run` is True.
        """
         
        # Guard contradictions about selected dir
        if run_id and resume_latest:
            raise ValueError("run_id and resume_latest both set the resume directory, "
                             "can be contradictory and are mutually exclusive.")
        
        # Guard contradictions about behavior
        if resume and force_new_run:
            raise ValueError("force_new_run wipes out resume directory, "
                             "so is mutually exclusive with resume") 
        
         # If targeting an existing dir, caller must choose resume or overwrite
        if (run_id or resume_latest):
            if not (resume or force_new_run): 
                raise ValueError(
                    "When selecting an existing run (run_id or resume_latest), "
                    "either resume=True or force_new_run=True are required."
                )            
        
        # Select run directory
        if run_id:
            run_dir = parent_dir / run_id
            if not run_dir.exists():
                raise FileNotFoundError(f"checkpoint run not found: {run_dir}")
            selected_dir_exists = True
        elif resume_latest:
            runs = sorted([d for d in parent_dir.glob("run-*") if d.is_dir()], key=lambda p: p.name)
            if not runs:
                # no existing; fall back to new
                run_dir = cls._create_new_run_dir(parent_dir, options_snapshot)
                selected_dir_exists = False
            else:
                run_dir = runs[-1]
                selected_dir_exists = True
        else:
            run_dir = cls._create_new_run_dir(parent_dir, options_snapshot)
            selected_dir_exists = False

        # If caller asked to resume but nothing exists, either error or auto-downgrade
        if resume and not selected_dir_exists:
            resume = False
            logging.warning("resume=True but no existing run directory was found to resume from."
                           "Continuing with resume=False.")

        # Purge directory if overwrite set and directory exists
        if selected_dir_exists and force_new_run:
            # purge ckpt artifacts so we truly start fresh
            shutil.rmtree(run_dir / "ckpt", ignore_errors=True)
            (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
            (run_dir / "status.json").unlink(missing_ok=True)
 
        mgr = cls(run_dir, video_path=video_path, resume=resume)
        mgr._cfg = dict(options_snapshot or {})

        logging.info(
            "ckpt: run_dir=%s parent=%s video=%s",
            mgr.root, parent_dir, video_path
        )

        # Only write "opened" when we're starting fresh:
        if not selected_dir_exists or force_new_run:
            try:
                mgr._write_status("opened")
            except Exception:
                pass

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
        st = self.read_status() or {}
        diffs = []
        for k in _PROTECTED_KEYS:
            if str(st.get(k)) != str(current.get(k)):
                diffs.append((k, st.get(k), current.get(k)))
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

        st = self.read_status() or {}

        # hard stop: different video (we’re inside a run for a specific video)
        if str(st.get("video_path")) != str(current.get("video_path")):
            raise ResumeSafetyError(
                f"[resume safety] video_path mismatch: was {st.get('video_path')!r}, now {current.get('video_path')!r}"
            )
        
            # Hard stop: schema mismatch
        sv = str(st.get("schema_version", ""))
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

        # Capture counts BEFORE any trimming for the log
        pre_obs = getattr(obs_collector, "count", lambda: None)()
        pre_emb = getattr(emb_collector, "count", lambda: None)()

        if trim_to_anchor:
            st = self.read_status() or {}

            # Safe fetch with sane defaults
            od = int(st.get("obs_rows_at_last_detection", 0) or 0)
            ed = int(st.get("emb_rows_at_last_detection", 0) or 0)
            lf = st.get("last_detection_frame")  # may be None
            ls = st.get("last_detection_shot")   # may be None
            lfirst = st.get("last_detection_shot_first_frame")

            self._last_det_frame = (int(lf) if lf is not None else None)
            self._last_det_shot  = (int(ls) if ls is not None else None)
            self._last_det_shot_first_frame = (int(lfirst) if lfirst is not None else None)
            self._obs_rows_at_det = int(od)
            self._emb_rows_at_det = int(ed)

            # Nothing to trim to — log and exit
            if (od == 0 and ed == 0):
                logging.info(
                    "ckpt:anchor no anchor present; pre_obs=%s pre_emb=%s loaded_obs=%s loaded_emb=%s",
                    pre_obs, pre_emb, loaded_obs, loaded_emb
                )
                return loaded_obs, loaded_emb

            # If anchor exceeds what we loaded, that indicates corrupted/incomplete NPZ.
            if (pre_obs is not None and od > pre_obs) or (pre_emb is not None and ed > pre_emb):
                logging.warning(
                    "ckpt:anchor anchor beyond loaded rows; "
                    "anchor_obs=%d anchor_emb=%d pre_obs=%s pre_emb=%s frame=%s shot=%s",
                    od, ed, pre_obs, pre_emb, lf, ls
                )
                # Best-effort clamp to what we have
                od = min(od, pre_obs or od)
                ed = min(ed, pre_emb or ed)

            # Do the trimming (guard each independently)
            try:
                if od and hasattr(obs_collector, "trim_to"):
                    obs_collector.trim_to(int(od))
            except Exception as e:
                logging.exception("ckpt:anchor obs trim_to(%d) failed: %s", od, e)

            try:
                if ed and hasattr(emb_collector, "trim_to"):
                    emb_collector.trim_to(int(ed))
            except Exception as e:
                logging.exception("ckpt:anchor emb trim_to(%d) failed: %s", ed, e)

            # Post-trim counts
            post_obs = getattr(obs_collector, "count", lambda: None)()
            post_emb = getattr(emb_collector, "count", lambda: None)()

            # Final summary line — this is the one you usually watch
            logging.info(
                "ckpt:anchor trimmed obs %s→%s (anchor=%d) emb %s→%s (anchor=%d) @ frame=%s shot=%s",
                pre_obs, post_obs, od, pre_emb, post_emb, ed, lf, ls
            )
        else:
            logging.debug(
                "ckpt:anchor trimming disabled; loaded_obs=%s loaded_emb=%s", loaded_obs, loaded_emb
            )

        return loaded_obs, loaded_emb


    # ---------- internals ----------

    def _write_status(self, note: str) -> None:
        # Safe defaults if start() wasn't called with a snapshot
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

