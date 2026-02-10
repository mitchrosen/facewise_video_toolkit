from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Mapping, Literal, Union, Callable
from typing import cast
from pathlib import Path
import numpy as np
import json
import os
import subprocess
import hashlib
import logging
from datetime import datetime, timezone
from facekit.common.obs_consts import (
    Source,
    src_to_code,
    code_to_src,
)
from facekit.utils.io import atomic_write_npz, atomic_write_npy
from facekit.tracking.face_structures import FaceObservation
from facekit.pipeline.checkpoint import CheckpointManager

XYXY = Tuple[float, float, float, float]

SCHEMA_VERSION_V2_0 = "2.0"
SCHEMA_VERSION_V2_1 = "2.1"

@dataclass
class V2WriterConfig:
    video_path: Optional[str] = None
    video_size: Optional[Tuple[int, int]] = None  # (W, H)
    total_frames: Optional[int] = None
    fps: Optional[float] = None
    normalize_to_percent: bool = True
    static_stddev_thresh_pct: float = 1.5
    emb_store: Literal["inline", "sidecar"] | None = "inline"  # None = don’t serialize embeddings
    emb_sidecar_path: Path | None = None  # only used for sidecar mode


def _bbox_to_center_size_xyxy(bbox: XYXY) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1)
    h = (y2 - y1)
    return cx, cy, w, h

def _clip_bbox_xyxy(
    b: Tuple[float, float, float, float], W: int, H: int
) -> Tuple[float, float, float, float]:
    """
    Clip bbox to frame bounds for summary statistics only.
    Raw per-frame bboxes remain untouched.
    """
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(float(W), x1))
    x2 = max(0.0, min(float(W), x2))
    y1 = max(0.0, min(float(H), y1))
    y2 = max(0.0, min(float(H), y2))
    # Ensure correct ordering after clipping
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)

def _normalize(cx: float, cy: float, w: float, h: float, W: int, H: int, as_percent: bool) -> Tuple[float,float,float,float]:
    if not W or not H:
        return cx, cy, w, h
    if as_percent:
        return (cx * 100.0 / W, cy * 100.0 / H, w * 100.0 / W, h * 100.0 / H)
    else:
        return (cx / W, cy / H, w / W, h / H)

def _track_label(track: Any) -> str:
    gid = getattr(track, "global_id", None)
    sid = getattr(track, "segment_id", None)
    if gid is not None:
        return f"face_{gid}"
    if sid is not None:
        return f"face_{sid}"
    tid = getattr(track, "track_id", None)
    return f"face_{tid if tid is not None else 0}"

def _track_first_last(track: Any) -> Tuple[int, int]:
    if hasattr(track, "first_frame") and callable(getattr(track, "first_frame")):
        f0 = int(track.first_frame())
    else:
        obs = getattr(track, "observations", [])
        f0 = int(obs[0].frame_idx) if obs else 0
    if hasattr(track, "last_frame") and callable(getattr(track, "last_frame")):
        f1 = int(track.last_frame())
    else:
        obs = getattr(track, "observations", [])
        f1 = int(obs[-1].frame_idx) if obs else -1
    return f0, f1

def _track_center_series(track: Any) -> List[Tuple[float,float]]:
    centers = []
    for obs in getattr(track, "observations", []):
        x1, y1, x2, y2 = obs.bbox
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return centers

def derive_face_metadata_from_observations(
    obs_collector: "ObservationsCollector",
    tracks: List[Any],
) -> List[Dict[str, Any]]:
    """
    Compute face_metadata by counting rows in the *observations* source of truth.
    This is deterministic across resume and golden runs.
    """
    # Build a stable map (shot_id, track_id) -> face_label from tracks
    id_to_label: Dict[Tuple[int, int], str] = {}
    for t in tracks:
        shot_id = int(getattr(t, "shot_id", 1))
        track_id = int(getattr(t, "track_id", -1))
        id_to_label[(shot_id, track_id)] = _track_label(t)

    counts: Dict[str, int] = {}
    # Iterate obs grouped by (shot, track_id); rows is ascending by frame
    for s, tid, rows in obs_collector.iter_tracks():
        label = id_to_label.get((int(s), int(tid)))
        if not label:
            # If a (shot,track) has no label mapping (should be rare), skip it
            # rather than inventing a transient label that would harm determinism.
            continue
        counts[label] = counts.get(label, 0) + int(len(rows))

    return [{"face_label": k, "occurance_count": int(v)} for k, v in sorted(counts.items())]


def _derive_face_metadata_from_tracks(shots_out: list[dict]) -> list[dict]:
    """
    Derive metadata straight from the *manifest's* tracks (v2.1):
    sum obs_count per face_label across all shots.
    This avoids any dependence on in-memory, pre-merge tracks and is resume-stable.
    """
    counts: dict[str, int] = {}
    for shot in shots_out or []:
        for t in shot.get("face_tracks", []):
            lbl = t.get("face_label")
            cnt = int(t.get("obs_count", 0))
            if not lbl:
                continue
            counts[lbl] = counts.get(lbl, 0) + cnt
    return [{"face_label": k, "occurance_count": int(v)} for k, v in sorted(counts.items())]


def _git_info(repo_dir: str | Path = ".") -> tuple[Optional[str], Optional[str]]:
    # Try git CLI
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        return commit, branch
    except Exception:
        pass
    # CI envs
    commit = os.getenv("GITHUB_SHA") or os.getenv("CI_COMMIT_SHA")
    branch = os.getenv("GITHUB_REF_NAME") or os.getenv("CI_COMMIT_BRANCH")
    if commit or branch:
        return commit, branch

    # Parse .git/HEAD best-effort
    try:
        head = Path(repo_dir, ".git", "HEAD")
        if head.exists():
            ref_line = head.read_text().strip()
            if ref_line.startswith("ref:"):
                rel = ref_line.split(" ", 1)[1]  # e.g. refs/heads/main
                ref_path = Path(repo_dir, ".git", rel)
                commit_full = ref_path.read_text().strip() if ref_path.exists() else None
                branch = Path(rel).name
                return (commit_full[:7] if commit_full else None), branch
            else:
                # detached head: HEAD contains a commit hash
                commit_full = ref_line
                return (commit_full[:7] if commit_full else None), "HEAD"
    except Exception:
        pass

    return None, None

def _compute_params_hash(generation: Dict[str, Any]) -> str:
    filt = {k: v for k, v in generation.items() if k != "created_utc"}
    s = json.dumps(filt, sort_keys=True, separators=(",",":"))
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()

def _emb_store_token(mode):
    return mode if mode in ("inline", "sidecar") else "none"

def get_legacy_last_frame(obs_npz_path: Union[str, Path]) -> Optional[int]:
    p = Path(obs_npz_path)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as data:
        arr = data.get("observations")
        if arr is None or arr.size == 0:
            return None
        return int(np.asarray(arr)["f"].max())
    
def _src_to_int(src_any) -> int:
    """
    Normalize source into an integer code:
      - int: pass-through (validate range in caller if needed)
      - Source: convert via .value -> src_to_code
      - str: convert via src_to_code
    Raises TypeError on unsupported types.
    """
    if isinstance(src_any, int):
        return int(src_any)
    if isinstance(src_any, Source):
        return int(src_to_code(src_any.value))
    if isinstance(src_any, str):
        return int(src_to_code(src_any))
    raise TypeError(f"'src' must be int|str|Source, not {type(src_any).__name__}: {src_any!r}")

def _to_int01(flag) -> int:
    # normalize truthy → 1, falsy → 0
    return 1 if bool(flag) else 0

ObsRow = np.dtype([
    ("f", np.int32),
    ("shot", np.int32),
    ("track_id", np.int32),
    ("bbox_xyxy", np.float32, (4,)),
    ("src", np.uint8),
    ("conf", np.float32),
    ("emb_idx", np.int32),
])

class ObservationsCollector:
    """Collects per-frame observations into a single structured array."""
    @property
    def columns(self) -> tuple[str, ...]:
        # Must match ObsRow exactly
        return ("f","shot","track_id","bbox_xyxy","src","conf","emb_idx")
    
    @property
    def schema(self) -> dict:
        return {"fields": list(self.columns)}
    
    def __init__(self) -> None:
        self._rows: List[np.ndarray] = []   # list of (k,) structured arrays
        self._count: int = 0
        # Cache for deterministic "sidecar order" view:
        # sidecar order is track-contiguous: (shot, track_id, frame)
        self._sorted_cache: np.ndarray | None = None
        self._slice_index: dict[tuple[int, int], tuple[int, int]] | None = None

    def _invalidate_index(self) -> None:
        self._sorted_cache = None
        self._slice_index = None

    def reset(self) -> None:
        self._rows.clear()
        self._count = 0
        self._invalidate_index()

    def find_rows(
        self,
        *,
        shot: int,
        track_id: int,
        frame_last: int | None = None,
        count: int | None = None,
        # optional filters
        only_unassigned: bool | None = None,
        source: int | None = None,
        **kwargs,
    ) -> list[tuple[int, int]]:
        """
        Return up to `count` positions [(block_idx, row_idx), ...] of rows that match:
        shot == shot AND track_id == track_id
        AND (emb_idx == -1 if only_unassigned)
        AND (f <= frame_last if provided)
        AND (src == source if provided)

        Ordering:
        Rows are returned newest→oldest (descending frame index), based on current storage order.

        Notes:
        - Accepts legacy alias `frame_max` via **kwargs.
        - Callers should not rely on ordering for correctness; use frame_at_pos(pos) if needed.
        """

        # Back-compat alias: some callers used frame_max
        if frame_last is None and "frame_max" in kwargs and kwargs["frame_max"] is not None:
            frame_last = int(kwargs["frame_max"])

        want = int(count) if count is not None else None
        out: list[tuple[int, int]] = []

        def _bbox_valid(bb):
            # bb is shape (4,) as float32: x1,y1,x2,y2
            return float(bb[2]) > float(bb[0]) and float(bb[3]) > float(bb[1])

        # Walk blocks from newest to oldest
        for b_idx in range(len(self._rows) - 1, -1, -1):
            block = self._rows[b_idx]
            for r_idx in range(block.shape[0] - 1, -1, -1):
                row = block[r_idx]
                if row["shot"] != shot or row["track_id"] != track_id:
                    continue
                if frame_last is not None and row["f"] > frame_last:
                    continue
                if only_unassigned and row["emb_idx"] != -1:
                    continue
                if source is not None and int(row["src"]) != int(source):
                    continue

                out.append((b_idx, r_idx))
                if want is not None and len(out) >= want:
                    return out
        return out

    def update_emb_idx(
        self,
        positions: list[tuple[int, int]],
        emb_indices: list[int] | tuple[int, ...] | np.ndarray,
    ) -> int:
        """
        Write the given emb_idx values into the provided (block_idx, row_idx) positions.

        Contract:
          - positions: list of (block_idx, row_idx) tuples
          - emb_indices: 1D sequence of ints (list/tuple/ndarray)
          - len(positions) == len(emb_indices)

        Returns:
          Number of rows updated.
        """

        # --- Validate positions ---
        if not isinstance(positions, list):
            raise TypeError(
                f"update_emb_idx: positions must be list[tuple[int,int]], "
                f"got {type(positions).__name__}"
            )

        norm_positions: list[tuple[int, int]] = []
        for pos in positions:
            if not (isinstance(pos, tuple) and len(pos) == 2):
                raise TypeError(
                    f"update_emb_idx: each position must be (block_idx, row_idx), "
                    f"got {pos!r}"
                )
            b_idx, r_idx = pos
            norm_positions.append((int(b_idx), int(r_idx)))

        # --- Normalize emb_indices to a list[int] ---
        if isinstance(emb_indices, np.ndarray):
            if emb_indices.ndim != 1:
                raise ValueError(
                    f"update_emb_idx: emb_indices must be 1D, got shape {emb_indices.shape}"
                )
            emb_list = [int(x) for x in emb_indices]
        elif isinstance(emb_indices, (list, tuple)):
            emb_list = [int(x) for x in emb_indices]
        else:
            raise TypeError(
                f"update_emb_idx: emb_indices must be list/tuple/ndarray of ints, "
                f"got {type(emb_indices).__name__}"
            )

        if len(norm_positions) != len(emb_list):
            raise ValueError(
                f"update_emb_idx length mismatch: positions={len(norm_positions)}, "
                f"emb_indices={len(emb_list)}"
            )

        # --- Apply updates ---
        for (b_idx, r_idx), emb_idx in zip(norm_positions, emb_list):
            self._rows[b_idx]["emb_idx"][r_idx] = int(emb_idx)

        return len(norm_positions)

    def append_track_obs(
        self,
        obs_items: List[Dict[str, Any]],
        emb_idx_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
        ) -> Tuple[int, int]:
        """
        Append obs for one track; returns (offset, count).
        Accepts dict-like or object-like observations.
          - Each item MUST provide: shot (int), track_id (int), f (int),
            bbox_xyxy (len=4) OR bbox_xywh (len=4), src in VALID_SOURCES.
          - conf is optional; emb_idx is computed via emb_idx_fn (or -1).
        Raises ValueError on any missing/invalid field.
        """
        if not obs_items:
            return (self._count, 0)

        k = len(obs_items)
        block = np.empty(k, dtype=ObsRow)

        def _validate_faceObs_dict(row: dict):
            if not isinstance(row, dict):
                raise TypeError("ObservationsCollector.append_track_obs expects normalized dicts.")

            missing = []
            # use .get to avoid KeyError so we can report everything at once
            if row.get("shot") is None:       missing.append("shot_id")
            if row.get("track_id") is None:   missing.append("track_id")
            if row.get("f") is None:          missing.append("frame_idx")
            if ("bbox_xyxy" not in row) and ("bbox_xywh" not in row):
                missing.append("bbox")
            if row.get("src") is None:
                missing.append("source")

            if missing:
                raise ValueError(f"Observation missing required fields: {missing} | row={row!r}")

            # --- normalize src to int code  ---
            return _src_to_int(row.get("src"))
        
        for i, obs in enumerate(obs_items):
            row = obs if isinstance(obs, dict) else dict(obs)  # ensure dict-like
            src_any = _validate_faceObs_dict(row)

            f = int(row["f"])
            shot = int(row["shot"])
            if "bbox_xyxy" in row and row["bbox_xyxy"] is not None:
                bb = [float(x) for x in row["bbox_xyxy"]]
            else:
                x, y, w, h = row["bbox_xywh"]
                bb = [float(x), float(y), float(x + w), float(y + h)]

            src_code =_src_to_int(src_any)
            conf = float(row["conf"]) if ("conf" in row and row["conf"] is not None) else np.nan

            emb_idx = -1
            if emb_idx_fn is not None:
                try:
                    emb_idx = int(emb_idx_fn(row))
                except Exception:
                    emb_idx = -1

            block[i]["f"] = f
            block[i]["shot"] = shot 
            block[i]["track_id"] = int(row["track_id"])
            block[i]["bbox_xyxy"] = bb
            block[i]["src"] = src_code
            block[i]["conf"] = conf
            block[i]["emb_idx"] = emb_idx

        offset = self._count
        self._rows.append(block)
        self._count += k
        self._invalidate_index()
        return (offset, k)
    
    def _sorted_array_for_sidecar(self) -> np.ndarray:
        """
        Return all collected rows, deterministically ordered for writing sidecars
        and computing obs_offset/obs_count slices:
          primary: shot
          secondary: track_id
          tertiary: frame index
        """
        if self._sorted_cache is not None:
            return self._sorted_cache

        arr = np.concatenate(self._rows, axis=0) if self._rows else np.empty(0, dtype=ObsRow)
        if arr.size == 0:
            self._sorted_cache = arr
            return arr

        # lexsort uses last key as primary, so order keys reversed:
        # primary shot, then track_id, then frame => keys (f, track_id, shot)
        idx = np.lexsort((arr["f"], arr["track_id"], arr["shot"]))
        arr_sorted = arr[idx].copy()
        self._sorted_cache = arr_sorted
        return arr_sorted

    def _build_slice_index(self) -> dict[tuple[int, int], tuple[int, int]]:
        """
        Build mapping (shot, track_id) -> (offset, count) in the deterministic
        sidecar ordering.
        """
        if self._slice_index is not None:
            return self._slice_index

        arr = self._sorted_array_for_sidecar()
        out: dict[tuple[int, int], tuple[int, int]] = {}
        if arr.size == 0:
            self._slice_index = out
            return out

        # Single pass because arr is already grouped by (shot, track_id)
        cur_key: tuple[int, int] | None = None
        start = 0
        for i in range(arr.shape[0]):
            key = (int(arr["shot"][i]), int(arr["track_id"][i]))
            if cur_key is None:
                cur_key = key
                start = i
                continue
            if key != cur_key:
                out[cur_key] = (int(start), int(i - start))
                cur_key = key
                start = i
        if cur_key is not None:
            out[cur_key] = (int(start), int(arr.shape[0] - start))

        self._slice_index = out
        return out

    def finalize_sidecar(
        self,
        out_path: Path,
        *,
        min_frame_exclusive: int | None = None,
    ) -> Dict[str, Any]:
        """
        Atomically write observations to an .npz (key: 'observations').
        Always writes a valid (possibly empty) structured array.
        """

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        arr = self._sorted_array_for_sidecar()
        if min_frame_exclusive is not None and arr.size:
            arr = arr[arr["f"] > int(min_frame_exclusive)]

        final_path = out_path if out_path.suffix.lower() == ".npz" else out_path.with_suffix(".npz")

        atomic_write_npz(final_path, observations=arr)

        return {
            "path": str(final_path),
            "format": "npz",
            "dtype": "structured",
            "fields": [
                {"name":"shot","type":"i4","desc":"shot number"},
                {"name":"track_id","type":"i4","desc":"per-shot track id"},
                {"name":"f","type":"i4","desc":"frame index"},
                {"name":"bbox_xyxy","type":"f4[4]","desc":"x1,y1,x2,y2"},
                {"name":"src","type":"u1","desc":"0=detected,1=tracked,2=flow"},
                {"name":"conf","type":"f4","desc":"NaN if absent"},
                {"name":"emb_idx","type":"i4","desc":"-1 if absent"},
            ],
            "count": int(arr.shape[0]),
        }

    def dump_npz(
        self,
        out_path: Union[str, Path],
        *,
        min_frame_exclusive: int | None = None,
    ) -> str:
        final = Path(out_path).with_suffix(".npz")
        arr = self._sorted_array_for_sidecar()
        if min_frame_exclusive is not None and arr.size:
            arr = arr[arr["f"] > int(min_frame_exclusive)]
        atomic_write_npz(final, observations=arr)
        return str(final)

    def load_npz(self, npz_path: Union[str, Path]) -> int:
        p = Path(npz_path)
        if not p.exists():
            return 0
        with np.load(p, allow_pickle=False) as data:
            arr = data.get("observations")
            if arr is None:
                return 0
            arr = np.asarray(arr, dtype=ObsRow)
        if arr.size == 0:
            return 0
        self._rows.append(arr.copy())
        self._count += int(arr.shape[0])
        self._invalidate_index()
        return int(arr.shape[0])

    def count(self) -> int:
        """Number of observation rows collected so far."""
        return self._count
    
    def trim_to(self, n: int) -> None:
        """Trim total rows to exactly n (drop newest first)."""
        n = max(0, int(n))
        if not self._rows:
            self._count = 0
            return
        cur = sum(int(r.shape[0]) for r in self._rows)
        if n >= cur:
            self._count = cur
            return
        # drop from the tail
        while self._rows and cur > n:
            last = self._rows[-1]
            if cur - last.shape[0] >= n:
                cur -= last.shape[0]
                self._rows.pop()
            else:
                keep = n - (cur - last.shape[0])
                self._rows[-1] = last[:keep].copy()
                cur = n
        self._count = cur
        self._invalidate_index()

    def iter_tracks(
        self,
        *,
        frame_max: int | None = None,
        shot: int | None = None,
        track_id: int | None = None,
    ):
        """
        Iterate grouped observations as (shot, track_id, rows) triples.
        - Coalesces across all internal blocks.
        - Optional filters: frame_max (inclusive), shot, track_id.
        - rows are dicts with at least: f, bbox_xyxy, src (Source enum), and optional conf.
        Order: by (shot, track_id), and rows are ascending by frame.
        """
        from facekit.common.obs_consts import CODE_TO_SRC  # already present
        groups: dict[tuple[int,int], list[dict]] = {}

        # gather across blocks
        for block in self._rows:
            for row in block:
                s = int(row["shot"])
                t = int(row["track_id"])
                if shot is not None and s != int(shot):
                    continue
                if track_id is not None and t != int(track_id):
                    continue
                f = int(row["f"])
                if frame_max is not None and f > int(frame_max):
                    continue

                d = {
                    "f": f,
                    "bbox_xyxy": [
                        float(row["bbox_xyxy"][0]),
                        float(row["bbox_xyxy"][1]),
                        float(row["bbox_xyxy"][2]),
                        float(row["bbox_xyxy"][3]),
                    ],
                    # API/UI boundary: expose enum
                    "src": CODE_TO_SRC[int(row["src"])],
                }
                conf = float(row["conf"])
                if not np.isnan(conf):
                    d["conf"] = conf

                groups.setdefault((s, t), []).append(d)

        # sort rows within each group by frame, then yield once per group
        for (s, t) in sorted(groups.keys()):
            rows = groups[(s, t)]
            rows.sort(key=lambda r: r["f"])
            yield (s, t, rows)


    def to_array(self) -> np.ndarray:
        """Return a single structured array similar to what finalize_sidecar writes."""
        return (np.concatenate(self._rows, axis=0)
                if self._rows else np.empty(0, dtype=ObsRow))
    
    def slice_for_track(self, shot: int, track_id: int) -> tuple[int, int]:
        """
        Return (offset, count) for all rows in this collector belonging to (shot, track_id).
        Offsets are in the deterministic sidecar ordering: (shot, track_id, frame).
        """
        idx = self._build_slice_index()
        key = (int(shot), int(track_id))
        if key not in idx:
            return (0, 0)
        return idx[key]

    def frame_at_pos(self, pos: tuple[int,int]) -> int:
        b_idx, r_idx = pos
        return int(self._rows[b_idx]["f"][r_idx])
    
    def trim_to_frame(self, max_frame_inclusive: int) -> None:
        maxf = int(max_frame_inclusive)
        if not self._rows:
            self._count = 0
            return
        kept_blocks = []
        new_count = 0
        for block in self._rows:
            if block.size == 0:
                continue
            mask = block["f"] <= maxf
            if mask.all():
                kept_blocks.append(block)
                new_count += int(block.shape[0])
            elif mask.any():
                sub = block[mask].copy()
                kept_blocks.append(sub)
                new_count += int(sub.shape[0])
        self._rows = kept_blocks
        self._count = new_count
        self._invalidate_index()

class EmbeddingCollector:
    """
    Handles inline vs sidecar storage of embeddings.

    Runtime/Checkpointing contract:
      - assign(vec) MUST return the assigned row index (int), or -1 if not stored.
        (This is what CheckpointManager expects.)

    JSON/Manifest contract:
      - fields_for_json(vec) returns a dict to attach to the obs:
          * {"embedding": [...]} for inline
          * {"emb_idx": i} for sidecar
          * {} when storage is disabled / vec is None
    """
    def __init__(
        self,
        mode: Literal["inline","sidecar"] | None,
        dim: int | None = None,
        *,
        base_offset: int = 0,
    ):
        self.mode = mode
        self.dim = dim
        self._embs: List[np.ndarray] = []
        self._base = int(base_offset)

    def _validate_vec(self, vec: np.ndarray | None) -> np.ndarray:
        if vec is None:
            # Treat as "nothing to do"
            return None  # type: ignore
        v = np.asarray(vec, dtype=np.float32).ravel()
        if self.dim is not None and v.size != self.dim:
            raise ValueError(f"Embedding dim mismatch; expected {self.dim}, got {v.size}")
        return v

    # === RUNTIME API expected by CheckpointManager ====================================
    def assign(self, vec: np.ndarray | None) -> int:
        """
        Add one embedding vector to sidecar storage and return its row index.
        Returns -1 if we are not storing embeddings (inline/None).
        """
        if self.mode is None:
            return -1  # embeddings disabled entirely

        v = self._validate_vec(vec)
        if v is None:
            return -1

        if self.mode == "inline":
            # Inline mode does not maintain an index in the sidecar; manifest will inline.
            return -1

        # sidecar
        idx = len(self._embs)
        self._embs.append(v.copy())
        return int(self._base + idx)

    # === JSON/Manifest helper =========================================================
    def fields_for_json(self, vec: np.ndarray | None) -> Dict[str, Any]:
        """
        Return the JSON fields to attach to an observation for this vector.
        - inline: {"embedding": [...]}
        - sidecar: {"emb_idx": i} (also records the vector via assign)
        - none/vec is None: {}
        """
        if self.mode is None or vec is None:
            return {}

        v = self._validate_vec(vec)
        if v is None:
            return {}

        if self.mode == "inline":
            return {"embedding": v.tolist()}

        # sidecar: ensure the row is recorded and return the index
        idx = self.assign(v)
        return {"emb_idx": int(idx)} if idx >= 0 else {}

    def finalize_sidecar(self, out_path: Path) -> Dict[str, Any]:
        """
        Atomic write of embeddings sidecar.
        - write .npz with key 'embeddings'
        """
        if self.mode != "sidecar":
            return {}

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self._embs:
            arr = np.vstack(self._embs).astype(np.float32, copy=False)
        else:
            arr = np.zeros((0, int(self.dim or 0)), dtype=np.float32)

        if out_path.suffix.lower() == ".npy":
            final_path = Path(atomic_write_npy(out_path, arr))
            fmt = "npy"
        else:
            final_path = Path(atomic_write_npz(out_path.with_suffix(".npz"), embeddings=arr))
            fmt = "npz"

        return {
            "path": str(final_path),
            "format": fmt,                 # <-- keep 'format' for the tests
            "dtype": "float32",
            "dim": int(arr.shape[1]) if arr.ndim == 2 else int(self.dim or 0),
            "count": int(arr.shape[0]),
        }

    def count(self) -> int:
        return len(self._embs)

    def dump_npz(self, out_path: Union[str, Path]) -> str:
        """Atomically write current embeddings as NPZ to out_path and return the final path."""
        final = Path(out_path).with_suffix(".npz")
        arr = self.to_array()
        atomic_write_npz(final, embeddings=arr)
        return str(final)

    def load_npz(self, npz_path: Union[str, Path]) -> int:
        """Append vectors from an NPZ file into this collector; returns rows loaded."""
        p = Path(npz_path)
        if not p.exists():
            return 0
        with np.load(p, allow_pickle=False) as data:
            arr = data.get("embeddings")
        if arr is None:
            return 0
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            return 0
        if self.dim is not None and arr.shape[1] != self.dim:
            raise ValueError(f"EmbeddingCollector.load_npz: dim mismatch {arr.shape[1]} != {self.dim}")
        self._embs.extend(row.copy() for row in arr)
        return int(arr.shape[0])

    def trim_to(self, n: int) -> None:
        n = max(0, int(n))
        if len(self._embs) <= n:
            return
        del self._embs[n:]

    def to_array(self) -> np.ndarray:
        if not self._embs:
            return np.zeros((0, int(self.dim or 0)), dtype=np.float32)
        return np.vstack(self._embs).astype(np.float32, copy=False)

    def get_many(self, indices) -> np.ndarray:
        idx = [int(i) for i in (indices or []) if int(i) >= 0]
        if not idx:
            return np.zeros((0, int(self.dim or 0)), dtype=np.float32)
        arr = self.to_array()
        return arr[idx, :] if arr.size else np.zeros((0, int(self.dim or 0)), dtype=np.float32)

def build_generation_from_objects(
    *,
    detector: Optional[Any] = None,
    embedder: Optional[Any] = None,
    tracking_params: Optional[Mapping[str, Any]] = None,
    validator: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Build the `generation` dict purely from live objects/settings.
    Excludes volatile fields from the params hash (e.g., created_utc).
    """
    commit, branch = _git_info()
    gen: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit or "unknown",
        "branch": branch or "unknown",
    }
    if detector and hasattr(detector, "provenance"):
        gen["detector"] = detector.provenance()
    if embedder and hasattr(embedder, "provenance"):
        gen["embedder"] = embedder.provenance()
    if tracking_params:
        gen["tracking"] = dict(tracking_params)
    if validator and hasattr(validator, "provenance"):
        gen["validator"] = validator.provenance()
    # hash only stable parts
    stable = {k: v for k, v in gen.items() if k not in ("created_utc",)}
    gen["params_hash"] = _compute_params_hash(stable)
    return gen

def build_generation(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Construct the 'generation' block. Commit/branch are derived here; any values
    provided in 'overrides' are merged (and take precedence only if non-empty).
    """
    commit, branch = _git_info()
    gen = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "emb_store": "inline",
    }

    if overrides:
        # Only overwrite commit/branch if caller provides explicit non-empty values.
        if overrides.get("commit"):
            gen["commit"] = overrides["commit"]
        if overrides.get("branch"):
            gen["branch"] = overrides["branch"]
        for k, v in overrides.items():
            if k not in ("commit", "branch"):
                gen[k] = v

    gen["params_hash"] = _compute_params_hash(gen)
    return gen



def build_v2_manifest_from_tracks(
        tracks: List[Any],
        cfg: V2WriterConfig,
        *,
        face_metadata: list[dict[str, Any]] | None = None,
        generation: dict[str, Any] | None = None,
        detector: Any | None = None,
        embedder: Any | None = None,
        tracking_params: Mapping[str, Any] | None = None,
        validator: Any | None = None,
        collector: EmbeddingCollector | None = None, 
) -> Dict[str, Any]:
    """
    returns manifest.
    - If cfg.emb_store is 'inline', embeddings are embedded into obs.
    - If cfg.emb_store is 'sidecar', obs carry 'emb_idx' and vectors are kept
      in the returned EmbeddingCollector for the caller to materialize.
    - If cfg.emb_store is None, embeddings are ignored.
    """
    embc = collector
    if embc is None:
        # create a throwaway for formatting only; will not get finalized
        embc = EmbeddingCollector(cfg.emb_store, dim=512)

    shots_map: Dict[int, List[Any]] = {}
    for t in tracks:
        shot_id = int(getattr(t, "shot_id", 1))
        shots_map.setdefault(shot_id, []).append(t)

    video = {}
    if cfg.video_path: video["path"] = str(cfg.video_path)
    if cfg.fps is not None: video["fps"] = float(cfg.fps)
    if cfg.video_size is not None: video["size"] = [int(cfg.video_size[0]), int(cfg.video_size[1])]
    if cfg.total_frames is not None: video["total_frames"] = int(cfg.total_frames)

    W, H = (cfg.video_size or (0, 0))

    shots_out: List[Dict[str, Any]] = []
    for shot_number in sorted(shots_map.keys()):
        tracks_out: List[Dict[str, Any]] = []
        for t in shots_map[shot_number]:
            f0, f1 = _track_first_last(t)
            obs = getattr(t, "observations", [])
            if obs:
                xs1 = sum(o.bbox[0] for o in obs) / len(obs)
                ys1 = sum(o.bbox[1] for o in obs) / len(obs)
                xs2 = sum(o.bbox[2] for o in obs) / len(obs)
                ys2 = sum(o.bbox[3] for o in obs) / len(obs)
                avg_bbox = (xs1, ys1, xs2, ys2)
            else:
                avg_bbox = (0.0, 0.0, 0.0, 0.0)

            # Track drift can push avg bbox outside frame; clip summary bbox.
            avg_bbox = _clip_bbox_xyxy(avg_bbox, W, H) if (W and H) else avg_bbox
            cx, cy, w, h = _bbox_to_center_size_xyxy(avg_bbox)
            cx, cy, w, h = _normalize(cx, cy, w, h, W, H, cfg.normalize_to_percent)

            centers = _track_center_series(t)
            if centers and W and H and np is not None:
                cx_series = np.array([c[0] * (100.0/W if cfg.normalize_to_percent else 1.0/W) for c in centers], dtype=float)
                cy_series = np.array([c[1] * (100.0/H if cfg.normalize_to_percent else 1.0/H) for c in centers], dtype=float)
                std_c = float((np.var(cx_series) + np.var(cy_series)) ** 0.5)
            else:
                std_c = 0.0
            is_static = std_c < cfg.static_stddev_thresh_pct

            # Normalize once and then adapt to v2.0 per-frame JSON shape
            shot_val = int(getattr(t, "shot_id", shot_number))
            track_val = int(getattr(t, "track_id", -1))
            items_out = normalize_obs_items_for_output(
                obs, shot_id=shot_val, track_id=track_val, emb_collector=embc
            )
            obs_json: List[Dict[str, Any]] = []
            for d in items_out:
                j = {
                    "f": d["f"],
                    "bbox_xyxy": d["bbox_xyxy"],
                    "src": d["src"],
                }
                if "conf" in d:
                    j["conf"] = d["conf"]
                # Inline or emb_idx already encoded by assign(); just pass through if present.
                if "embedding" in d:
                    j["embedding"] = d["embedding"]
                if "emb_idx" in d:
                    j["emb_idx"] = d["emb_idx"]
                obs_json.append(j)

            tracks_out.append({
                "first_frame": int(f0),
                "last_frame": int(f1),
                "face_label": _track_label(t),
                "avg_center_x": round(float(cx), 2),
                "avg_center_y": round(float(cy), 2),
                "avg_face_width": round(float(w), 2),
                "avg_face_height": round(float(h), 2),
                "is_static": bool(is_static),
                "obs": obs_json,
            })
        if not tracks_out:
            continue
        shot_first = min(ft["first_frame"] for ft in tracks_out)
        shot_last = max(ft["last_frame"] for ft in tracks_out)
        shots_out.append({
            "shot_number": int(shot_number),
            "first_frame": int(shot_first),
            "last_frame": int(shot_last),
            "face_tracks": tracks_out,
        })

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V2_0,
        "video": video,
        "shots": shots_out,
    }
    if face_metadata is not None:
        manifest["face_metadata"] = face_metadata

    # generation: prefer explicit overrides; otherwise derive from live objects
    if generation:
        base_gen = build_generation({"emb_store": _emb_store_token(cfg.emb_store)})
        base_gen = dict(base_gen)
        base_gen.update(generation)  # caller’s fields win (commit/branch etc.)
    else:
        base_gen = build_generation_from_objects(
            detector=detector,
            embedder=embedder,
            tracking_params=tracking_params,
            validator=validator,
        )
        base_gen["emb_store"] = _emb_store_token(cfg.emb_store)

    # finalize params hash on the merged/stable view
    stable = {k: v for k, v in base_gen.items() if k != "created_utc"}
    base_gen["params_hash"] = _compute_params_hash(stable)
    manifest["generation"] = base_gen

    return manifest

def fix_shots(manifest: Dict[str, Any], shot_defs: List[Dict[str, Any]]) -> None:
    """
    Ensure manifest['shots']:
      - contains one entry per shot in `shot_defs`
      - each shot's (first_frame, last_frame) matches the shot segmentation,
        regardless of whether the shot has tracks.

    Mutates `manifest` in-place, including `totals.num_shots` and `totals.num_tracks` if present.
    """
    if not shot_defs:
        return

    shots = manifest.get("shots") or []

    # Index existing shots by shot_number
    existing_by_num: Dict[int, Dict[str, Any]] = {}
    for s in shots:
        if "shot_number" not in s:
            continue
        sn = int(s["shot_number"])
        existing_by_num[sn] = s

    # For every known shot definition:
    #   - create a new shot entry if missing
    #   - normalize first_frame/last_frame to the segmentation
    for sd in shot_defs:
        sn = int(sd["shot_number"])
        first = int(sd.get("first_frame", 0))
        last = int(sd.get("last_frame", max(first, first)))

        entry = existing_by_num.get(sn)
        if entry is None:
            # No tracks for this shot -> trackless/graphics-only shot
            entry = {
                "shot_number": sn,
                "first_frame": first,
                "last_frame": last,
                "num_tracks": 0,
                "face_tracks": [],
            }
            shots.append(entry)
            existing_by_num[sn] = entry
        else:
            # Shot already exists (has tracks); normalize coverage to full shot span
            entry["first_frame"] = first
            entry["last_frame"] = last

    # Keep shots sorted by shot_number for stable output
    shots.sort(key=lambda s: int(s["shot_number"]))
    manifest["shots"] = shots

    # Fix totals, if present
    totals = manifest.get("totals")
    if isinstance(totals, dict):
        totals["num_shots"] = len(shots)
        totals["num_tracks"] = sum(int(s.get("num_tracks", 0)) for s in shots)
        manifest["totals"] = totals

def normalize_obs_items_for_output(
    items: list[FaceObservation],
    *,
    shot_id: int | None = None,
    track_id: int | None = None,
    emb_collector: EmbeddingCollector | None = None,
) -> list[dict]:
    out: list[dict] = []
    for ob in items:
        if not isinstance(ob, FaceObservation):
            raise TypeError(f"normalize_obs_items_for_output expects FaceObservation, got {type(ob).__name__}")

        if not isinstance(ob.source, Source):
            raise TypeError(
                f"Observation.source must be Source enum, got {type(ob.source).__name__}"
            )

        x1, y1, x2, y2 = map(int, ob.bbox) if ob.bbox is not None else (0, 0, 0, 0)
        row = {
            "f": int(ob.frame_idx),
            "bbox_xyxy": [x1, y1, x2, y2],
            # JSON schema expects a string ("detected"/"tracked"/"flow"); disk sidecars use int codes.
            "src": str(ob.source),
        }

        if ob.confidence is not None:
            row["conf"] = float(ob.confidence)

        # If caller supplied track/shot, include them (some exporters like having it)
        if track_id is not None:
            row["track_id"] = int(track_id)
        if shot_id is not None:
            row["shot"] = int(shot_id)

        # Attach embedding (inline or sidecar index) if requested
        if emb_collector is not None:
            # You’ll pass the actual vector in your caller, but if it’s already on the FaceObservation
            # (e.g., ob.embedding), wire it through; otherwise leave empty.
            vec = getattr(ob, "embedding", None)
            fields = emb_collector.fields_for_json(vec)
            if fields:
                row.update(fields)

        out.append(row)
    return out

def write_v2_json(path: str, manifest: Dict[str, Any]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))
    return str(p)

def build_v2_1_manifest_from_tracks(
    tracks: list,
    cfg: V2WriterConfig,
    *,
    face_metadata: Optional[list[dict]] = None,
    generation: Optional[dict] = None,
    detector: Optional[object] = None,
    embedder: Optional[object] = None,
    tracking_params: Optional[Mapping[str, Any]] = None,
    validator: Optional[object] = None,
    emb_collector: Optional[EmbeddingCollector] = None,
    obs_collector: Optional[ObservationsCollector] = None,
) -> dict:
    """
    Schema 2.1 writer: per-frame observations go to an NPZ sidecar via
    ObservationsCollector. Tracks get (obs_offset, obs_count). Embeddings:
      - recommended 'sidecar' via EmbeddingCollector
      - 'none' supported
      - 'inline' is NOT supported in 2.1 (no per-frame JSON). We coerce to sidecar.
    NOTE: This function has no side effects (does not write files).
    """

    # Schema 2.1 contract: per-track rows live in an observations sidecar, so
    # obs_offset/obs_count must be computed from an ObservationsCollector.
    #
    # If obs_collector is missing, emitting (0,0) silently produces invalid manifests
    # (all slices point at nothing) and breaks downstream consumers.
    if obs_collector is None:
        # Allow truly-empty track lists (e.g. graphics-only passes) without requiring a collector.
        if tracks:
            raise ValueError(
                "Schema 2.1 writer requires obs_collector to compute obs_offset/obs_count "
                "(pass ObservationsCollector, or use schema 2.0 / a different output mode)."
            )

    # Coerce embeddings inline->sidecar in 2.1 to keep model simple
    if emb_collector and getattr(emb_collector, "mode", None) == "inline":
        emb_collector.mode = "sidecar"

    W, H = (cfg.video_size or (0, 0))

    # Pack tracks by shot
    shots_map: dict[int, list] = {}
    for t in tracks:
        shot_id = int(getattr(t, "shot_id", 1))
        shots_map.setdefault(shot_id, []).append(t)

    # Minimal video block
    video = {}
    if cfg.video_path: video["path"] = str(cfg.video_path)
    if cfg.fps is not None: video["fps"] = float(cfg.fps)
    if cfg.video_size is not None: video["size"] = [int(cfg.video_size[0]), int(cfg.video_size[1])]
    if cfg.total_frames is not None: video["total_frames"] = int(cfg.total_frames)

    shots: list[dict] = []

    for shot_number in sorted(shots_map.keys()):
        tracks: list[dict] = []
        for t in shots_map[shot_number]:
            f0, f1 = _track_first_last(t)

            obs = getattr(t, "observations", [])
            # average bbox stats (same as 2.0)
            if obs:
                xs1 = sum(o.bbox[0] for o in obs) / len(obs)
                ys1 = sum(o.bbox[1] for o in obs) / len(obs)
                xs2 = sum(o.bbox[2] for o in obs) / len(obs)
                ys2 = sum(o.bbox[3] for o in obs) / len(obs)
                avg_bbox = (xs1, ys1, xs2, ys2)
            else:
                avg_bbox = (0.0, 0.0, 0.0, 0.0)

            # Track drift can push avg bbox outside frame; clip summary bbox.
            avg_bbox = _clip_bbox_xyxy(avg_bbox, W, H) if (W and H) else avg_bbox
            cx, cy, w, h = _bbox_to_center_size_xyxy(avg_bbox)
            cx, cy, w, h = _normalize(cx, cy, w, h, W, H, cfg.normalize_to_percent)

            # movement heuristic (as in 2.0)
            centers = _track_center_series(t)
            if centers and W and H and np is not None:
                cx_series = np.array([c[0] * (100.0/W if cfg.normalize_to_percent else 1.0/W) for c in centers], dtype=float)
                cy_series = np.array([c[1] * (100.0/H if cfg.normalize_to_percent else 1.0/H) for c in centers], dtype=float)
                std_c = float((np.var(cx_series) + np.var(cy_series)) ** 0.5)
            else:
                std_c = 0.0
            is_static = std_c < cfg.static_stddev_thresh_pct

            # Offload per-frame obs to sidecar and store slice
            shot_val = int(getattr(t, "shot_id", shot_number))
            track_val = int(getattr(t, "track_id", -1))
            obs_offset, obs_count = obs_collector.slice_for_track(shot_val, track_val)

            tracks.append({
                "first_frame": int(f0),
                "last_frame": int(f1),
                "face_label": _track_label(t),
                "avg_center_x": round(float(cx), 2),
                "avg_center_y": round(float(cy), 2),
                "avg_face_width": round(float(w), 2),
                "avg_face_height": round(float(h), 2),
                "is_static": bool(is_static),
                "obs_offset": int(obs_offset),
                "obs_count": int(obs_count),
            })

        if not tracks:
            continue
        shot_first = min(ft["first_frame"] for ft in tracks)
        shot_last  = max(ft["last_frame"] for ft in tracks)
        shots.append({
            "shot_number": int(shot_number),
            "first_frame": int(shot_first),
            "last_frame": int(shot_last),
            "num_tracks": int(len(tracks)),
            "face_tracks": tracks,
        })

    manifest: dict = {
        "schema_version": SCHEMA_VERSION_V2_1,
        "video": video,
        "shots": shots,
    }
    # Face metadata: prefer caller; otherwise derive from the observations (canonical)
    if face_metadata is not None:
        manifest["face_metadata"] = face_metadata
    elif obs_collector is not None:
        manifest["face_metadata"] = _derive_face_metadata_from_tracks(shots)

    if generation:
        base_gen = build_generation({"emb_store": ("sidecar" if emb_collector else "none")})
        base_gen.update(generation)
    else:
        base_gen = build_generation_from_objects(
            detector=detector,
            embedder=embedder,
            tracking_params=tracking_params,
            validator=validator,
        )
        base_gen["emb_store"] = "sidecar" if emb_collector else "none"

    stable = {k: v for k, v in base_gen.items() if k != "created_utc"}
    base_gen["params_hash"] = _compute_params_hash(stable)
    manifest["generation"] = base_gen

    return manifest
