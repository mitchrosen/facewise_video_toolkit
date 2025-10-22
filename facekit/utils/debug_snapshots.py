# facekit/utils/debug_snapshots.py
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import typing as _t
import numpy as np

from facekit.common.obs_consts import Source

SchemaVersion = "1.0"

@dataclass
class TrackBrief:
    shot_id: int
    track_id: int
    segment_id: _t.Optional[int]
    open: bool
    first_frame: _t.Optional[int]
    last_frame: _t.Optional[int]
    len_obs: int
    len_det: int
    len_trk: int
    has_embedding: bool
    avg_emb_norm: _t.Optional[float]
    last_det_bbox: _t.Optional[tuple[int,int,int,int]]
    last_det_frame: int

def _last_det_bbox_and_frame(tr) -> tuple[_t.Optional[tuple[int,int,int,int]], int]:
    last_det_frame = -1
    last_det_bbox = None
    for o in getattr(tr, "observations", []) or []:
        if getattr(o, "source", None) == Source.DETECTED and o.bbox is not None:
            if int(o.frame_idx) > last_det_frame:
                last_det_frame = int(o.frame_idx)
                x1,y1,x2,y2 = [int(v) for v in o.bbox[:4]]
                last_det_bbox = (x1,y1,x2,y2)
    return last_det_bbox, last_det_frame

def _avg_emb_norm(tr) -> _t.Optional[float]:
    if hasattr(tr, "has_embedding") and tr.has_embedding():
        try:
            v = tr.compute_average_embedding()
            if v is None:
                return None
            v = np.asarray(v, dtype=np.float32)
            return float(np.linalg.norm(v))
        except Exception:
            return None
    return None

def brief_track(tr) -> TrackBrief:
    dets = [o for o in (tr.observations or []) if getattr(o, "source", None) == Source.DETECTED]
    trks = [o for o in (tr.observations or []) if getattr(o, "source", None) == Source.TRACKED]
    last_det_bbox, last_det_frame = _last_det_bbox_and_frame(tr)
    seg = getattr(tr, "segment_id", None)
    if seg is not None:
        try: seg = int(seg)
        except Exception: seg = None
    return TrackBrief(
        shot_id      = int(getattr(tr, "shot_id", -1)),
        track_id     = int(getattr(tr, "track_id", -1)),
        segment_id   = seg,
        open         = (not tr.is_closed()),
        first_frame  = (tr.first_frame() if hasattr(tr, "first_frame") else None),
        last_frame   = (tr.last_frame()  if hasattr(tr, "last_frame")  else None),
        len_obs      = len(getattr(tr, "observations", []) or []),
        len_det      = len(dets),
        len_trk      = len(trks),
        has_embedding= bool(getattr(tr, "has_embedding", lambda: False)()),
        avg_emb_norm = _avg_emb_norm(tr),
        last_det_bbox= last_det_bbox,
        last_det_frame= last_det_frame,
    )

def snapshot_from_aggregator(
    aggregator,
    *,
    frame_idx: int,
    shot_number: int,
    shot_first_frame: int,
    track_order: dict[tuple[int,int], int] | None,
    run_id: str | None = None,
) -> dict:
    """
    Build a pre-GID snapshot of the current world at a checkpoint boundary.
    """
    briefs = []
    for tr in getattr(aggregator, "tracks", []) or []:
        briefs.append(asdict(brief_track(tr)))

    # Persist a tiny “open tracks” fast view for resume warm-start debugging
    open_fast = []
    for tr in getattr(aggregator, "tracks", []) or []:
        if not tr.is_closed():
            last_bbox, last_det = _last_det_bbox_and_frame(tr)
            open_fast.append({
                "track_id": int(getattr(tr, "track_id", -1)),
                "last_frame": tr.last_frame() if hasattr(tr, "last_frame") else None,
                "last_det_frame": last_det,
                "last_det_bbox": last_bbox,
            })

    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema": SchemaVersion,
        "ts": now,
        "run_id": run_id,
        "frame_idx": int(frame_idx),
        "shot_number": int(shot_number),
        "shot_first_frame": int(shot_first_frame),
        "aggregator": {
            "next_track_id": int(getattr(aggregator, "next_track_id", -1)),
            "track_count": len(getattr(aggregator, "tracks", []) or []),
            "open_track_count": len([t for t in getattr(aggregator, "tracks", []) or [] if not t.is_closed()]),
            "open_fast": open_fast,
        },
        "track_order": {f"{k[0]}:{k[1]}": int(v) for k, v in (track_order or {}).items()},
        "tracks": briefs,
    }

def write_snapshot_atomic(base_dir: Path, name: str, payload: dict) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp = base_dir / (name + ".tmp")
    out = base_dir / (name + ".json")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, out)
    return out

def load_latest_snapshot(snapshots_dir: Path, *, up_to_frame: int | None = None) -> dict | None:
    snapshots_dir = Path(snapshots_dir)
    if not snapshots_dir.exists():
        return None
    cands = sorted(snapshots_dir.glob("frame_*_shot_*.json"))
    if not cands:
        return None
    if up_to_frame is None:
        return json.loads(cands[-1].read_text(encoding="utf-8"))
    # choose the latest <= up_to_frame
    best = None
    best_frame = -1
    for p in cands:
        try:
            # name format: frame_{f}_shot_{s}.json
            parts = p.stem.split("_")
            f = int(parts[1])
        except Exception:
            continue
        if f <= up_to_frame and f > best_frame:
            best = p; best_frame = f
    return json.loads(best.read_text(encoding="utf-8")) if best else None

def diff_snapshot_vs_rehydrate(snapshot: dict, prior_tracks: list) -> list[str]:
    """
    Return human-readable diff lines comparing snapshot (pre-interruption) vs. rehydration.
    Focuses on per-(shot,track) last frames, open/closed, and last DET frame.
    """
    # Build maps
    def _key(tr) -> tuple[int,int]:
        return (int(getattr(tr, "shot_id", -1)), int(getattr(tr, "track_id", -1)))

    S = {}
    for t in snapshot.get("tracks", []):
        S[(int(t["shot_id"]), int(t["track_id"]))] = {
            "open": bool(t["open"]),
            "last_frame": (t["last_frame"] if t["last_frame"] is not None else -1),
            "last_det_frame": int(t["last_det_frame"]),
            "segment_id": (t["segment_id"] if t["segment_id"] is not None else None),
        }

    R = {}
    for tr in prior_tracks or []:
        last_det_bbox, last_det_frame = _last_det_bbox_and_frame(tr)
        R[_key(tr)] = {
            "open": (not tr.is_closed()),
            "last_frame": (tr.last_frame() if hasattr(tr,"last_frame") else -1),
            "last_det_frame": last_det_frame,
            "segment_id": getattr(tr, "segment_id", None),
        }

    lines = []
    keys = sorted(set(S.keys()) | set(R.keys()))
    for k in keys:
        s = S.get(k); r = R.get(k)
        if s is None:
            lines.append(f"[resume-diff] NEW on rehydrate: key={k} state={r}")
            continue
        if r is None:
            lines.append(f"[resume-diff] MISSING on rehydrate: key={k} prior_state={s}")
            continue
        # compare fields
        for fld in ("open","last_frame","last_det_frame","segment_id"):
            if s.get(fld) != r.get(fld):
                lines.append(
                    f"[resume-diff] key={k} {fld}: prior={s.get(fld)} now={r.get(fld)}"
                )
    if not lines:
        lines.append("[resume-diff] exact parity between snapshot and rehydrate (tracked fields)")
    return lines

def materialize_structured_embeddings(
    obs: np.ndarray,
    emb: np.ndarray,
    *,
    only_with_valid_idx: bool = True,
) -> np.ndarray:
    """
    Build a structured view that joins obs rows to the embedding matrix.

    - obs: structured array with fields including at least
      'shot', 'f', 'track_id', 'src', 'emb_idx', and optionally 'has_crop'.
    - emb: 2-D float array of shape (N, D), dtype float32/float64, no field names.

    Returns a structured array with fields:
      ('shot', 'frame', 'track_id', 'src', 'emb_idx', 'has_crop', 'vec')

    If only_with_valid_idx=True, rows whose emb_idx < 0 or >= emb.shape[0]
    are dropped. Otherwise they are included with 'vec' left as zeros.
    """
    if obs.dtype.names is None:
        raise ValueError("obs must be a structured array")

    if emb.ndim != 2:
        raise ValueError(f"emb must be 2-D, got shape {emb.shape!r}")

    n_obs = obs.shape[0]
    dim = emb.shape[1] if emb.shape[0] > 0 else 0

    # Determine which obs rows have valid embedding indices
    if "emb_idx" not in obs.dtype.names:
        raise ValueError("obs has no 'emb_idx' field")

    emb_idx = obs["emb_idx"].astype(int)
    valid_mask = (emb_idx >= 0) & (emb_idx < emb.shape[0])

    if only_with_valid_idx:
        obs_used = obs[valid_mask]
        emb_idx_used = emb_idx[valid_mask]
    else:
        obs_used = obs
        emb_idx_used = np.clip(emb_idx, -1, max(emb.shape[0] - 1, 0))

    n = obs_used.shape[0]

    dtype = [
        ("shot", np.int32),
        ("frame", np.int32),
        ("track_id", np.int32),
        ("src", np.int32),
        ("emb_idx", np.int32),
        ("has_crop", np.int8),
        ("vec", np.float32, (dim,)),
    ]
    out = np.zeros(n, dtype=dtype)

    # Copy scalar fields
    out["shot"] = obs_used["shot"].astype(np.int32)
    out["frame"] = obs_used["f"].astype(np.int32)
    out["track_id"] = obs_used["track_id"].astype(np.int32)
    out["src"] = obs_used["src"].astype(np.int32)
    out["emb_idx"] = emb_idx_used.astype(np.int32)

    if "has_crop" in obs.dtype.names:
        out["has_crop"] = obs_used["has_crop"].astype(np.int8)
    else:
        out["has_crop"] = -1  # or 0, if you prefer

    # Fill vectors for valid indices
    valid_for_vec = (emb_idx_used >= 0) & (emb_idx_used < emb.shape[0])
    if emb.shape[0] > 0 and valid_for_vec.any():
        out["vec"][valid_for_vec] = emb[emb_idx_used[valid_for_vec]]

    return out