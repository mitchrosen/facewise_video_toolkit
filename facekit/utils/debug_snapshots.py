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
