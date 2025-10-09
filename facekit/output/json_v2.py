from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Mapping, Literal
from pathlib import Path
import json, os, subprocess, hashlib
from datetime import datetime, timezone

try:
    import numpy as np
except Exception:
    np = None

XYXY = Tuple[float, float, float, float]

@dataclass
class V2WriterConfig:
    schema_version: str = "2.0"
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

def derive_face_metadata(tracks: List[Any]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for t in tracks:
        label = _track_label(t)
        n = len(getattr(t, "observations", []))
        counts[label] = counts.get(label, 0) + int(n)
    return [{"face_label": k, "occurance_count": v} for k, v in sorted(counts.items())]

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
        head = Path(repo_dir) / ".git" / "HEAD"
        if head.exists():
            ref_line = head.read_text().strip()
            if ref_line.startswith("ref:"):
                rel = ref_line.split(" ", 1)[1]  # e.g. refs/heads/main
                ref_path = Path(repo_dir) / ".git" / rel
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

class EmbeddingCollector:
    """
    Handles inline vs sidecar storage of emeddings.
    - inline: returns the list[float] to embed directly in JSON and no indexing.
    - sidecar: assigns a stable integer index for each vector and stores it in memory.
    - None: no-op
    """
    def __init__(self, mode: Literal["inline","sidecar"] | None, dim: int | None = None):
        self.mode = mode
        self.dim = dim
        self._embs: List[np.ndarray] = []

    def assign(self, vec: np.ndarray | None) -> Dict[str, Any]:
        """Return the JSON field(s) to add to an obs given an embedding vec or None."""
        if self.mode is None or vec is None:
            return {}
        v = np.asarray(vec, dtype=np.float32).ravel()
        if self.dim is not None and v.size != self.dim:
            # Be strict; avoid silent format drift
            raise ValueError(f"Embedding dim mismatch; expected {self.dim}, got {v.size}")
        if self.mode == "inline":
            return {"embedding": v.tolist()}
        # sidecar
        idx = len(self._embs)
        self._embs.append(v.copy())
        return {"emb_idx": idx}

    def finalize_sidecar(self, out_path: Path) -> Dict[str, Any]:
        """
        No-op unless in 'sidecar' mode.
        Writes the sidecar array and returns a descriptor for the manifest.
        """
        if self.mode != "sidecar":
            return {}  # nothing to write
        
        out_path = Path(out_path)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self._embs:
            arr = np.vstack(self._embs).astype(np.float32, copy=False) 
        else: 
            arr = np.zeros((0, self.dim or 0), dtype=np.float32)

        # default to .npy if suffix looks like .npy; otherwise enforce .npz
        fmt = "npy" if out_path.suffix.lower()==".npy" else "npz"
        if fmt == "npz":
            out_path = out_path.with_suffix(".npz")
            np.savez(out_path, embeddings=arr)
        else:
            np.save(out_path, arr)

        desc = {
            "path": str(out_path),
            "format": fmt,
            "dtype": "float32",
            "dim": int(arr.shape[1]) if arr.ndim == 2 else int(self.dim or 0),
            "count": int(arr.shape[0]),
        }
        return desc

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
    returns (manifest, emb_collector).
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
            cx, cy, w, h = _bbox_to_center_size_xyxy(avg_bbox)
            cx, cy, w, h = _normalize(cx, cy, w, h, W, H, cfg.normalize_to_percent)

            centers = _track_center_series(t)
            if centers and W and H and np is not None:
                import numpy as _np
                cx_series = _np.array([c[0] * (100.0/W if cfg.normalize_to_percent else 1.0/W) for c in centers], dtype=float)
                cy_series = _np.array([c[1] * (100.0/H if cfg.normalize_to_percent else 1.0/H) for c in centers], dtype=float)
                std_c = float((_np.var(cx_series) + _np.var(cy_series)) ** 0.5)
            else:
                std_c = 0.0
            is_static = std_c < cfg.static_stddev_thresh_pct

            obs_json: List[Dict[str, Any]] = []
            for o in obs:
                src = getattr(o, "source", None)
                if src == "detection": src = "detected"
                elif src == "tracking": src = "tracked"
                item = {
                    "f": int(o.frame_idx),
                    "bbox_xyxy": [float(o.bbox[0]), float(o.bbox[1]), float(o.bbox[2]), float(o.bbox[3])],
                    "src": src or "detected",
                }
                if getattr(o, "confidence", None) is not None:
                    item["conf"] = float(o.confidence)
                # Attach embedding inline or as index, if present
                if embc is not None:
                    emb = getattr(o, "embedding", None)
                    item.update(embc.assign(emb))
                obs_json.append(item)

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
        "schema_version": cfg.schema_version,
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

    # # finalize sidecar if requested
    # if cfg.emb_store == "sidecar":
    #     # semantics:
    #     # - emb_sidecar_path is None → use default next to video (*.embeddings.npz)
    #     # - emb_sidecar_path is a Path → write there (validate parent)
    #     if cfg.emb_sidecar_path is not None:
    #         sidecar_path = cfg.emb_sidecar_path
    #     else:
    #         sidecar_path = Path(cfg.video_path).with_suffix(".embeddings.npz")
    #     manifest["embedding_sidecar"] = embc.finalize_sidecar(Path(sidecar_path))
    # return manifest

# def materialize_sidecar_if_needed(
#     embc: Optional[EmbeddingCollector],
#     cfg: V2WriterConfig,
# ) -> Optional[Dict[str, Any]]:
#     """
#     Writes the sidecar only if cfg.emb_store == 'sidecar' and embc is provided.
#     Returns the descriptor to be inserted into the manifest, or None if no-op.
#     """
#     if cfg.emb_store != "sidecar" or embc is None:
#         return None

#     # resolve path semantics:
#     # - emb_sidecar_path None → default next to video (*.embeddings.npz)
#     # - emb_sidecar_path Path  → write there (parent mkdir)
#     # (we always write; no “I already wrote elsewhere” ambiguity)
#     if cfg.emb_sidecar_path is not None:
#         sidecar_path = cfg.emb_sidecar_path
#     else:
#         sidecar_path = Path(cfg.video_path).with_suffix(".embeddings.npz")

#     desc = embc.finalize_sidecar(Path(sidecar_path))
#     return desc

def write_v2_json(path: str, manifest: Dict[str, Any]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))
    return str(p)
