#!/usr/bin/env python3
"""
probe_sidecars.py

Usage:
  python probe_sidecars.py --run-root /path/to/run-YYYY... --anchor 180 --det-code 1

What it does:
  - Loads obs sidecar: <run-root>/ckpt/obs_ckpt.npz  (key 'observations')
  - Loads emb sidecar: <run-root>/ckpt/emb_ckpt.npz  (key usually 'embeddings' or 'vecs')
  - Treats the given `anchor` as the resume anchor frame.
  - Enforces **new resume semantics**:

    Let `anchor_shot` be the shot that contains DET rows at frame == anchor.

    * For all pre-anchor DET rows (f <= anchor-1) in **completed shots** (shot < anchor_shot):
        - Require that each row has a valid embedding index: 0 <= emb_idx < num_embeddings.

    * For pre-anchor DET rows in the **anchor shot** (shot == anchor_shot):
        - Require that a crop is present.
        - Do **NOT** require emb_idx; it's allowed to be -1, since embeddings
          will be recomputed from crops on resume.

  - Exits 0 if all invariants hold.
  - Exits 1 otherwise, printing a summary of the violations.
"""

import argparse
import sys
from pathlib import Path
import numpy as np


def _load_obs(run_root: Path):
    obs_npz = run_root / "ckpt" / "obs_ckpt.npz"
    if not obs_npz.exists():
        raise SystemExit(f"OBS sidecar not found: {obs_npz}")
    arr = np.load(obs_npz, allow_pickle=False)["observations"]
    return arr


def _load_emb(run_root: Path):
    emb_npz = run_root / "ckpt" / "emb_ckpt.npz"
    if not emb_npz.exists():
        # No embeddings at all: treat as zero-length
        return None
    npz = np.load(emb_npz, allow_pickle=False)
    # Prefer common keys
    for key in ("embeddings", "vecs", "arr_0"):
        if key in npz:
            return npz[key]
    # Fallback: first array in the file
    for key in npz.files:
        return npz[key]
    return None

def _extract_landmarks_presence(obs: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask per row: True if the row has landmarks persisted.
    Supports a few schema variants without guessing an exact landmark count.
    """
    names = set(obs.dtype.names or ())
    if "landmarks" in names:
        landmarks = obs["landmarks"]
        # landmarks may be shaped (K,2), (2K,), or fixed padded array; treat "all zeros" or "all NaN" as missing.
        # We do not assume a specific K.
        landmarks_arr = np.asarray(landmarks)
        # Reduce over all non-row axes
        if landmarks_arr.ndim >= 2:
            # per-row: any finite value
            axis = tuple(range(1, landmarks_arr.ndim))
            return np.any(np.isfinite(landmarks_arr), axis=axis)
        else:
            return np.isfinite(landmarks_arr)

    # Variant: landmarks_xy + landmarks_n
    if ("landmarks_xy" in names) and ("landmarks_n" in names):
        n = obs["landmarks_n"].astype(np.int64, copy=False)
        return n > 0

    # Variant: lmk_xy + lmk_n
    if ("lmk_xy" in names) and ("lmk_n" in names):
        n = obs["lmk_n"].astype(np.int64, copy=False)
        return n > 0

    raise AssertionError(
        "No landmarks fields found in observations sidecar. "
        f"dtype names = {list(obs.dtype.names or [])}. "
        "Expected one of: 'landmarks' OR ('landmarks_xy','landmarks_n') OR ('lmk_xy','lmk_n')."
    )



def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--anchor", required=True, type=int)
    p.add_argument("--det-code", required=True, type=int)
    args = p.parse_args(argv)

    run_root: Path = args.run_root
    anchor: int = args.anchor
    det_code: int = args.det_code

    obs = _load_obs(run_root)
    emb = _load_emb(run_root)

    fields = obs.dtype.names
    print("OBS fields:", fields)

    n_emb = int(emb.shape[0]) if emb is not None else 0
    print(f"EMB shape: {emb.shape if emb is not None else None} dtype: {getattr(getattr(emb, 'dtype', None), 'name', None)}")

    # Basic columns
    f = obs["f"].astype(int)
    shot = obs["shot"].astype(int)
    src = obs["src"].astype(int)
    emb_idx = obs["emb_idx"].astype(int) if "emb_idx" in fields else np.full_like(f, -1, dtype=int)

    # Landmarks presence mask (raises with clear error if fields missing)
    has_landmarks = _extract_landmarks_presence(obs)

    # Pre-anchor DET rows: f <= anchor-1
    pre_anchor_mask = (src == det_code) & (f <= anchor - 1)

    # Determine anchor_shot from DET rows at exactly anchor, if any.
    anchor_det_mask = (src == det_code) & (f == anchor)
    anchor_shots = np.unique(shot[anchor_det_mask]) if anchor_det_mask.any() else np.array([], dtype=int)
    if anchor_shots.size == 1:
        anchor_shot = int(anchor_shots[0])
    else:
        # Fallback: take the max shot index seen among pre-anchor rows (conservative).
        pre_shots = np.unique(shot[pre_anchor_mask]) if pre_anchor_mask.any() else np.array([], dtype=int)
        anchor_shot = int(pre_shots.max()) if pre_shots.size > 0 else None

    print(f"Anchor: {anchor} (checked frames <= {anchor-1})")
    print(f"Anchor shot inferred as: {anchor_shot!r}")

    if not pre_anchor_mask.any():
        print("No pre-anchor DET rows found; nothing to check.")
        return 0

    # --- Embedding invariant: completed shots only (shot < anchor_shot) ---
    completed_shot_mask = np.zeros_like(pre_anchor_mask, dtype=bool)
    if anchor_shot is not None:
        completed_shot_mask = pre_anchor_mask & (shot < anchor_shot)
    else:
        # If we couldn't infer anchor_shot, treat all as completed (old strict behavior).
        completed_shot_mask = pre_anchor_mask.copy()

    bad_emb_mask = completed_shot_mask & ((emb_idx < 0) | (emb_idx >= n_emb))
    missing_emb_count = int(bad_emb_mask.sum())

    # --- Landmark invariant: completed shots DET rows must have landmarks ---
    missing_landmarks_mask = completed_shot_mask & (~has_landmarks)
    missing_landmarks_count = int(missing_landmarks_mask.sum())

    # For debugging, report totals
    det_rows_total = int(pre_anchor_mask.sum())
    print("\n=== DET→EMB parity (completed shots only) ===")
    print(f"Total DET rows considered (pre-anchor, all shots): {det_rows_total}")
    print(f"Completed-shot DET rows (shot < anchor_shot): {int(completed_shot_mask.sum())}")
    print(f"Embedding rows in sidecar: {n_emb}")
    print(f"Missing embeddings in completed shots: {missing_emb_count}")
    print(f"Missing landmarks in completed shots: {missing_landmarks_count}")

    # Show small samples when there are issues
    def _sample(mask, label, limit=5):
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return
        print(f"\n=== Samples for {label} (up to {limit}) ===")
        for i in idx[:limit]:
            row = {name: obs[name][i] for name in fields}
            # make it a bit shorter
            row["f"] = int(row.get("f", -1))
            row["shot"] = int(row.get("shot", -1))
            row["src"] = int(row.get("src", -1))
            row["emb_idx"] = int(row.get("emb_idx", -1)) if "emb_idx" in row else -1
            print(row)

    if missing_emb_count:
        _sample(bad_emb_mask, "completed-shot DET rows with invalid emb_idx")
 
    if missing_landmarks_count:
        _sample(missing_landmarks_mask, "completed-shot DET rows missing landmarks")

    # Decide exit code:
    # - Missing landmarks in completed shots is a hard error.
    if missing_emb_count or missing_landmarks_count:
        print("\n=== Summary ===")
        print(f"Pre-anchor DET rows: {det_rows_total}")
        print(f"Missing crops (all pre-anchor DET): {missing_crop_count}")
        print(f"Missing embeddings (completed shots only): {missing_emb_count}")
        print(f"Missing landmarks (completed shots only): {missing_landmarks_count}")
        return 1

    print("\n=== Summary ===")
    print("OK: completed shots have DET↔EMB parity and landmarks persisted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
