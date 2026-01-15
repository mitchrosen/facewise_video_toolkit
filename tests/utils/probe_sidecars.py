#!/usr/bin/env python3
"""
probe_sidecars.py

Usage:
  python probe_sidecars.py --run-root /path/to/run-YYYY... --anchor 180 --det-code 1

What it does:
  - Loads obs sidecar: <run-root>/ckpt/obs_ckpt.npz  (key 'observations')
  - Loads emb sidecar: <run-root>/ckpt/emb_ckpt.npz  (key usually 'embeddings' or 'vecs')
  - Treats the given `anchor` as the resume anchor frame.
  - Enforces **current resume semantics** (no crops):

    Let `anchor_shot` be the shot that contains DET rows at frame == anchor.

    * For all pre-anchor DET rows (f <= anchor-1) in **completed shots** (shot < anchor_shot):
        - Require that each row has a valid embedding index:
              0 <= emb_idx < num_embeddings
        - Require that each row has valid landmarks:
              * landmarks present
              * numeric
              * finite (no NaN/Inf)
              * shape either (10,) flat or (5,2)

    * For pre-anchor DET rows in the **anchor shot** (shot == anchor_shot):
        - Require valid landmarks (same rules as above).
        - Do **NOT** require emb_idx; it's allowed to be -1 because embeddings may be
          computed/batched at end-of-shot and the anchor shot may be incomplete at crash time.

  - Exits 0 if all invariants hold.
  - Exits 1 otherwise, printing a summary of the violations.

Notes:
  - This script intentionally does not check anything about crops. Crops are not persisted
    under the current contract; landmarks are the persisted alignment primitive.
"""

import argparse
import sys
from pathlib import Path
import numpy as np


# ---- Loading helpers ---------------------------------------------------------

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


# ---- Landmarks helpers -------------------------------------------------------

def _select_landmarks_field(fields: tuple[str, ...]) -> str | None:
    """
    Return the preferred landmarks field name present in `fields`, else None.

    We explicitly avoid the abbreviation 'lm'. If legacy schemas used it, we can
    choose to support it, but we won't print or refer to it as 'lm'.
    """
    # Current canonical (based on your printed dtype): 'landmarks_flat10'
    preferred = (
        "landmarks_flat10",   # flat [x1,y1,...,x5,y5]
        "landmarks",          # generic
        "landmarks_5pt",      # explicit
        "landmarks5",         # alternative legacy-ish
        "five_point_landmarks",
        # If you *must* support older data, add exact legacy field names here.
        # Avoid "lm" in new code; only include if you really have old files:
        # "lm",
        # "lms",
    )
    for name in preferred:
        if name in fields:
            return name
    return None


def _landmarks_missing_or_invalid_mask(
    obs: np.ndarray,
    det_mask: np.ndarray,
) -> tuple[np.ndarray, str | None]:
    """
    Return (bad_mask, landmarks_field_name).

    bad_mask is True for DET rows (per det_mask) that are missing landmarks OR have invalid landmarks.

    Valid landmarks:
      - present in sidecar under a supported field name
      - numeric array
      - finite (no NaN/Inf)
      - shape either:
          * (10,)   flat
          * (5, 2)  5 points x/y
    """
    fields = obs.dtype.names or ()
    landmarks_field = _select_landmarks_field(fields)
    if landmarks_field is None:
        # If we cannot find any landmarks field at all, all DET rows are invalid.
        return det_mask.copy(), None

    # If a has_landmarks flag exists, honor it. Otherwise infer "should have landmarks"
    # for DET rows (conservative: DET rows must have landmarks under the current contract).
    if "has_landmarks" in fields:
        should_have = det_mask & (obs["has_landmarks"].astype(int) == 1)
    else:
        should_have = det_mask.copy()

    # Pull landmarks array; expected numeric dtype (float32)
    landmarks = obs[landmarks_field]

    bad = np.zeros(det_mask.shape[0], dtype=bool)

    # Shape handling:
    # - If landmarks is (N, 10): each row flat10
    # - If landmarks is (N, 5, 2): each row 5x2
    # - If landmarks is object/structured, we'll attempt per-row conversion (slower but robust)
    try:
        arr = np.asarray(landmarks)
        # Fast-path numeric
        if arr.dtype != object:
            # Per-row validation depending on dimensionality
            if arr.ndim == 2 and arr.shape[1] == 10:
                # Check finiteness for required rows
                finite = np.all(np.isfinite(arr), axis=1)
                bad |= should_have & ~finite
            elif arr.ndim == 3 and arr.shape[1:] == (5, 2):
                finite = np.all(np.isfinite(arr.reshape(arr.shape[0], -1)), axis=1)
                bad |= should_have & ~finite
            else:
                # Unexpected shape: mark required rows bad
                bad |= should_have
        else:
            # Object array: validate row-by-row
            idx = np.nonzero(should_have)[0]
            for i in idx:
                try:
                    one = np.asarray(landmarks[i], dtype=np.float32)
                    if one.shape == (10,):
                        ok_shape = True
                        flat = one
                    elif one.shape == (5, 2):
                        ok_shape = True
                        flat = one.reshape(-1)
                    else:
                        ok_shape = False
                        flat = None
                    if (not ok_shape) or (flat is None) or (not np.all(np.isfinite(flat))):
                        bad[i] = True
                except Exception:
                    bad[i] = True
    except Exception:
        # If anything goes wrong at array level, mark required rows bad.
        bad |= should_have

    # Also treat "all zeros" as suspicious? (Optional)
    # Under your pipeline, zeros are unlikely; but if you want to enforce nonzero:
    # (keeping it OFF by default because it can false-positive on edge cases)
    #
    # if landmarks_field is not None:
    #     try:
    #         arr = np.asarray(obs[landmarks_field], dtype=np.float32)
    #         if arr.ndim == 2 and arr.shape[1] == 10:
    #             nonzero = np.any(arr != 0, axis=1)
    #             bad |= should_have & ~nonzero
    #     except Exception:
    #         pass

    return bad, landmarks_field


# ---- Main --------------------------------------------------------------------

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

    fields = obs.dtype.names or ()
    print("OBS fields:", fields)

    n_emb = int(emb.shape[0]) if emb is not None else 0
    print(f"EMB shape: {emb.shape if emb is not None else None} dtype: {getattr(getattr(emb, 'dtype', None), 'name', None)}")

    # Basic columns
    f = obs["f"].astype(int)
    shot = obs["shot"].astype(int)
    src = obs["src"].astype(int)
    emb_idx = obs["emb_idx"].astype(int) if "emb_idx" in fields else np.full_like(f, -1, dtype=int)

    # Pre-anchor DET rows: f <= anchor-1
    pre_anchor_det_mask = (src == det_code) & (f <= anchor - 1)

    # Determine anchor_shot from DET rows at exactly anchor, if any.
    anchor_det_mask = (src == det_code) & (f == anchor)
    anchor_shots = np.unique(shot[anchor_det_mask]) if anchor_det_mask.any() else np.array([], dtype=int)
    if anchor_shots.size == 1:
        anchor_shot = int(anchor_shots[0])
    else:
        # Fallback: take the max shot index seen among pre-anchor DET rows (conservative).
        pre_shots = np.unique(shot[pre_anchor_det_mask]) if pre_anchor_det_mask.any() else np.array([], dtype=int)
        anchor_shot = int(pre_shots.max()) if pre_shots.size > 0 else None

    print(f"Anchor: {anchor} (checked frames <= {anchor-1})")
    print(f"Anchor shot inferred as: {anchor_shot!r}")

    if not pre_anchor_det_mask.any():
        print("No pre-anchor DET rows found; nothing to check.")
        return 0

    # --- Landmarks invariant: all pre-anchor DET rows must have valid landmarks ---
    bad_landmarks_mask, landmarks_field_name = _landmarks_missing_or_invalid_mask(obs, pre_anchor_det_mask)
    bad_landmarks_count = int(bad_landmarks_mask.sum())

    # --- Embedding invariant: completed shots only (shot < anchor_shot) ---
    if anchor_shot is not None:
        completed_shot_mask = pre_anchor_det_mask & (shot < anchor_shot)
    else:
        # If we couldn't infer anchor_shot, treat all as completed (strict fallback).
        completed_shot_mask = pre_anchor_det_mask.copy()

    bad_emb_mask = completed_shot_mask & ((emb_idx < 0) | (emb_idx >= n_emb))
    missing_emb_count = int(bad_emb_mask.sum())

    # For debugging, report totals
    det_rows_total = int(pre_anchor_det_mask.sum())
    print("\n=== DET→LANDMARKS and DET→EMB checks (pre-anchor) ===")
    print(f"Total DET rows considered (pre-anchor, all shots): {det_rows_total}")
    print(f"Completed-shot DET rows (shot < anchor_shot): {int(completed_shot_mask.sum())}")
    print(f"Embedding rows in sidecar: {n_emb}")
    print(f"Landmarks field used: {landmarks_field_name!r}")
    print(f"Pre-anchor DET rows missing/invalid landmarks: {bad_landmarks_count}")
    print(f"Missing embeddings in completed shots: {missing_emb_count}")

    # Show small samples when there are issues
    def _sample(mask, label, limit=5):
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return
        print(f"\n=== Samples for {label} (up to {limit}) ===")
        for i in idx[:limit]:
            row = {name: obs[name][i] for name in fields}
            # make it a bit shorter / consistent
            row["f"] = int(row.get("f", -1))
            row["shot"] = int(row.get("shot", -1))
            row["src"] = int(row.get("src", -1))
            if "emb_idx" in row:
                row["emb_idx"] = int(row["emb_idx"])
            if "has_landmarks" in row:
                row["has_landmarks"] = int(row["has_landmarks"])
            # Optional: truncate very long landmarks prints
            if landmarks_field_name and landmarks_field_name in row:
                try:
                    lm_arr = np.asarray(row[landmarks_field_name], dtype=np.float32).reshape(-1)
                    if lm_arr.size > 10:
                        row[landmarks_field_name] = lm_arr[:10]
                except Exception:
                    pass
            print(row)

    if bad_landmarks_count:
        _sample(bad_landmarks_mask, "pre-anchor DET rows missing/invalid landmarks")

    if missing_emb_count:
        _sample(bad_emb_mask, "completed-shot DET rows with invalid emb_idx")

    # Decide exit code:
    #  - Any missing/invalid landmarks in pre-anchor DET rows is a hard error.
    #  - Any missing embedding in completed shots is a hard error.
    if bad_landmarks_count or missing_emb_count:
        print("\n=== Summary ===")
        print(f"Pre-anchor DET rows: {det_rows_total}")
        print(f"Missing/invalid landmarks (all pre-anchor DET): {bad_landmarks_count}")
        print(f"Missing embeddings (completed shots only): {missing_emb_count}")
        return 1

    print("\n=== Summary ===")
    print("OK: all pre-anchor DET rows have valid landmarks; completed shots have DET↔EMB parity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
