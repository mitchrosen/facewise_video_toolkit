#!/usr/bin/env python3
"""
probe_sidecars.py

Usage:
  python probe_sidecars.py --run-root /path/to/run-YYYY... --anchor 180 --det-code 1

What it does:
  - Loads obs sidecar: <run-root>/ckpt/obs_ckpt.npz  (key 'observations')
  - Loads emb sidecar: <run-root>/ckpt/emb_ckpt.npz  (key usually 'embeddings' or 'vecs')
  - Treats the given `anchor` as the **embedding-safe** resume anchor frame.
  - Enforces a narrow, enforceable invariant:
      * We cannot infer which DET rows produced a valid aligned face.
      * Therefore we do NOT require DET rows pre-anchor to have embeddings.
      * We only enforce: any DET row pre-anchor that *claims* an embedding (emb_idx >= 0)
        must reference a valid embedding row (emb_idx < num_embeddings).

  - Exits 0 if all invariants hold.
  - Exits 1 otherwise, printing a summary of the violations.

Notes:
  - This script intentionally does not check anything about crops or landmarks.
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

    print(f"Anchor: {anchor} (checked frames <= {anchor-1})")
    print(f"DET code: {det_code}")

    pre_anchor_det_mask = (src == det_code) & (f <= anchor - 1)
    if not pre_anchor_det_mask.any():
        print("No pre-anchor DET rows found; nothing to check.")
        return 0

    # Only enforce emb_idx validity for rows that claim embeddings.
    claims_emb_mask = pre_anchor_det_mask & (emb_idx >= 0)
    bad_emb_mask = claims_emb_mask & (emb_idx >= n_emb)
    bad_count = int(bad_emb_mask.sum())

    # For debugging, report totals
    det_rows_total = int(pre_anchor_det_mask.sum())
    print("\n=== DET→EMB checks (pre-anchor) ===")
    print(f"Total DET rows considered (pre-anchor, all shots): {det_rows_total}")
    print(f"DET pre-anchor rows claiming embeddings (emb_idx >= 0): {int(claims_emb_mask.sum())}")
    print(f"Embedding rows in sidecar: {n_emb}")
    print(f"Invalid emb_idx among claimed embeddings: {bad_count}")

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

    if bad_count:
        _sample(bad_emb_mask, "completed-shot DET rows with invalid emb_idx")

    # Decide exit code: any out-of-range emb_idx on a row that claims an embedding is an error.
    if bad_count:
        print("\n=== Summary ===")
        print(f"Pre-anchor DET rows: {det_rows_total}")
        print(f"Invalid emb_idx among claimed embeddings: {bad_count}")
        return 1

    print("\n=== Summary ===")
    print("OK: all claimed embeddings (DET pre-anchor rows with emb_idx>=0) are resolvable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
