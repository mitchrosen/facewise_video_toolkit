# facekit/speaker/speaker_heuristics.py
from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

Landmarks = List[List[float]]         # [[x,y], ...] length >= 5 (LE, RE, N, ML, MR)
FrameLandmarks = Dict[int, Landmarks] # {global_id: landmarks}
WindowResult = Tuple[Tuple[int, int], Optional[int], float, float, float] #((start_idx, end_idx), global_id of detected speaker or None, combined_score, var_vert, var_width)

def compute_mouth_features_5pt(landmarks: Landmarks) -> Tuple[float, float]:
    """
    Extract two normalized features from 5-point landmarks:
      - mouth_width_ratio: |ML - MR| / |LE - RE|
      - mouth_mid_vertical_offset_ratio: ((ML_y+MR_y)/2 - N_y) / |LE - RE|

    Landmarks order: [left_eye, right_eye, nose, mouth_left, mouth_right]
    Returns (0.0, 0.0) if invalid.
    """
    if len(landmarks) < 5:
        return 0.0, 0.0

    le, re, n, ml, mr = map(np.asarray, landmarks[:5])
    eye_dist = float(np.linalg.norm(le - re))
    if eye_dist < 1e-6:
        return 0.0, 0.0

    mouth_width_ratio = float(np.linalg.norm(ml - mr) / eye_dist)
    mouth_mid_y = 0.5 * (ml[1] + mr[1])
    mouth_mid_vertical_offset_ratio = float((mouth_mid_y - n[1]) / eye_dist)

    return mouth_width_ratio, mouth_mid_vertical_offset_ratio


def active_speaker_windows_from_5pt(
    frames_landmarks: List[FrameLandmarks],
    window_size: int = 30,
    stride: Optional[int] = None,
    w_vert: float = 1.0,
    w_width: float = 0.5,
    min_samples: int = 5,
    force_winner: bool = True,
    ema_alpha: Optional[float] = None,
) -> List[WindowResult]:
    """
    Pick a likely speaker per window using 5-pt landmarks only.

    Args:
        frames_landmarks: list over time of {global_id: landmarks(5x2)} dicts.
        window_size: #frames per analysis window.
        stride: hop between windows; defaults to window_size (non-overlapping).
        w_vert: weight on variance of vertical-offset feature.
        w_width: weight on variance of mouth-width feature.
        min_samples: min #valid samples per ID in window to score it.
        force_winner: if True, always return the argmax (even if all weak).
        ema_alpha: if set (0<alpha<=1), apply EMA smoothing to features
                   inside the window before computing variance (reduces jitter).

    Returns:
        List of results per window:
            ((start_idx, end_idx), global_id of detected speaker or None, combined_score, var_vert, var_width)
        where combined_score = w_vert*var_vert + w_width*var_width for the winner.
    """
    if stride is None:
        stride = window_size

    results: List[WindowResult] = []
    n_frames = len(frames_landmarks)

    def _ema(vals: List[float], alpha: float) -> List[float]:
        if not vals:
            return vals
        out = [vals[0]]
        for x in vals[1:]:
            out.append(alpha * x + (1 - alpha) * out[-1])
        return out

    for start in range(0, n_frames, stride):
        end = min(start + window_size, n_frames)
        if end - start <= 1:
            break

        # Accumulate time series per global_id
        widths: Dict[int, List[float]] = defaultdict(list)
        verts: Dict[int, List[float]] = defaultdict(list)

        for f in range(start, end):
            lm_by_id = frames_landmarks[f]
            for gid, lm in lm_by_id.items():
                w, v = compute_mouth_features_5pt(lm)
                # Keep raw signed vertical (variance cares about oscillation)
                widths[gid].append(w)
                verts[gid].append(v)

        # Compute per-ID variances (optionally after smoothing)
        scores: Dict[int, Tuple[float, float, float]] = {}  # gid -> (combined, var_v, var_w)
        for gid in set(list(widths.keys()) + list(verts.keys())):
            w_series = widths.get(gid, [])
            v_series = verts.get(gid, [])
            if len(w_series) < min_samples or len(v_series) < min_samples:
                continue

            if ema_alpha is not None:
                w_series = _ema(w_series, ema_alpha)
                v_series = _ema(v_series, ema_alpha)

            var_w = float(np.var(w_series)) if len(w_series) > 1 else 0.0
            var_v = float(np.var(v_series)) if len(v_series) > 1 else 0.0
            combined = w_vert * var_v + w_width * var_w
            scores[gid] = (combined, var_v, var_w)

        if scores:
            # Argmax combined score; break ties by larger var_v, then var_w, then lowest gid
            winner = max(
                scores.items(),
                key=lambda kv: (kv[1][0], kv[1][1], kv[1][2], -kv[0])  # prefer higher vars
            )[0]
            combined, var_v, var_w = scores[winner]
            results.append(((start, end), winner, combined, var_v, var_w))
        else:
            # No ID had enough samples; return None or force-winner via weakest rule
            if force_winner:
                # Consider any ID with at least 2 samples; pick the one with max crude variance
                fallback: Dict[int, Tuple[float, float, float]] = {}
                for gid in set(list(widths.keys()) + list(verts.keys())):
                    w_series = widths.get(gid, [])
                    v_series = verts.get(gid, [])
                    if len(w_series) >= 2 and len(v_series) >= 2:
                        var_w = float(np.var(w_series))
                        var_v = float(np.var(v_series))
                        combined = w_vert * var_v + w_width * var_w
                        fallback[gid] = (combined, var_v, var_w)
                if fallback:
                    winner = max(
                        fallback.items(),
                        key=lambda kv: (kv[1][0], kv[1][1], kv[1][2], -kv[0])
                    )[0]
                    combined, var_v, var_w = fallback[winner]
                    results.append(((start, end), winner, combined, var_v, var_w))
                else:
                    results.append(((start, end), None, 0.0, 0.0, 0.0))
            else:
                results.append(((start, end), None, 0.0, 0.0, 0.0))

    return results
