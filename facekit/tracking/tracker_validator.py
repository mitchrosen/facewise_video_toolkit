from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import cv2
import numbers
from facekit.utils.geometry import compute_iou
import logging

BBoxXYWH = Tuple[float, float, float, float]

@dataclass
class ValidatorParams:
    iou_thresh: float = 0.5           # IoU must be >= this
    area_delta_max: float = 0.50      # max fractional change in area (e.g., 0.5 = ±50%)
    asp_ratio_delta_max: float = 0.50 # max fractional change in aspect ratio (w/h)
    v_max: float = 0.80               # max center shift normalized by previous diag
    hsv_thresh: float = 0.35          # 1 - cosine(hist_prev, hist_curr) must be <= this

    def __post_init__(self):
        def _is_num(x): return isinstance(x, numbers.Real) and np.isfinite(x)

        if not (_is_num(self.iou_thresh) and 0.0 <= self.iou_thresh <= 1.0):
            raise ValueError(f"iou_thresh must be in [0,1]; got {self.iou_thresh}")

        for name, val in (("area_delta_max", self.area_delta_max),
                          ("asp_ratio_delta_max", self.asp_ratio_delta_max),
                          ("v_max", self.v_max)):
            if not (_is_num(val) and val >= 0.0):
                raise ValueError(f"{name} must be >= 0; got {val}")

        if not (_is_num(self.hsv_thresh) and -1.0 <= self.hsv_thresh <= 1.0):
            raise ValueError(f"hsv_thresh must be in [-1,1]; got {self.hsv_thresh}")
        
    def to_generation_dict(self) -> dict:
        """Stable names."""
        return {
            "iou": float(self.iou_thresh),
            "area_delta": float(self.area_delta_max),
            "asp_delta": float(self.asp_ratio_delta_max),
            "v_max": float(self.v_max),
            "hsv_thresh": float(self.hsv_thresh),
        }

class TrackerValidator:
    """
    Stateful validator for frame-to-frame tracking continuity within a shot.

    Usage:
      - Construct per-shot with the shot's frames and first_frame_idx.
      - Call validate(tracked_boxes, frame_idx) on tracking frames.
        * On the first call (or on non-consecutive frames), it seeds itself and returns True.
        * On success, it updates its internal baseline to the current boxes.
        * On failure, it clears its baseline and returns False (coordinator should trigger detection).
      - Optionally call set_baseline() right after a detection (re)init to seed with detector boxes.

    Internal state:
      - _prev_boxes: Dict[track_id, (x,y,w,h)] for last accepted frame
      - _prev_idx: last accepted absolute frame index
      - _sig_by_tid: optional appearance signatures (HSV hist) for last accepted frame
    """

    def __init__(self, frames: List[np.ndarray], first_frame_idx: int, params: ValidatorParams):
        assert 0.0 < params.iou_thresh <= 1.0
        assert params.area_delta_max >= 0.0
        assert params.asp_ratio_delta_max >= 0.0
        assert params.v_max >= 0.0
        self._frames = frames
        self._first = first_frame_idx
        self.p = params

        self._prev_boxes: Optional[Dict[int, BBoxXYWH]] = None
        self._prev_idx: Optional[int] = None
        self._sig_by_tid: Dict[int, np.ndarray] = {}

    def set_baseline(self, boxes_xywh: Dict[int, BBoxXYWH], frame_idx: int) -> None:
        self._prev_boxes = dict(boxes_xywh)
        self._prev_idx = frame_idx
        self._sig_by_tid = {}
        
        frame = self._frame_from_abs(frame_idx)
        for tid, b in boxes_xywh.items():
            sig = self._hsv_sig(frame, b)
            if sig is not None:
                self._sig_by_tid[tid] = sig

    def validate(self, curr_boxes: Dict[int, BBoxXYWH], 
                 frame_idx: int,
                 verbose: bool=False) -> bool:
        """
        Validate continuity against last accepted state.
        - If we have no baseline or frames aren’t consecutive, seed baseline from current and return True.
        - Otherwise, require:
            * all previous tids still present
            * IoU(prev, curr) >= iou_thresh
            * |area_delta| <= area_delta_max
            * |asp_ratio_delta| <= asp_ratio_delta_max
            * center shift / prev_diag <= v_max
            * (optional) HSV appearance distance <= hsv_thresh
        On success: baseline <- current (including HSV signatures).
        On failure: baseline cleared (so next call won’t compare against a stale state).
        """
        if not curr_boxes:
            self._clear_baseline()
            return False

        # First time or non-consecutive → seed & accept
        if (self._prev_boxes is None) or (self._prev_idx is None) or (frame_idx != self._prev_idx + 1):
            self.set_baseline(curr_boxes, frame_idx)
            return True

        prev = self._prev_boxes
        prev_ids = set(prev.keys())
        curr_ids = set(curr_boxes.keys())

        # No missing tracks
        if not prev_ids.issubset(curr_ids):
            self._clear_baseline()
            if verbose:
                logging.debug(f"[TRACK VALIDATION] missing ids: {sorted(prev_ids - curr_ids)}")
            return False

        # iterate through tracks
        for tid in prev_ids:

            # No missing boxes
            b0 = prev[tid]; b1 = curr_boxes[tid]
            if b0 is None or b1 is None:
                self._clear_baseline()
                if verbose: logging.debug(f"[TRACK VALIDATION] tid={tid} missing box")
                return False

            # IoU is within bounds
            iou = compute_iou(self._xywh_to_xyxy(b0), self._xywh_to_xyxy(b1))
            if iou < self.p.iou_thresh:
                self._clear_baseline()
                if verbose: logging.debug(f"[TRACK VALIDATION] tid={tid} IoU={iou:.3f} (min {self.p.iou_thresh})")
                return False

            # Area changes are within bounds
            if not self._area_ok(b0, b1, self.p.area_delta_max, verbose):
                self._clear_baseline()
                if verbose: logging.debug(f"[TRACK VALIDATION] tid={tid} area not within bounds (max {self.p.area_delta_max})")
                return False

            # aspect ratio (use asp_ratio terminology)
            if not self._asp_ratio_ok(b0, b1, self.p.asp_ratio_delta_max):
                self._clear_baseline()
                if verbose: logging.debug(f"[TRACK VALIDATION] tid={tid} asp ratio not within bounds (max {self.p.asp_ratio_delta_max})")
                return False

            # velocity
            if not self._velocity_ok(b0, b1, self.p.v_max):
                self._clear_baseline()
                if verbose: logging.debug(f"[TRACK VALIDATION] tid={tid} velocity not within bounds (max {self.p.v_max})")
                return False

            # appearance
            frame = self._frame_from_abs(frame_idx)
            sig_prev = self._sig_by_tid.get(tid)
            sig_curr = self._hsv_sig(frame, b1)
            if sig_prev is None or sig_curr is None:
                self._clear_baseline()
                if verbose: logging.debug(f"[TRACK VALIDATION] tid={tid} appearance fails due to lack of previous signature")
                return False
            if self._hsv_dist(sig_prev, sig_curr) > self.p.hsv_thresh:
                self._clear_baseline()
                if verbose: logging.debug(f"[TRACK VALIDATION] tid={tid} appearance not within bounds (max {self.p.hsv_thresh})")
                return False

        # Success; update baseline to current
        self.set_baseline(curr_boxes, frame_idx)
        return True
    
    def provenance(self) -> dict:
        return self.p.to_generation_dict()

    # --- helpers ---
    def _clear_baseline(self):
        self._prev_boxes = None
        self._prev_idx = None
        self._sig_by_tid.clear()

    def _frame_from_abs(self, abs_idx: int) -> np.ndarray:
        local = abs_idx - self._first
        if local < 0 or local >= len(self._frames):
            raise IndexError(f"[TRACK VALIDATION] Frame {abs_idx} out of shot range [{self._first}, {self._first+len(self._frames)-1}]")
        return self._frames[local]

    @staticmethod
    def _xywh_to_xyxy(b: BBoxXYWH):
        x,y,w,h = b
        return (int(x), int(y), int(x+w), int(y+h))

    @staticmethod
    def _area_ok(b0: BBoxXYWH, b1: BBoxXYWH, max_frac: float, eps: float = 1e-6) -> bool:
        _,_,w0,h0 = b0; _,_,w1,h1 = b1
        a0 = max(w0*h0, eps); a1 = max(w1*h1, eps)
        return abs(a1 - a0) / a0 <= max_frac

    @staticmethod
    def _asp_ratio_ok(b0: BBoxXYWH, b1: BBoxXYWH, max_frac: float, eps: float = 1e-6) -> bool:
        _,_,w0,h0 = b0; _,_,w1,h1 = b1
        asp0 = w0 / max(h0, eps)
        asp1 = w1 / max(h1, eps)
        return abs(asp1 - asp0) / max(asp0, eps) <= max_frac

    @staticmethod
    def _velocity_ok(b0: BBoxXYWH, b1: BBoxXYWH, v_max: float, eps: float = 1e-6) -> bool:
        x0,y0,w0,h0 = b0; x1,y1,w1,h1 = b1
        c0x, c0y = x0 + 0.5*w0, y0 + 0.5*h0
        c1x, c1y = x1 + 0.5*w1, y1 + 0.5*h1
        shift = ((c1x - c0x)**2 + (c1y - c0y)**2) ** 0.5
        d0 = (w0*w0 + h0*h0) ** 0.5
        return (shift / max(d0, eps)) <= v_max

    @staticmethod
    def _hsv_sig(frame: np.ndarray, b: BBoxXYWH) -> Optional[np.ndarray]:
        x,y,w,h = map(int, b)
        if w <= 0 or h <= 0: return None
        H,W = frame.shape[:2]
        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        x2 = max(x+1, min(x+w, W)); y2 = max(y+1, min(y+h, H))

        crop = frame[y:y2, x:x2]
        if crop.size == 0: return None
        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # 16x8x8 bins across H,S,V; normalized
        hist = cv2.calcHist([hsv], [0,1,2], None, [8,4,4], [0,180, 0,256, 0,256]).flatten()
        hist /= (hist.sum() + 1e-6)
        return hist

    @staticmethod
    def _hsv_dist(a: np.ndarray, b: np.ndarray) -> float:
        # 1 - cosine similarity
        num = float((a*b).sum())
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        return 1.0 - (num / den)
