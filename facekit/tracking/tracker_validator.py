from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import cv2
import numbers
from facekit.utils.geometry import compute_iou
import logging

BBox_XYWH = Tuple[float, float, float, float]

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

        if not (_is_num(self.hsv_thresh) and 0 <= self.hsv_thresh <= 2):
            raise ValueError(f"hsv_thresh must be in [0,2]; got {self.hsv_thresh}")
        
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
      - Construct per-shot
      - Call validate(tracked_boxes, frame_idx) on tracking frames.
        * On the first call (or on non-consecutive frames), it seeds itself and returns True.
        * On success, it updates its internal baseline to the current boxes.
        * On failure, it clears its baseline and returns False (coordinator should trigger detection).
 
    Internal state:
      - _prev_boxes: Dict[track_id, (x,y,w,h)] for last accepted frame
      - _prev_idx: last accepted absolute frame index
      - _signature_by_tid: optional appearance signatures (HSV hist) for last accepted frame
    """

    def __init__(self, params: ValidatorParams):
        assert 0.0 <= params.iou_thresh <= 1.0
        assert params.area_delta_max >= 0.0
        assert params.asp_ratio_delta_max >= 0.0
        assert params.v_max >= 0.0
        self.params = params

        self._prev_boxes: Optional[Dict[int, BBox_XYWH]] = None
        self._prev_idx: Optional[int] = None
        self._prev_hsv_signature_by_tid: Dict[int, np.ndarray] = {}

    def seed_validator(self, boxes_xywh: Dict[int, BBox_XYWH], frame_idx: int, frame:np.ndarray) -> None:
        hsv_signature_by_tid = self.hsv_signature_dict(boxes_xywh, frame)
        self.set_baseline(boxes_xywh, frame_idx, hsv_signature_by_tid)

    def hsv_signature_dict(self, boxes_xywh: Dict[int, BBox_XYWH], frame:np.ndarray) -> Dict[int, np.ndarray]:
        hsv_signature_by_tid = {}
        tids = set(boxes_xywh.keys())
        for tid in tids:
            box = boxes_xywh[tid]
            if box is None:
                hsv_signature_by_tid = None
            else:
                signature_curr = self._calc_hsv_signature(frame, box)
                hsv_signature_by_tid[tid] = signature_curr
        return hsv_signature_by_tid
        
    def set_baseline(self, boxes_xywh: Dict[int, BBox_XYWH], frame_idx: int, hsv_signature_by_tid: Dict[int, np.ndarray]) -> None:
        self._prev_boxes = dict(boxes_xywh)
        self._prev_idx = frame_idx
        self._prev_hsv_signature_by_tid = dict(hsv_signature_by_tid)

    def validate(self, 
                 curr_boxes: Dict[int, BBox_XYWH],
                 current_frame: np.ndarray,
                 frame_idx: int) -> bool:
        """
        Validate continuity against last accepted state.
        - If we have no baseline or frames aren’t consecutive, seed baseline from current and return True.
        - Otherwise, require:
            * all previous tids to still be present
            * IoU(prev, curr) >= iou_thresh
            * |area_delta| <= area_delta_max
            * |asp_ratio_delta| <= asp_ratio_delta_max
            * center shift / prev_diag <= v_max
            * HSV appearance distance <= hsv_thresh
        On success: baseline <- current.
        On failure: baseline cleared (so next call won’t compare against a stale state).
        """
        if not curr_boxes:
            self._clear_baseline()
            return False

        # First time or non-consecutive → seed & accept
        if (self._prev_boxes is None or
            self._prev_idx is None or
            frame_idx != self._prev_idx + 1):

            self.seed_validator(curr_boxes, frame_idx, current_frame)
            return True

        prev = self._prev_boxes
        prev_ids = set(prev.keys())
        curr_ids = set(curr_boxes.keys())

        # No missing tracks
        if not prev_ids.issubset(curr_ids):
            self._clear_baseline()
            return False

        curr_hsv_signature_by_tid = {}

        # iterate through tracks
        for tid in prev_ids:

            # No missing boxes
            b0 = prev[tid]
            b1 = curr_boxes[tid]
            if b0 is None or b1 is None:
                self._clear_baseline()
                return False

            # IoU is within bounds
            iou = compute_iou(self._xywh_to_xyxy(b0), self._xywh_to_xyxy(b1))
            if iou < self.params.iou_thresh:
                self._clear_baseline()
                return False

            # Area changes are within bounds
            if not self._area_ok(b0, b1, self.params.area_delta_max):
                self._clear_baseline()
                return False

            # aspect ratio
            if not self._asp_ratio_ok(b0, b1, self.params.asp_ratio_delta_max):
                self._clear_baseline()
                return False

            # velocity
            if not self._velocity_ok(b0, b1, self.params.v_max):
                self._clear_baseline()
                return False

            # appearance
            signature_prev = self._prev_hsv_signature_by_tid.get(tid)
            signature_curr = self._calc_hsv_signature(current_frame, b1)
            if signature_prev is None or signature_curr is None:
                self._clear_baseline()
                return False
            if self._hsv_dist(signature_prev, signature_curr) > self.params.hsv_thresh:
                self._clear_baseline()
                return False
            
            curr_hsv_signature_by_tid[tid] = signature_curr

        # Success; update baseline to current
        self.set_baseline(curr_boxes, frame_idx, curr_hsv_signature_by_tid)
        return True
    
    def provenance(self) -> dict:
        return self.params.to_generation_dict()

    # --- helpers ---
    def _clear_baseline(self):
        self._prev_boxes = None
        self._prev_idx = None
        self._prev_hsv_signature_by_tid.clear()

    @staticmethod
    def _xywh_to_xyxy(b: BBox_XYWH):
        x,y,w,h = b
        return (x, y, x+w, y+h)

    @staticmethod
    def _area_ok(b0: BBox_XYWH, b1: BBox_XYWH, max_frac: float, eps: float = 1e-6) -> bool:
        _,_,w0,h0 = b0; _,_,w1,h1 = b1
        a0 = max(w0*h0, eps); a1 = max(w1*h1, eps)
        return abs(a1 - a0) / a0 <= max_frac

    @staticmethod
    def _asp_ratio_ok(b0: BBox_XYWH, b1: BBox_XYWH, max_frac: float, eps: float = 1e-6) -> bool:
        _,_,w0,h0 = b0; _,_,w1,h1 = b1
        asp0 = w0 / max(h0, eps)
        asp1 = w1 / max(h1, eps)
        return abs(asp1 - asp0) / max(asp0, eps) <= max_frac

    @staticmethod
    def _velocity_ok(b0: BBox_XYWH, b1: BBox_XYWH, v_max: float, eps: float = 1e-6) -> bool:
        x0,y0,w0,h0 = b0; x1,y1,w1,h1 = b1
        c0x, c0y = x0 + 0.5*w0, y0 + 0.5*h0
        c1x, c1y = x1 + 0.5*w1, y1 + 0.5*h1
        shift = ((c1x - c0x)**2 + (c1y - c0y)**2) ** 0.5
        d0 = (w0*w0 + h0*h0) ** 0.5
        return (shift / max(d0, eps)) <= v_max

    @staticmethod
    def _calc_hsv_signature(frame: np.ndarray, b: BBox_XYWH) -> Optional[np.ndarray]:
        x,y,w,h = map(int, b)
        if w <= 0 or h <= 0: return None
        H,W = frame.shape[:2]
        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        x2 = max(x+1, min(x+w, W)); y2 = max(y+1, min(y+h, H))

        crop = frame[y:y2, x:x2]
        if crop.size == 0: return None
        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0,1,2], None, [8,4,4], [0,180, 0,256, 0,256]).flatten()
        hist /= (hist.sum() + 1e-6)
        return hist

    @staticmethod
    def _hsv_dist(a: np.ndarray, b: np.ndarray) -> float:
        # 1 - cosine similarity
        num = float((a*b).sum())
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        return 1.0 - (num / den)
