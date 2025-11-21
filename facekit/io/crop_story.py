from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional
import cv2
import numpy as np
import tempfile
import os

class AlignedCropStore:
    """
    Saves 112x112 RGB-aligned face crops as PNGs under a stable path:

        <base_dir>/crops/shot_<shot>/tid_<tid>/frame_<frame>.png

    - Saves atomically (tmp file + os.replace).
    - Converts RGB <-> BGR for OpenCV I/O.
    - Provides batch load helpers to gather crops for specific frames.
    """

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.root = self.base_dir  # alias for readability
        (self.root).mkdir(parents=True, exist_ok=True)

    # ---------- path helpers ----------
    def crop_path(self, shot: int, tid: int, frame: int) -> Path:
        return self.root / f"crops/shot_{int(shot)}/tid_{int(tid)}/frame_{int(frame)}.png"

    def ensure_dirs(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- save/load ----------
    def save_png(self, *, shot: int, tid: int, frame: int, img_rgb: np.ndarray) -> Path:
        """
        Save a single 112x112 RGB crop as PNG (atomic). Returns final path.
        """
        if not isinstance(img_rgb, np.ndarray):
            raise TypeError("img_rgb must be a numpy array")
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"img_rgb must be HxWx3 RGB; got shape={img_rgb.shape}")

        dst = self.crop_path(shot, tid, frame)
        self.ensure_dirs(dst)

        # cv2.imwrite expects BGR; convert from RGB
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # atomic write
        with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False, suffix=".png") as tmp:
            tmp_path = Path(tmp.name)
            ok = cv2.imwrite(str(tmp_path), img_bgr)
            if not ok:
                try:
                    tmp.close()
                finally:
                    tmp_path.unlink(missing_ok=True)
                raise IOError(f"Failed to write PNG to {tmp_path}")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, dst)
        # fsync parent to harden (best-effort)
        try:
            fd = os.open(str(dst.parent), os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            pass

        return dst

    def load_png(self, *, shot: int, tid: int, frame: int) -> Optional[np.ndarray]:
        """
        Load a single crop as RGB. Returns None if missing or unreadable.
        """
        path = self.crop_path(shot, tid, frame)
        if not path.exists():
            return None
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def load_many(self, *, shot: int, tid: int, frames: Iterable[int]) -> List[np.ndarray]:
        """
        Load crops for the given frame indices (in order). Skips missing.
        Returns a list of RGB np.ndarrays.
        """
        out: List[np.ndarray] = []
        for f in frames:
            img = self.load_png(shot=shot, tid=tid, frame=int(f))
            if img is not None:
                out.append(img)
        return out

    def list_available_frames(self, *, shot: int, tid: int) -> List[int]:
        """
        List frames for which a crop exists (sorted).
        """
        d = self.root / f"crops/shot_{int(shot)}/tid_{int(tid)}"
        if not d.exists():
            return []
        frames = []
        for p in d.glob("frame_*.png"):
            try:
                n = int(p.stem.split("_", 1)[1])
                frames.append(n)
            except Exception:
                continue
        return sorted(frames)
