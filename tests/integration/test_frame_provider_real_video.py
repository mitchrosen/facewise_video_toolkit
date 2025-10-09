import os
from pathlib import Path
import cv2
import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim
from facekit.io.frame_provider import ReaderCoordinator

IDX_LIST = [0, 5, 50, 100]

def _resolve_video_path() -> str:
    env = os.environ.get("FACEKIT_TEST_VIDEO")
    if env and Path(env).exists():
        return env
    candidate = Path(__file__).parents[2] / "tests" / "data" / "interview-sam-altman_5sec_snippet.mp4"
    if candidate.exists():
        return str(candidate)
    pytest.skip("Set FACEKIT_TEST_VIDEO or place the sample video at tests/data/…")

@pytest.fixture(scope="module")
def video_path():
    return _resolve_video_path()

@pytest.fixture(scope="module")
def ocv_frames(video_path):
    """Read once with OpenCV up to max index; cache only what we need."""
    need = set(IDX_LIST)
    max_idx = max(need)
    out = {}
    cap = cv2.VideoCapture(video_path)
    try:
        i = -1
        while i < max_idx:
            ok, f = cap.read()
            i += 1
            if not ok:
                break
            if i in need:
                out[i] = f.copy()
    finally:
        cap.release()
    # ensure all requested frames were captured
    missing = [i for i in IDX_LIST if i not in out]
    if missing:
        pytest.skip(f"Video shorter than required indices; missing {missing}")
    return out

@pytest.fixture(scope="module")
def prov(video_path):
    p = ReaderCoordinator(video_path)
    try:
        _ = p.get_frame(0)  # warm-up (build index/metadata)
        yield p
    finally:
        p.close()

@pytest.mark.integration
def test_readercoordinator_matches_opencv_random_access(prov, ocv_frames):
    errors = []
    for idx in IDX_LIST:
        ra = prov.get_frame(idx)
        ocv = ocv_frames[idx]
        if ra is None:
            errors.append(f"[parity] idx={idx}: ReaderCoordinator returned None")
            continue
        if ra.shape != ocv.shape:
            errors.append(f"[parity] idx={idx}: shape {ra.shape} != {ocv.shape}")
            continue
        if not np.array_equal(ra, ocv):
            g1 = cv2.cvtColor(ra,  cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(ocv, cv2.COLOR_BGR2GRAY)
            v = ssim(g1, g2)
            if v < 0.999:
                errors.append(f"[parity] idx={idx}: SSIM {v:.6f} < 0.999")
    assert not errors, ";\n".join(errors)

@pytest.mark.integration
def test_readercoordinator_sequential_matches_opencv(prov, ocv_frames):
    errors = []
    for idx in IDX_LIST:
        prov.reset_to_frame(idx)
        seq = prov.next()
        ocv = ocv_frames[idx]
        if seq is None:
            errors.append(f"[seq parity] idx={idx}: next() returned None")
            continue
        if seq.shape != ocv.shape:
            errors.append(f"[seq parity] idx={idx}: shape {seq.shape} != {ocv.shape}")
            continue
        if not np.array_equal(seq, ocv):
            g1 = cv2.cvtColor(seq, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(ocv, cv2.COLOR_BGR2GRAY)
            v = ssim(g1, g2)
            if v < 0.999:
                errors.append(f"[seq parity] idx={idx}: SSIM {v:.6f} < 0.999")
    assert not errors, ";\n".join(errors)
