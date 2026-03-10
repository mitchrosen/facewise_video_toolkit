# To run:
# pytest -s tests/special/test_collect_bboxes_min_face1.py

import json
from pathlib import Path

import numpy as np
import pytest

from facekit.io.frame_provider import ReaderCoordinator
from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.detection.face_detector import FaceDetector


def _bbox_h(b):
    # bbox xyxy -> height
    x1, y1, x2, y2 = [int(v) for v in b[:4]]
    return max(0, y2 - y1)

pytest.mark.integration
def test_collect_bboxes_min_face1_smoke(tmp_path: Path):
    """
    Collect all detected bboxes on every frame with min_face=1.

    This is meant as a *data gathering* special test: it produces a JSON
    summary so we can pick realistic min_face thresholds for an integration test.
    """

    # --- YOU EDIT THESE PATHS ---
    video = Path("tests/assets/videos/crowd_test.mp4")  
    weights = Path("models/detector/yolov5n_state_dict.pt")           
    config = Path("models/detector/yolov5n.yaml")               
    device = "cpu"  # or "cuda" if you want

    assert video.exists(), f"Missing video clip: {video}"
    assert weights.exists(), f"Missing weights: {weights}"
    assert config.exists(), f"Missing config: {config}"

    # min_face=1: effectively no size filtering in the model wrapper
    yolo = load_yolo5face_model(str(weights), str(config), min_face=1, device=device)
    detector = FaceDetector(yolo)

    per_frame = []
    all_heights = []

    with ReaderCoordinator(str(video)) as fp:
        total = int(fp.total_frames() or 0)

        # If your ReaderCoordinator doesn’t support total_frames well for tiny clips,
        # we’ll just iterate until fp.next() returns None.
        fp.reset_to_frame(0)

        frame_idx = 0
        while True:
            frame = fp.next()
            if frame is None:
                break

            det = detector.detect_faces_in_frame(frame, target_size=640)
            if not det:
                per_frame.append({"frame": frame_idx, "count": 0, "bboxes": [], "heights": []})
                frame_idx += 1
                continue

            boxes, lms, confs = det

            bxs = [tuple(int(v) for v in b[:4]) for b in boxes]
            hs = [_bbox_h(b) for b in bxs]

            all_heights.extend(hs)

            per_frame.append(
                {
                    "frame": frame_idx,
                    "count": len(bxs),
                    "bboxes": bxs,
                    "heights": hs,
                    "confs": [float(c) if c is not None else None for c in confs],
                }
            )
            frame_idx += 1

    all_heights = [int(h) for h in all_heights if h is not None]
    all_heights.sort()

    summary = {
        "video": str(video),
        "min_face": 1,
        "frames": len(per_frame),
        "total_detections": int(sum(f["count"] for f in per_frame)),
        "height_stats": {
            "min": int(all_heights[0]) if all_heights else None,
            "p10": int(np.percentile(all_heights, 10)) if all_heights else None,
            "p25": int(np.percentile(all_heights, 25)) if all_heights else None,
            "p50": int(np.percentile(all_heights, 50)) if all_heights else None,
            "p75": int(np.percentile(all_heights, 75)) if all_heights else None,
            "p90": int(np.percentile(all_heights, 90)) if all_heights else None,
            "max": int(all_heights[-1]) if all_heights else None,
        },
        "per_frame": per_frame,
    }

    out = tmp_path / "bbox_report_min_face_1.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote bbox report: {out}\n")

    # assertions intentionally weak, this is data collection.
    assert summary["frames"] > 0
    assert summary["total_detections"] >= 0
