import os
import json
from pathlib import Path

import numpy as np
import pytest

from facekit.io.frame_provider import ReaderCoordinator
from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.detection.face_detector import FaceDetector


def _count_detections(video_path: Path, *, min_face: int, model_path: Path, config_path: Path, device: str) -> int:
    """
    Runs the real detector on every frame of the clip and returns total number of detections.
    Assumes the YOLO wrapper enforces min_face filtering internally.
    """
    yolo = load_yolo5face_model(
        str(model_path),
        str(config_path),
        device=device,
        min_face=int(min_face),
    )
    detector = FaceDetector(yolo)

    total = 0
    with ReaderCoordinator(str(video_path)) as fp:
        n = int(fp.total_frames())
        fp.reset_to_frame(0)
        for frame_idx in range(n):
            frame = fp.next()
            if frame is None:
                continue
            det = detector.detect_faces_in_frame(frame)
            if not det:
                continue
            boxes, lms, confs = det
            total += len(boxes)
    return int(total)


def _expected_counts_from_report(report_path: Path, thresholds: list[int]) -> dict[int, int]:
    """
    Given a min_face=1 report containing per-detection heights, compute the expected
    detection totals for each threshold: count(height >= threshold).
    """
    data = json.loads(report_path.read_text())
    heights = []
    for fr in data["per_frame"]:
        heights.extend(fr["heights"])
    arr = np.asarray(heights, dtype=np.float32)

    expected = {}
    for t in thresholds:
        expected[int(t)] = int((arr >= float(t)).sum())
    return expected


@pytest.mark.integration
def test_min_face_filtering_matches_report_distribution():
    video_path = Path("tests/assets/videos/crowd_test.mp4")
    report_path = Path("tests/assets/reports/bbox_report_min_face_1.json")

    model_path = Path("models/detector/yolov5n_state_dict.pt")
    config_path = Path("models/detector/yolov5n.yaml")

    assert video_path.exists(), f"Missing clip: {video_path}"
    assert report_path.exists(), f"Missing report: {report_path}"
    assert model_path.exists(), f"Missing model: {model_path}"
    assert config_path.exists(), f"Missing config: {config_path}"

    device = os.getenv("FACEKIT_DEVICE", "cpu")

    # Thresholds derived from report’s percentiles
    thresholds = [1, 43, 72, 97, 120]

    expected = _expected_counts_from_report(report_path, thresholds)

    # Sanity: report-based expected[1] should equal the report’s own total detections
    report_total = json.loads(report_path.read_text())["total_detections"]
    assert expected[1] == int(report_total)

    # Now run the real detector for each threshold and compare totals
    for t in thresholds:
        got = _count_detections(video_path, min_face=t, model_path=model_path, config_path=config_path, device=device)
        assert got == expected[t], f"min_face={t}: expected {expected[t]} detections, got {got}"
