import cv2
from typing import List, Tuple, Optional


def get_mouth_box_from_landmarks(landmarks: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """
    Accepts 5-point landmark format and returns a bounding box around the mouth region.

    Assumes:
    - landmark 2 = center (approximate middle of mouth)
    - landmark 3 = left corner
    - landmark 4 = right corner

    Adds a small padding.
    """
    if len(landmarks) < 5:
        raise ValueError(f"Expected at least 5 landmarks, got {len(landmarks)}")

    mouth_pts = [landmarks[2], landmarks[3], landmarks[4]]
    xs = [pt[0] for pt in mouth_pts]
    ys = [pt[1] for pt in mouth_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad = 5
    return (
        max(0, x_min - pad),
        max(0, y_min - pad),
        (x_max - x_min) + 2 * pad,
        (y_max - y_min) + 2 * pad,
    )


class MouthTracker:
    def __init__(self, tracker_type: str = "CSRT"):
        self.tracker_type = tracker_type
        self.trackers = []
        self.tracker_boxes = []
        self.track_fail_counts = []

    def _create_tracker(self):
        if self.tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        raise ValueError(f"Unsupported tracker type: {self.tracker_type}")

    def init_trackers(self, frame, landmark_sets: List[List[Tuple[int, int]]]):
        self.trackers.clear()
        self.tracker_boxes.clear()
        self.track_fail_counts.clear()
        for landmarks in landmark_sets:
            box = get_mouth_box_from_landmarks(landmarks)
            tracker = self._create_tracker()
            tracker.init(frame, box)
            self.trackers.append(tracker)
            self.tracker_boxes.append(box)
            self.track_fail_counts.append(0)

    def update_trackers(self, frame) -> List[Optional[Tuple[int, int, int, int]]]:
        new_boxes = []
        for i, tracker in enumerate(self.trackers):
            success, box = tracker.update(frame)
            if success:
                self.track_fail_counts[i] = 0
                new_boxes.append(box)
            else:
                self.track_fail_counts[i] += 1
                new_boxes.append(None)
        self.tracker_boxes = new_boxes
        return new_boxes
