import cv2
import numpy as np

class FaceTracker:
    def __init__(self, tracker_type="CSRT"):
        self.tracker_type = tracker_type
        self.trackers = []  # list of (track_id, tracker) tuples

    def _create_tracker(self):
        # Try modern API first
        if self.tracker_type == "CSRT":
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
            elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                return cv2.legacy.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            if hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()
            elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
                return cv2.legacy.TrackerKCF_create()
        raise RuntimeError(f"Unsupported tracker type: {self.tracker_type}")

    def init_trackers(self, frame, boxes, track_ids):
        """Initialize one tracker per box and store associated track_id."""
        self.trackers = []
        for box, track_id in zip(boxes, track_ids):
            tracker = self._create_tracker()
            try:
                tracker.init(frame, tuple(box))
                self.trackers.append((track_id, tracker))
            except Exception as e:
                print(f"Failed to initialize tracker for track_id={track_id} box={box}: {e}", flush=True)

    def update_trackers(self, frame):
        """Return dict mapping track_id to updated box, only if update succeeded."""
        updated = {}
        surviving_trackers = []

        for track_id, tracker in self.trackers:
            success, box = tracker.update(frame)
            if success:
                updated[track_id] = box
                surviving_trackers.append((track_id, tracker))
            else:
                print(f"[DEBUG] Tracker for track_id={track_id} failed — removing")

        self.trackers = surviving_trackers
        return updated

def draw_tracked_face_box(frame, box, color_name="tracked"):
    """
    Draws a face bounding box with a specified color:
        - "detected": Yellow (scheduled detection)
        - "fallback": Orange (detection due to tracker failure)
        - "tracked": Blue (from tracker)
    """
    color_map = {
        "detected": (0, 255, 255),  # yellow
        "fallback": (0, 90, 255),  # orange
        "tracked": (255, 0, 0)      # blue
    }
    color = color_map.get(color_name, (255, 0, 255))  # fallback: magenta

    x, y, w, h = map(int, box)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
