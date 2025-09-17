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
        """
        Update all active trackers with `frame`.

        Returns
        -------
        dict[int, tuple|None]
            Mapping track_id -> (x, y, w, h) on success, or None if that tracker failed.
            NOTE: Failed trackers are removed from self.trackers after reporting None.
        """
        results = {}
        surviving = []

        # Snapshot the ids we’re attempting this frame, so every id gets a result
        for track_id, tracker in list(self.trackers):
            ok, box = tracker.update(frame)
            if ok:
                results[track_id] = box  # (x, y, w, h)
                surviving.append((track_id, tracker))
            else:
                results[track_id] = None

        self.trackers = surviving
        return results

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
