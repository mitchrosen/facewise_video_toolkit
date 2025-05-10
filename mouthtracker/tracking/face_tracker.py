import cv2
import numpy as np

class FaceTracker:
    def __init__(self, tracker_type="CSRT"):
        self.tracker_type = tracker_type
        self.trackers = []
        self.track_fail_counts = []

    def _create_tracker(self):
        if self.tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        else:
            raise ValueError(f"Unsupported tracker type: {self.tracker_type}")

    def init_trackers(self, frame, boxes):
        self.trackers = []
        self.track_fail_counts = []
        print(f"Initializing tracker with boxes {boxes}", flush=True)
        for box in boxes:
            tracker = self._create_tracker()
            try:
                tracker.init(frame, tuple(box))
                self.trackers.append(tracker)
                self.track_fail_counts.append(0)
                print(f"✅ Tracker initialized for box {box}", flush=True)
            except Exception as e:
                print(f"❌ Failed to initialize tracker for box {box}: {e}", flush=True)

    def update_trackers(self, frame):
        updated_boxes = []
        for tracker in self.trackers:
            success, box = tracker.update(frame)
            if success:
                updated_boxes.append(box)
            else:
                updated_boxes.append(None)
        return updated_boxes


def draw_tracked_face_box(frame, box, color_name="tracked"):
    """
    Draws a face bounding box with a specified color:
        - "detected": Yellow (scheduled detection)
        - "fallback": Orange (detection due to tracker failure)
        - "tracked": Blue (from tracker)
    """
    color_map = {
        "detected": (0, 255, 255),  # yellow
        "fallback": (0, 165, 255),  # orange
        "tracked": (255, 0, 0)      # blue
    }
    color = color_map.get(color_name, (255, 0, 255))  # fallback: magenta

    x, y, w, h = map(int, box)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
