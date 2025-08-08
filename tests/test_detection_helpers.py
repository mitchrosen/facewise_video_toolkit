from pathlib import Path
from datetime import datetime
import unittest
import numpy as np
import cv2
from unittest.mock import MagicMock

from facekit.detection.detection_helpers import draw_faces_and_mouths
from facekit.detection.face_detector import FaceDetector

def save_debug_image(image: np.ndarray, test_name: str):
    """Save frame to disk for debugging."""
    debug_dir = Path("test_debug_outputs")
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = debug_dir / f"{test_name}_{timestamp}.png"
    cv2.imwrite(str(out_path), image)
    print(f"⚠️ Debug frame saved to {out_path}")

def visualize_mask_regions(
        original: np.ndarray,
        dot_mask: np.ndarray,
        box_mask: np.ndarray,
        conf_mask: np.ndarray,
        leaked_changes: np.ndarray,
        test_name: str
) -> None:
    """
    Create an image showing which regions are expected (dot, box, confidence)
    and highlight unexpected modifications (leaked pixels).
    """
    vis = np.zeros_like(original)

    # Set channels for different regions
    vis[dot_mask.astype(bool)] = [0, 255, 0]      # Green for dots
    vis[box_mask.astype(bool)] = [255, 0, 0]      # Blue for box
    vis[conf_mask.astype(bool)] = [0, 255, 255]   # Yellow for confidence
    vis[leaked_changes] = [0, 0, 255]             # Red for unexpected change

    out_dir = Path("test_debug_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"mask_debug_{test_name}_{timestamp}.png"
    cv2.imwrite(str(out_path), vis)
    print(f"🧩 Mask visualization saved to {out_path}")


class TestDrawFacesAndMouths(unittest.TestCase):
    IMG_SIZE = 500

    def setUp(self):
        self.frame = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
        self.boxes = [[100, 100, 400, 400]]
        self.landmarks = [[
            [150, 150],  # dummy left eye
            [350, 150],  # dummy right eye
            [250, 200],  # dummy nose
            [200, 300],  # mouth left
            [300, 300]   # mouth right
        ]]
        self.confidences = [0.95]

    def test_only_expected_pixels_changed(self):
        modified = self.frame.copy()
        face_count = draw_faces_and_mouths(modified, self.boxes, self.landmarks, self.confidences)

        try:
            self.assertEqual(face_count, 1)

            red_dot_radius = 3
            green_dot_radius = 5
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            pad = 2
            extra_pad = 4

            pts = self.landmarks[0]
            mouth_left = pts[3]
            mouth_right = pts[4]
            mouth_center = [
                int((mouth_left[0] + mouth_right[0]) / 2),
                int((mouth_left[1] + mouth_right[1]) / 2)
            ]
            mouth_pts = [(mouth_left, red_dot_radius),
                         (mouth_right, red_dot_radius),
                         (mouth_center, green_dot_radius)]

            dot_mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.uint8)
            for (x, y), r in mouth_pts:
                cv2.circle(dot_mask, (int(x), int(y)), r, 1, thickness=-1)

            box_mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.uint8)
            x1, y1, x2, y2 = self.boxes[0]
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 1, thickness=2)

            conf_mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.uint8)
            text = f"{self.confidences[0]:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            text_x1 = max(0, x1 - pad)
            text_y1 = max(0, y1 - text_height - baseline - pad - extra_pad)
            text_x2 = min(self.IMG_SIZE, x1 + text_width + pad)
            text_y2 = min(self.IMG_SIZE, y1 + pad)
            cv2.rectangle(conf_mask, (text_x1, text_y1), (text_x2, text_y2), 1, thickness=-1)

            safe_mask = ((dot_mask | box_mask | conf_mask) > 0)

            changed = np.any(modified[dot_mask.astype(bool)] != 0)
            self.assertTrue(changed, "Expected dot regions to be nonzero")

            diff_mask = np.any(modified != self.frame, axis=-1)
            leaked_changes = diff_mask & (~safe_mask.astype(bool))
            num_unexpected = np.count_nonzero(leaked_changes)

            if num_unexpected > 0:
                visualize_mask_regions(
                    original=self.frame,
                    dot_mask=dot_mask,
                    box_mask=box_mask,
                    conf_mask=conf_mask,
                    leaked_changes=leaked_changes,
                    test_name="only_expected_pixels_changed"
                )
                debug_vis = modified.copy()
                debug_vis[leaked_changes] = [0, 0, 255]
                save_debug_image(debug_vis, "unexpected_modified_pixels")

            self.assertEqual(num_unexpected, 0,
                             f"{num_unexpected} pixels were unexpectedly modified outside safe regions")

        except AssertionError:
            save_debug_image(modified, "test_draws_mouth_dots_and_confidence")
            raise

    def test_handles_empty_inputs(self):
        blank = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
        result = draw_faces_and_mouths(blank, [], [], [])
        try:
            self.assertEqual(result, 0)
            self.assertTrue(np.all(blank == 0), "Nothing should be drawn")
        except AssertionError:
            save_debug_image(blank, "test_handles_empty_inputs")
            raise

    def test_raises_on_insufficient_landmarks(self):
        frame = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
        boxes = [[100, 100, 400, 400]]
        bad_landmarks = [[[20, 50], [80, 50]]]
        confidences = [0.95]

        with self.assertRaises(IndexError):
            draw_faces_and_mouths(frame, boxes, bad_landmarks, confidences)


class TestDetectFacesInFrame(unittest.TestCase):
    def test_detect_faces_returns_expected_outputs(self):
        frame = np.zeros((640, 480, 3), dtype=np.uint8)

        fake_boxes = [[10, 10, 100, 100]]
        fake_landmarks = [[[15, 15], [90, 15], [50, 30], [30, 70], [70, 70]]]
        fake_confidences = [0.95]

        mock_model = MagicMock()
        mock_model.return_value = (fake_boxes, fake_landmarks, fake_confidences)
        detector = FaceDetector(mock_model)

        boxes, landmarks, confidences = detector.detect_faces_in_frame(frame)

        self.assertEqual(boxes, fake_boxes)
        self.assertEqual(landmarks, fake_landmarks)
        self.assertEqual(confidences, fake_confidences)

    def test_detect_faces_handles_none(self):
        frame = np.zeros((640, 480, 3), dtype=np.uint8)

        mock_model = MagicMock()
        mock_model.return_value = None
        detector = FaceDetector(mock_model)

        boxes, landmarks, confidences = detector.detect_faces_in_frame(frame)

        self.assertEqual(boxes, [])
        self.assertEqual(landmarks, [])
        self.assertEqual(confidences, [])