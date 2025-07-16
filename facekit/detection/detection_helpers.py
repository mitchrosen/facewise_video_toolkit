import cv2
import numpy as np
from typing import List, Tuple, Union
from numpy import ndarray
from typing import Optional, Tuple, List
from facekit.tracking.face_structures import FaceObservation
from facekit.detection.yolo5face_model import load_yolo5face_model

def detect_faces_in_frame(
    model: object,
    frame: ndarray,
    target_size: int = 640
) -> Optional[Tuple[List[List[int]], List[List[List[int]]], List[float]]]:
    """
    Run face detection on a single video frame using the specified YOLO-based model.

    Args:
        frame (np.ndarray): The image frame to process.
        model (Any): The face detection model object with a callable interface.
        target_size (int): Optional resizing dimension before detection (default is 640).

    Returns:
        Optional[Tuple]: A tuple containing:
            - List of bounding boxes (each box is [x1, y1, x2, y2])
            - List of facial landmarks (each face has a list of 5 [x, y] points)
            - List of confidence scores for each detected face
        Returns None if no faces are detected or an error occurs.
    """

    try:
        results = model(frame, target_size=target_size)
        if (
            isinstance(results, tuple) and
            len(results) == 3 and
            all(isinstance(r, list) for r in results)
        ):
            return results  # (boxes, landmarks, confidences)
        else:
            return None
    except Exception:
        return None

import cv2
import numpy as np
from typing import List

def draw_faces_and_mouths(
    frame: np.ndarray,
    boxes: List[List[float]],
    landmarks: List[List[List[int]]],
    confidences: List[float]
) -> int:
    """
    Draws bounding boxes and mouth landmarks on a frame for each detected face.

    Each face is expected to have exactly 5 landmarks in the following order:
        [left_eye, right_eye, nose, mouth_left, mouth_right]

    - A yellow bounding box is drawn around the face.
    - Red dots are drawn at mouth_left and mouth_right.
    - A green dot is drawn at the midpoint between mouth_left and mouth_right.
    - Confidence is shown as text near the bounding box.

    Args:
        frame (np.ndarray): The image frame to draw on (modified in place).
        boxes (List[List[int]]): List of bounding box coordinates per face.
        landmarks (List[List[List[int]]]): List of 5-point facial landmarks per face.
        confidences (List[float]): List of confidence scores per face.

    Returns:
        int: Number of faces successfully drawn.

    Raises:
        IndexError: If fewer than 5 landmarks are provided for a face.
    """
    height, width = frame.shape[:2]
    face_count = 0

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        conf = float(confidences[i])

        # Sanity check bounding box is within frame
        if not (0 <= x1 < width and 0 <= y1 < height and 0 <= x2 <= width and 0 <= y2 <= height):
            print(f"⚠️ Skipping face {i}: Box out of bounds")
            continue

        # Draw bounding box and confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if i >= len(landmarks):
            raise IndexError(f"Missing landmarks for face {i}")

        pts = landmarks[i]
        if len(pts) < 5:
            raise IndexError(f"Landmark set for face {i} must contain at least 5 points")

        mouth_left = tuple(map(int, pts[3]))
        mouth_right = tuple(map(int, pts[4]))

        # Clamp inside image just to be safe
        for pt in (mouth_left, mouth_right):
            if not (0 <= pt[0] < width and 0 <= pt[1] < height):
                print(f"⚠️ Skipping mouth landmark for face {i}: Out of bounds")
                continue

        mouth_center = (
            int((mouth_left[0] + mouth_right[0]) / 2),
            int((mouth_left[1] + mouth_right[1]) / 2)
        )

        cv2.circle(frame, mouth_left, 3, (0, 0, 255), -1)
        cv2.circle(frame, mouth_right, 3, (0, 0, 255), -1)
        cv2.circle(frame, mouth_center, 5, (0, 255, 0), -1)

        face_count += 1

    return face_count

device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
model_path = "models/yolov5n_state_dict.pt"
config_path = "models/yolov5n.yaml"
model = load_yolo5face_model(model_path=model_path, config_path=config_path, device=device)

def compute_expanded_bbox(bbox, all_boxes, img_w, img_h, margin=20):
    """
    Expand bbox as much as possible without overlapping other boxes or image boundaries.
    Optionally add a fixed margin if space allows.
    """
    x1, y1, x2, y2 = bbox

    # Distance to image boundaries
    left_limit = x1
    top_limit = y1
    right_limit = img_w - x2
    bottom_limit = img_h - y2

    # Distance to nearest neighbor in each direction
    for other in all_boxes:
        if other == bbox:
            continue
        ox1, oy1, ox2, oy2 = other

        # To the left: other box must be fully left
        if ox2 <= x1:
            left_limit = min(left_limit, x1 - ox2)
        # To the right: other box must be fully right
        if ox1 >= x2:
            right_limit = min(right_limit, ox1 - x2)
        # Above
        if oy2 <= y1:
            top_limit = min(top_limit, y1 - oy2)
        # Below
        if oy1 >= y2:
            bottom_limit = min(bottom_limit, oy1 - y2)

    # Now decide actual expansion amount (bounded by image and neighbors)
    expand_left = max(left_limit, margin)
    expand_top = max(top_limit, margin)
    expand_right = max(right_limit, margin)
    expand_bottom = max(bottom_limit, margin)

    return (
        max(0, x1 - expand_left),
        max(0, y1 - expand_top),
        min(img_w, x2 + expand_right),
        min(img_h, y2 + expand_bottom),
    )

def detect_faces_and_embeddings(frame, frame_idx: int, embedder=None) -> List[FaceObservation]:
    """
    Detect faces and return a list of FaceObservation objects.
    Expands bounding boxes for embedding context without overlapping other faces.

    Args:
        frame (np.ndarray): The current video frame (BGR).
        frame_idx (int): Index of the current frame.
        embedder (object): Face embedding extractor with a `get_embedding(frame, bbox)` method. 
            Used to generate embeddings for identity resolution across shots.

    Returns:
        List[FaceObservation]: Observations with bbox and (optional) embeddings.
    """
    observations = []

    result = detect_faces_in_frame(model, frame)
    if result is None:
        return observations

    boxes, landmarks, confidences = result
    height, width = frame.shape[:2]

    # Expand boxes for embedding context
    expanded_boxes = []
    for i, bbox in enumerate(boxes):
        expanded_box = compute_expanded_bbox(bbox, boxes, width, height, margin=20)
        expanded_boxes.append(expanded_box)

    for bbox, expanded_bbox, conf in zip(boxes, expanded_boxes, confidences):
        bbox = tuple(int(x) for x in bbox)
        expanded_bbox = tuple(int(x) for x in expanded_bbox)

        # Crop using expanded bbox
        face_crop = frame[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]

        embedding = None
        if embedder is not None:
            embedding = embedder.get_embedding(face_crop)

        obs = FaceObservation(
            frame_idx=frame_idx,
            bbox=bbox,  # Keep original for consistency
            embedding=embedding,
            confidence=conf
        )
        observations.append(obs)

    return observations
