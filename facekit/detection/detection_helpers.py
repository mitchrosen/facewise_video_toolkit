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

# ready for plug in of an actual face embedding model
embedding_model = None  # e.g., InsightFace for example
device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
model_path = "models/yolov5n_state_dict.pt"
config_path = "models/yolov5n.yaml"
model = load_yolo5face_model(model_path=model_path, config_path=config_path, device=device)

def detect_faces_and_embeddings(frame, frame_idx: int) -> List[FaceObservation]:
    """
    Detect faces and return a list of FaceObservation objects.

    Args:
        frame (np.ndarray): The current video frame (BGR).
        frame_idx (int): Index of the current frame.

    Returns:
        List[FaceObservation]: Observations with bbox and (optional) embeddings.
    """
    observations = []

    result = detect_faces_in_frame(model, frame)
    if result is None:
        return observations

    boxes, landmarks, confidences = result

    for bbox, conf in zip(boxes, confidences):
        bbox = tuple(int(x) for x in bbox)
        face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        embedding = None
        if embedding_model is not None:
            embedding = embedding_model.get_embedding(face_crop)

        obs = FaceObservation(
            frame_idx=frame_idx,
            bbox=bbox,
            embedding=embedding,
            confidence=conf
        )
        observations.append(obs)

    return observations