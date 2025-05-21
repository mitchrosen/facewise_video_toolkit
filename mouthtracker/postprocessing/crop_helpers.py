import cv2
import numpy as np
from math import atan2, degrees, radians, cos, sin
from typing import List, Dict, Tuple


def crop_for_single_face(
    face: Dict[str, List[int]],
    frame: np.ndarray,
    aspect_ratio: float,
    output_size: Tuple[int, int] = (720, 1560)
) -> np.ndarray:
    """
    Perform standard portrait crop centered on a single face.

    Uses full height of the input frame and centers horizontally around the face.

    Parameters:
        face (dict): A dictionary with key "bbox" = [x1, y1, x2, y2].
        frame (np.ndarray): The input video frame (H x W x 3).
        aspect_ratio (float): Target portrait aspect ratio (width / height).
        output_size (tuple): Final output size as (width, height).

    Returns:
        np.ndarray: Cropped and resized portrait frame.
    """
  
    frame_h, frame_w = frame.shape[:2]
    crop_h = frame_h
    crop_w = int(crop_h / aspect_ratio)

    x1, _, x2, _ = face["bbox"]
    face_cx = (x1 + x2) // 2

    crop_x1 = face_cx - crop_w // 2
    crop_x2 = crop_x1 + crop_w

    if crop_x1 < 0:
        crop_x1 = 0
        crop_x2 = crop_w
    elif crop_x2 > frame_w:
        crop_x2 = frame_w
        crop_x1 = crop_x2 - crop_w

    crop_x1 = max(0, crop_x1)
    crop_x2 = min(frame_w, crop_x2)

    cropped = frame[:, crop_x1:crop_x2]
    return cv2.resize(cropped, output_size)


def crop_for_two_faces(
    faces: List[Dict[str, List[int]]],
    frame: np.ndarray,
    aspect_ratio: float,
    output_size: Tuple[int, int] = (720, 1560),
    max_angle: float = 30,
    max_scale: float = 1.5
) -> np.ndarray:
    """
    Crop a portrait-oriented region from a frame that includes two faces.

    Attempts an upright crop first. If the faces can't fit within a portrait
    crop width (even after up to 10% width shrink), falls back to a rotated
    and scaled crop that realigns the portrait box to fit both faces.

    Returns:
        np.ndarray: Cropped and resized portrait frame.
    """
    if upright_crop_possible(faces, frame.shape[0], aspect_ratio, shrink_tolerance=0.9):
        return upright_crop_for_two_faces(faces, frame, aspect_ratio, output_size)
    else:
        return angular_crop_for_two_faces(faces, frame, aspect_ratio, output_size, max_angle, max_scale)


def upright_crop_possible(
    faces: List[Dict[str, List[int]]],
    frame_h: int,
    aspect_ratio: float,
    shrink_tolerance: float = 0.9
) -> bool:
    """
    Checks if an upright portrait crop can contain both faces
    with up to `shrink_tolerance` shrink in width.

    Returns:
        bool: True if upright crop is sufficient, else False.
    """
    x1a, _, x2a, _ = faces[0]["bbox"]
    x1b, _, x2b, _ = faces[1]["bbox"]
    required_width = max(x2a, x2b) - min(x1a, x1b)
    full_crop_width = frame_h/aspect_ratio
    shrink_factor = required_width / full_crop_width
    return shrink_factor <= 1.0 and shrink_factor >= shrink_tolerance


def upright_crop_for_two_faces(
    faces: List[Dict[str, List[int]]],
    frame: np.ndarray,
    aspect_ratio: float,
    output_size: Tuple[int, int]
) -> np.ndarray:
    """
    Crops upright portrait window centered between two faces.

    Uses full frame height and standard portrait aspect ratio to
    determine crop width, centered horizontally between the two faces.

    Returns:
        np.ndarray: Cropped and resized portrait frame.
    """
    x1a, _, x2a, _ = faces[0]["bbox"]
    x1b, _, x2b, _ = faces[1]["bbox"]
    cx1 = (x1a + x2a) / 2
    cx2 = (x1b + x2b) / 2
    center_x = (cx1 + cx2) / 2

    frame_h, frame_w = frame.shape[:2]
    crop_h = frame_h
    crop_w = int(crop_h / aspect_ratio)

    crop_x1 = int(center_x - crop_w / 2)
    crop_x2 = crop_x1 + crop_w

    # Clamp to frame boundaries
    if crop_x1 < 0:
        crop_x1 = 0
        crop_x2 = crop_w
    elif crop_x2 > frame_w:
        crop_x2 = frame_w
        crop_x1 = frame_w - crop_w

    cropped = frame[:, crop_x1:crop_x2]
    return cv2.resize(cropped, output_size)

def angular_crop_for_two_faces(
    faces: List[Dict[str, List[int]]],
    frame: np.ndarray,
    aspect_ratio: float,
    output_size: Tuple[int, int],
    max_angle: float,
    max_scale: float
) -> np.ndarray:
    """
    Rotates and scales a portrait crop box to include two faces aligned diagonally,
    then re-aligns upright and extracts a consistent-aspect crop that includes both faces.
    """
    frame_h, frame_w = frame.shape[:2]

    # Face centers
    (x1a, y1a, x2a, y2a) = faces[0]["bbox"]
    (x1b, y1b, x2b, y2b) = faces[1]["bbox"]
    cx1, cy1 = (x1a + x2a) / 2, (y1a + y2a) / 2
    cx2, cy2 = (x1b + x2b) / 2, (y1b + y2b) / 2
    mid_x, mid_y = (cx1 + cx2) / 2, (cy1 + cy2) / 2

    # Angle between faces
    dx = cx2 - cx1
    dy = cy2 - cy1
    angle = degrees(atan2(dy, dx))
    rotation_angle = max(-max_angle, min(max_angle, angle - 90))

    # Fixed portrait box
    base_crop_h = frame_h
    base_crop_w = base_crop_h / aspect_ratio

    # Inverse rotate face corners to measure bounding box
    theta = radians(rotation_angle)
    inv_rot = np.array([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]])
    all_rotated_corners = []

    for face in faces:
        fx1, fy1, fx2, fy2 = face["bbox"]
        corners = np.array([
            [fx1, fy1], [fx2, fy1], [fx2, fy2], [fx1, fy2]
        ])
        rotated = np.dot(corners - np.array([mid_x, mid_y]), inv_rot.T)
        all_rotated_corners.append(rotated)

    all_rotated = np.vstack(all_rotated_corners)
    required_w = np.ptp(all_rotated[:, 0])
    required_h = np.ptp(all_rotated[:, 1])
    required_scale = max(required_w / base_crop_w, required_h / base_crop_h)
    scale = 1.0 / required_scale

    # Apply scaled warp
    rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y), -rotation_angle, scale)
    warped = cv2.warpAffine(frame, rot_mat, (frame_w, frame_h), flags=cv2.INTER_LINEAR)

    # Recompute center after warp
    rotated_center = np.dot(rot_mat[:, :2], np.array([mid_x, mid_y])) + rot_mat[:, 2]
    rot_mid_x, rot_mid_y = rotated_center

    # Use base crop dimensions (scale already applied in warp)
    crop_w = int(base_crop_w)
    crop_h = int(base_crop_h)

    crop_x1 = int(rot_mid_x - crop_w / 2)
    crop_y1 = int(rot_mid_y - crop_h / 2)

    # Clamp to image bounds
    crop_x1 = max(0, min(frame_w - crop_w, crop_x1))
    crop_y1 = max(0, min(frame_h - crop_h, crop_y1))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    final_crop = warped[crop_y1:crop_y2, crop_x1:crop_x2]
    return cv2.resize(final_crop, output_size)

def crop_for_three_faces(
    faces: List[Dict[str, List[int]]],
    frame: np.ndarray,
    aspect_ratio: float,
    output_size: Tuple[int, int] = (720, 1560),
    max_angle: float = 30
) -> np.ndarray:
    """
    Crop a portrait-oriented region that includes three faces.
    Rotates and scales as needed to align the faces vertically
    and ensure all are included in the final crop.

    Returns:
        np.ndarray: Cropped and resized portrait frame.
    """
    frame_h, frame_w = frame.shape[:2]

    # Get center of each face
    centers = []
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

    # Compute average center and bounding box
    cx_all = [c[0] for c in centers]
    cy_all = [c[1] for c in centers]
    mid_x = sum(cx_all) / 3
    mid_y = sum(cy_all) / 3

    # Compute principal direction via linear regression on centers
    dx = max(cx_all) - min(cx_all)
    dy = max(cy_all) - min(cy_all)
    angle = degrees(atan2(dy, dx))
    rotation_angle = max(-max_angle, min(max_angle, angle - 90))

    # Base crop box
    base_crop_h = frame_h
    base_crop_w = base_crop_h / aspect_ratio

    # Rotate face boxes to upright space
    theta = radians(rotation_angle)
    inv_rot = np.array([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]])
    all_rotated_corners = []

    for face in faces:
        fx1, fy1, fx2, fy2 = face["bbox"]
        corners = np.array([
            [fx1, fy1], [fx2, fy1], [fx2, fy2], [fx1, fy2]
        ])
        rotated = np.dot(corners - np.array([mid_x, mid_y]), inv_rot.T)
        all_rotated_corners.append(rotated)

    all_rotated = np.vstack(all_rotated_corners)
    required_w = np.ptp(all_rotated[:, 0])
    required_h = np.ptp(all_rotated[:, 1])
    scale = max(required_w / base_crop_w, required_h / base_crop_h)

    # Warp frame
    rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y), -rotation_angle, 1.0 / scale)
    warped = cv2.warpAffine(frame, rot_mat, (frame_w, frame_h), flags=cv2.INTER_LINEAR)

    # New center after rotation
    rotated_center = np.dot(rot_mat[:, :2], np.array([mid_x, mid_y])) + rot_mat[:, 2]
    rot_mid_x, rot_mid_y = rotated_center

    # Use original base dimensions (since scaling already applied)
    crop_w = int(base_crop_w)
    crop_h = int(base_crop_h)

    crop_x1 = int(rot_mid_x - crop_w / 2)
    crop_y1 = int(rot_mid_y - crop_h / 2)

    # Clamp
    crop_x1 = max(0, min(frame_w - crop_w, crop_x1))
    crop_y1 = max(0, min(frame_h - crop_h, crop_y1))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    final_crop = warped[crop_y1:crop_y2, crop_x1:crop_x2]
    return cv2.resize(final_crop, output_size)

def letterbox_frame(frame, aspect_ratio, output_size):
    """
    Resize the frame to fit within the target width and height
    while preserving aspect ratio. Adds black padding as needed.

    Parameters:
        frame (np.ndarray): Input image.
        width (int): Target output width.
        height (int): Target output height.
        aspect_ratio (float): Not used here but kept for API consistency.

    Returns:
        np.ndarray: Letterboxed (padded) image of shape (height, width, 3)
    """
    width = output_size[0]
    height = output_size[1]
    frame_h, frame_w = frame.shape[:2]
    scale = min(width / frame_w, height / frame_h)
    resized_w = int(frame_w * scale)
    resized_h = int(frame_h * scale)

    resized = cv2.resize(frame, (resized_w, resized_h))

    # Create black background
    result = np.zeros((height, width, 3), dtype=np.uint8)

    # Center the resized image
    x_offset = (width - resized_w) // 2
    y_offset = (height - resized_h) // 2
    result[y_offset:y_offset+resized_h, x_offset:x_offset+resized_w] = resized

    return result