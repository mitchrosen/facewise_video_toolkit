import cv2
import numpy as np
from math import atan2, degrees, radians, cos, sin


def crop_for_single_face(face, frame, aspect_ratio, output_size=(720, 1560)):
    """
    Perform standard portrait crop centered on a single face.

    Uses full height of the input frame and centers horizontally around the face.

    Returns:
        np.ndarray: Cropped and resized frame
    """
    frame_h, frame_w = frame.shape[:2]

    # Full frame height, portrait width
    crop_h = frame_h
    crop_w = int(aspect_ratio * crop_h)

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
    resized = cv2.resize(cropped, output_size)
    return resized


def crop_for_two_faces(faces, frame, aspect_ratio, output_size=(720, 1560), max_angle=30, max_scale=1.5):
    """
    Crop a portrait-oriented region from a frame that includes two faces,
    optionally rotating and scaling the crop window to ensure both faces fit.

    The method computes the angle between the centers of the two faces. If the angle
    deviates from vertical by less than or equal to `max_angle` degrees, a portrait crop 
    window is rotated by that angle and scaled minimally (up to `max_scale`) to include both 
    face bounding boxes. The cropped region is then re-rotated to upright portrait orientation.

    If the angle exceeds `max_angle`, the rotation is clamped to ±`max_angle`.

    Parameters:
        faces (list of dict): A list of two face dicts, each with key "bbox" = [x1, y1, x2, y2].
        frame (np.ndarray): The input video frame (H x W x 3).
        aspect_ratio (float): Desired portrait aspect ratio (e.g., 2.17 for 720x1560).
        output_size (tuple): Final output dimensions (width, height), default (720, 1560).
        max_angle (float): Maximum allowed rotation in degrees from vertical (default 30).
        max_scale (float): Maximum allowed scale factor for the crop box (default 1.5).
        debug (bool): If True, adds overlays to the input frame for debugging (default False).

    Returns:
        np.ndarray: A portrait-oriented cropped and resized frame containing both faces.
    """    
    frame_h, frame_w = frame.shape[:2]
    
    # Step 1: Face centers
    (x1a, y1a, x2a, y2a) = faces[0]["bbox"]
    (x1b, y1b, x2b, y2b) = faces[1]["bbox"]

    cx1 = (x1a + x2a) / 2
    cy1 = (y1a + y2a) / 2
    cx2 = (x1b + x2b) / 2
    cy2 = (y1b + y2b) / 2

    mid_x = (cx1 + cx2) / 2
    mid_y = (cy1 + cy2) / 2

    # Step 2: Rotation angle
    dx = cx2 - cx1
    dy = cy2 - cy1
    angle = degrees(atan2(dy, dx))
    rotation_angle = angle - 90  # deviation from vertical

    # Step 3: Clamp to ±max_angle
    rotation_angle = max(-max_angle, min(max_angle, rotation_angle))

    # Step 4: Initial portrait crop box
    crop_h = frame_h
    crop_w = int(aspect_ratio * crop_h)
    scale = 1.0

    # Step 5: Iterate scale until both faces fit inside rotated box
    while scale <= max_scale:
        test_w = crop_w * scale
        test_h = crop_h * scale

        # Get 4 corners of crop box (centered at mid_x, mid_y)
        box = np.array([
            [-test_w/2, -test_h/2],
            [ test_w/2, -test_h/2],
            [ test_w/2,  test_h/2],
            [-test_w/2,  test_h/2]
        ])

        # Rotation matrix
        theta = radians(rotation_angle)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        rotated_box = np.dot(box, rot.T) + np.array([mid_x, mid_y])

        # Convert rotated box to bounding rect
        x_min = np.min(rotated_box[:,0])
        x_max = np.max(rotated_box[:,0])
        y_min = np.min(rotated_box[:,1])
        y_max = np.max(rotated_box[:,1])

        # Check if all face corners are inside this rotated region
        def face_inside(face):
            fx1, fy1, fx2, fy2 = face["bbox"]
            corners = np.array([
                [fx1, fy1],
                [fx2, fy1],
                [fx2, fy2],
                [fx1, fy2]
            ])
            # Inverse rotate corners around mid
            inv_rot = np.array([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]])
            aligned = np.dot(corners - np.array([mid_x, mid_y]), inv_rot.T)
            half_w, half_h = test_w / 2, test_h / 2
            return np.all(
                (aligned[:,0] >= -half_w) & (aligned[:,0] <= half_w) &
                (aligned[:,1] >= -half_h) & (aligned[:,1] <= half_h)
            )

        if face_inside(faces[0]) and face_inside(faces[1]):
            break

        scale += 0.05

    # Step 6: Crop & rotate the region
    # Compute affine warp to rotate the crop upright
    src_center = (mid_x, mid_y)
    rot_mat = cv2.getRotationMatrix2D(src_center, rotation_angle, scale)

    warped = cv2.warpAffine(frame, rot_mat, (frame_w, frame_h), flags=cv2.INTER_LINEAR)

    # Step 7: Extract axis-aligned upright portrait box from rotated frame
    crop_x1 = int(mid_x - (crop_w / 2))
    crop_y1 = int(mid_y - (crop_h / 2))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    # Clamp to image bounds
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(frame_w, crop_x2)
    crop_y2 = min(frame_h, crop_y2)

    final_crop = warped[crop_y1:crop_y2, crop_x1:crop_x2]
    final_resized = cv2.resize(final_crop, output_size)

    return final_resized

def crop_for_three_faces(faces, width, height, aspect_ratio):
    # Stub: return crop based on middle face
    return (0, 0, width, height)

def letterbox_frame(frame, width, height, aspect_ratio):
    # Stub: return full frame
    return (0, 0, width, height)