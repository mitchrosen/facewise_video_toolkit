def normalize_face_bbox(bbox, frame_w, frame_h):
    """
    Normalize a bounding box from pixel coordinates to percent [0-100] values.

    Args:
        bbox (tuple): (x1, y1, x2, y2) in pixels.
        frame_w (int): Frame width in pixels.
        frame_h (int): Frame height in pixels.

    Returns:
        dict: Normalized face details.
    """
    x1, y1, x2, y2 = bbox
    center_x = 100 * (x1 + x2) / 2 / frame_w
    center_y = 100 * (y1 + y2) / 2 / frame_h
    face_width = 100 * (x2 - x1) / frame_w
    face_height = 100 * (y2 - y1) / frame_h
    return {
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2),
        "face_width": round(face_width, 2),
        "face_height": round(face_height, 2)
    }