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

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        boxA (tuple): (x1, y1, x2, y2) for the first box.
        boxB (tuple): (x1, y1, x2, y2) for the second box.

    Returns:
        float: IoU value between 0.0 and 1.0
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area