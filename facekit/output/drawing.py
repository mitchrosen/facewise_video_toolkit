import cv2

def draw_bbox_with_track_id(frame, bbox, track_id):
    """
    Draw a rectangle and label it with the track ID.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"ID {track_id}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )
