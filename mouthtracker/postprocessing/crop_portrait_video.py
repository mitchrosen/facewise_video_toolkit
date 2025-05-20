import cv2
import json
from mouthtracker.postprocessing.crop_helpers import (
    crop_for_single_face,
    crop_for_two_faces,
    crop_for_three_faces,
    letterbox_frame
)

def crop_video_from_json(json_path, video_path, output_path, aspect_ratio=2.17, output_size=(720, 1560)):
    """
    Apply portrait-mode cropping to a video based on face tracking data in a JSON file.

    Parameters:
        json_path (str): Path to the JSON file with face bounding boxes per frame.
        video_path (str): Path to the input video file.
        output_path (str): Path to save the cropped output video.
        aspect_ratio (float): Target portrait aspect ratio (e.g., 2.17).
        output_size (tuple): Final output frame size (width, height).
    """
    with open(json_path, 'r') as f:
        tracking_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    for frame_idx, frame_data in enumerate(tracking_data["frames"]):
        ret, frame = cap.read()
        if not ret:
            break

        faces = frame_data.get("faces", [])
        face_count = len(faces)

        if face_count == 1:
            cropped = crop_for_single_face(faces[0], frame, aspect_ratio, output_size)
        elif face_count == 2:
            cropped = crop_for_two_faces(faces, frame, aspect_ratio, output_size)
        elif face_count == 3:
            cropped = crop_for_three_faces(faces, frame, aspect_ratio, output_size)
        else:
            cropped = letterbox_frame(frame, aspect_ratio, output_size)

        out.write(cropped)

    cap.release()
    out.release()
    
