# mouthtracker/postprocessing/crop_portrait_video.py

import cv2
import json

def crop_video_from_json(json_path, video_path, output_path, aspect_ratio=2.17):
    with open(json_path, 'r') as f:
        tracking_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx, frame_data in enumerate(tracking_data["frames"]):
        ret, frame = cap.read()
        if not ret:
            break

        faces = frame_data.get("faces", [])

        match len(faces):
            case 1:
                crop_box = crop_for_single_face(faces[0], width, height, aspect_ratio)
            case 2:
                crop_box = crop_for_two_faces(faces, width, height, aspect_ratio)
            case 3:
                crop_box = crop_for_three_faces(faces, width, height, aspect_ratio)
            case _:
                crop_box = letterbox_frame(frame, width, height, aspect_ratio)

        cropped = apply_crop_and_resize(frame, crop_box, aspect_ratio)
        out.write(cropped)

    cap.release()
    out.release()

def crop_for_single_face(face, width, height, aspect_ratio):
    # Stub: use center of face box and expand to aspect ratio
    return (0, 0, width, height)

def crop_for_two_faces(faces, width, height, aspect_ratio):
    # Stub: return crop centered between faces
    return (0, 0, width, height)

def crop_for_three_faces(faces, width, height, aspect_ratio):
    # Stub: return crop based on middle face
    return (0, 0, width, height)

def letterbox_frame(frame, width, height, aspect_ratio):
    # Stub: return full frame
    return (0, 0, width, height)

def apply_crop_and_resize(frame, crop_box, aspect_ratio):
    x1, y1, x2, y2 = crop_box
    cropped = frame[y1:y2, x1:x2]
    # Resize or pad to match target aspect_ratio
    return cv2.resize(cropped, (frame.shape[1], frame.shape[0]))  # Placeholder
