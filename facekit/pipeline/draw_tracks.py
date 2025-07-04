from typing import List, Callable, Optional
import cv2
from pathlib import Path
from facekit.tracking.face_tracks import FaceTrack

def draw_tracks_on_video(
    video_path: str,
    output_path: str,
    tracks: List[FaceTrack],
    label_fmt: Optional[Callable[[FaceTrack], str]] = None
) -> None:
    """
    Render a video with bounding boxes and labels for each tracked face.

    Args:
        video_path (str): Path to the original video file.
        output_path (str): Path to write the output annotated video.
        tracks (List[FaceTrack]): A list of face tracks to render.
        label_fmt (Callable[[FaceTrack], str], optional): Function to format track labels.
                                                           Defaults to using just track_id.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    overlay_map = {}
    for track in tracks:
        label = label_fmt(track) if label_fmt else f"{track.track_id}"
        for obs in track.observations:
            overlay_map.setdefault(obs.frame_idx, []).append((obs.bbox, label))

    for frame_idx in range(total_frames):
        
        ret, frame = cap.read()
        if not ret:
            break

        overlays = overlay_map.get(frame_idx, [])
        for bbox, label in overlays:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(label), (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
