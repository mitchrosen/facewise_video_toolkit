import cv2
import os
from typing import Optional
from facekit.detection.detection_helpers import detect_faces_and_embeddings
from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.output.drawing import draw_bbox_with_track_id
from facekit.config import DEFAULT_OUTPUT_DIR
# temporarily add vchunk_id export
from facekit.output.json_writer import export_vchunk_id_map

def run_tracking_on_video(
    input_path: str,
    output_path: Optional[str] = None,
    draw: bool = True,
    max_frames: Optional[int] = None
):
    """
    Run face detection and tracking on a video and optionally save output.

    Args:
        input_path (str): Path to the input video file.
        output_path (str, optional): Where to save the output video with boxes.
        draw (bool): Whether to overlay bounding boxes on the output.
        max_frames (int, optional): Limit the number of frames to process.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        basename = os.path.basename(input_path)
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"tracked_{basename}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) if draw else None

    aggregator = ShotFaceTrackAggregator(shot_number=0)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames is not None and frame_idx >= max_frames):
            break

        observations = detect_faces_and_embeddings(frame, frame_idx)
        aggregator.update_tracks_with_frame(frame_idx, observations)

        if draw:
            for track in aggregator.get_tracks_in_frame(frame_idx):
                obs = track.get_bbox_by_frame(frame_idx)
                if obs:
                    draw_bbox_with_track_id(frame, obs, track.track_id)
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    print(f"Tracking complete. Output saved to: {output_path}")

    #Temporarily add logic to resolve vchunk IDs
    # return aggregator.finalize_tracks()

    tracks = aggregator.finalize_tracks()

    # TEMP: assign vchunk IDs and export map
    aggregator.resolve_vchunk_ids(
        vchunk_idcounter=0,
        embedding_threshold=0.7
    )
    export_vchunk_id_map(tracks, output_path.replace(".mp4", "_vchunk_ids.json"))
    print(f"vchunk ID map written to: {output_path.replace('.mp4', '_vchunk_ids.json')}")

    return tracks
