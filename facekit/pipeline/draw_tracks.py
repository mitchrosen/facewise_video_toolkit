from typing import List, Callable, Optional
import cv2
import time
from pathlib import Path
from facekit.tracking.face_structures import FaceTrack

def draw_tracks_on_video(
    video_path: str,
    output_path: str,
    tracks: List[FaceTrack],
    label_fmt: Optional[Callable[[FaceTrack, int], str]] = None
) -> None:
    """
    Render a video with bounding boxes and labels for each tracked face.
    Uses MJPEG for faster debug video writing and provides timing stats.

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

    # ✅ Use MJPEG for speed and .avi extension
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if output_path.suffix.lower() != ".avi":
        output_path = output_path.with_suffix(".avi")

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Build overlay map
    overlay_map = {}
    for track in tracks:
        for obs in track.observations:
            if label_fmt:
                # Prefer 2-arg labelers (track, frame_idx); fall back to legacy 1-arg
                try:
                    label = label_fmt(track, obs.frame_idx)
                except TypeError:
                    label = label_fmt(track)
            else:
                label = f"{track.track_id}"
            overlay_map.setdefault(obs.frame_idx, []).append((obs.bbox, label))

    print(f"[DEBUG] total_frames (video) = {total_frames}")

    if overlay_map:
        min_overlay_frame = min(overlay_map.keys())
        max_overlay_frame = max(overlay_map.keys())
        print(f"[DEBUG] overlay_map frames: min={min_overlay_frame}, max={max_overlay_frame}, count={len(overlay_map)}")
        # Frames that have overlays but are beyond the video length (they will never be drawn)
        extra_overlay_frames = sorted(k for k in overlay_map.keys() if k >= total_frames)[:10]
        if extra_overlay_frames:
            print(f"[WARN] {len(extra_overlay_frames)} overlay frames >= total_frames (showing up to 10): {extra_overlay_frames}")
        # Quick look at last 10 video frames and whether overlays exist
        tail_probe_start = max(0, total_frames - 10)
        tail_probe = [(f, len(overlay_map.get(f, []))) for f in range(tail_probe_start, total_frames)]
        print(f"[DEBUG] overlays per last 10 video frames: {tail_probe}")
    else:
        max_overlay_frame = 0
        print("[DEBUG] overlay_map is EMPTY (no observations to draw)")

    end_frame = max(total_frames, max_overlay_frame + 1)

    # ✅ Timing variables
    total_read, total_draw, total_write = 0.0, 0.0, 0.0

    for frame_idx in range(end_frame):
        # Read frame
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] cap.read() returned False at frame_idx={frame_idx}. "
                f"end_frame={end_frame}, total_frames={total_frames}. Stopping read.")
            break
        t1 = time.time()
        total_read += (t1 - t0)

        # Draw overlays
        t0 = time.time()
        overlays = overlay_map.get(frame_idx, [])

        # --- DEBUG: tail frames and per-frame overlay details ---
        if frame_idx >= total_frames - 5:  # focus on the last 5 video frames
            labels_here = [lbl for _, lbl in overlays]
            print(f"[DEBUG] f={frame_idx}/{total_frames-1} "
                  f"overlays={len(overlays)} labels={labels_here} ")
            # If present, show one bbox sample to verify coordinates look sane
            if overlays:
                (x1, y1, x2, y2), _ = overlays[0]
                print(f"[DEBUG] sample bbox on f={frame_idx}: ({x1},{y1},{x2},{y2})")
        
        for bbox, label in overlays:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(label), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        t1 = time.time()
        total_draw += (t1 - t0)

        # Write frame
        t0 = time.time()
        out.write(frame)
        t1 = time.time()
        total_write += (t1 - t0)

    cap.release()
    out.release()

    # --- DEBUG: loop end summary ---
    print(f"[DEBUG] Finished draw loop at frame_idx={frame_idx if 'frame_idx' in locals() else 'N/A'}")

    print(f"\n✅ Video with overlays written to {output_path}")
    print(f"[TIMING SUMMARY]")
    print(f"Frame Reading    : {total_read:.2f}s")
    print(f"Overlay Drawing  : {total_draw:.2f}s")
    print(f"Frame Writing    : {total_write:.2f}s")
