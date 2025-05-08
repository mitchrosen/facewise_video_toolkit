import cv2
import torch
from typing import Optional
from mouthtracker.detection.yolo5face_model import load_yolo5face_model
from mouthtracker.detection.detection_helpers import detect_faces_in_frame, draw_faces_and_mouths
from mouthtracker.output.audio_tools import restore_audio_from_source
from mouthtracker.tracking.mouth_tracker import MouthTracker

try:
    from google.colab.patches import cv2_imshow
except ImportError:
    def cv2_imshow(img):
        import matplotlib.pyplot as plt
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

def multiface_mouthtrack(
    input_path: str,
    output_path: str,
    model_path: str,
    config_path: str,
    show_periodic: bool = False,
    display_interval_sec: float = 0.5,
    require_gpu: bool = True,
    min_face: int = 10,
    track_interval: int = 30
) -> None:
    """
    Tracks mouths in a video by alternating between face detection and mouth tracking.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the output video.
        model_path (str): Path to YOLOv5 face detector weights.
        config_path (str): Path to detector config YAML.
        show_periodic (bool): Whether to display frames during processing.
        display_interval_sec (float): Interval (sec) for periodic display.
        require_gpu (bool): If True, raise error if CUDA not available.
        min_face (int): Minimum face size in pixels.
        track_interval (int): Number of frames to skip between detections.
    """

    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("❌ GPU required but CUDA is not available.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_yolo5face_model(
        model_path=model_path,
        config_path=config_path,
        device=device,
        min_face=min_face
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_interval = max(int(fps * display_interval_sec), 1)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_num = 0
    max_faces = 0
    mouth_tracker = MouthTracker()
    last_detection_frame = -track_interval

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num - last_detection_frame >= track_interval:
            result = detect_faces_in_frame(model, frame, target_size=640)

            if result is not None:
                boxes, landmarks, confidences = result
                face_count = draw_faces_and_mouths(frame, boxes, landmarks, confidences)
                mouth_tracker.init_trackers(frame, landmarks)
                last_detection_frame = frame_num
                max_faces = max(max_faces, face_count)
            else:
                face_count = 0
                cv2.putText(frame, "No Faces Found", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            mouth_boxes = mouth_tracker.update_trackers(frame)
            face_count = 0
            for box in mouth_boxes:
                if box is not None:
                    x, y, w, h = map(int, box)
                    center = (x + w // 2, y + h // 2)
                    cv2.circle(frame, center, 4, (0, 255, 255), -1)
                    face_count += 1

        cv2.putText(frame, f"Faces tracked: {face_count}", (20, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if show_periodic and frame_num % display_interval == 0:
            print(f"Processing frame {frame_num}")
            cv2_imshow(frame)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    restore_audio_from_source(input_path, output_path)

    print(f"✅ Mouth tracking complete. Output saved to: {output_path}")
    print(f"📊 Max faces detected at once: {max_faces}")
