import cv2
import torch
from typing import Optional
from mouthtracker.detection.yolo5face_model import load_yolo5face_model
from mouthtracker.detection.detection_helpers import detect_faces_in_frame, draw_faces_and_mouths
from mouthtracker.output.audio_tools import restore_audio_from_source
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    def cv2_imshow(img):
        import cv2
        from matplotlib import pyplot as plt
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

def level3_multiface_yolo5face_mouthtrack_v2(
    input_path: str,
    output_path: str,
    model_path: str,
    config_path: str,
    show_periodic: bool = False,
    display_interval_sec: float = 0.5,
    require_gpu: bool = True,
    min_face: int = 10
) -> None:
    """
    Performs frame-by-frame face detection and visual mouth tracking using a YOLOv5-based model.

    For each frame of the input video, this function detects all faces and overlays:
    - A green dot at the average center of the mouth
    - A red dot at the left-most mouth corner
    - A red dot at the right-most mouth corner
    - Confidence scores above each face (if available)

    Optionally displays frames periodically during processing and restores the original audio afterward.

    Args:
        input_path (str): Path to the source video file.
        output_path (str): Path to save the output video with drawn indicators.
        model_path (str): Path to the .pt file containing YOLOv5 face detection weights.
        config_path (str): Path to the YOLOv5 model configuration YAML file.
        show_periodic (bool): Whether to show frames periodically during processing. Default is False.
        display_interval_sec (float): Interval (in seconds) between displayed frames if `show_periodic` is True.
        require_gpu (bool): Whether to require a CUDA-enabled GPU. Raises an error if set and unavailable.
        min_face (int): Minimum pixel size of a face to be considered valid.

    Returns:
        None
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_faces_in_frame(model, frame, target_size=640)

        if result is not None:
            boxes, landmarks, confidences = result
            face_count = draw_faces_and_mouths(frame, boxes, landmarks, confidences)
            max_faces = max(max_faces, face_count)
        else:
            face_count = 0
            cv2.putText(frame, "No Faces Found", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Faces detected: {face_count}", (20, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if show_periodic and frame_num % display_interval == 0:
            print(f"Processing frame {frame_num}")
            cv2_imshow(frame)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()

    restore_audio_from_source(input_path, output_path)

    print(f"✅ Detection complete. Output saved to: {output_path}")
    print(f"📊 Max faces in any frame: {max_faces}")