import cv2
import torch
from typing import Optional
from mouthtracker.detection.yolo5face_model import load_yolo5face_model
from mouthtracker.detection.detection_helpers import detect_faces_in_frame, draw_faces_and_mouths
from mouthtracker.output.audio_tools import restore_audio_from_source
from mouthtracker.tracking.face_tracker import FaceTracker, draw_tracked_face_box

def multiface_mouthtrack(
    input_path: str,
    output_path: str,
    model_path: str,
    config_path: str,
    show_periodic: bool = False,
    display_interval_sec: float = 0.5,
    require_gpu: bool = True,
    min_face: int = 10,
    track_interval: int = 30,
    tracker_type="CSRT"
) -> None:
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
    tracker = FaceTracker(tracker_type=tracker_type)
    prev_face_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        should_detect = not prev_face_count or (frame_num % track_interval == 0)
        do_fallback = False
        face_count = 0

        if not should_detect:
            tracked_boxes = tracker.update_trackers(frame)
            for box in tracked_boxes:
                if box is not None:
                    draw_tracked_face_box(frame, box, color_name="tracked")
                    face_count += 1
                else:
                    do_fallback = True
        
        if should_detect or do_fallback:
            result = detect_faces_in_frame(model, frame, target_size=640)
            if result is not None:
                boxes, landmarks, confidences = result
                if boxes:
                    face_count = draw_faces_and_mouths(frame, boxes, landmarks, confidences)
                    tracker.init_trackers(frame, [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes])
                    _ = tracker.update_trackers(frame)
                    for box in boxes:
                        draw_tracked_face_box(
                            frame,
                            (box[0], box[1], box[2] - box[0], box[3] - box[1]),
                            color_name="detected" if should_detect else "fallback"
                        )
                    max_faces = max(max_faces, face_count)
                else:
                    cv2.putText(frame, "No Faces Found", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No Faces Found", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        prev_face_count = face_count
        if face_count > 0:
            label = f"{'Faces detected' if should_detect else 'Faces tracked'}: {face_count}"
            
            cv2.putText(frame, label, (20, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if show_periodic and frame_num % display_interval == 0:
            print(f"Processing frame {frame_num}")
            try:
                from google.colab.patches import cv2_imshow
            except ImportError:
                import matplotlib.pyplot as plt
                def cv2_imshow(img):
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()
            cv2_imshow(frame)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    restore_audio_from_source(input_path, output_path)

    print(f"✅ Mouth tracking complete. Output saved to: {output_path}")
    print(f"📊 Max faces detected at once: {max_faces}")
