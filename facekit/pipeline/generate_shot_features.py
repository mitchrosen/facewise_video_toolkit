import json
import cv2
from pathlib import Path
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.detection.detection_helpers import detect_faces_in_frame
from facekit.utils.geometry import normalize_face_bbox
from facekit.postprocessing.validate_shot_features_json import validate_shot_features_json

def detect_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    return scene_manager.get_scene_list()

def get_frame_at(video_capture, frame_num):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = video_capture.read()
    if not success:
        raise RuntimeError(f"Failed to read frame {frame_num}")
    return frame

def extract_faces(frame, model, frame_w, frame_h):
    boxes, _, _ = detect_faces_in_frame(model, frame, target_size=640)
    return [normalize_face_bbox((x1, y1, x2, y2), frame_w, frame_h) for x1, y1, x2, y2 in boxes]

def generate_shot_features_json(video_path: str, output_json_path: str,
                                 model_path: str = "facekit/yolo5faceInference/yolo5face/yolov5s-face.pt",
                                 config_path: str = "facekit/yolo5faceInference/yolo5face/yolov5n.yaml"):
    video_path = Path(video_path)
    output_path = Path(output_json_path)
    scenes = detect_scenes(video_path)

    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    model = load_yolo5face_model(model_path=model_path, config_path=config_path, device=device)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    shots = []
    for idx, (start_time, end_time) in enumerate(scenes, start=1):
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()
        mid_frame = (start_frame + end_frame) // 2

        try:
            frame = get_frame_at(cap, mid_frame)
            face_boxes = extract_faces(frame, model, frame_w, frame_h)
        except Exception as e:
            print(f"⚠️  Could not extract faces for shot {idx}: {e}")
            face_boxes = []

        shots.append({
            "shot_number": idx,
            "first_frame": start_frame,
            "last_frame": end_frame,
            "detected_faces": {
                "face_count": len(face_boxes),
                "face_details": face_boxes
            },
            "detected_graphics": {}
        })

    cap.release()

    result = {"shots": shots}
    output_path.write_text(json.dumps(result, indent=2))

    errors = validate_shot_features_json(str(output_path), "schemas/shot_features.schema.json", total_frames)
    if errors:
        print("❌ Validation errors:")
        for e in errors:
            print(" -", e)
    else:
        print(f"✅ JSON valid. Saved to {output_path}")

