# facekit/pipeline/generate_shot_features.py

import json
import cv2
from pathlib import Path
from typing import Optional
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode

from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.detection.face_detector import FaceDetector
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

def extract_faces(frame, detector: FaceDetector, frame_w, frame_h):
    result = detector.detect_faces_in_frame(frame, target_size=640)
    if result is None:
        return []
    boxes, _, _ = result
    return [normalize_face_bbox((x1, y1, x2, y2), frame_w, frame_h) for x1, y1, x2, y2 in boxes]

def generate_shot_features_json(video_path: str, output_json_path: str,
                                 detector_model_path: str = "models/detector/yolov5n_state_dict.pt",
                                 config_path: str = "models/detector/yolov5n.yaml",
                                 threshold: float = 30.0):
    import time
    start_time = time.time()
    video_path = Path(video_path)
    output_path = Path(output_json_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    elapsed = time.time() - start_time
    print(f"⏱️ setup time: {elapsed:.2f} seconds")
   
    start_time = time.time()
    scenes = detect_scenes(video_path, threshold)
    elapsed = time.time() - start_time
    print(f"⏱️ detect_scenes time: {elapsed:.2f} seconds")

    if not scenes:
        fps = cap.get(cv2.CAP_PROP_FPS)
        scenes = [(FrameTimecode(0, fps), FrameTimecode(total_frames - 1, fps))]

    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    detector_model = load_yolo5face_model(detector_model_path=detector_model_path, config_path=config_path, device=device)
    detector = FaceDetector(detector_model)

    start_time = time.time()
    shots = []
    for idx, (scene_start, scene_end) in enumerate(scenes, start=1):
        start_frame_num = scene_start.get_frames()
        end_frame_num = scene_end.get_frames() - 1
        mid_frame_num = (start_frame_num + end_frame_num) // 2

        frame = get_frame_at(cap, mid_frame_num)

        try:
            face_boxes = extract_faces(frame, detector, frame_w, frame_h)
        except Exception as e:
            print(f"⚠️  Could not extract faces for shot {idx}: {e}")
            face_boxes = []

        shots.append({
            "shot_number": idx,
            "first_frame": start_frame_num,
            "last_frame": end_frame_num,
            "detected_faces": {
                "face_count": len(face_boxes),
                "face_details": face_boxes
            },
            "detected_graphics": {}
        })

    if shots and int(shots[-1]["last_frame"]) < total_frames - 1:
       shots[-1]["last_frame"] = total_frames - 1

    cap.release()
    elapsed = time.time() - start_time
    print(f"⏱️ extract_faces and build json struct time: {elapsed:.2f} seconds")

    start_time = time.time()
    result = {"shots": shots}

    print(f"[DEBUG] total_frames={total_frames}")
    for s in shots:
        print(f"[DEBUG] shot {s['shot_number']}: {s['first_frame']}..{s['last_frame']}")
    max_last = max(s['last_frame'] for s in shots) if shots else -1
    print(f"[DEBUG] max last_frame in shots={max_last}")

    output_path.write_text(json.dumps(result, indent=2))
    elapsed = time.time() - start_time
    print(f"⏱️ write json file time: {elapsed:.2f} seconds")

    errors = validate_shot_features_json(str(output_path), "schemas/shot_features.schema.json", total_frames)
    if errors:
        print("❌ Validation errors:")
        for e in errors:
            print(" -", e)
    else:
        print(f"✅ JSON valid. Saved to {output_path}")
