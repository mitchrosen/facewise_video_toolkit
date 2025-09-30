from contextlib import ExitStack
from pathlib import Path
from importlib.resources import files
import json
import torch
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
import numpy as np

from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.detection.face_detector import FaceDetector
from facekit.utils.geometry import normalize_face_bbox
from facekit.validation.json.validate_shot_features_json_v1 import validate_shot_features_json_v1
from facekit.io.frame_provider import ReaderCoordinator

def detect_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([str(video_path)])
    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(frame_source=video_manager)

    return scene_manager.get_scene_list()

def extract_faces(frame, detector: FaceDetector, frame_w, frame_h):
    result = detector.detect_faces_in_frame(frame, target_size=640)
    if result is None:
        return []
    boxes, _, _ = result
    return [normalize_face_bbox((x1, y1, x2, y2), frame_w, frame_h) for x1, y1, x2, y2 in boxes]

def generate_shot_features_json(
        video_path: str, 
        output_json_path: str,
        detector_model_path: str = "models/detector/yolov5n_state_dict.pt",
        config_path: str = "models/detector/yolov5n.yaml",
        threshold: float = 30.0
):
    import time
    t0 = time.time()
    video_path = Path(video_path)
    output_path = Path(output_json_path)

    shots = []

    with ExitStack() as stack:
        frame_provider = stack.enter_context(ReaderCoordinator(str(video_path)))  # auto-close
       
        # Basic metadata via provider (avoid separate cv2 VideoCapture)
        total_frames = frame_provider.total_frames()
        fps = frame_provider.fps() or 30.0
        size = frame_provider.size() or (0, 0)
        frame_w, frame_h = size if size != (0, 0) else (0, 0)

        # Scene detection
        t1 = time.time()
        scenes = detect_scenes(video_path, threshold)
        t2 = time.time()
        print(f"setup+scene detect: {(t2 - t0):.2f}s (scenes in {t2-t1:.2f}s)")

        if not scenes:
            tf = total_frames if total_frames > 0 else 1 # If video has 0 frames, synthesize a trivial 0..0 scene
            scenes = [(FrameTimecode(0, fps), FrameTimecode(tf - 1, fps))]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        detector_model = load_yolo5face_model(detector_model_path=detector_model_path, config_path=config_path, device=device)
        detector = FaceDetector(detector_model)

        t3 = time.time()
        shots = []
        for idx, (scene_start, scene_end) in enumerate(scenes, start=1):
            start_frame_num = scene_start.get_frames()
            end_frame_num = scene_end.get_frames() - 1
            mid_frame_num = (start_frame_num + end_frame_num) // 2

            frame = frame_provider.get_frame(frame_idx=mid_frame_num)
            
            if frame is None:
                face_boxes = []
            else:
                try:
                    face_boxes = extract_faces(frame, detector, frame_w, frame_h)
                except Exception as e:
                    print(f"Could not extract faces for shot {idx}: {e}")
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
        print(f"face sampling+json build: {(time.time()-t2):.2f}s")

    if total_frames and shots:
        shots[-1]["last_frame"] = min(shots[-1]["last_frame"], total_frames - 1)

    elapsed = time.time() - t0
    print(f"extract_faces and build json struct time: {elapsed:.2f} seconds")

    t4 = time.time()
    result = {"shots": shots}

    output_path.write_text(json.dumps(result, indent=2))
    elapsed = time.time() - t4
    print(f"write json file time: {elapsed:.2f} seconds")

    SCHEMA_PATH = Path("schemas/shot_features_v1.schema.json")
    errors = validate_shot_features_json_v1(str(output_path), SCHEMA_PATH, total_frames)
    if errors:
        print("Validation errors:")
        for e in errors:
            print(" -", e)
    else:
        print(f"JSON valid. Saved to {output_path}")
