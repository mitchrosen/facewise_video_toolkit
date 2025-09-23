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
from facekit.utils.video_reader import VideoReader

def detect_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    return scene_manager.get_scene_list()

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

    reader = VideoReader(str(video_path))
    stream = reader.stream
    fps = reader.fps
    frame_w = stream.codec_context.width
    frame_h = stream.codec_context.height
    elapsed = time.time() - start_time
    print(f"setup time: {elapsed:.2f} seconds")
   
    # Scene detection
    start_time = time.time()
    scenes = detect_scenes(video_path, threshold)
    elapsed = time.time() - start_time
    print(f"detect_scenes time: {elapsed:.2f} seconds")

    # If no scenes, create a single scene covering the whole video.
    # Derive a frame count from PyAV if available; otherwise estimate from duration.
    if not scenes:
        # Try to estimate total frames robustly
        total_frames_guess = int(stream.frames) if getattr(stream, "frames", 0) else 0
        if total_frames_guess <= 0 and getattr(stream, "duration", None):
            # duration is in "time_base" units; convert to seconds then to frames
            seconds = float(stream.duration * reader.time_base)
            total_frames_guess = max(1, int(round(seconds * fps)))
        if total_frames_guess <= 0:
            total_frames_guess = 1  # last resort
        scenes = [(FrameTimecode(0, fps), FrameTimecode(total_frames_guess, fps))]

    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    detector_model = load_yolo5face_model(detector_model_path=detector_model_path, config_path=config_path, device=device)
    detector = FaceDetector(detector_model)

    # Helper: fetch exactly one frame by index via PyAV reader
    def _get_frame_at(reader: VideoReader, frame_num: int):
        arrs = reader.get_frames(frame_num, frame_num)
        if not arrs:
            raise RuntimeError(f"Failed to read frame {frame_num}")
        return arrs[0]

    start_time = time.time()
    shots = []
    for idx, (scene_start, scene_end) in enumerate(scenes, start=1):
        start_frame_num = scene_start.get_frames()
        end_frame_num = scene_end.get_frames() - 1
        mid_frame_num = (start_frame_num + end_frame_num) // 2

        frame = _get_frame_at(reader, mid_frame_num)

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

    # Use the scene end for a consistent total frame count under VFR
    total_frames = scenes[-1][1].get_frames()

    if shots and int(shots[-1]["last_frame"]) < total_frames - 1:
       shots[-1]["last_frame"] = total_frames - 1

    reader.close()
    elapsed = time.time() - start_time
    print(f"extract_faces and build json struct time: {elapsed:.2f} seconds")

    start_time = time.time()
    result = {"shots": shots}


    output_path.write_text(json.dumps(result, indent=2))
    elapsed = time.time() - start_time
    print(f"write json file time: {elapsed:.2f} seconds")

    errors = validate_shot_features_json(str(output_path), "schemas/shot_features.schema.json", total_frames)
    if errors:
        print("Validation errors:")
        for e in errors:
            print(" -", e)
    else:
        print(f"JSON valid. Saved to {output_path}")
