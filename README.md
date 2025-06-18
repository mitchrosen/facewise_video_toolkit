# FaceWise Video Toolkit

A comprehensive Python toolkit for face detection, tracking, and video processing with advanced portrait cropping capabilities. Built on YOLOv5Face for robust face detection and OpenCV for video processing.

## Features

### 🎯 Face Detection & Tracking
- **YOLOv5Face Integration**: High-performance face detection using YOLOv5Face models
- **Multi-face Tracking**: Track multiple faces across video frames using CSRT or KCF trackers
- **Adaptive Detection**: Intelligent detection scheduling with fallback mechanisms
- **Mouth Tracking**: Specialized mouth region tracking with landmark-based bounding boxes

### 📹 Video Processing Pipelines
- **Shot Feature Extraction**: Automatic scene detection and face analysis per shot
- **Frame-by-Frame Processing**: Real-time face tracking with visual feedback
- **JSON Output**: Structured data export for face coordinates and tracking metadata

### ✂️ Portrait Video Cropping
- **Smart Cropping**: Automatic portrait-mode cropping based on face positions
- **Multi-face Support**: Handles 1-3 faces with different cropping strategies:
  - Single face: Centered portrait crop
  - Two faces: Upright or angular crop with rotation/scaling
  - Three faces: Optimized vertical alignment
- **Aspect Ratio Control**: Configurable output dimensions (default: 720x1560, 2.17:1)
- **Audio Preservation**: Automatic audio track restoration using FFmpeg

### 🔧 Post-processing Tools
- **JSON Validation**: Schema-based validation for shot features output
- **Audio Tools**: FFmpeg integration for audio track management
- **Crop Helpers**: Advanced geometric transformations for face alignment

## Installation

### Runtime Dependencies
```bash
pip install -r requirements.txt
```

### Development and Testing
```bash
pip install -r requirements-dev.txt
```

## Quick Start

### 1. Extract Shot Features
Analyze a video to detect scenes and extract face information per shot:

```bash
python -m facekit.cli.shot_features_cli input_video.mp4 --output video_shots.json
```

### 2. Face Tracking with Visual Output
Process a video with real-time face tracking visualization:

```python
from facekit.pipeline.mouthtrack_frame_by_frame import multiface_mouthtrack

multiface_mouthtrack(
    input_path="input_video.mp4",
    output_path="tracked_output.mp4",
    model_path="models/yolov5n_state_dict.pt",
    config_path="models/yolov5n.yaml",
    show_periodic=True
)
```

### 3. Generate Tracking JSON
Export face tracking data to JSON format:

```python
from facekit.pipeline.json_writer import multiface_tracking_to_json

multiface_tracking_to_json(
    input_path="input_video.mp4",
    output_json_path="tracking_data.json",
    model_path="models/yolov5n_state_dict.pt",
    config_path="models/yolov5n.yaml"
)
```

### 4. Portrait Video Cropping
Create portrait-mode videos from tracking data:

```python
from facekit.postprocessing.crop_portrait_video import crop_video_from_json

crop_video_from_json(
    json_path="tracking_data.json",
    video_path="input_video.mp4",
    output_path="portrait_output.mp4",
    aspect_ratio=2.17,
    output_size=(720, 1560)
)
```

## Project Structure

```
facekit/
├── cli/                    # Command-line interfaces
│   ├── shot_features_cli.py
│   └── visualize_shots_cli.py
├── detection/              # Face detection core
│   ├── detection_helpers.py
│   ├── tracker_base.py
│   └── yolo5face_model.py
├── pipeline/               # Processing pipelines
│   ├── generate_shot_features.py
│   ├── json_writer.py
│   └── mouthtrack_frame_by_frame.py
├── postprocessing/         # Video post-processing
│   ├── crop_helpers.py
│   ├── crop_portrait_video.py
│   └── validate_shot_features_json.py
├── tracking/               # Face and mouth tracking
│   ├── face_tracker.py
│   └── mouth_tracker.py
├── output/                 # Output utilities
│   ├── audio_tools.py
│   └── json_writer.py
├── utils/                  # Utility functions
│   └── geometry.py
└── yolov5faceInference/    # YOLOv5Face wrapper
```

## Configuration

### Model Files
- **YOLOv5 Weights**: `models/yolov5n_state_dict.pt`
- **Model Config**: `models/yolov5n.yaml`

### Key Parameters
- `min_face`: Minimum face size in pixels (default: 10)
- `track_interval`: Frames between detections (default: 30)
- `tracker_type`: "CSRT" or "KCF" (default: "CSRT")
- `target_size`: Detection input size (default: 640)

## Output Formats

### Shot Features JSON
Structured output with scene detection and face analysis:
```json
{
  "shots": [
    {
      "shot_number": 1,
      "first_frame": 0,
      "last_frame": 299,
      "detected_faces": {
        "face_count": 2,
        "face_details": [
          {
            "center_x": 45.5,
            "center_y": 32.1,
            "face_width": 15.2,
            "face_height": 20.8
          }
        ]
      },
      "detected_graphics": {}
    }
  ]
}
```

### Tracking JSON
Frame-by-frame face tracking data:
```json
[
  {
    "frame": 0,
    "faces": [
      {
        "x": 100,
        "y": 150,
        "w": 80,
        "h": 100,
        "conf": 0.95,
        "source": "detected"
      }
    ]
  }
]
```

## Testing

Run the test suite:
```bash
pytest
```

Test coverage includes:
- Face detection and tracking algorithms
- Video processing pipelines
- Portrait cropping functionality
- JSON validation and output formats
- Audio processing tools

## Requirements

- Python 3.10+
- OpenCV (cv2)
- PyTorch
- NumPy
- JSONSchema
- PySceneDetect
- FFmpeg (for audio processing)

## License

This project incorporates the YOLOv5Face model wrapper, which is released under the MIT License.
