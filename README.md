# FaceWise Video Toolkit v1.0

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

### Prerequisites
- Python 3.10+ (tested with Python 3.13)
- Git (for submodule management)
- FFmpeg (for audio processing)

### Clone Repository with Submodules
```bash
# Clone with submodules (recommended)
git clone --recursive <repository-url>

# Or if already cloned without submodules
git submodule update --init --recursive
```

### Setup Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### Install Dependencies

#### Runtime Dependencies
```bash
pip install -r requirements.txt
```

#### Development and Testing
```bash
pip install -r requirements-dev.txt
```

### Verify Installation
```bash
# Test imports
python -c "import facekit; print('✅ FaceKit installed successfully')"

# Run test suite
pytest tests/ -v

# Test CLI tools
python -m facekit.cli.shot_features_cli --help
```

## Quick Start

### 0. Verify Installation
First, ensure everything is working correctly:
```bash
# Activate virtual environment
source venv/bin/activate

# Test the installation
python -c "import facekit; print('✅ FaceKit ready!')"

# Run tests (optional but recommended)
pytest tests/ -v
```

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
The toolkit includes pre-trained YOLOv5Face models:
- **YOLOv5 Weights**: `models/yolov5n_state_dict.pt` (included)
- **Model Config**: `models/yolov5n.yaml` (included)
- **Alternative Model**: `facekit/yolov5faceInference/yolo5face/yolov5s-face.pt` (larger model)

> **Note**: The default models are included in the repository. For custom models, update the paths in your function calls.

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

### Core Dependencies
- Python 3.10+ (tested with Python 3.13)
- PyTorch 2.7.1+
- torchvision 0.22.1+
- OpenCV (opencv-python, opencv-contrib-python)
- NumPy 2.3.0+
- PyYAML 6.0.2+
- Pillow 11.2.1+
- JSONSchema 4.24.0+
- PySceneDetect 0.6.6+

### System Dependencies
- FFmpeg (for audio processing)

### Development Dependencies
- pytest 8.4.1+
- All runtime dependencies

## Git Submodule Management

This project uses Git submodules for the YOLOv5Face integration. Here's how to work with them properly:

### Initial Setup
```bash
# Clone with submodules
git clone --recursive <repository-url>

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### Working with Submodules

#### Pulling Latest Changes
```bash
# Pull main repository changes
git pull origin <branch-name>

# Update submodules to match the references
git submodule update --recursive
```

#### Making Changes to Submodules
If you need to modify the `yolov5faceInference` submodule:

```bash
# Navigate to submodule
cd facekit/yolov5faceInference

# Create a new branch for your changes
git checkout -b feature/your-changes

# Make your changes and commit them
git add .
git commit -m "Your descriptive commit message"

# Return to main repository
cd ../..

# Add the submodule reference update
git add facekit/yolov5faceInference
git commit -m "Update yolov5faceInference submodule: Brief description"
```

#### Checking Submodule Status
```bash
# Check if submodules are up to date
git submodule status

# See submodule changes
git diff --submodule
```

### Recent Submodule Updates
- **v2024.1**: Fixed module import structure in `yolov5faceInference`
  - Added missing `__init__.py` files for proper Python module structure
  - Submodule updated to commit `486c8b2` on branch `fix/module-imports`

### Quick Reference: Common Submodule Commands
```bash
# Check submodule status
git submodule status

# Update all submodules to latest committed references
git submodule update --recursive

# See what changed in submodules
git diff --submodule

# Initialize missing submodules
git submodule update --init --recursive

# Force update submodules (discards local changes)
git submodule update --force --recursive
```

## Development & Testing

### Running Tests
The project includes comprehensive test coverage (32 tests):

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_detection_helpers.py -v
pytest tests/test_pipeline_shot_features.py -v
pytest tests/test_postprocessing_crop.py -v
```

### Test Coverage
- ✅ Face detection and tracking algorithms
- ✅ Video processing pipelines
- ✅ Portrait cropping functionality (1-3 faces)
- ✅ JSON validation and output formats
- ✅ Audio processing tools
- ✅ YOLOv5Face model integration
- ✅ Mouth tracking and landmark detection

### Recent Fixes (v2024.1)
- 🔧 Fixed module import structure with proper `__init__.py` files
- 🔧 Added missing dependencies (PyYAML, torchvision, pillow)
- 🔧 Resolved YOLOv5Face integration issues
- 🔧 Updated requirements.txt with complete dependency list
- ✅ All 32 tests now passing

## Troubleshooting

### Common Issues

#### Import Errors
If you encounter module import errors:
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python -c "import facekit; print('Success')"
```

#### Missing Dependencies
If you get `ModuleNotFoundError`:
```bash
# Install missing packages individually
pip install PyYAML torchvision pillow

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

#### CUDA/GPU Issues
For GPU acceleration:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CUDA-compatible PyTorch if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### FFmpeg Not Found
For audio processing features:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

#### Submodule Issues

**Submodule not found or empty:**
```bash
# Initialize and update submodules
git submodule update --init --recursive

# Force update if needed
git submodule update --force --recursive
```

**Import errors from yolov5faceInference:**
```bash
# Check submodule status
git submodule status

# Ensure you're on the correct submodule commit
cd facekit/yolov5faceInference
git log --oneline -5
# Should show commit 486c8b2 or later with "Fix module imports"

# Return to main repo and test
cd ../..
python -c "import facekit.yolov5faceInference.yolo5face.yoloface.face_detector; print('✅ Submodule imports working')"
```

**Submodule shows "modified content":**
```bash
# Check what changed in the submodule
git diff --submodule

# If you want to discard submodule changes
git submodule update --force

# If you want to commit submodule changes, see "Git Submodule Management" section above
```

**Team member can't access submodule changes:**
```bash
# Ensure they have the latest main repository
git pull origin <branch-name>

# Update their submodules
git submodule update --recursive

# If still having issues, re-initialize
git submodule deinit facekit/yolov5faceInference
git submodule update --init facekit/yolov5faceInference
```

## License

This project incorporates the YOLOv5Face model wrapper, which is released under the MIT License.
