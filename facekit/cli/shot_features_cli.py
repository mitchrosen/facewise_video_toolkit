import argparse
from pathlib import Path
from facekit.pipeline.generate_shot_features import generate_shot_features_json

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-shot face features from a video and write JSON output."
    )
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (defaults to <video_basename>_shots.json)"
    )
    parser.add_argument(
        "--detector_model_path", type=str,
        # default="facekit/yolov5faceInference/yolo5face/yolov5s-face.pt",
        default="models/detector/yolov5n_state_dict.pt",
        help="Path to the YOLOv5 face detector weights"
    )
    parser.add_argument(
        "--config_path", type=str,
        # default="facekit/yolov5faceInference/yolo5face/yolov5n.yaml",
        default="models/detector/yolov5n.yaml",
        help="Path to YOLOv5 face model config file"
    )
    args = parser.parse_args()

    input_path = Path(args.video_path)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_shots.json")
    )

    generate_shot_features_json(
        video_path=str(input_path),
        output_json_path=str(output_path),
        detector_model_path=args.detector_model_path,
        config_path=args.config_path
    )

if __name__ == "__main__":
    main()
