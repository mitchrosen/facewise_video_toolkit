import argparse
import sys
from pathlib import Path
from ast import literal_eval

# Ensure internal modules are importable
sys.path.append(str(Path(__file__).resolve().parent / "mouthtracker"))

from pipeline.mouthtrack_frame_by_frame import level3_multiface_yolo5face_mouthtrack_v2


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv5 face detection + mouth tracking on a video file.")
    
    parser.add_argument("--input_path", required=True, help="Path to input video")
    parser.add_argument("--output_path", required=True, help="Path to output video (with overlays)")
    parser.add_argument("--model_path", required=True, help="Path to YOLOv5 face weights (.pt file)")
    parser.add_argument("--config_path", required=True, help="Path to YOLOv5 face config (.yaml file)")

    parser.add_argument("--require_gpu",
                        type=literal_eval,
                        choices=[True, False],
                        default=argparse.SUPPRESS,
                        help="Set GPU requirement explicitly. If omitted, uses function's default.")
    
    parser.add_argument("--show_periodic",
                        type=literal_eval,
                        choices=[True, False],
                        default=argparse.SUPPRESS,
                        help="Show frames periodically.")
    
    parser.add_argument("--display_interval_sec", 
                        type=float,
                        default=argparse.SUPPRESS,
                        help="Display frame interval in seconds (ignored if show_periodic is False)")
    
    parser.add_argument("--min_face", 
                        type=int,
                        default=argparse.SUPPRESS,
                        help="Minimum face size to track (in pixels)")

    args = parser.parse_args()

    # Collect only explicitly passed optional arguments
    kwargs = {}
    for optional in ["require_gpu", "show_periodic", "display_interval_sec", "min_face"]:
        if hasattr(args, optional):
            kwargs[optional] = getattr(args, optional)

    # Required arguments passed directly
    level3_multiface_yolo5face_mouthtrack_v2(
        input_path=args.input_path,
        output_path=args.output_path,
        model_path=args.model_path,
        config_path=args.config_path,
        **kwargs
    )


if __name__ == "__main__":
    main()
