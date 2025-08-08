import argparse
from pathlib import Path
from facekit.pipeline.run_tracking_on_video import run_tracking_on_video

def main():
    parser = argparse.ArgumentParser(description="Run face tracking on a video.")

    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input video file"
    )

    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to save output video with bounding boxes"
    )

    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit number of frames to process (useful for debugging)"
    )

    parser.add_argument(
        "--no-draw", action="store_true",
        help="Disable drawing bounding boxes and saving video output"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_faceIDs.mp4")
    )

    run_tracking_on_video(
        input_path=input_path,
        output_path=output_path,
        draw=not args.no_draw,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
