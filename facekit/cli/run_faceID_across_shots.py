import argparse
from pathlib import Path
import json

from facekit.pipeline.track_across_shots import track_across_shots
from facekit.tracking.serialize import tracks_to_json_dict
from facekit.pipeline.generate_shot_features import generate_shot_features_json
from facekit.pipeline.draw_tracks import draw_tracks_on_video
from facekit.tracking.face_tracks import FaceTrack

def main():
    parser = argparse.ArgumentParser(description="Track faces across shots in a video")
    parser.add_argument("--input", 
                        required=True, 
                        type=str,
                        help="Path to input video file")
    parser.add_argument("--shot_segmentation", 
                        type=str,
                        default=None, 
                        help="Optional path to shot segmentation metadata JSON - built if doesn't exit")
    parser.add_argument("--output_tracks", 
                        type=str,
                        default=False, 
                        help="Path to full set of face tracks JSON")
    parser.add_argument("--model", 
                        type=str,
                        default="models/yolov5n_state_dict.pt")
    parser.add_argument("--config",
                        type=str,
                        default="models/yolov5n.yaml")
    parser.add_argument(
        "--output_video",
        nargs="?",
        const=True,
        default=None,
        help="Optional: if provided without value, saves marked video to default path; if a path is provided, saves there"
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    shot_segmentation_path= (
        Path(arg.shot_segmentation)
        if args.shot_segmentation
        else input_path.with_name(f"{input_path.stem}_shot_segmentation.json")
    )

    # If missing, auto-generate shot segmentation
    if not shot_segmentation_path.exists():
        print(f"⚙️  Generating shot segmentation: {shot_segmentation_path}")
        generate_shot_features_json(video_path=str(input_path), output_json_path=str(shot_segmentation_path))
    else:
        print(f"🎞️  Using existing shot segmentation file: {shot_segmentation_path}")

    output_tracks_path = (
        Path(args.output_tracks)
        if args.output_tracks
        else input_path.with_name(f"{input_path.stem}_face_tracks.json")
    )

    tracks = track_across_shots(
        video_path=input_path,
        shot_json_path=shot_segmentation_path,
        model_path=args.model,
        config_path=args.config
    )

    # Serialize to JSON
    output_tracks_path.parent.mkdir(parents=True, exist_ok=True)
    json_data = tracks_to_json_dict(tracks)
    output_tracks_path.write_text(json.dumps(json_data, indent=2))
    print(f"✅ Wrote {len(tracks)} tracks to {output_tracks_path}")

    # Optionally write out video with bounding boxes and track IDs
    if args.output_video:
        if args.output_video is True:
            output_video_path = input_path.with_name(f"{input_path.stem}_trackedFaceIDs.mp4")
        else:
            output_video_path = Path(args.output_video)

        def label_with_shot_and_id(track: FaceTrack) -> str:
            return f"S{track.shot_id}_T{track.track_id}"

        print(f"🎬 Rendering output video with overlays to {output_video_path}")
        draw_tracks_on_video(
            video_path=str(input_path),
            output_path=str(output_video_path),
            tracks=tracks,
            label_fmt=label_with_shot_and_id
        )


if __name__ == "__main__":
    main()
