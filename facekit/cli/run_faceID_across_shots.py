import argparse
from pathlib import Path
import json
import cv2

from facekit.pipeline.track_across_shots import track_across_shots
from facekit.pipeline.generate_shot_features import generate_shot_features_json
from facekit.pipeline.draw_tracks import draw_tracks_on_video
from facekit.tracking.serialize import tracks_to_json_dict
from facekit.tracking.face_structures import FaceTrack
from facekit.embedding.embedder import FaceEmbedder
from facekit.detection.face_detector import FaceDetector

def main():
    parser = argparse.ArgumentParser(description="Track faces across shots in a video")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--shot_segmentation", default=None, help="Path to shot segmentation JSON (optional)")
    parser.add_argument("--output_tracks", default=None, help="Path to save face tracks JSON")
    parser.add_argument("--detector_model", default="models/detector/yolov5n_state_dict.pt", help="Path to YOLOv5 model weights")
    parser.add_argument("--config", default="models/detector/yolov5n.yaml", help="Path to YOLOv5 config")
    parser.add_argument("--embedding_model", default="models/embedding/glintr100_dynamic.onnx", help="Path to ArcFace ONNX model")
    parser.add_argument("--output_video", nargs="?", const=True, default=None,
                        help="Optionally render labeled video (provide path or leave empty for default)")
    args = parser.parse_args()

    input_path = Path(args.input)

    # Resolve shot segmentation path
    shot_segmentation_path = (
        Path(args.shot_segmentation)
        if args.shot_segmentation
        else input_path.with_name(f"{input_path.stem}_shot_segmentation.json")
    )

    # Auto-generate shot segmentation if missing
    if not shot_segmentation_path.exists():
        print(f"⚙️ Generating shot segmentation: {shot_segmentation_path}")
        generate_shot_features_json(
            video_path=str(input_path),
            output_json_path=str(shot_segmentation_path),
            detector_model_path=args.detector_model,
            config_path=args.config
        )
    else:
        print(f"🎞️ Using existing shot segmentation: {shot_segmentation_path}")

    # Auto-select device
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    # Initialize detector and embedder
    print(f"Initializing FaceDetector ({device})...")
    detector = FaceDetector(model_path=args.detector_model, config_path=args.config, device=device)
    print(f"Initializing FaceEmbedder ({device})...")
    embedder = FaceEmbedder(embedding_model_path=args.embedding_model, device=device)

    # Track faces across shots
    tracks = track_across_shots(
        video_path=str(input_path),
        shot_json_path=str(shot_segmentation_path),
        detector=detector,
        embedder=embedder
    )

    # Serialize tracks to JSON
    output_tracks_path = (
        Path(args.output_tracks)
        if args.output_tracks
        else input_path.with_name(f"{input_path.stem}_face_tracks.json")
    )
    output_tracks_path.parent.mkdir(parents=True, exist_ok=True)
    output_tracks_path.write_text(json.dumps(tracks_to_json_dict(tracks), indent=2))
    print(f"Wrote {len(tracks)} tracks to {output_tracks_path}")

    # Optionally render video
    if args.output_video:
        output_video_path = (
            input_path.with_name(f"{input_path.stem}_trackedFaceIDs.mp4")
            if args.output_video is True
            else Path(args.output_video)
        )

        def label_with_shot_and_id(track: FaceTrack) -> str:
            return f"S{track.shot_number}_C{track.vchunk_id}"

        print(f"Rendering labeled video to {output_video_path}")
        draw_tracks_on_video(
            video_path=str(input_path),
            output_path=str(output_video_path),
            tracks=tracks,
            label_fmt=label_with_shot_and_id
        )

if __name__ == "__main__":
    main()
