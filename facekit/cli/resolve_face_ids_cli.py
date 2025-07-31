import argparse
from pathlib import Path
import json
import cv2

from facekit.pipeline.track_across_shots import track_across_shots
from facekit.tracking.tracking_resolution import GlobalIdentityResolver
from facekit.tracking.serialize import tracks_to_json_dict
from facekit.pipeline.draw_tracks import draw_tracks_on_video
from facekit.tracking.face_structures import FaceTrack
from facekit.pipeline.generate_shot_features import generate_shot_features_json
from facekit.embedding.embedder import FaceEmbedder
from facekit.detection.face_detector import FaceDetector
from facekit.detection.yolo5face_model import load_yolo5face_model

def main():
    parser = argparse.ArgumentParser(description="Track faces and resolve global identities across shots")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--detector_model", default="models/detector/yolov5n_state_dict.pt", help="Path to YOLOv5 model weights")
    parser.add_argument("--embedding_model", default="models/embedding/glintr100_dynamic.onnx", help="Path to ArcFace ONNX model") 
    parser.add_argument("--config", default="models/detector/yolov5n.yaml", help="Path to YOLOv5 model config")
    parser.add_argument("--shot_segmentation", default=None, help="Path to shot segmentation JSON (optional)")
    parser.add_argument("--output_vchunk_json", default=None, help="Optional path to save vchunk-only tracks JSON")
    parser.add_argument("--output_global_json", default=None, help="Optional path to save resolved global ID tracks JSON")
    parser.add_argument("--output_video", nargs="?", const=True, default=None,
                        help="Optionally render labeled video with global + vchunk IDs")

    args = parser.parse_args()

    input_path = Path(args.input)
    shot_json = (
        Path(args.shot_segmentation)
        if args.shot_segmentation
        else input_path.with_name(f"{input_path.stem}_shot_segmentation.json")
    )

    # Auto-generate shot segmentation if missing
    if not shot_json.exists():
        print(f"⚠️ Shot segmentation file not found at {shot_json}. Generating it now...")
        generate_shot_features_json(
            video_path=str(input_path),
            output_json_path=str(shot_json),
            detector_model_path=args.detector_model,
            config_path=args.config
        )

    # Validate and fix shot segmentation coverage
    with open(shot_json, "r") as f:
        shot_data = json.load(f)

    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if shot_data["shots"] and int(shot_data["shots"][-1]["last_frame"]) < total_frames - 1:
        print(f"⚠️ Last shot ends at {shot_data['shots'][-1]['last_frame']} but video has {total_frames-1}. Extending it.")
        shot_data["shots"][-1]["last_frame"] = total_frames - 1
        shot_json.write_text(json.dumps(shot_data, indent=2))
        print("Shot segmentation JSON updated to include final frame.")

    # Initialize embedder
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    # Initialize detector and embedder
    print("Initializing detector and embedder...")

    # Remove --config from parser.add_argument section

    # Initialize detector and embedder
    print("Initializing detector and embedder...")
    detector_model = load_yolo5face_model(
        detector_model_path=args.detector_model,
        config_path=args.config,
        device=device)
    detector = FaceDetector(detector_model=detector_model)
    embedder = FaceEmbedder(embedding_model_path=args.embedding_model, device=device)

    # Track across shots
    tracks = track_across_shots(
        video_path=str(input_path),
        shot_json_path=str(shot_json),
        detector=detector,
        embedder=embedder
    )

    # Save vchunk-only JSON if requested
    if args.output_vchunk_json:
        vchunk_path = Path(args.output_vchunk_json)
        vchunk_path.parent.mkdir(parents=True, exist_ok=True)
        vchunk_path.write_text(json.dumps(tracks_to_json_dict(tracks), indent=2))
        print(f"Wrote vchunk tracks to {vchunk_path}")

    # Resolve global IDs based on embeddings
    resolver = GlobalIdentityResolver(embedding_threshold=0.70)
    resolver.resolve_global_ids(tracks)

    # Save globally resolved JSON if requested
    if args.output_global_json:
        global_path = Path(args.output_global_json)
        global_path.parent.mkdir(parents=True, exist_ok=True)
        global_path.write_text(json.dumps(tracks_to_json_dict(tracks), indent=2))
        print(f"Wrote globally resolved tracks to {global_path}")

    # Optionally render annotated video
    if args.output_video:
        if args.output_video is True:
            output_video_path = input_path.with_name(f"{input_path.stem}_global_faceIDs.mp4")
        else:
            output_video_path = Path(args.output_video)

        def label_with_global_and_vchunk(track: FaceTrack) -> str:
            gid = track.global_id if hasattr(track, "global_id") and track.global_id is not None else "?"
            vid = track.vchunk_id if hasattr(track, "vchunk_id") and track.vchunk_id is not None else "?"
            return f"G{gid}_V{vid}"

        print(f"Rendering labeled video to {output_video_path}")
        draw_tracks_on_video(
            video_path=str(input_path),
            output_path=str(output_video_path),
            tracks=tracks,
            label_fmt=label_with_global_and_vchunk
        )


if __name__ == "__main__":
    main()
