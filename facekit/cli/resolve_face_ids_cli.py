import argparse
from pathlib import Path
import json
import cv2
import torch

from facekit.pipeline.track_across_segments import track_across_segments
from facekit.tracking.tracking_resolution import GlobalIdentityResolver
from facekit.tracking.serialize import tracks_to_json_dict
from facekit.pipeline.draw_tracks import draw_tracks_on_video
from facekit.tracking.face_structures import FaceTrack
from facekit.pipeline.generate_shot_features import generate_shot_features_json
from facekit.embedding.embedder import FaceEmbedder
from facekit.detection.face_detector import FaceDetector
from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.io.frame_provider import ReaderCoordinator

from facekit.output.json_v2 import (
    V2WriterConfig,
    build_v2_manifest_from_tracks,
    derive_face_metadata,
    write_v2_json,
    EmbeddingCollector,
)
from facekit.validation import validate_manifest


def _resolve_device(arg_device: str) -> str:
    """
    Choose 'cuda' or 'cpu'.
    - 'cuda'/'cpu': honored explicitly
    - 'auto' (default): prefer CUDA if torch.cuda.is_available()
    """
    arg_device = (arg_device or "auto").lower()
    if arg_device in ("cuda", "gpu"):
        return "cuda"
    if arg_device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Track faces and resolve global identities across shots (V2 JSON writer)"
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--detector_model", default="models/detector/yolov5n_state_dict.pt",
                        help="Path to YOLOv5 model weights")
    parser.add_argument("--embedding_model", default="models/embedding/glintr100_dynamic.onnx",
                        help="Path to ArcFace ONNX model")
    parser.add_argument("--config", default="models/detector/yolov5n.yaml",
                        help="Path to YOLOv5 model config")
    parser.add_argument("--shot_segmentation", default=None,
                        help="Path to shot segmentation JSON (optional)")
    parser.add_argument("--output_segment_json", default=None,
                        help="Optional path to save segment-only tracks JSON")
    parser.add_argument("--output_global_json", default=None,
                        help="Path to write the **JSON V2** manifest for resolved global IDs")
    parser.add_argument("--output_video", nargs="?", const=True, default=None,
                        help="Optionally render labeled video with global + segment IDs")
    parser.add_argument("--resolver-debug-json", default=None,
                        help="If set, write resolver assignment debug JSON to this path.")
    parser.add_argument("--resolver-seed-map", default=None,
                        help="Path to JSON: {'seeds': [[shot,segment,global_id], ...]} to pin IDs for known groups.")

    parser.add_argument("--detect_interval", type=int, default=30)
    parser.add_argument("--embedding_batch_size_max", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Compute device for detector/embedder (default: auto)")
    # logging
    parser.add_argument("--log", default="INFO",
                    choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    parser.add_argument("--log-file", default=None)

    # embedding storage controls for V2
    parser.add_argument(
        "--emb-store",
        choices=["inline", "sidecar", "none"],
        default="inline",
        help="How to serialize embeddings in V2 manifest (default: inline).",
    )
    parser.add_argument(
        "--emb-sidecar-path",
        default=None,
        help=("Path to sidecar embeddings file (only used when --emb-store=sidecar). "
              "If omitted, uses '<video>.embeddings.npz'."),
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    # --- Prepare or load shot segmentation  ----------------
    shot_json_path = (
        Path(args.shot_segmentation)
        if args.shot_segmentation
        else input_path.with_name(f"{input_path.stem}_shot_segmentation.json")
    )

    if not shot_json_path.exists():
        print(f"Shot segmentation file not found at {shot_json_path}. Generating it now...")
        generate_shot_features_json(
            video_path=str(input_path),
            output_json_path=str(shot_json_path),
            detector_model_path=args.detector_model,
            config_path=args.config,
        )

    # Validate and fix coverage to final frame
    with open(shot_json_path, "r") as f:
        shot_data = json.load(f)
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if shot_data.get("shots") and int(shot_data["shots"][-1]["last_frame"]) < total_frames - 1:
        print(f"Last shot ends at {shot_data['shots'][-1]['last_frame']} but video has {total_frames-1}. Extending it.")
        shot_data["shots"][-1]["last_frame"] = total_frames - 1
        shot_json_path.write_text(json.dumps(shot_data, indent=2))
        print("Shot segmentation JSON updated to include final frame.")

    # --- Detector + embedder ------------------------------------------------
    device = _resolve_device(args.device)
    print(f"Using device: {device} (torch.cuda.is_available()={torch.cuda.is_available()})")

    print("Initializing detector and embedder...")
    detector_model = load_yolo5face_model(
        detector_model_path=args.detector_model,
        config_path=args.config,
        device=device,
    )
    detector = FaceDetector(detector_model=detector_model)
    embedder = FaceEmbedder(embedding_model_path=args.embedding_model, device=device)
    # Optional runtime knob
    if hasattr(embedder, "set_max_batch_size"):
        embedder.set_max_batch_size(int(args.embedding_batch_size_max))
    elif hasattr(embedder, "max_batch_size"):
        try:
            setattr(embedder, "max_batch_size", int(args.embedding_batch_size_max))
        except Exception:
            pass

    # Quick ORT provider visibility (best-effort)
    try:
        prov = getattr(embedder.embedding_model, "session", None)
        if prov is not None:
            print("ArcFace ORT providers:", embedder.embedding_model.session.get_providers())
    except Exception:
        pass

    # --- Tracking across segments ------------------------------------------
    with ReaderCoordinator(str(input_path)) as frame_provider:
        tracks = track_across_segments(
            frame_source=frame_provider,  
            shot_json_path=str(shot_json_path),
            detector=detector,
            embedder=embedder,
            detect_interval=args.detect_interval,
            embedding_batch_size_max=args.embedding_batch_size_max,
        )

    # Save segment-only JSON if requested
    if args.output_segment_json:
        segment_path = Path(args.output_segment_json)
        segment_path.parent.mkdir(parents=True, exist_ok=True)
        segment_path.write_text(json.dumps(tracks_to_json_dict(tracks), indent=2))
        print(f"Wrote segment tracks to {segment_path}")

    # Resolve global IDs and (optionally) render
    seed_map = None
    if args.resolver_seed_map:
        with open(args.resolver_seed_map, "r") as f:
            payload = json.load(f)
        seed_map = {(int(s), int(seg)): int(gid) for s, seg, gid in payload.get("seeds", [])}

    resolver = GlobalIdentityResolver(
        embedding_threshold=0.70,
        debug_dump_path=args.resolver_debug_json,
        seed_map=seed_map,
    )
    resolver.resolve_global_ids(tracks)

    if args.output_video:
        if args.output_video is True:
            output_video_path = input_path.with_name(f"{input_path.stem}_global_faceIDs.mp4")
        else:
            output_video_path = Path(args.output_video)

        def label_with_shot_track_face_segment_frame_ids(track: FaceTrack, frame_num: int) -> str:
            def q(x): return "?" if x is None else x
            sid = q(getattr(track, "shot_id", None))
            tid = q(getattr(track, "track_id", None))
            gid = q(getattr(track, "global_id", None))
            seg = q(getattr(track, "segment_id", None))
            return f"ShotID{sid}_TrackID{tid}_FaceSegID{seg}_FaceGlobID{gid}_Frame#{frame_num}"

        print(f"Rendering labeled video to {output_video_path}")
        draw_tracks_on_video(
            video_path=str(input_path),
            output_path=str(output_video_path),
            tracks=tracks,
            label_fmt=label_with_shot_track_face_segment_frame_ids,
        )

    # --- JSON V2 manifest (authoritative) ----------------------------------
    if args.output_global_json:
        # Map CLI to config value
        emb_store = None if args.emb_store == "none" else args.emb_store  # None | 'inline' | 'sidecar'

        # Decide sidecar path before building cfg (only meaningful for 'sidecar')
        sidecar_path = None
        if emb_store == "sidecar":
            sidecar_path = Path(args.emb_sidecar_path) if args.emb_sidecar_path else input_path.with_suffix(".embeddings.npz")

        cfg = V2WriterConfig(
            video_path=str(input_path),
            video_size=(width, height),
            fps=float(fps),
            total_frames=int(total_frames),
            normalize_to_percent=True,
            emb_store=emb_store,
            emb_sidecar_path=sidecar_path,
        )

        collector = (
            EmbeddingCollector(cfg.emb_store, dim=512)
            if cfg.emb_store in ("inline", "sidecar")
            else None
        )

        face_meta = derive_face_metadata(tracks)

        manifest = build_v2_manifest_from_tracks(
            tracks,
            cfg,
            face_metadata=face_meta,
            generation=None,  # derive commit/branch inside writer
            detector=detector,
            embedder=embedder,
            tracking_params={"detect_interval": int(args.detect_interval)},
            validator=None,
            collector=collector,
        )

        # Write sidecar only if requested
        if cfg.emb_store == "sidecar" and collector is not None:
            # Use the path already decided above (defaults to <video>.embeddings.npz)
            manifest["embedding_sidecar"] = collector.finalize_sidecar(cfg.emb_sidecar_path)

        # Write the V2 JSON
        out_glob = Path(args.output_global_json)
        write_v2_json(str(out_glob), manifest)
        print(f"Wrote V2 JSON to {out_glob}")
        if "embedding_sidecar" in manifest:
            print(f"Wrote embedding sidecar to {manifest['embedding_sidecar']['path']}")

        # Validate against schema
        errs = validate_manifest(out_glob, total_frame_count=int(total_frames))
        if errs:
            print("Validation errors:")
            for e in errs:
                print(" -", e)
            raise SystemExit(2)
        print("Validation OK.")


if __name__ == "__main__":
    main()
