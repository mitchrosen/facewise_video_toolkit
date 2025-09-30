import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from contextlib import ExitStack
import torch
import logging, sys

from facekit.io.frame_provider import ReaderCoordinator
from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.detection.face_detector import FaceDetector
from facekit.embedding.embedder import FaceEmbedder
from facekit.pipeline.draw_tracks import draw_tracks_on_video
from facekit.pipeline.generate_shot_features import generate_shot_features_json
from facekit.pipeline.track_across_segments import track_across_segments
from facekit.tracking.tracking_resolution import GlobalIdentityResolver
from facekit.tracking.face_structures import FaceTrack

from facekit.output.json_v2 import (
    V2WriterConfig,
    build_v2_manifest_from_tracks,
    derive_face_metadata,
    write_v2_json,
    EmbeddingCollector,
)
from facekit.validation import validate_manifest

def _fix_shot_coverage(shot_json_path: Path, total_frames: int) -> None:
    try:
        data = json.loads(shot_json_path.read_text())
        shots = data.get("shots", [])
        if shots:
            expected_last = max(0, int(total_frames) - 1)
            if int(shots[-1]["last_frame"]) != expected_last:
                shots[-1]["last_frame"] = expected_last
                shot_json_path.write_text(json.dumps(data, indent=2))
                print(f"[fix] Extended final shot to last_frame={expected_last} for full coverage.")
    except Exception as e:
        print(f"[warn] Could not adjust shot coverage: {e!r}")

def _device(arg_device: str) -> str:
    if arg_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if arg_device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_manifest_dict_and_frames(manifest: dict, total_frame_count: int | None):
    # Convenience for CLI: let dispatcher read via a temp path
    with NamedTemporaryFile("w+", suffix=".json", delete=True) as tmp:
        tmp.write(json.dumps(manifest))
        tmp.flush()
        return validate_manifest(Path(tmp.name), total_frame_count=total_frame_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track faces and resolve global identities across shots (JSON V2 writer)"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to input video file"
    )
    parser.add_argument(
        "--detector_model",
        default="models/detector/yolov5n_state_dict.pt",
        help="Path to YOLOv5 model weights",
    )
    parser.add_argument(
        "--embedding_model",
        default="models/embedding/glintr100_dynamic.onnx",
        help="Path to ArcFace ONNX model",
    )
    parser.add_argument(
        "--config",
        default="models/detector/yolov5n.yaml",
        help="Path to YOLOv5 model config",
    )
    parser.add_argument(
        "--shot_segmentation",
        default=None,
        help="Path to shot segmentation JSON (optional)",
    )
    parser.add_argument(
        "--output_segment_json", 
        default=None,
        help="Optional path to save segment-only tracks JSON"
    )
    parser.add_argument(
        "--output_global_json",
        nargs="?",           # optional value
        const=True,          # present with no value => True
        default=None,        # absent => None
        help=(
            "Output JSON V2 manifest. "
            "If used with no value, writes '<video>_v2.json'. "
            "If a path is provided, writes there."
        ),
    )
    parser.add_argument(
        "--output_video", 
        nargs="?", 
        const=True, 
        default=None,
        help=(
            "Render video with global + segment IDs. "
            "If used with no value, writes '<video>_global_faceIDs.avi'. "
            "If a path is provided, writes there.")
    )
    parser.add_argument(
        "--detect_interval", 
        type=int, 
        default=30
    )
    parser.add_argument(
        "--embedding_batch_size_max", 
        type=int, 
        default=32
    )
    parser.add_argument(
        "--device", 
        choices=["auto", "cuda", "cpu"], 
        default="auto",
        help="Compute device for detector/embedder (default: auto)"
    )

    # Embedding storage controls
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
              "If omitted, a default '<video>.embeddings.npz' is used."),
    )

    # logging
    parser.add_argument(
        "--log", 
        default="INFO",
        choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"]
    )
    parser.add_argument(
        "--log-file", 
        default=None
    )

    args = parser.parse_args()

    # ---- logging setup -------------------------------------------------------------
    # normalize level (accepts "info", "INFO", etc.)
    lvl = logging._nameToLevel.get(str(getattr(args, "log", "INFO")).upper(), logging.INFO)

    kwargs = {
        "level": lvl,
        "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
        "force": True,           # reconfigure even if someone logged earlier
    }

    log_file = getattr(args, "log_file", None)
    if log_file:                # treat empty string as stdout
        kwargs["filename"] = log_file
    else:
        kwargs["stream"] = sys.stdout

    logging.basicConfig(**kwargs)

    # ---- programmatic setup ---------------------------------------------------------
    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = _device(args.device)

    # Detector / embedder
    yolo = load_yolo5face_model(
        detector_model_path=args.detector_model,
        config_path=args.config,
        device=device,
    )
    detector = FaceDetector(yolo)
    embedder = FaceEmbedder(args.embedding_model, device=device)

    if hasattr(embedder, "set_max_batch_size"):
        embedder.set_max_batch_size(int(args.embedding_batch_size_max))
    elif hasattr(embedder, "max_batch_size"):
        try:
            setattr(embedder, "max_batch_size", int(args.embedding_batch_size_max))
        except Exception:
            pass

    with ExitStack() as stack:
        # Accurate props via PyAV-backed ReaderCoordinator
        fp = stack.enter_context(ReaderCoordinator(str(video_path)))
        fps = float(fp.fps() or 0.0)
        size = fp.size() or (0, 0)
        width, height = int(size[0]), int(size[1])
        total_frames = int(fp.total_frames() or 0)

        # Shots (if not provided, build to a temp file then read back)
        if args.shot_segmentation:
            shot_json_path = str(Path(args.shot_segmentation))
        else:
            # generate_shot_features_json writes a file → make a temp file and pass its path
            with NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            generate_shot_features_json(
                video_path=str(video_path),
                output_json_path=str(tmp_path),
                detector_model_path=args.detector_model,
                config_path=args.config,
            )
            shot_json_path = tmp_path

        _fix_shot_coverage(shot_json_path, total_frames)

        # ---- tracking + identity resolution -----------------------------------
        tracks = track_across_segments(
            frame_source=fp,                 # or str(video_path) if you prefer
            shot_json_path=shot_json_path,   # <<< path, not dict
            detector=detector,
            embedder=embedder,
            detect_interval=int(args.detect_interval),
            embedding_batch_size_max=int(args.embedding_batch_size_max),
            # iou_thresh / embedding_thresh → use defaults unless you want to expose flags
        )
        GlobalIdentityResolver().resolve_global_ids(tracks)

    # Optional: segment JSON (unchanged behavior)
    if args.output_segment_json:
        out_seg = Path(args.output_segment_json)
        out_seg.write_text(json.dumps(
            [t.to_dict_segment() for t in tracks] if tracks and hasattr(tracks[0], "to_dict_segment") else [],
            indent=2
        ))
        print(f"Wrote segment-only tracks to {out_seg}")

    # ---- optional render ---------------------------------------------------
    if args.output_video:
        if args.output_video is True:
            output_video_path = video_path.with_name(f"{video_path.stem}_global_faceIDs.mp4")
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
            video_path=str(video_path),
            output_path=str(output_video_path),
            tracks=tracks,
            label_fmt=label_with_shot_track_face_segment_frame_ids,
        )

    # ---- V2 manifest (authoritative) --------------------------------------
    if args.output_global_json:
        want_global = args.output_global_json not in (None, False)
        if want_global:
            # Decide manifest output path
            out_glob = (
                video_path.with_name(f"{video_path.stem}_v2.json")
                if args.output_global_json is True
                else Path(args.output_global_json)
            )

            # emb_store mapping and sidecar path
            emb_store = None if args.emb_store == "none" else args.emb_store
            sidecar_path = (
                (
                    Path(args.emb_sidecar_path) 
                    if args.emb_sidecar_path 
                    else video_path.with_name(f"{video_path.stem}.embeddings.npz")
                )
                if emb_store == "sidecar"
                else None
            )

            cfg = V2WriterConfig(
                video_path=str(video_path),
                video_size=(width, height),
                fps=fps,
                total_frames=total_frames,
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
                generation=None,
                detector=detector,
                embedder=embedder,
                tracking_params={"detect_interval": int(args.detect_interval)},
                validator=None,
                collector=collector,
            )

            if cfg.emb_store == "sidecar" and collector is not None:
                manifest["embedding_sidecar"] = collector.finalize_sidecar(sidecar_path)

            errs = validate_manifest_dict_and_frames(manifest, total_frames)
            if errs:
                print("Validation errors:")
                for e in errs:
                    print(" -", e)
                raise SystemExit(2)

            write_v2_json(str(out_glob), manifest)
            print(f"Wrote V2 JSON to {out_glob}")
            if "embedding_sidecar" in manifest:
                print(f"Wrote embedding sidecar to {manifest['embedding_sidecar']['path']}")

if __name__ == "__main__":
    main()
