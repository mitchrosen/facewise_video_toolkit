import argparse
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from contextlib import ExitStack
import torch
import os
import sys
from typing import Literal, List, Optional
from jsonschema import ValidationError
from dataclasses import dataclass, field
import logging

from facekit.io.frame_provider import ReaderCoordinator
from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.detection.face_detector import FaceDetector
from facekit.embedding.embedder import FaceEmbedder
from facekit.pipeline.draw_tracks import draw_tracks_on_video
from facekit.pipeline.generate_shot_features import generate_shot_features_json
import facekit.pipeline.track_across_segments as track_across_segments
from facekit.pipeline.checkpoint import CheckpointManager
from facekit.tracking.tracking_resolution import GlobalIdentityResolver
from facekit.tracking.face_structures import FaceTrack
from facekit.validation import validate_manifest
from facekit.output.json_v2 import (
    V2WriterConfig,
    build_v2_manifest_from_tracks,
    build_v2_1_manifest_from_tracks,
    fix_shots,
    derive_face_metadata_from_observations,
    write_v2_json,
    EmbeddingCollector,
    ObservationsCollector
)
from facekit.errors import ResumeSafetyError

@dataclass
class SimpleObs:
    frame_idx: int
    bbox: tuple[float, float, float, float]
    source: str = "detected"
    confidence: Optional[float] = None

@dataclass
class SimpleTrack:
    shot_id: int
    track_id: int
    observations: List[SimpleObs] = field(default_factory=list)
    segment_id: Optional[int] = None
    global_id: Optional[int] = None
    def first_frame(self) -> int:
        return min(o.frame_idx for o in self.observations) if self.observations else 0
    def last_frame(self) -> int:
        return max(o.frame_idx for o in self.observations) if self.observations else -1

def _fix_shot_coverage(shot_json_path: str | Path, total_frames: int) -> None:
    p = Path(shot_json_path)
    try:
        data = json.loads(p.read_text())
        shots = data.get("shots", [])
        if shots:
            expected_last = max(0, int(total_frames) - 1)
            if int(shots[-1]["last_frame"]) != expected_last:
                shots[-1]["last_frame"] = expected_last
                p.write_text(json.dumps(data, indent=2))
                logging.info(f"[fix] Extended final shot to last_frame={expected_last} for full coverage.")
    except Exception as e:
        logging.warning(f"[warn] Could not adjust shot coverage: {e!r}")

def _device(arg_device: str) -> str:
    if arg_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if arg_device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _atomic_copy(src: Path, dst: Path) -> None:
    dst = Path(dst)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)

def _validate_manifest_dict(
        manifest: dict, 
        schema_version: Literal["2.0", "2.1"],
        total_frame_count: int | None,
        schema_dir: str | None = None) -> list[str]:
    # Convenience for CLI: let dispatcher read via a temp path
    with NamedTemporaryFile("w+", suffix=".json", delete=True) as tmp:
        tmp.write(json.dumps(manifest))
        tmp.flush()
        return validate_manifest(Path(tmp.name), 
                                 schema_version=schema_version, 
                                 total_frame_count=total_frame_count,
                                 schema_dir=schema_dir,
                                 )

def run_pipeline(args):
    # ---- forbid inline embeddings for 2.1 ----
    if args.schema_version == "2.1" and args.emb_store == "inline":
        raise ResumeSafetyError("Schema 2.1 does not support inline embeddings. Use --emb-store sidecar or none.")

    # ---- resume flag sanity -------------------------------------------------------
    # Disallow obviously conflicting flags early (clear user signal => clear behavior).
    if args.no_resume and (args.resume_latest or args.checkpoint_run_id):
        raise ResumeSafetyError("--no-resume cannot be combined with --resume-latest/--checkpoint-run-id")
    if args.new_run and (args.resume_latest or args.checkpoint_run_id):
        raise ResumeSafetyError("--new-run cannot be combined with --resume-latest/--checkpoint-run-id")

    # ---- logging setup -------------------------------------------------------------
    # normalize level (accepts "info", "INFO", etc.)
    lvl = logging._nameToLevel.get(str(getattr(args, "log", "INFO")).upper(), logging.INFO)
    kwargs = {
        "level": lvl,
        "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
        "force": True,           # reconfigure even if someone logged earlier
    }
    if args.log_file:
        kwargs["filename"] = args.log_file
    else:
        kwargs["stream"] = sys.stdout
    logging.basicConfig(**kwargs)

    # ---- programmatic setup ---------------------------------------------------------
    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = _device(args.device)

    with ExitStack() as stack:
        # Accurate props via PyAV-backed ReaderCoordinator
        fp = stack.enter_context(ReaderCoordinator(str(video_path)))
        fps = float(fp.fps() or 0.0)
        size = fp.size() or (0, 0)
        width, height = int(size[0]), int(size[1])
        total_frames = int(fp.total_frames() or 0)

        # Shots (if not provided, build to a temp file then read back)
        if args.shot_segmentation:
            shot_json_path = Path(args.shot_segmentation)
        else:
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

        # Read the full shot definitions *once*; this is the authoritative shot list.
        shot_data = json.loads(shot_json_path.read_text())
        shot_defs = shot_data.get("shots", [])

        # Detector / embedder
        yolo = load_yolo5face_model(args.detector_model, args.config, device=device)
        detector = FaceDetector(yolo)
        embedder = FaceEmbedder(args.embedding_model, device=device)
        if hasattr(embedder, "set_max_batch_size"):
            embedder.set_max_batch_size(int(args.embedding_batch_size_max))

        # ---- Checkpointing setup -----------------------------------------------
        # pick parent dir (stable for a video)
        parent_ckpt_dir = (
            Path(args.checkpoint_dir)
            if args.checkpoint_dir
            else CheckpointManager.compute_parent_dir(Path("checkpoints"), video_path)
        )
        # ---- collectors ---------------------------------------------------------
        # Single source of truth for the whole run (fresh or resume):
        # These are used by checkpointing during the run AND by the final writers.
        obs_collector = ObservationsCollector()
        emb_collector = EmbeddingCollector(mode="sidecar", dim=512, base_offset=0)

        options_snapshot = {
            "schema_version": args.schema_version,
            "video_path": str(video_path.resolve()),
            "detect_interval": int(args.detect_interval),
            "embedding_batch_size_max": int(args.embedding_batch_size_max),
            "device": args.device,
            "emb_store": ("none" if args.emb_store == "none" else args.emb_store),
            "emb_sidecar_path": (str(args.emb_sidecar_path) if args.emb_sidecar_path else None),
            "obs_sidecar_path": (str(args.obs_sidecar_path) if args.obs_sidecar_path else None),
            "detector_model_path": str(args.detector_model),
            "embedding_model_path": str(args.embedding_model),
            "yolo_config_path": str(args.config),
            "shot_segmentation_path": (str(args.shot_segmentation) if args.shot_segmentation else None),
            "log_level": args.log,
            "log_file": args.log_file,
        }

        ckpt = CheckpointManager.open(
            parent_dir=parent_ckpt_dir,
            video_path=video_path,
            options_snapshot=options_snapshot,
            no_resume=args.no_resume,
            force_new_run=bool(args.new_run),
            run_id=args.checkpoint_run_id,
            resume_latest=bool(args.resume_latest),
        )

        # Summarize selected run + resume intent
        status = ckpt.read_status() or {}
        logging.info(
            "Checkpoint selection: dir=%s | resume_enabled=%s | run_id=%s | resume_latest=%s | new_run=%s",
            ckpt.root, bool(ckpt.resume_enabled), args.checkpoint_run_id or "-", bool(args.resume_latest), bool(args.new_run)
        )
        logging.info("status.json: last_detection_frame=%s last_detection_shot=%s",
                     status.get("last_detection_frame"), status.get("last_detection_shot"))

        # If resume is possible, load status and *enforce* immutable params unless --force

        if not args.no_resume:
            ckpt.validate_resume_or_raise(
                options_snapshot,
                force=args.force,
            )          

        ckpt.start(
            obs_collector,
            emb_collector, 
            frames_done=0, 
            shots_done=0, 
            tracks_seen=0,
            options_snapshot=options_snapshot)
        
        # Hydrate + anchor only on a true resume (manager says resume is enabled AND anchor available)
        anchor_summary = None
        resume_anchor_f: Optional[int] = None
        if (not args.no_resume) and bool(ckpt.resume_enabled):
            anchor_summary = ckpt.rehydrate_runtime(obs_collector, emb_collector, trim_to_anchor=True)
            af = anchor_summary.get("anchor_frame")
            ashot = anchor_summary.get("anchor_shot")
            ashot_first = anchor_summary.get("anchor_shot_first_frame")
            resume_anchor_f = af if isinstance(af, int) else None
            logging.info("resume: hydrated collectors; anchor_frame=%r anchor_shot=%r anchor_shot_first=%r",
                         af, ashot, ashot_first)
            logging.info("resume: stable segment labeling enabled (track_order entries=%d)",
                         int(anchor_summary.get("track_order_entries", 0)))
            # Emit an easily parsable line for tests:
            print(f"ANCHOR:{int(resume_anchor_f or 0)}", flush=True)
        else:
            logging.info("resume: disabled or no prior state; starting cold.")
    
        # ---- tracking with progress hooks --------------------------------
        _resume_enabled = (not args.no_resume) and bool(ckpt.resume_enabled)
        logging.info("tracker: resume_enabled=%s (no_resume=%s, ckpt.resume_enabled=%s)",
                     _resume_enabled, bool(args.no_resume), bool(ckpt.resume_enabled))
        
        tracks = track_across_segments.track_across_segments(
            frame_source=fp,
            shot_json_path=str(shot_json_path),
            detector=detector,
            embedder=embedder,
            detect_interval=int(args.detect_interval),
            embedding_batch_size_max=int(args.embedding_batch_size_max),
            checkpoint=ckpt,
            resume_enabled=_resume_enabled,
        )

        # --- Global identity resolution (always run) ---
        if not tracks:
            logging.info("global-id: no tracks produced; skipping resolver.")
        else:
            # Warn (but proceed) if any track is missing embeddings.
            def _has_emb(t) -> bool:
                embs = getattr(t, "embeddings", None)
                return bool(embs) and len(embs) > 0
            total = len(tracks)
            missing = sum(1 for t in tracks if not _has_emb(t))
            if missing:
                logging.warning(
                    "global-id: proceeding with missing embeddings on %d/%d tracks "
                    "(these will be assigned unique IDs deterministically).",
                    missing, total
                )
            resolver = GlobalIdentityResolver()
            next_gid = resolver.resolve_global_ids(tracks, start_id=0)
            logging.info("global-id: assignment complete (next_gid=%d)", next_gid)

        ckpt.finalize()

    # Optional: segment JSON (unchanged behavior)
    if args.output_segment_json:
        out_seg = Path(args.output_segment_json)
        payload = [t.to_dict_segment() for t in tracks] if tracks and hasattr(tracks[0], "to_dict_segment") else []
        out_seg.write_text(json.dumps(payload, indent=2))
        logging.info(f"Wrote segment-only tracks to {out_seg}")

    # ---- Optional overlay render ---------------------------------------------------
    if args.output_video:
        if args.output_video is True:
            output_video_path = video_path.with_name(f"{video_path.stem}_global_faceIDs.avi")
        else:
            output_video_path = Path(args.output_video)

        def label_fmt(track: FaceTrack, frame_num: int) -> str:
            def q(x): return "?" if x is None else x
            sid = q(getattr(track, "shot_id", None))
            tid = q(getattr(track, "track_id", None))
            gid = q(getattr(track, "global_id", None))
            seg = q(getattr(track, "segment_id", None))
            return f"ShotID{sid}_TrackID{tid}_FaceSegID{seg}_FaceGlobID{gid}_Frame#{frame_num}"

        logging.info(f"Rendering labeled video to {output_video_path}")
        draw_tracks_on_video(
            video_path=str(video_path),
            output_path=str(output_video_path),
            tracks=tracks,
            label_fmt=label_fmt,
        )

    # ---- V2 manifest (authoritative) --------------------------------------
    if args.output_global_json:
        # choose default filename based on schema when flag has no path
        if args.output_global_json is True:
            out_glob = video_path.with_name(
                f"{video_path.stem}_{'v2_1' if args.schema_version=='2.1' else 'v2'}.json"
            )
        else:
            out_glob = Path(args.output_global_json)

        emb_store = None if args.emb_store == "none" else args.emb_store

        # resolve embedding sidecar path (if needed)
        sidecar_path = (
            (Path(args.emb_sidecar_path) if args.emb_sidecar_path else video_path.with_suffix(".embeddings.npz"))
            if emb_store == "sidecar" else None
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
        face_meta = derive_face_metadata_from_observations(obs_collector, tracks)


        if args.schema_version == "2.0":
            writer_collector: EmbeddingCollector | None = None
            if emb_store == "inline":
                writer_collector = EmbeddingCollector(mode="inline", dim=512)
            elif emb_store == "sidecar":
                # Reuse the runtime one so indices match what you already assigned during tracking
                writer_collector = emb_collector
            else:  # none
                writer_collector = None

            manifest = build_v2_manifest_from_tracks(
                tracks,
                cfg,
                face_metadata=face_meta,
                generation=None,
                detector=detector,
                embedder=embedder,
                tracking_params={"detect_interval": int(args.detect_interval)},
                validator=None,
                collector=writer_collector,
            )

        else:  # 2.1
            # default observations sidecar path
            obs_path = Path(args.obs_sidecar_path) if args.obs_sidecar_path else video_path.with_suffix(".observations.npz")

            manifest = build_v2_1_manifest_from_tracks(
                tracks, 
                cfg,
                face_metadata=face_meta,
                generation=None,
                detector=detector,
                embedder=embedder,
                tracking_params={"detect_interval": int(args.detect_interval)},
                validator=None,
                emb_collector=(emb_collector if emb_store == "sidecar" else None),
                obs_collector=obs_collector, 
            )

            fix_shots(manifest, shot_defs)

            # finalize observations sidecar
            manifest["observations_sidecar"] = obs_collector.finalize_sidecar(
                obs_path,
                # If we resumed, only persist rows strictly after the anchor frame.
                min_frame_exclusive=None,
            )

            # Ensure manifest has at least one shot, even if no faces/tracks were found.
            shots = manifest.get("shots")
            if not shots:
                # Fallback: single shot covering the whole video.
                # If total_frames is unknown/0, clamp to [0, 0].
                last = max(total_frames - 1, 0)
                manifest["shots"] = [{
                    "shot_number": 1,
                    "first_frame": 0,
                    "last_frame": last,
                    "num_tracks": 0,
                    "face_tracks": []
                }]        

        # ---- finalize embedding sidecar  ---------------------
        if emb_store == "sidecar":
            epath = cfg.emb_sidecar_path or video_path.with_suffix(".embeddings.npz")
            manifest["embedding_sidecar"] = emb_collector.finalize_sidecar(epath)

        errs = _validate_manifest_dict(
            manifest,
            schema_version=args.schema_version,
            total_frame_count=total_frames,
            schema_dir=args.schema_dir,
        )
        if errs:
            logging.error("Validation errors:")
            for e in errs:
                logging.error(" - %s", e)
            raise SystemExit(2)

        write_v2_json(str(out_glob), manifest)
        logging.info(f"Wrote V2 JSON to {out_glob}")
        if "embedding_sidecar" in manifest:
            logging.info(f"Wrote embedding sidecar to {manifest['embedding_sidecar']['path']}")
        
        if ckpt is not None:
            # Copy the live checkpoint sidecars to the user-requested paths.
            obs_out = getattr(args, "obs_sidecar_path", None)
            # Leave here for future support of embedding sidecar
            emb_out = None

            if ckpt is not None:
                ckpt.copy_ckpt_sidecars_to_final(
                    obs_sidecar_path=None,   # <-- prevent overwrite
                    emb_sidecar_path=None
                )
                ckpt.mark_completed()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track faces and resolve global identities across shots (JSON V2 writer)"
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--detector-model", default="models/detector/yolov5n_state_dict.pt", 
                        help="Path to YOLOv5 model weights",)
    parser.add_argument("--embedding-model", default="models/embedding/glintr100_dynamic.onnx",
                        help="Path to ArcFace ONNX model",)
    parser.add_argument("--config", default="models/detector/yolov5n.yaml",
                        help="Path to YOLOv5 model config",)
    parser.add_argument("--shot-segmentation", default=None, help="Path to shot segmentation JSON (optional)",)
    parser.add_argument("--schema-version", default="2.0", choices=["2.0","2.1"],
                        help = "Manifest schema version to write (default: 2.0).")
    parser.add_argument("--schema-dir", default=None,
                        help = "Directory containing schema JSON files. "
                        "If omitted, uses FACEKIT_SCHEMA_DIR or the package default.")
    parser.add_argument("--output-segment-json", default=None, 
                        help="Optional path to save segment-only tracks JSON")
    parser.add_argument("--output-global-json", nargs="?", const=True, default=None,
                        help=("Output JSON manifest. Flag only: writes '<video>_v2.json' for --schema_version=2.0 "
                                "or '<video>_v2_1.json' for 2.1. If a path is provided, writes there."))
    parser.add_argument("--output-video", nargs="?", const=True, default=None,
                        help=("Render video with global + segment IDs. "
                                "If used with no value, writes '<video>_global_faceIDs.avi'. "
                                "If a path is provided, writes there."))
    parser.add_argument("--detect-interval", type=int, default=30,
                        help="frame count between forced face detection frame")
    parser.add_argument("--embedding-batch-size-max", type=int, default=32)
    parser.add_argument("--device",  choices=["auto", "cuda", "cpu"], default="auto",
                        help="Compute device for detector/embedder (default: auto)")
    # Storage controls
    parser.add_argument("--emb-store", choices=["inline", "sidecar", "none"], default="inline",
                        help="How to serialize embeddings in V2 manifest (default: inline).",)
    parser.add_argument("--emb-sidecar-path", default=None,
                        help=("Path to sidecar embeddings file (only used when --emb-store=sidecar). "
                                "If omitted, a default '<video>.embeddings.npz' is used."),)
    parser.add_argument("--obs-sidecar-path", default=None,
                        help=("Path to sidecar observations file (2.1 only). "
                                "If omitted, a default '<video>.observations.npz' is used."))
    # Checkpointing
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory to place checkpoint sidecars + status.json (default: ./checkpoints/<video_stem>__<hash>/)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume from checkpoint dir even if present.")
    parser.add_argument("--force", action="store_true",
                        help="Allow parameter overrides on resume that would normally be rejected.")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Resume from the latest run inside the checkpoint parent dir (if any).")
    parser.add_argument("--checkpoint-run-id", default=None,
                        help="Resume from a specific run id (e.g., run-20250101T010203Z-deadbeef).")
    parser.add_argument("--new-run", action="store_true",
                        help="Force creation of a fresh run directory even if previous runs exist.")    # logging
    parser.add_argument("--log", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    parser.add_argument("--log-file", default=None)

    args = parser.parse_args()

    try:
        run_pipeline(args)
    except ResumeSafetyError as e:
        logging.error(str(e))
        raise SystemExit(2)           # user/config error during resume
    except ValidationError as e:
        logging.error(str(e))
        raise SystemExit(2)           # schema/manifest invalid -> user-facing input
    except Exception as e:
        logging.exception("Unhandled error:")
        raise SystemExit(1)           # real failure/bug

if __name__ == "__main__":
    main()
