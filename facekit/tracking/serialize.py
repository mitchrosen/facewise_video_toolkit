import json
from pathlib import Path
import numpy as np
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.common.obs_consts import Source

def to_v2_manifest(
    tracks,
    *,
    video_path: str,
    video_size: tuple[int, int],
    fps: float,
    total_frames: int,
    face_metadata: list[dict] | None = None,
    generation_overrides: dict | None = None,
) -> dict:
    """
    Build the canonical **JSON V2** manifest from in-memory tracking results.

    This is a thin adapter that delegates to the public writer in
    `facekit.output.json_v2` so there is a **single source of truth** for the
    file format. Use this any time you want to publish/export results that
    downstream tools will consume.

    What it does
    ------------
    - Groups `tracks` by `track.shot_id` into `shots[]`.
    - Emits each track with:
        * `first_frame` / `last_frame` (absolute frame indices)
        * `face_label` (prefers `global_id`, then `segment_id`, else `track_id`)
        * `avg_center_x`, `avg_center_y`, `avg_face_width`, `avg_face_height`
          normalized to **percent [0,100]** with 2-decimal rounding
        * `is_static` heuristic (low center variance ⇒ True)
        * `obs[]` per observation with:
            - `f` (frame index)
            - `bbox_xyxy` (x1,y1,x2,y2)
            - `src` in {"detected","tracked","flow"}  (spelled-out)
            - optional `conf` if present on the observation
    - Adds top-level `video` (path/fps/size/total_frames).
    - Adds top-level `generation`:
        * auto-fills `created_utc`, `commit`, `branch`, `emb_store` (default "inline")
        * computes stable `params_hash` (sha256 over generation minus `created_utc`)
        * merges any keys from `generation_overrides` (e.g., detector/embedder/tracking/validator).
    - Optionally adds `face_metadata` (e.g., `{"face_label": "face_0", "occurance_count": 27}`).

    Guarantees
    ----------
    - Output conforms to `schemas/shot_features_v2.0.schema.json`
      (use the validator at `facekit.validation.validate_shot_features_v2`
      for enforcement in tests/CI).
    - Normalization uses video_size you provide; be sure it matches the frames
      used during detection/tracking.
    - Embeddings are **not** included in JSON V2 (provenance only via `generation`).

    Parameters
    ----------
    tracks : list[FaceTrack]
        Sequence of finished tracks; each item must expose:
        - `shot_id`, `observations` (with `.frame_idx`, `.bbox`, optional `.source`, `.confidence`)
        - optional `global_id` / `segment_id` / `track_id`
        - optional `first_frame()` / `last_frame()` helpers
    video_path : str
        Source video path written to `video.path`.
    video_size : (int, int)
        (width, height) used for normalization and emitted in `video.size`.
    fps : float
        Video frames-per-second written to `video.fps`.
    total_frames : int
        Total frame count written to `video.total_frames`.
    face_metadata : list[dict], optional
        Precomputed face metadata array; if omitted you can build one with
        `facekit.output.json_v2.derive_face_metadata(tracks)`.
    generation_overrides : dict, optional
        Any fields to merge into `generation` (e.g., detector/embedder/tracking/validator,
        or explicit `commit`, `branch`, `emb_store`). Missing required fields are backfilled.

    Returns
    -------
    dict
        The JSON-serializable manifest (use `write_v2_json(path, manifest)` to publish).

    Example
    -------
    >>> manifest = to_v2_manifest(
    ...     tracks,
    ...     video_path="/data/input.mp4",
    ...     video_size=(1920,1080),
    ...     fps=29.97,
    ...     total_frames=12345,
    ...     face_metadata=derive_face_metadata(tracks),
    ...     generation_overrides={
    ...         "detector": {"name": "yolov5s-face", "weights": "...", "config": "..."},
    ...         "embedder": {"name": "arcface-r50", "dim": 512},
    ...         "tracking": {"tracker": "CSRT", "detect_interval": 10},
    ...         "validator": {"iou": 0.5, "area_delta": 0.5, "asp_delta": 0.5, "v_max": 0.8, "hsv_thresh": 0.35},
    ...         "emb_store": "inline",
    ...     },
    ... )
    >>> from facekit.output.json_v2 import write_v2_json
    >>> write_v2_json("out/video.v2.json", manifest)

    See also
    --------
    - facekit.output.json_v2.V2WriterConfig
    - facekit.output.json_v2.build_v2_manifest_from_tracks
    - facekit.output.json_v2.derive_face_metadata
    - facekit.validation.validate_shot_features_v2
    """
    # Import inside to avoid import cycles.
    from facekit.output.json_v2 import V2WriterConfig, build_v2_manifest_from_tracks

    cfg = V2WriterConfig(
        video_path=video_path,
        video_size=video_size,
        fps=fps,
        total_frames=total_frames,
        normalize_to_percent=True,
    )
    return build_v2_manifest_from_tracks(
        tracks,
        cfg,
        face_metadata=face_metadata,
        generation=generation_overrides,
    )


def tracks_to_json_dict(tracks, include_embeddings=False):
    """
    Convert a list of FaceTrack objects into a JSON-serializable dictionary.

    Args:
        tracks (List[FaceTrack]): List of face tracks
        include_embeddings (bool): If True, include embedding vectors in output

    Returns:
        dict: JSON-compatible dictionary
    """
    out_tracks = []
    for track in tracks:
        obs_rows = []
        for idx, obs in enumerate(track.observations or []):
            # Prefer the true enum if present; otherwise infer legacy default: first=detected, rest=tracked.
            if hasattr(obs, "source") and isinstance(obs.source, Source):
                src_str = obs.source.value
            else:
                src_str = Source.DETECTED.value if idx == 0 else Source.TRACKED.value

            row = {
                "frame_idx": int(obs.frame_idx),
                "bbox": list(obs.bbox) if obs.bbox is not None else None,
                "confidence": obs.confidence,
                "src": src_str,
            }
            if include_embeddings and getattr(obs, "embedding", None) is not None:
                row["embedding"] = obs.embedding.tolist()
            obs_rows.append(row)

        out_tracks.append({
            "shot_id": track.shot_id,
            "track_id": track.track_id,
            "face_label": getattr(track, "global_id", None),
            "observations": obs_rows,
        })

    return {"tracks": out_tracks}


def load_tracks_from_json_dict(json_dict):
    """
    Convert a JSON-compatible dictionary into a list of FaceTrack objects.

    Args:
        json_dict (dict): Dictionary produced by tracks_to_json_dict()

    Returns:
        List[FaceTrack]: List of FaceTrack instances
    """
    tracks = []
    for t in json_dict.get("tracks", []):
        shot_id = t["shot_id"]
        track_id = t["track_id"]
        observations = []
        for idx, obs in enumerate(t.get("observations", [])):
            # Prefer stored string "src" (detected/tracked/flow/fallback) → enum; else infer legacy default.
            if "src" in obs and obs["src"] is not None:
                try:
                    src_enum = Source(str(obs["src"]).lower())
                except ValueError as e:
                    raise ValueError(f"Unknown observation src {obs['src']!r} for track {track_id} at index {idx}") from e
            else:
                src_enum = Source.DETECTED if idx == 0 else Source.TRACKED

            bbox = tuple(obs["bbox"]) if obs.get("bbox") is not None else None
            emb = np.array(obs["embedding"]) if "embedding" in obs else None

            observations.append(FaceObservation(
                frame_idx=int(obs["frame_idx"]),
                bbox=bbox,
                confidence=obs.get("confidence", 1.0),
                embedding=emb,
                source=src_enum,   # <-- enforce source here
            ))

        tracks.append(FaceTrack(shot_id=shot_id, track_id=track_id, observations=observations))
    return tracks

def save_tracks_to_json_file(tracks, output_path, include_embeddings=False):
    """
    Save tracks to a JSON file.

    Args:
        tracks (List[FaceTrack])
        output_path (str or Path): Where to save the file
        include_embeddings (bool): Whether to include embeddings in output
    """
    data = tracks_to_json_dict(tracks, include_embeddings=include_embeddings)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_tracks_from_json_file(json_path):
    """
    Load FaceTrack objects from a JSON file.

    Args:
        json_path (str or Path): Path to the JSON file

    Returns:
        List[FaceTrack]: List of FaceTrack instances
    """
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
    return load_tracks_from_json_dict(data)
