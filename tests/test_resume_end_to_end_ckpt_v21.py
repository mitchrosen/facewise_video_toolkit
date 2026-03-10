import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pytest
from typing import List, Dict

from facekit.pipeline.checkpoint import CheckpointManager, ResumeSafetyError
from facekit.output.json_v2 import (
    ObservationsCollector,
    EmbeddingCollector,
    V2WriterConfig,
    build_v2_1_manifest_from_tracks,
)
from facekit.pipeline.track_across_segments import track_across_segments
from facekit.io.frame_provider import ReaderCoordinator
from facekit.common.obs_consts import Source, src_to_code

@dataclass
class SimpleObs:
    frame_idx: int
    bbox: tuple[float, float, float, float]
    source: str = "detected"
    confidence: float | None = None

@dataclass
class SimpleTrack:
    shot_id: int
    track_id: int
    observations: list[SimpleObs] = field(default_factory=list)
    segment_id: int | None = None
    global_id: int | None = None
    def first_frame(self) -> int:
        return min(o.frame_idx for o in self.observations) if self.observations else 0
    def last_frame(self) -> int:
        return max(o.frame_idx for o in self.observations) if self.observations else -1

def _write_shots_fixed_width(path: Path, first: int, last: int, width: int) -> None:
    """
    Write a shots.json with fixed-width shots (inclusive frame ranges).
    width=10 on a 0..59 video yields shots: [0..9],[10..19],...,[50..59].
    """
    first_i = int(first)
    last_i = int(last)
    w = int(width)
    assert w > 0
    shots: List[Dict[str, int]] = []
    s = first_i
    while s <= last_i:
        e = min(s + w - 1, last_i)
        shots.append({"shot_number": len(shots) + 1, "first_frame": s, "last_frame": e})
        s = e + 1
    path.write_text(json.dumps({"shots": shots}, indent=2))

# --- Tiny fake models -------------------------------------------------------

class CrashAfterNDetections:
    """
    Deterministic detector:
    - returns one detection per call until 'crash_at' (1-based calls), then raises KeyboardInterrupt
    - landmarks are a 5-point dummy list per YOLO-face style.
    """
    def __init__(self, crash_at: int | None):
        self.calls = 0
        self.crash_at = crash_at

    def detect_faces_in_frame(self, frame):
        self.calls += 1
        if self.crash_at is not None and self.calls == self.crash_at:
            raise KeyboardInterrupt("simulated crash")
        # One in-bounds XYXY box based on frame size so OpenCV ROI is valid
        H, W = frame.shape[:2]
        x1, y1 = int(W * 0.10), int(H * 0.10)
        x2, y2 = int(W * 0.30), int(H * 0.40)
        box = np.array([x1, y1, x2, y2, 0.0], dtype=np.float32)
        landmarks = np.array([[120,120],[180,120],[150,150],[130,180],[170,180]], dtype=np.float32)
        conf = 0.9
        return [box], [landmarks], [conf]

class NoCrashDetector(CrashAfterNDetections):
    def __init__(self):
        super().__init__(crash_at=None)

class DummyEmbedder:
    """Return (K,512) zeros float32 for any K aligned faces."""
    def get_embedding_batch(self, aligned_faces, batch_size=32):
        K = len(aligned_faces or [])
        return np.zeros((K, 512), dtype=np.float32)

# --- Shot JSON helper -------------------------------------------------------

def _write_shots(path: Path, first: int, last: int):
    shots = {"shots": [{"shot_number": 1, "first_frame": int(first), "last_frame": int(last)}]}
    path.write_text(json.dumps(shots, indent=2))

# --- Core end-to-end test ---------------------------------------------------

def test_resume_reconstructs_prior_tracks_and_writes_v21_sidecar(tmp_path: Path):
    """
    First run: detect every frame, crash after anchor -> checkpoint exists.
    Expect:
      - returned tracks include obs < anchor and >= anchor
      - obs sidecar count equals collector count (pre + post)
      - embeddings collector kept (sidecar mode) and anchor trim respected
    """
    # Video: create a tiny solid-color MP4 (ReaderCoordinator only needs real file)
    vid = tmp_path / "toy.mp4"
    # Generate a 2s / 60f 192x108 clip (fast) using OpenCV (no audio)
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(vid), fourcc, 30.0, (192,108))
    for _ in range(60):
        vw.write(np.zeros((108,192,3), dtype=np.uint8))
    vw.release()

    shots_path = tmp_path / "shots.json"
    # Force multiple shot boundaries so an aligned-face flush can occur before the crash.
    _write_shots_fixed_width(shots_path, 0, 59, width=10)

    # Checkpoint parent
    parent = tmp_path / "ckpt"
    parent.mkdir()

    # Collectors used across runs (as real code does)
    obsA = ObservationsCollector()
    embA = EmbeddingCollector(mode="sidecar", dim=512)

    # Open run dir (fresh)
    opts = {
        "schema_version": "2.1",
        "video_path": str(vid),
        "detect_interval": 1,     # detect every frame so checkpoint every frame
        "embedding_batch_size_max": 16,
        "device": "cpu",
        "emb_store": "sidecar",
        "emb_sidecar_path": None,
        "obs_sidecar_path": None,
        "detector_model_path": "X",
        "embedding_model_path": "Y",
        "yolo_config_path": "Z",
        "shot_segmentation_path": str(shots_path),
        "log_level": "INFO",
        "log_file": None,
    }
    ckpt = CheckpointManager.open(
        parent_dir=parent, video_path=vid, options_snapshot=opts,
        no_resume=True, force_new_run=False, resume_latest=False
    )
    ckpt.start(obsA, embA, options_snapshot=opts)

    # INITIAL run: crash after we've crossed at least one shot boundary (>= 10).
    # This should allow a flush/anchor to be established once the code is updated.
    det_crash = CrashAfterNDetections(crash_at=30)
    emb = DummyEmbedder()

    with ReaderCoordinator(str(vid)) as fp:
        with pytest.raises(KeyboardInterrupt):
            track_across_segments(
                frame_source=fp,
                shot_json_path=str(shots_path),
                detector=det_crash,
                embedder=emb,
                detect_interval=1,                # detect each frame
                embedding_batch_size_max=16,
                checkpoint=ckpt,
            )

    # After crash, status.json should reflect an anchor (at last detection frame).
    st = ckpt.read_status()
    assert st is not None
    anchor_f = int(st["last_embedding_safe_frame"])
    assert anchor_f >= 0
    # We expect at least 1 observation row
    assert int(st["obs_rows_at_last_embedding_safe"]) > 0

    # Simulate a fresh process: new collectors for the resume run
    obsB = ObservationsCollector()
    embB = EmbeddingCollector(mode="sidecar", dim=512)

    # Reopen selected run with resume
    ckpt2 = CheckpointManager.open(
        parent_dir=parent, video_path=vid, options_snapshot=opts,
        no_resume=False, resume_latest=True, force_new_run=False
    )
    # Enforce resume safety; should pass (same video/options)
    assert ckpt2.validate_resume_or_raise(opts, force=False) is True

    # Hydrate + trim
    loaded_obs, loaded_emb = ckpt2.load_and_anchor_collectors(obsB, embB)
    assert loaded_obs == int(st["obs_rows_at_last_embedding_safe"])
    assert loaded_emb == int(st["emb_rows_at_last_embedding_safe"])

    ckpt2.obs_collector = obsB
    
    # This test creates 6 shots (shot_number 1..6) with one track per shot (track_id=0).
    # Track-order must cover every (shot_number, track_id) present in the persisted obs.
    setattr(ckpt2, "get_track_order", lambda: {(s, 0): 0 for s in range(1, 7)})

    # Resume run (no crash)
    det_ok = NoCrashDetector()
    with ReaderCoordinator(str(vid)) as fp:
        tracks = track_across_segments(
            frame_source=fp,
            shot_json_path=str(shots_path),
            detector=det_ok,
            embedder=emb,
            detect_interval=1,
            embedding_batch_size_max=16,
            checkpoint=ckpt2,
        )

    # Assert combined coverage: there must be obs < anchor and >= anchor across tracks
    all_frames = [o.frame_idx for t in tracks for o in getattr(t, "observations", [])]
    assert any(f < anchor_f for f in all_frames), "Pre-resume frames should be present"
    assert any(f >= anchor_f for f in all_frames), "Post-resume frames should be present"

    # Build a v2.1 manifest using the *live* obs collector (which now has both pre and post)
    cfg = V2WriterConfig(
        video_path=str(vid),
        video_size=(192,108),
        total_frames=60,
        fps=30.0,
        normalize_to_percent=True,
        emb_store="sidecar",
        emb_sidecar_path=tmp_path / "embeddings.npz",
    )
    manifest = build_v2_1_manifest_from_tracks(
        tracks, cfg,
        face_metadata=None,
        generation=None,
        detector=None, embedder=None,
        tracking_params={"detect_interval": 1},
        validator=None,
        emb_collector=embB,          # append-time indices flow into obs rows when created
        obs_collector=obsB,          # this carries all rows (pre+post) after hydration+resume run
    )
    # finalize obs sidecar
    obs_info = obsB.finalize_sidecar(tmp_path / "observations_ckpt_sidecar.npz")
    assert int(obs_info["count"]) == obsB.count()
    assert Path(obs_info["path"]).exists()

def test_iter_tracks_filters_and_groups(tmp_path: Path):
    oc = ObservationsCollector()

    rows = [
        {
            "shot": 1,
            "track_id": 7,
            "f": 5,
            "bbox_xyxy": [0, 0, 10, 10],
            "src": int(src_to_code(Source.DETECTED.value)),
        },
        # TRACKED row: landmarks optional
        {
            "shot": 1,
            "track_id": 7,
            "f": 9,
            "bbox_xyxy": [1, 1, 11, 11],
            "src": int(src_to_code(Source.TRACKED.value)),
        },
        {
            "shot": 1,
            "track_id": 7,
            "f": 12,
            "bbox_xyxy": [2, 2, 12, 12],
            "src": int(src_to_code(Source.DETECTED.value)),
        },
    ]

    oc.append_track_obs(rows, emb_idx_fn=lambda _: -1)

    groups_all = list(oc.iter_tracks(frame_max=None))
    groups_10 = list(oc.iter_tracks(frame_max=10))

    assert len(groups_all) == 1 and groups_all[0][0:2] == (1, 7)
    assert [d["f"] for d in groups_10[0][2]] == [5, 9]

def test_embeddingcollector_get_many():
    ec = EmbeddingCollector(mode="sidecar", dim=4)
    # assign three small vectors
    for v in (np.ones(4), np.arange(4), np.zeros(4)):
        ec.assign(v.astype(np.float32))
    sub = ec.get_many([0, 2])
    assert sub.shape == (2, 4)
    assert np.allclose(sub[0], np.ones(4))

def test_resume_safety_video_mismatch_raises(tmp_path: Path):
    # Minimal status + files in a run dir
    parent = tmp_path / "ck"
    parent.mkdir()
    vidA = tmp_path / "A.mp4"
    vidA.write_bytes(b"not a real video")
    run = CheckpointManager._create_new_run_dir(parent, {"video_path": str(vidA)})
    Path(run, "ckpt").mkdir()
    obs_sidecar = Path(run, "ckpt", "obs_ckpt.npz")
    emb_sidecar = Path(run, "ckpt", "emb_ckpt.npz")

    # Minimal but valid obs sidecar with required key 'observations'
    obs_dtype = np.dtype([
        ("shot", "i4"),
        ("track_id", "i4"),
        ("f", "i4"),
        ("bbox_xyxy", "f4", (4,)),
        ("src", "i4"),
        ("conf", "f4"),
        ("emb_idx", "i4"),
    ])
    np.savez(obs_sidecar, observations=np.zeros(0, dtype=obs_dtype))

    # Minimal but valid emb sidecar with required key 'embeddings'
    np.savez(emb_sidecar, embeddings=np.zeros((0, 512), dtype=np.float32))

    st = {
        "video_path": str(vidA),
        "schema_version": "2.1",
        "detect_interval": 1,
        "embedding_batch_size_max": 16,
        "device": "cpu",
        "emb_store": "sidecar",
        "emb_sidecar_path": None,
        "obs_sidecar_path": None,
        "detector_model_path": "X",
        "embedding_model_path": "Y",
        "yolo_config_path": "Z",
        "shot_segmentation_path": None,
        "checkpoint_dir": str(run),
        "log_level": "INFO",
        "log_file": None,
        "last_embedding_safe_frame": 0,
        "last_embedding_safe_shot_number": 1,
        "last_embedding_safe_shot_first_frame": 0,
        "obs_rows_at_last_embedding_safe": 0,
        "emb_rows_at_last_embedding_safe": 0,
        "frames_done": 0, "shots_done": 0, "tracks_seen": 0,
        "obs_rows": 0, "emb_rows": 0,
        "last_saved_utc": "now", "note": "opened",
        "track_order": [{"shot": 1, "track_id": 0, "order": 0}],
    }
    Path(run, "status.json").write_text(json.dumps(st, indent=2))

    # Try to open with a different video path
    vidB = tmp_path / "B.mp4"
    vidB.write_bytes(b"not a real video")

    mgr = CheckpointManager.open(
        parent_dir=parent, video_path=vidB, options_snapshot={"video_path": str(vidB)}, no_resume=False, resume_latest=True
    )
    with pytest.raises(ResumeSafetyError):
        mgr.validate_resume_or_raise({"video_path": str(vidB), "schema_version": "2.1"}, force=False)

def test_resume_safety_schema_mismatch_raises(tmp_path: Path):
    parent = Path(tmp_path, "ck2")
    parent.mkdir()
    vid = Path(tmp_path, "v.mp4")
    vid.write_bytes(b"v")
    run = CheckpointManager._create_new_run_dir(parent, {"video_path": str(vid)})
    Path(run, "ckpt").mkdir()

    obs_sidecar = Path(run, "ckpt", "obs_ckpt.npz")
    emb_sidecar = Path(run, "ckpt", "emb_ckpt.npz")

    obs_dtype = np.dtype([
        ("shot", "i4"),
        ("track_id", "i4"),
        ("f", "i4"),
        ("bbox_xyxy", "f4", (4,)),
        ("src", "i4"),
        ("conf", "f4"),
        ("emb_idx", "i4"),
    ])
    np.savez(obs_sidecar, observations=np.zeros(0, dtype=obs_dtype))
    np.savez(emb_sidecar, embeddings=np.zeros((0, 512), dtype=np.float32))

    st = {"video_path": str(vid), "schema_version": "2.0",
              "track_order": [{"shot": 1, "track_id": 0, "order": 0}]}  # <- mismatch
    Path(run, "status.json").write_text(json.dumps(st))
    mgr = CheckpointManager.open(parent_dir=parent, video_path=vid, options_snapshot={"video_path": str(vid)}, no_resume=False, resume_latest=True)
    with pytest.raises(ResumeSafetyError):
        mgr.validate_resume_or_raise({"video_path": str(vid), "schema_version": "2.1"}, force=False)
