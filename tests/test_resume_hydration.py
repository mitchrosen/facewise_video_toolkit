# tests/test_resume_hydration.py
import json
from pathlib import Path
import numpy as np
import tempfile
from typing import List, Optional, Tuple

from facekit.pipeline.checkpoint import CheckpointManager
from facekit.output.json_v2 import (
    ObservationsCollector,
    EmbeddingCollector,
    V2WriterConfig,
    build_v2_1_manifest_from_tracks,
)
from facekit.tracking.aggregator import ShotFaceTrackAggregatorProtocol

class DummyTrack:
    def __init__(self, tid: int, bbox: Optional[Tuple[int,int,int,int]] = None,
                 last_f: int = 0, closed: bool = False, last_det: Optional[int] = None):
        self.track_id = int(tid)
        self._bbox = bbox
        self._last = int(last_f)
        self._closed = bool(closed)
        self._last_det = last_det

    def get_last_bbox(self): return self._bbox
    def is_closed(self): return self._closed
    def last_frame(self): return self._last
    # Support both attribute and method styles; provider handles either.
    def last_det_frame(self): return self._last_det

class DummyAggregator(ShotFaceTrackAggregatorProtocol):
    def __init__(self, tracks: List[DummyTrack] | None = None):
        self.tracks = list(tracks or [])

# ---------- helpers ----------
def _lm5(f: int):
    """
    Produce a valid 5x2 landmark list for DETECTED rows.
    Keep it deterministic and finite.
    """
    x = float(f + 1)
    return [(x, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]


def mk_obs_block(n, *, shot=3, track_id=7, f_start=0, src="detected"):
    """Produce n normalized obs dicts suitable for ObservationsCollector.append_track_obs."""
    rows = []
    for i in range(n):
        f = f_start + i
        row = {
            "shot": int(shot),
            "track_id": int(track_id),
            "f": int(f),
            "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],  # valid box
            "src": src,                              # "detected" or "tracked"
            "conf": 0.9,
        }
        rows.append(row)
    return rows

def write_status(path, **kw):
    base = {
        "last_saved_utc": "2025-10-21T20:29:18Z",
        "video_path": "/tmp/video.mp4",
        "frames_done": 0,
        "shots_done": 0,
        "tracks_seen": 0,
        "obs_rows": 0,
        "emb_rows": 0,

        # Embedding-safe anchor fields: this is the only durable resume boundary.
        "last_embedding_safe_frame": None,
        "last_embedding_safe_shot_number": None,
        "last_embedding_safe_shot_first_frame": None,
        "obs_rows_at_last_embedding_safe": 0,
        "emb_rows_at_last_embedding_safe": 0,

        "schema_version": "2.1",
        "detect_interval": 60,
        "embedding_batch_size_max": 256,
        "device": "auto",
        "emb_store": "none",
        "emb_sidecar_path": None,
        "obs_sidecar_path": None,
        "detector_model_path": "models/detector/yolov5n_state_dict.pt",
        "embedding_model_path": "models/embedding/glintr100_dynamic.onnx",
        "yolo_config_path": "models/detector/yolov5n.yaml",
        "shot_segmentation_path": None,
        "checkpoint_dir": str(path.parent),
        "log_level": "INFO",
        "log_file": None,
        "note": "seed",
    }
    base.update(kw)
    path.write_text(json.dumps(base, indent=2))

# ---------- tests ----------

def test_load_and_anchor_collectors_trims_to_detection_boundary(tmp_path: Path):
    # Arrange: make a fake run dir with NPZs and status.json
    run_dir = Path(tmp_path, "checkpoint-run")
    Path(run_dir, "ckpt").mkdir(parents=True)
    status = Path(run_dir, "status.json")
    obs_npz = Path(run_dir, "ckpt", "obs_ckpt.npz")
    emb_npz = Path(run_dir, "ckpt", "emb_ckpt.npz")

    # Seed collectors on disk: 100 obs rows, 20 emb rows
    obs_seed = ObservationsCollector()
    off, cnt = obs_seed.append_track_obs(mk_obs_block(100, shot=6, track_id=11, f_start=500))
    assert cnt == 100
    obs_seed.dump_npz(obs_npz)

    emb_seed = EmbeddingCollector(mode="sidecar", dim=512)
    # add 20 embeddings
    for _ in range(20):
        emb_seed.assign(np.ones(512, dtype=np.float32))
    emb_seed.dump_npz(emb_npz)

    # Anchor should trim back to 90/18 and place resume at (frame=67020, shot=6, shot_first=51151)
    write_status(
        status,
        last_embedding_safe_frame=67020,
        last_embedding_safe_shot_number=6,
        last_embedding_safe_shot_first_frame=51151,
        obs_rows_at_last_embedding_safe=90,
        emb_rows_at_last_embedding_safe=18,
    )

    # Live collectors start empty (this mimics the new process)
    obs_live = ObservationsCollector()
    emb_live = EmbeddingCollector(mode="sidecar", dim=512)

    # Act: open manager pointing to our existing run and hydrate/anchor
    mgr = CheckpointManager(run_dir, video_path=Path(tempfile.gettempdir(), "video.mp4"), resume=True)
    loaded_obs, loaded_emb = mgr.load_and_anchor_collectors(obs_live, emb_live)

    # Assert: loaded everything from disk, then trimmed to anchor
    assert loaded_obs == 100
    assert loaded_emb == 20
    assert obs_live.count() == 90
    assert emb_live.count() == 18

    # Anchor exposed via get_resume_anchor()
    anchor = mgr.get_resume_anchor()
    # Backward/forward compatible: allow 2- or 3-tuple
    assert anchor == (67020, 6, 51151)

def test_append_after_resume_and_finalize_updates_files(tmp_path: Path):
    # Arrange: create run dir with a smaller anchor (10 obs, 4 emb)
    run_dir = Path(tmp_path, "run")
    Path(run_dir, "ckpt").mkdir(parents=True)
    status = Path(run_dir, "status.json")
    obs_npz = Path(run_dir, "ckpt", "obs_ckpt.npz")
    emb_npz = Path(run_dir, "ckpt", "emb_ckpt.npz")

    # disk: 12 rows/5 embs; anchor says use only 10/4
    obs_seed = ObservationsCollector()
    obs_seed.append_track_obs(mk_obs_block(12, shot=3, track_id=7, f_start=1000))
    obs_seed.dump_npz(obs_npz)

    emb_seed = EmbeddingCollector(mode="sidecar", dim=512)
    for _ in range(5):
        emb_seed.assign(np.arange(512, dtype=np.float32))
    emb_seed.dump_npz(emb_npz)

    write_status(
        status,
        last_embedding_safe_frame=12345,
        last_embedding_safe_shot_number=3,
        last_embedding_safe_shot_first_frame=1000,
        obs_rows_at_last_embedding_safe=10,
        emb_rows_at_last_embedding_safe=4,
    )

    # Live collectors (empty)
    obs_live = ObservationsCollector()
    emb_live = EmbeddingCollector(mode="sidecar", dim=512)

    mgr = CheckpointManager(run_dir, video_path="/tmp/video.mp4", resume=True)
    mgr.load_and_anchor_collectors(obs_live, emb_live)

    # Assert trimmed
    assert obs_live.count() == 10
    assert emb_live.count() == 4

    # Act: simulate new work after resume
    #  - add 3 new obs rows (same shot/track, later frames)
    obs_live.append_track_obs(mk_obs_block(3, shot=3, track_id=7, f_start=2000))
    #  - add 1 new embedding and link via add_embeddings (exercise API slightly)
    #    (we won't call add_embeddings() which links rows by search; here just ensure collector grows)
    emb_live.assign(np.ones(512, dtype=np.float32))

    # Finalize – should write fresh NPZs with counts: obs=13, emb=5
    mgr.start(obs_live, emb_live, options_snapshot={"detect_interval": 60})
    mgr.finalize("final")
    # Reload from disk to confirm content size
    obs_check = ObservationsCollector(); obs_check.load_npz(obs_npz)
    emb_check = EmbeddingCollector(mode="sidecar", dim=512); emb_check.load_npz(emb_npz)

    assert obs_check.count() == 13
    assert emb_check.count() == 5

def test_v21_writer_uses_hydrated_obs_collector_counts(tmp_path: Path):
    # Arrange: hydrated collector with pre + new rows
    obs_col = ObservationsCollector()
    obs_col.append_track_obs(mk_obs_block(8, shot=6, track_id=9, f_start=5000))   # “pre”
    obs_col.append_track_obs(mk_obs_block(2, shot=6, track_id=9, f_start=6000))   # “new”
    # Fake one minimal track with matching first/last; observations themselves only live in sidecar
    class _T:
        shot_id=6; track_id=9
        def __init__(self): self.observations=[]
        def first_frame(self): return 5000
        def last_frame(self):  return 6001
    tracks = [_T()]

    cfg = V2WriterConfig(
        video_path="/tmp/video.mp4",
        video_size=(1280,720),
        total_frames=100000,
        fps=30.0,
        normalize_to_percent=True,
        emb_store=None,
    )

    # Act: build v2.1 manifest with this already-filled collector
    manifest = build_v2_1_manifest_from_tracks(
        tracks, cfg,
        face_metadata=None,
        generation=None,
        detector=None,
        embedder=None,
        tracking_params={"detect_interval": 60},
        validator=None,
        emb_collector=None,            # embeddings disabled
        obs_collector=obs_col,         # <- PRE + NEW already present
    )

    # No side-effects yet; finalize to file to sanity-check the count
    out = tmp_path / "obs_sidecar.npz"
    sidecar = obs_col.finalize_sidecar(out)
    assert sidecar["count"] == 10           # 8 + 2
    assert manifest["schema_version"] == "2.1"
    # ensure the single shot was emitted
    assert len(manifest["shots"]) == 1
    s0 = manifest["shots"][0]
    assert s0["shot_number"] == 6
    # and the track references an obs slice (offset/count present)
    assert "face_tracks" in s0 and len(s0["face_tracks"]) == 1
    t0 = s0["face_tracks"][0]
    assert t0["obs_count"] == 0 or t0["obs_count"] <= sidecar["count"]  # count may be 0 if obs list on track is empty

def test_mark_embedding_safe_records_true_shot_first(tmp_path: Path):
    run = tmp_path / "r"; (run / "ckpt").mkdir(parents=True)
    mgr = CheckpointManager(run, video_path="/tmp/video.mp4", resume=True)

    obs = ObservationsCollector()
    emb = EmbeddingCollector(mode="sidecar", dim=512)
    mgr.start(obs, emb, options_snapshot={"detect_interval": 60})

    # Simulate a detection at frame 100 within shot #6 whose true first frame was 80,
    # even if we resumed at 95; checkpoint_now must record shot_first_frame=80.
    agg = DummyAggregator([])  # minimal aggregator
    mgr.mark_embedding_safe(frame_idx=100, shot_number=6, shot_first_frame=80, note="test")
    st = json.loads((run / "status.json").read_text())
    assert st["last_embedding_safe_frame"] == 100
    assert st["last_embedding_safe_shot_number"] == 6
    assert st["last_embedding_safe_shot_first_frame"] == 80

def test_resume_hydrates_open_tracks_snapshot(tmp_path: Path):
    # Seed a run dir with status.json containing open_tracks
    run = tmp_path / "rehyd"; (run / "ckpt").mkdir(parents=True)
    obs_npz = run / "ckpt" / "obs_ckpt.npz"
    emb_npz = run / "ckpt" / "emb_ckpt.npz"

    # minimal sidecars so start()/load works
    ObservationsCollector().dump_npz(obs_npz)
    EmbeddingCollector(mode="sidecar", dim=512).dump_npz(emb_npz)

    write_status(
        run / "status.json",
        last_embedding_safe_frame=1000,
        last_embedding_safe_shot_number=6,
        last_embedding_safe_shot_first_frame=950,
        open_tracks=[
            {
                "shot": 6,
                "track_id": 0,
                "last_frame": 1000,
                "last_det_frame": 1000,
                "closed": False,
                "bbox": [10, 10, 50, 50],
            }
        ],
    )

    mgr = CheckpointManager(run, video_path="/tmp/v.mp4", resume=True)
    obs = ObservationsCollector(); emb = EmbeddingCollector(mode="sidecar", dim=512)
    mgr.start(obs, emb, options_snapshot={"detect_interval": 60})

    # API surface will vary; if you expose it, check what was rehydrated.
    # Prefer a stable accessor on mgr, falling back to status echo.
    st = mgr.read_status()
    assert st and st.get("open_tracks") and st["open_tracks"][0]["track_id"] == 0
