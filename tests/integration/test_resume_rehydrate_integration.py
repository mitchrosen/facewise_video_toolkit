from pathlib import Path
import json
import numpy as np

from facekit.pipeline.track_across_segments import track_across_segments
from facekit.pipeline.checkpoint import CheckpointManager
from facekit.output.json_v2 import ObservationsCollector

class DummyDetector:
    def detect_faces_in_frame(self, frame): return None

class DummyEmbedder:
    def get_embedding_batch(self, crops, batch_size=32):
        return np.zeros((len(crops), 512), dtype=np.float32)

def _write_shots(p: Path, f0: int, f1: int):
    p.write_text(json.dumps({"shots":[{"shot_number":1,"first_frame":f0,"last_frame":f1}]}))

def test_resume_invokes_rehydrate_once_with_anchor_minus_one(tmp_path, monkeypatch):
    # --- minimal frame provider to avoid real I/O ---
    class DummyFP:
        def __init__(self, total=60, w=192, h=108, fps=30.0):
            import numpy as _np
            self._total = total
            self._w, self._h = w, h
            self._fps = fps
            self._idx = 0
            self._blank = _np.zeros((h, w, 3), dtype=_np.uint8)
        @property
        def fps(self): return self._fps
        @property
        def size(self): return (self._w, self._h)
        @property
        def total_frames(self): return self._total
        def reset_to_frame(self, i): self._idx = int(i)
        def next(self):
            if self._idx >= self._total: return None
            self._idx += 1
            return self._blank

    fp = DummyFP()

    # --- shots file ---
    shots = tmp_path / "shots.json"
    _write_shots(shots, 0, 59)

    # --- checkpoint with an anchor at 30 ---
    vid = tmp_path / "dummy.mp4"
    vid.write_text("placeholder")
    parent = tmp_path / "ck"
    parent.mkdir()
    opts = {
        "schema_version":"2.1","video_path":str(vid),"detect_interval":10,
        "embedding_batch_size_max":32,"device":"cpu","emb_store":"sidecar",
        "emb_sidecar_path":None,"obs_sidecar_path":None,"detector_model_path":"x",
        "embedding_model_path":"y","yolo_config_path":"z","shot_segmentation_path":str(shots),
        "log_level":"INFO","log_file":None
    }
    ckpt = CheckpointManager.open(
        parent_dir=parent, video_path=vid, options_snapshot=opts,
        no_resume=True, force_new_run=False
    )
    # simulate prior run’s last detection
    ckpt._last_det_frame = 30
    ckpt._last_det_shot = 1
    ckpt._last_det_shot_first_frame = 0

    # --- pre-anchor observations in collector ---
    oc = ObservationsCollector()
    oc.append_track_obs(
        [{"shot":1,"track_id":1,"f":10,"bbox_xyxy":[0,0,10,10],"src":"detected"}],
        emb_idx_fn=lambda _: -1
    )
    ckpt.obs_collector = oc

    # --- monkeypatches ---
    called = {}
    def fake_rehydrate(collector, frame_max, **kwargs):
        called["args"] = (collector, frame_max, kwargs.get("track_order"))
        return []

    # ensure pipeline uses our fake
    monkeypatch.setattr(
        "facekit.pipeline.track_across_segments.rehydrate_tracks",
        fake_rehydrate
    )

    # provide non-empty track_order (we didn't really persist one)
    monkeypatch.setattr(ckpt, "get_track_order", lambda: {(1, 1): 0})

    # --- run ---
    _ = track_across_segments(
        frame_source=fp,
        shot_json_path=str(shots),
        detector=DummyDetector(),
        embedder=DummyEmbedder(),
        checkpoint=ckpt,
    )

    # --- assertions ---
    assert "args" in called
    _, fm, to = called["args"]
    assert fm == 29                      # anchor - 1
    assert isinstance(to, dict) and to   # non-empty track_order
    # First frame processed should be the anchor (30) or later.
    # Our DummyFP increments before returning, so check its internal cursor:
    assert fp._idx >= 30
