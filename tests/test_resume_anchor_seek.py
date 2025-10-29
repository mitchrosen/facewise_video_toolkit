from pathlib import Path
import json
import numpy as np
import pytest

from facekit.pipeline.track_across_segments import track_across_segments
from facekit.pipeline.checkpoint import CheckpointManager
from facekit.output.json_v2 import ObservationsCollector
from facekit.tracking import aggregator as _agg

# --- local helpers (no cross-imports) ---

def _write_shots(path: Path, first: int, last: int, per_shot: int = None):
    if per_shot is None:
        shots = [{"shot_number": 1, "first_frame": first, "last_frame": last}]
    else:
        shots, s, sn = [], first, 1
        while s <= last:
            e = min(s + per_shot - 1, last)
            shots.append({"shot_number": sn, "first_frame": s, "last_frame": e})
            s, sn = e + 1, sn + 1
    path.write_text(json.dumps({"shots": shots}))

class DummyDetector:
    def detect_faces_in_frame(self, frame):
        # Return a single fake detection every time: (boxes, landmarks, confidences)
        return ([(0, 0, 10, 10)], [[(0,0)]*5], [0.9])

class DummyEmbedder:
    def get_embedding_batch(self, crops, batch_size=32):
        # Return 512-d unit vectors
        return np.ones((len(crops), 512), dtype=np.float32)

class SpyFP:
    def __init__(self, total=400, w=64, h=48, fps=30.0):
        self._total = total
        self._w, self._h = w, h
        self._fps = fps
        self._idx = 0
        self._blank = np.zeros((h, w, 3), dtype=np.uint8)
        self.reset_calls = []
        self.first_next_idx = None
    @property
    def fps(self): return self._fps
    @property
    def size(self): return (self._w, self._h)
    @property
    def total_frames(self): return self._total
    def reset_to_frame(self, i):
        self._idx = int(i)
        self.reset_calls.append(self._idx)
    def next(self):
        if self.first_next_idx is None:
            self.first_next_idx = self._idx
        if self._idx >= self._total:
            return None
        self._idx += 1
        return self._blank

# --- the test ---

def test_resume_starts_at_anchor_abs_frame(tmp_path: Path, monkeypatch):
    # shots: [0..119], [120..239], [240..359]
    shots = tmp_path / "shots.json"
    _write_shots(shots, 0, 359, per_shot=120)

    parent = tmp_path / "ck"
    parent.mkdir()
    vid = tmp_path / "dummy.mp4"
    vid.write_text("x")

    opts = {
        "schema_version":"2.1","video_path":str(vid),
        "detect_interval":60,"embedding_batch_size_max":8,"device":"cpu",
        "emb_store":"sidecar","emb_sidecar_path":None,"obs_sidecar_path":None,
        "detector_model_path":"x","embedding_model_path":"y","yolo_config_path":"z",
        "shot_segmentation_path":str(shots),"log_level":"INFO","log_file":None,
    }
    ckpt = CheckpointManager.open(parent_dir=parent, video_path=vid, options_snapshot=opts,
                                  no_resume=True, force_new_run=False)

    # Simulate anchor at abs frame 180 (inside shot #2)
    ckpt._last_det_frame = 180
    ckpt._last_det_shot = 2
    ckpt._last_det_shot_first_frame = 120

    # Collector is required by pipeline; add one pre-anchor obs so rehydrate path executes
    oc = ObservationsCollector()
    oc.append_track_obs(
        [{"shot":2,"track_id":1,"f":150,"bbox_xyxy":[0,0,10,10],"src":"detected"}],
        emb_idx_fn=lambda _: -1
    )
    ckpt.obs_collector = oc
    setattr(ckpt, "get_track_order", lambda: {(2, 1): 0})

    monkeypatch.setattr(_agg.ShotFaceTrackAggregator, "resolve_segment_ids", lambda self, **kw: 0)

    fp = SpyFP(total=400)

    _ = track_across_segments(
        frame_source=fp,
        shot_json_path=str(shots),
        detector=DummyDetector(),
        embedder=DummyEmbedder(),
        checkpoint=ckpt,
    )

    # Must seek to the anchor (>=180), not 0
    assert fp.reset_calls, "Frame provider reset_to_frame should be called"
    assert fp.reset_calls[0] >= 180
    assert fp.first_next_idx >= 180
