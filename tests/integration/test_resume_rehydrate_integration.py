from pathlib import Path
import json
import numpy as np

from facekit.pipeline.track_across_segments import track_across_segments
from facekit.pipeline.checkpoint import CheckpointManager
from facekit.output.json_v2 import ObservationsCollector, EmbeddingCollector
import facekit.pipeline.resume_rehydrate as resume_rehydrate


# ------------------- Fakes -------------------

class DummyDetector:
    # Must return the 3-tuple shape, not None
    def detect_faces_in_frame(self, frame):
        return ([], [], [])


class DummyEmbedder:
    def get_embedding_batch(self, crops, batch_size=32):
        return np.zeros((len(crops), 512), dtype=np.float32)


# ------------------- Helpers -------------------

def _write_shots(p: Path, f0: int, f1: int):
    p.write_text(json.dumps({"shots": [{"shot_number": 1, "first_frame": f0, "last_frame": f1}]}))


# ------------------- Test -------------------

def test_resume_invokes_rehydrate_once_with_anchor_minus_one(tmp_path, monkeypatch):
    # ---- minimal frame provider with instrumentation ----
    class DummyFP:
        def __init__(self, total=60, w=192, h=108, fps=30.0):
            self._total = total
            self._w, self._h = w, h
            self._fps = fps
            self._idx = 0
            self._blank = np.zeros((h, w, 3), dtype=np.uint8)
            self.ops = []   # ("reset", i) or ("next", i)

        @property
        def fps(self): return self._fps
        @property
        def size(self): return (self._w, self._h)
        @property
        def total_frames(self): return self._total

        def reset_to_frame(self, i):
            self._idx = int(i)
            self.ops.append(("reset", self._idx))

        def next(self):
            self.ops.append(("next", self._idx))
            if self._idx >= self._total:
                return None
            self._idx += 1
            return self._blank

        def first_next_after_reset(self, target):
            reset_pos = None
            for i, (op, val) in enumerate(self.ops):
                if op == "reset" and val == target:
                    reset_pos = i
                    break
            if reset_pos is None:
                return None
            for op, val in self.ops[reset_pos + 1:]:
                if op == "next":
                    return val
            return None

    fp = DummyFP()

    # ---- shots ----
    shots = tmp_path / "shots.json"
    _write_shots(shots, 0, 59)

    # ---- checkpoint with anchor=30 ----
    vid = tmp_path / "dummy.mp4"
    vid.write_text("placeholder")
    parent = tmp_path / "ck"
    parent.mkdir()

    opts = {
        "schema_version": "2.1",
        "video_path": str(vid),
        "detect_interval": 10,
        "embedding_batch_size_max": 32,
        "device": "cpu",
        "emb_store": "sidecar",
        "emb_sidecar_path": None,
        "obs_sidecar_path": None,
        "detector_model_path": "x",
        "embedding_model_path": "y",
        "yolo_config_path": "z",
        "shot_segmentation_path": str(shots),
        "log_level": "INFO",
        "log_file": None,
    }

    ckpt = CheckpointManager.open(
        parent_dir=parent,
        video_path=vid,
        options_snapshot=opts,
        no_resume=True,
        force_new_run=False,
    )

    anchor = 30
    ckpt._last_emb_safe_frame = anchor
    ckpt._last_emb_safe_shot = 1
    ckpt._last_emb_safe_shot_first_frame = 0

    # ---- wire collectors exactly like prod ----
    oc = ObservationsCollector()
    embc = EmbeddingCollector(mode="sidecar", dim=512)
    ckpt.start(oc, embc, options_snapshot=opts)

    oc.append_track_obs(
        [{
            "shot": 1,
            "track_id": 1,
            "f": 10,
            "bbox_xyxy": [0, 0, 10, 10],
            "src": "detected",
        }],
        emb_idx_fn=lambda _: -1,
    )

    # ---- spy on rehydrate_tracks ----
    called = {"count": 0, "args": None}

    def fake_rehydrate(collector, frame_max, **kwargs):
        called["count"] += 1
        called["args"] = (collector, int(frame_max), kwargs.get("track_order"))
        return []

    monkeypatch.setattr(resume_rehydrate, "rehydrate_tracks", fake_rehydrate)

    # non-empty track_order required
    monkeypatch.setattr(ckpt, "get_track_order", lambda: {(1, 1): 0})

    # ---- run ----
    _ = track_across_segments(
        frame_source=fp,
        shot_json_path=str(shots),
        detector=DummyDetector(),
        embedder=DummyEmbedder(),
        checkpoint=ckpt,
    )

    # ---- assertions ----

    # A) rehydrate called once strictly before the anchor; resume starts at anchor+1
    assert called["count"] == 1
    _, fm, to = called["args"]
    assert fm == anchor - 1
    assert isinstance(to, dict) and to

    # B) seek + start at anchor
    assert ("reset", anchor + 1) in fp.ops
    first_after = fp.first_next_after_reset(anchor + 1)
    assert first_after == anchor + 1
