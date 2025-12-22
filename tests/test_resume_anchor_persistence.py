from pathlib import Path
import json
import numpy as np
import pytest

from facekit.pipeline.track_across_segments import track_across_segments
from facekit.pipeline.checkpoint import CheckpointManager
from facekit.output.json_v2 import ObservationsCollector, EmbeddingCollector
from facekit.tracking import aggregator as _agg
from facekit.common.obs_consts import Source

def _make_landmarks(k: int = 5) -> np.ndarray:
    # Small, deterministic landmark set
    xy = np.zeros((k, 2), dtype=np.float32)
    for i in range(k):
        xy[i, 0] = 10.0 + i
        xy[i, 1] = 20.0 + 2 * i
    return xy

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
        return ([(0, 0, 10, 10)], [[(0,0)]*5], [0.9])

class DummyEmbedder:
    def get_embedding_batch(self, crops, batch_size=32):
        return np.ones((len(crops), 512), dtype=np.float32)

class SpyFP:
    def __init__(self, total=400, w=64, h=48, fps=30.0):
        self._total = total
        self._w, self._h = w, h
        self._fps = fps
        self._idx = 0
        self._blank = np.zeros((h, w, 3), dtype=np.uint8)
        self.reset_calls = []
        self.ops: list[tuple[str, int]] = []
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
        self.ops.append(("reset", self._idx))
    def next(self):
        if self.first_next_idx is None:
            self.first_next_idx = self._idx
        self.ops.append(("next", self._idx))
        if self._idx >= self._total:
            return None
        self._idx += 1
        return self._blank
    def first_next_after_anchor_seek(self, anchor: int) -> int | None:
        """
        Return the frame index of the first `next()` *after* we have sought to the anchor.
        This ignores any pre-anchor backfill I/O.
        """
        anchor_seek_pos = None
        for i, (op, val) in enumerate(self.ops):
            if op == "reset" and val == anchor:
                anchor_seek_pos = i
                break
        if anchor_seek_pos is None:
            return None
        for op, val in self.ops[anchor_seek_pos + 1:]:
            if op == "next":
                return val
        return None

def test_resume_starts_at_anchor_and_never_persists_preanchor(tmp_path: Path, monkeypatch):
    # shots: [0..119], [120..239], [240..359]
    shots = tmp_path / "shots.json"
    _write_shots(shots, 0, 359, per_shot=120)

    parent = tmp_path / "ck"; parent.mkdir()
    vid = tmp_path / "dummy.mp4"; vid.write_text("x")

    opts = {
        "schema_version": "2.1", "video_path": str(vid),
        "detect_interval": 60, "embedding_batch_size_max": 8, "device": "cpu",
        "emb_store": "sidecar", "emb_sidecar_path": None, "obs_sidecar_path": None,
        "detector_model_path": "x", "embedding_model_path": "y", "yolo_config_path": "z",
        "shot_segmentation_path": str(shots), "log_level": "INFO", "log_file": None,
    }
    
    ckpt = CheckpointManager.open(
        parent_dir=parent,
        video_path=vid,
        options_snapshot=opts,
        no_resume=True,
        force_new_run=False,
    )

    # Wire up collectors exactly like the real pipeline would.
    obs_collector = ObservationsCollector()
    emb_collector = EmbeddingCollector(mode="sidecar", dim=512)

    # Start checkpoint manager with live collectors and options snapshot.
    ckpt.start(obs_collector, emb_collector, options_snapshot=opts)

    # Anchor at 180 (inside shot #2)
    anchor = 180
    ckpt._last_det_frame = anchor
    ckpt._last_det_shot = 2
    ckpt._last_det_shot_first_frame = 120

    # Seed one pre-anchor obs to exercise rehydrate
    obs_collector.append_track_obs(
        [{
            "shot": 2,
            "track_id": 1,
            "f": 100,
            "bbox_xyxy": [10, 10, 50, 50],
            "src": Source.DETECTED,
            "conf": 0.9,
            "landmarks": _make_landmarks(5),
        },
        {
            "shot": 2,
            "track_id": 1,
            "f": 150,
            "bbox_xyxy": [0, 0, 10, 10],
            "src": Source.DETECTED,
            "landmarks": _make_landmarks(5),
        }],
        emb_idx_fn=lambda _: -1,
    )

    orig_find_rows = obs_collector.find_rows

    def _flat_find_rows(*args, **kwargs):
        rows = orig_find_rows(*args, **kwargs)
        flat = []
        for pos in rows:
            if isinstance(pos, tuple):
                if len(pos) != 2:
                    raise TypeError(f"unexpected find_rows position tuple: {pos!r}")
                block_idx, row_idx = pos
                # Current collector should only use a single block 0; assert to catch surprises.
                assert int(block_idx) == 0, f"unexpected block index from find_rows: {pos!r}"
                flat.append(int(row_idx))
            else:
                flat.append(int(pos))
        return flat

    obs_collector.find_rows = _flat_find_rows

    # This now uses the live EmbeddingCollector wired in via start()
    ckpt.add_embeddings(
        shot_number=2,
        track_id=1,
        frame_idx_last=100,
        embs=np.ones((1, 512), dtype=np.float32),
    )

    ckpt.add_embeddings(
        shot_number=2,
        track_id=1,
        frame_idx_last=150,
        embs=np.ones((1, 512), dtype=np.float32),
    )

    setattr(ckpt, "get_track_order", lambda: {(2, 1): 0})

    # keep segment-id resolution out of scope
    monkeypatch.setattr(_agg.ShotFaceTrackAggregator, "resolve_segment_ids", lambda self, **kw: 0)

    # wrap persistence calls
    add_obs_calls = []
    add_emb_calls = []
    orig_add_observations = ckpt.add_observations
    orig_add_embeddings = ckpt.add_embeddings

    def wrapped_add_observations(shot_number, frame_idx, obs_batch):
        add_obs_calls.append((int(shot_number), int(frame_idx)))
        return orig_add_observations(shot_number, frame_idx, obs_batch)

    def wrapped_add_embeddings(shot_number, track_id, last_idx, embs):
        add_emb_calls.append((int(shot_number), int(track_id), int(last_idx)))
        return orig_add_embeddings(shot_number, track_id, last_idx, embs)

    monkeypatch.setattr(ckpt, "add_observations", wrapped_add_observations)
    monkeypatch.setattr(ckpt, "add_embeddings", wrapped_add_embeddings)

    # Sanity check: can the checkpoint see the pre-anchor frames and embeddings?
    ckpt.finalize(note="pre-anchor-sanity")
    frames = ckpt.get_det_frames_for_track(2, 1, frame_max=anchor - 1)
    assert frames == [100, 150], f"Unexpected det frames: {frames}"

    embs = ckpt.get_embeddings_by_frames(2, frames)
    assert embs is not None, "No embeddings returned for seeded frames"
    assert embs.shape[0] == 2 and embs.shape[1] == 512, f"Bad emb shape: {getattr(embs, 'shape', None)}"

    fp = SpyFP(total=400)
    _ = track_across_segments(
        frame_source=fp,
        shot_json_path=str(shots),
        detector=DummyDetector(),
        embedder=DummyEmbedder(),
        checkpoint=ckpt,
    )

    # A) We must actually start the *main loop* at the anchor frame.
    # Allow pre-anchor backfill, but the first next() after resetting to the anchor must be >= anchor.
    first_after_seek = fp.first_next_after_anchor_seek(anchor)
    assert first_after_seek is not None, "did not observe a next() after seeking to the anchor"
    assert first_after_seek >= anchor, f"first processed frame after anchor-seek {first_after_seek} < anchor {anchor}"

    # B1) NEVER persist observations < anchor
    for (_shot, f) in add_obs_calls:
        assert f >= anchor, f"persisted observations at pre-anchor frame {f} < {anchor}"

    # B2) Embeddings rule:
    #  - allowed: pre-anchor last_idx IF it belongs to the anchor shot
    #  - forbidden: any embeddings for shots strictly before the anchor shot
    with open(shots, "r") as f:
        _data = json.load(f)
    def _shot_of(frame):
        for s in _data["shots"]:
            if s["first_frame"] <= frame <= s["last_frame"]:
                return s["shot_number"]
        return None
    anchor_shot = _shot_of(anchor)
    # shots strictly before the anchor are those with last_frame < anchor
    pre_anchor_shots = {s["shot_number"] for s in _data["shots"] if s["last_frame"] < anchor}

    for (shot_num, _tid, last_idx) in add_emb_calls:
        if last_idx < anchor:
            assert shot_num == anchor_shot, (
                f"persisted embeddings for pre-anchor frame {last_idx} in non-anchor shot {shot_num} "
                f"(anchor shot is {anchor_shot})"
            )
        # also ensure we never embed for shots strictly before anchor
        assert shot_num not in pre_anchor_shots, (
            f"persisted embeddings for shot {shot_num} which is strictly before anchor (anchor={anchor})"
        )