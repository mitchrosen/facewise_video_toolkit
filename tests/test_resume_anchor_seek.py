from pathlib import Path
import json
import numpy as np
import pytest

from facekit.pipeline.track_across_segments import track_across_segments
from facekit.pipeline.checkpoint import CheckpointManager
from facekit.output.json_v2 import ObservationsCollector, EmbeddingCollector
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
    def __init__(self):
        self.calls = 0

    def detect_faces_in_frame(self, frame):
        # Return a single fake detection every time: (boxes, landmarks, confidences)
        # Encode a non-constant value in landmarks[0][0] so tests can sanity-check it.
        self.calls += 1
        x = float(self.calls)  # deterministic, non-constant across calls
        return ([(0, 0, 10, 10)], [[(x, 0.0)] + [(0.0, 0.0)] * 4], [0.9])

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
        self.ops = []   # list[tuple[str,int]] e.g. ('reset', 180) or ('next', 180)
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


# --- Phase 1: simulate interrupted run and persist sidecars + status ---


def _seed_preanchor_run(
    parent: Path,
    vid: Path,
    shots_json: Path,
    opts: dict,
    *,
    anchor: int = 180,
) -> str:
    """
    Phase 1: simulate a run that reached `anchor`, wrote sidecars + status.json,
    and then "crashed". Returns the run_id of the checkpoint dir.
    """
    ckpt = CheckpointManager.open(
        parent_dir=parent,
        video_path=vid,
        options_snapshot=opts,
        no_resume=True,
        force_new_run=False,
    )

    # Wire up collectors exactly like the real pipeline would.
    oc = ObservationsCollector()
    embc = EmbeddingCollector(mode="sidecar", dim=512)
    ckpt.start(oc, embc, options_snapshot=opts)

    # Anchor at 180 (inside shot #2: frames [120..239])
    # New design: the durable resume boundary is the *embedding-safe* anchor.
    anchor_shot = 2
    anchor_shot_first_frame = 120

    if hasattr(ckpt, "_last_embedding_safe_frame"):
        ckpt._last_embedding_safe_frame = anchor
    if hasattr(ckpt, "_last_embedding_safe_shot"):
        ckpt._last_embedding_safe_shot = anchor_shot
    if hasattr(ckpt, "_last_embedding_safe_shot_first_frame"):
        ckpt._last_embedding_safe_shot_first_frame = anchor_shot_first_frame

    # Seed one pre-anchor DET obs in the anchor shot to exercise rehydrate.
    # This mirrors what track_across_segments would have persisted before crash.
    oc.append_track_obs(
        [
            {
                "shot": anchor_shot,
                "track_id": 1,
                "f": 150,
                "bbox_xyxy": [0, 0, 10, 10],
                "src": "detected",
            }
        ],
        emb_idx_fn=lambda _: -1,  # no embedding index yet; will be filled via add_embeddings
    )

    # Add an embedding row for that DET (what would happen at shot end).
    ckpt.add_embeddings(
        shot_number=anchor_shot,
        track_id=1,
        frame_idx_last=150,
        embs=np.ones((1, 512), dtype=np.float32),
    )

    # Finalize to actually write obs_ckpt.npz / emb_ckpt.npz to disk.
    ckpt.finalize()

    # Now patch status.json to reflect the anchor and a minimal track_order,
    # mimicking a partially-complete but structurally valid status file.
    st = ckpt.read_status() or {}
    st["last_embedding_safe_frame"] = anchor
    st["last_embedding_safe_shot_number"] = anchor_shot
    st["last_embedding_safe_shot_first_frame"] = anchor_shot_first_frame
    # Minimal track_order: single track (shot=2, track_id=1).
    st["track_order"] = [{"shot": anchor_shot, "track_id": 1, "order": 0}]
    ckpt.status_path.write_text(json.dumps(st, indent=2))

    return ckpt.run_id


# --- Phase 2: real resume using on-disk state ---


def test_resume_starts_at_anchor_abs_frame(tmp_path: Path, monkeypatch):
    # shots: [0..119], [120..239], [240..359]
    shots = tmp_path / "shots.json"
    _write_shots(shots, 0, 359, per_shot=120)

    parent = tmp_path / "ck"
    parent.mkdir()
    vid = tmp_path / "dummy.mp4"
    vid.write_text("x")

    opts = {
        "schema_version": "2.1", "video_path": str(vid),
        "detect_interval": 60, "embedding_batch_size_max": 8, "device": "cpu",
        "emb_store": "sidecar", "emb_sidecar_path": None, "obs_sidecar_path": None,
        "detector_model_path": "x", "embedding_model_path": "y", "yolo_config_path": "z",
        "shot_segmentation_path": str(shots), "log_level": "INFO", "log_file": None,
    }

    anchor = 180
    anchor_shot = 2
    anchor_shot_first, anchor_shot_last = 120, 239

    # ---------- Phase 1: write sidecars + status as if a run had crashed ----------
    run_id = _seed_preanchor_run(
        parent=parent,
        vid=vid,
        shots_json=shots,
        opts=opts,
        anchor=anchor,
    )

    # ---------- Phase 2: open for resume from that on-disk state ----------
    ckpt = CheckpointManager.open(
        parent_dir=parent,
        video_path=vid,
        options_snapshot=opts,
        no_resume=False,
        force_new_run=False,
        run_id=run_id,  # <<< resume from the seeded run
    )

    # Avoid persist-integrity guards by disabling writes.
    if hasattr(ckpt, "write_disabled"):
        ckpt.write_disabled = True

    # Ensure track-order resolution is trivial for this test
    setattr(ckpt, "get_track_order", lambda: {(anchor_shot, 1): 0})

    # Keep segment-id resolution out of scope for this test
    monkeypatch.setattr(
        _agg.ShotFaceTrackAggregator,
        "resolve_segment_ids",
        lambda self, **kw: 0,
    )

    # ---- Spy/wrap persistence calls so we can assert business invariants ----
    add_obs_calls = []
    add_emb_calls = []

    orig_add_observations = ckpt.add_observations
    orig_add_embeddings = ckpt.add_embeddings

    def wrapped_add_observations(shot_number, frame_idx, obs_batch):
        add_obs_calls.append((int(shot_number), int(frame_idx), obs_batch))
        return orig_add_observations(shot_number, frame_idx, obs_batch)

    def wrapped_add_embeddings(shot_number, track_id, last_idx, embs):
        add_emb_calls.append((int(shot_number), int(track_id), int(last_idx), embs))
        return orig_add_embeddings(shot_number, track_id, last_idx, embs)

    # Attach wrappers to the *resumed* manager
    monkeypatch.setattr(ckpt, "add_observations", wrapped_add_observations)
    monkeypatch.setattr(ckpt, "add_embeddings", wrapped_add_embeddings)

    # Frame provider spy (records first frame actually consumed by the main loop)
    fp = SpyFP(total=400)

    _ = track_across_segments(
        frame_source=fp,
        shot_json_path=str(shots),
        detector=DummyDetector(),
        embedder=DummyEmbedder(),
        checkpoint=ckpt,
    )

    # ---------------- A) Anchor seek behavior ----------------
    # Anchor is inclusive; resume should start at the first frame *after* the anchor.
    start = anchor + 1
    assert any(v == start for v in fp.reset_calls), \
        "expected a reset_to_frame(anchor+1) but didn't see one"

    first_after = fp.first_next_after_anchor_seek(start)
    assert first_after == start, \
        f"first processed frame {first_after} != start {start}"

    # ---------------- B) No pre-anchor OBS writes ----------------
    # We must not rewrite history for frames < anchor.
    for (_shotnum, f, _batch) in add_obs_calls:
        assert f >= anchor, f"persisted observations for pre-anchor frame {f} < {anchor}"

    # Embeddings ARE allowed for pre-anchor DET frames in the anchor-containing shot
    for (shotnum, _tid, last_idx, _embs) in add_emb_calls:
        # must be for the anchor shot
        assert shotnum == anchor_shot, f"unexpected embeddings for non-anchor shot {shotnum}"
        # frame must be within that shot’s bounds (can be < anchor due to backfill)
        assert anchor_shot_first <= last_idx <= anchor_shot_last, (
            f"embedding tagged to frame {last_idx} outside shot-{anchor_shot} "
            f"[{anchor_shot_first}..{anchor_shot_last}]"
        )
