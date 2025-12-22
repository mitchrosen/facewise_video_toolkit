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
        # Return a single fake detection every time: (boxes, landmarks, confidences)
        return ([(0, 0, 10, 10)], [[(0, 0)] * 5], [0.9])


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
    obs_collector = ObservationsCollector()
    emb_collector = EmbeddingCollector(mode="sidecar", dim=512)
    ckpt.start(obs_collector, emb_collector, options_snapshot=opts)

    # Anchor at 180 (inside shot #2: frames [120..239])
    anchor_shot = 2
    anchor_shot_first_frame = 120

    ckpt._last_det_frame = anchor
    ckpt._last_det_shot = anchor_shot
    ckpt._last_det_shot_first_frame = anchor_shot_first_frame

    # Seed one pre-anchor DET obs in the anchor shot to exercise rehydrate.
    # This mirrors what track_across_segments would have persisted before crash.
    obs_collector.append_track_obs(
        [{
            "shot": anchor_shot,
            "track_id": 1,
            "f": 150,
            "bbox_xyxy": [0, 0, 10, 10],
            "src": Source.DETECTED,
            "landmarks": _make_landmarks(5),
        }],
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
    st["last_detection_frame"] = anchor
    st["last_detection_shot"] = anchor_shot
    st["last_detection_shot_first_frame"] = anchor_shot_first_frame
    st["shot_segmentation_path"] = str(shots_json)
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
    ckpt_now_calls = []

    orig_add_observations = ckpt.add_observations
    orig_add_embeddings = ckpt.add_embeddings
    orig_checkpoint_now = ckpt.checkpoint_now

    def wrapped_add_observations(shot_number, frame_idx, obs_batch):
        add_obs_calls.append((int(shot_number), int(frame_idx), obs_batch))
        return orig_add_observations(shot_number, frame_idx, obs_batch)

    def wrapped_add_embeddings(shot_number, track_id, last_idx, embs):
        add_emb_calls.append((int(shot_number), int(track_id), int(last_idx), embs))
        return orig_add_embeddings(shot_number, track_id, last_idx, embs)

    def wrapped_checkpoint_now(*, frame_idx, shot_number, aggregator, shot_first_frame, note=None):
        ckpt_now_calls.append(dict(
            frame_idx=int(frame_idx), shot_number=int(shot_number),
            shot_first_frame=int(shot_first_frame), note=note or ""
        ))
        return orig_checkpoint_now(
            frame_idx=frame_idx,
            shot_number=shot_number,
            aggregator=aggregator,
            shot_first_frame=shot_first_frame,
            note=note,
        )

    # Attach wrappers to the *resumed* manager
    monkeypatch.setattr(ckpt, "add_observations", wrapped_add_observations)
    monkeypatch.setattr(ckpt, "add_embeddings", wrapped_add_embeddings)
    monkeypatch.setattr(ckpt, "checkpoint_now", wrapped_checkpoint_now)

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
    # We must seek to the anchor frame at resume.
    assert any(v == anchor for v in fp.reset_calls), \
        "expected a reset_to_frame(anchor) but didn't see one"

    first_after = fp.first_next_after_anchor_seek(anchor)
    assert first_after == anchor, \
        f"first processed frame {first_after} != anchor {anchor}"

    # ---------------- B) No pre-anchor OBS writes ----------------
    # We must not rewrite history for frames < anchor.
    for (_shotnum, f, _batch) in add_obs_calls:
        assert f >= anchor, f"persisted observations for pre-anchor frame {f} < {anchor}"

    # Embeddings ARE allowed for pre-anchor DET frames in the anchor-containing shot
    # (backfill crops -> embed at shot end). They must NOT be from earlier shots.
    for (shotnum, _tid, last_idx, _embs) in add_emb_calls:
        # must be for the anchor shot
        assert shotnum == anchor_shot, f"unexpected embeddings for non-anchor shot {shotnum}"
        # frame must be within that shot’s bounds (can be < anchor due to backfill)
        assert anchor_shot_first <= last_idx <= anchor_shot_last, (
            f"embedding tagged to frame {last_idx} outside shot-{anchor_shot} "
            f"[{anchor_shot_first}..{anchor_shot_last}]"
        )

    # ---------------- C) First checkpoint at anchor ----------------
    assert ckpt_now_calls, "expected checkpoint_now to be called at the anchor"
    first_ck = ckpt_now_calls[0]
    assert first_ck["frame_idx"] == anchor, \
        f"first checkpoint at {first_ck['frame_idx']} != anchor {anchor}"
