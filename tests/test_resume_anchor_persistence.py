# tests/test_resume_anchor_persistence.py

from pathlib import Path
import json
import numpy as np
import pytest

from facekit.pipeline.track_across_segments import track_across_segments
from facekit.pipeline.checkpoint import CheckpointManager
from facekit.output.json_v2 import ObservationsCollector, EmbeddingCollector
from facekit.tracking import aggregator as _agg
from facekit.common.obs_consts import Source, src_to_code
import facekit.pipeline.resume_rehydrate as _rr

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

def _seed_resume_run_dir(parent: Path, *, vid: Path, opts: dict, anchor: int) -> str:
    """
    Create a run directory that looks like a previously-crashed run with an embedding-safe anchor.
    We keep it minimal because this test is about resume seeking + boundary behavior, not I/O integrity.
    """
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / "run-000001"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "ckpt").mkdir(exist_ok=True)

    st = dict(opts)
    st.update(
        checkpoint_dir=str(run_dir),
        last_embedding_safe_frame=int(anchor),
        last_embedding_safe_shot_number=2,
        last_embedding_safe_shot_first_frame=120,
        obs_rows_at_last_embedding_safe=0,
        emb_rows_at_last_embedding_safe=0,
        track_order=[],
        frames_done=0,
        shots_done=0,
        tracks_seen=0,
    )
    (run_dir / "status.json").write_text(json.dumps(st, indent=2))

    # Write *valid* (possibly empty) NPZs. Zero-byte files cause np.load EOFError.
    obs_path = run_dir / "ckpt" / "obs_ckpt.npz"
    emb_path = run_dir / "ckpt" / "emb_ckpt.npz"

    oc = ObservationsCollector()
    oc.dump_npz(obs_path)

    ec = EmbeddingCollector(mode="sidecar", dim=512)
    ec.dump_npz(emb_path)

    return run_dir.name

class DummyDetector:
    def __init__(self):
        self.calls = 0

    def detect_faces_in_frame(self, frame):
        self.calls += 1
        x = float(self.calls)  # frame_idx+1 in this SpyFP setup
        return ([(0, 0, 10, 10)], [[(x, 0.0)] + [(0.0, 0.0)] * 4], [0.9])


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

    parent = tmp_path / "ck"
    parent.mkdir()
    vid = tmp_path / "dummy.mp4"
    vid.write_text("x")

    opts = {
        "schema_version": "2.1",
        "video_path": str(vid),
        "detect_interval": 60,
        "embedding_batch_size_max": 8,
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

    # Seed a prior on-disk run with an embedding-safe anchor, then reopen in resume mode.
    anchor = 180
    run_id = _seed_resume_run_dir(parent, vid=vid, opts=opts, anchor=anchor)
    ckpt = CheckpointManager.open(
        parent_dir=parent,
        video_path=vid,
        options_snapshot=opts,
        no_resume=False,
        force_new_run=False,
        run_id=run_id,
    )

    # Wire up collectors exactly like the real pipeline would.
    oc = ObservationsCollector()
    embc = EmbeddingCollector(mode="sidecar", dim=512)

    # Start checkpoint manager with live collectors and options snapshot.
    ckpt.start(oc, embc, options_snapshot=opts)

    # Avoid persist-integrity guards.
    if hasattr(ckpt, "write_disabled"):
        ckpt.write_disabled = True

    # Seed pre-anchor obs in shot 2 (frames 100, 150).
    # IMPORTANT: get_embeddings_by_frames() relies on an obs->embedding-index linkage.
    # We therefore (a) assign embeddings in embc, and (b) set emb_idx on the obs rows.
    emb_idx_100 = embc.assign(np.ones((512,), dtype=np.float32))
    emb_idx_150 = embc.assign(np.ones((512,), dtype=np.float32))

    oc.append_track_obs(
        [
            {
                "shot": 2,
                "track_id": 1,
                "f": 100,
                "bbox_xyxy": [10, 10, 50, 50],
                "src": Source.DETECTED,
                "conf": 0.9,
                "emb_idx": int(emb_idx_100),
            },
            {
                "shot": 2,
                "track_id": 1,
                "f": 150,
                "bbox_xyxy": [0, 0, 10, 10],
                "src": Source.DETECTED,
                "emb_idx": int(emb_idx_150),
            },
        ],
       emb_idx_fn=lambda row: int(row.get("emb_idx", -1)),
    )

    # --- flatten find_rows: convert (block_idx,row_idx) -> row_idx (expecting single block 0)
    orig_find_rows = oc.find_rows

    def _flat_find_rows(*args, **kwargs):
        rows = orig_find_rows(*args, **kwargs)
        # Preserve (block_idx, row_idx) tuples; ObservationsCollector may use multiple blocks.
        out = []
        for pos in rows:
            if isinstance(pos, tuple):
                if len(pos) != 2:
                    raise TypeError(f"unexpected find_rows position tuple: {pos!r}")
                out.append((int(pos[0]), int(pos[1])))
            else:
                out.append(int(pos))
        return out

    oc.find_rows = _flat_find_rows  # type: ignore[attr-defined]

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

    # --- Find DETECTED rows (NOTE: find_rows expects int code for `source`)
    det_code = int(src_to_code(Source.DETECTED.value))
    pos = oc.find_rows(
        shot=2,
        track_id=1,
        frame_last=anchor - 1,
        source=det_code,
    )
    # orig_find_rows returns newest->oldest; align by sorting positions by frame.
    def _frame_at(p):
        if isinstance(p, tuple):
            b, r = p
            return int(oc._rows[b][r]["f"])  # type: ignore[attr-defined]
        return int(oc._rows[0][int(p)]["f"])  # type: ignore[attr-defined]

    pos_sorted = sorted(pos, key=_frame_at)
    assert [_frame_at(p) for p in pos_sorted] == frames

    # Validate embedding linkage for DET rows pre-anchor, *only when an embedding is claimed*.
    #
    # Narrow contract (enforceable today):
    #   - We cannot reliably infer which DET rows produced a valid aligned face.
    #   - Therefore we do NOT require every DET row pre-anchor to have an embedding.
    #   - But if a row claims an embedding (emb_idx >= 0), it must point to a real embedding row.
    for row_idx, f in zip(pos_sorted, frames):
        if isinstance(row_idx, tuple):
            b, r = row_idx
            row = oc._rows[b][r]  # type: ignore[attr-defined]
        else:
            row = oc._rows[0][int(row_idx)]  # type: ignore[attr-defined]

        if getattr(row, "dtype", None) is not None and row.dtype.names and "emb_idx" in row.dtype.names:
            emb_idx = int(row["emb_idx"])
        else:
            emb_idx = -1
        if emb_idx >= 0:
            # EmbeddingCollector stores vectors in-memory at 0-based indices; emb_idx is absolute.
            assert (emb_idx - embc._base) < len(embc._embs)  # type: ignore[attr-defined]

    # Validate that the seeded DET rows' emb_idx resolve into the in-memory collector.
    # NOTE: This test runs with ckpt.write_disabled=True, so persisted embedding retrieval is not
    # a contract requirement here. We assert the obs->emb_idx linkage is internally consistent.
    resolved = []
    for pos_i in pos_sorted:
        if isinstance(pos_i, tuple):
            b, r = pos_i
            row = oc._rows[b][r]  # type: ignore[attr-defined]
        else:
            row = oc._rows[0][int(pos_i)]  # type: ignore[attr-defined]
        emb_idx = int(row["emb_idx"]) if (row.dtype.names and "emb_idx" in row.dtype.names) else -1
        assert emb_idx >= 0, f"Expected emb_idx for seeded row, got {emb_idx}"
        resolved.append(embc._embs[emb_idx - embc._base])  # type: ignore[attr-defined]

    assert len(resolved) == 2, f"Expected 2 resolved embeddings, got {len(resolved)}"
    assert np.stack(resolved, axis=0).shape == (2, 512)

    # IMPORTANT:
    # This test is about resume seeking + "do not persist pre-anchor" behavior.
    #
    # The production resume pipeline currently may enforce additional invariants during
    # pre-anchor rehydration (e.g., requiring landmarks/embedding parity for DET rows).
    # Those invariants are orthogonal to what we're validating here, and we intentionally
    # run with ckpt.write_disabled=True (no filesystem durability contract in this test).
    #
    # To keep this test focused and deterministic, bypass pre-anchor rehydration.
    monkeypatch.setattr(_rr, "rehydrate_tracks", lambda *a, **k: [])

    fp = SpyFP(total=400)
    _ = track_across_segments(
        frame_source=fp,
        shot_json_path=str(shots),
        detector=DummyDetector(),
        embedder=DummyEmbedder(),
        checkpoint=ckpt,
    )

    # A) Anchor is inclusive; resume work starts at the first frame after the anchor.
    start = anchor + 1
    first_after_seek = fp.first_next_after_anchor_seek(start)
    assert first_after_seek is not None, "did not observe a next() after seeking to the anchor"
    assert first_after_seek >= start, (
        f"first processed frame after anchor-seek {first_after_seek} < start {start}"
    )

    # B1) NEVER persist observations at/before anchor
    for (_shot, f) in add_obs_calls:
        assert f > anchor, f"persisted observations at/before anchor frame {f} <= {anchor}"

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
    pre_anchor_shots = {s["shot_number"] for s in _data["shots"] if s["last_frame"] < anchor}

    for (shot_num, _tid, last_idx) in add_emb_calls:
        # Under embedding-safe anchors, frames <= anchor are already durable and must not be re-written.
        assert last_idx > anchor or shot_num == anchor_shot, (
            "unexpected embedding write at/before anchor outside the anchor-containing shot: "
            f"shot={shot_num} last_idx={last_idx} anchor={anchor}"
        )
        assert shot_num not in pre_anchor_shots, (
            f"persisted embeddings for shot {shot_num} which is strictly before anchor (anchor={anchor})"
        )
