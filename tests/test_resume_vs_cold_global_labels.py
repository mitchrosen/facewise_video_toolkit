# tests/test_resume_vs_cold_global_labels.py
from pathlib import Path
import json
import numpy as np

from facekit.pipeline.track_across_segments import track_across_segments
from facekit.tracking.tracking_resolution import GlobalIdentityResolver
from facekit.pipeline.checkpoint import CheckpointManager
from facekit.output.json_v2 import ObservationsCollector, EmbeddingCollector


# --- Deterministic test doubles -------------------------------------------------

class DummyDetector:
    """
    Encodes *frame index* into the first landmark x coordinate as (frame_idx + 1).

    We infer frame_idx from SpyFP's pixel stamp at [0,0,0:2] so this is stable
    across cold vs resume (unlike detector call-count).
    """
    def __init__(self):
        self.box = (0, 0, 10, 10)

    def detect_faces_in_frame(self, frame):
        lo = int(frame[0, 0, 0])
        hi = int(frame[0, 0, 1])
        frame_idx = lo + 256 * hi

        x = float(frame_idx + 1)
        lms = [[(x, 0.0)] + [(0.0, 0.0)] * 4]   # (5,2)
        return ([self.box], lms, [0.9])


class DummyEmbedder:
    def get_embedding_batch(self, crops, batch_size=32):
        # deterministic unit vectors (shape [N, 512])
        return np.tile(np.eye(1, 512, 0, dtype=np.float32), (len(crops), 1))


# --- Helpers -------------------------------------------------------------------

def _write_two_shots(path: Path, s0=(0, 102), s1=(103, 299)):
    shots = [
        {"shot_number": 1, "first_frame": s0[0], "last_frame": s0[1]},
        {"shot_number": 2, "first_frame": s1[0], "last_frame": s1[1]},
    ]
    path.write_text(json.dumps({"shots": shots}))


class SpyFP:
    """
    Minimal frame provider:
      - next() for main loop
      - get_frame() for re-reads during embedding collection
      - stamp frame_idx into pixel [0,0,0:2] so DummyDetector can decode it.
    """
    def __init__(self, total=400, w=64, h=48, fps=30.0):
        self._total = int(total)
        self._w, self._h = int(w), int(h)
        self._fps = float(fps)
        self._idx = 0
        self._blank = np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def fps(self):
        return self._fps

    def size(self):
        return (self._w, self._h)

    def total_frames(self):
        return self._total

    def _frame_for(self, idx: int):
        img = self._blank.copy()
        img[0, 0, 0] = idx % 256
        img[0, 0, 1] = (idx // 256) % 256
        return img

    def get_frame(self, frame_idx: int):
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= self._total:
            raise IndexError(frame_idx)
        return self._frame_for(frame_idx)

    def reset_to_frame(self, i):
        self._idx = int(i)

    def next(self):
        if self._idx >= self._total:
            return None
        frame = self._frame_for(self._idx)
        self._idx += 1
        return frame


def _open_ckpt(tmp_path: Path, shots_path: Path, vid_name: str):
    """
    Build a CheckpointManager and mimic enough of the CLI lifecycle that:
      - obs/emb collectors exist
      - obs_ckpt.npz exists before the pipeline tries to read it
    """
    parent = tmp_path / "ck"
    parent.mkdir(exist_ok=True)

    vid = tmp_path / vid_name
    vid.write_text("x")

    opts = {
        "schema_version": "2.1",
        "video_path": str(vid),
        "detect_interval": 60,
        "embedding_batch_size_max": 32,
        "device": "cpu",
        "emb_store": "sidecar",
        "emb_sidecar_path": None,
        "obs_sidecar_path": None,
        "detector_model_path": "x",
        "embedding_model_path": "y",
        "yolo_config_path": "z",
        "shot_segmentation_path": str(shots_path),
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

    obs = ObservationsCollector()
    emb = EmbeddingCollector(mode="sidecar", dim=512, base_offset=0)

    ckpt.start(
        obs_collector=obs,
        emb_collector=emb,
        frames_done=0,
        shots_done=0,
        tracks_seen=0,
        options_snapshot=opts,
    )

    # Ensure obs_ckpt.npz exists (some paths read it while persisting embeddings)
    ckpt_dir = getattr(ckpt, "ckpt_dir", None) or (ckpt.root / "ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = ckpt_dir / "obs_ckpt.npz"
    if not p.exists():
        # Best-effort dtype selection
        if hasattr(ckpt, "_obs_dtype"):
            dtype = ckpt._obs_dtype()
        elif hasattr(CheckpointManager, "OBS_DTYPE"):
            dtype = CheckpointManager.OBS_DTYPE
        else:
            dtype = np.dtype([("f", "<i4"), ("shot", "<i4"), ("track_id", "<i4")])
        empty = np.zeros((0,), dtype=dtype)
        np.savez(p, observations=empty)

    return ckpt


def _is_detected(obs):
    s = getattr(obs, "source", "")
    if isinstance(s, str):
        return s.lower() == "detected"
    return getattr(s, "value", "").lower() == "detected"


def _lm5x2(obs):
    lm = getattr(obs, "landmarks", None)
    assert lm is not None, "DETECTED observation missing landmarks"
    arr = np.asarray(lm, dtype=np.float32)
    if arr.shape == (1, 5, 2):
        arr = arr[0]
    assert arr.shape == (5, 2), f"expected (5,2), got {arr.shape}"
    assert np.all(np.isfinite(arr)), "landmarks contain NaN/Inf"
    return arr


# --- The test ------------------------------------------------------------------

def test_straight_through_vs_resume_same_global_labels(tmp_path: Path, monkeypatch):
    """
    Cold run then resume run over the same two-shot plan:
      - Shot 1: one face (A)
      - Shot 2: same face (A) continues.

    Contract:
      (1) DETECTED observations always carry valid (5,2) landmarks
      (2) On resume, DETECTED landmarks remain semantically correct (arr[0,0]=frame_idx+1)
          for frames strictly after the resume anchor.
      (3) Global IDs for tracks strictly after the anchor match between cold and resume
          when re-resolved from the same seed on the post-anchor slices.
    """
    anchor = 180
    shots = tmp_path / "shots.json"
    _write_two_shots(shots)

    # Make alignment deterministic and enforce "landmarks must be present"
    def _dummy_align(frame, landmarks, frame_idx=None, source=None, *, return_meta=False):
        assert landmarks is not None, "aligner called with landmarks=None"
        lm = np.asarray(landmarks, dtype=np.float32)
        if lm.shape == (1, 5, 2):
            lm = lm[0]
        assert lm.shape == (5, 2)
        assert np.all(np.isfinite(lm))
        chip = np.zeros((112, 112, 3), dtype=np.uint8)
        if return_meta:
            return chip, {"frame_idx": frame_idx, "source": source}
        return chip

    monkeypatch.setattr(
        "facekit.pipeline.track_across_segments.align_face_for_arcface",
        _dummy_align,
        raising=True,
    )

    def _ordered(trs):
        return sorted(trs, key=lambda t: (getattr(t, "shot_id", 0), t.first_frame(), t.track_id))

    # -------------------- Cold run --------------------
    ckpt_cold = _open_ckpt(tmp_path, shots, "cold.mp4")
    cold_tracks = track_across_segments(
        frame_source=SpyFP(total=320),
        shot_json_path=str(shots),
        detector=DummyDetector(),
        embedder=DummyEmbedder(),
        checkpoint=ckpt_cold,
        resume_enabled=False,
    )
    cold_tracks = _ordered(cold_tracks)

    # Track-order map used by resume rehydrate ordering safety checks
    track_order = {(int(t.shot_id), int(t.track_id)): i for i, t in enumerate(cold_tracks)}
    assert track_order, "expected non-empty track_order"

    # Flush cold sidecars, then copy obs_ckpt.npz to resume run
    ckpt_cold.finalize()
    cold_ckpt_dir = getattr(ckpt_cold, "ckpt_dir", None) or (ckpt_cold.root / "ckpt")
    cold_obs = cold_ckpt_dir / "obs_ckpt.npz"
    assert cold_obs.exists(), f"cold obs sidecar missing: {cold_obs}"

    # Contract check: DET rows have landmarks and they encode frame_idx+1
    cold_det_frames = set()
    for t in cold_tracks:
        for o in getattr(t, "observations", []):
            if not _is_detected(o):
                continue
            cold_det_frames.add(int(o.frame_idx))
            arr = _lm5x2(o)
            expected = float(int(o.frame_idx) + 1)
            assert float(arr[0, 0]) == expected, (
                f"Cold landmark semantic mismatch at frame {o.frame_idx}: "
                f"expected arr[0,0]={expected}, got {arr[0,0]}"
            )
    assert cold_det_frames, "expected at least one DET observation in cold run"

    # -------------------- Resume run --------------------
    ckpt_resume = _open_ckpt(tmp_path, shots, "resume.mp4")

    # Provide ordering to rehydrate (avoid ResumeSafetyError)
    monkeypatch.setattr(ckpt_resume, "get_track_order", lambda: track_order, raising=False)
    monkeypatch.setattr(ckpt_resume, "read_track_order", lambda: track_order, raising=False)

    # Ensure resume obs sidecar matches cold sidecar
    resume_ckpt_dir = getattr(ckpt_resume, "ckpt_dir", None) or (ckpt_resume.root / "ckpt")
    resume_ckpt_dir.mkdir(parents=True, exist_ok=True)
    (resume_ckpt_dir / "obs_ckpt.npz").write_bytes(cold_obs.read_bytes())

    # Force the resume anchor
    ckpt_resume.get_resume_anchor = lambda: (anchor,)

    # (Optional consistency hints for logging)
    ckpt_resume._last_det_frame = anchor
    ckpt_resume._last_det_shot = 2
    ckpt_resume._last_det_shot_first_frame = 103

    resume_tracks = track_across_segments(
        frame_source=SpyFP(total=320),
        shot_json_path=str(shots),
        detector=DummyDetector(),
        embedder=DummyEmbedder(),
        checkpoint=ckpt_resume,
        resume_enabled=True,
    )
    resume_tracks = _ordered(resume_tracks)

    # Contract check: post-anchor DET landmarks encode frame_idx+1
    post_anchor_det_seen = False
    for t in resume_tracks:
        for o in getattr(t, "observations", []):
            if not _is_detected(o):
                continue
            f = int(o.frame_idx)
            if f <= anchor:
                continue
            post_anchor_det_seen = True
            arr = _lm5x2(o)
            expected = float(f + 1)
            assert float(arr[0, 0]) == expected, (
                f"Resume landmark corruption at frame {f}: "
                f"expected arr[0,0]={expected}, got {arr[0,0]}"
            )
    assert post_anchor_det_seen, "expected at least one post-anchor DET observation in resume run"

    # -------------------- Global ID contract (post-anchor) --------------------
    def _post_anchor(trs, a):
        return [t for t in trs if t.first_frame() > a]

    cold_post = _ordered(_post_anchor(cold_tracks, anchor))
    resm_post = _ordered(_post_anchor(resume_tracks, anchor))

    # Resolve IDs on post-anchor slices with same seed
    GlobalIdentityResolver().resolve_global_ids(cold_post, start_id=0)
    GlobalIdentityResolver().resolve_global_ids(resm_post, start_id=0)

    assert len(cold_post) == len(resm_post), (
        f"post-anchor (> {anchor}) track count differs: cold={len(cold_post)} resume={len(resm_post)}"
    )

    for a, b in zip(cold_post, resm_post):
        assert a.global_id == b.global_id, (
            "global label drift post-anchor (> anchor): "
            f"cold gid={a.global_id} vs resume gid={b.global_id} "
            f"at shot={getattr(a, 'shot_id', None)} frame={a.first_frame()}"
        )
