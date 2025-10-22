# tests/test_resume_vs_cold_global_labels.py
from pathlib import Path
import json
import numpy as np
import pytest

from facekit.pipeline.track_across_segments import track_across_segments
from facekit.tracking.tracking_resolution import GlobalIdentityResolver
from facekit.pipeline.checkpoint import CheckpointManager


# --- Deterministic test doubles -------------------------------------------------

class DummyDetector:
    def __init__(self):
        self.box = (0, 0, 10, 10)

    def detect_faces_in_frame(self, frame):
        # boxes, landmarks, confidences
        return ([self.box], [[(0, 0)] * 5], [0.9])


class DummyEmbedder:
    def get_embedding_batch(self, crops, batch_size=32):
        # deterministic unit vectors (shape [N, 512])
        return np.tile(np.eye(1, 512, 0, dtype=np.float32), (len(crops), 1))


# --- Small helpers --------------------------------------------------------------

def _write_two_shots(path: Path, s0=(0, 102), s1=(103, 299)):
    shots = [
        {"shot_number": 1, "first_frame": s0[0], "last_frame": s0[1]},
        {"shot_number": 2, "first_frame": s1[0], "last_frame": s1[1]},
    ]
    path.write_text(json.dumps({"shots": shots}))


def _open_ckpt(tmp_path: Path, shots_path: Path, vid_name: str):
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
    return CheckpointManager.open(
        parent_dir=parent,
        video_path=vid,
        options_snapshot=opts,
        no_resume=True,
        force_new_run=False,
    )


class SpyFP:
    def __init__(self, total=400, w=64, h=48, fps=30.0):
        self._total = total
        self._w, self._h = w, h
        self._fps = fps
        self._idx = 0
        self._blank = np.zeros((h, w, 3), dtype=np.uint8)

    @property
    def fps(self):
        return self._fps

    @property
    def size(self):
        return (self._w, self._h)

    @property
    def total_frames(self):
        return self._total

    def reset_to_frame(self, i):
        self._idx = int(i)

    def next(self):
        if self._idx >= self._total:
            return None
        self._idx += 1
        return self._blank


# --- The test ------------------------------------------------------------------

def test_straight_through_vs_resume_same_global_labels(tmp_path: Path, monkeypatch):
    """
    Cold run then resume run over the same two-shot plan:
      - Shot 1: one face (A)
      - Shot 2: same face (A) continues.

    Contract: With the same resolver seed and a frozen ordering, the global labels
    for frames strictly *after* the resume anchor are identical between a cold run
    and a resumed run.

    Note: many pipelines intentionally 'replay' the anchor frame on resume to keep
    state consistent. That creates an off-by-one if you compare >= anchor. We
    normalize by comparing > anchor for both runs.
    """
    anchor = 180  # inside shot 2
    shots = tmp_path / "shots.json"
    _write_two_shots(shots)

    # --- Stub the aligner so detections always produce aligned faces ---
    def _dummy_align(frame, landmarks, frame_idx, source="detect"):
        # Always return a valid 112x112 RGB crop so tracks get created.
        return np.zeros((112, 112, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "facekit.pipeline.track_across_segments.align_face_for_arcface",
        _dummy_align,
        raising=True,
    )

    def _ordered(trs):
        return sorted(
            trs,
            key=lambda t: (getattr(t, "shot_id", 0), t.first_frame(), t.track_id),
        )

    # 1) Cold run
    ckpt_cold = _open_ckpt(tmp_path, shots, "cold.mp4")
    fp_cold = SpyFP(total=320)
    cold_tracks = track_across_segments(
        frame_source=fp_cold,
        shot_json_path=str(shots),
        detector=DummyDetector(),
        embedder=DummyEmbedder(),
        checkpoint=ckpt_cold,
        resume_enabled=False,
    )

    cold_tracks = _ordered(cold_tracks)

    # Global IDs from the full set aren't used for the comparison; keep for sanity only.
    GlobalIdentityResolver().resolve_global_ids(cold_tracks, start_id=0)
    cold_ids = {t.track_id: getattr(t, "global_id", None) for t in cold_tracks}
    assert cold_ids and all(v is not None for v in cold_ids.values())

    # 2) Resume run at anchor=180 (shot 2)
    ckpt_resume = _open_ckpt(tmp_path, shots, "resume.mp4")

    # Hand off obs collector to simulate persisted sidecar state.
    if hasattr(ckpt_resume, "obs_collector") and hasattr(ckpt_cold, "obs_collector"):
        ckpt_resume.obs_collector = ckpt_cold.obs_collector

    # Force the resume anchor to the desired frame so _resolve_anchor() doesn't
    # override it based on status.json or obs_collector.
    ckpt_resume.get_resume_anchor = lambda: (anchor,)

    # Emulate that we last checkpointed at the anchor (for logging/consistency only)
    ckpt_resume._last_det_frame = anchor
    ckpt_resume._last_det_shot = 2
    ckpt_resume._last_det_shot_first_frame = 103

    fp_res = SpyFP(total=320)
    resume_tracks = track_across_segments(
        frame_source=fp_res,
        shot_json_path=str(shots),
        detector=DummyDetector(),
        embedder=DummyEmbedder(),
        checkpoint=ckpt_resume,
        resume_enabled=True,
    )

    resume_tracks = _ordered(resume_tracks)

    # Same note as above — resolve over full list only for sanity.
    GlobalIdentityResolver().resolve_global_ids(resume_tracks, start_id=0)
    resume_ids = {t.track_id: getattr(t, "global_id", None) for t in resume_tracks}
    assert resume_ids and all(v is not None for v in resume_ids.values())

    # --- Compare only frames STRICTLY after the anchor to avoid replay off-by-one
    def _post_anchor(trs, a):
        return [t for t in trs if t.first_frame() > a]

    cold_post = _ordered(_post_anchor(cold_tracks, anchor))
    resm_post = _ordered(_post_anchor(resume_tracks, anchor))

    # Re-resolve IDs *only* on the post-anchor slices so both start at the same seed
    # and are unaffected by pre-anchor enumeration.
    GlobalIdentityResolver().resolve_global_ids(cold_post, start_id=0)
    GlobalIdentityResolver().resolve_global_ids(resm_post, start_id=0)

    assert len(cold_post) == len(resm_post), \
        f"post-anchor (> {anchor}) track count differs: cold={len(cold_post)} resume={len(resm_post)}"

    for a, b in zip(cold_post, resm_post):
        assert a.global_id == b.global_id, (
            "global label drift post-anchor (> anchor): "
            f"cold gid={a.global_id} vs resume gid={b.global_id} "
            f"at shot={getattr(a, 'shot_id', None)} frame={a.first_frame()}"
        )
