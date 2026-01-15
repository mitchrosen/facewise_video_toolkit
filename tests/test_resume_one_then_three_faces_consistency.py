# tests/test_resume_one_then_three_faces_consistency.py

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


# -------------------- helpers --------------------

def _with_repo_env() -> dict:
    pkg_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(pkg_root) + (os.pathsep + existing if existing else "")
    return env


def _run_python(shim: Path, *args, ok=(0,), env=None, cwd=None):
    cp = subprocess.run(
        [sys.executable, str(shim), *args],
        text=True,
        capture_output=True,
        env=env,
        cwd=cwd or shim.parent,
    )
    if cp.returncode not in ok:
        raise AssertionError(
            f"RC {cp.returncode} not in {ok}\n=== STDOUT ===\n{cp.stdout}\n=== STDERR ===\n{cp.stderr}\n"
        )
    return cp


def _extract_anchor(stdout: str, stderr: str) -> int:
    txt = stdout + "\n" + stderr
    m = re.search(r"ANCHOR:(\d+)", txt)
    assert m, f"Could not parse anchor from logs:\n{txt}"
    return int(m.group(1))


def _load_obs_npz(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        assert "observations" in data.files, f"sidecar missing observations: {npz_path}"
        return data["observations"]


def _load_emb_npz(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        assert "embeddings" in data.files, f"sidecar missing embeddings: {npz_path}"
        return data["embeddings"]


def _ordered_tracks_json(json_path: Path):
    js = json.loads(json_path.read_text())
    tracks = js.get("tracks", [])

    def _ff(t):
        return int(t.get("first_frame", 10**9))

    tracks.sort(key=lambda t: (int(t.get("shot_id", 0)), _ff(t), int(t.get("track_id", 0))))
    return tracks


# -------------------- subprocess shim --------------------

SHIM = r'''
import sys
import json
import traceback
from pathlib import Path

import numpy as np

from facekit.pipeline.checkpoint import CheckpointManager
from facekit.pipeline.track_across_segments import track_across_segments
from facekit.tracking.tracking_resolution import GlobalIdentityResolver

from facekit.output.json_v2 import ObservationsCollector, EmbeddingCollector


def _det_align(frame, landmarks, frame_idx=None, source=None, *, return_meta=False):
    arr = np.zeros((10,10,3), np.uint8)
    arr.flags.writeable = True
    if return_meta:
        return arr, {"frame_idx": frame_idx, "source": source}
    return arr


class SpyFP:
    def __init__(self, total=320, w=64, h=48, fps=30.0):
        self._total = int(total)
        self._w = int(w)
        self._h = int(h)
        self._fps = float(fps)
        self._idx = 0
        self._blank = np.zeros((h,w,3), np.uint8)

    @property
    def fps(self): return self._fps
    @property
    def size(self): return (self._w, self._h)
    @property
    def total_frames(self): return self._total

    def reset_to_frame(self, i): self._idx = int(i)

    def next(self):
        if self._idx >= self._total:
            return None
        self._idx += 1
        return self._blank

    def get_frame(self, frame_idx: int):
        if frame_idx < 0 or frame_idx >= self._total:
            raise IndexError(f"frame_idx out of range: {frame_idx}")
        return self._blank


class DummyDetector:
    def __init__(self, fp, shot1_last=102):
        self.fp = fp
        self.s1 = int(shot1_last)

    def detect_faces_in_frame(self, frame):
        fidx = self.fp._idx - 1
        if fidx <= self.s1:
            boxes = [(5,5,15,15)]
        else:
            boxes = [(5,5,15,15), (25,5,35,15), (45,5,55,15)]

        conf = [0.99]*len(boxes)

        def cx(b): return (int(b[0]) + int(b[2])) // 2
        landmarks = [[(cx(b), 0)] + [(0,0)]*4 for b in boxes]
        return boxes, landmarks, conf


class _DummyEmbedder:
    def __init__(self, *a, **k):
        pass
    def get_embedding_batch(self, chips, batch_size=None, **kwargs):
        vecs = []
        for chip in chips:
            h = int(np.uint64(chip.sum() + chip.shape[0]*1009 + chip.shape[1]*2741))
            rng = np.random.RandomState(h % (2**32))
            v = rng.rand(512).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            vecs.append(v)
        return np.stack(vecs, axis=0)
    def get_embedding(self, chip, **kwargs):
        return self.get_embedding_batch([chip], **kwargs)[0]


class CrashyCheckpoint:
    def __init__(self, inner, crash_frame: int):
        self._inner = inner
        self._crash_frame = crash_frame

    def on_frame(self, frame_idx: int) -> None:
        if self._crash_frame is not None and frame_idx == self._crash_frame:
            raise RuntimeError(f"boom at frame {frame_idx} (injected crash)")
        return self._inner.on_frame(frame_idx)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _main():
    import argparse
    import facekit.pipeline.track_across_segments as mod

    mod.align_face_for_arcface = _det_align

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cold","crash","resume"], required=True)
    p.add_argument("--shots-json", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--obs-npz", required=True)
    p.add_argument("--emb-npz", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--detect-interval", type=int, default=10)
    p.add_argument("--crash-frame", type=int, default=None)
    args = p.parse_args()

    shots_path = Path(args.shots_json)

    opts = {
        "schema_version": "2.1",
        "video_path": str(Path(args.ckpt_dir, "dummy.mp4")),
        "detect_interval": args.detect_interval,
        "embedding_batch_size_max": 32,
        "device": "cpu",

        "emb_store": "sidecar",
        "emb_sidecar_path": str(args.emb_npz),
        "obs_sidecar_path": str(args.obs_npz),

        "detector_model_path": "x",
        "embedding_model_path": "y",
        "yolo_config_path": "z",
        "shot_segmentation_path": str(shots_path),
        "log_level": "INFO",
        "log_file": None,
    }

    no_resume = (args.mode != "resume")
    force_new = (args.mode != "resume")

    mgr = CheckpointManager.open(
        parent_dir=Path(args.ckpt_dir),
        video_path=Path(opts["video_path"]),
        options_snapshot=opts,
        no_resume=no_resume,
        force_new_run=force_new,
        resume_latest=(args.mode == "resume"),
    )

    obs = ObservationsCollector()
    emb = EmbeddingCollector("sidecar", dim=512)

    # IMPORTANT: do NOT preload sidecars into collectors.
    # - cold/crash are independent runs that write their own sidecars
    # - resume rehydrates from checkpoint; preloading obs causes duplicates
    mgr.start(obs, emb)

    fp = SpyFP(total=320)
    det = DummyDetector(fp, shot1_last=102)

    if args.mode == "crash" and args.crash_frame is not None:
        mgr = CrashyCheckpoint(mgr, args.crash_frame)

    exit_code = 0
    tracks = []
    try:
        tracks = track_across_segments(
            frame_source=fp,
            shot_json_path=str(shots_path),
            detector=det,
            embedder=_DummyEmbedder(),
            checkpoint=mgr,
            detect_interval=args.detect_interval,
            resume_enabled=(args.mode == "resume"),
        )

        tracks = sorted(tracks, key=lambda t: (int(getattr(t,"shot_id",0)), t.first_frame(), int(getattr(t,"track_id",0))))
        GlobalIdentityResolver().resolve_global_ids(tracks, start_id=0)

    except Exception:
        traceback.print_exc()
        exit_code = 2
    finally:
        try:
            obs.dump_npz(Path(args.obs_npz))
        except Exception:
            traceback.print_exc()
            if exit_code == 0: exit_code = 3

        try:
            emb.dump_npz(Path(args.emb_npz))
        except Exception:
            traceback.print_exc()
            if exit_code == 0: exit_code = 3

    try:
        out = {"tracks": []}
        for t in tracks:
            out["tracks"].append({
                "shot_id": int(getattr(t,"shot_id",0)),
                "track_id": int(getattr(t,"track_id",0)),
                "global_id": int(getattr(t,"global_id",0)),
                "first_frame": int(t.first_frame()),
                "last_frame": int(t.last_frame()),
            })
        Path(args.out_json).write_text(json.dumps(out))
    except Exception:
        traceback.print_exc()
        if exit_code == 0: exit_code = 4

    try:
        status = mgr.read_status() or {}
        anchor = int(status.get("last_detection_frame") or 0)
    except Exception:
        anchor = 0

    print(f"ANCHOR:{anchor}", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    _main()
'''


# -------------------- test --------------------

@pytest.mark.integration
def test_resume_three_phase_isolated(tmp_path: Path):
    """
    Three-phase cold / crash / resume with 1→3 faces.

    Contract:
      - Completed shots must have embeddings persisted for DETECTED frames.
      - The anchor shot may have no embeddings persisted yet.

    Under this contract, anchor is the last *completed-shot* detection frame.
    Shot 1 detection frames are 0,10,...,100, so expected anchor is 100.
    """
    shim = tmp_path / "run_three_phase.py"
    shim.write_text(SHIM)

    shots = {
        "shots": [
            {"shot_number": 1, "first_frame": 0, "last_frame": 102},
            {"shot_number": 2, "first_frame": 103, "last_frame": 299},
        ]
    }
    shots_path = tmp_path / "shots.json"
    shots_path.write_text(json.dumps(shots))

    ckpt_parent = tmp_path / "ckpt_parent"
    ckpt_parent.mkdir()

    cold_json = tmp_path / "cold_tracks.json"
    crash_json = tmp_path / "crash_tracks.json"
    resume_json = tmp_path / "resume_tracks.json"

    # IMPORTANT: isolate cold sidecars from crash/resume sidecars
    cold_obs_npz = tmp_path / "cold_obs_sidecar.npz"
    cold_emb_npz = tmp_path / "cold_emb_sidecar.npz"

    run_obs_npz = tmp_path / "run_obs_sidecar.npz"
    run_emb_npz = tmp_path / "run_emb_sidecar.npz"

    # ---- A) Cold run (baseline) ----
    _run_python(
        shim,
        "--mode", "cold",
        "--shots-json", str(shots_path),
        "--ckpt-dir", str(ckpt_parent),
        "--detect-interval", "10",
        "--obs-npz", str(cold_obs_npz),
        "--emb-npz", str(cold_emb_npz),
        "--out-json", str(cold_json),
        ok=(0,),
        env=_with_repo_env(),
    )
    cold_tracks = _ordered_tracks_json(cold_json)

    # ---- B) Crash run (creates checkpoint + sidecars for the run-to-resume) ----
    cp_crash = _run_python(
        shim,
        "--mode", "crash",
        "--shots-json", str(shots_path),
        "--ckpt-dir", str(ckpt_parent),
        "--detect-interval", "10",
        "--obs-npz", str(run_obs_npz),
        "--emb-npz", str(run_emb_npz),
        "--out-json", str(crash_json),
        "--crash-frame", "124",
        ok=(0, 1, 2),
        env=_with_repo_env(),
    )
    anchor = _extract_anchor(cp_crash.stdout, cp_crash.stderr)
    assert anchor == 120, f"expected anchor 120 for crash at frame 124 under completed-shot anchoring, got {anchor}"

    # ---- Contract check on the crash-run sidecars ----
    obs_pre = _load_obs_npz(run_obs_npz)
    emb_pre = _load_emb_npz(run_emb_npz)
    emb_count_pre = int(emb_pre.shape[0])

    DET_CODE = 0

    # Shot 1 is completed by anchor, so all its DETECTED rows must have embeddings.
    shot1_det = obs_pre[(obs_pre["shot"] == 1) & (obs_pre["src"] == DET_CODE)]

    # Single face => exactly one DET row per detection frame (0..100 step 10)
    det_frames = sorted(set(int(f) for f in shot1_det["f"].astype(int).tolist()))
    assert det_frames == list(range(0, 101, 10)), f"unexpected shot1 DET frames: {det_frames}"
    assert shot1_det.shape[0] == 11, f"expected 11 DET rows for shot1 (single face), got {shot1_det.shape[0]}"

    emb_idx = shot1_det["emb_idx"].astype(int)
    assert np.all(emb_idx >= 0), f"found missing emb_idx on completed-shot DET rows: {emb_idx.tolist()}"

    max_idx = int(emb_idx.max()) if emb_idx.size else -1
    assert emb_count_pre > max_idx, (
        f"embedding sidecar too small for completed-shot DET rows: "
        f"emb_count={emb_count_pre}, max_emb_idx={max_idx}"
    )

    # ---- C) Resume run (must use the crash-run sidecars) ----
    _run_python(
        shim,
        "--mode", "resume",
        "--shots-json", str(shots_path),
        "--ckpt-dir", str(ckpt_parent),
        "--detect-interval", "10",
        "--obs-npz", str(run_obs_npz),
        "--emb-npz", str(run_emb_npz),
        "--out-json", str(resume_json),
        ok=(0,),
        env=_with_repo_env(),
    )
    resume_tracks = _ordered_tracks_json(resume_json)

    # ---- Assertions ----

    post_obs = _load_obs_npz(run_obs_npz)
    assert post_obs.size > 0, "resume wrote an empty observations sidecar"

    def _post_anchor(trs):
        return [t for t in trs if int(t["first_frame"]) > anchor]

    cold_post = _post_anchor(cold_tracks)
    resume_post = _post_anchor(resume_tracks)

    assert len(cold_post) == len(resume_post), (
        f"post-anchor track count drift: cold={len(cold_post)} resume={len(resume_post)} (anchor={anchor})"
    )

    for a, b in zip(cold_post, resume_post):
        assert int(a["shot_id"]) == int(b["shot_id"])
        assert int(a["first_frame"]) == int(b["first_frame"])
        assert int(a["last_frame"]) == int(b["last_frame"])
        assert int(a["global_id"]) == int(b["global_id"]), (
            f"GID drift: cold={a['global_id']} resume={b['global_id']} "
            f"shot={a['shot_id']} span=[{a['first_frame']},{a['last_frame']}]"
        )
