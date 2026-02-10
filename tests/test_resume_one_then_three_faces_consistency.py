# tests/test_resume_one_then_three_faces_consistency.py

import json
import os
import re
import textwrap
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from facekit.common.obs_consts import src_to_code, Source

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
    m = re.search(r"EMB_SAFE_ANCHOR:(\d+)", txt)
    assert m, f"Could not parse anchor from logs:\n{txt}"
    return int(m.group(1))

def _extract_mark_safe_frames(stdout: str, stderr: str) -> list[int]:
    txt = stdout + "\n" + stderr
    m = re.search(r"MARK_SAFE_FRAMES:([0-9,]*)", txt)
    assert m, f"Could not parse MARK_SAFE_FRAMES from logs:\n{txt}"
    blob = m.group(1).strip()
    if not blob:
        return []
    return [int(x) for x in blob.split(",") if x.strip()]

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
import facekit.pipeline.track_across_segments as _tas_mod
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

def _force_embed_queue_max_pending(max_pending: int) -> None:
    """
    Make the embed queue deterministic for this integration test by forcing max_pending.

    We do NOT want this test to depend on any particular options key name being plumbed
    through production. We patch the queue constructor used by track_across_segments.
    """
    from facekit.embedding.embedding_queue import AlignedFaceEmbeddingQueue as _Q

    class _TestQueue(_Q):
        def __init__(self, *a, **k):
            k.setdefault("max_pending", int(max_pending))
            super().__init__(*a, **k)

    _tas_mod.AlignedFaceEmbeddingQueue = _TestQueue  # type: ignore[attr-defined]

 
def _main():
    import argparse
    import facekit.pipeline.track_across_segments as mod

    mod.align_face_for_arcface = _det_align

    # Deterministic contract for this test: flush fence is max_pending=7.
    _force_embed_queue_max_pending(7)

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
        "embedding_batch_size_max": 7,
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

    # Record exactly which frames the pipeline marks as embedding-safe.
    _mark_safe_frames = []
    _orig_mark_embedding_safe = getattr(mgr, "mark_embedding_safe", None)
    if _orig_mark_embedding_safe is not None:
        def _wrapped_mark_embedding_safe(*a, **k):
            # Accept either positional or keyword `frame_idx`.
            if "frame_idx" in k:
                f = int(k["frame_idx"])
            elif len(a) >= 1:
                f = int(a[0])
            else:
                f = -1
            _mark_safe_frames.append(f)
            return _orig_mark_embedding_safe(*a, **k)
        mgr.mark_embedding_safe = _wrapped_mark_embedding_safe

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
            embedding_queue_max_pending=7,
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
        anchor = int(status.get("last_embedding_safe_frame") or 0)
    except Exception:
        anchor = 0

    # Emit the safe-frame marks for the outer test to validate.
    try:
        print("MARK_SAFE_FRAMES:" + ",".join(str(x) for x in _mark_safe_frames), flush=True)
    except Exception:
        print("MARK_SAFE_FRAMES:", flush=True)

    print(f"EMB_SAFE_ANCHOR:{anchor}", flush=True)
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
      - Embeddings are persisted in batches (flush occurs when the aligned-face cache reaches
        embedding_batch_size_max, and possibly at end-of-shot).
      - For this test we use embedding_batch_size_max=7, and we assume every detection produces a
        valid aligned faces per every face present in the frame.
      - For this test, Shot 1 has 1 face per frame, Shot 2 has 3 faces per frame.
      - Anchor is the last embedding-safe frame.
      - IMPORTANT:
        "embedding payload frame" and "embedding-safe frame" are not always the same thing.
        The embedding payload frame is the newest frame that actually contributed embeddings in the
        flush/drain batch. The embedding-safe frame is the frame whose processing caused that
        flush/drain boundary to complete. Once that boundary completes, all embeddings up to the
        embedding-safe frame are durable.
      - IMPORTANT: the aligned-face cache is RESET at shot boundaries.
        At each shot boundary we do a FINAL FLUSH of whatever remains in the cache.
      - IMPORTANT:
        The first frame of a new shot is always a detection frame, even when that frame is not
        aligned to a detect_interval. For this test, the first frame of shot 2 is not aligned
        with the detection interval.

    With detect_interval=10 and crash at frame 134:
      - Shot 1 (frames <= 102) has 1 face per detection frame (0..100 step 10): 11 faces.
        With max_pending=7:
          * We flush at frame 60 (the 7th face arrives).
          * 4 remain queued aligned faces (at frames 70,80,90,100) are flushed at the shot boundary.
        For Shot 1:
          * frame 60 is both the embedding payload frame and the embedding-safe frame for the first flush
          * frame 100 is the embedding payload frame for the shot-end drain
          * frame 102 is the embedding-safe frame for the shot-end drain, because the drain is called
            while processing the final frame of the shot
      - Shot 2 begins with an EMPTY cache (reset at shot boundary).
        With detect_interval=10, the cadence-aligned detection frames before crashing at 134 are:
        110, 120, 130.
      - Shot 2 begins on frame 103 which is treated as a detection frame (as happens at every shot boundary).
        Therefore Shot 2 detection frames before crashing at 134 are: 103, 110, 120, 130.

        Each detection frame in Shot 1 has 1 face.
        Pending and persisted progression:
          * after 0  -> 1 pending, 0 total persisted
          * after 10 -> 2 pending, 0 total persisted
          * after 20 -> 3 pending, 0 total persisted
          * after 30 -> 4 pending, 0 total persisted
          * after 40 -> 5 pending, 0 total persisted
          * after 50 -> 6 pending, 0 total persisted
          * at 60 -> 7 pending, flush triggers and persists all 7; queue empty
                     7 total persisted
          * after 70 -> 1 pending, 7 total persisted
          * after 80 -> 2 pending, 7 total persisted
          * after 90 -> 3 pending, 7 total persisted
          * after 100 -> 4 pending, 7 total persisted
          * at shot end (102) -> end-of-shot drain persists remaining 4; queue empty
                                 embedding payload frame = 100, embedding-safe frame = 102
                                 11 total persisted

        Each detection frame in Shot 2 has 3 faces.
        Pending and persisted progression:
          * after 103 -> 3 pending, 11 total persisted
          * after 110 -> 6 pending, 11 total persisted
          * at 120, first face -> 7 pending, flush triggers and persists all 7; queue empty
                                 embedding payload frame = 120, embedding-safe frame = 120
                                 18 total persisted
                    2 more faces remain from that frame -> 2 pending
                                 end-of-frame drain persists those 2; queue empty
                                 20 total persisted
          * after 130 -> 3 pending, 20 total persisted
          * at 134 -> crash; frame-130 faces are not yet persisted
                      Final: 20 total persisted

        Therefore the crash-run embedding-safe marks should be [60, 102, 120].
        The final embedding-safe frame before the crash is 120, with 20 embeddings persisted.
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
        "--crash-frame", "134",
        ok=(0, 1, 2),
        env=_with_repo_env(),
    )
    anchor = _extract_anchor(cp_crash.stdout, cp_crash.stderr)
    mark_safe_frames = _extract_mark_safe_frames(cp_crash.stdout, cp_crash.stderr)
    assert mark_safe_frames, f"expected at least one mark_embedding_safe call, got none.\n{cp_crash.stdout}\n{cp_crash.stderr}"

    # Current contract:
    # - embedding-safe refers to the frame whose processing completed a flush/drain boundary
    # - shot 1 contributes safe marks at 60 and 102:
    #     * 60 when the 7th pending face arrives
    #     * 102 when the remaining 4 pending faces are drained at shot end
    # - shot 2 begins at frame 103, which is always treated as a detection frame at shot start
    # - then scheduled detections occur at 110 and 120
    # - frame 120 crosses the max_pending=7 flush fence and becomes the last embedding-safe frame
    #   before the crash at 134
    assert mark_safe_frames == [60, 102, 120], (
        f"expected safe marks [60, 102, 120], got {mark_safe_frames}"
    )

    # Current contract:
    # Final crash-run anchor remains 120 because shot 2 advances the safe point beyond shot 1.
    # Note that Shot 1's last embedding payload frame is 100, but its shot-end embedding-safe frame is 102.
    # - At start of Shot 2 Frame 103 is detected immediately
    # - then scheduled detections occur at 110 and 120
    # - frame 120 crosses the max_pending=7 flush fence and becomes the last embedding-safe frame
    #   before the crash at 134
    assert max(mark_safe_frames) == 120, (
        f"expected mark_embedding_safe max frame 120, got {max(mark_safe_frames)}. "
        f"all={mark_safe_frames} anchor={anchor}"
    )
    assert anchor == 120, (
        f"expected anchor 120, got {anchor}. mark_safe_frames={mark_safe_frames}"
    )

    # ---- Contract check on the crash-run sidecars ----
    obs_pre = _load_obs_npz(run_obs_npz)
    emb_pre = _load_emb_npz(run_emb_npz)
    emb_count_pre = int(emb_pre.shape[0])

    DET_CODE = int(src_to_code(Source.DETECTED.value))

    # Shot 1 is completed by anchor, so all its DETECTED rows must have embeddings.
    shot1_det = obs_pre[(obs_pre["shot"] == 1) & (obs_pre["src"] == DET_CODE)]

    # Single face => exactly one DET row per detection frame (0..100 step 10)
    det_frames = sorted(set(int(f) for f in shot1_det["f"].astype(int).tolist()))
    assert det_frames == list(range(0, 101, 10)), f"unexpected shot1 DET frames: {det_frames}"
    assert shot1_det.shape[0] == 11, f"expected 11 DET rows for shot1 (single face), got {shot1_det.shape[0]}"

    emb_idx = shot1_det["emb_idx"].astype(int)
    # Deterministic invariant for this test:
    # - With max_pending=7 and 1 face/detection in shot 1:
    #   * frame 60 triggers the 7th pending -> flush, so 60 is both payload frame and safe frame
    #   * the remaining 4 (frames 70,80,90,100) flush at shot boundary
    #   * for that shot-boundary drain, 100 is the embedding payload frame, while 102 is the
    #     embedding-safe frame because the drain completes while processing the final frame
    # - Therefore ALL shot1 DET rows must have an embedding pre-crash.
    assert np.all(emb_idx >= 0), f"found missing emb_idx on shot1 DET rows: {emb_idx.tolist()}"

    # Shot 1 persistence contract:
    # - flush fence max_pending=7 is reached at frame 60
    # - the first 7 detection frames therefore persist before frame 70 arrives
    # - the remaining 4 detection frames (70,80,90,100) persist at the shot boundary drain
    # - because shot 1 has exactly one face per detection frame, the persisted embedding rows
    #   referenced by shot-1 DET observations are exactly indices 0..10
    shot1_emb_idx = shot1_det["emb_idx"].astype(int)
    assert sorted(shot1_emb_idx.tolist()) == list(range(11)), (
        "shot1 DET rows should reference exactly the first 11 persisted embeddings "
        "(one per detection frame from 0 through 100). "
        f"got {sorted(shot1_emb_idx.tolist())}"
    )
    # The persisted embedding payload for shot 1 ends at frame 100, even though the shot-end
    # embedding-safe mark should be 102 under the current intended contract.
    assert int(shot1_emb_idx.min()) == 0
    assert int(shot1_emb_idx.max()) == 10

    # Shot 2 pre-crash contract:
    # - The new shot starts at frame 103.
    # - Because the shot begins with no active tracker / no open tracks, the pipeline performs an
    #   immediate detection at 103 even though 103 is not on the detect_interval cadence.
    # - With 3 faces per detection frame and max_pending=7:
    #     103 -> 3 pending
    #     110 -> 6 pending
    #     120 -> first face reaches 7, causing a flush; end-of-frame drain persists the other two
    #            faces from frame 120 as well; for shot 2, payload frame and safe frame are both 120
    # - Therefore pre-crash DET observation rows in shot 2 may exist for frames 103, 110, 120, 130.
    # - Frame 130 may appear as a DET observation row because observations can be persisted ahead
    #   of embedding durability.
    # - However, only frames 103, 110, and 120 are embedding-safe before the crash.
    # - Frame 130 observations exist, but their embeddings are not yet persisted/linked.

    # Absolute persisted-embedding count pre-crash:
    # - shot 1: 11 faces persisted (7 at frame 60 + 4 at shot boundary)
    # - shot 2 through frame 120: 9 faces persisted (3 @ 103, 3 @ 110, 3 @ 120)
    # Total = 20
    assert emb_count_pre == 20, f"expected 20 persisted embeddings pre-crash, got {emb_count_pre}"

    shot2_det = obs_pre[(obs_pre["shot"] == 2) & (obs_pre["src"] == DET_CODE)]
    shot2_det_frames = sorted(set(int(f) for f in shot2_det["f"].astype(int).tolist()))
    assert shot2_det_frames == [103, 110, 120, 130], (
        "unexpected DET frames for shot 2 before crash; "
        "under the current contract, DET observations may already exist for frame 130 "
        "even though frame 130 is not yet embedding-safe.\n"
        f"got {shot2_det_frames}"
    )

    # Separate persisted-vs-not-yet-persisted by emb_idx linkage.
    shot2_det_safe = shot2_det[shot2_det["emb_idx"].astype(int) >= 0]
    shot2_det_unsafe = shot2_det[shot2_det["emb_idx"].astype(int) < 0]

    shot2_det_safe_frames = sorted(set(int(f) for f in shot2_det_safe["f"].astype(int).tolist()))
    shot2_det_unsafe_frames = sorted(set(int(f) for f in shot2_det_unsafe["f"].astype(int).tolist()))

    assert shot2_det_safe_frames == [103, 110, 120], (
        "unexpected embedding-safe DET frames for shot 2 before crash; "
        "only 103, 110, and 120 should have persisted embeddings pre-crash.\n"
        f"got {shot2_det_safe_frames}"
    )
    assert shot2_det_unsafe_frames == [130], (
        "unexpected non-embedding-safe DET frames for shot 2 before crash; "
        "frame 130 should exist only as unembedded DET observations at crash time.\n"
        f"got {shot2_det_unsafe_frames}"
    )

    assert shot2_det_safe.shape[0] == 9, (
        f"expected 9 embedding-safe DET rows for shot 2 pre-crash "
        f"(3 faces x 3 frames: 103, 110, 120), got {shot2_det_safe.shape[0]}"
    )
    assert shot2_det_unsafe.shape[0] == 3, (
        f"expected 3 non-embedding-safe DET rows for frame 130 pre-crash, "
        f"got {shot2_det_unsafe.shape[0]}"
    )

    # emb_idx should reference valid embedding rows.
    max_idx = int(emb_idx.max()) if emb_idx.size else -1
    assert max_idx == 10, f"expected max emb_idx 10 for shot1 (0-based, 11 rows), got {max_idx}"
    assert emb_count_pre > max_idx, (
        f"embedding sidecar too small for shot1 DET rows: "
        f"emb_count={emb_count_pre}, max_emb_idx={max_idx}"
    )

    # Across both shots, pre-crash persisted embeddings should cover:
    # - shot 1 frames 0..100 inclusive on the 10-frame cadence (11 embeddings total)
    # - shot 2 frames 103, 110, 120 with 3 faces each (9 embeddings total)
    # This also documents why the pre-crash anchor is 120 rather than 130, and why Shot 1's
    # final embedding payload frame (100) differs from its shot-end embedding-safe frame (102).
    shot2_max_idx = int(shot2_det["emb_idx"].astype(int).max()) if shot2_det.size else -1
    assert shot2_max_idx == 19, (
        f"expected last pre-crash embedding index to be 19, got {shot2_max_idx}"
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

    def _post_anchor_segments(trs):
        out = []
        for t in trs:
            first = int(t["first_frame"])
            last = int(t["last_frame"])
            if last <= anchor:
                continue
            out.append({
                "shot_id": int(t["shot_id"]),
                "track_id": int(t["track_id"]),
                "global_id": int(t["global_id"]),
                "first_frame": max(first, anchor + 1),
                "last_frame": last,
            })
        out.sort(key=lambda t: (t["shot_id"], t["first_frame"], t["last_frame"], t["track_id"]))
        return out

    cold_post = _post_anchor_segments(cold_tracks)
    resume_post = _post_anchor_segments(resume_tracks)

    assert len(cold_post) == len(resume_post), (
        f"post-anchor segment count drift: cold={len(cold_post)} resume={len(resume_post)} "
        f"(anchor={anchor}, cold={cold_post}, resume={resume_post})"
    )

    for a, b in zip(cold_post, resume_post):
        assert int(a["shot_id"]) == int(b["shot_id"])
        assert int(a["first_frame"]) == int(b["first_frame"])
        assert int(a["last_frame"]) == int(b["last_frame"])
        assert int(a["global_id"]) == int(b["global_id"]), (
            f"GID drift: cold={a['global_id']} resume={b['global_id']} "
            f"shot={a['shot_id']} span=[{a['first_frame']},{a['last_frame']}]"
        )