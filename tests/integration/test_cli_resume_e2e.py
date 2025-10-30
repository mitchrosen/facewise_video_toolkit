import json
import sys
from pathlib import Path

import numpy as np
import pytest


# ------------------------- tiny helpers & dummies -------------------------

# --- add near the other helpers ---
class _EmitOneThenCrash:
    """
    Emits exactly one bbox per frame (so obs rows grow), then raises at N to simulate a mid-run crash.
    """
    def __init__(self, crash_at: int):
        self.crash_at = crash_at
        self.n = 0
    def detect_faces_in_frame(self, frame, frame_index=None):
        import numpy as np
        self.n += 1
        if self.n >= self.crash_at:
            raise RuntimeError("boom (injected)")
        # boxes_xyxy, landmarks(any shape tolerated), confidences
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        lms   = np.zeros((1, 5, 2), dtype=np.float32)  # shape not used by the pipeline in this test
        conf  = np.array([0.99], dtype=np.float32)
        return boxes, lms, conf

class _EmitOneNoCrash(_EmitOneThenCrash):
    def __init__(self):
        super().__init__(crash_at=10**9)
    def detect_faces_in_frame(self, frame, frame_index=None):
        import numpy as np
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        lms   = np.zeros((1, 5, 2), dtype=np.float32)
        conf  = np.array([0.99], dtype=np.float32)
        return boxes, lms, conf
    
class _CrashAfterNDetections:
    """Detector stub: increments on each detect() and raises at N to simulate a crash."""
    def __init__(self, crash_at: int):
        self.crash_at = crash_at
        self.n = 0
    def detect_faces_in_frame(self, frame, frame_index=None):
        self.n += 1
        if self.n >= self.crash_at:
            raise RuntimeError("boom (injected)")
        # boxes_xyxy, landmarks, confidences
        return [], [], []

class _NoCrashDetector(_CrashAfterNDetections):
    def __init__(self):
        super().__init__(crash_at=10**9)
    def detect_faces_in_frame(self, frame, frame_index=None):
        return [], [], []

class _DummyEmbedder:
    def __init__(self, *a, **k):
        pass

    # Return a deterministic 512-dim vector per chip based on pixel content
    def get_embedding_batch(self, chips):
        import numpy as np
        vecs = []
        for chip in chips:
            # hash-like reduction: same chip -> same vector; independent of process
            h = int(np.uint64(chip.sum() + chip.shape[0]*1009 + chip.shape[1]*2741))
            rng = np.random.RandomState(h % (2**32))
            v = rng.rand(512).astype(np.float32)
            # normalize like ArcFace
            v /= np.linalg.norm(v) + 1e-12
            vecs.append(v)
        return np.stack(vecs, axis=0)
    
def _make_tiny_video(path: Path, frames=60, size=(192, 108)):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, size)
    blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for _ in range(frames):
        vw.write(blank)
    vw.release()

def _write_shots_json(path: Path, first: int, last: int):
    shots = {"shots": [{"shot_number": 1, "first_frame": first, "last_frame": last}]}
    path.write_text(json.dumps(shots))

def _latest_in(base: Path, pattern: str) -> Path:
    """Relative glob under base; return newest by mtime."""
    candidates = list(base.glob(pattern))
    assert candidates, f"No files matched {pattern!r} under {base}"
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _load_frames_v21(npz_path: Path):
    """
    Load frames from an observations NPZ.
    Primary (legacy/mainline): structured array under 'observations' with field 'f'.
    Fallback: flat NPZ with key 'frame'.
    """
    import numpy as np
    with np.load(npz_path) as data:
        if "observations" in data.files:
            arr = data["observations"]
            return (arr["f"].astype(int) if arr.size else np.array([], dtype=int))
        # flat fallback
        if "frame" in data.files:
            return data["frame"].astype(int)
        return np.array([], dtype=int)

def _columns_from_npz(npz_path: Path):
    """
    Normalize both legacy structured and flat layouts into a dict of columns:
      frame, shot_id, track_id, x, y, w, h, source
    Missing x/y/w/h in structured are derived from bbox_xyxy.
    """
    import numpy as np
    with np.load(npz_path) as data:
        # Legacy / mainline: single structured array under 'observations'
        if "observations" in data.files:
            obs = data["observations"]
            n = int(obs.shape[0]) if obs.size else 0
            if n == 0:
                return {
                    "frame": np.array([], dtype=int),
                    "shot_id": np.array([], dtype=int),
                    "track_id": np.array([], dtype=int),
                    "x": np.array([], dtype=np.float32),
                    "y": np.array([], dtype=np.float32),
                    "w": np.array([], dtype=np.float32),
                    "h": np.array([], dtype=np.float32),
                    "source": np.array([], dtype=int),
                }
            frame = obs["f"].astype(int, copy=False)
            shot_id = obs["shot"].astype(int, copy=False)
            track_id = obs["track_id"].astype(int, copy=False)
            source = obs["src"].astype(int, copy=False)
            bbox = obs["bbox_xyxy"].astype(np.float32, copy=False)
            x1 = bbox[:, 0]; y1 = bbox[:, 1]; x2 = bbox[:, 2]; y2 = bbox[:, 3]
            x = x1
            y = y1
            w = (x2 - x1)
            h = (y2 - y1)
            return {
                "frame": frame,
                "shot_id": shot_id,
                "track_id": track_id,
                "x": x.astype(np.float32, copy=False),
                "y": y.astype(np.float32, copy=False),
                "w": w.astype(np.float32, copy=False),
                "h": h.astype(np.float32, copy=False),
                "source": source,
            }
        # Flat fallback (if ever produced)
        cols = {}
        for k in ("frame","shot_id","track_id","x","y","w","h","source"):
            if k in data.files:
                cols[k] = data[k]
        # Ensure all required keys exist (even if empty arrays)
        def _ensure(name, dtype):
            if name not in cols:
                cols[name] = np.array([], dtype=dtype)
        _ensure("frame", int)
        _ensure("shot_id", int)
        _ensure("track_id", int)
        _ensure("x", np.float32)
        _ensure("y", np.float32)
        _ensure("w", np.float32)
        _ensure("h", np.float32)
        _ensure("source", int)
        return cols
    
# ----------------------------------- TEST -----------------------------------

@pytest.mark.integration
def test_cli_resume_exact_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    E2E through the CLI with real checkpointing and sidecars:

      Run #1:
        - detect_interval=1 (detect every frame)
        - simulated detector raises mid-run -> CLI exits non-zero
        - checkpoint dir populated (status.json / obs_ckpt.npz)

      Run #2:
        - resume via --resume-latest
        - must start at the recorded anchor (no rewind to frame 0)
        - pre-anchor observations are preserved exactly
        - global frame order is non-decreasing
    """
    # Import after collection for monkeypatching
    from facekit.cli import resolve_face_ids_v2_cli as cli_mod
    from facekit.pipeline import track_across_segments as track_mod

    # 1) Patch the EXACT callable that track_across_segments uses
    def _fake_align(frame, landmarks, frame_idx=None, **k):
        import numpy as np
        # Return a single aligned chip per detection call
        return np.zeros((112, 112, 3), dtype=np.uint8)
    
    monkeypatch.setattr(track_mod, "align_face_for_arcface", _fake_align, raising=True)

    # 2) Make the embedder actually emit vectors
    monkeypatch.setattr("facekit.embedding.embedder.FaceEmbedder", _DummyEmbedder, raising=True)

    total_frames = 60
    vid = tmp_path / "toy.mp4"
    _make_tiny_video(vid, frames=total_frames, size=(192, 108))
    shots_path = tmp_path / "shots.json"
    _write_shots_json(shots_path, 0, total_frames - 1)

    ckpt_parent = tmp_path / "ckpt"
    ckpt_parent.mkdir()
    obs_sidecar = tmp_path / "obs_sidecar.npz"
    out_json = tmp_path / "out_v2_1.json"

    # Force CPU + avoid real model loads
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr("facekit.detection.yolo5face_model.load_yolo5face_model",
                        lambda *a, **k: object())
     
    # Inject our detectors into track_across_segments
    replacement = {"det": None}
    _orig_track = track_mod.track_across_segments
    def _wrapped_track(*args, **kwargs):
        if replacement["det"] is not None:
            kwargs["detector"] = replacement["det"]
        return _orig_track(*args, **kwargs)
    monkeypatch.setattr(track_mod, "track_across_segments", _wrapped_track)

    # ------------------------------ RUN 1 (crash) ------------------------------
    replacement["det"] = _EmitOneThenCrash(crash_at=22)  # ensures obs_rows > 0 before the crash
    argv1 = [
        "prog",
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--output-global-json", str(out_json),
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--emb-store", "none",
        "--obs-sidecar-path", str(obs_sidecar),
        "--checkpoint-dir", str(ckpt_parent),
        "--log", "INFO",
    ]
    monkeypatch.setattr(sys, "argv", argv1)
    with pytest.raises(SystemExit):  # expected due to injected crash
        cli_mod.main()

    # --- Inspect checkpoint (robust: derive run root from obs_ckpt)
    obs_ckpt_path = _latest_in(ckpt_parent, "run-*/ckpt/obs_ckpt.npz")
    run_root = obs_ckpt_path.parent.parent  # .../run-.../ckpt -> .../run-...
    status_path = run_root / "status.json"
    if not status_path.exists():
        # fallback for any legacy layout
        maybe = run_root / "ckpt" / "status.json"
        assert maybe.exists(), f"status.json not found under {run_root}"
        status_path = maybe

    status1 = json.loads(status_path.read_text())
    obs1 = np.load(obs_ckpt_path)

    frames_pre = _load_frames_v21(obs_ckpt_path)

    anchor_in_status = status1.get("last_detection_frame")
    if anchor_in_status is not None:
        anchor_frame = int(anchor_in_status)
    elif frames_pre.size:
        anchor_frame = int(frames_pre.max())
    else:
        anchor_frame = 0

    assert 0 <= anchor_frame < total_frames, f"bad anchor_frame={anchor_frame} total={total_frames}"
    pre_anchor_rows = int((frames_pre < anchor_frame).sum()) if frames_pre.size else 0
   # ------------------------------ RUN 2 (resume) -----------------------------
    replacement["det"] = _EmitOneNoCrash()  # same emission pattern, no crash
    argv2 = [
        "prog",
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--output-global-json", str(out_json),
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--emb-store", "none",
        "--obs-sidecar-path", str(obs_sidecar),
        "--checkpoint-dir", str(ckpt_parent),
        "--resume-latest",
        "--log", "INFO",
    ]
    monkeypatch.setattr(sys, "argv", argv2)
    cli_mod.main()  # should complete cleanly

    # --- After RUN 2 completes, validate resume behavior strictly
    # Load resumed sidecar/manifest
    obs_ckpt_latest = _latest_in(ckpt_parent, "run-*/ckpt/obs_ckpt.npz")
    obs2 = np.load(obs_ckpt_latest)
    frames2 = _load_frames_v21(obs_ckpt_latest)
    
    assert frames2.size > 0, "no observations after resume"

    # Pre-anchor rows should be preserved exactly; no new rows < anchor_frame.
    assert (frames2 < anchor_frame).sum() == pre_anchor_rows, (
        f"unexpected change to pre-anchor rows: had {pre_anchor_rows}, "
        f"now {(frames2 < anchor_frame).sum()} (anchor={anchor_frame})"
    )
    # First newly appended row must be at/after the anchor.
    if frames2.size > pre_anchor_rows:
        assert frames2[pre_anchor_rows] >= anchor_frame, (
            f"first appended row is before anchor: frames2[{pre_anchor_rows}]={frames2[pre_anchor_rows]} < {anchor_frame}"
        )
    # Only enforce “pre-anchor preserved exactly” if we actually had any in Run 1
    if pre_anchor_rows > 0:
        pre2 = (frames2 < anchor_frame).sum()
        assert pre2 == pre_anchor_rows, f"pre-anchor rows changed ({pre2} != {pre_anchor_rows})"

    # Global frame order non-decreasing within each (shot,track) sequence
    # Normalize columns from structured/flat ckpt NPZ for per-(shot,track) monotonic checks
    cols_ckpt = _columns_from_npz(obs_ckpt_latest)
    shot = cols_ckpt["shot_id"].astype(int)
    track = cols_ckpt["track_id"].astype(int)
    for sid in np.unique(shot):
        for tid in np.unique(track[shot==sid]):
            seq = frames2[(shot==sid) & (track==tid)]
            assert np.all(seq[:-1] <= seq[1:]), f"non-monotonic order in shot={sid}, track={tid}"

    # 3) No duplicates: (shot_id, track_id, frame) tuples must be unique
    triples = np.stack([shot, track, frames2], axis=1)
    uniq = np.unique(triples, axis=0)
    assert uniq.shape[0] == triples.shape[0], "duplicate (shot,track,frame) records after resume"

    # ------------------------------ POST-ASSERTS -------------------------------
    final_cols = _columns_from_npz(obs_sidecar)
    final_frames = final_cols["frame"].astype(int)

    # (1) Pre-anchor observations preserved exactly — only if any existed pre-crash
    if pre_anchor_rows > 0:
        assert final_frames.size >= pre_anchor_rows, "lost rows after resume"
        assert int((final_frames < anchor_frame).sum()) == pre_anchor_rows, \
            "pre-anchor observations not preserved exactly"
    
    # (2) No rewind: resume begins at the anchor (or later if no frames at anchor)
    if final_frames.size:
        # index of first frame >= anchor
        idxs = np.where(final_frames >= anchor_frame)[0]
        if idxs.size:
            min_post = int(final_frames[idxs.min()])
            assert min_post == anchor_frame, f"resume started at {min_post}, expected anchor {anchor_frame}"

    # (3) Global non-decreasing order
    if final_frames.size > 1:
        diffs = np.diff(final_frames)
        assert (diffs >= 0).all(), f"frames out of order; first negative @ {np.where(diffs < 0)[0][:5]}"

    # ------------------------------ RUN 3 (golden, no crash) ------------------------------
    replacement["det"] = _EmitOneNoCrash()  # must match RUN 2 so sidecars are byte-for-byte comparable
    gold_ckpt = tmp_path / "gold_ckpt"
    gold_ckpt.mkdir()

    argv3 = [
        "prog",
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--output-global-json", str(tmp_path / "gold_v2_1.json"),
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--emb-store", "none",
        "--obs-sidecar-path", str(tmp_path / "gold_obs_sidecar.npz"),
        "--checkpoint-dir", str(gold_ckpt),
        "--new-run",                 # ensure clean run
        "--log", "INFO",
    ]
    monkeypatch.setattr(sys, "argv", argv3)
    cli_mod.main()

    # Compare sidecars (order + contents)
    obs_resume = np.load(_latest_in(ckpt_parent, "run-*/ckpt/obs_ckpt.npz"))
    obs_golden = np.load(_latest_in(gold_ckpt, "run-*/ckpt/obs_ckpt.npz"))

    def _rows_from_path(npz_path: Path):
        import numpy as np
        cols = _columns_from_npz(npz_path)
        stack = [cols[k].reshape(-1,1) for k in ("shot_id","track_id","frame","x","y","w","h","source")]
        return np.concatenate(stack, axis=1) if stack and stack[0].size else np.empty((0,0))

    R = _rows_from_path(_latest_in(ckpt_parent, "run-*/ckpt/obs_ckpt.npz"))
    G = _rows_from_path(_latest_in(gold_ckpt, "run-*/ckpt/obs_ckpt.npz"))

    # Same length + exact elementwise equality → identical retained data/order
    assert R.shape == G.shape, f"resume rows {R.shape} != golden rows {G.shape}"
    assert np.array_equal(R, G), "resume data differs from uninterrupted golden run"

    # Compare manifests (field-by-field, ignore non-deterministic/derived bits)
    gold_manifest = json.loads((tmp_path/"gold_v2_1.json").read_text())
    resume_manifest = json.loads(out_json.read_text())

    def _fields_signature(sidecar: dict):
        # Normalize the fields list into a simple (name, type) signature.
        # Some writers use 'type', older ones used 'typ' – handle both.
        fields = sidecar.get("fields", [])
        return [(f["name"], f.get("type", f.get("typ"))) for f in fields]

    # 1) Core invariants about the run itself
    assert gold_manifest.get("schema_version") == resume_manifest.get("schema_version")
    assert gold_manifest.get("face_metadata") == resume_manifest.get("face_metadata")

    # 2) Inputs (paths should match; if you prefer, wrap in os.path.abspath() first)
    g_in, r_in = gold_manifest.get("input", {}), resume_manifest.get("input", {})
    assert g_in.get("video") == r_in.get("video")
    assert g_in.get("shots") == r_in.get("shots")

    # 3) Observations sidecar invariants
    g_obs, r_obs = gold_manifest.get("observations_sidecar", {}), resume_manifest.get("observations_sidecar", {})

    # Stable numeric/enum properties
    for k in ("count", "dtype", "format"):
        assert g_obs.get(k) == r_obs.get(k), f"{k} differs: {g_obs.get(k)} vs {r_obs.get(k)}"

    # Schema/shape invariants (ignore ordering differences beyond (name,type) pairs if desired)
    assert _fields_signature(g_obs) == _fields_signature(r_obs), "sidecar fields signature differs"

    # Optionally ensure frame bounds or other derived invariants if present
    for k in ("min_frame", "max_frame", "tracks"):
        if k in g_obs or k in r_obs:
            assert g_obs.get(k) == r_obs.get(k), f"{k} differs: {g_obs.get(k)} vs {r_obs.get(k)}"

    # 4) Ignore volatile bits explicitly (timestamps, run ids, byte sizes, hashes, tool metadata, etc.)
    def _scrub(m: dict):
        m = dict(m)
        for k in ("created_at", "run_id", "version", "tool_meta"):
            m.pop(k, None)
        if "observations_sidecar" in m:
            for vk in ("path", "size_bytes", "sha256"):
                m["observations_sidecar"].pop(vk, None)
        return m