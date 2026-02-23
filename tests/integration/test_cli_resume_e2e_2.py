import json
import subprocess
import sys
from pathlib import Path
import re
import numpy as np
import pytest
import os

# ============================== Helpers (self-contained) ==============================

def _with_repo_env() -> dict:
    """
    Build an env that includes the project root on PYTHONPATH so the shim can import `facekit`.
    Assumes this test lives in <repo>/tests/integration/... -> repo = __file__.parents[2].
    If your layout differs, adjust the parents index or compute dynamically.
    """
    repo_root = Path(__file__).resolve().parents[2]
    # Prepend repo_root to existing PYTHONPATH; also fold in current sys.path entries defensively.
    existing = os.environ.get("PYTHONPATH", "")
    extras = [str(repo_root)] + [p for p in sys.path if p]  # directories only
    combined = os.pathsep.join([*extras, existing]) if existing else os.pathsep.join(extras)
    env = dict(os.environ)
    env["PYTHONPATH"] = combined
    return env

def _make_tiny_video(path: Path, frames=60, size=(192, 108)):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, size)
    blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for _ in range(frames):
        vw.write(blank)
    vw.release()

def _write_shots_json(path: Path, first: int, last: int):
    shots = {"shots": [{"shot_number": 1, "first_frame": int(first), "last_frame": int(last)}]}
    path.write_text(json.dumps(shots))

def _latest_in(base: Path, pattern: str) -> Path:
    candidates = list(base.glob(pattern))
    assert candidates, f"No files matched {pattern!r} under {base}"
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _load_frames_v21(npz_path: Path) -> np.ndarray:
    """Return int frames array from observations NPZ (v2.1 struct or flat fallback)."""
    with np.load(npz_path) as data:
        if "observations" in data.files:
            arr = data["observations"]
            return (arr["f"].astype(int) if arr.size else np.array([], dtype=int))
        if "frame" in data.files:
            return data["frame"].astype(int)
        return np.array([], dtype=int)

def _columns_from_npz(npz_path: Path):
    """
    Normalize both structured and flat layouts into a dict of columns:
      frame, shot_id, track_id, x, y, w, h, source
    """
    with np.load(npz_path) as data:
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
        # Flat fallback
        cols = {}
        for k in ("frame","shot_id","track_id","x","y","w","h","source"):
            if k in data.files:
                cols[k] = data[k]
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

def _run_python(shim: Path, *args, ok=(0,), env=None, cwd=None):
    cp = subprocess.run([sys.executable, str(shim), *args],
                        text=True, capture_output=True, env=env,
                        cwd=cwd or shim.parent)  # <-- important
    if cp.returncode not in ok:
        raise AssertionError(
            f"Return code {cp.returncode} not in {ok}\n"
            f"=== STDOUT ===\n{cp.stdout}\n"
            f"=== STDERR ===\n{cp.stderr}\n"
        )
    return cp

def _extract_run_root_from_logs(stdout: str, stderr: str) -> Path | None:
    txt = stdout + "\n" + stderr
    m = re.search(r"Checkpoint selection:\s*dir=([^\s|]+)", txt)
    if m: 
        return Path(m.group(1))
    # accept older/alternate formats
    m = re.search(r"ckpt\.root\s*=\s*([^\s]+)", txt)
    return Path(m.group(1)) if m else None

def _find_status_json_from_ckpt_root(root: Path) -> Path:
    # Try both …/status.json and …/ckpt/status.json
    candidates = Path(root,"status.json"), Path(root,"ckpt", "status.json")
    for p in candidates:
        if p.exists(): return p
    # recursive fallback
    hits = list(root.rglob("status.json"))
    assert hits, f"No status.json found under {root}"
    return max(hits, key=lambda p: p.stat().st_mtime)

def _list_tree_for_debug(root: Path, max_items: int = 200) -> str:
    rows = []
    try:
        for p in sorted(root.rglob("*")):
            rows.append(str(p.relative_to(root)))
            if len(rows) >= max_items:
                rows.append("... (truncated)")
                break
    except Exception as e:
        rows.append(f"<error listing tree: {e!r}>")
    return "\n".join(rows)

def _latest_run_dir(ckpt_parent: Path) -> Path:
    """
    Return the newest run-* directory under ckpt_parent. Works with both:
      ckpt_parent/run-*/...
      ckpt_parent/<video_hash>/run-*/...
    """
    candidates = []
    # direct runs
    candidates += [p for p in ckpt_parent.glob("run-*") if p.is_dir()]
    # hash-nested runs
    for h in ckpt_parent.iterdir():
        if h.is_dir():
            candidates += [p for p in h.glob("run-*") if p.is_dir()]
    assert candidates, (
        f"No run-* directories under {ckpt_parent}\n"
        f"--- tree ---\n{_list_tree_for_debug(ckpt_parent)}"
    )
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _read_status_anchor(run_root: Path) -> int:
    status_path = run_root / "status.json"
    assert status_path.exists(), (
        f"status.json not found in run root: {run_root}\n"
        f"--- run tree ---\n{_list_tree_for_debug(run_root)}"
    )
    import json
    status = json.loads(status_path.read_text() or "{}")
    assert "last_detection_frame" in status, (
        f"status.json missing last_detection_frame\n"
        f"status.json:\n{status_path.read_text()}"
    )
    return int(status.get("last_detection_frame") or 0)


def _find_status_json_under_run_root(run_root: Path) -> Path:
    for cand in (Path(run_root, "status.json"), Path(run_root, "ckpt","status.json")):
        if cand.exists():
            return cand
    hits = list(run_root.rglob("status.json"))
    assert hits, f"No status.json under {run_root}\n--- run tree ---\n{_list_tree_for_debug(run_root)}"
    return max(hits, key=lambda p: p.stat().st_mtime)

def _find_ckpt_files(ckpt_parent: Path) -> tuple[Path, Path | None]:
    run_dir = _latest_run_dir(ckpt_parent)
    ckpt_dir = run_dir / "ckpt"   # <— not "checkpoint"
    obs = max(ckpt_dir.glob("obs_ckpt*.npz"), default=None, key=lambda p: p.stat().st_mtime)
    emb = max(ckpt_dir.glob("emb_ckpt*.npz"), default=None, key=lambda p: p.stat().st_mtime)
    assert obs is not None, f"No obs_ckpt*.npz in {ckpt_dir}\n--- ckpt tree ---\n{_list_tree_for_debug(run_dir)}"
    return obs, emb

def _find_status_json(ckpt_parent: Path) -> Path:
    run_dir = _latest_run_dir(ckpt_parent)
    # prefer top-level status.json; fall back to ckpt/status.json
    for cand in (Path(run_dir, "status.json"), Path(run_dir, "ckpt", "status.json")):
        if cand.exists():
            return cand
    hits = list(run_dir.rglob("status.json"))
    assert hits, (
        f"No status.json under {run_dir}\n"
        f"--- run tree ---\n{_list_tree_for_debug(run_dir)}"
    )
    return max(hits, key=lambda p: p.stat().st_mtime)


# ============================== Subprocess shim (written to tmp) ==============================

SHIM_SOURCE = r"""
import os, sys, json, numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# 1) Patch modules *before* the CLI imports bind names via 'from ... import ...'
# -- dummy YOLO loader
def _dummy_loader(*a, **k): return object()
import facekit.detection.yolo5face_model as y5
y5.load_yolo5face_model = _dummy_loader

# -- dummy embedder class
class _DummyEmbedder:
    def __init__(self, *a, **k):
        pass
    def get_embedding_batch(self, chips, batch_size=None, **kwargs):
        vecs = []
        for chip in chips:
            h = int(np.uint64(chip.sum() + chip.shape[0]*1009 + chip.shape[1]*2741))
            rng = np.random.RandomState(h % (2**32))
            v = rng.rand(512).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)  # unit-norm to match prod expectations
            vecs.append(v)
        return np.stack(vecs, axis=0)
    # optional single-image API for future-proofing
    def get_embedding(self, chip, **kwargs):
        return self.get_embedding_batch([chip], **kwargs)[0]

import facekit.embedding.embedder as emb_mod
emb_mod.FaceEmbedder = _DummyEmbedder

# -- import the tracking module so we can patch aligner & detector injection
from facekit.pipeline import track_across_segments as track_mod

# 2) Now import the CLI (it will see the patched loader/embedder)
from facekit.cli import resolve_face_ids_v2_cli as cli_mod

# 3) Stubs used by the tracker
class _EmitOneThenCrash:
    def __init__(self, crash_at): self.crash_at, self.n = int(crash_at), 0
    def detect_faces_in_frame(self, frame, frame_index=None):
        self.n += 1
        if self.n >= self.crash_at:
            raise RuntimeError("boom (injected)")
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], np.float32)
        lms   = np.zeros((1, 5, 2), np.float32)
        conf  = np.array([0.99], np.float32)
        return boxes, lms, conf

class _EmitOneNoCrash(_EmitOneThenCrash):
    def __init__(self): super().__init__(10**9)

def _fake_align(frame, landmarks, frame_idx=None, **k):
    return np.zeros((112, 112, 3), np.uint8)

# 4) Patch aligner + wrap tracker to inject stub detector
track_mod.align_face_for_arcface = _fake_align
_orig_track = track_mod.track_across_segments

mode = None
argv = []
i = 1
while i < len(sys.argv):
    if sys.argv[i] == "--stub-mode" and i+1 < len(sys.argv):
        mode = sys.argv[i+1]; i += 2
    else:
        argv.append(sys.argv[i]); i += 1

def _wrapped_track(*a, **k):
    if mode:
        if mode.startswith("crash:"):
            n = int(mode.split(":",1)[1])
            k["detector"] = _EmitOneThenCrash(n)
        elif mode == "emit_one":
            k["detector"] = _EmitOneNoCrash()
    return _orig_track(*a, **k)

track_mod.track_across_segments = _wrapped_track

# 5) Run CLI
sys.argv = ["prog", *argv]
try:
    cli_mod.main()
except SystemExit as e:
    raise
"""

# ============================== The test ==============================

@pytest.mark.integration
def test_resume_exact_replay_subprocess(tmp_path: Path):
    # Write subprocess shim
    shim = tmp_path / "run_cli_with_stubs.py"
    shim.write_text(SHIM_SOURCE)

    # Inputs
    total_frames = 60
    vid = tmp_path / "toy.mp4"
    shots_path = tmp_path / "shots.json"
    _make_tiny_video(vid, frames=total_frames, size=(192, 108))
    _write_shots_json(shots_path, 0, total_frames - 1)

    # Shared checkpoint dir for runs 1+2
    ckpt_parent = tmp_path / "ckpt"
    ckpt_parent.mkdir()
    obs_sidecar = tmp_path / "obs_sidecar.npz"
    out_json = tmp_path / "out_v2_1.json"

    common = [
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--output-global-json", str(out_json),
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--embedding-queue-max-pending", "16",
        "--emb-store", "none",
        "--obs-sidecar-path", str(obs_sidecar),
        "--checkpoint-dir", str(ckpt_parent),
        "--log", "INFO",
    ]

    # ---------------- RUN 1 (crash) ----------------
    cp1 = _run_python(
        shim, 
        "--stub-mode", 
        "crash:22", 
        *common, 
        ok=(1, 2), 
        cwd=tmp_path,
        env=_with_repo_env(),)

    run_root1 = _extract_run_root_from_logs(cp1.stdout, cp1.stderr)
    if run_root1 is None:
        raise AssertionError(
            "Could not parse run root from logs.\n"
            f"STDOUT:\n{cp1.stdout}\n\nSTDERR:\n{cp1.stderr}\n\n"
            f"ckpt_parent tree:\n{_list_tree_for_debug(ckpt_parent)}"
        )
    status1 = json.loads(_find_status_json_under_run_root(run_root1).read_text())
    # Single source of truth for resume anchor
    anchor = status1.get("last_embedding_safe_frame")
    assert anchor is not None, (
        f"missing last_embedding_safe_frame in status.json (keys={sorted(status1.keys())})"
    )
    anchor = int(anchor)
    assert 0 <= anchor < total_frames, f"bad anchor_frame={anchor} total={total_frames}"
    # Pre-anchor baseline comes from RUN 1’s checkpoint, not the shared sidecar.
    obs_ckpt1 = (run_root1 / "ckpt" / "obs_ckpt.npz")
    assert obs_ckpt1.exists(), f"missing {obs_ckpt1}"
    frames_pre_ckpt = _load_frames_v21(obs_ckpt1)
    pre_rows = int((frames_pre_ckpt < anchor).sum()) if frames_pre_ckpt.size else 0

    # ---------------- RUN 2 (resume) ----------------
    cp2 = _run_python(
        shim, 
        "--stub-mode", 
        "emit_one", 
        *common, 
        "--resume-latest", 
        ok=(0,), 
        cwd=tmp_path,
        env=_with_repo_env(),)
    # After resume, validate against the same shared sidecar path
    run_root2 = _extract_run_root_from_logs(cp2.stdout, cp2.stderr) or run_root1
    final_cols = _columns_from_npz(obs_sidecar)
    frames2 = final_cols["frame"].astype(int)
    assert frames2.size > 0, "no observations after resume"

    # Pre-anchor preserved exactly
    assert int((frames2 < anchor).sum()) == pre_rows, \
        f"pre-anchor rows changed: had {pre_rows}, now {(frames2 < anchor).sum()} (anchor={anchor})"
    # No rewind: first appended row >= anchor
    if frames2.size > pre_rows:
        assert frames2[pre_rows] >= anchor, \
            f"first appended row is before anchor: frames2[{pre_rows}]={frames2[pre_rows]} < {anchor}"

    # Per-(shot,track) monotonic + no duplicates
    shot2 = final_cols["shot_id"].astype(int)
    track2 = final_cols["track_id"].astype(int)
    for sid in np.unique(shot2):
        mask_s = (shot2 == sid)
        tids = np.unique(track2[mask_s])
        for tid in tids:
            seq = frames2[mask_s & (track2 == tid)]
            assert np.all(seq[:-1] <= seq[1:]), f"non-monotonic order in shot={sid}, track={tid}"
    triples = np.stack([shot2, track2, frames2], axis=1)
    uniq = np.unique(triples, axis=0)
    assert uniq.shape[0] == triples.shape[0], "duplicate (shot,track,frame) records after resume"

    # Validate first resumed frame is exactly at anchor
    if frames2.size:
        idxs = np.where(frames2 >= anchor)[0]
        if idxs.size:
            assert int(frames2[idxs.min()]) == anchor, (
                f"resume started at {int(frames2[idxs.min()])}, expected {anchor}"
            )    
    if frames2.size > 1:
        diffs = np.diff(frames2)
        assert (diffs >= 0).all(), "frames out of order in final sidecar"

    # ---------------- RUN 3 (golden clean) ----------------
    gold_ckpt = tmp_path / "gold_ckpt"
    gold_ckpt.mkdir()
    gold_obs = tmp_path / "gold_obs_sidecar.npz"
    gold_json = tmp_path / "gold_v2_1.json"

    common_golden = [
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--output-global-json", str(gold_json),
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--emb-store", "none",
        "--obs-sidecar-path", str(gold_obs),
        "--checkpoint-dir", str(gold_ckpt),
        "--new-run",
        "--log", "INFO",
    ]
    _run_python(
        shim, 
        "--stub-mode", 
        "emit_one", 
        *common_golden, 
        ok=(0,),
        env=_with_repo_env(),)

    # Compare resume vs golden **sidecar** rows (order + elementwise equality)
    def _rows_from_path(npz_path: Path) -> np.ndarray:
        cols = _columns_from_npz(npz_path)
        stack = [cols[k].reshape(-1, 1) for k in ("shot_id","track_id","frame","x","y","w","h","source")]
        return np.concatenate(stack, axis=1) if stack and stack[0].size else np.empty((0, 0))

    R = _rows_from_path(obs_sidecar)
    G = _rows_from_path(gold_obs)
    assert R.shape == G.shape, f"resume rows {R.shape} != golden rows {G.shape}"
    assert np.array_equal(R, G), "resume data differs from uninterrupted golden run"

    # Compare manifests (ignore volatile fields)
    resume_manifest = json.loads(out_json.read_text())
    gold_manifest = json.loads(Path(gold_json).read_text())

    def _fields_signature(sidecar: dict):
        fields = sidecar.get("fields", [])
        return [(f["name"], f.get("type", f.get("typ"))) for f in fields]

    # core invariants
    assert resume_manifest.get("schema_version") == gold_manifest.get("schema_version")
    assert resume_manifest.get("face_metadata") == gold_manifest.get("face_metadata")

    r_in, g_in = resume_manifest.get("input", {}), gold_manifest.get("input", {})
    assert r_in.get("video") == g_in.get("video")
    assert r_in.get("shots") == g_in.get("shots")

    r_obs, g_obs = resume_manifest.get("observations_sidecar", {}), gold_manifest.get("observations_sidecar", {})
    for k in ("count", "dtype", "format"):
        assert r_obs.get(k) == g_obs.get(k), f"{k} differs: {r_obs.get(k)} vs {g_obs.get(k)}"
    assert _fields_signature(r_obs) == _fields_signature(g_obs), "sidecar fields signature differs"
    for k in ("min_frame", "max_frame", "tracks"):
        if (k in r_obs) or (k in g_obs):
            assert r_obs.get(k) == g_obs.get(k), f"{k} differs: {r_obs.get(k)} vs {g_obs.get(k)}"
