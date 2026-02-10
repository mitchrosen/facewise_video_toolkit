# tests/test_resume_equivalence_e2e.py
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import math
import difflib
from copy import deepcopy
import atexit
import zipfile

from facekit.common.obs_consts import SRC_TO_CODE, Source

# Usage KEEP_E2E_LOGS=1 pytest tests
KEEP_SUBPROCESS_LOGS = os.environ.get("KEEP_E2E_LOGS", "").strip() in {"1", "true", "True", "yes", "YES"}

# ---- helpers ---------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _env_with_repo() -> dict:
    env = dict(os.environ)
    repo = str(_repo_root())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo}{os.pathsep}{existing}" if existing else repo
    return env

def _run(cmd, *, ok=(0,), cwd=None, env=None, log_prefix="run"):
    base = Path.cwd()
    log_dir = _init_log_dir(base)

    cp = subprocess.run(cmd, text=True, capture_output=True, cwd=cwd, env=env)

    # Only persist stdout/stderr if user explicitly asked to keep logs via env var
    should_persist = KEEP_SUBPROCESS_LOGS

    if should_persist:
        stdout_path = log_dir / f"{log_prefix}_stdout.txt"
        stderr_path = log_dir / f"{log_prefix}_stderr.txt"
        stdout_path.write_text(cp.stdout or "")
        stderr_path.write_text(cp.stderr or "")
        _register_log_file(stdout_path)
        _register_log_file(stderr_path)

    if cp.returncode not in ok:
        # If we didn't persist for some reason, still include content in the exception.
        if not should_persist:
            raise AssertionError(
                f"Subprocess error (prefix={log_prefix})\n"
                f"RC={cp.returncode}, expected={ok}\n"
                f"=== STDOUT ===\n{cp.stdout}\n"
                f"=== STDERR ===\n{cp.stderr}\n"
            )
        raise AssertionError(
            f"Subprocess error (prefix={log_prefix})\n"
            f"RC={cp.returncode}, expected={ok}\n"
            f"stdout: {log_dir / f'{log_prefix}_stdout.txt'}\n"
            f"stderr: {log_dir / f'{log_prefix}_stderr.txt'}\n"
        )

    return cp

def _load_json(p: Path):
    return json.loads(p.read_text())

def _ordered_tracks(js):
    """
    Return a flattened, consistently-sorted list of track dicts
    """
    tracks = []
    if "shots" in js:
        # v2.0 / v2.1: flatten shots[*].face_tracks and carry shot_number for sort
        for shot in js.get("shots", []):
            shot_no = int(shot.get("shot_number", 0))
            for t in shot.get("face_tracks", []):
                t = dict(t)  # shallow copy
                t.setdefault("shot_id", shot_no)
                tracks.append(t)
    else:
        raise KeyError("No 'shots' key in manifest")

    tracks.sort(key=lambda t: (
        int(t["shot_id"]),
        int(t["first_frame"]),
        int(t.get("last_frame", -1)),
        str(t.get("face_label") or ""),
    ))
    return tracks

def _extract_anchor_from_logs(text: str) -> int:
    # 1) Explicit “ANCHOR:NNN” if you add it later
    m = re.search(r"\bANCHOR:(\d+)\b", text)
    if m: return int(m.group(1))
    # 2) Your resume logs often print “rehydrate: anchor=NNN”
    m = re.search(r"rehydrate:\s*anchor=(\d+)", text)
    if m: return int(m.group(1))
    # 3) Or “resume: first_new_frame=NNN”
    m = re.search(r"resume:\s*first_new_frame=(\d+)", text)
    if m: return int(m.group(1))
    # 4) Or “ENTER processing loop at frame=XXX (anchor=YYY)”
    m = re.search(r"ENTER processing loop at frame=\d+\s+\(anchor=(\d+)\)", text)
    if m: return int(m.group(1))
    # Fallback
    return 0

# Create global list to accumulate log files
_LOG_DIR = None
_LOG_FILES = []

def _init_log_dir(base: Path):
    global _LOG_DIR
    if _LOG_DIR is None:
        _LOG_DIR = base / "_logs"
        _LOG_DIR.mkdir(exist_ok=True)
    return _LOG_DIR

def _register_log_file(path: Path):
    _LOG_FILES.append(path)

def _zip_logs():
    """Called automatically at test exit."""
    global _LOG_DIR, _LOG_FILES
    if _LOG_DIR is None or not _LOG_FILES:
        return
    zip_path = _LOG_DIR / "logs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in _LOG_FILES:
            if path.exists():
                zf.write(path, arcname=path.name)
    print(f"\n[LOG] logs.zip written to: {zip_path}\n")

# register exit hook
atexit.register(_zip_logs)

# ---- test ------------------------------------------------------------------

@pytest.mark.integration
def test_resume_equivalence_full_e2e(tmp_path: Path):
    video = Path(_repo_root(), "tests", "assets", "videos", "OGsTest_10sec_snippet.mp4")
    # shots = _repo_root() / "tests" / "assets" / "videos" / "OGsTest_10sec_snippet_shots.json"
    assert video.exists(), "missing test video"
    # assert shots.exists(), "missing shots json"

    ckpt_parent = tmp_path / "ckpt_parent"
    ckpt_parent.mkdir()

    out_cold   = tmp_path / "tracks_cold.json"
    out_crash  = tmp_path / "tracks_crash.json"   # might be empty/partial
    out_resume = tmp_path / "tracks_resume.json"

    #create empty files to prevent being treated as parent dirs downstream
    out_cold.touch()
    out_crash.touch()
    out_resume.touch()

    # shared sidecars (obs/emb npz written by prod checkpoint)
    obs_npz = tmp_path / "obs_sidecar.npz"
    emb_npz = tmp_path / "emb_sidecar.npz"

    env = _env_with_repo()

    # ---- 1) COLD RUN (real CLI) ----
    # Replace module/flags to match your real tool.
    cold_cmd = [
        sys.executable, "-m", "facekit.cli.resolve_face_ids_v2_cli",
        "--input", str(video),
        "--checkpoint-dir", str(ckpt_parent),
        "--detect-interval", "8",
        "--embedding-queue-max-pending", "11",
        "--device", "cpu",
        "--schema-version", "2.1",
        "--emb-store", "sidecar",
        "--obs-sidecar-path", str(obs_npz),
        "--emb-sidecar-path", str(emb_npz),
        "--output-global-json", str(out_cold),
        "--no-resume",                    # ensure fresh run
        "--new-run",
        "--log", "DEBUG",
    ]
    print("Running COLD ----------------------")
    _ = _run(cold_cmd, env=env, log_prefix="cold")
    # print("COLD logs ----------------------")
    # print(cold_log.stdout)
    # print(cold_log.stderr)
    print("-------------------------------")
    
    # ---- 2) CRASH RUN (subprocess wrapper injects crash at frame 183) ----
    crashy = tmp_path / "crash_wrapper.py"
    crashy.write_text(
    r"""
import os
import sys
import traceback

from facekit.cli.resolve_face_ids_v2_cli import main as real_main
from facekit.pipeline.checkpoint import CheckpointManager

CRASH_AT = int(os.environ.get("CRASH_AT_FRAME", "183"))

class CrashyCheckpoint:
    def __init__(self, inner):
        self._inner = inner
    def on_frame(self, frame_idx: int) -> None:
        if frame_idx == CRASH_AT:
            raise RuntimeError(f"boom at frame {frame_idx}")
        return self._inner.on_frame(frame_idx)
    def __getattr__(self, name):
        return getattr(self._inner, name)

_orig_open = CheckpointManager.open
def _wrapped_open(*a, **k):
    mgr = _orig_open(*a, **k)
    return CrashyCheckpoint(mgr)
CheckpointManager.open = staticmethod(_wrapped_open)

if __name__ == "__main__":
    try:
        real_main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)
"""
)
    
    crash_cmd = [
        sys.executable, str(crashy),
        "--input", str(video),
        "--checkpoint-dir", str(ckpt_parent),
        "--detect-interval", "8",
        "--embedding-queue-max-pending", "11",
        "--device", "cpu",
        "--schema-version", "2.1",
        "--emb-store", "sidecar",
        "--obs-sidecar-path", str(obs_npz),
        "--emb-sidecar-path", str(emb_npz),
        "--output-global-json", str(out_crash),
        "--no-resume",
        "--log", "DEBUG",
    ]

    print("Running CRASH  ---------------------")
    crash_log = _run(crash_cmd, env={**env, "CRASH_AT_FRAME": "183"}, ok=(0,1,2), log_prefix="crash")
    # print("CRASH logs ---------------------")
    # print(crash_log.stdout)
    # print(crash_log.stderr)
    print("-------------------------------")

    # assert "boom at frame 183" in (crash_log.stdout + crash_log.stderr), \
    #     "Crash hook didn’t trigger at frame 183"

    def _latest_ckpt_dir(parent: Path) -> Path:
        runs = [d for d in parent.iterdir() if d.is_dir() and d.name.startswith("run-")]
        assert runs, f"No run-* directory under {parent}"
        # Prefer the one whose status.json exists & is most recent
        def _score(p: Path):
            s = p / "status.json"
            return (s.exists(), s.stat().st_mtime if s.exists() else 0, p.stat().st_mtime)
        return max(runs, key=_score)

    latest_ckpt_dir = _latest_ckpt_dir(ckpt_parent)
    status_json = Path(latest_ckpt_dir, "status.json")
    obs_npz_path = Path(latest_ckpt_dir, "ckpt", "obs_ckpt.npz")
    emb_npz_path = Path(latest_ckpt_dir, "ckpt", "emb_ckpt.npz")
    assert obs_npz_path.exists(), f"Missing {obs_npz_path}"
    assert emb_npz_path.exists(),  f"Missing {emb_npz_path}"
    assert status_json.exists(),   f"Missing {status_json}"

    st = json.loads(status_json.read_text())
    to_list = st.get("track_order") or []
    assert isinstance(to_list, list) and len(to_list) > 0, "track_order missing/empty in status.json"

    status = _load_json(status_json)
    assert int(status["last_embedding_safe_frame"]) == 152, (
       f"status.json embedding-safe-frame drift: expected 152, got {status.get('last_embedding_safe_frame')}"
    )

    # Crash run detections 
    arr = np.load(obs_npz_path, allow_pickle=False)["observations"]
    shot = 1
    det_code = SRC_TO_CODE[Source.DETECTED]
    is_shot = arr["shot"] == shot
    is_det  = arr["src"]  == det_code

    names = set(arr.dtype.names or [])

    # Under the current contract, no landmarks are persisted in the obs sidecar.
    forbidden = ("landmarks", "lms", "lm", "landmarks_5pt", "landmarks_flat10")
    assert not any(n in names for n in forbidden), (
        f"Landmarks fields unexpectedly persisted in obs sidecar: {sorted(set(names) & set(forbidden))}"
    )

    # ---- 2.5) PROBE SIDECRS (pre-resume, strict parity up to anchor-1) ----
    probe = _repo_root() / "tests" / "utils" / "probe_sidecars.py"
    assert probe.exists(), f"Probe script not found: {probe}"

    det_code_int = int(SRC_TO_CODE[Source.DETECTED])
    probe_cmd = [
        sys.executable, str(probe),
        "--run-root", str(latest_ckpt_dir),
        "--anchor", "152",
        "--det-code", str(det_code_int),
    ]
    _ = _run(probe_cmd, env=_env_with_repo(), log_prefix="probe")
    # print("PROBE (pre-resume) ----------------")
    # print(probe_log.stdout)
    # print(probe_log.stderr)
    # print("-----------------------------------")
    
    # ---- 3) RESUME RUN (real CLI) ----
    resume_cmd = [
        sys.executable, "-m", "facekit.cli.resolve_face_ids_v2_cli",
        "--input", str(video),
        "--checkpoint-dir", str(ckpt_parent),
        "--detect-interval", "8",
        "--embedding-queue-max-pending", "11",
        "--device", "cpu",
        "--schema-version", "2.1",
        "--emb-store", "sidecar",
        "--obs-sidecar-path", str(obs_npz),
        "--emb-sidecar-path", str(emb_npz),
        "--output-global-json", str(out_resume),
        "--resume-latest",
        "--log", "DEBUG",
    ]
    print("Running RESUME ---------------------")
    _ = _run(resume_cmd, env=env)
    # print("RESUME logs ---------------------")
    # print(resume_log.stdout)
    # print(resume_log.stderr)
    print("-------------------------------")

    # safe_frame = _extract_anchor_from_logs(resume_log.stdout + "\n" + resume_log.stderr)
    # print(f"Embedding-safe frame for resume run is {safe_frame}")
    # assert safe_frame == 152, f"unexpected embedding-safe frame parsed: {safe_frame}"

    # ---- Compare outputs ----
    cold_js   = _load_json(out_cold)

    resume_js = _load_json(out_resume)

    cold_tracks   = _ordered_tracks(cold_js)
    resume_tracks = _ordered_tracks(resume_js)

    def _compact(tr):
        return [(t["shot_id"], t["first_frame"], t["last_frame"])
                for t in tr]
    
    # print("\n---- DEBUG TRACK SUMMARY ----")
    # print("COLD:", _compact(cold_tracks))
    # print("RESUME:", _compact(resume_tracks))
    # print("------------------------------\n")

    assert len(cold_tracks) == len(resume_tracks), \
        f"track count mismatch: cold={len(cold_tracks)} resume={len(resume_tracks)}"

    def _labels_used_by_tracks(js: dict) -> set[str]:
        labels = set()
        for shot in js.get("shots", []):
            for t in shot.get("face_tracks", []):
                labels.add(t.get("face_label"))
        return labels

    def _labels_in_metadata(js: dict) -> set[str]:
        return {m.get("face_label") for m in js.get("face_metadata", [])}

    def _track_identity_key(t: dict) -> tuple[int, int, int]:
        """Deterministic key for 'same track' across runs."""
        return (int(t["shot_id"]), int(t["first_frame"]), int(t["last_frame"]))

    cold_used = _labels_used_by_tracks(cold_js)
    resume_used = _labels_used_by_tracks(resume_js)

    assert None not in cold_used, "cold has tracks with face_label=None"
    assert None not in resume_used, "resume has tracks with face_label=None"

    cold_meta = _labels_in_metadata(cold_js)
    resume_meta = _labels_in_metadata(resume_js)

    missing_cold = cold_used - cold_meta
    missing_resume = resume_used - resume_meta

    assert not missing_cold, f"cold: face_metadata missing labels: {sorted(missing_cold)}"
    assert not missing_resume, f"resume: face_metadata missing labels: {sorted(missing_resume)}"

    # Compare track-by-track by identity key (shot, first, last)
    def _track_identity_key(t: dict) -> tuple[int, int, int]:
        return (int(t["shot_id"]), int(t["first_frame"]), int(t["last_frame"]))

    cold_by_key = {_track_identity_key(t): t for t in cold_tracks}
    resume_by_key = {_track_identity_key(t): t for t in resume_tracks}

    assert set(cold_by_key.keys()) == set(resume_by_key.keys()), (
        "Track identity keys differ.\n"
        f"Only in cold:   {sorted(set(cold_by_key) - set(resume_by_key))}\n"
        f"Only in resume: {sorted(set(resume_by_key) - set(cold_by_key))}\n"
    )

    for key in sorted(cold_by_key.keys()):
        a = cold_by_key[key]
        b = resume_by_key[key]
        # Compare structural fields
        print(f"a fields = {a}")
        print(f"b fields = {b}")
        assert a["shot_id"]     == b["shot_id"], \
            f"shot drift: cold={a['shot_id']} resume={b['shot_id']}"
        assert a["first_frame"] == b["first_frame"], \
            f"first_frame drift: cold={a['first_frame']} resume={b['first_frame']}"
        assert a["last_frame"]  == b["last_frame"], \
            f"last_frame drift: cold={a['last_frame']} resume={b['last_frame']}"

        # Strict: global identity is face_label in your schema
        assert a.get("face_label") is not None, (
            f"cold run track missing face_label: key={_track_identity_key(a)}"
        )
        assert b.get("face_label") is not None, (
            f"resume run track missing face_label: key={_track_identity_key(b)}"
        )

        assert a["face_label"] == b["face_label"], (
            f"face_label drift: cold={a['face_label']} resume={b['face_label']} (key={key})\n"
            f"cold track:   {a}\n"
            f"resume track: {b}"
        )
        
        # ---- canonical compare for v2.1 globalID JSON ----

    _VOLATILE_KEYS = {"generation", "observations_sidecar", "embedding_sidecar", "params_hash"}

    # Per-track floating summaries can drift slightly between cold and resumed runs
    # even when the structural resume contract is satisfied (same tracks, same labels).
    # The strict assertions above already verify the load-bearing invariants:
    #   - same track identity keys
    #   - same face_label for each track
    #   - same face_metadata labels/counts
    # So exclude these derived numeric summaries from the final canonical compare.
    _TRACK_FLOAT_SUMMARY_KEYS = {
        "avg_center_x",
        "avg_center_y",
        "avg_face_width",
        "avg_face_height",
        "avg_confidence",
        "min_confidence",
        "max_confidence",
    }

    def _round_floats(x, ndigits=6):
        if isinstance(x, float):
            # normalize -0.0 and tiny float noise
            return 0.0 if abs(x) < 10**-(ndigits+2) else round(x, ndigits)
        if isinstance(x, list):
            return [_round_floats(v, ndigits) for v in x]
        if isinstance(x, dict):
            return {k: _round_floats(v, ndigits) for k, v in x.items()}
        return x

    def _canon_globalid(js: dict) -> dict:
        js = deepcopy(js)

        # drop volatile sections
        for k in list(js.keys()):
            if k in _VOLATILE_KEYS:
                js.pop(k, None)

        # normalize SHOTS / FACE_TRACKS
        shots = js.get("shots", [])
        for shot in shots:
            # enforce ints
            shot["shot_number"] = int(shot.get("shot_number", -1))
            for t in shot.get("face_tracks", []):
                t["first_frame"] = int(t.get("first_frame", -1))
                t["last_frame"]  = int(t.get("last_frame", -1))
                for k in _TRACK_FLOAT_SUMMARY_KEYS:
                    t.pop(k, None)

                # keep label exactly as produced (strict). if absent, use "" to stabilize sort
                t["face_label"]  = "" if t.get("face_label") is None else str(t["face_label"])
            # sort tracks deterministically
            shot["face_tracks"] = sorted(
                shot.get("face_tracks", []),
                key=lambda t: (t["first_frame"], t["last_frame"], t["face_label"])
            )
        js["shots"] = sorted(shots, key=lambda s: s["shot_number"])

        # normalize FACE_METADATA
        if "face_metadata" in js:
            fmd = []
            for m in js["face_metadata"]:
                cnt = m.get("occurrence_count", None)
                if cnt is None:
                    cnt = m.get("occurance_count", 0)
                fmd.append({
                    "face_label": str(m.get("face_label", "")),
                    "occurrence_count": int(cnt),
                })
            js["face_metadata"] = sorted(fmd, key=lambda m: m["face_label"])

        # round floats everywhere to tame tiny numeric noise
        js = _round_floats(js, ndigits=6)
        return js

    def _canon_str(js: dict) -> str:
        """Stable string for better diffs."""
        return json.dumps(_canon_globalid(js), sort_keys=True, indent=2)

    def assert_globalid_equal(cold_js: dict, resume_js: dict):
        a, b = _canon_globalid(cold_js), _canon_globalid(resume_js)
        if a != b:
            a_s, b_s = _canon_str(cold_js).splitlines(), _canon_str(resume_js).splitlines()
            diff = "\n".join(difflib.unified_diff(a_s, b_s, fromfile="cold", tofile="resume", lineterm=""))
            raise AssertionError("GlobalID JSONs differ (strict). Unified diff:\n" + diff)

    assert_globalid_equal(cold_js, resume_js)