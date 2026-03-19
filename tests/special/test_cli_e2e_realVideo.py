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
import shutil

from facekit.common.obs_consts import SRC_TO_CODE, Source

# Usage:
#   E2E_LIVE=1 pytest -s tests/integration/test_cli_e2e_realVideo.py
#   KEEP_E2E_LOGS=1 pytest tests
KEEP_SUBPROCESS_LOGS = os.environ.get("KEEP_E2E_LOGS", "").strip() in {"1", "true", "True", "yes", "YES"}
LIVE_SUBPROCESS_OUTPUT = os.environ.get("E2E_LIVE", "").strip() in {"1", "true", "True", "yes", "YES"}

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

    should_persist = KEEP_SUBPROCESS_LOGS

    if LIVE_SUBPROCESS_OUTPUT:
        # Stream live to terminal (no capture). Great for debugging.
        cp = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        # When streaming, we don't have cp.stdout/cp.stderr strings.
        if cp.returncode not in ok:
            raise AssertionError(
                f"Subprocess error (prefix={log_prefix})\n"
                f"RC={cp.returncode}, expected={ok}\n"
                f"(E2E_LIVE=1 streams output; nothing captured.)\n"
            )
        return cp

    # Default: capture output (your existing behavior)
    cp = subprocess.run(cmd, text=True, capture_output=True, cwd=cwd, env=env)

    if should_persist:
        stdout_path = log_dir / f"{log_prefix}_stdout.txt"
        stderr_path = log_dir / f"{log_prefix}_stderr.txt"
        stdout_path.write_text(cp.stdout or "")
        stderr_path.write_text(cp.stderr or "")
        _register_log_file(stdout_path)
        _register_log_file(stderr_path)

    if cp.returncode not in ok:
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
        int(t["first_frame"])
    ))
    return tracks

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

def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def _avi_to_mov(avi: Path, mov: Path) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(avi), "-c", "copy", str(mov)],
            check=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(avi), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mov)],
            check=True,
        )


# ---- test ------------------------------------------------------------------

@pytest.mark.integration
def test_cli_e2e_realVideo(tmp_path: Path):

    tests_dir = _repo_root() / "tests"
    results_dir = tests_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # video_name = "53-man-roster-decisions"
    # video_name = "hot-to-go"
    video_name = "OGsTest_10sec_snippet"
    # video_name = "3fiUUxKMiBGn6kF4Puczvi"

    print(f"Video: {video_name}")

    input_video = Path(tests_dir, "assets", "videos", f"{video_name}.mp4")
    # shots = _repo_root() / "tests" / "assets" / "videos" / "OGsTest_10sec_snippet_shots.json"
    assert input_video.exists(), "missing test video"
    # assert shots.exists(), "missing shots json"

    ckpt_parent = tmp_path / "ckpt_parent"
    ckpt_parent.mkdir()

    out_cold   = results_dir / f"{video_name}_tracks.json"
    out_video = results_dir / f"{video_name}_tracks_global_faceIDs.avi"
    out_video_mov = results_dir / f"{video_name}_tracks_global_faceIDs.mov"

    #create empty files to prevent being treated as parent dirs downstream
    out_cold.touch()

    # shared sidecars (obs/emb npz written by prod checkpoint)
    obs_npz = tmp_path / "obs_sidecar.npz"
    emb_npz = tmp_path / "emb_sidecar.npz"

    env = _env_with_repo()

    # ---- 1) COLD RUN (real CLI) ----
    cold_cmd = [
        sys.executable, "-m", "facekit.cli.resolve_face_ids_v2_cli",
        "--input", str(input_video),
        "--checkpoint-dir", str(ckpt_parent),
        "--detect-interval", "10",
        "--track-sample-interval", "5",
        "--min-face", "10",
        "--post-min-track-len", "80",
        "--device", "cpu",
        "--schema-version", "2.1",
        "--emb-store", "sidecar",
        "--obs-sidecar-path", str(obs_npz),
        "--emb-sidecar-path", str(emb_npz),
        "--output-global-json", str(out_cold),
        "--output-video", str(out_video),
        "--no-resume",                    # ensure fresh run
        # "--new-run",  # incompatible with --no-checkpoint-write
        "--no-checkpoint-write", # incompatible with --new-run
        "--log", "INFO",
    ]
    print("Running COLD ----------------------")
    cold_log = _run(cold_cmd, env=env, log_prefix="cold")
    # print("COLD logs ----------------------")
    # print(cold_log.stdout)
    # print(cold_log.stderr)
    print("-------------------------------")
    
    cold_js   = _load_json(out_cold)

    _   = _ordered_tracks(cold_js)

    def _compact(tr):
        return [(t["shot_id"], t["first_frame"], t["last_frame"])
                for t in tr]
    
    # print("\n---- DEBUG TRACK SUMMARY ----")
    # print("COLD:", _compact(cold_tracks))
    # print("------------------------------\n")

    def _labels_used_by_tracks(js: dict) -> set[str]:
        labels = set()
        for shot in js.get("shots", []):
            for t in shot.get("face_tracks", []):
                labels.add(t.get("face_label"))
        return labels

    def _labels_in_metadata(js: dict) -> set[str]:
        return {m.get("face_label") for m in js.get("face_metadata", [])}

    cold_used = _labels_used_by_tracks(cold_js)

    assert None not in cold_used, "cold has tracks with face_label=None"

    cold_meta = _labels_in_metadata(cold_js)

    missing_cold = cold_used - cold_meta

    assert not missing_cold, f"cold: face_metadata missing labels: {sorted(missing_cold)}"

    if not _have_ffmpeg():
        pytest.skip("ffmpeg not installed")

    _avi_to_mov(out_video, out_video_mov)

    from facekit.output.audio_tools import restore_audio_from_source
    restore_audio_from_source(str(input_video), str(out_video_mov))
