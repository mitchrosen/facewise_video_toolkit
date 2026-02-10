# tests/integration/test_resume_anchor_shot_continuity.py

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


# ---- helpers ---------------------------------------------------------------

_LOG_DIRS_TO_ZIP: list[Path] = []


def _repo_root() -> Path:
    # This file lives at tests/integration/test_resume_anchor_shot_continuity.py
    return Path(__file__).resolve().parents[2]


def _env_with_repo() -> dict[str, str]:
    env = os.environ.copy()
    repo = str(_repo_root())
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo if not old else f"{repo}{os.pathsep}{old}"
    return env


class RunResult:
    def __init__(self, proc: subprocess.CompletedProcess[str]):
        self.args = proc.args
        self.returncode = proc.returncode
        self.stdout = proc.stdout
        self.stderr = proc.stderr


def _run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    ok: tuple[int, ...] = (0,),
    log_prefix: str = "run",
) -> RunResult:
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
    )
    rr = RunResult(proc)

    log_dir = Path.cwd() / ".pytest_resume_logs" / log_prefix
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_DIRS_TO_ZIP.append(log_dir)

    (log_dir / "stdout.txt").write_text(rr.stdout or "")
    (log_dir / "stderr.txt").write_text(rr.stderr or "")
    (log_dir / "cmd.txt").write_text(" ".join(cmd))

    if rr.returncode not in ok:
        raise AssertionError(
            f"Command failed with rc={rr.returncode}\n"
            f"CMD: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{rr.stdout}\n\n"
            f"STDERR:\n{rr.stderr}"
        )

    return rr


def _zip_logs() -> None:
    if not _LOG_DIRS_TO_ZIP:
        return
    try:
        root = Path.cwd() / ".pytest_resume_logs"
        if root.exists():
            shutil.make_archive(str(root), "zip", root)
    except Exception:
        pass


atexit.register(_zip_logs)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def _ordered_tracks(js: dict) -> list[dict]:
    """
    Return a flattened, consistently-sorted list of track dicts.
    Matches the helper used by the original integration test.
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

# ---- test ------------------------------------------------------------------

@pytest.mark.integration
def test_resume_anchor_shot_continuity(tmp_path: Path):
    """
    Narrower/faster diagnostic than the full cold-vs-crash-vs-resume equivalence test.

    This test does only:
      1) crash a run at frame 183
      2) verify the persisted embedding-safe anchor is 152
      3) resume from that checkpoint
      4) inspect only shot-2 track continuity in the resumed output

    It is intentionally aimed at the known bad shape where live anchor-shot tracks
    get split into:
      - pre-anchor fragments ending near 151/152
      - fresh tracks beginning at 153
    """
    video = Path(_repo_root(), "tests", "assets", "videos", "OGsTest_10sec_snippet.mp4")
    assert video.exists(), "missing test video"

    ckpt_parent = tmp_path / "ckpt_parent"
    ckpt_parent.mkdir()

    out_crash = tmp_path / "tracks_crash.json"
    out_resume = tmp_path / "tracks_resume.json"
    out_crash.touch()
    out_resume.touch()

    obs_npz = tmp_path / "obs_sidecar.npz"
    emb_npz = tmp_path / "emb_sidecar.npz"

    env = _env_with_repo()

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
        "--new-run",
        "--log", "DEBUG",
    ]

    print("Running CRASH ----------------------")
    _ = _run(
        crash_cmd,
        env={**env, "CRASH_AT_FRAME": "183"},
        ok=(0, 1, 2),
        log_prefix="diag_crash",
    )
    print("-----------------------------------")

    def _latest_ckpt_dir(parent: Path) -> Path:
        runs = [d for d in parent.iterdir() if d.is_dir() and d.name.startswith("run-")]
        assert runs, f"No run-* directory under {parent}"

        def _score(p: Path):
            s = p / "status.json"
            return (s.exists(), s.stat().st_mtime if s.exists() else 0, p.stat().st_mtime)

        return max(runs, key=_score)

    latest_ckpt_dir = _latest_ckpt_dir(ckpt_parent)
    status_json = latest_ckpt_dir / "status.json"
    assert status_json.exists(), f"Missing {status_json}"

    status = _load_json(status_json)
    assert int(status["last_embedding_safe_frame"]) == 152, (
        f"status.json embedding-safe-frame drift: expected 152, "
        f"got {status.get('last_embedding_safe_frame')}"
    )
    assert status.get("open_tracks"), f"expected open_tracks at crash anchor, got {status.get('open_tracks')!r}"


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
    _ = _run(resume_cmd, env=env, log_prefix="diag_resume")
    print("-----------------------------------")

    resume_js = _load_json(out_resume)
    resume_tracks = _ordered_tracks(resume_js)

    shot2 = [
        (int(t["first_frame"]), int(t["last_frame"]))
        for t in resume_tracks
        if int(t["shot_id"]) == 2
    ]

    expected_shot2 = [
        (103, 111),
        (103, 223),
        (103, 299),
        (120, 191),
        (208, 215),
        (224, 299),
        (232, 299),
    ]

    fresh_153 = [r for r in shot2 if r[0] == 153]
    truncated_pre_anchor = [r for r in shot2 if r[1] in (151, 152)]

    assert shot2 == expected_shot2, (
        "Resumed shot-2 tracks do not match the expected continuity shape.\n"
        f"Expected shot-2 ranges: {expected_shot2}\n"
        f"Actual shot-2 ranges:   {shot2}\n"
        f"Fresh 153-start ranges: {fresh_153}\n"
        f"Pre-anchor truncations: {truncated_pre_anchor}\n"
        f"Open tracks at crash anchor: {status.get('open_tracks')}\n"
        f"Resume status anchor: {status.get('last_embedding_safe_frame')}\n"
    )