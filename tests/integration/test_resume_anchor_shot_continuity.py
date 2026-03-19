from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from facekit.common.obs_consts import Source, src_to_code


_LOG_DIRS_TO_ZIP: list[Path] = []


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_with_repo() -> dict[str, str]:
    env = os.environ.copy()
    repo = str(_repo_root())
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo if not old else f"{repo}{os.pathsep}{old}"
    return env


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_single_array_from_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if len(data.files) != 1:
            raise AssertionError(f"expected exactly one array in {path}, found {data.files}")
        return data[data.files[0]]


def _ordered_tracks(js: dict) -> list[dict]:
    tracks: list[dict] = []
    shots = js.get("shots")
    if shots is None:
        raise KeyError("No 'shots' key in manifest")

    for shot in shots:
        shot_no = int(shot.get("shot_number", 0))
        for track in shot.get("face_tracks", []):
            row = dict(track)
            row.setdefault("shot_id", shot_no)
            tracks.append(row)

    tracks.sort(
        key=lambda t: (
            int(t["shot_id"]),
            int(t["first_frame"]),
            int(t.get("last_frame", -1)),
            str(t.get("face_label") or ""),
        )
    )
    return tracks


def _latest_ckpt_dir(parent: Path) -> Path:
    runs = [d for d in parent.iterdir() if d.is_dir() and d.name.startswith("run-")]
    assert runs, f"No run-* directory under {parent}"

    def _score(path: Path) -> tuple[bool, float, float]:
        status = path / "status.json"
        return (
            status.exists(),
            status.stat().st_mtime if status.exists() else 0.0,
            path.stat().st_mtime,
        )

    return max(runs, key=_score)


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


def _resolve_obs_sidecar_path(latest_ckpt_dir: Path, requested_path: Path) -> Path:
    if requested_path.exists():
        return requested_path

    status_json = latest_ckpt_dir / "status.json"
    if status_json.exists():
        status = _load_json(status_json)
        for key in (
            "obs_sidecar_path",
            "observations_sidecar_path",
            "obs_npz_path",
        ):
            value = status.get(key)
            if value:
                candidate = Path(value)
                if candidate.exists():
                    return candidate

    candidates = sorted(latest_ckpt_dir.rglob("*.npz"))
    obs_like = [p for p in candidates if "obs" in p.name.lower()]
    if len(obs_like) == 1:
        return obs_like[0]

    raise AssertionError(
        "Could not locate observation sidecar.\n"
        f"requested_path={requested_path}\n"
        f"latest_ckpt_dir={latest_ckpt_dir}\n"
        f"npz_candidates={[str(p) for p in candidates]}"
    )

atexit.register(_zip_logs)


@pytest.mark.integration
def test_resume_anchor_shot_continuity(tmp_path: Path):
    """
    Narrower/faster diagnostic than the full cold-vs-crash-vs-resume equivalence test.

    This test:
      1) crashes a run at frame 183
      2) verifies the persisted embedding-safe anchor under the current
         sampled-embedding pipeline configuration
      3) verifies that TRACKED observations with embeddings were persisted
      4) resumes from that checkpoint
      5) inspects only shot-2 continuity in the resumed output

    With tracked-frame sampling enabled, later sampled TRACKED observations can
    be embedded and persisted before the crash, advancing the durable boundary
    deeper into shot 2.
    """
    video = _repo_root() / "tests" / "assets" / "videos" / "OGsTest_10sec_snippet.mp4"
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
    _run(
        crash_cmd,
        env={**env, "CRASH_AT_FRAME": "183"},
        ok=(0, 1, 2),
        log_prefix="diag_crash",
    )
    print("-----------------------------------")

    latest_ckpt_dir = _latest_ckpt_dir(ckpt_parent)
    status_json = latest_ckpt_dir / "status.json"
    assert status_json.exists(), f"Missing {status_json}"

    status = _load_json(status_json)
    anchor_frame = int(status["last_embedding_safe_frame"])

    assert anchor_frame == 176, (
        f"status.json embedding-safe-frame drift: expected 176, got {status.get('last_embedding_safe_frame')}"
    )
    assert status.get("open_tracks"), (
        f"expected open_tracks at crash anchor, got {status.get('open_tracks')!r}"
    )

    obs_sidecar_path = _resolve_obs_sidecar_path(latest_ckpt_dir, obs_npz)
    obs_arr = _load_single_array_from_npz(obs_sidecar_path)
    tracked_code = int(src_to_code(Source.TRACKED.value))

    tracked_embedded_rows = obs_arr[
        (obs_arr["shot"] == 2)
        & (obs_arr["src"] == tracked_code)
        & (obs_arr["emb_idx"].astype(int) >= 0)
    ]
    tracked_embedded_frames = sorted(
        set(int(f) for f in tracked_embedded_rows["f"].astype(int).tolist())
    )

    assert tracked_embedded_frames, (
        "Expected at least one persisted TRACKED observation with an embedding in shot 2.\n"
        f"obs_sidecar_path={obs_sidecar_path}\n"
        f"tracked_embedded_frames={tracked_embedded_frames}\n"
        f"anchor_frame={anchor_frame}\n"
    )

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
    _run(resume_cmd, env=env, log_prefix="diag_resume")
    print("-----------------------------------")

    resume_js = _load_json(out_resume)
    resume_tracks = _ordered_tracks(resume_js)

    shot2 = [
        (int(t["first_frame"]), int(t["last_frame"]))
        for t in resume_tracks
        if int(t["shot_id"]) == 2
    ]

    expected_shot2 = [
        (103, 223),
        (103, 299),
        (120, 191),
        (224, 299),
    ]

    first_resumed_frame = anchor_frame + 1
    fresh_at_resume_boundary = [r for r in shot2 if r[0] == first_resumed_frame]
    truncated_at_anchor = [r for r in shot2 if r[1] in (anchor_frame - 1, anchor_frame)]

    assert fresh_at_resume_boundary == [], (
        "Unexpected fresh shot-2 tracks starting exactly at the first resumed frame.\n"
        f"anchor_frame={anchor_frame}\n"
        f"first_resumed_frame={first_resumed_frame}\n"
        f"fresh_at_resume_boundary={fresh_at_resume_boundary}\n"
        f"shot2={shot2}\n"
    )

    assert truncated_at_anchor == [], (
        "Unexpected shot-2 tracks truncated at the embedding-safe anchor boundary.\n"
        f"anchor_frame={anchor_frame}\n"
        f"truncated_at_anchor={truncated_at_anchor}\n"
        f"shot2={shot2}\n"
    )

    assert shot2 == expected_shot2, (
        "Resumed shot-2 tracks do not match the expected continuity shape.\n"
        f"Expected shot-2 ranges: {expected_shot2}\n"
        f"Actual shot-2 ranges:   {shot2}\n"
        f"Fresh at resume-boundary: {fresh_at_resume_boundary}\n"
        f"Truncated at anchor: {truncated_at_anchor}\n"
        f"Open tracks at crash anchor: {status.get('open_tracks')}\n"
        f"Resume status anchor: {anchor_frame}\n"
    )