from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.integration.test_resume_completed_run_frame_ranges import (
    SHIM_SOURCE,
    _load_rows,
    _make_tiny_video,
    _run_python,
    _with_repo_env,
    _write_shots_json,
)


@pytest.mark.integration
@pytest.mark.parametrize(
    "start_frame,end_frame",
    [
        (10, 20),      # both before crash
        (40, 50),      # both after crash
        (20, 40),      # straddles crash
        (30, 40),      # start == crash frame
        (20, 30),      # end == crash frame
        (20, None),    # start before crash, no end
        (40, None),    # start after crash, no end
        (None, 20),    # no start, end before crash
        (None, 40),    # no start, end after crash
    ],
)
def test_resume_after_crash_matches_full_run_subset(
    tmp_path: Path,
    start_frame: int | None,
    end_frame: int | None,
):
    """
    History-preserving crash+resume behavior:

    1. Produce a gold full run.
    2. Produce a crashed run.
    3. Resume with explicit frame-range arguments.
    4. Verify exported observations exactly match the equivalent subset
       from the gold full run.
    """

    shim = tmp_path / "run_cli_with_stubs.py"
    shim.write_text(SHIM_SOURCE)

    total_frames = 60

    vid = tmp_path / "toy.mp4"
    shots_path = tmp_path / "shots.json"

    _make_tiny_video(
        vid,
        frames=total_frames,
        size=(192, 108),
    )

    _write_shots_json(
        shots_path,
        0,
        total_frames - 1,
    )

    # ------------------------------------------------------------------
    # GOLD FULL RUN
    # ------------------------------------------------------------------

    gold_ckpt = tmp_path / "gold_ckpt"
    gold_ckpt.mkdir()

    gold_obs = tmp_path / "gold_obs.npz"

    common_gold = [
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--embedding-queue-max-pending", "16",
        "--emb-store", "none",
        "--checkpoint-dir", str(gold_ckpt),
        "--obs-sidecar-path", str(gold_obs),
        "--output-global-json", str(tmp_path / "gold.json"),
        "--log", "INFO",
    ]

    _run_python(
        shim,
        *common_gold,
        ok=(0,),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    gold_rows = _load_rows(gold_obs)

    # ------------------------------------------------------------------
    # CRASH RUN
    # ------------------------------------------------------------------

    crash_ckpt = tmp_path / "crash_ckpt"
    crash_ckpt.mkdir()

    crash_obs = tmp_path / "crash_obs.npz"

    common_crash = [
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--embedding-queue-max-pending", "16",
        "--emb-store", "none",
        "--checkpoint-dir", str(crash_ckpt),
        "--obs-sidecar-path", str(crash_obs),
        "--output-global-json", str(tmp_path / "crash.json"),
        "--log", "INFO",
    ]

    # crash around frame 30
    _run_python(
        shim,
        "--stub-mode",
        "crash:30",
        *common_crash,
        ok=(1, 2),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    # ------------------------------------------------------------------
    # RESUME RUN
    # ------------------------------------------------------------------

    resumed_obs = tmp_path / "resumed_obs.npz"

    resume_args = [
        "--resume-latest",
    ]

    if start_frame is not None:
        resume_args.extend(["--start-frame", str(start_frame)])

    if end_frame is not None:
        resume_args.extend(["--end-frame", str(end_frame)])

    _run_python(
        shim,
        *resume_args,
        *common_crash,
        "--obs-sidecar-path", str(resumed_obs),
        "--output-global-json", str(tmp_path / "resumed.json"),
        ok=(0,),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    resumed_rows = _load_rows(resumed_obs)

    # ------------------------------------------------------------------
    # EXPECTED SUBSET
    # ------------------------------------------------------------------

    effective_start = 0 if start_frame is None else start_frame
    effective_end = (
        total_frames - 1
        if end_frame is None
        else end_frame
    )

    expected_rows = gold_rows[
        (gold_rows[:, 2] >= effective_start)
        & (gold_rows[:, 2] <= effective_end)
    ]

    assert expected_rows.shape[0] > 0
    assert resumed_rows.shape == expected_rows.shape

    np.testing.assert_array_equal(
        resumed_rows,
        expected_rows,
    )