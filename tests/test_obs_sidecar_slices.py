import numpy as np
from pathlib import Path
import pytest

from facekit.output.json_v2 import ObservationsCollector
from facekit.output.json_v2 import build_v2_1_manifest_from_tracks, V2WriterConfig
from facekit.tracking.face_structures import FaceObservation
from facekit.common.obs_consts import Source


def _row(shot: int, tid: int, f: int):
    # Minimal row dict accepted by append_track_obs
    return {
        "shot": int(shot),
        "track_id": int(tid),
        "f": int(f),
        "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
        # "tracked" is fine here; we avoid DETECTED landmarks requirements.
        "src": "tracked",
        "conf": 0.9,
        # no landmarks required for tracked
    }


def test_obs_sidecar_slices_are_track_contiguous_even_if_appended_interleaved(tmp_path: Path):
    c = ObservationsCollector()

    # Interleave appends:
    # track 0: frame 0, then later frame 2
    # track 1: frame 1
    c.append_track_obs([_row(shot=1, tid=0, f=0)])
    c.append_track_obs([_row(shot=1, tid=1, f=1)])
    c.append_track_obs([_row(shot=1, tid=0, f=2)])

    # The deterministic sidecar ordering is (shot, track_id, frame):
    # (1,0,f=0), (1,0,f=2), (1,1,f=1)
    off0, cnt0 = c.slice_for_track(1, 0)
    off1, cnt1 = c.slice_for_track(1, 1)

    assert (off0, cnt0) == (0, 2)
    assert (off1, cnt1) == (2, 1)

    # Validate the actual sidecar contents match the slice.
    out_npz = tmp_path / "obs.npz"
    c.finalize_sidecar(out_npz)
    with np.load(out_npz, allow_pickle=False) as data:
        arr = data["observations"]

    # Slice for track 0 is contiguous and ordered by frame
    t0 = arr[off0 : off0 + cnt0]
    assert list(map(int, t0["track_id"])) == [0, 0]
    assert list(map(int, t0["f"])) == [0, 2]

    # Slice for track 1
    t1 = arr[off1 : off1 + cnt1]
    assert list(map(int, t1["track_id"])) == [1]
    assert list(map(int, t1["f"])) == [1]

def test_v21_manifest_requires_obs_collector():
    class _Track:
        def __init__(self, shot_id: int, track_id: int, frames: list[int]):
            self.shot_id = shot_id
            self.track_id = track_id
            self.observations = [
                FaceObservation(
                    frame_idx=f,
                    source=Source.TRACKED,
                    track_id=track_id,
                    bbox=(0, 0, 10, 10),
                    confidence=0.9,
                )
                for f in frames
            ]

    tracks = [_Track(shot_id=1, track_id=0, frames=[0, 1])]
    cfg = V2WriterConfig(video_size=(100, 100))

    with pytest.raises(ValueError, match=r"requires obs_collector"):
        build_v2_1_manifest_from_tracks(
            tracks,
            cfg,
            obs_collector=None,
        )