import pytest
import numpy as np

from facekit.output.json_v2 import (
    V2WriterConfig,
    build_v2_1_manifest_from_tracks,
    ObservationsCollector,
)
from facekit.tracking.face_structures import FaceObservation
from facekit.common.obs_consts import Source


class _Track:
    """Minimal track stub for json_v2 writers."""
    def __init__(self, shot_id: int, track_id: int, obs):
        self.shot_id = int(shot_id)
        self.track_id = int(track_id)
        self.observations = list(obs)

    def first_frame(self):
        return int(self.observations[0].frame_idx)

    def last_frame(self):
        return int(self.observations[-1].frame_idx)


def test_v2_1_summary_stats_clipped_but_raw_obs_untouched(tmp_path):
    """
    Regression: tracking can drift outside frame -> summary stats must be clipped
    so normalized widths/heights stay <= 100, but raw obs boxes must remain unchanged.
    """
    W, H = 100, 100  # easy: percent == pixels

    # Raw obs drift outside frame (negative + >W/H)
    raw_bbox = (-50, -10, 150, 210)  # width=200, height=220 -> would normalize to 200%, 220%
    obs = [
        FaceObservation(frame_idx=0, source=Source.TRACKED, bbox=raw_bbox, confidence=0.9),
        FaceObservation(frame_idx=1, source=Source.TRACKED, bbox=raw_bbox, confidence=0.9),
    ]
    t = _Track(shot_id=1, track_id=0, obs=obs)

    cfg = V2WriterConfig(
        video_path="toy.mp4",
        video_size=(W, H),
        total_frames=2,
        fps=30.0,
        normalize_to_percent=True,
        emb_store=None,  # embeddings irrelevant here
    )

    c = ObservationsCollector()
    c.append_track_obs(
        [
            {
                "shot": 1,
                "track_id": 0,
                "f": 0,
                "bbox_xyxy": list(raw_bbox),
                "src": "tracked",
                "conf": 0.9,
            },
            {
                "shot": 1,
                "track_id": 0,
                "f": 1,
                "bbox_xyxy": list(raw_bbox),
                "src": "tracked",
                "conf": 0.9,
            },
        ]
    )

    manifest = build_v2_1_manifest_from_tracks([t], cfg, obs_collector=c)

    # --- Verify summary stats got clipped (schema-friendly) ---
    shot = manifest["shots"][0]
    face_track = shot["face_tracks"][0]

    # With clipping, bbox becomes (0,0,100,100) => width/height 100%
    assert face_track["avg_face_width"] <= 100.0
    assert face_track["avg_face_height"] <= 100.0
    assert 0.0 <= face_track["avg_center_x"] <= 100.0
    assert 0.0 <= face_track["avg_center_y"] <= 100.0

    # --- Verify raw per-frame obs are NOT modified ---
    # In v2.1, the authoritative raw obs are in the NPZ sidecar. Verify those bboxes are unchanged.
    out_npz = tmp_path / "obs_sidecar.npz"
    c.finalize_sidecar(out_npz)
    with np.load(out_npz, allow_pickle=False) as data:
        arr = data["observations"]

    assert arr.size == 2
    assert list(map(int, arr["track_id"])) == [0, 0]
    assert [tuple(map(float, bb)) for bb in arr["bbox_xyxy"]] == [tuple(map(float, raw_bbox)), tuple(map(float, raw_bbox))]


def test_v2_1_no_video_size_does_not_clip(tmp_path):
    """
    If video_size is missing (W/H=0), we should not clip, and we should not crash.
    """
    raw_bbox = (-50, -10, 150, 210)
    obs = [
        FaceObservation(frame_idx=0, source=Source.TRACKED, bbox=raw_bbox),
    ]
    t = _Track(shot_id=1, track_id=0, obs=obs)

    cfg = V2WriterConfig(
        video_path="toy.mp4",
        video_size=None,            # <- triggers W/H = (0,0)
        normalize_to_percent=True,  # normalization becomes no-op without W/H
        emb_store=None,
    )

    c = ObservationsCollector()
    c.append_track_obs(
        [
            {
                "shot": 1,
                "track_id": 0,
                "f": 0,
                "bbox_xyxy": list(raw_bbox),
                "src": "tracked",
                "conf": 0.9,
            },
            {
                "shot": 1,
                "track_id": 0,
                "f": 1,
                "bbox_xyxy": list(raw_bbox),
                "src": "tracked",
                "conf": 0.9,
            },
        ]
    )

    manifest = build_v2_1_manifest_from_tracks([t], cfg, obs_collector=c)

    ft = manifest["shots"][0]["face_tracks"][0]
    # With W/H=0, _normalize returns raw center/size; values may exceed 100 and that's OK
    # because schema validation should only be run when video_size is known.
    assert "avg_face_width" in ft
    assert "avg_face_height" in ft

    # Still: sidecar should be writable and contain the raw bbox as-is (no clipping without video_size).
    out_npz = tmp_path / "obs_sidecar.npz"
    c.finalize_sidecar(out_npz)
    with np.load(out_npz, allow_pickle=False) as data:
        arr = data["observations"]
    assert arr.size >= 1
    assert tuple(map(float, arr["bbox_xyxy"][0])) == tuple(map(float, raw_bbox))


def test_v2_1_offsets_slice_npz_correctly_when_appends_interleaved(tmp_path):
    """
    For interleaved observations across tracks, the manifest's
    (obs_offset, obs_count) must slice the NPZ correctly for each track.
    """
    c = ObservationsCollector()

    # Interleaved appends:
    # track 0: f=0, then later f=2
    # track 1: f=1
    c.append_track_obs([{"shot": 1, "track_id": 0, "f": 0, "bbox_xyxy": [0, 0, 10, 10], "src": "tracked", "conf": 0.9}])
    c.append_track_obs([{"shot": 1, "track_id": 1, "f": 1, "bbox_xyxy": [0, 0, 10, 10], "src": "tracked", "conf": 0.9}])
    c.append_track_obs([{"shot": 1, "track_id": 0, "f": 2, "bbox_xyxy": [0, 0, 10, 10], "src": "tracked", "conf": 0.9}])

    t0 = _Track(shot_id=1, track_id=0, obs=[
        FaceObservation(frame_idx=0, source=Source.TRACKED, track_id=0, bbox=(0, 0, 10, 10), confidence=0.9),
        FaceObservation(frame_idx=2, source=Source.TRACKED, track_id=0, bbox=(0, 0, 10, 10), confidence=0.9),
    ])
    t1 = _Track(shot_id=1, track_id=1, obs=[
        FaceObservation(frame_idx=1, source=Source.TRACKED, track_id=1, bbox=(0, 0, 10, 10), confidence=0.9),
    ])

    cfg = V2WriterConfig(video_size=(100, 100), emb_store=None)
    manifest = build_v2_1_manifest_from_tracks([t0, t1], cfg, obs_collector=c)

    out_npz = tmp_path / "obs_sidecar.npz"
    c.finalize_sidecar(out_npz)
    with np.load(out_npz, allow_pickle=False) as data:
        arr = data["observations"]

    shot = manifest["shots"][0]
    tracks = shot["face_tracks"]
    assert len(tracks) == 2

    # Identify which manifest entry corresponds to track_id 0 vs 1.
    # In this stub setup, face_label should be face_{track_id}.
    by_label = {t["face_label"]: t for t in tracks}
    ft0 = by_label["face_0"]
    ft1 = by_label["face_1"]

    off0, cnt0 = int(ft0["obs_offset"]), int(ft0["obs_count"])
    off1, cnt1 = int(ft1["obs_offset"]), int(ft1["obs_count"])

    assert (off0, cnt0) == (0, 2)
    assert (off1, cnt1) == (2, 1)

    sl0 = arr[off0:off0+cnt0]
    sl1 = arr[off1:off1+cnt1]
    assert list(map(int, sl0["track_id"])) == [0, 0]
    assert list(map(int, sl0["f"])) == [0, 2]
    assert list(map(int, sl1["track_id"])) == [1]
    assert list(map(int, sl1["f"])) == [1]