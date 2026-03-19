import pytest

from facekit.common.obs_consts import Source
from facekit.tracking.face_structures import FaceTrack, FaceObservation

from facekit.tracking.track_consolidation_and_pruning import (
    apply_track_consolidation_and_pruning,
)


# ----------------------------
# Test helpers
# ----------------------------

def obs(frame_idx: int, bbox, *, src=Source.DETECTED) -> FaceObservation:
    return FaceObservation(
        frame_idx=int(frame_idx),
        bbox=tuple(map(int, bbox)),
        source=src,
        track_id=None,
        embedding=None,
        confidence=None,
        aligned_face=None,
        landmarks=None,
    )

def make_track(
    *,
    shot_id: int,
    track_id: int,
    global_id: int | None,
    frames_and_bboxes: list[tuple[int, tuple[int, int, int, int]]],
) -> FaceTrack:
    t = FaceTrack(shot_id=int(shot_id), track_id=int(track_id), global_id=global_id)
    # Ensure open so add_observation works in your implementation
    if hasattr(t, "mark_open"):
        t.mark_open()
    for f, bb in frames_and_bboxes:
        o = obs(f, bb, src=Source.DETECTED)
        o.track_id = int(track_id)
        try:
            o.shot_id = int(shot_id)
        except Exception:
            pass
        t.add_observation(o)
    return t

def frames(t: FaceTrack) -> list[int]:
    return list(map(int, t.get_frame_indices()))

def sources_by_frame(t: FaceTrack) -> dict[int, Source]:
    return {int(o.frame_idx): o.source for o in (t.observations or [])}

# ----------------------------
# Tests: reassignment + final survival contract
# ----------------------------

def test_reassigned_short_tracks_are_pruned_from_final_output():
    """
    Two short tracks can still participate in reassignment competition, but
    final output should prune them unconditionally if they remain below
    min_track_len.
    """
    shot = 1

    L10 = make_track(
        shot_id=shot, track_id=100, global_id=10,
        frames_and_bboxes=[(4, (10, 10, 50, 50))]
    )
    L11 = make_track(
        shot_id=shot, track_id=101, global_id=11,
        frames_and_bboxes=[(4, (100, 100, 140, 140))]
    )

    S1 = make_track(
        shot_id=shot, track_id=1, global_id=None,
        frames_and_bboxes=[(5, (11, 11, 51, 51)), (6, (12, 12, 52, 52))]
    )
    S2 = make_track(
        shot_id=shot, track_id=2, global_id=None,
        frames_and_bboxes=[(6, (102, 102, 142, 142)), (7, (103, 103, 143, 143))]
    )

    out = apply_track_consolidation_and_pruning(
        [L10, L11, S1, S2],
        min_gap_len=10,
        min_track_len=3,   # S1/S2 are len 2, so they should be pruned
        iou_threshold=0.1,
    )

    by_tid = {t.track_id: t for t in out}
    assert 1 not in by_tid
    assert 2 not in by_tid

def test_long_enough_overlapping_track_survives_final_output():
    """
    Final prune should keep a track that is long enough, even if it does not
    receive a fallback global_id in this end-to-end path.
    """
    shot = 1

    fixed10 = make_track(
        shot_id=shot, track_id=200, global_id=10,
        frames_and_bboxes=[(5, (10, 10, 50, 50)), (20, (10, 10, 50, 50))]
    )
    S = make_track(
        shot_id=shot, track_id=3, global_id=None,
        frames_and_bboxes=[
            (10, (11, 11, 51, 51)),
            (11, (12, 12, 52, 52)),
            (12, (13, 13, 53, 53)),
        ],
    )

    out = apply_track_consolidation_and_pruning(
        [fixed10, S],
        min_gap_len=10,
        min_track_len=3,
        iou_threshold=0.1,
    )

    by_tid = {t.track_id: t for t in out}
    assert 3 in by_tid

def test_short_track_that_cannot_take_overlapping_gid_is_pruned():
    """
    A short track that cannot take gid=10 because it overlaps a fixed gid=10
    track may still fall back internally, but it should not survive final
    output if it remains shorter than min_track_len.
    """
    shot = 1

    fixed10 = make_track(
        shot_id=shot, track_id=200, global_id=10,
        frames_and_bboxes=[(5, (10, 10, 50, 50)), (20, (10, 10, 50, 50))]
    )
    L10 = make_track(
        shot_id=shot, track_id=201, global_id=10,
        frames_and_bboxes=[(9, (10, 10, 50, 50))]
    )
    L11 = make_track(
        shot_id=shot, track_id=202, global_id=11,
        frames_and_bboxes=[(9, (12, 12, 52, 52))]
    )

    S = make_track(
        shot_id=shot, track_id=3, global_id=None,
        frames_and_bboxes=[(10, (11, 11, 51, 51)), (11, (12, 12, 52, 52))]
    )

    out = apply_track_consolidation_and_pruning(
        [fixed10, L10, L11, S],
        min_gap_len=10,
        min_track_len=3,   # S is len 2, so it should be pruned
        iou_threshold=0.1,
    )

    by_tid = {t.track_id: t for t in out}
    assert 3 not in by_tid

def test_long_enough_unassigned_track_survives_final_output():
    """
    Final prune should keep tracks whose length meets min_track_len,
    even if they still have global_id=None.
    """
    shot = 1
    T = make_track(
        shot_id=shot, track_id=1, global_id=None,
        frames_and_bboxes=[
            (5, (11, 11, 51, 51)),
            (6, (12, 12, 52, 52)),
            (7, (13, 13, 53, 53)),
        ],
    )

    out = apply_track_consolidation_and_pruning(
        [T],
        min_gap_len=10,
        min_track_len=3,
        iou_threshold=0.1,
    )

    by_tid = {t.track_id: t for t in out}
    assert 1 in by_tid
    assert by_tid[1].global_id is None

# ----------------------------
# Tests: Phase 3 gap filling + merge
# ----------------------------

def test_fill_gap_merges_two_tracks_and_injects_interpolated_frames():
    """
    Same gid tracks A then B with a short gap should:
      - inject interpolated observations into A for every gap frame
      - merge B into A
      - remove B from output list
    """
    if not hasattr(Source, "INTERPOLATED"):
        pytest.skip("Source.INTERPOLATED not defined yet")

    shot = 1

    A = make_track(
        shot_id=shot, track_id=10, global_id=1,
        frames_and_bboxes=[(0, (0, 0, 10, 10)), (2, (0, 0, 10, 10))]
    )
    # Gap frames are 3 and 4 (since B starts at 5)
    B = make_track(
        shot_id=shot, track_id=11, global_id=1,
        frames_and_bboxes=[(5, (1, 1, 11, 11)), (6, (1, 1, 11, 11))]
    )

    tracks = [A, B]

    out = apply_track_consolidation_and_pruning(
        tracks,
        min_gap_len=5,       # gap_len=2 => fill
        min_track_len=1,     # no pruning concerns
        iou_threshold=0.0,   # ensure boundary passes
    )

    # Should now be ONE surviving gid=1 track in this shot
    gid1 = [t for t in out if t.shot_id == shot and t.global_id == 1]
    assert len(gid1) == 1

    T = gid1[0]
    fs = frames(T)
    assert fs == [0, 2, 3, 4, 5, 6] or fs == [0, 2, 3, 4, 5, 6], "Expected merged + gap frames"

    srcs = sources_by_frame(T)
    assert srcs[3] == Source.INTERPOLATED
    assert srcs[4] == Source.INTERPOLATED
    assert srcs[5] == Source.DETECTED
    assert srcs[6] == Source.DETECTED


def test_fill_gap_is_blocked_if_any_gap_frame_is_occupied_by_any_track():
    """
    No frame skipping: if any gap frame is occupied by ANY track in the shot,
    we do not fill and therefore do not merge.
    """
    if not hasattr(Source, "INTERPOLATED"):
        pytest.skip("Source.INTERPOLATED not defined yet")

    shot = 1

    A = make_track(
        shot_id=shot, track_id=10, global_id=1,
        frames_and_bboxes=[(0, (0, 0, 10, 10)), (2, (0, 0, 10, 10))]
    )
    B = make_track(
        shot_id=shot, track_id=11, global_id=1,
        frames_and_bboxes=[(5, (1, 1, 11, 11))]
    )

    # Occupier has an observation at frame 3 (in the gap 3..4)
    OCC = make_track(
        shot_id=shot, track_id=99, global_id=999,
        frames_and_bboxes=[(3, (200, 200, 210, 210))]
    )

    tracks = [A, B, OCC]

    out = apply_track_consolidation_and_pruning(
        tracks,
        min_gap_len=5,
        min_track_len=1,
        iou_threshold=0.0,
    )

    # Should NOT merge; both gid=1 tracks remain
    gid1 = sorted([t for t in out if t.shot_id == shot and t.global_id == 1], key=lambda t: t.track_id)
    assert len(gid1) == 2

    # And no interpolated observations should have been added to A
    srcsA = sources_by_frame(gid1[0])
    assert 3 not in srcsA or srcsA[3] != Source.INTERPOLATED
    assert 4 not in srcsA or srcsA[4] != Source.INTERPOLATED
 

 # ----------------------------
# Tests: ordering + final prune contract
# ----------------------------

def test_apply_consolidation_runs_before_final_prune(monkeypatch):
    """
    Desired contract:
      short tracks get a chance to be rescued by consolidation before final prune.

    Under the current implementation, prune runs first, so this test should fail.
    """
    class DummyTrack:
        def __init__(self, shot_id, track_id, length, global_id=None):
            self.shot_id = shot_id
            self.track_id = track_id
            self.global_id = global_id
            self._length = length

    rescued = DummyTrack(shot_id=1, track_id=10, length=2, global_id=None)

    def fake_group_tracks_by_shot(tracks):
        return {1: [rescued]}

    def fake_propose_gid_reassignments_for_short_tracks(
        shot_tracks, *, min_gap_len, min_track_len, iou_threshold
    ):
        return []

    def fake_apply_assigned_global_ids(shot_tracks, assignments):
        return None

    def fake_fill_short_gaps_within_shot_and_merge(
        shot_tracks, *, min_gap_len, iou_threshold
    ):
        # Under the desired ordering, the short track should still be present here
        # and can be "rescued" before final prune.
        assert len(shot_tracks) == 1, (
            "track was pruned before consolidation ran"
        )
        shot_tracks[0]._length = 100

    def fake_track_len(track):
        return int(track._length)

    monkeypatch.setattr(
        "facekit.tracking.track_consolidation_and_pruning.group_tracks_by_shot",
        fake_group_tracks_by_shot,
    )
    monkeypatch.setattr(
        "facekit.tracking.track_consolidation_and_pruning.propose_gid_reassignments_for_short_tracks",
        fake_propose_gid_reassignments_for_short_tracks,
    )
    monkeypatch.setattr(
        "facekit.tracking.track_consolidation_and_pruning.apply_assigned_global_ids",
        fake_apply_assigned_global_ids,
    )
    monkeypatch.setattr(
        "facekit.tracking.track_consolidation_and_pruning.fill_short_gaps_within_shot_and_merge",
        fake_fill_short_gaps_within_shot_and_merge,
    )
    monkeypatch.setattr(
        "facekit.tracking.track_consolidation_and_pruning.track_len",
        fake_track_len,
    )

    out = apply_track_consolidation_and_pruning(
        [rescued],
        min_gap_len=5,
        min_track_len=70,
        iou_threshold=0.0,
    )

    assert len(out) == 1
    assert out[0].track_id == 10
    assert out[0]._length == 100


def test_final_prune_drops_short_track_even_if_it_has_global_id():
    """
    Desired contract:
      after all reassignment / merge attempts are done, any remaining track
      shorter than min_track_len is dropped unconditionally.

    This track has a global_id, but it is still too short and cannot merge with
    anything else, so it should not survive final output.
    """
    shot = 1

    short_with_gid = make_track(
        shot_id=shot, track_id=50, global_id=123,
        frames_and_bboxes=[(10, (50, 50, 60, 60)), (11, (50, 50, 60, 60))]
    )

    out = apply_track_consolidation_and_pruning(
        [short_with_gid],
        min_gap_len=5,
        min_track_len=5,
        iou_threshold=0.0,
    )

    by_tid = {t.track_id: t for t in out}
    assert 50 not in by_tid, (
        "Short track should be pruned from final output even though it has a global_id"
    )