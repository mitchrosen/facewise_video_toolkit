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
# Tests: Phase 1 reassignment
# ----------------------------

def test_short_track_competition_same_gid_winner_keeps_loser_falls_back():
    """
    Two short tracks overlap in time and both want gid=10.
    The better-fitting one (higher IoU evidence) keeps gid=10.
    The loser should fall back to gid=11 (its second-best candidate),
    not be pruned just because it lost.
    """
    shot = 1

    # Left caps:
    # L10 ends at 4; bbox is very close to short1 start bbox -> strong IoU
    L10 = make_track(
        shot_id=shot, track_id=100, global_id=10,
        frames_and_bboxes=[(4, (10, 10, 50, 50))]
    )
    # L11 also ends at 4; bbox is closer to short2 start bbox -> good fallback
    L11 = make_track(
        shot_id=shot, track_id=101, global_id=11,
        frames_and_bboxes=[(4, (100, 100, 140, 140))]
    )

    # Two short tracks that overlap in time:
    # short1: frames 5-6 near L10 bbox => wants gid=10
    S1 = make_track(
        shot_id=shot, track_id=1, global_id=None,
        frames_and_bboxes=[(5, (11, 11, 51, 51)), (6, (12, 12, 52, 52))]
    )
    # short2: frames 6-7 closer to L11 bbox, but still can "see" L10 as a candidate
    S2 = make_track(
        shot_id=shot, track_id=2, global_id=None,
        frames_and_bboxes=[(6, (102, 102, 142, 142)), (7, (103, 103, 143, 143))]
    )

    tracks = [L10, L11, S1, S2]

    out = apply_track_consolidation_and_pruning(
        tracks,
        min_gap_len=10,
        min_track_len=3,      # S1 and S2 are short (len 2)
        iou_threshold=0.1,    # permissive
    )

    # S1 should take gid=10; S2 should take gid=11 (fallback)
    by_tid = {t.track_id: t for t in out}
    assert by_tid[1].global_id == 10
    assert by_tid[2].global_id == 11

    # Both should survive (they now have gids)
    assert 1 in by_tid and 2 in by_tid


def test_short_track_cannot_take_gid_if_it_overlaps_fixed_track_with_that_gid():
    """
    If a short track overlaps in time with a fixed (non-short) track that already has gid=10,
    it must not be reassigned to gid=10 even if IoU would suggest it.
    It should fall back to gid=11 if available.
    """
    shot = 1

    # Fixed (long) track with gid=10 spanning 5..20
    fixed10 = make_track(
        shot_id=shot, track_id=200, global_id=10,
        frames_and_bboxes=[(5, (10, 10, 50, 50)), (20, (10, 10, 50, 50))]
    )
    # Left caps for gid=10 and gid=11 end at 9, within min_gap_len of the short track at 10
    L10 = make_track(
        shot_id=shot, track_id=201, global_id=10,
        frames_and_bboxes=[(9, (10, 10, 50, 50))]
    )
    L11 = make_track(
        shot_id=shot, track_id=202, global_id=11,
        frames_and_bboxes=[(9, (12, 12, 52, 52))]
    )

    # Short track overlaps fixed10 in time (10..11)
    S = make_track(
        shot_id=shot, track_id=3, global_id=None,
        frames_and_bboxes=[(10, (11, 11, 51, 51)), (11, (12, 12, 52, 52))]
    )

    tracks = [fixed10, L10, L11, S]

    out = apply_track_consolidation_and_pruning(
        tracks,
        min_gap_len=10,
        min_track_len=3,     # S is short
        iou_threshold=0.1,
    )

    by_tid = {t.track_id: t for t in out}
    assert by_tid[3].global_id == 11, "Should fall back because gid=10 overlaps fixed gid=10 track"


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
