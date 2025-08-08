from fractions import Fraction
import numpy as np
import pytest
from facekit.utils.video_reader import VideoReader


def test_inclusive_end_no_tail_drop_with_pts_jitter(patch_av_open, mk_frames, mk_container):
    n, fps, tb = 144, 30.0, Fraction(1, 30)

    def jitter(i): return 0.0005 if i >= n - 5 else 0.0

    frames = mk_frames(n, fps=fps, time_base=tb, jitter=jitter)
    container = mk_container(frames, fps_num=30, fps_den=1, time_base=tb)
    patch_av_open(container)

    vr = VideoReader("dummy.mp4")
    imgs = vr.get_frames(0, n - 1)
    assert len(imgs) == n
    assert imgs[0][0, 0, 0] == 0
    assert imgs[-1][0, 0, 0] == n - 1
    vr.close()

def test_seek_backward_preroll_and_exact_start(patch_av_open, mk_frames, mk_container):
    n, fps, tb = 100, 30.0, Fraction(1, 30)
    frames = mk_frames(n, fps=fps, time_base=tb)
    container = mk_container(frames, fps_num=30, fps_den=1, time_base=tb)
    patch_av_open(container)

    start, end = 30, 60
    vr = VideoReader("dummy.mp4")
    imgs = vr.get_frames(start, end)
    assert len(imgs) == (end - start + 1)
    assert imgs[0][0, 0, 0] == start
    assert imgs[-1][0, 0, 0] == end
    vr.close()

def test_consecutive_shots_abut_no_overlap(patch_av_open, mk_frames, mk_container):
    n, fps, tb = 8, 30.0, Fraction(1, 30)
    frames = mk_frames(n, fps=fps, time_base=tb)
    container = mk_container(frames, fps_num=30, fps_den=1, time_base=tb)
    patch_av_open(container)

    vr = VideoReader("dummy.mp4")
    a = vr.get_frames(0, 2)
    b = vr.get_frames(3, 7)
    all_vals = [img[0, 0, 0].item() for img in (a + b)]
    assert len(a) == 3 and len(b) == 5
    assert all_vals == list(range(8))
    vr.close()

def test_handles_pts_none_uses_time_then_count(patch_av_open, mk_frames, mk_container):
    n, fps, tb = 20, 30.0, Fraction(1, 30)
    frames = mk_frames(n, fps=fps, time_base=tb, pts_none=True)
    container = mk_container(frames, fps_num=30, fps_den=1, time_base=tb)
    patch_av_open(container)

    start, end = 5, 12
    vr = VideoReader("dummy.mp4")
    imgs = vr.get_frames(start, end)
    assert len(imgs) == (end - start + 1)
    assert imgs[0][0, 0, 0] == start
    assert imgs[-1][0, 0, 0] == end
    vr.close()

def test_handles_no_pts_no_time_magicmocks_uses_count_preroll(patch_av_open, mk_frames, mk_container):
    n, fps, tb = 40, 30.0, Fraction(1, 30)
    frames = mk_frames(n, fps=fps, time_base=tb, pts_mock=True, time_mock=True)
    container = mk_container(frames, fps_num=30, fps_den=1, time_base=tb)
    patch_av_open(container)

    start, end = 25, 30
    vr = VideoReader("dummy.mp4")
    imgs = vr.get_frames(start, end)
    assert len(imgs) == (end - start + 1)
    assert imgs[0][0, 0, 0] == start
    assert imgs[-1][0, 0, 0] == end
    vr.close()

def test_fallback_when_seek_raises_uses_sequential_and_count(patch_av_open, mk_frames, mk_container, monkeypatch):
    n, fps, tb = 15, 30.0, Fraction(1, 30)
    frames = mk_frames(n, fps=fps, time_base=tb)

    # First av.open → container whose seek raises; second → fresh container for fallback reopen
    class RaisingContainer(mk_container(frames).__class__):
        def seek(self, *a, **k): raise OSError("seek not supported")

    first = RaisingContainer(frames)
    second = mk_container(frames)
    patch_av_open([first, second])  # side_effect sequence

    vr = VideoReader("dummy.mp4")
    # Assert and capture the expected fallback warning
    import pytest
    with pytest.warns(RuntimeWarning, match=r"Seek not supported"):
        imgs = vr.get_frames(4, 10)

    assert len(imgs) == (10 - 4 + 1)
    assert imgs[0][0, 0, 0] == 4
    assert imgs[-1][0, 0, 0] == 10
    vr.close()
