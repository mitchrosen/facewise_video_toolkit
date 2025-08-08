# tests/conftest.py
import pytest
from unittest.mock import patch
from fractions import Fraction

from tests.utils.pyav_fakes import FakeContainer, make_pts_time_sequence

@pytest.fixture
def patch_av_open():
    """
    Yields a function to patch facekit.utils.video_reader.av.open
    with a provided FakeContainer (or a sequence via side_effect).
    """
    patches = []

    def _apply(container_or_side_effect):
        p = patch("facekit.utils.video_reader.av.open", return_value=container_or_side_effect)
        if hasattr(container_or_side_effect, "__iter__") and not isinstance(container_or_side_effect, (bytes, str)):
            p = patch("facekit.utils.video_reader.av.open", side_effect=container_or_side_effect)
        patches.append(p)
        return p.__enter__()

    yield _apply
    for p in patches:
        p.__exit__(None, None, None)

@pytest.fixture
def mk_frames():
    return make_pts_time_sequence

@pytest.fixture
def mk_container():
    def _mk(frames, fps_num=30, fps_den=1, time_base=Fraction(1, 30)):
        return FakeContainer(frames, fps_num=fps_num, fps_den=fps_den, time_base=time_base)
    return _mk
