import numpy as np
import pytest

from facekit.io.frame_provider import ReaderCoordinator
from tests.utils.video_mocks import synthetic_frame

@pytest.fixture(autouse=True)
def patch_videoreader(monkeypatch):
    # Import the fake here so it's definitely in scope for the patch
    from tests.utils.video_mocks import FakeVideoReader
    import facekit.io.frame_provider as fp_mod
    monkeypatch.setattr(fp_mod, "VideoReader", FakeVideoReader)

@pytest.fixture()
def reader():
    # Small seq_chunk to exercise buffer refills across boundaries
    rd = ReaderCoordinator("dummy.mp4", seq_chunk=4, lru_size=8)
    try:
        yield rd
    finally:
        rd.close()

def test_random_access_absolute(reader):
    # Spot-check a few indices map 1:1 to our synthetic generator
    for i in [0, 1, 2, 5, 10, 50, 90, 100, 108, 109]:
        ra = reader.get_frame(i)
        exp = synthetic_frame(i)
        assert ra is not None
        assert np.array_equal(ra, exp), f"rand access mismatch at {i}"

def test_seq_after_reset_returns_exact_index(reader):
    i = 42
    reader.reset_to_frame(i)
    seq = reader.next()
    exp = synthetic_frame(i)
    assert seq is not None
    assert np.array_equal(seq, exp), "reset_to_frame(i); next() did not yield frame i"

def test_seq_vs_rand_same_index(reader):
    i = 77
    reader.reset_to_frame(i)
    a = reader.next()
    b = reader.get_frame(i)
    assert a is not None and b is not None
    assert np.array_equal(a, b), "seq next() != rand get_frame(i)"

def test_sequential_order_across_buffer_boundaries(reader):
    # With seq_chunk=4 this crosses several fills
    start, count = 6, 15
    reader.reset_to_frame(start)
    for k in range(count):
        f = reader.next()
        exp = synthetic_frame(start + k)
        assert np.array_equal(f, exp), f"sequential mismatch at {start+k}"

def test_negative_indices_rejected(reader):
    with pytest.raises(ValueError):
        reader.get_frame(-1)
    with pytest.raises(ValueError):
        reader.reset_to_frame(-5)

def test_out_of_range_returns_none(reader):
    tf = reader.total_frames()
    assert isinstance(tf, int) and tf > 0
    assert reader.get_frame(tf) is None  # last valid is tf-1
