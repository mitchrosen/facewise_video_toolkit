import pytest

# Import the module so we can call positive_int directly
import facekit.cli.resolve_face_ids_v2_cli as cli


@pytest.mark.parametrize("bad", ["0", "-1", "-999", "bananas", "3.14", ""])
def test_positive_int_rejects_invalid(bad):
    with pytest.raises(Exception) as e:
        cli.positive_int(bad)
    # argparse.ArgumentTypeError is a subclass of Exception; keep it loose
    assert "must be > 0" in str(e.value) or "not a valid integer" in str(e.value)


@pytest.mark.parametrize("good", ["1", "2", "10", "999"])
def test_positive_int_accepts_valid(good):
    assert cli.positive_int(good) == int(good)