from __future__ import annotations

from pathlib import Path
import pytest


HERE = Path(__file__).resolve().parent  # tests/e2e_realvideo


def _arg_selects_this_dir(arg: str) -> bool:
    """
    Return True iff this CLI arg is a path (optionally with ::nodeid)
    that resolves to something under HERE.
    """
    if not arg or arg.startswith("-"):
        return False

    # Handle nodeid forms like: path/to/test.py::test_name
    path_part = arg.split("::", 1)[0]
    p = Path(path_part)

    # If it doesn't look like a filesystem path, ignore it.
    # (This avoids treating -k expressions, markers, etc. as selectors.)
    # You can tweak this if you intentionally pass non-existing paths.
    if not (p.exists() or (Path.cwd() / p).exists()):
        return False

    try:
        rp = (p if p.is_absolute() else (Path.cwd() / p)).resolve()
    except Exception:
        return False

    # Python 3.11 has is_relative_to
    return rp.is_relative_to(HERE)


def pytest_collection_modifyitems(config, items):
    # Only allow these tests if the user explicitly selected a path under HERE.
    explicitly_selected = any(
        _arg_selects_this_dir(arg) for arg in config.invocation_params.args
    )

    if explicitly_selected:
        return

    skip_marker = pytest.mark.skip(
        reason=f"Tests under {HERE} are skipped unless explicitly selected by path."
    )

    for item in items:
        # Only skip tests that are in/under this directory
        try:
            item_path = Path(str(item.fspath)).resolve()
        except Exception:
            item.add_marker(skip_marker)
            continue

        if item_path.is_relative_to(HERE):
            item.add_marker(skip_marker)
