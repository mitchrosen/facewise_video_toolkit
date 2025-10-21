# facekit/utils/io.py
import os
import errno
from pathlib import Path
from typing import Union

def fsync_parent_dir(final_path: Union[str, Path]) -> None:
    """
    Best-effort: fsync the *directory that contains* `final_path`.
    Useful after atomic os.replace(tmp, final_path) to reduce risk of metadata loss
    on crash/power failure. Safe no-op on filesystems/platforms that don't support it.

    Parameters
    ----------
    final_path : str | Path
        Path to the final file that was just atomically replaced/written.
    """
    p = Path(final_path)
    dir_path = p if p.is_dir() else p.parent

    # Some platforms lack O_DIRECTORY. Use it when available; otherwise fall back.
    flags = getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_RDONLY", 0)
    try:
        dirfd = os.open(str(dir_path), flags)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except FileNotFoundError:
        # Directory vanished? Nothing we can do.
        return
    except OSError as e:
        # Common on FUSE/Google Drive or filesystems that don't support dir fsync.
        # Ignore only "operation not supported"/"invalid argument".
        if e.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EPERM):
            # Re-raise unexpected errors so we don't hide real problems.
            raise
