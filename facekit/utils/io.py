from __future__ import annotations
from pathlib import Path
import os
import tempfile
import numpy as np
from typing import Union

def fsync_parent_dir(path: Union[str, os.PathLike]) -> None:
    """Best-effort fsync() of the parent dir to durably persist a prior os.replace()."""
    p = Path(path)
    try:
        fd = os.open(str(p.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass

def atomic_write_npz(
    dst: Union[str, Path],
    **named_arrays: np.ndarray,
) -> str:
    """
    Atomically write one or more named numpy arrays to an .npz file.

    Usage:
        atomic_write_npz(path, observations=arr)
        atomic_write_npz(path, embeddings=emb_arr)
        atomic_write_npz(path, observations=arr, embeddings=emb_arr)
    Args:
        dst: Target file path ('.npz' suffix enforced).
        **named_arrays: Mapping of name -> array(s) to store in the NPZ.
    Returns:
        Final file path as a string.
    Raises:
        ValueError if no arrays are provided.
    """
    if not named_arrays:
        raise ValueError("Provide at least one named array, e.g. observations=arr")

    dst = Path(dst).with_suffix(".npz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    named_arrays = {k: np.asarray(v) for k, v in named_arrays.items()}

    tmp = tempfile.NamedTemporaryFile(dir=dst.parent, suffix=".tmp", delete=False)
    try:
        np.savez(tmp, **named_arrays)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, dst)
        fsync_parent_dir(dst)
        return str(dst)
    finally:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass

def atomic_write_npy(dst: Union[str, Path], array: np.ndarray) -> str:
    """
    Atomically write single ndarray to an .npy file.

    Usage:
        atomic_write_npy(path, observations=arr)
        atomic_write_npy(path, embeddings=emb_arr)
        atomic_write_npy(path, observations=arr, embeddings=emb_arr)
    Args:
        dst: Target file path ('.npy' suffix enforced).
        **named_arrays: Mapping of name -> array(s) to store in the NPY.
    Returns:
        Final file path as a string.
    Raises:
        ValueError if no arrays are provided.
    """
    dst = Path(dst).with_suffix(".npy")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(dir=dst.parent, suffix=".tmp", delete=False)
    try:
        np.save(tmp, array)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, dst)
        fsync_parent_dir(dst)
        return str(dst)
    finally:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass
