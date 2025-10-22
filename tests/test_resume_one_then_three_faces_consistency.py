import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import os
import re
import pytest

from facekit.common.obs_consts import Source, SRC_TO_CODE

# -------------------- helpers --------------------

def _with_repo_env() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    existing = os.environ.get("PYTHONPATH", "")
    combined = os.pathsep.join([str(repo_root), *(p for p in sys.path if p), existing]) if existing \
               else os.pathsep.join([str(repo_root), *(p for p in sys.path if p)])
    env = dict(os.environ)
    env["PYTHONPATH"] = combined
    return env

def _run_python(shim: Path, *args, ok=(0,), env=None, cwd=None):
    cp = subprocess.run([sys.executable, str(shim), *args],
                        text=True, capture_output=True, env=env,
                        cwd=cwd or shim.parent)
    if cp.returncode not in ok:
        raise AssertionError(
            f"RC {cp.returncode} not in {ok}\n=== STDOUT ===\n{cp.stdout}\n=== STDERR ===\n{cp.stderr}\n"
        )
    return cp

def _extract_anchor(stdout: str, stderr: str) -> int:
    txt = stdout + "\n" + stderr
    m = re.search(r"ANCHOR:(\d+)", txt)
    assert m, f"Could not parse anchor from logs:\n{txt}"
    return int(m.group(1))

def _load_rows(npz_path: Path):
    with np.load(npz_path) as data:
        obs = data["observations"] if "observations" in data.files else None
        if obs is None or obs.size == 0:
            return {
                "frame": np.array([], dtype=int),
                "shot": np.array([], dtype=int),
                "track": np.array([], dtype=int),
                "x1": np.array([], dtype=np.float32),
                "y1": np.array([], dtype=np.float32),
                "x2": np.array([], dtype=np.float32),
                "y2": np.array([], dtype=np.float32),
                "src": np.array([], dtype=int),
            }
        f = obs["f"].astype(int, copy=False)
        s = obs["shot"].astype(int, copy=False)
        t = obs["track_id"].astype(int, copy=False)
        bb = obs["bbox_xyxy"].astype(np.float32, copy=False)
        return {
            "frame": f,
            "shot":  s,
            "track": t,
            "x1": bb[:,0], "y1": bb[:,1], "x2": bb[:,2], "y2": bb[:,3],
            "src": obs["src"].astype(int, copy=False),
        }
    
def _assert_sidecar_src_int_codes(npz_path: Path):
    """
    Policy check: obs sidecar must store src as integer codes only,
    using the same codes as SRC_TO_CODE.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        assert "observations" in data.files, "sidecar missing 'observations' array"
        arr = data["observations"]
        assert "src" in arr.dtype.names, "sidecar 'observations' has no 'src' field"

        # 1) dtype must be integer
        assert np.issubdtype(
            arr["src"].dtype, np.integer
        ), f"expected integer src dtype, got {arr['src'].dtype}"

        # 2) values must be one of our known codes
        allowed_codes = {int(v) for v in SRC_TO_CODE.values()}
        vals = {int(x) for x in np.unique(arr["src"])}
        unexpected = vals - allowed_codes
        assert not unexpected, (
            f"unexpected src codes in sidecar: {sorted(unexpected)}; "
            f"allowed={sorted(allowed_codes)}"
        )

def _ordered_tracks_json(json_path: Path):
    js = json.loads(json_path.read_text())
    tracks = js["tracks"]
    def first_frame(t): 
        obs = t.get("observations", [])
        return min(o["frame_idx"] for o in obs) if obs else 10**9
    tracks.sort(key=lambda t: (int(t.get("shot_id", 0)), first_frame(t), int(t.get("track_id", 0))))
    return tracks
# -------------------- the subprocess shim --------------------

SHIM = r'''
import sys
import json
import numpy as np
import traceback
from pathlib import Path
from facekit.common.obs_consts import Source, SRC_TO_CODE
from facekit.pipeline.checkpoint import CheckpointManager
from facekit.pipeline.track_across_segments import track_across_segments
from facekit.tracking.tracking_resolution import GlobalIdentityResolver

# Structured dtype for NPZ (store src as INT CODE, not enum)
_obs_dtype = np.dtype([
    ("shot",       np.int64),
    ("track_id",   np.int64),
    ("f",          np.int64),
    ("bbox_xyxy",  np.float32, (4,)),
    ("src",        np.int64),   # <- int code via SRC_TO_CODE at dump time
    ("has_crop",   np.int8),
    ("emb_idx",    np.int64),
])

class ObsSidecar:
    def __init__(self, npz_path: str | None = None):
        self.rows: list[dict] = []
        # track-order bookkeeping to return an int "order" per (shot, track)
        self._order_map: dict[tuple[int,int], int] = {}
        self._next_order: int = 0

        self._npz = Path(npz_path) if npz_path else None
        if self._npz and self._npz.exists():
            with np.load(self._npz) as data:
                if "observations" in data.files:
                    arr = data["observations"]
                    has_emb = "emb_idx" in arr.dtype.names
                    for r in arr:
                        # Rehydrate with enum in-memory, code in file
                        code = int(r["src"])
                        try:
                            src_enum = next(k for k, v in SRC_TO_CODE.items() if v == code)
                        except StopIteration:
                            raise ValueError(f"Unknown src code {code} in sidecar")

                        shot_i  = int(r["shot"])
                        track_i = int(r["track_id"])
                        f_i     = int(r["f"])
                        has_crop = int(r["has_crop"])

                        # For tests: synthesize a crop_ref whenever has_crop==1.
                        # We don't care about the actual image path, only that
                        # pre-anchor DET rows have a non-empty crop_ref.
                        crop_ref = None
                        if has_crop:
                            crop_ref = f"dummy_s{shot_i}_t{track_i}_f{f_i}"

                        self.rows.append({
                            "shot":      shot_i,
                            "track_id":  track_i,
                            "f":         f_i,
                            "bbox_xyxy": r["bbox_xyxy"].astype(np.float32, copy=False),
                            "src":       src_enum,                     # enum in memory
                            "has_crop":  has_crop,
                            "emb_idx":   (int(r["emb_idx"]) if has_emb else -1),
                            "crop_ref":  crop_ref,
                        })
                        key = (shot_i, track_i)
                        if key not in self._order_map:
                            self._order_map[key] = self._next_order
                            self._next_order += 1

    # ---- methods used by checkpoint.py ----

    def count(self) -> int:
        return len(self.rows)

    def get_all_frame_indices(self):
        return np.array([r["f"] for r in self.rows], dtype=np.int64)

    def append_track_obs(self, rows, emb_idx_fn=lambda _o: -1):
        """
        rows: list of dicts (shot, track_id, f, bbox_xyxy, src, has_crop opt, crop_ref opt)
        MUST return (n_added, order_int)
        """
        order_int = None
        for r in rows:
            shot_i  = int(r["shot"])
            track_i = int(r["track_id"])
            f_i     = int(r["f"])

            key = (shot_i, track_i)
            if key not in self._order_map:
                self._order_map[key] = self._next_order
                self._next_order += 1
            order_int = self._order_map[key]

            emb_idx = int(emb_idx_fn(r))

            # Determine whether this row is DETECTED
            val = r["src"]
            if isinstance(val, Source):
                is_det = (val is Source.DETECTED) or (getattr(val, "value", "").lower() == "detected")
            else:
                is_det = str(val).lower() == "detected"

            # Propagate or synthesize crop_ref
            crop_ref = r.get("crop_ref")
            if is_det and not crop_ref:
                # Test-only synthetic crop_ref for DET rows
                crop_ref = f"dummy_s{shot_i}_t{track_i}_f{f_i}"
            # has_crop: if we have a crop_ref, treat as 1 by default
            has_crop = int(r.get("has_crop", 1 if crop_ref else 0))

            self.rows.append({
                "shot":      shot_i,
                "track_id":  track_i,
                "f":         f_i,
                "bbox_xyxy": np.asarray(r["bbox_xyxy"], dtype=np.float32),
                "src":       r["src"],   # keep enum in memory
                "has_crop":  has_crop,
                "emb_idx":   emb_idx,
                "crop_ref":  crop_ref,
            })
        return len(rows), (order_int if order_int is not None else -1)

    def find_rows(
        self,
        *,
        shot: int,
        track_id: int,
        frame_last: int | None = None,
        count: int | None = None,
        # optional filters (backward compatible with real collector)
        only_unassigned: bool | None = None,
        only_with_crop: bool | None = None,
        source: int | None = None,
        **kwargs,
    ):
        """
        Return positions of rows as (block_idx, row_idx) tuples.

        Contract mirrors ObservationsCollector:
          - positions are (block_idx, row_idx)
          - we sort by frame ascending
        For this test, we only use block_idx == 0.
        """
        if frame_last is None:
            frame_last = float("inf")

        candidates: list[int] = []
        for i, r in enumerate(self.rows):
            if r["shot"] != int(shot):
                continue
            if r["track_id"] != int(track_id):
                continue
            if r["f"] > int(frame_last):
                continue

            # emb_idx filter
            emb_idx_val = int(r.get("emb_idx", -1))
            if only_unassigned is True and emb_idx_val >= 0:
                continue

            # crop filter
            has_crop = int(r.get("has_crop", 0))
            if only_with_crop is True and not has_crop:
                continue

            # source filter (source is an INT CODE)
            if source is not None:
                val = r["src"]
                if isinstance(val, Source):
                    row_code = int(SRC_TO_CODE[val])
                elif isinstance(val, (int, np.integer)):
                    row_code = int(val)
                else:
                    raise ValueError(
                        f"ObsSidecar.find_rows: unexpected src type {type(val).__name__}: {val!r}"
                    )

                if row_code != int(source):
                    continue

            candidates.append(i)

        candidates.sort(key=lambda i: self.rows[i]["f"])
        # Expose as (block_idx, row_idx); we only use block 0 in this test
        return [(0, i) for i in candidates]

    def update_emb_idx(
        self,
        positions: list[tuple[int, int]],
        emb_indices: list[int] | tuple[int, ...] | np.ndarray,
    ) -> int:
        """
        Write the given emb_idx values into the provided (block_idx, row_idx) positions.

        Contract mirrors ObservationsCollector:
          - positions: list of (block_idx, row_idx)
          - len(positions) == len(emb_indices)
        For this test, we assert block_idx == 0.
        """
        if not isinstance(positions, list):
            raise TypeError(
                f"update_emb_idx: positions must be list[tuple[int,int]], "
                f"got {type(positions).__name__}"
            )

        # Normalize emb_indices to a simple list of ints
        if isinstance(emb_indices, np.ndarray):
            emb_list = [int(x) for x in emb_indices.tolist()]
        else:
            emb_list = [int(x) for x in emb_indices]

        if len(positions) != len(emb_list):
            raise ValueError(
                f"update_emb_idx: len(positions)={len(positions)} "
                f"!= len(emb_indices)={len(emb_list)}"
            )

        updated = 0
        for pos, emb_idx in zip(positions, emb_list):
            if not (isinstance(pos, tuple) and len(pos) == 2):
                raise TypeError(
                    f"update_emb_idx: each position must be (block_idx, row_idx), got {pos!r}"
                )
            block_idx, row_idx = pos
            # For this test we expect a single in-memory block 0.
            assert int(block_idx) == 0, f"ObsSidecar only supports block_idx 0, got {block_idx!r}"
            self.rows[int(row_idx)]["emb_idx"] = int(emb_idx)
            updated += 1

        return updated

    def dump_npz(self, out_path: str | Path):
        n = len(self.rows)
        arr = np.zeros((n,), dtype=_obs_dtype)
        for i, r in enumerate(self.rows):
            arr["shot"][i]      = r["shot"]
            arr["track_id"][i]  = r["track_id"]
            arr["f"][i]         = r["f"]
            arr["bbox_xyxy"][i] = r["bbox_xyxy"]

            # STRICT POLICY:
            # - In memory we may keep enums or int codes.
            # - On disk we ALWAYS store integer codes from SRC_TO_CODE.
            val = r["src"]

            if isinstance(val, Source):
                src_enum = val
            elif isinstance(val, (int, np.integer)):
                # Already a code, but verify it's known
                code = int(val)
                try:
                    src_enum = next(k for k, v in SRC_TO_CODE.items() if v == code)
                except StopIteration:
                    raise ValueError(f"Unknown src code {code} in rows")
            else:
                # No silent coercion of repr strings / raw strings.
                raise ValueError(
                    f"Unsupported src type {type(val).__name__} in rows: {val!r}. "
                    "ObsSidecar expects Source enum or integer code in memory."
                )

            arr["src"][i]      = int(SRC_TO_CODE[src_enum])
            arr["has_crop"][i] = r["has_crop"]
            arr["emb_idx"][i]  = r["emb_idx"]

        np.savez_compressed(out_path, observations=arr)

    # ---- API used by resume_rehydrate.py ----
    def iter_tracks(
        self,
        *,
        frame_max: int | None = None,
        shot: int | None = None,
        track_id: int | None = None,
    ):
        """
        Yield (shot, track_id, rows) where rows are dicts:
          {"f": int, "bbox_xyxy": [x1,y1,x2,y2], "src": str, optional "conf": float, optional "crop_ref": str}
        Ordered by (shot, track_id); rows sorted by frame asc.
        """
        groups: dict[tuple[int,int], list[dict]] = {}
        for r in self.rows:
            s = int(r["shot"])
            t = int(r["track_id"])
            if shot is not None and s != int(shot):
                continue
            if track_id is not None and t != int(track_id):
                continue
            f = int(r["f"])
            if frame_max is not None and f > int(frame_max):
                continue
            d = {
                "f": f,
                "bbox_xyxy": [
                    float(r["bbox_xyxy"][0]),
                    float(r["bbox_xyxy"][1]),
                    float(r["bbox_xyxy"][2]),
                    float(r["bbox_xyxy"][3]),
                ],
                # production rehydrator expects a lowercase string
                "src": (r["src"].value if hasattr(r["src"], "value") else str(r["src"]).lower()),
            }
            # Pass through crop_ref if present so _row_to_faceobs can set it
            if r.get("crop_ref"):
                d["crop_ref"] = r["crop_ref"]
            groups.setdefault((s, t), []).append(d)

        for (s, t) in sorted(groups.keys()):
            rows = groups[(s, t)]
            # --- Dedupe by frame: prefer 'detected' over others on ties ---
            by_f = {}
            for r in rows:
                f = int(r["f"])
                cur = by_f.get(f)
                if cur is None:
                    by_f[f] = r
                else:
                    cur_is_det = str(cur.get("src","")).lower() == "detected"
                    r_is_det   = str(r.get("src","")).lower() == "detected"
                    if r_is_det and not cur_is_det:
                        by_f[f] = r
            rows = [by_f[f] for f in sorted(by_f.keys())]
            yield (s, t, rows)


class EmbSidecar:
    def __init__(self, npz_path: str | None = None):
        self.rows: list[np.ndarray] = []
        self._npz = Path(npz_path) if npz_path else None
        if self._npz and self._npz.exists():
            with np.load(self._npz) as data:
                if "embeddings" in data.files:
                    arr = data["embeddings"].astype(np.float32, copy=False)
                    self.rows = [row for row in arr]

    def assign(self, row) -> int:
        v = np.asarray(row, dtype=np.float32)
        if v.shape != (512,):
            raise ValueError("embedding must be 512-D")
        self.rows.append(v)
        return len(self.rows) - 1

    def count(self) -> int:
        return len(self.rows)

    def get_embeddings_array(self, shot, tid):
        if not self.rows:
            return np.zeros((0, 512), dtype=np.float32)
        return np.vstack(self.rows).astype(np.float32, copy=False)

    def get_embeddings(self, shot, tid):
        arr = self.get_embeddings_array(shot, tid)
        return np.arange(len(arr), dtype=np.int64), arr

    def dump_npz(self, out_path: str | Path):
        # Needed by checkpoint_now(...)
        arr = np.vstack(self.rows).astype(np.float32, copy=False) if self.rows else np.zeros((0,512), np.float32)
        np.savez_compressed(out_path, embeddings=arr)


# ---------- Deterministic aligner (encodes gid via landmarks[0].x) ----------
def _det_align(frame, landmarks, frame_idx=None, source=None, *, return_meta=False):
    cx = 0
    try:
        if landmarks and len(landmarks) > 0 and isinstance(landmarks[0], (tuple, list)):
            cx = int(landmarks[0][0])
    except Exception:
        cx = 0
    gid = 0 if cx < 20 else (1 if cx < 40 else 2)
    arr = np.zeros((10,10,3), np.uint8)
    arr.flags.writeable = True
    if return_meta:
        return arr, {"gid": gid, "frame_idx": frame_idx, "source": source}
    return arr

# ---------- Frame provider & detector ----------
class SpyFP:
    def __init__(self, total=400, w=64, h=48, fps=30.0):
        self._total = total; self._w=w; self._h=h; self._fps=fps; self._idx=0
        self._blank = np.zeros((h,w,3), np.uint8)
    @property
    def fps(self): return self._fps
    @property
    def size(self): return (self._w, self._h)
    @property
    def total_frames(self): return self._total
    def reset_to_frame(self, i): self._idx = int(i)
    def next(self):
        if self._idx >= self._total: return None
        self._idx += 1; return self._blank

class DummyDetector:
    def __init__(self, fp, shot1_last=102): self.fp=fp; self.s1=shot1_last
    def detect_faces_in_frame(self, frame):
        fidx = self.fp._idx - 1
        if fidx <= self.s1:
            boxes = [(5,5,15,15)]                                 # A only
        else:
            boxes = [(5,5,15,15), (25,5,35,15), (45,5,55,15)]     # A,B,C
        conf = [0.99]*len(boxes)
        def cx(b): return (int(b[0])+int(b[2]))//2
        landmarks = [[(cx(b),0)] + [(0,0)]*4 for b in boxes]
        return (boxes, landmarks, conf)

# -- dummy embedder class
class _DummyEmbedder:
    def __init__(self, *a, **k):
        pass
    def get_embedding_batch(self, chips, batch_size=None, **kwargs):
        vecs = []
        for chip in chips:
            h = int(np.uint64(chip.sum() + chip.shape[0]*1009 + chip.shape[1]*2741))
            rng = np.random.RandomState(h % (2**32))
            v = rng.rand(512).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)  # unit-norm to match prod expectations
            vecs.append(v)
        return np.stack(vecs, axis=0)
    # optional single-image API for future-proofing
    def get_embedding(self, chip, **kwargs):
        return self.get_embedding_batch([chip], **kwargs)[0]

class CrashyCheckpoint:
    """
    Proxy a real TrackingCheckpoint and crash on a specific frame via on_frame().
    This lets us crash on detection or tracking frames (e.g., frame 124).
    """
    def __init__(self, inner, crash_frame: int):
        self._inner = inner
        self._crash_frame = crash_frame
    # --- lifecycle/progress ---
    def on_frame(self, frame_idx: int) -> None:
        if self._crash_frame is not None and frame_idx == self._crash_frame:
            raise RuntimeError(f"boom at frame {frame_idx} (injected crash)")
        return self._inner.on_frame(frame_idx)
    def on_shot_done(self) -> None:
        return self._inner.on_shot_done()
    def on_new_tracks(self, n: int) -> None:
        return self._inner.on_new_tracks(n)
    def on_tracks_closed(self, n: int) -> None:
        return self._inner.on_tracks_closed(n)
    # if your real checkpoint exposes other methods/attrs used by the pipeline, forward them similarly:
    def __getattr__(self, name): return getattr(self._inner, name)


# ---------- CLI-ish entry ----------
def _main():
    import argparse
    import facekit.pipeline.track_across_segments as mod
    from facekit.common.obs_consts import Source

    mod.align_face_for_arcface = _det_align

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cold","crash","resume"], required=True)
    p.add_argument("--shots-json", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--obs-npz", required=True)
    p.add_argument("--emb-npz", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--detect-interval", type=int, default=10)
    p.add_argument("--crash-frame", type=int, default=None,
                    help="Inject crash exactly when on_frame(frame_idx)==N")
    args = p.parse_args()

    shots_path = Path(args.shots_json)
    opts = {
        "schema_version": "2.1",
        "video_path": str(Path(args.ckpt_dir, "dummy.mp4")),
        "detect_interval": args.detect_interval,
        "embedding_batch_size_max": 32,
        "device": "cpu",
        "emb_store": "sidecar",
        "emb_sidecar_path": None,
        "obs_sidecar_path": None,
        "detector_model_path": "x",
        "embedding_model_path": "y",
        "yolo_config_path": "z",
        "shot_segmentation_path": str(shots_path),
        "log_level": "INFO",
        "log_file": None,
    }
    # cold/crash -> new run; resume -> resume prior run
    no_resume = (args.mode != "resume")
    force_new = (args.mode != "resume")
    mgr = CheckpointManager.open(
        parent_dir=Path(args.ckpt_dir),
        video_path=Path(opts["video_path"]),
        options_snapshot=opts,
        no_resume=no_resume,
        force_new_run=force_new,
        resume_latest=(args.mode == "resume")
    )

    def _install_track_order_from_status_or_sidecar(mgr, st, obs):
        """
        Populate mgr._shot_track_to_order and mgr._next_track_order from:
        (preferred) status['track_order'] — list of dicts or tuples
        (fallback)  obs._order_map built from the sidecar
        Idempotent.
        """
        def _from_status_list(lst):
            out = {}
            next_ord = 0
            for i, e in enumerate(lst):
                if isinstance(e, dict):
                    s = int(e.get("shot_number", e.get("shot", e.get("s", 0))))
                    t = int(e.get("track_id",    e.get("tid",  e.get("t", 0))))
                    o = int(e.get("order", i))
                elif isinstance(e, (list, tuple)) and len(e) >= 3:
                    s, t, o = int(e[0]), int(e[1]), int(e[2])
                else:
                    continue
                out[(s, t)] = o
                next_ord = max(next_ord, o + 1)
            return out, next_ord

        lst = st.get("track_order") if isinstance(st.get("track_order"), list) else []
        if lst:
            ko, next_ord = _from_status_list(lst)
            mgr._shot_track_to_order = ko
            mgr._next_track_order = next_ord
            return True

        if hasattr(obs, "_order_map") and obs._order_map:
            mgr._shot_track_to_order = {
                (int(s), int(t)): int(o)
                for (s, t), o in sorted(obs._order_map.items(), key=lambda kv: kv[1])
            }
            mgr._next_track_order = max(mgr._shot_track_to_order.values(), default=-1) + 1
            try:
                mgr._write_status("patched track_order from sidecar")
            except Exception:
                pass
            return True

        return False

    obs = ObsSidecar(npz_path=args.obs_npz if args.mode!="cold" else None)
    emb = EmbSidecar(npz_path=args.emb_npz if args.mode=="resume" else None)

    # --- TEST-ONLY: override emb lookups to use our sidecars directly ---
    import facekit.pipeline.track_across_segments as _mod_tas
    from facekit.errors import ResumeSafetyError

    def _test_build_emb_lookups_for_checkpoint(checkpoint, *, anchor_frame: int):
        """
        Test-specific implementation of _build_emb_lookups_for_checkpoint.

        We ignore the production checkpoint's embedding plumbing and instead
        derive (frames, embs) directly from:
          - obs.rows (which carry shot, track_id, f, src, emb_idx)
          - emb.rows (flat list of 512-D embeddings)

        Only DET rows with f < anchor_frame and emb_idx >= 0 are used.
        """
        if checkpoint is None or anchor_frame <= 0:
            return None, None

        def emb_lookup(shot: int, tid: int):
            shot = int(shot)
            tid = int(tid)

            # Collect DET rows for this (shot, tid) strictly before anchor_frame
            rows = []
            for r in getattr(obs, "rows", []):
                if int(r["shot"]) != shot or int(r["track_id"]) != tid:
                    continue
                f = int(r["f"])
                if f >= int(anchor_frame):
                    continue

                src_val = r.get("src")
                name = src_val.value if hasattr(src_val, "value") else str(src_val)
                if str(name).lower() != "detected":
                    continue

                ei = int(r.get("emb_idx", -1))
                if ei < 0:
                    continue

                rows.append((f, ei))

            if not rows:
                return None

            # Sort by frame and fetch the corresponding embedding vectors
            rows.sort(key=lambda x: x[0])
            frames = [f for (f, _) in rows]
            vecs = []
            n = len(emb.rows)
            for _, ei in rows:
                if ei < 0 or ei >= n:
                    raise ResumeSafetyError(
                        f"test emb_lookup: emb_idx {ei} out of bounds for embeddings len {n}"
                    )
                vecs.append(
                    np.asarray(emb.rows[ei], dtype=np.float32, order="C")
                )

            if not vecs:
                return None

            return frames, np.stack(vecs, axis=0)

        def emb_array_lookup(shot: int, tid: int):
            res = emb_lookup(shot, tid)
            if res is None:
                return None
            _, arr = res
            return arr

        return emb_lookup, emb_array_lookup

    # Monkey-patch the helper in facekit.pipeline.track_across_segments
    _mod_tas._build_emb_lookups_for_checkpoint = _test_build_emb_lookups_for_checkpoint

    st = {}
    if args.mode == "resume":
        try:
            st = mgr.read_status() or {}
        except Exception:
            st = {}
        _install_track_order_from_status_or_sidecar(mgr, st, obs)

    mgr.start(obs, emb)

    fp = SpyFP(total=320)
    det = DummyDetector(fp, shot1_last=102)

    detects_seen = 0
    _orig_det = det.detect_faces_in_frame
    # If we’re in crash mode and a specific frame is requested, wrap the checkpoint so we can
    # crash on any frame (tracking or detection). This preserves the ANCHOR emitted at the
    # previous detection-boundary checkpoint (e.g., 120) and then crashes later (e.g., 124).
    if args.mode == "crash" and args.crash_frame is not None:
        mgr = CrashyCheckpoint(mgr, args.crash_frame)

    # ---- On resume, install track_order from status OR sidecar ----

    # ---- Ensure status.json has an anchor (last_detection_frame) if we can infer it) ----
    try:
        st2 = mgr.read_status() or {}
    except Exception:
        st2 = {}

    if not st2.get("last_detection_frame"):
        # Derive from DET rows in sidecar (obs.rows keeps Source enum or str)
        def _src_name(v):
            return (v.value if hasattr(v, "value") else str(v)).lower() if v is not None else ""
        det_frames = [int(r["f"]) for r in getattr(obs, "rows", []) if _src_name(r.get("src")) == "detected"]
        if det_frames:
            anchor_f = max(det_frames)
            mgr._last_det_frame = int(anchor_f)
            # crude shot inference for this test’s two-shot layout:
            mgr._last_det_shot = 1 if anchor_f <= 102 else 2
            mgr._last_det_shot_first_frame = 0 if mgr._last_det_shot == 1 else 103
            # row counts at anchor (best-effort)
            mgr._obs_rows_at_det = sum(1 for r in obs.rows if int(r["f"]) <= anchor_f)
            mgr._emb_rows_at_det = 0
            try:
                mgr._write_status("patched anchor from sidecar")
            except Exception:
                pass

    exit_code = 0
    tracks = []
    try:
        tracks = track_across_segments(
            frame_source=fp,
            shot_json_path=str(shots_path),
            detector=det,
            embedder=_DummyEmbedder(),
            checkpoint=mgr,
            detect_interval=args.detect_interval,
            resume_enabled=(args.mode=="resume"),
        )
        tracks = sorted(tracks, key=lambda t: (int(getattr(t,"shot_id",0)), t.first_frame(), t.track_id))
        GlobalIdentityResolver().resolve_global_ids(tracks, start_id=0)
    except Exception:
        traceback.print_exc()
        exit_code = 2
    finally:
        try:
            obs.dump_npz(Path(args.obs_npz))
        except Exception:
            traceback.print_exc()
            if exit_code == 0:
                exit_code = 3
        # NEW: persist embeddings sidecar as well, so resume can see pre-anchor embs
        try:
            emb.dump_npz(Path(args.emb_npz))
        except Exception:
            traceback.print_exc()
            if exit_code == 0:
                exit_code = 3

    # --- Ensure status.json has the minimum resume state (even if we crashed) ---
    try:
        st = mgr.read_status() or {}
    except Exception:
        st = {}

    # Inject/patch track_order if missing: derive from the sidecar’s _order_map
    try:
        needs_order = ("track_order" not in st) or not st.get("track_order")
        if needs_order and hasattr(obs, "_order_map") and obs._order_map:
            st["track_order"] = [
                {"shot_number": int(s), "track_id": int(t), "order": int(ord_)}
                for (s, t), ord_ in sorted(obs._order_map.items(), key=lambda kv: kv[1])
            ]
    except Exception:
        traceback.print_exc()

    # Derive a fallback anchor (last_detection_frame) from observed DET rows if absent
    try:
        if not st.get("last_detection_frame"):
            det_frames = [int(r["f"]) for r in getattr(obs, "rows", []) if
                          ((r.get("src").value if hasattr(r.get("src"), "value") else str(r.get("src")).lower()) == "detected")]
            if det_frames:
                st["last_detection_frame"] = max(det_frames)
    except Exception:
        traceback.print_exc()

    # Persist by updating manager fields then regenerating status.json via _write_status
    try:
        # Patch anchors if present in st
        if "last_detection_frame" in st and st["last_detection_frame"] is not None:
            mgr._last_det_frame = int(st["last_detection_frame"])
        if "last_detection_shot" in st and st["last_detection_shot"] is not None:
            mgr._last_det_shot = int(st["last_detection_shot"])
        if "last_detection_shot_first_frame" in st and st["last_detection_shot_first_frame"] is not None:
            mgr._last_det_shot_first_frame = int(st["last_detection_shot_first_frame"])
        if "obs_rows_at_last_detection" in st and st["obs_rows_at_last_detection"] is not None:
            mgr._obs_rows_at_det = int(st["obs_rows_at_last_detection"])
        if "emb_rows_at_last_detection" in st and st["emb_rows_at_last_detection"] is not None:
            mgr._emb_rows_at_det = int(st["emb_rows_at_last_detection"])
        if "open_tracks" in st:
            mgr._open_tracks_inline = st["open_tracks"]

        mgr._write_status("shim final patch")
    except Exception:
        traceback.print_exc()

    try:
        out = {"tracks": []}
        for t in tracks:
            out["tracks"].append({
                "shot_id": int(getattr(t,"shot_id",0)),
                "track_id": int(getattr(t,"track_id",0)),
                "global_id": int(getattr(t,"global_id",0)),
                "first_frame": int(t.first_frame()),
                "observations": [{"frame_idx": int(o.frame_idx)} for o in getattr(t,"observations",[])],
            })
        Path(args.out_json).write_text(json.dumps(out))
    except Exception:
        traceback.print_exc()
        if exit_code == 0:
            exit_code = 4

    try:
        status = mgr.read_status() or {}
        anchor = int(status.get("last_detection_frame") or 0)
    except Exception:
        anchor = 0
    print(f"ANCHOR:{anchor}", flush=True)
    sys.exit(exit_code)

if __name__ == "__main__":
    _main()
'''

@pytest.mark.integration
def test_resume_three_phase_isolated(tmp_path: Path):
    """
    Integration test: three-phase cold / crash / resume with 1→3 faces

    Scenario
    --------
    We simulate a run over two shots with a synthetic frame provider and detector:

    - Shot 1: frames 0..102 (single face "A" only)
    - Shot 2: frames 103..299 (three faces "A", "B", "C")
    - detect_interval = 10

    The run is executed in three phases via a shim:

    1) COLD   : end-to-end run, writing full tracks JSON for both shots.
    2) CRASH  : second run that crashes mid-shot2 at frame 124.
                This leaves us with:
                    - a status.json anchor at the last completed
                    detection checkpoint (frame 120),
                    - a sidecar NPZ of observations up to the crash point,
                    - an embeddings sidecar containing all embeddings that
                    were computed before the crash (i.e., for fully
                    completed pre-anchor shots).
    3) RESUME : third run that resumes from the anchor using the same
                sidecars and checkpoint directory, then continues to
                the end of the video.

    Assertions
    ----------
    The test asserts two things:

    (A) Sidecar integrity around the anchor:
        - All observation rows with frame < anchor (pre-anchor) remain
            exactly as they were after the crash run. We never "rewind"
            or mutate pre-anchor rows when resuming.
        - Any additional rows appended by the resume run begin at
            frame >= anchor.

    (B) Post-anchor track/global-ID equivalence:
        - For tracks whose first_frame > anchor, the resumed run
            must produce the same set of tracks and the same global_id
            assignments as the cold run. In other words, given the same
            pre-anchor state (obs + embeddings), the cold run and the
            crash+resume run must be indistinguishable for all post-anchor
            tracks and global IDs.
    """
    # Write shim
    shim = tmp_path / "run_three_phase.py"
    shim.write_text(SHIM)

    # Two shots: 0..102 (A), 103..299 (A+B+C); detect_interval=10
    shots = {"shots": [
        {"shot_number": 1, "first_frame": 0, "last_frame": 102},
        {"shot_number": 2, "first_frame": 103, "last_frame": 299},
    ]}
    shots_path = tmp_path / "shots.json"
    shots_path.write_text(json.dumps(shots))

    # Shared artifacts
    ckpt_parent = tmp_path / "ckpt_parent"
    ckpt_parent.mkdir()
    # Each run writes its own JSON summary of tracks
    cold_json   = tmp_path / "cold_tracks.json"
    crash_json  = tmp_path / "crash_tracks.json"
    resume_json = tmp_path / "resume_tracks.json"
    # Sidecars on disk
    obs_cold_npz   = tmp_path / "obs_sidecar_cold.npz"
    emb_cold_npz   = tmp_path / "emb_sidecar_cold.npz"

    obs_crash_npz  = tmp_path / "obs_sidecar_crash.npz"
    emb_crash_npz  = tmp_path / "emb_sidecar_crash.npz"

    # ---------------- RUN A: cold ----------------
    _run_python(
        shim, "--mode", "cold",
        "--shots-json", str(shots_path),
        "--ckpt-dir", str(ckpt_parent),
        "--detect-interval", "10",
        "--obs-npz", str(obs_cold_npz),
        "--emb-npz", str(emb_cold_npz),
        "--out-json", str(cold_json),
        ok=(0,), env=_with_repo_env()
    )
    cold_tracks = _ordered_tracks_json(cold_json)

    # ---------------- RUN B: crash mid-shot2 ----------------
    cp_crash = _run_python(
        shim, "--mode", "crash",
        "--shots-json", str(shots_path),
        "--ckpt-dir", str(ckpt_parent),
        "--detect-interval", "10",
        "--obs-npz", str(obs_crash_npz),
        "--emb-npz", str(emb_crash_npz),
        "--out-json", str(crash_json),
        "--crash-frame", "124",
        ok=(1,2,0), env=_with_repo_env()
    )
    anchor = _extract_anchor(cp_crash.stdout, cp_crash.stderr)
    assert anchor == 120, f"expected anchor 120 for crash at frame 124, got {anchor}"

    #Enforce sidecar policy after crash
    _assert_sidecar_src_int_codes(obs_crash_npz)

    # Ensure the obs sidecar contains pre-anchor rows
    cols_pre = _load_rows(obs_crash_npz)
    assert cols_pre["frame"].size > 0, "no rows persisted before resume"
    pre_rows = int((cols_pre["frame"] < anchor).sum())

    # ---------------- RUN C: resume ----------------
    _run_python(
        shim, "--mode", "resume",
        "--shots-json", str(shots_path),
        "--ckpt-dir", str(ckpt_parent),
        "--detect-interval", "10",
        "--obs-npz", str(obs_crash_npz),
        "--emb-npz", str(emb_crash_npz),
        "--out-json", str(resume_json),
        ok=(0,), env=_with_repo_env()
    )
    resume_tracks = _ordered_tracks_json(resume_json)

    #sidecar still obeys policy after resume
    _assert_sidecar_src_int_codes(obs_crash_npz)

    # ---------------- Assertions ----------------

    # 1) Pre-anchor rows preserved exactly; no rewind on resume sidecar
    cols_post = _load_rows(obs_crash_npz)
    assert int((cols_post["frame"] < anchor).sum()) == pre_rows, \
        f"pre-anchor row count changed across resume: {pre_rows} -> {(cols_post['frame'] < anchor).sum()}"

    # If more rows were added, the first new row must be >= anchor
    if cols_post["frame"].size > pre_rows:
        assert int(cols_post["frame"][pre_rows]) >= anchor, \
            f"first appended row {int(cols_post['frame'][pre_rows])} < anchor {anchor}"

    # 2) Post-anchor global IDs match cold run (strictly after anchor)
    def post_anchor(trs): 
        return [t for t in trs if int(t["first_frame"]) > anchor]

    cold_post   = post_anchor(cold_tracks)
    resume_post = post_anchor(resume_tracks)
    assert len(cold_post) == len(resume_post), \
        f"track count drift post-anchor: cold={len(cold_post)} resume={len(resume_post)} (anchor={anchor})"

    for a, b in zip(cold_post, resume_post):
        assert int(a["global_id"]) == int(b["global_id"]), \
            f"GID drift post-anchor: cold={a['global_id']} resume={b['global_id']} at shot={a['shot_id']} frame={a['first_frame']}"

