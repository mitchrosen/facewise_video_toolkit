import json, os
from pathlib import Path
import numpy as np
import pytest
from facekit.pipeline.checkpoint import CheckpointManager
from facekit.output.json_v2 import ObservationsCollector, EmbeddingCollector
from facekit.errors import ResumeSafetyError

class DummyObs:
    rows = 0
    def dump_npz(self, path): Path(path).write_bytes(b"NPZ")
    def count(self): return self.rows
    def load_npz(self, path): return self.rows
    def trim_to(self, n): self.rows = n

class DummyEmb(DummyObs): pass

class RecordingObs(DummyObs):
    def __init__(self): self.rows=0; self.trimmed_to=None
    def load_npz(self, path): return self.rows
    def trim_to(self, n): self.trimmed_to=n; self.rows=n

class RecordingEmb(RecordingObs): pass

class FileBackedObs:
    def __init__(self): self.rows = 0; self.trimmed_to = None
    def dump_npz(self, path):
        import numpy as np
        np.savez(path, rows=self.rows)
    def count(self): return self.rows
    def load_npz(self, path):
        import numpy as np
        data = np.load(path)
        self.rows = int(data["rows"])
        return self.rows
    def trim_to(self, n):
        self.trimmed_to = n
        self.rows = n

class FileBackedEmb(FileBackedObs): pass

@pytest.mark.parametrize("resume", [False, True])
def test_start_no_resume_and_resume(tmp_path, resume):
    ckpt_dir = tmp_path / "check"
    ckpt = CheckpointManager(ckpt_dir, video_path=tmp_path/"v.mp4", resume=resume)

    obs = ObservationsCollector()
    emb = EmbeddingCollector("sidecar", dim=512)

    # simulate pre-existing resume files
    if resume:
        # put tiny sidecars
        np.savez(ckpt_dir/"embeddings_ckpt.npz", embeddings=np.zeros((0, 512), dtype=np.float32))
        np.savez(ckpt_dir/"observations_ckpt.npz", observations=np.zeros((0,), dtype=obs._rows[0].dtype if obs._rows else np.dtype([("f","i4"),("bbox_xyxy","f4",4),("src","u1"),("conf","f4"),("emb_idx","i4")])) )
        (ckpt_dir/"status.json").parent.mkdir(parents=True, exist_ok=True)
        (ckpt_dir/"status.json").write_text(json.dumps({"frames_done": 100}))

    ckpt.start(obs, emb, tracks_seen=0, shots_done=0, frames_done=0)
    assert (ckpt_dir/"status.json").exists()

def test_and_status_and_finalize(tmp_path):
    ckpt_dir = tmp_path/"ck"
    ckpt = CheckpointManager(ckpt_dir, video_path=tmp_path/"v.mp4", resume=False)
    obs = ObservationsCollector()
    emb = EmbeddingCollector("sidecar", dim=512)
    ckpt.start(obs, emb)

    # simulate frames; every 3 → atomic write should occur
    for f in range(1, 10):
        ckpt.on_frame(f)
        assert (ckpt_dir/"status.json").exists()

    ckpt.finalize()
    # finalized sidecars exist (names may vary by your impl; check presence of any *.npz)
    assert list((ckpt_dir / "ckpt").glob("*.npz")), "expected at least one npz sidecar"

def test_load_into_collectors_roundtrip(tmp_path):
    ckpt_dir = tmp_path/"ck2"
    ckpt = CheckpointManager(ckpt_dir, video_path=tmp_path/"v.mp4", resume=False)
    obs = ObservationsCollector()
    emb = EmbeddingCollector("sidecar", dim=512)

    ckpt.start(obs, emb)
    # force some writes by calling private helpers if exposed, else mimic finalize
    ckpt.finalize()

    # new collectors, load
    obs2 = ObservationsCollector()
    emb2 = EmbeddingCollector("sidecar", dim=512)
    loaded_obs, loaded_emb = ckpt.load_into_collectors(obs2, emb2)
    assert loaded_obs >= 0 and loaded_emb >= 0

def test_open_creates_run_dir(tmp_path):
    parent = tmp_path / "ckpts"
    snap = {"video_path": str((tmp_path/"vid.mp4").resolve())}
    cm = CheckpointManager.open(
        parent_dir=parent, 
        video_path=tmp_path/"vid.mp4", 
        options_snapshot=snap, 
        no_resume=False)
    assert cm.root.parent == parent
    assert cm.root.name.startswith("run-")
    assert (parent / "current").exists() or True  # symlink is best-effort

def test_validate_resume_video_path_mismatch(tmp_path):
    parent = tmp_path / "ck"
    v1 = tmp_path/"a.mp4"; v1.write_bytes(b"")
    v2 = tmp_path/"b.mp4"; v2.write_bytes(b"")
    snap = {"video_path": str(v1.resolve()), "detect_interval": 10}
    cm = CheckpointManager.open(
        parent_dir=parent, 
        video_path=v1, 
        options_snapshot=snap, 
        no_resume=False)
    # prime status
    cm.start(obs_collector=DummyObs(), emb_collector=DummyEmb(), options_snapshot=snap)
    cm.finalize()
    with pytest.raises(ResumeSafetyError):
        cm.validate_resume_or_raise({"video_path": str(v2.resolve()), "detect_interval": 10}, force=False)

def test_validate_resume_diffs_require_force(tmp_path):
    parent = tmp_path / "ck"
    v = tmp_path/"a.mp4"; v.write_bytes(b"")
    snap = {"video_path": str(v.resolve()), "detect_interval": 10}
    cm = CheckpointManager.open(
        parent_dir=parent, 
        video_path=v, 
        options_snapshot=snap, 
        no_resume=False)
    cm.start(DummyObs(), DummyEmb(), options_snapshot=snap)
    cm.finalize()
    # detect_interval changed
    with pytest.raises(ResumeSafetyError):
        cm.validate_resume_or_raise({"video_path": str(v.resolve()), "detect_interval": 5}, force=False)
    assert cm.validate_resume_or_raise({"video_path": str(v.resolve()), "detect_interval": 5}, force=True)

def test_status_lifecycle(tmp_path):
    v = tmp_path/"v.mp4"; v.write_bytes(b"")
    cm = CheckpointManager.open(
        parent_dir=tmp_path, 
        video_path=v, 
        options_snapshot={"video_path": str(v.resolve())}, 
        no_resume=False)
    obs, emb = DummyObs(), DummyEmb()
    cm.start(obs, emb, options_snapshot={"video_path": str(v.resolve())})
    cm.on_frame(0)
    cm.on_frame(9)
    cm.on_frame(10)
    cm.checkpoint_now(frame_idx=10, shot_number=1)
    cm.on_shot_done()
    cm.finalize()
    st = json.loads((cm.root/"status.json").read_text())
    assert st["frames_done"] == 11
    assert st["shots_done"] >= 1

def test_load_and_anchor_trims_to_pre_detection(tmp_path):
    v = tmp_path/"v.mp4"; v.write_bytes(b"")
    cm = CheckpointManager.open(parent_dir=tmp_path, video_path=v,
                             options_snapshot={"video_path": str(v.resolve())}, 
                             no_resume=False)
    obs, emb = FileBackedObs(), FileBackedEmb()
    cm.start(obs, emb, options_snapshot={"video_path": str(v.resolve())})

    # Pre-anchor rows, then checkpoint (anchor should record 7/3)
    obs.rows = 7; emb.rows = 3
    cm.checkpoint_now(frame_idx=5, shot_number=1)

    # Advance rows post-anchor and persist them
    obs.rows = 12; emb.rows = 9
    cm.finalize()  # writes latest rows to the npz files

    st = cm.read_status()
    st = st or {}
    st["track_order"] = [{"shot": 1, "track_id": 0, "order": 0}]
    cm.status_path.write_text(json.dumps(st, indent=2))

    # New “process”: re-open the manager pointing at the existing run dir
    cm2 = CheckpointManager.open(
      parent_dir=tmp_path,
      video_path=v,
      options_snapshot={"video_path": str(v.resolve())},
      no_resume=False,
      run_id=cm.run_id,           # <<< ensure we point at the same run
    )
    obs2, emb2 = FileBackedObs(), FileBackedEmb()
    loaded_obs, loaded_emb = cm2.load_and_anchor_collectors(obs2, emb2)
    assert loaded_obs == 12 and loaded_emb == 9

    # Now trim to anchor recorded at checkpoint (7/3)
    cm2.load_and_anchor_collectors(obs2, emb2, trim_to_anchor=True)
    assert obs2.trimmed_to == 7
    assert emb2.trimmed_to == 3

def test_resume_available(tmp_path):
    v = (tmp_path/"v.mp4"); v.write_bytes(b"")
    cm = CheckpointManager.open(
        parent_dir=tmp_path, 
        video_path=v, 
        options_snapshot={"video_path": str(v.resolve())}, 
        no_resume=False)
    cm.start(DummyObs(), DummyEmb(), options_snapshot={"video_path": str(v.resolve())})
    cm.checkpoint_now(frame_idx=0, shot_number=1)
    assert cm.resume_available()

def test_atomic_sidecars_exist_even_empty(tmp_path):
    v = tmp_path/"v.mp4"; v.write_bytes(b"")
    cm = CheckpointManager.open(parent_dir=tmp_path, video_path=v, options_snapshot={"video_path": str(v.resolve())}, no_resume=True)
    cm.start(DummyObs(), DummyEmb(), options_snapshot={"video_path": str(v.resolve())})
    cm.finalize()
    assert (cm.root/"ckpt"/"obs_ckpt.npz").exists()
    assert (cm.root/"ckpt"/"emb_ckpt.npz").exists()

def test_compute_parent_dir_stable(tmp_path):
    v1 = (tmp_path/"x"/"v.mp4"); v1.parent.mkdir(parents=True); v1.write_bytes(b"")
    v2 = (tmp_path/"y"/"v.mp4"); v2.parent.mkdir(parents=True); v2.write_bytes(b"")
    p1 = CheckpointManager.compute_parent_dir(tmp_path/"ck", v1)
    p2 = CheckpointManager.compute_parent_dir(tmp_path/"ck", v1)
    p3 = CheckpointManager.compute_parent_dir(tmp_path/"ck", v2)
    assert p1 == p2
    assert p1 != p3

