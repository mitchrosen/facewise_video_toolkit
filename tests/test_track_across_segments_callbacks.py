import pytest
from facekit.pipeline.track_across_segments import track_across_segments
from typing import List
import numpy as np
from facekit.pipeline.checkpoint import TrackingCheckpoint
from facekit.tracking.aggregator import ShotFaceTrackAggregatorProtocol

def test_callbacks_invoked(monkeypatch):
    # stub detector/embedder
    class _Det:
        def detect_faces_in_frame(self, frame, target_size=640):
            return ([], [], [])

    class _Emb:
        def set_max_batch_size(self, n): pass
        def get_embedding_batch(self, aligned, batch_size=32):
            return np.zeros((len(aligned), 512), dtype=np.float32)

    # fake FrameProvider
    class FP:
        def __init__(self):
            self._i = 0
        # sequential API
        def reset_to_frame(self, start_idx: int) -> None:
            self._i = int(start_idx)

        def next(self):
            ok, frame = self.get(self._i)
            self._i += 1
            return frame if ok else None
        # random access API
        def get(self, idx):
            return True, b"\x00" * (320 * 240 * 3)
        # required lifecycle
        def close(self) -> None:
            pass
        # metadata API
        def fps(self): return 30.0
        def size(self): return (320, 240)
        def total_frames(self): return 30

    # Fake TrackingCheckpoint
    class TrackingCheckpointStub(TrackingCheckpoint):
        def __init__(self):
            self.frames: List[int] = []
            self.checkpoints: List[tuple[int,int]] = []  # (frame_idx, shot_number)
            self.new_tracks_total = 0
            self.shots_done = 0
            self.obs_calls: List[tuple[int,int,int]] = []  # (shot, frame, count)
            self.emb_calls: List[tuple[int,int,int]] = []  # (shot, track_id, count)
            self.status_path = "/dev/null"
            self.finalized_notes: List[str] = []
            self.write_disabled = False
            self.resume_enabled = False


        # called every processed frame
        def on_frame(self, frame_idx: int) -> None:
            self.frames.append(int(frame_idx))

        # called when N new tracks are created on a frame
        def on_new_tracks(self, n: int) -> None:
            self.new_tracks_total += int(n)

        # called right before a detection (anchor)
        def checkpoint_now(
                self, 
                *, 
                frame_idx: int, 
                shot_number: int, 
                aggregator: ShotFaceTrackAggregatorProtocol,
                shot_first_frame: int | None = None,
                note: str = "checkpoint") -> None:
            self.checkpoints.append((int(frame_idx), int(shot_number)))

        # called at end of shot
        def on_shot_done(self) -> None:
            self.shots_done += 1

        # persistence helpers (no-ops for this test)
        def add_observations(self, shot_number, frame_idx, det_obs) -> None:
            self.obs_calls.append((int(shot_number), int(frame_idx), len(det_obs)))
        
        def add_embeddings(self, shot_number, track_id, frame_idx, embs) -> None:
            self.emb_calls.append((int(shot_number), int(track_id), int(getattr(embs, "shape", (0,0))[0])))
        
        def get_pending_detection_cursor(self):
            # (shot_number, frame_idx, shot_first_frame, reason)
            return (None, None, None, None)
        def get_track_order(self) -> dict[tuple[int, int], int]:
            return getattr(self, "_track_order", {})
        
        def finalize(self, note: str = "final") -> None:
            self.finalized_notes.append(str(note))
        
        def mark_completed(self, note: str | None = None) -> None:
            """
            track_across_segments calls checkpoint.mark_completed() at the end.
            For this stub we just record that it happened.
            """
            # keep behavior simple: mirror finalize-style recording
            self.finalized_notes.append(str(note) if note is not None else "completed")
            
    # a minimal shot json file
    import json, tempfile, pathlib
    shots = {"shots":[{"shot_number":1,"first_frame":0,"last_frame":9}]}
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
        pathlib.Path(tmp.name).write_text(json.dumps(shots))
        shot_json = tmp.name

    frames = []
    tracks_started = []
    def on_frame(i): frames.append(i)
    def on_new(): tracks_started.append(1)

    cp = TrackingCheckpointStub()

    tracks = track_across_segments(
        frame_source=FP(),
        shot_json_path=shot_json,
        detector=_Det(),
        embedder=_Emb(),
        detect_interval=3,
        embedding_batch_size_max=16,
        checkpoint = cp,
    )
    # --- assertions -------------------------------------------------------------
    # processed exactly the frames in the shot
    assert cp.frames[0] == 0
    assert cp.frames[-1] == 9
    assert len(cp.frames) == 10

    # end-of-shot callback fired
    assert cp.shots_done == 1

    # run finalization/completion callback fired
    assert cp.finalized_notes
    assert cp.finalized_notes[-1] == "completed"

    # with empty detections, no tracks are created
    assert cp.new_tracks_total == 0
    assert tracks == []
