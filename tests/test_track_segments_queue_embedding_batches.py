import numpy as np

from facekit.pipeline.track_across_segments import track_across_segments
from facekit.common.obs_consts import Source


class TinyFrameProvider:
    def __init__(self, frames):
        self._frames = list(frames)
        self.fps = 30.0
        self.size = (self._frames[0].shape[1], self._frames[0].shape[0])
        self.total_frames = len(self._frames)
        self._i = 0

    def reset_to_frame(self, i: int):
        self._i = int(i)

    def next(self):
        if self._i < 0 or self._i >= len(self._frames):
            return None
        f = self._frames[self._i]
        self._i += 1
        return f


class FakeDetectorEveryFrame:
    def detect_faces_in_frame(self, frame):
        boxes = [(10, 10, 50, 50)]
        landmarks = [[(1.0, 2.0)] * 5]
        confs = [0.9]
        return boxes, landmarks, confs


class RecordingEmbedder:
    def __init__(self):
        self.calls = []

    def get_embedding_batch(self, aligned_faces, batch_size=32):
        self.calls.append(len(aligned_faces))
        n = len(aligned_faces)
        # Return correct dtype/shape
        return np.zeros((n, 512), dtype=np.float32)


def test_track_across_segments_embeds_from_cached_aligned_faces_in_bounded_batches(tmp_path, monkeypatch):
    """
    Integration contract:
    - aligned_face is computed on detection frames and cached
    - embedding is produced without rereading frames later
    - batching follows queue semantics (max_pending=3 => calls [3,2] for 5 items)
    - aligned_face is cleared after embedding
    """
    import facekit.pipeline.track_across_segments as tas

    # Patch align_face_for_arcface to produce a unique aligned face per frame
    def fake_align(frame, landmarks, *args, **kwargs):
        # encode frame identity using first pixel to help debugging if needed
        out = np.zeros((112, 112, 3), dtype=np.uint8)
        out[0, 0, 0] = int(frame[0, 0, 0])
        return out

    monkeypatch.setattr(tas, "align_face_for_arcface", fake_align)

    # Patch queue max_pending to 3 by monkeypatching the class default ctor usage later.
    # This test assumes you'll instantiate AlignedFaceEmbeddingQueue(max_pending=3) in the shot loop.
    # If you expose it as a parameter later, update the test accordingly.

    # 5 frames => expect [3,2] batches if you maybe_flush each iteration and final flush at shot end
    frames = []
    for i in range(5):
        fr = np.zeros((64, 64, 3), dtype=np.uint8)
        fr[0, 0, 0] = i
        frames.append(fr)

    fp = TinyFrameProvider(frames)

    shot_json = tmp_path / "shots.json"
    shot_json.write_text(
        '{"shots":[{"shot_number":1,"first_frame":0,"last_frame":4}]}',
        encoding="utf-8",
    )

    embedder = RecordingEmbedder()

    tracks = track_across_segments(
        fp,
        str(shot_json),
        detector=FakeDetectorEveryFrame(),
        embedder=embedder,
        detect_interval=1,  # detection every frame 
        checkpoint=None,
        resume_enabled=False,
        embedding_batch_size_max=32,
        embedding_queue_max_pending=3,  # force mid-shot flush at 3 => calls [3,2]
    )

    # Once you integrate the queue with max_pending=3, this should be [3,2].
    assert embedder.calls == [3, 2]

    # Verify aligned_face cleared after embedding (to keep memory bounded)
    for t in tracks:
        for ob in getattr(t, "observations", []) or []:
            if ob.source == Source.DETECTED and getattr(ob, "embedding", None) is not None:
                assert getattr(ob, "aligned_face", None) is None
