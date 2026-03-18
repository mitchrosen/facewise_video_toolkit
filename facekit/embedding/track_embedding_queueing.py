from facekit.common.obs_consts import Source
from facekit.embedding.embedding_sampling import should_sample_track_observation


def maybe_enqueue_track_observation_for_embedding(
    *,
    observation,
    track_local_index: int,
    track_sample_interval: int,
    frame,
    align_face_fn,
    embedding_queue,
) -> bool:
    if not should_sample_track_observation(
        track_local_index=track_local_index,
        is_detection_frame=(observation.source == Source.DETECTED),
        track_sample_interval=track_sample_interval,
    ):
        return False

    landmarks = getattr(observation, "landmarks", None)
    if landmarks is None:
        return False

    aligned_face = align_face_fn(frame, landmarks)
    if aligned_face is None:
        return False

    observation.aligned_face = aligned_face
    embedding_queue.enqueue(observation)
    return True