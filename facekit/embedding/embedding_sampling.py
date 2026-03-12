"""Helpers for deciding when to capture embedding samples for a track."""


def should_sample_track_observation(
    *,
    track_local_index: int,
    is_detection_frame: bool,
    track_sample_interval: int,
) -> bool:
    """Return True when a track observation should be sampled for embedding.

    Rules:
    - detection frames are always sampled
    - otherwise, sample every N track-local frames
    - non-detection frames are never sampled when interval <= 0
    """
    if is_detection_frame:
        return True

    if track_sample_interval <= 0:
        return False

    return track_local_index % track_sample_interval == 0