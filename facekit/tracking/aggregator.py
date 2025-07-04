from typing import List, Dict, Optional
import numpy as np
from .face_tracks import FaceTrack, FaceObservation

class ShotFaceTrackAggregator:
    """
    Aggregates and manages face tracks for a single shot in a video.

    This class processes per-frame face observations, associating them with
    ongoing face tracks or initializing new ones when no matches are found.
    Matching is based on spatial (IoU) and optional embedding similarity.

    Attributes:
        shot_number (int): Identifier for the video shot.
        iou_threshold (float): Minimum IoU required to match observations to existing tracks.
        embedding_threshold (float): Maximum cosine distance (1 - similarity) for embedding match.
        tracks (List[FaceTrack]): List of all face tracks built for the shot.
        next_track_id (int): Counter for assigning new track IDs.
    """

    def __init__(
        self,
        shot_number: int,
        iou_threshold: float = 0.5,
        embedding_threshold: float = 0.6
    ):
        """
        Initialize the aggregator for a specific shot.

        Args:
            shot_number (int): Unique identifier for the shot.
            iou_threshold (float): IoU threshold to use for spatial matching.
            embedding_threshold (float): Cosine distance threshold for embedding similarity.
        """
        self.shot_number = shot_number
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.tracks: List[FaceTrack] = []
        self.next_track_id = 0

    def add_frame_observations(self, frame_idx: int, observations: List[FaceObservation]):
        """
        Assigns face observations from a single frame to existing tracks or creates new ones.

        Args:
            frame_idx (int): Frame index corresponding to the observations.
            observations (List[FaceObservation]): List of face observations in the frame.
        """
        for obs in observations:
            matched = False
            for track in self.tracks:
                if not track.is_active:
                    continue
                if self._is_match(track, obs):  # uses instance thresholds by default
                    track.add_observation(obs)
                    matched = True
                    break
            if not matched:
                new_track = FaceTrack(track_id=self.next_track_id, shot_id=self.shot_number)
                new_track.add_observation(obs)
                self.tracks.append(new_track)
                self.next_track_id += 1

    def finalize_tracks(self) -> List[FaceTrack]:
        """
        Returns all completed face tracks for the shot.

        Returns:
            List[FaceTrack]: The full list of face tracks built during aggregation.
        """
        return self.tracks

    def get_tracks_in_frame(self, frame_idx: int) -> List[FaceTrack]:
        """
        Retrieve all tracks that include a face observation in the specified frame.

        Args:
            frame_idx (int): Frame index to query.

        Returns:
            List[FaceTrack]: Tracks with an observation at the given frame.
        """
        return [track for track in self.tracks if frame_idx in track.get_frame_indices()]

    def get_shot_embeddings(self) -> Dict[int, Optional[np.ndarray]]:
        """
        Compute and return a representative embedding for each track in the shot.

        Returns:
            Dict[int, Optional[np.ndarray]]: Mapping of track ID to average embedding vector.
                                             If a track has no embeddings, the value is None.
        """
        return {
            track.track_id: track.compute_average_embedding()
            for track in self.tracks
            if track.observations
        }

    def _is_match(
        self,
        track: FaceTrack,
        obs: FaceObservation,
        iou_threshold: Optional[float] = None,
        embedding_threshold: Optional[float] = None
    ) -> bool:
        """
        Determines whether a new observation matches an existing track.

        This method builds a temporary one-observation track to compare against
        the target track using `FaceTrack.can_merge_with()` logic.

        Args:
            track (FaceTrack): The existing track to test for a match.
            obs (FaceObservation): The new observation to be matched.
            iou_threshold (Optional[float]): Override for IoU threshold (default: instance value).
            embedding_threshold (Optional[float]): Override for embedding threshold (default: instance value).

        Returns:
            bool: True if the observation is compatible with the track, False otherwise.
        """
        if not track.observations:
            return False
        temp = FaceTrack(track_id=-1, shot_id=-1, observations=[obs])
        return track.can_merge_with(
            temp,
            iou_thresh=iou_threshold if iou_threshold is not None else self.iou_threshold,
            embedding_thresh=embedding_threshold if embedding_threshold is not None else self.embedding_threshold
        )
