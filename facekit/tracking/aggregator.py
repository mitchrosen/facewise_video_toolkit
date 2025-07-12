from typing import List, Dict, Optional
import numpy as np
from .face_structures import FaceTrack, FaceObservation

class ShotFaceTrackAggregator:
    """
    Aggregates and manages face tracks for a single shot in a video.

    This class processes per-frame face observations, associating them with
    ongoing face tracks or initializing new ones when no matches are found.
    Matching is based on spatial (IoU) and optional embedding distance (1 - similarity).

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
            embedding_threshold (float): Maximum cosine distance (1 - similarity) for embedding match.
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
    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        return 1 - np.dot(a, b)

    def resolve_global_ids(
        self,
        prior_tracks: List[FaceTrack],
        global_id_counter: int,
        embedding_threshold: float = 0.6
    ) -> int:
        """
        Assigns global IDs to current shot's tracks using a 3-pass strategy:
        1. Match against active tracks in this shot (IoU + embedding)
        2. Match against inactive tracks in this shot (embedding only)
        3. Match against prior shots' tracks (embedding only)
        4. Assign new global ID if no match

        Modifies each track in-place to set .global_id
        Returns the updated global_id_counter
        """
        for track in self.tracks:
            if track.global_id is not None:
                continue  # Do not overwrite existing global_id!
            
            if not track.has_embedding():
                track.global_id = global_id_counter
                global_id_counter += 1
                continue

            best_match = None
            best_score = float("inf")

            # Pass 1: match active tracks in this shot (IoU + embedding)
            for candidate in self.tracks:
                if candidate is track or not candidate.is_active:
                    continue
                if candidate.global_id is None:
                    continue
                if not candidate.has_embedding():
                    continue
                if track.can_merge_with(candidate, iou_thresh=self.iou_threshold, embedding_thresh=embedding_threshold):
                    best_match = candidate
                    break  # Prefer exact match and stop early

            # Pass 2: match inactive tracks in this shot (embedding only)
            if best_match is None:
                for candidate in self.tracks:
                    if candidate is track or candidate.is_active:
                        continue
                    if candidate.global_id is None:
                        continue
                    if not candidate.has_embedding():
                        continue

                    # sim = self.cosine_distance(track.compute_average_embedding(), 
                    #                       candidate.compute_average_embedding())

                    emb1 = track.compute_average_embedding()
                    emb2 = candidate.compute_average_embedding()
                    sim = 1 - np.dot(emb1 / np.linalg.norm(emb1), emb2 / np.linalg.norm(emb2))

                    if sim < embedding_threshold and sim < best_score:
                        best_score = sim
                        best_match = candidate

            # Pass 3: match against global prior tracks (embedding only)
            if best_match is None:
                for prior in prior_tracks:
                    if not prior.has_embedding():
                        continue
                    sim = self.cosine_distance(track.compute_average_embedding(), 
                                          prior.compute_average_embedding())

                    if sim < embedding_threshold and sim < best_score:
                        best_score = sim
                        best_match = prior

            # Assign result
            if best_match:
                track.global_id = best_match.global_id
            else:
                track.global_id = global_id_counter
                global_id_counter += 1

        return global_id_counter

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
