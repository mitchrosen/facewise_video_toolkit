from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

from facekit.tracking.face_tracks import FaceTrack, FaceObservation
from facekit.utils.geometry import compute_iou


class ShotFaceTrackAggregator:
    """
    Maintains and updates face tracks over a single video shot.
    Tracks are matched per-frame based on IoU and optionally embedding similarity.
    """

    def __init__(self, shot_number: int):
        self.shot_number = shot_number
        self.tracks: List[FaceTrack] = []
        self.next_track_id = 0

    def add_frame_observations(self, frame_idx: int, observations: List[FaceObservation]):
        """
        Assign incoming frame observations to existing tracks, or create new ones.

        Args:
            frame_idx (int): Frame number (included in FaceObservation, redundant).
            observations (List[FaceObservation]): Face detections in this frame.
        """
        for obs in observations:
            matched = False
            for track in self.tracks:
                if not track.is_active:
                    continue
                if self._is_match(track, obs):
                    track.add_observation(obs)
                    matched = True
                    break
            if not matched:
                new_track = FaceTrack(track_id=self.next_track_id)
                new_track.add_observation(obs)
                self.tracks.append(new_track)
                self.next_track_id += 1

    def finalize_tracks(self) -> List[FaceTrack]:
        """
        Finalize and return all tracks created during this shot.
        """
        return self.tracks

    def get_shot_embeddings(self) -> Dict[int, Optional[np.ndarray]]:
        """
        Return representative average embeddings for each completed track.
        """
        return {
            track.track_id: track.compute_average_embedding()
            for track in self.tracks
            if track.observations
        }

    def get_tracks_in_frame(self, frame_idx: int) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """
        Return (track_id, bbox) pairs for all tracks with observations in the given frame.

        Args:
            frame_idx (int): Frame index to query.

        Returns:
            List[Tuple[int, bbox]]: List of (track_id, bbox) pairs in that frame.
        """
        result = []
        for track in self.tracks:
            obs = track._frame_index_map.get(frame_idx)
            if obs:
                result.append((track.track_id, obs.bbox))
        return result

    def _is_match(self, track: FaceTrack, obs: FaceObservation, iou_threshold: float = 0.5) -> bool:
        """
        Determine whether a track and a new observation match based on IoU only.

        You may substitute this with track.can_merge_with() if embeddings should be considered.
        """
        last_bbox = track.get_last_bbox()
        if last_bbox is None or obs.bbox is None:
            return False
        return compute_iou(last_bbox, obs.bbox) > iou_threshold
