import numpy as np
from typing import List
from facekit.tracking.face_structures import FaceTrack

class GlobalIdentityResolver:
    def __init__(self, embedding_threshold: float = 0.6):
        self.embedding_threshold = embedding_threshold

    def resolve(self, tracks: List[FaceTrack], starting_global_id: int = 0) -> int:
        """
        Assign global_ids to FaceTracks based on embedding similarity.

        Args:
            tracks (List[FaceTrack]): All FaceTracks with vchunk_id assigned.
            starting_global_id (int): Initial value for global_id counter.

        Returns:
            int: The next available global_id after assignment.
        """
        # Filter tracks with usable embeddings
        track_embeddings = []
        for i, track in enumerate(tracks):
            avg_emb = track.compute_average_embedding()
            if avg_emb is not None:
                norm = np.linalg.norm(avg_emb)
                if norm > 0:
                    track_embeddings.append((i, avg_emb / norm))

        # No valid embeddings to match
        if not track_embeddings:
            for track in tracks:
                track.global_id = starting_global_id
                starting_global_id += 1
            return starting_global_id

        # Initialize each track's global_id to None
        for track in tracks:
            track.global_id = None

        next_global_id = starting_global_id

        # Greedy grouping of similar tracks
        for i, emb_i in track_embeddings:
            track_i = tracks[i]
            if track_i.global_id is not None:
                continue  # already assigned

            # Start new group
            track_i.global_id = next_global_id
            for j, emb_j in track_embeddings:
                if j == i:
                    continue
                track_j = tracks[j]
                if track_j.global_id is not None:
                    continue
                similarity = np.dot(emb_i, emb_j)
                if similarity >= (1 - self.embedding_threshold):
                    track_j.global_id = next_global_id
            next_global_id += 1

        # Assign global_id to any tracks without embeddings
        for track in tracks:
            if track.global_id is None:
                track.global_id = next_global_id
                next_global_id += 1

        return next_global_id
