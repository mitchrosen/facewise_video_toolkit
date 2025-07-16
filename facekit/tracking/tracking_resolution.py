import numpy as np
from typing import List
from facekit.tracking.face_structures import FaceTrack

class GlobalIdentityResolver:
    def __init__(self, embedding_threshold: float = 0.7):
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
        print(f"[DEBUG] Starting global resolution: {len(tracks)} tracks")

        # Filter tracks with usable embeddings
        track_embeddings = []
        for i, track in enumerate(tracks):
            print(f"[DEBUG] Track {track.track_id}@Shot{track.shot_id}#Vchunk{track.vchunk_id} "
                f"| embeddings={len(track.embeddings)} | has_embedding={track.has_embedding()}")
            avg_emb = track.compute_average_embedding()
            if avg_emb is not None:
                norm = np.linalg.norm(avg_emb)
                if norm > 0:
                    track_embeddings.append((i, avg_emb / norm))

        # No valid embeddings to match
        if not track_embeddings:
            print("[DEBUG] No embeddings found, assigning unique global IDs to all tracks")
            for track in tracks:
                track.global_id = starting_global_id
                print(f"[DEBUG] Track {track.track_id}@Shot{track.shot_id} → global_id {starting_global_id}")
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
            print(f"[DEBUG] Starting new group with Track {track_i.track_id}@Shot{track_i.shot_id} "
                f"→ global_id {next_global_id}")

            for j, emb_j in track_embeddings:
                if j == i:
                    continue
                track_j = tracks[j]
                if track_j.global_id is not None:
                    continue

                similarity = np.dot(emb_i, emb_j)
                print(f"[DEBUG] Compare T{track_i.track_id}@S{track_i.shot_id} vs "
                    f"T{track_j.track_id}@S{track_j.shot_id} → sim={similarity:.4f}")

                # ✅ Use your threshold logic: same face if similarity > 0.06
                if similarity > 0.06:
                    track_j.global_id = next_global_id
                    print(f"[DEBUG] Assigned Track {track_j.track_id}@Shot{track_j.shot_id} "
                        f"to global_id {next_global_id}")

            next_global_id += 1

        # Assign global_id to any tracks without embeddings
        for track in tracks:
            if track.global_id is None:
                track.global_id = next_global_id
                print(f"[DEBUG] No embedding match → Track {track.track_id}@Shot{track.shot_id} "
                    f"gets new global_id {next_global_id}")
                next_global_id += 1

        print(f"[DEBUG] Finished global resolution. Next available global_id={next_global_id}")
        return next_global_id
