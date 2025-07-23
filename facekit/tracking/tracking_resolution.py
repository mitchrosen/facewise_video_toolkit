import numpy as np
from typing import List
from facekit.tracking.face_structures import FaceTrack
import torch

class GlobalIdentityResolver:
    def __init__(self, embedding_threshold: float = 0.7):
        """
        Resolves global IDs for FaceTracks using clustering based on embedding similarity.
        
        Args:
            embedding_threshold (float): Cosine similarity threshold for linking tracks.
        """
        self.embedding_threshold = embedding_threshold

    def resolve_global_ids(self, tracks: List[FaceTrack], start_id: int = 0) -> int:
        """
        Assign global_ids to FaceTracks by clustering based on embedding similarity.

        Uses a graph-based approach where edges are formed between tracks whose
        average embeddings exceed the similarity threshold (with floating-point tolerance).
        Connected components represent clusters of tracks that belong to the same global identity.

        Args:
            tracks (List[FaceTrack]): All FaceTracks (possibly across shots).
            start_id (int): Starting global_id counter.

        Returns:
            int: The next available global_id after assignment.
        """
        print(f"[DEBUG] Starting global resolution: {len(tracks)} tracks")

        # Normalize embeddings and store valid ones
        valid_tracks = []
        for i, track in enumerate(tracks):
            avg_emb = track.compute_average_embedding()
            print(f"[DEBUG] Track {track.track_id}@Shot{track.shot_id} "
                  f"| embeddings={len(track.embeddings)} | has_embedding={track.has_embedding()}")
            if avg_emb is not None:
                norm = np.linalg.norm(avg_emb)
                if norm > 0:
                    valid_tracks.append((i, avg_emb / norm))

        if not valid_tracks:
            print("[DEBUG] No valid embeddings found; assigning unique IDs")
            for track in tracks:
                track.global_id = start_id
                print(f"[DEBUG] Track {track.track_id}@Shot{track.shot_id} → global_id {start_id}")
                start_id += 1
            return start_id

        # Prepare adjacency graph
        n = len(tracks)
        adjacency = [[] for _ in range(n)]
        print(f"[DEBUG] Building adjacency list with threshold {self.embedding_threshold}")

        # ✅ GPU-optimized similarity computation for large sets
        indices = [idx for idx, _ in valid_tracks]
        embeddings = np.stack([emb for _, emb in valid_tracks])  # shape: (k, dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)

        # Compute similarity matrix using PyTorch for speed
        with torch.no_grad():
            sim_matrix = torch.matmul(emb_tensor, emb_tensor.T).cpu().numpy()

        # ✅ Build adjacency using similarity threshold + floating-point tolerance
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sim = sim_matrix[i, j]
                if sim >= self.embedding_threshold or np.isclose(sim, self.embedding_threshold, atol=1e-6):
                    adjacency[indices[i]].append(indices[j])
                    adjacency[indices[j]].append(indices[i])
                    print(f"[DEBUG] Link T{indices[i]} <-> T{indices[j]} sim={sim:.4f}")

        # DFS to find connected components (clusters)
        visited = [False] * n
        next_global_id = start_id

        def dfs(node, comp):
            visited[node] = True
            comp.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    dfs(neighbor, comp)

        for i in range(n):
            if visited[i]:
                continue
            component = []
            dfs(i, component)
            if not component:
                continue

            print(f"[DEBUG] Starting DFS for new component at track {i}")
            print(f"[DEBUG] Component nodes: {component}")

            # Assign same global_id to all tracks in this component
            for idx in component:
                tracks[idx].global_id = next_global_id
            print(f"[DEBUG] Assigned global_id {next_global_id} to cluster {component}")
            next_global_id += 1

        # Assign new IDs to tracks without embeddings (or completely isolated)
        for track in tracks:
            if track.global_id is None:
                track.global_id = next_global_id
                print(f"[DEBUG] No embedding match → Track {track.track_id}@Shot{track.shot_id} "
                      f"gets global_id {next_global_id}")
                next_global_id += 1

        print("[DEBUG] Final assignments:")
        for t in tracks:
            print(f"    Track {t.track_id}@Shot{t.shot_id} → global_id {t.global_id}")

        print(f"[DEBUG] Finished global resolution. Next available global_id={next_global_id}")
        return next_global_id
