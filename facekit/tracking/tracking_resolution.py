import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from facekit.tracking.face_structures import FaceTrack


class GlobalIdentityResolver:
    def __init__(self, embedding_threshold: float = 0.7, device: str = "auto"):
        """
        Resolves global IDs for FaceTracks using clustering based on embedding similarity.
        Supports GPU acceleration for similarity computation.

        Args:
            embedding_threshold (float): Cosine similarity threshold for linking tracks.
            device (str): "auto", "cpu", or "cuda". 
                - "auto": Use GPU if available, else CPU.
                - "cpu": Force CPU.
                - "cuda": Require GPU (raise error if not available).
        """
        self.embedding_threshold = embedding_threshold

        # Device selection logic
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Please check your environment.")
        else:
            self.device = device

        print(f"[INFO] GlobalIdentityResolver initialized on {self.device}")

    def resolve_global_ids(self, tracks: List[FaceTrack], start_id: int = 0) -> int:
        """
        Assign global_ids to FaceTracks by clustering based on embedding similarity.

        Uses a graph-based approach where edges are formed between tracks whose
        average embeddings exceed the similarity threshold. Connected components
        represent clusters of tracks belonging to the same global identity.

        Args:
            tracks (List[FaceTrack]): All FaceTracks (possibly across shots).
            start_id (int): Starting global_id counter.

        Returns:
            int: The next available global_id after assignment.
        """
        print(f"[DEBUG] Starting global resolution: {len(tracks)} tracks")

        # Collect normalized embeddings for valid tracks
        valid_indices, embeddings = [], []
        for i, track in enumerate(tracks):
            avg_emb = track.compute_average_embedding()
            if avg_emb is not None and np.linalg.norm(avg_emb) > 0:
                valid_indices.append(i)
                embeddings.append(avg_emb)

        if not embeddings:
            print("[DEBUG] No valid embeddings; assigning unique IDs")
            for track in tracks:
                track.global_id = start_id
                start_id += 1
            return start_id

        # Convert embeddings to torch tensor and normalize on chosen device
        emb_tensor = torch.tensor(np.stack(embeddings), dtype=torch.float32, device=self.device)
        emb_tensor = F.normalize(emb_tensor, p=2, dim=1)

        # Compute similarity matrix on GPU (if available)
        sim_matrix = torch.mm(emb_tensor, emb_tensor.T)

        # Threshold on GPU, then move to CPU for DFS
        adjacency = (sim_matrix >= self.embedding_threshold).cpu().numpy()

        # Build adjacency list and find connected components
        visited = set()
        components = []

        def dfs(node, comp):
            visited.add(node)
            comp.append(node)
            for neighbor, connected in enumerate(adjacency[node]):
                if connected and neighbor not in visited:
                    dfs(neighbor, comp)

        for node in range(len(valid_indices)):
            if node not in visited:
                comp = []
                dfs(node, comp)
                components.append(comp)

        # Assign global IDs
        for comp in components:
            for idx in comp:
                tracks[valid_indices[idx]].global_id = start_id
            start_id += 1

        # Assign unique IDs to tracks without embeddings
        assigned_set = set(valid_indices)
        for i, track in enumerate(tracks):
            if i not in assigned_set:
                track.global_id = start_id
                start_id += 1

        print(f"[DEBUG] Global ID assignment complete. Next ID: {start_id}")
        return start_id
