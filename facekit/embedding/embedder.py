import numpy as np
from typing import List, Optional
from insightface.model_zoo import ArcFaceONNX

class FaceEmbedder:
    def __init__(self, embedding_model_path: str, device: str = "cpu"):
        """
        Initialize ONNX ArcFace model for embeddings.
        Args:
            embedding_model_path: Path to ArcFace ONNX model file.
            device: 'cpu' or 'cuda'.
        """
        # Debug: Check if the model file exists
        import os
        if not os.path.exists(embedding_model_path):
            print(f"DEBUG Model file not found: {embedding_model_path}")
        else:
            print(f"DEBUG Model file found: {embedding_model_path} (size: {os.path.getsize(embedding_model_path) / (1024*1024):.2f} MB)")

        self.embedding_model = ArcFaceONNX(model_file=embedding_model_path)
        ctx_id = 0 if device == "cpu" else -1
        self.embedding_model.prepare(ctx_id=ctx_id)
        if hasattr(self.embedding_model, "input_size") and hasattr(self.embedding_model, "session"):
            print(f"DEBUG prepare(): {self.embedding_model.input_size}, {self.embedding_model.session}")
        
        self.input_size = (112, 112)

    def get_embedding_batch(self, aligned_faces: List[np.ndarray], batch_size: int = 32) -> List[np.ndarray]:
        """
        Compute embeddings for a batch of aligned faces.
        Args:
            aligned_faces: List of aligned face images (112x112 RGB).
            batch_size: Max number of faces to process per batch.
        Returns:
            List of normalized embeddings (float32 arrays).
        """
        if not isinstance(aligned_faces, (list, tuple)) or not all(isinstance(f, np.ndarray) for f in aligned_faces):
            raise TypeError("aligned_faces must be a list of numpy arrays.")
        if not aligned_faces:
            return []
        if not all(f.shape == (112, 112, 3) for f in aligned_faces):
            raise ValueError("Each face must be aligned to (112,112,3) RGB.")
    
        embeddings = []
        for i in range(0, len(aligned_faces), batch_size):
            batch = aligned_faces[i:i + batch_size]

            # # Apply ArcFace normalization: (img - 127.5) / 128.0
            # batch = np.stack([(face.astype(np.float32) - 127.5) / 128.0 for face in batch])  # N,H,W,C
            # batch = batch.transpose(0, 3, 1, 2)  # N,C,H,W      
            
            batch_embeddings = self.embedding_model.get_feat(batch)
            batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32, order="C")  #ensure float32, contiguous       # ensure float32, contiguous

            # L2 normalize
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)                 # guard divide-by-zero

            batch_embeddings = batch_embeddings / norms
            embeddings.extend(batch_embeddings)
        
        embeddings_array = np.vstack(embeddings)               # Concatenate per‑batch results into array, cache- and vectorized-ops-friendly                              # (K, 512)
        return embeddings_array
