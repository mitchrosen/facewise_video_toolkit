import numpy as np
from typing import List, Optional
from insightface.app import FaceAnalysis


class FaceEmbedder:
    def __init__(self, model_name: str = "buffalo_l", device: str = "cpu"):
        """
        Initialize the face embedding model.
        Args:
            model_name: Name of the InsightFace model to use.
            device: Device to load model onto ('cpu' or 'cuda').
        """
        self.device = device
        provider = "CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"

        self.app = FaceAnalysis(name=model_name, providers=[provider])
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

    def get_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a normalized 512-dimensional embedding from the most prominent face in the frame.

        Args:
            frame: Original BGR image as a NumPy array.

        Returns:
            A 512-dimensional float32 NumPy array (L2-normalized), or None if no face is found.
        """
        import warnings
        with warnings.catch_warnings():     # Suppress FutureWarning from insightface (as of 7/9/2025) due to internal use of np.linalg.lstsq.
                                            # Safe to remove this filter once insightface updates and no longer triggers the warning.
            warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
            faces = self.app.get(frame)

        if not faces or faces[0].embedding is None:
            return None

        embedding = np.asarray(faces[0].embedding, dtype=np.float32)

        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]  # flatten

        if embedding.shape != (512,):
            raise ValueError(f"Unexpected embedding shape: {embedding.shape} (expected (512,))")

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding
