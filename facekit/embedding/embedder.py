import os
from typing import List
import numpy as np
from insightface.model_zoo import ArcFaceONNX
import logging


class FaceEmbedder:
    def __init__(self, embedding_model_path: str, device: str = "cuda"):
        """
        Face embedder using InsightFace ArcFace ONNX.

        Args:
            embedding_model_path: Path to ArcFace .onnx file (expects 112x112 input).
            device: 'cuda' to prefer GPU, 'cpu' to force CPU.
                    ArcFaceONNX: ctx_id >= 0 → GPU, ctx_id < 0 → CPU.
        """
        self.embedding_model = ArcFaceONNX(model_file=embedding_model_path)
        # Keep explicit paths/knobs for provenance
        self.model_path: str = embedding_model_path
        self.device: str = (device or "cpu").lower()
        self.embedding_dim: int = 512
        self.max_batch_size: int = 512

        # Prepare the model (ArcFaceONNX controls ORT session creation)
        dev = self.device
        ctx_id = 0 if dev == "cuda" else -1
        self.embedding_model.prepare(ctx_id=ctx_id)

        # --- Enforce ONNX Runtime providers to match requested device ---
        try:
            import onnxruntime as ort
            sess = getattr(self.embedding_model, "session", None)
            model_file = getattr(self.embedding_model, "model_file", embedding_model_path)
            if sess is not None:
                if dev == "cuda":
                    avail = ort.get_available_providers()
                    want = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
                    try:
                        sess.set_providers(want)
                    except Exception:
                        so = ort.SessionOptions()
                        self.embedding_model.session = ort.InferenceSession(model_file, so, providers=want)
                else:
                    try:
                        sess.set_providers(["CPUExecutionProvider"])
                    except Exception:
                        so = ort.SessionOptions()
                        self.embedding_model.session = ort.InferenceSession(
                            model_file, so, providers=["CPUExecutionProvider"]
                        )
        except Exception as e:
            logging.warning("WARN: could not enforce ORT providers:", repr(e))

        self.input_size = (112, 112)

    def set_max_batch_size(self, batch_size: int) -> None:
        try:
            self.max_batch_size = int(batch_size)
        except Exception:
            pass

    def provenance(self) -> dict:
        """
        Stable, minimal provenance.
        """
        # Try to get the final model file path from the underlying object, fall back to arg
        model_file = getattr(self.embedding_model, "model_file", None) or self.model_path
        return {
            "name": "arcface",
            "model": str(model_file),
            "dim": int(self.embedding_dim),
            "batch_max": int(self.max_batch_size),
            "device": self.device,
        }

    def get_embedding_batch(self, aligned_faces: List[np.ndarray], batch_size: int = 512) -> np.ndarray:
        """
        Compute embeddings for aligned faces in chunks (true batching).

        Args:
            aligned_faces: List of (112,112,3) uint8 aligned RGB/BGR crops.
            batch_size: Chunk size per ONNX call.

        Returns:
            (N, 512) float32 L2-normalized embeddings.
        """
        self.set_max_batch_size(batch_size)

        if not isinstance(aligned_faces, (list, tuple)) or not all(isinstance(f, np.ndarray) for f in aligned_faces):
            raise TypeError("aligned_faces must be a list of numpy arrays.")
        if not aligned_faces:
            return np.zeros((0, 512), dtype=np.float32)
        if not all(f.shape == (112, 112, 3) for f in aligned_faces):
            raise ValueError("Each face must be aligned to (112,112,3).")

        out_chunks = []

        for i in range(0, len(aligned_faces), batch_size):
            batch = aligned_faces[i:i + batch_size]
            # ArcFaceONNX.get_feat accepts a list and does its own preprocessing.
            embs = self.embedding_model.get_feat(batch)  # (B, 512)
            embs = np.asarray(embs, dtype=np.float32, order="C")
            # L2-normalize
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            out_chunks.append(embs / norms)

        return np.vstack(out_chunks)

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """Convenience: single face → (512,) float32."""
        return self.get_embedding_batch([aligned_face], batch_size=1)[0]
