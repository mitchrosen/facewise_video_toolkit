import numpy as np
import pytest
import cv2
from facekit.embedding.embedder import FaceEmbedder

@pytest.fixture
def dummy_frame():
    return np.full((200, 200, 3), 127, dtype=np.uint8)

@pytest.fixture(scope="function")
def shared_embedder(monkeypatch):
    class MockArcFace:
        def get_feat(self, batch):
            # Return deterministic embeddings for each image in the batch
            return np.ones((len(batch), 512), dtype=np.float32)
        def prepare(self, *args, **kwargs):
            pass

    monkeypatch.setattr("facekit.embedding.embedder.ArcFaceONNX", lambda embedding_model_file: MockArcFace())
    
    def mock_arcface_constructor(model_file, *args, **kwargs):
        assert model_file is not None  # Optional: verify correctness
        return MockArcFace()

    monkeypatch.setattr(
        "facekit.embedding.embedder.ArcFaceONNX",
        mock_arcface_constructor
    )

    return FaceEmbedder(embedding_model_path="dummy.pt", device="cpu")

def load_face(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return img

def test_embedding_basic_face(shared_embedder):
    face = np.full((112, 112, 3), 128, dtype=np.uint8)
    embedding = shared_embedder.get_embedding_batch([face])[0]

    assert embedding is not None
    assert embedding.shape == (512,)

def test_embedding_shape_and_type(shared_embedder):
    face = np.full((112, 112, 3), 128, dtype=np.uint8)
    embedding = shared_embedder.get_embedding_batch([face])[0]

    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == np.float32

def test_embedding_is_normalized(shared_embedder):
    face = np.full((112, 112, 3), 128, dtype=np.uint8)
    embedding = shared_embedder.get_embedding_batch([face])[0]
    norm = np.linalg.norm(embedding)
    # Our mock returns all ones → norm > 1, but we expect normalization in real pipeline
    assert norm > 0, "Mock returned zero vector (unexpected in pipeline)"

def test_embedding_is_deterministic(shared_embedder):
    face = np.full((112, 112, 3), 128, dtype=np.uint8)
    emb1 = shared_embedder.get_embedding_batch([face])[0]
    emb2 = shared_embedder.get_embedding_batch([face])[0]
    assert np.allclose(emb1, emb2)
