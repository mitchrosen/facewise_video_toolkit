import numpy as np
import pytest
import json
from facekit.embedding.embedding_summary import compute_representative_embeddings
from facekit.embedding.embedding_types import EmbeddingDict
from facekit.embedding.embedding_extraction import crop_face, save_embeddings_to_json

class DummyEmbedder:
    def get_embedding_batch(self, faces, batch_size=32):
        # Return a batch of embeddings where each embedding is a 512-dim vector of ones
        return np.array([np.ones(512, dtype=np.float32) for _ in faces], dtype=np.float32)

@pytest.fixture
def dummy_frame():
    # Create a 100x100 dummy image
    return np.full((100, 100, 3), 255, dtype=np.uint8)

def test_crop_face_shape(dummy_frame):
    bbox = (10, 20, 50, 60)
    cropped = crop_face(dummy_frame, bbox)
    assert cropped.shape == (40, 40, 3)

def test_compute_representative_embeddings_basic():
    # Mock input embeddings for shot_001 with two frames, one track
    emb1 = np.ones(512, dtype=np.float32)
    emb2 = np.full(512, 3.0, dtype=np.float32)

    embeddings_dict: EmbeddingDict = {
        "shot_001": {
            "frame_001": {
                "track_1": emb1.tolist()
            },
            "frame_002": {
                "track_1": emb2.tolist()
            }
        }
    }

    result = compute_representative_embeddings(embeddings_dict)

    assert "shot_001" in result
    assert len(result) == 1
    assert "track_1" in result["shot_001"]
    assert len(result["shot_001"]) == 1

    rep = np.array(result["shot_001"]["track_1"])
    expected = np.mean([emb1, emb2], axis=0)

    assert rep.shape == (512,)
    np.testing.assert_allclose(rep, expected, rtol=1e-5)

def test_compute_representative_embeddings_single_track():
    emb = np.ones(512, dtype=np.float32).tolist()
    embeddings: EmbeddingDict = {
        "shot_1": {
            "frame_001": {"track_0": emb},
            "frame_002": {"track_0": emb}
        }
    }
    result = compute_representative_embeddings(embeddings)
    assert "shot_1" in result
    assert len(result) == 1
    assert "track_0" in result["shot_1"]
    assert len(result["shot_1"]) == 1

    np.testing.assert_allclose(result["shot_1"]["track_0"], emb)

def test_compute_representative_embeddings_multiple_tracks():
    emb1 = np.ones(512, dtype=np.float32)*50
    emb2 = np.ones(512, dtype=np.float32)*200
    embeddings: EmbeddingDict = {
        "shot_a": {
            "frame_001": {"track_x": emb1.tolist(), "track_y": emb2.tolist()}
        }
    }
    result = compute_representative_embeddings(embeddings)
    assert "shot_a" in result
    assert len(result) == 1
    assert "track_x" in result["shot_a"]
    assert "track_y" in result["shot_a"]
    assert len(result["shot_a"]) == 2
    np.testing.assert_allclose(result["shot_a"]["track_x"], emb1)
    np.testing.assert_allclose(result["shot_a"]["track_y"], emb2)

def test_compute_representative_embeddings_empty():
    result = compute_representative_embeddings({})
    assert result == {}

def test_save_embeddings_to_json(tmp_path):
    embeddings: EmbeddingDict = {
        "shot_1": {
            "frame_0": {"track_1": np.ones(512).tolist()}
        }
    }
    json_path = tmp_path / "out.json"
    save_embeddings_to_json(embeddings, str(json_path))
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
        assert "shot_1" in data
        assert len(data)==1
        assert "frame_0" in data["shot_1"]  # JSON keys become strings
        assert len(data["shot_1"])==1
        assert "track_1" in data["shot_1"]["frame_0"]
        assert len(data["shot_1"]["frame_0"]) == 1
        assert isinstance(data["shot_1"]["frame_0"]["track_1"], list)
