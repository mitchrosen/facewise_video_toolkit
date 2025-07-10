import numpy as np
import pytest
import json
from facekit.embedding.embedding_summary import compute_representative_embeddings
from facekit.embedding.embedding_extraction import crop_face, extract_embedding_for_track, save_embeddings_to_json
from facekit.embedding.embedding_types import EmbeddingDict

class DummyEmbedder:
    def get_embedding(self, face_crop):
        return np.ones(512, dtype=np.float32)

@pytest.fixture
def dummy_frame():
    # Create a 100x100 dummy image
    return np.full((100, 100, 3), 255, dtype=np.uint8)

def test_crop_face_shape(dummy_frame):
    bbox = (10, 20, 50, 60)
    cropped = crop_face(dummy_frame, bbox)
    assert cropped.shape == (40, 40, 3)

def test_extract_embedding_structure(dummy_frame):
    embeddings: EmbeddingDict = {}
    shot_id = "shot_001"
    frame_idx = 12
    frame_id = f"frame_{frame_idx}"
    track_id = "track_0"
    bbox = (10, 20, 50, 60)
    extract_embedding_for_track(embeddings, dummy_frame, shot_id, frame_idx, track_id, bbox, DummyEmbedder())

    assert shot_id in embeddings
    assert len(embeddings) == 1
    assert frame_id in embeddings[shot_id]
    assert len(embeddings[shot_id]) == 1
    assert track_id in embeddings[shot_id][frame_id]
    assert len(embeddings[shot_id][frame_id]) == 1
    assert isinstance(embeddings[shot_id][frame_id][track_id], list)
    assert len(embeddings[shot_id][frame_id][track_id]) == 512

def test_extract_embedding_none_skipped(dummy_frame):
    class NoneEmbedder:
        def get_embedding(self, face_crop):
            return None

    embeddings: EmbeddingDict = {}
    extract_embedding_for_track(embeddings, dummy_frame, "shot_001", 0, "track_0", (10, 10, 20, 20), NoneEmbedder())
    assert embeddings == {}

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
