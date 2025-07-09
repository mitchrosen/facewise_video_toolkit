import numpy as np
import pytest
import cv2
import matplotlib.pyplot as plt
from facekit.embedding.embedder import FaceEmbedder

import sys
print("Python executable:", sys.executable)
print("sys.path:", sys.path)

@pytest.fixture
def dummy_frame():
    # 200x200 gray image
    return np.full((200, 200, 3), 127, dtype=np.uint8)

@pytest.fixture
def dummy_bbox():
    # Valid centered face box
    return [50, 50, 150, 150]

def load_face(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return img

def test_embedding_basic_face():
    face = load_face("tests/assets/faces/face1_1.jpg")
    embedder = FaceEmbedder()
    embedding = embedder.get_embedding(face)

    assert embedding is not None, "Embedding is None"
    assert embedding.shape == (512,), f"Unexpected shape: {embedding.shape}"

def test_embedding_shape_and_type(dummy_frame):
    face = load_face("tests/assets/faces/face1_1.jpg")
    embedder = FaceEmbedder()
    embedding = embedder.get_embedding(face)

    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == np.float32
    assert embedding.shape == (512,)

def test_embedding_is_normalized(dummy_frame):
    face = load_face("tests/assets/faces/face1_1.jpg")
    embedder = FaceEmbedder()
    embedding = embedder.get_embedding(face)

    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-3), f"Embedding norm was {norm}"

def test_embedding_is_deterministic(dummy_frame):
    face = load_face("tests/assets/faces/face1_1.jpg")
    embedder = FaceEmbedder()

    emb1 = embedder.get_embedding(face)
    emb2 = embedder.get_embedding(face)

    assert np.allclose(emb1, emb2), "Embedding not deterministic"

def show_face_image(img: np.ndarray, title: str = "Face Image") -> None:
    """
    Displays a face image using matplotlib.
    
    Args:
        img: A NumPy array representing the image (BGR format).
        title: Optional title for the plot.
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Expected a NumPy image array, got None or invalid type.")

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot the image
    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

def test_embedding_similarity_close_images():
    embedder = FaceEmbedder()

    for face_num in range(1,4):
        emb = []
        for face_version in range(1,5):
            face = load_face(f"tests/assets/faces/face{face_num}_{face_version}.jpg")
            show_face_image(face, f"test_embedding_similarity_close_images(), face{face_num}_{face_version}")
            emb.append(embedder.get_embedding(face))

        for face_version1 in range(4):
            for face_version2 in range(4):
                if face_version1 == face_version2:
                    continue
                similarity = np.dot(emb[face_version1], emb[face_version2])
                assert similarity > 0.30, f"for face{face_num} comparing version {face_version1} vs {face_version2}, expected high similarity, got {similarity:.4f}"

def test_embedding_similarity_different_images():
    embedder = FaceEmbedder()

    for face_version in range(1,5):
        emb = []
        for face_num in range(1,4):
            face = load_face(f"tests/assets/faces/face{face_num}_{face_version}.jpg")
            show_face_image(face, f"test_embedding_similarity_close_images(), face{face_num}_{face_version}")
            emb.append(embedder.get_embedding(face))

        for face_num1 in range(3):
            for face_num2 in range(3):
                if face_num1 == face_num2:
                    continue
                similarity = np.dot(emb[face_num1], emb[face_num2])
                assert similarity < 0.06, f"for face version{face_version} comparing face nums {face_num1} vs {face_num2}, expected low similarity, got {similarity:.4f}"