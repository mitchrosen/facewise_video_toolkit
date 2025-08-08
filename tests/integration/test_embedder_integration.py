import os
import cv2
import numpy as np
import pytest
import torch

from facekit.embedding.embedder import FaceEmbedder
from facekit.embedding.alignment import align_face_for_arcface
from facekit.detection.face_detector import FaceDetector
from facekit.detection.yolo5face_model import load_yolo5face_model

@pytest.mark.integration
def test_real_arcface_similarity():
    """
    Full integration test:
    1. Load two images of the same person and one of a different person.
    2. Detect faces and landmarks using YOLOv5Face.
    3. Align faces using ArcFace template.
    4. Compute embeddings with FaceEmbedder.
    5. Assert that same-person similarity > different-person similarity.
    """

    embedding_model_path = "models/embedding/glintr100_dynamic.onnx"
    img1_path = "tests/assets/faces/face1_1.jpg"
    img2_path = "tests/assets/faces/face1_2.jpg"
    img3_path = "tests/assets/faces/face2_1.jpg"

    if not all(os.path.exists(p) for p in [embedding_model_path, img1_path, img2_path, img3_path]):
        pytest.skip("Required model or image assets are missing")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load ArcFace embedder
    embedder = FaceEmbedder(embedding_model_path=embedding_model_path, device="cpu")

    # Load YOLOv5Face detector and wrap with FaceDetector
    detector_model = load_yolo5face_model(
        detector_model_path="models/detector/yolov5n_state_dict.pt", 
        config_path="models/detector/yolov5n.yaml", 
        device="cpu"
    )
    detector = FaceDetector(detector_model)

    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)
    assert img1 is not None and img2 is not None and img3 is not None

    # Detect faces and extract landmarks
    def detect_and_align(img):
        result = detector.detect_faces_in_frame(img)
        assert result is not None, "Detection failed"
        boxes, landmarks, confs = result
        assert len(boxes) > 0, "No face detected"
        return align_face_for_arcface(img, landmarks[0])  # Align first face

    aligned1 = detect_and_align(img1)
    aligned2 = detect_and_align(img2)
    aligned3 = detect_and_align(img3)

    # Compute embeddings
    embeddings = embedder.get_embedding_batch([aligned1, aligned2, aligned3])
    assert len(embeddings) == 3
    emb1, emb2, emb3 = embeddings

    # Cosine similarity
    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_same = cosine_similarity(emb1, emb2)
    sim_diff = cosine_similarity(emb1, emb3)

    print(f"[DEBUG] sim_same={sim_same:.4f}, sim_diff={sim_diff:.4f}")
    assert sim_same > sim_diff, f"Expected same-person sim ({sim_same:.4f}) > diff-person sim ({sim_diff:.4f})"
