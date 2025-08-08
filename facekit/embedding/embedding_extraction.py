import json
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict
from facekit.embedding.embedding_types import EmbeddingDict, RepresentativeEmbeddingDict
from facekit.embedding.embedder import FaceEmbedder


def save_embeddings_to_json(embeddings_dict: EmbeddingDict, output_path: str):
    """
    Write nested dict structure to JSON. Converts numpy arrays to lists.
    """
    with open(output_path, 'w') as f:
        json.dump(embeddings_dict, f, indent=2)


def crop_face(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop a face from the frame using the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


