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


def extract_embedding_for_track(
    embeddings_dict: EmbeddingDict,
    frame: np.ndarray,
    shot_id: int,
    frame_idx: int,
    track_id: str,
    bbox: Tuple[int, int, int, int],
    face_embedder: FaceEmbedder
):
    """
    Extracts an embedding for the cropped face and stores it in the nested dict under shot_id → frame_id → track_id.
    """
    face_crop = crop_face(frame, bbox)
    embedding = face_embedder.get_embedding(face_crop)
    if embedding is not None:
        embedding = embedding.tolist()
        frame_id = f"frame_{frame_idx}"
        if shot_id not in embeddings_dict:
            embeddings_dict[shot_id] = {}
        if frame_id not in embeddings_dict[shot_id]:
            embeddings_dict[shot_id][frame_id] = {}
        embeddings_dict[shot_id][frame_id][track_id] = embedding

