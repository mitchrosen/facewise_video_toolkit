from typing import Dict, List
import numpy as np
from facekit.embedding.embedding_types import EmbeddingDict, RepresentativeEmbeddingDict


# Output format: shot → track → mean embedding
def compute_representative_embeddings(embeddings_dict: EmbeddingDict) -> Dict[str, Dict[str, List[float]]]:
    """
    Computes a representative (mean) embedding per track within each shot.

    Args:
        embeddings_dict: Nested dict: shot_id → frame_id → track_id → embedding (list of floats)

    Returns:
        Dict[shot_id][track_id] = mean embedding (list of floats)
    """
    output: Dict[str, Dict[str, List[float]]] = {}

    for shot_id, frames in embeddings_dict.items():
        track_accumulator: Dict[str, List[np.ndarray]] = {}

        for _, track_dict in frames.items():
            for track_id, emb_list in track_dict.items():
                emb = np.array(emb_list, dtype=np.float32)
                track_accumulator.setdefault(track_id, []).append(emb)

        output[shot_id] = {
            track_id: np.mean(embs, axis=0).tolist() for track_id, embs in track_accumulator.items()
        }

    return output
