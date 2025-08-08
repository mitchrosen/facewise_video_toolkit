from typing import Dict, List

# shot_id → frame_id → track_id → embedding (list of floats)
EmbeddingDict = Dict[str, Dict[str, Dict[str, List[float]]]]

# shot_id → track_id → mean embedding
RepresentativeEmbeddingDict = Dict[str, Dict[str, List[float]]]
