import json
from pathlib import Path
from facekit.tracking.face_structures import FaceTrack, FaceObservation


def tracks_to_json_dict(tracks, include_embeddings=False):
    """
    Convert a list of FaceTrack objects into a JSON-serializable dictionary.

    Args:
        tracks (List[FaceTrack]): List of face tracks
        include_embeddings (bool): If True, include embedding vectors in output

    Returns:
        dict: JSON-compatible dictionary
    """
    return {
        "tracks": [
            {
                "shot_id": track.shot_id,
                "track_id": track.track_id,
                "face_label": track.global_id,  # <-- Added field
                "observations": [
                    {
                        "frame_idx": obs.frame_idx,
                        "bbox": list(obs.bbox),
                        "confidence": obs.confidence,
                        **({"embedding": obs.embedding.tolist()} if include_embeddings and obs.embedding is not None else {})
                    }
                    for obs in track.observations
                ]
            }
            for track in tracks
        ]
    }

def load_tracks_from_json_dict(json_dict):
    """
    Convert a JSON-compatible dictionary into a list of FaceTrack objects.

    Args:
        json_dict (dict): Dictionary produced by tracks_to_json_dict()

    Returns:
        List[FaceTrack]: List of FaceTrack instances
    """
    tracks = []
    for t in json_dict.get("tracks", []):
        shot_id = t["shot_id"]
        track_id = t["track_id"]
        observations = [
            FaceObservation(
                frame_idx=obs["frame_idx"],
                bbox=tuple(obs["bbox"]),
                confidence=obs.get("confidence", 1.0),
                embedding=np.array(obs["embedding"]) if "embedding" in obs else None
            )
            for obs in t["observations"]
        ]
        tracks.append(FaceTrack(shot_id= shot_id, track_id=track_id, observations=observations))
    return tracks

def save_tracks_to_json_file(tracks, output_path, include_embeddings=False):
    """
    Save tracks to a JSON file.

    Args:
        tracks (List[FaceTrack])
        output_path (str or Path): Where to save the file
        include_embeddings (bool): Whether to include embeddings in output
    """
    data = tracks_to_json_dict(tracks, include_embeddings=include_embeddings)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_tracks_from_json_file(json_path):
    """
    Load FaceTrack objects from a JSON file.

    Args:
        json_path (str or Path): Path to the JSON file

    Returns:
        List[FaceTrack]: List of FaceTrack instances
    """
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
    return load_tracks_from_json_dict(data)
