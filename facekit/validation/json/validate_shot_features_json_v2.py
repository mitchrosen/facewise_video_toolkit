import json
from pathlib import Path
from jsonschema import Draft202012Validator
from typing import List

def validate_shot_features_json_v2(
        json_path: Path, 
        schema_path: Path, 
        *, 
        total_frame_count: int | None = None
        ) -> List[str]:
    data = json.loads(Path(json_path).read_text())
    schema = json.loads(Path(schema_path).read_text())

    errs: List[str] = []
    for e in Draft202012Validator(schema).iter_errors(data):
        errs.append(f"schema at {'.'.join(map(str, e.path))}: {e.message}")

    if errs:
        return errs

    # -------- Business rules ----------
    emb_store = (data.get("generation") or {}).get("emb_store", "inline")

    # sidecar present iff emb_store == "sidecar"
    has_sidecar = "embedding_sidecar" in data
    if emb_store == "sidecar" and not has_sidecar:
        errs.append("business: emb_store=sidecar requires top-level embedding_sidecar descriptor")
    if emb_store != "sidecar" and has_sidecar:
        errs.append("business: embedding_sidecar present but emb_store != sidecar")

    # per-observation embedding fields must match emb_store
    for s_i, shot in enumerate(data.get("shots", [])):
        for ft_i, ft in enumerate(shot.get("face_tracks", [])):
            for ob_i, ob in enumerate(ft.get("obs", [])):
                if emb_store == "inline":
                    if "emb_idx" in ob:
                        errs.append(f"business at shots.{s_i}.face_tracks.{ft_i}.obs.{ob_i}: emb_idx not allowed when emb_store=inline")
                    # "embedding" is allowed but optional
                elif emb_store == "sidecar":
                    if "embedding" in ob:
                        errs.append(f"business at shots.{s_i}.face_tracks.{ft_i}.obs.{ob_i}: embedding not allowed when emb_store=sidecar")
                    # emb_idx is allowed (also optional)
                else:  # "none"
                    if "embedding" in ob or "emb_idx" in ob:
                        errs.append(f"business at shots.{s_i}.face_tracks.{ft_i}.obs.{ob_i}: no embedding fields allowed when emb_store=none")

    # coverage rule (if provided)
    if total_frame_count is not None:
        last = max((s.get("last_frame", -1) for s in data.get("shots", [])), default=-1)
        if last != total_frame_count - 1:
            errs.append(f"coverage: final shot last_frame {last} != total_frame_count-1 ({total_frame_count-1})")

    return errs
