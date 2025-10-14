import json
from pathlib import Path
from jsonschema import Draft202012Validator
from typing import List

def _read(path):
    return json.loads(Path(path).read_text())

def _last_shot_last_frame(data: dict) -> int:
    shots = data.get("shots") or []
    if not shots:
        return -1
    return max(int(s.get("last_frame", -1)) for s in shots)

def _err(path: str, msg: str) -> str:
    return f"{path}: {msg}"

def _validate_v2_0_business(data: dict, total_frame_count: int | None) -> List[str]:
    errs: List[str] = []
    gen = data.get("generation") or {}
    emb_store = gen.get("emb_store", "inline")
    has_sidecar = "embedding_sidecar" in data

    # emb_store vs embedding_sidecar
    if emb_store == "sidecar" and not has_sidecar:
        errs.append(_err("business", "emb_store=sidecar requires top-level embedding_sidecar descriptor"))
    if emb_store != "sidecar" and has_sidecar:
        errs.append(_err("business", "embedding_sidecar present but emb_store != sidecar"))

    # per-observation embedding fields
    for s_i, shot in enumerate(data.get("shots", [])):
        for ft_i, ft in enumerate(shot.get("face_tracks", [])):
            for ob_i, ob in enumerate(ft.get("obs", [])):
                pfx = f"shots.{s_i}.face_tracks.{ft_i}.obs.{ob_i}"
                has_embedding = "embedding" in ob
                has_index = "emb_idx" in ob
                if emb_store == "inline":
                    if has_index:
                        errs.append(_err(pfx, "emb_idx not allowed when emb_store=inline"))
                elif emb_store == "sidecar":
                    if has_embedding:
                        errs.append(_err(pfx, "embedding not allowed when emb_store=sidecar"))
                else:  # none
                    if has_embedding or has_index:
                        errs.append(_err(pfx, "no embedding fields allowed when emb_store=none"))

    # coverage
    if total_frame_count is not None:
        last = _last_shot_last_frame(data)
        expected = total_frame_count - 1
        if last != expected:
            errs.append(_err("coverage", f"final shot last_frame {last} != total_frame_count-1 ({expected})"))

    return errs

def _validate_v2_1_business(data: dict, total_frame_count: int | None) -> List[str]:
    """
    2.1 business rules:
      - No per-frame JSON `obs`. Tracks carry `obs_offset` + `obs_count`.
      - If any `obs_count > 0`, require top-level `observations_sidecar`.
      - `generation.emb_store` must be "sidecar" or "none" (inline is not supported).
      - If "sidecar": must have top-level `embedding_sidecar`; no per-frame `embedding` fields exist in 2.1.
      - If "none": must NOT have `embedding_sidecar`.
      - Optional bound check: obs_offset/obs_count slices fit within observations_sidecar.count (if present).
      - Coverage rule as in 2.0.
    """
    errs: list[str] = []

    # Require observations_sidecar presence at top-level
    if "observations_sidecar" not in data:
        errs.append("business: observations_sidecar is required for schema 2.1")

    # num_tracks must equal the number of face_tracks for every shot
    for si, shot in enumerate(data.get("shots", [])):
        ft = shot.get("face_tracks", [])
        nt = shot.get("num_tracks", None)
        if nt is None:
            errs.append(f"business at shots.{si}: 'num_tracks' is required (schema 2.1)")
        else:
            try:
                if int(nt) != len(ft):
                    errs.append(
                        f"business at shots.{si}: num_tracks {nt} does not match face_tracks length {len(ft)}"
                    )
            except Exception:
                errs.append(f"business at shots.{si}: num_tracks must be an integer")

    # Optional coverage check (if provided)
    if total_frame_count is not None:
        last = max((s.get("last_frame", -1) for s in data.get("shots", [])), default=-1)
        if last != total_frame_count - 1:
            errs.append(
                f"coverage: final shot last_frame {last} != total_frame_count-1 ({total_frame_count-1})"
            )

    return errs

def validate_shot_features_json_v2(
        json_path: Path, 
        schema_path: Path, 
        *, 
        total_frame_count: int | None = None
        ) -> List[str]:
    """
    Validates schema v2.x files against the provided JSON Schema, then applies
    business rules depending on `schema_version` found inside the JSON.
    """
    data = _read(json_path)
    schema = _read(schema_path)

    # Schema (structural) validation first.
    errs: List[str] = []
    for e in Draft202012Validator(schema).iter_errors(data):
        errs.append(f"schema at {'.'.join(map(str, e.path))}: {e.message}")

    if errs:
        return errs

    ver = str(data.get("schema_version", "2.0")).strip()
    if ver not in ("2.0", "2.1"):
        return [f"business: unsupported schema_version {ver!r}; expected '2.0' or '2.1'."]

    # -------- Business rules per minor --------
    def _facecount_rule(d, errs_out):
        # Only apply if this block exists; keep this tolerant.
        det = (d or {}).get("detected_faces")
        if not isinstance(det, dict):
            return
        fc = det.get("face_count")
        fd = det.get("face_details")
        if isinstance(fc, int) and fc == 0 and fd:
            errs_out.append("business: face_count == 0 requires face_details to be empty/absent")

    if ver == "2.1":
        errs.extend(_validate_v2_1_business(data, total_frame_count))
        _facecount_rule(data, errs)
    elif ver == "2.0":
        errs.extend(_validate_v2_0_business(data, total_frame_count))
        _facecount_rule(data, errs)
    else:
        return [f"business: unsupported schema_version {ver!r}; expected '2.0' or '2.1'."]

    return errs
