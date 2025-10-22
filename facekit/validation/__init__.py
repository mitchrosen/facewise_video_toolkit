from __future__ import annotations
from pathlib import Path
import json as _json
import re
import os
from typing import Callable, Protocol, Dict, Tuple, List
from importlib import resources

from facekit.errors import UnknownSchemaVersion

# ---- Registry of implemented validators (source of truth) ------------------
class JsonValidator(Protocol):
    def __call__(self,
                 json_path: Path | str,
                 schema_path: Path | str,
                 total_frame_count: int | None = None) -> list[str]:
        ...

def _wrap_validator(fn) -> JsonValidator:
    # Normalize to a uniform calling convention.
    def _call(json_path: Path | str,
              schema_path: Path | str,
              total_frame_count: int | None = None) -> list[str]:
        # Try keyword first (most robust), fall back to positional if needed.
        try:
            return fn(json_path, schema_path, total_frame_count=total_frame_count)
        except TypeError:
            return fn(json_path, schema_path, total_frame_count)
    return _call

from .json.validate_shot_features_json_v1 import validate_shot_features_json_v1 as _v1
from .json.validate_shot_features_json_v2 import validate_shot_features_json_v2 as _v2

SUPPORTED_VALIDATORS: Dict[int, JsonValidator] = {
    1: _wrap_validator(_v1),
    2: _wrap_validator(_v2),
}

# ---- Schema discovery ------------------------------------------------------
_SCHEMA_RX = re.compile(r"^shot_features_v(?P<ver>\d+(?:\.\d+)?)\.schema\.json$")

def _parse_ver(v: str) -> Tuple[int,int]:
    if not re.fullmatch(r"\d+(?:\.\d+)?", v):
        raise UnknownSchemaVersion(f"Bad schema_version format: {v!r} (use '1', '1.0', '2', '2.0', or 'latest')")
    if "." in v:
        a,b = v.split(".",1)
        return int(a), int(b)
    return int(v), 0

def _schemas_base(schema_dir: str | Path | None = None) -> Path:
    """
    Where schemas live, in priority order:
      1) explicit schema_dir argument
      2) FACEKIT_SCHEMA_DIR env var
      3) package-relative default (facekit/validation/schemas)
    """
    # Explicit arg
    if schema_dir is not None:
        return Path(schema_dir).resolve()
    
    # If not above, then try environment override
    env = os.getenv("FACEKIT_SCHEMA_DIR")
    if env:
        return Path(env).resolve()
    
    # If not above, walk up from this file looking for a top-level "schemas"
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        candidate = parent.parent / "schemas" if parent.name == "validation" else parent / "schemas"
        if candidate.is_dir():
            # sanity: looks like our schemas (pattern match one file)
            if any(_SCHEMA_RX.match(p.name) for p in candidate.iterdir() if p.is_file()):
                return candidate.resolve()

    # If not above, package data (installed wheel): facekit/validation/schemas
    try:
        pkg_dir = Path(resources.files("facekit.validation"), "schemas")
        # Traversable -> best effort path conversion
        pkg_path = Path(str(pkg_dir))
        if pkg_path.is_dir():
            return pkg_path.resolve()
    except Exception:
        pass

    # If not above, fallback to module-adjacent
    mod_adjacent = Path(__file__).parent / "schemas"
    return mod_adjacent.resolve()

def _available_schema_versions(schema_dir: str | Path | None = None) -> List[Tuple[str, Path]]:
    base = _schemas_base(schema_dir)
    out: List[Tuple[str, Path]] = []
    if not base.exists():
        raise FileNotFoundError(
            f"No schemas directory found at: {base}\n"
            f"Hint: pass --schema-dir, or set FACEKIT_SCHEMA_DIR to your repo's schemas directory."
        )
    for p in base.iterdir():
        p = Path(p)
        if not p.is_file(): 
            continue
        m = _SCHEMA_RX.match(p.name)
        if m:
            out.append((m.group("ver"), p))
    out.sort(key=lambda vp: _parse_ver(vp[0]))
    return out

def get_schema_path(version: str | None, *, schema_dir: str | Path | None = None) -> Path:
    """
    Resolve a bundled schema path by filename pattern (no hardcoded table):
      - 'latest'          -> highest discovered version
      - 'N'               -> highest discovered N.x; else error
      - 'N.x'             -> exact file must exist; else error
      - ''/None           -> prefer '1.0' if present; else lowest discovered
      - 'N.x' missing     -> error (no silent fallback)
      - unknown major 'Q' -> error (until a Q.x file exists)
    """
    discovered = _available_schema_versions(schema_dir)
    if not discovered:
        base = _schemas_base(schema_dir)
        raise FileNotFoundError(f"No schemas found under {base}")

    by_exact = {v: p for v,p in discovered}
    by_major: Dict[int, List[Tuple[str,Path]]] = {}
    for v,p in discovered:
        maj,_ = _parse_ver(v)
        by_major.setdefault(maj, []).append((v,p))

    req = (version or "").strip()
    if req.lower() == "latest":
        return discovered[-1][1]
    if req == "" or req is None:
        return by_exact.get("1.0", discovered[0][1])
    if req in by_exact:
        return by_exact[req]
    if re.fullmatch(r"\d+", req):
        maj = int(req)
        if maj in by_major:
            return by_major[maj][-1][1]  # highest minor of that major
        raise UnknownSchemaVersion(
            f"No schemas found for major {maj}. Available: {', '.join(v for v,_ in discovered)}"
        )
    if re.fullmatch(r"\d+\.\d+", req):
        raise UnknownSchemaVersion(
            f"Requested schema_version {req} not found. Available: {', '.join(v for v,_ in discovered)}"
        )
    raise UnknownSchemaVersion(
        f"Unrecognized schema_version spec: {req!r}. "
        "Use 'latest', '<MAJOR>' (e.g. '2'), or '<MAJOR>.<MINOR>' (e.g. '2.0')."
    )

def _major_for_schema_path(path: Path) -> int:
    m = _SCHEMA_RX.match(path.name)
    if not m:
        raise UnknownSchemaVersion(
            f"Schema filename {path.name!r} does not match expected pattern "
            f"'shot_features_v<MAJOR>[.<MINOR>].schema.json'"
        )
    maj,_ = _parse_ver(m.group("ver"))
    return maj

# ---- Public dispatcher -----------------------------------------------------
def validate_manifest(json_path: str | Path,
                      *,
                      total_frame_count: int | None = None,
                      schema_path: str | Path | None = None,
                      schema_version: str | None = None,
                      schema_dir: str | Path | None = None) -> list[str]:
    """
    Version-dispatching validator (schema auto-discovery + explicit registry):
      - If `schema_path` is provided, use it (and verify filename pattern).
      - Else, if `schema_version` is provided, resolve via get_schema_path().
      - Else, read `schema_version` from the JSON (default '1.0') and resolve.
      - Picks the validator by **major version** using SUPPORTED_VALIDATORS.
      - Returns [] on success; list of error strings otherwise.
    """
    json_path = Path(json_path)

    if schema_path is None:
        if schema_version is None:
            data = _json.loads(json_path.read_text())
            schema_version = str(data.get("schema_version", "1.0"))
        schema_path = get_schema_path(schema_version, schema_dir=schema_dir)

    schema_path = Path(schema_path)
    major = _major_for_schema_path(schema_path)

    validator = SUPPORTED_VALIDATORS.get(major)
    if validator is None:
        discovered = _available_schema_versions()
        supported = ", ".join(map(str, sorted(SUPPORTED_VALIDATORS)))
        available = ", ".join(v for v,_ in discovered) or "none"
        raise UnknownSchemaVersion(
            f"No validator implemented for major version {major}. "
            f"Supported majors: {supported}. Discovered schemas: {available}."
        )

    return validator(json_path, schema_path, total_frame_count=total_frame_count)
