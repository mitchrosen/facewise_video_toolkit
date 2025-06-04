"""
Validation helper for shot_features.json files.

Usage:
    errors = validate_shot_features(json_path, schema_path,
                                    total_frame_count=<optional int>)
    if errors:
        print("Validation failed:")
        for e in errors:
            print(" -", e)
    else:
        print("All good!")
"""

import json
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker

# ---------- core validator -------------------------------------------------- #
def validate_shot_features_json(json_path: str | Path,
                           schema_path: str | Path,
                           total_frame_count: int | None = None) -> list[str]:
    """Validate file against schema + supplemental business rules.

    Returns a list of error strings (empty list = success).
    """
    json_path  = Path(json_path)
    schema_path = Path(schema_path)

    data   = json.loads(json_path.read_text())
    schema = json.loads(schema_path.read_text())

    # 1. draft-2020-12 structural validation
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = [f"schema at {'.'.join(map(str, e.absolute_path))}: {e.message}"
          for e in validator.iter_errors(data)]

    shots = data.get("shots", [])
    if not shots:
        errors.append("no shots found")
        return errors

    # 2. supplemental rules
    # 2a. shot_number must start at 1 and increment by 1
    for idx, shot in enumerate(shots, start=1):
        if shot["shot_number"] != idx:
            errors.append(
                f"shot_number mismatch at array index {idx-1}: "
                f"expected {idx}, got {shot['shot_number']}"
            )

    # 2b. frames must be contiguous
    for prev, curr in zip(shots[:-1], shots[1:]):
        if curr["first_frame"] != prev["last_frame"] + 1:
            errors.append(
                f"contiguity: shot {curr['shot_number']} first_frame "
                f"{curr['first_frame']} != previous last_frame+1 "
                f"({prev['last_frame']+1})"
            )

    # 2c. last frame must cover whole video (if total_frame_count provided)
    if total_frame_count is not None:
        expected_last = total_frame_count - 1
        if shots[-1]["last_frame"] != expected_last:
            errors.append(
                f"coverage: final shot last_frame {shots[-1]['last_frame']} "
                f"!= total_frame_count-1 ({expected_last})"
            )

    # 2d. face_count vs. face_details consistency
    for shot in shots:
        shot_num = shot.get("shot_number")
        detected_faces = shot.get("detected_faces", {})
        face_count = detected_faces.get("face_count")

        if face_count is None:
            errors.append(f"shot {shot_num}: missing face_count in detected_faces")
            continue

        has_details = "face_details" in detected_faces
        face_details = detected_faces.get("face_details", [])

        if face_count > 0:
            if not has_details:
                errors.append(f"shot {shot_num}: face_count > 0 but face_details is missing")
            elif len(face_details) != face_count:
                errors.append(
                    f"shot {shot_num}: face_count ({face_count}) does not match len(face_details) ({len(face_details)})"
                )
        elif face_count == 0 and has_details:
            errors.append(f"shot {shot_num}: face_count == 0 but face_details is present")


    return errors
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("json_file", help="shot_features.json to validate")
    p.add_argument("--schema", default="shot_features_schema.json",
                   help="Path to the JSON Schema file")
    p.add_argument("--total-frames", type=int,
                   help="Total number of frames in the source video")
    args = p.parse_args()

    errs = validate_shot_features(args.json_file, args.schema,
                                  total_frame_count=args.total_frame_count)
    if errs:
        print("❌ Validation errors:")
        for e in errs:
            print(" -", e)
        sys.exit(1)
    else:
        print("✅ shot_features.json is valid!")

