from __future__ import annotations
from enum import StrEnum
from typing import Final, FrozenSet, Iterable, Union, cast

class Source(StrEnum):
    """Canonical face-observation sources."""
    DETECTED = "detected"
    TRACKED  = "tracked"
    FLOW     = "flow"
    FALLBACK = "fallback"

# Valid set
VALID_SOURCES = {Source.DETECTED, Source.TRACKED, Source.FLOW}

# Integer codes used by ObservationsCollector sidecar format.
SRC_TO_CODE = {
    Source.DETECTED: 0,
    Source.TRACKED:  1,
    Source.FLOW:     2,
}

CODE_TO_SRC = {v: k for k, v in SRC_TO_CODE.items()}