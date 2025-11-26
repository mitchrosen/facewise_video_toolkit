from enum import StrEnum

class Source(StrEnum):
    """Canonical face-observation sources."""
    DETECTED = "detected"
    TRACKED  = "tracked"
    FLOW     = "flow"
    FALLBACK = "fallback"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"

    def __str__(self) -> str:  # makes, for example, print(member) -> "detected" 
        return self.value


SRC_TO_CODE = {
    Source.DETECTED:     0,
    Source.TRACKED:      1,
    Source.FLOW:         2,
    Source.FALLBACK:     3,
    Source.INTERPOLATED: 4,
    Source.EXTRAPOLATED: 5,
}
CODE_TO_SRC = {v: k for (k, v) in SRC_TO_CODE.items()}

def src_to_code(src: "Source|str") -> int:
    """
    Accept a Source enum or value-string ('detected'|'tracked'|'flow'|'fallback').
    Return the canonical int code.
    """
    if isinstance(src, Source):
        return SRC_TO_CODE[src]
    if isinstance(src, str):
        try:
            # Normalize to value, then index via enum → code
            return SRC_TO_CODE[Source(src.lower())]
        except ValueError:
            pass
    raise ValueError(f"Unknown src (expect Source or one of {sorted(SRC_TO_CODE)}): {src!r}")

def code_to_src(code: int) -> Source:
    """
    Accept an int code and return the Source enum.
    """
    try:
        return CODE_TO_SRC[int(code)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Unknown src code (expect one of {sorted(CODE_TO_SRC)}): {code!r}")

def ensure_src_code(src_any) -> int:
    """
    Normalize any accepted src representation to an int code.
    Accepted: int, Source, str ("detected"/"tracked"/"flow"/"fallback").
    """
    if isinstance(src_any, int):
        return int(src_any)
    try:
        return int(src_to_code(src_any if isinstance(src_any, str) else str(src_any)))
    except Exception as e:
        raise TypeError(f"Bad src: expected int|str|Source, got {src_any!r}") from e