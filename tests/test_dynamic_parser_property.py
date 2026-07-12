"""Property-based tests for the LLM-output JSON parser.

`_extract_first_json_object` walks the input character by character to
isolate the first balanced top-level `{...}` block. The function lives
on the hot path between the LLM and `validate_generated_tool_code`, so
a single edge case (escaped quotes, braces in strings, nested objects)
that fools the parser ends up writing untrusted code to disk.

We use `hypothesis` to generate any JSON object, wrap it in arbitrary
prose noise, and assert two invariants:

  1. The extractor returns a non-empty string that itself parses as
     valid JSON (no false positives that break downstream parsing).
  2. The parsed value equals the original generated object (no silent
     truncation).
"""

from __future__ import annotations

import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from autoagent.dynamic import _extract_first_json_object, _parse_json_object

# JSON-compatible scalar strategy. Avoids NaN/Infinity since stdlib json
# does not round-trip those by default.
_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(min_size=0, max_size=20),
)


def _json_objects(depth: int = 2) -> st.SearchStrategy[dict]:
    """A recursive JSON-object strategy bounded in depth."""
    if depth <= 0:
        return st.dictionaries(st.text(min_size=1, max_size=10), _json_scalars, max_size=4)
    children = st.one_of(_json_scalars, _json_objects(depth - 1))
    return st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=4)


# Prose noise that the LLM might wrap around the JSON. Must NOT contain
# `{` or `}` so it cannot accidentally start a competing JSON block.
_prose = st.text(alphabet=st.characters(blacklist_characters="{}\\"), max_size=40)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(obj=_json_objects(depth=2), prefix=_prose, suffix=_prose)
def test_extractor_finds_object_inside_arbitrary_prose(obj: dict, prefix: str, suffix: str) -> None:
    encoded = json.dumps(obj, ensure_ascii=False)
    text = f"{prefix}{encoded}{suffix}"
    extracted = _extract_first_json_object(text)
    assert extracted is not None, f"Extractor missed the JSON block in {text!r}"
    # Must be valid JSON on its own.
    reparsed = json.loads(extracted)
    assert reparsed == obj, f"Round-trip mismatch: {reparsed!r} != {obj!r}"


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(obj=_json_objects(depth=2))
def test_parse_json_object_roundtrip(obj: dict) -> None:
    encoded = json.dumps(obj, ensure_ascii=False)
    assert _parse_json_object(encoded) == obj


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(obj=_json_objects(depth=2), prefix=_prose, suffix=_prose)
def test_parse_extracts_from_prose(obj: dict, prefix: str, suffix: str) -> None:
    encoded = json.dumps(obj, ensure_ascii=False)
    text = f"{prefix}{encoded}{suffix}"
    assert _parse_json_object(text) == obj
