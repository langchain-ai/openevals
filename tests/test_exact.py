from scorers.base import ScorerResult
from scorers.exact import exact_match
# import pytest


def test_exact_matcher():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    assert exact_match(inputs=inputs, outputs=outputs) == ScorerResult(
        key="equal", score=1.0
    )


def test_exact_matcher_with_different_values():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 3}
    assert exact_match(inputs=inputs, outputs=outputs) == ScorerResult(
        key="equal", score=0.0
    )
