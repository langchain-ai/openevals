from matchers.base import MatchResult
from matchers.exact import exact_matcher
# import pytest


def test_exact_matcher():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    assert exact_matcher(inputs=inputs, outputs=outputs) == MatchResult(
        key="equal", score=1.0
    )


def test_exact_matcher_with_different_values():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 3}
    assert exact_matcher(inputs=inputs, outputs=outputs) == MatchResult(
        key="equal", score=0.0
    )
