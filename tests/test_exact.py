from evaluators.exact import exact_match
from evaluators.types import EvaluatorResult

import pytest


@pytest.mark.langsmith
def test_exact_matcher():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    assert exact_match(inputs=inputs, outputs=outputs) == EvaluatorResult(
        key="exact_match", score=1.0
    )


@pytest.mark.langsmith
def test_exact_matcher_with_different_values():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 3}
    assert exact_match(inputs=inputs, outputs=outputs) == EvaluatorResult(
        key="exact_match", score=0.0
    )
