from evaluators.exact import exact_match, exact_match_async
from evaluators.types import EvaluatorResult

import pytest


@pytest.mark.langsmith
def test_exact_matcher():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    assert exact_match(inputs=inputs, outputs=outputs) == EvaluatorResult(
        key="exact_match", score=True, comment=None
    )


@pytest.mark.langsmith
def test_exact_matcher_with_different_values():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 3}
    assert exact_match(inputs=inputs, outputs=outputs) == EvaluatorResult(
        key="exact_match", score=False, comment=None
    )


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_exact_matcher_async():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    assert await exact_match_async(inputs=inputs, outputs=outputs) == EvaluatorResult(
        key="exact_match", score=True, comment=None
    )


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_exact_matcher_with_different_values_async():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 3}
    assert await exact_match_async(inputs=inputs, outputs=outputs) == EvaluatorResult(
        key="exact_match", score=False, comment=None
    )
