from evaluators.exact import exact_match, exact_match_async
from evaluators.types import EvaluatorResult

import pytest


@pytest.mark.langsmith
def test_exact_matcher():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 2}
    assert exact_match(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key="exact_match", score=True, comment=None)


@pytest.mark.langsmith
def test_exact_matcher_with_different_values():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    assert exact_match(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key="exact_match", score=False, comment=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_exact_matcher_async():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 2}
    assert await exact_match_async(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key="exact_match", score=True, comment=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_exact_matcher_with_different_values_async():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    assert await exact_match_async(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key="exact_match", score=False, comment=None)
