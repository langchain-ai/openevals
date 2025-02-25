from openevals.exact import exact_match, exact_match_async
from openevals.types import EvaluatorResult
from langsmith import Client
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


@pytest.mark.langsmith
def test_exact_evaluate():
    client = Client()
    evaluator = exact_match
    res = client.evaluate(lambda x: x, data="json", evaluators=[evaluator])
    for r in res:
        assert r["evaluation_results"]["results"][0].score is not None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_exact_aevaluate():
    client = Client()

    async def target(inputs):
        return inputs

    evaluator = exact_match_async
    res = await client.aevaluate(target, data="json", evaluators=[evaluator])
    async for r in res:
        assert r["evaluation_results"]["results"][0].score is not None
