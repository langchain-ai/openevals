from openevals.string.embedding_similarity import (
    create_embedding_similarity_evaluator,
    create_async_embedding_similarity_evaluator,
)
from langsmith import Client
import pytest


@pytest.mark.langsmith
def test_embedding_similarity():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 2}
    embedding_similarity = create_embedding_similarity_evaluator()
    res = embedding_similarity(outputs=outputs, reference_outputs=reference_outputs)
    assert res["key"] == "embedding_similarity"
    assert res["score"] == 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
def test_embedding_similarity_with_different_values():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    embedding_similarity = create_embedding_similarity_evaluator()
    res = embedding_similarity(outputs=outputs, reference_outputs=reference_outputs)
    assert res["key"] == "embedding_similarity"
    assert res["score"] < 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_embedding_similarity_async():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 2}
    embedding_similarity = create_async_embedding_similarity_evaluator()
    res = await embedding_similarity(
        outputs=outputs, reference_outputs=reference_outputs
    )
    assert res["key"] == "embedding_similarity"
    assert res["score"] == 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_embedding_similarity_with_different_values_async():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    embedding_similarity = create_async_embedding_similarity_evaluator()
    res = await embedding_similarity(
        outputs=outputs, reference_outputs=reference_outputs
    )
    assert res["key"] == "embedding_similarity"
    assert res["score"] < 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
def test_embedding_similarity_evaluate():
    client = Client()
    evaluator = create_embedding_similarity_evaluator()
    res = client.evaluate(lambda x: x, data="json", evaluators=[evaluator])
    for r in res:
        assert r["evaluation_results"]["results"][0].score is not None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_embedding_similarity_aevaluate():
    client = Client()

    async def target(inputs):
        return inputs

    evaluator = create_async_embedding_similarity_evaluator()
    res = await client.aevaluate(target, data="json", evaluators=[evaluator])
    async for r in res:
        assert r["evaluation_results"]["results"][0].score is not None
