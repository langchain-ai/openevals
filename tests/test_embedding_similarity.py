from evaluators.string.embedding_similarity import (
    create_embedding_similarity_evaluator,
    create_async_embedding_similarity_evaluator,
)

import pytest


@pytest.mark.langsmith
def test_embedding_similarity():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    embedding_similarity = create_embedding_similarity_evaluator()
    res = embedding_similarity(inputs=inputs, outputs=outputs)
    assert res["key"] == "embedding_similarity"
    assert res["score"] == 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
def test_embedding_similarity_with_different_values():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 3}
    embedding_similarity = create_embedding_similarity_evaluator()
    res = embedding_similarity(inputs=inputs, outputs=outputs)
    assert res["key"] == "embedding_similarity"
    assert res["score"] < 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_embedding_similarity_async():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    embedding_similarity = create_async_embedding_similarity_evaluator()
    res = await embedding_similarity(inputs=inputs, outputs=outputs)
    assert res["key"] == "embedding_similarity"
    assert res["score"] == 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_embedding_similarity_with_different_values_async():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 3}
    embedding_similarity = create_async_embedding_similarity_evaluator()
    res = await embedding_similarity(inputs=inputs, outputs=outputs)
    assert res["key"] == "embedding_similarity"
    assert res["score"] < 1.0
    assert res["comment"] is None
