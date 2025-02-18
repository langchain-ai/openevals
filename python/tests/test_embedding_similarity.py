from evaluators.string.embedding_similarity import (
    create_embedding_similarity_evaluator,
    create_async_embedding_similarity_evaluator,
)

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
