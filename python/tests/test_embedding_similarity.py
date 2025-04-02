from openevals.string.embedding_similarity import (
    create_embedding_similarity_evaluator,
    create_async_embedding_similarity_evaluator,
)
from langsmith import Client
import pytest


@pytest.mark.langsmith
def test_embedding_similarity_documents():
    evaluator = create_embedding_similarity_evaluator()

    inputs = "Where was the first president of FoobarLand born?"

    context = "\n".join(
        [
            "BazQuxLand is a new country located on the dark side of the moon",
            "Space dolphins are native to BazQuxLand",
            "BazQuxLand is a constitutional democracy whose first president was Bagatur Askaryan",
            "The current weather in BazQuxLand is 80 degrees and clear.",
        ]
    )

    result = evaluator(
        outputs=context,
        reference_outputs=inputs,
    )

    print(result)


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
