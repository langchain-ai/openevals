import hashlib

from openevals.string.embedding_similarity import (
    create_embedding_similarity_evaluator,
    create_async_embedding_similarity_evaluator,
)
from langchain_core.embeddings import Embeddings
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


def test_embedding_similarity_evaluate():
    client = Client()
    evaluator = create_embedding_similarity_evaluator()
    res = client.evaluate(lambda x: x, data="json", evaluators=[evaluator])
    for r in res:
        assert r["evaluation_results"]["results"][0].score is not None


@pytest.mark.asyncio
async def test_embedding_similarity_aevaluate():
    client = Client()

    async def target(inputs):
        return inputs

    evaluator = create_async_embedding_similarity_evaluator()
    res = await client.aevaluate(target, data="json", evaluators=[evaluator])
    async for r in res:
        assert r["evaluation_results"]["results"][0].score is not None


class _HashEmbeddings(Embeddings):
    """Offline embeddings: identical text maps to an identical vector."""

    def embed_query(self, text: str) -> list[float]:
        return [float(b) for b in hashlib.sha256(text.encode()).digest()]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


@pytest.fixture
def fake_embeddings(monkeypatch):
    def fake_init_embeddings(model):
        return _HashEmbeddings()

    monkeypatch.setattr(
        "openevals.string.embedding_similarity.init_embeddings", fake_init_embeddings
    )
    monkeypatch.setattr("langchain.embeddings.init_embeddings", fake_init_embeddings)


def test_embedding_similarity_ignores_key_order(fake_embeddings):
    outputs = {"city": "Paris", "unit": "C", "details": {"lat": 48.85, "lon": 2.35}}
    reference_outputs = {
        "unit": "C",
        "details": {"lon": 2.35, "lat": 48.85},
        "city": "Paris",
    }
    embedding_similarity = create_embedding_similarity_evaluator()
    res = embedding_similarity(outputs=outputs, reference_outputs=reference_outputs)
    assert res["key"] == "embedding_similarity"
    assert res["score"] == 1.0


@pytest.mark.asyncio
async def test_embedding_similarity_ignores_key_order_async(fake_embeddings):
    outputs = {"city": "Paris", "unit": "C", "details": {"lat": 48.85, "lon": 2.35}}
    reference_outputs = {
        "unit": "C",
        "details": {"lon": 2.35, "lat": 48.85},
        "city": "Paris",
    }
    embedding_similarity = create_async_embedding_similarity_evaluator()
    res = await embedding_similarity(
        outputs=outputs, reference_outputs=reference_outputs
    )
    assert res["key"] == "embedding_similarity"
    assert res["score"] == 1.0
