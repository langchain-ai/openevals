import json
from openevals.types import (
    EvaluatorResult,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
)
from openevals.utils import _run_evaluator, _arun_evaluator
from langchain.embeddings import init_embeddings

from typing import Any


def _handle_embedding_outputs(
    algorithm: str, received_embedding: list[float], expected_embedding: list[float]
) -> float:
    def dot_product(v1, v2):
        return sum(a * b for a, b in zip(v1, v2))

    def vector_magnitude(v):
        return (sum(x * x for x in v)) ** 0.5

    def cosine_similarity(v1, v2):
        dot_prod = dot_product(v1, v2)
        magnitude1 = vector_magnitude(v1)
        magnitude2 = vector_magnitude(v2)
        return dot_prod / (magnitude1 * magnitude2)

    # Calculate similarity based on chosen algorithm
    if algorithm == "cosine":
        similarity = cosine_similarity(received_embedding, expected_embedding)
    else:
        similarity = dot_product(received_embedding, expected_embedding)
    return round(similarity, 2)


def create_embedding_similarity_evaluator(
    *, model: str = "openai:text-embedding-3-small", algorithm: str = "cosine"
) -> SimpleEvaluator:
    """
    Create an evaluator that compares outputs and reference outputs for similarity by text embedding distance.

    Args:
        model (str): The model to use for embedding similarity
        algorithm (str): The algorithm to use for embedding similarity. Defaults to 'cosine'. 'dot_product' is also supported.

    Returns:
        EvaluatorResult: A score representing the embedding similarity
    """

    if algorithm != "cosine" and algorithm != "dot_product":
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. Only 'cosine' and 'dot_product' are supported."
        )

    def wrapped_evaluator(
        *, outputs: Any, reference_outputs: Any, **kwargs: Any
    ) -> EvaluatorResult:
        if outputs is None or reference_outputs is None:
            raise ValueError(
                "Embedding similarity requires both outputs and reference_outputs"
            )

        if not isinstance(outputs, str):
            outputs = json.dumps(outputs)
        if not isinstance(reference_outputs, str):
            reference_outputs = json.dumps(reference_outputs)

        def get_score():
            embeddings = init_embeddings(model)
            received_embedding = embeddings.embed_query(outputs)
            expected_embedding = embeddings.embed_query(reference_outputs)

            similarity = _handle_embedding_outputs(
                algorithm, received_embedding, expected_embedding
            )

            return similarity

        res = _run_evaluator(
            run_name="embedding_similarity",
            scorer=get_score,
            feedback_key="embedding_similarity",
        )
        if isinstance(res, list):
            return res[0]
        return res

    return wrapped_evaluator  # type: ignore


def create_async_embedding_similarity_evaluator(
    *, model: str = "openai:text-embedding-3-small", algorithm: str = "cosine"
) -> SimpleAsyncEvaluator:
    """
    Create an evaluator that compares the actual output and reference output for similarity by text embedding distance.

    Args:
        model (str): The model to use for embedding similarity
        algorithm (str): The algorithm to use for embedding similarity. Defaults to 'cosine'. 'dot_product' is also supported.

    Returns:
        EvaluatorResult: A score representing the embedding similarity
    """
    if algorithm != "cosine" and algorithm != "dot_product":
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. Only 'cosine' and 'dot_product' are supported."
        )

    async def wrapped_evaluator(
        *, outputs: Any, reference_outputs: Any, **kwargs: Any
    ) -> EvaluatorResult:
        if outputs is None or reference_outputs is None:
            raise ValueError(
                "Embedding similarity requires both outputs and reference_outputs"
            )

        if not isinstance(outputs, str):
            outputs = json.dumps(outputs)
        if not isinstance(reference_outputs, str):
            reference_outputs = json.dumps(reference_outputs)

        async def get_score():
            from langchain.embeddings import init_embeddings

            embeddings = init_embeddings(model)
            received_embedding = await embeddings.aembed_query(outputs)
            expected_embedding = await embeddings.aembed_query(reference_outputs)

            similarity = _handle_embedding_outputs(
                algorithm, received_embedding, expected_embedding
            )

            return similarity

        res = await _arun_evaluator(
            run_name="embedding_similarity",
            scorer=get_score,
            feedback_key="embedding_similarity",
        )
        if isinstance(res, list):
            return res[0]
        return res

    return wrapped_evaluator  # type: ignore
