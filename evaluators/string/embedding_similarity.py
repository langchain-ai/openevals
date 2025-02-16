from evaluators.types import EvaluatorResult, SimpleEvaluator
from evaluators.utils import _run_evaluator

from typing import Any


def create_embedding_similarity_evaluator(
    *, model: str = "openai:text-embedding-3-small", algorithm: str = "cosine"
) -> SimpleEvaluator:
    """
    Create an evaluator that compares the actual output and reference output for similarity by text embedding distance.

    Args:
        model (str): The model to use for embedding similarity
        algorithm (str): The algorithm to use for embedding similarity. Defaults to 'cosine'. 'dot_product' is also supported.

    Returns:
        EvaluatorResult: A score representing the embedding similarity
    """

    def wrapped_evaluator(
        *, outputs: str, reference_outputs: str, **kwargs: Any
    ) -> EvaluatorResult:
        def get_score():
            if outputs is None or reference_outputs is None:
                raise ValueError(
                    "Embedding similarity requires both outputs and reference_outputs"
                )
            from langchain.embeddings import init_embeddings

            embeddings = init_embeddings(model)
            received_embedding = embeddings.embed_query(outputs)
            expected_embedding = embeddings.embed_query(reference_outputs)

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
            elif algorithm == "dot_product":
                similarity = dot_product(received_embedding, expected_embedding)
            else:
                raise ValueError(
                    f"Unsupported algorithm: {algorithm}. Only 'cosine' and 'dot_product' are supported."
                )

            return similarity

        return _run_evaluator(
            run_name="embedding_similarity",
            scorer=get_score,
            feedback_key="embedding_similarity",
        )

    return wrapped_evaluator
