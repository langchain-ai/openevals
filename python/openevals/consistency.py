"""Label-free consistency evaluators: how much do N runs of the same input agree?

These evaluators take the outputs of several runs of one input and need no reference
outputs. They are useful to check run-to-run reliability, and to check that nothing changed
after upgrading a model, a prompt or a retriever.

Consistency is not correctness: a system that is consistently wrong scores 1.0.

The pair sampling, similarity measures and judge rubric come from the ``repeatability-evals``
package (``pip install "repeatability-evals>=0.3.0"``), imported when an evaluator is created.
"""

import json
from typing import Any, Callable, Literal, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel

from openevals.types import (
    EvaluatorResult,
    SimpleAsyncEvaluator,
    SimpleEvaluator,
)
from openevals.utils import _arun_evaluator, _run_evaluator


def _lib() -> Any:
    try:
        import repeatability_evals
    except ImportError as e:
        raise ImportError(
            "Consistency evaluators need the `repeatability-evals` package. "
            'Install it with `pip install "openevals[consistency]"`.'
        ) from e
    if not hasattr(repeatability_evals, "render_prompt"):
        raise ImportError(
            "Consistency evaluators need repeatability-evals>=0.3.0. "
            'Upgrade with `pip install -U "repeatability-evals>=0.3.0"`.'
        )
    return repeatability_evals


def _runs(outputs: Any) -> list:
    if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
        raise ValueError(
            "Consistency evaluators require `outputs` to be a list of at least two "
            "runs of the same input."
        )
    return list(outputs)


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _query(inputs: Any) -> str:
    return "" if inputs is None else _as_text(inputs)


def _sampler(lib: Any, max_comparisons: Optional[int]) -> Any:
    return lib.CycleSampler(budget=max_comparisons)


def _results(feedback_key: str, result: Any) -> dict:
    """Headline score plus, when the judge scored several dimensions, one score per dimension."""
    out: dict = {
        feedback_key: {
            "score": result.score,
            "reasoning": _comment(result),
            "metadata": {
                "n_runs": result.n_items,
                "n_comparisons": result.n_pairs,
                "min_pair_score": result.min_score,
            },
        }
    }
    for name, value in (result.details.get("dimensions") or {}).items():
        out[f"{feedback_key}.{name}"] = value
    return out


def _comment(result: Any) -> Optional[str]:
    if not result.pairs:
        return None
    worst = min(result.pairs, key=lambda p: p.score)
    text = (
        f"Compared {result.n_pairs} pairs of {result.n_items} runs; most different "
        f"pair is runs {worst.i + 1} and {worst.j + 1} ({worst.score:.2f})"
    )
    return f"{text}: {worst.reason}" if worst.reason else f"{text}."


# --------------------------------------------------------------------------- retrieval
def _retrieval_similarity(lib: Any, mode: str, ngram_size: int) -> Any:
    if mode == "id":
        return lib.JaccardSimilarity()
    if mode == "ngram":
        return lib.NgramJaccard(n=ngram_size)
    raise ValueError(f"Unsupported mode: {mode}. Use 'id' or 'ngram'.")


def _retrieval_runs(outputs: Any) -> list:
    return [
        [run] if isinstance(run, str) else [_as_text(c) for c in run]
        for run in _runs(outputs)
    ]


def create_retrieval_consistency_evaluator(
    *,
    mode: Literal["id", "ngram"] = "ngram",
    ngram_size: int = 3,
    max_comparisons: Optional[int] = None,
    feedback_key: str = "retrieval_consistency",
) -> SimpleEvaluator:
    """
    Create an evaluator that measures how consistently a retriever returns the same evidence
    across repeated runs of the same query. Needs no relevance labels and no LLM.

    `outputs` must be a list of runs; each run is the list of chunks that retrieval returned
    (chunk IDs for `mode="id"`, chunk texts for `mode="ngram"`). Order is ignored.

    Args:
        mode: "id" compares sets of chunk IDs with Jaccard similarity. "ngram" compares the
            word n-grams of the pooled chunk text, which survives re-chunking and unstable IDs.
        ngram_size: n-gram length for `mode="ngram"`.
        max_comparisons: number of run pairs to compare. Defaults to the number of runs
            (a random cycle, so every run is compared at least twice).
        feedback_key: key of the returned result.

    Returns:
        EvaluatorResult: the mean similarity over compared pairs, in [0, 1] (1 = identical).
    """
    lib = _lib()
    metric = lib.PairwiseConsistency(
        _retrieval_similarity(lib, mode, ngram_size),
        sampler=_sampler(lib, max_comparisons),
    )

    def wrapped_evaluator(
        *, inputs: Optional[Any] = None, outputs: Any, **kwargs: Any
    ) -> EvaluatorResult:
        runs = _retrieval_runs(outputs)

        def get_score(**kwargs: Any) -> dict:
            return _results(feedback_key, metric.evaluate(runs))

        res = _run_evaluator(
            run_name="retrieval_consistency",
            scorer=get_score,
            feedback_key=feedback_key,
            inputs=inputs,
            outputs=outputs,
        )
        return res[0] if isinstance(res, list) else res

    return wrapped_evaluator  # type: ignore


def create_async_retrieval_consistency_evaluator(
    *,
    mode: Literal["id", "ngram"] = "ngram",
    ngram_size: int = 3,
    max_comparisons: Optional[int] = None,
    feedback_key: str = "retrieval_consistency",
) -> SimpleAsyncEvaluator:
    """Async version of `create_retrieval_consistency_evaluator`."""
    lib = _lib()
    metric = lib.PairwiseConsistency(
        _retrieval_similarity(lib, mode, ngram_size),
        sampler=_sampler(lib, max_comparisons),
    )

    async def wrapped_evaluator(
        *, inputs: Optional[Any] = None, outputs: Any, **kwargs: Any
    ) -> EvaluatorResult:
        runs = _retrieval_runs(outputs)

        async def get_score(**kwargs: Any) -> dict:
            return _results(feedback_key, await metric.aevaluate(runs))

        res = await _arun_evaluator(
            run_name="retrieval_consistency",
            scorer=get_score,
            feedback_key=feedback_key,
            inputs=inputs,
            outputs=outputs,
        )
        return res[0] if isinstance(res, list) else res

    return wrapped_evaluator  # type: ignore


# --------------------------------------------------------------------------------- answers
def _chat_model(judge: Optional[BaseChatModel], model: Optional[str]) -> BaseChatModel:
    if judge is not None:
        if not isinstance(judge, BaseChatModel):
            raise TypeError("`judge` must be a LangChain BaseChatModel.")
        return judge
    if model is None:
        raise ValueError("Provide either `judge` or `model`.")
    from langchain.chat_models import init_chat_model

    return init_chat_model(model)


def _content_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return "".join(
        part if isinstance(part, str) else part.get("text", "") for part in content
    )


def _answer_metric(
    judge: Optional[BaseChatModel], model: Optional[str], max_comparisons: Optional[int]
) -> Any:
    lib = _lib()
    chat = _chat_model(judge, model)
    rubric = lib.RubricJudge(
        lambda prompt: _content_text(chat.invoke(prompt)),
        allm=_async_complete(chat),
    )
    return lib.PairwiseConsistency(rubric, sampler=_sampler(lib, max_comparisons))


def _async_complete(chat: BaseChatModel) -> Callable[[str], Any]:
    async def complete(prompt: str) -> str:
        return _content_text(await chat.ainvoke(prompt))

    return complete


def create_answer_consistency_evaluator(
    *,
    judge: Optional[BaseChatModel] = None,
    model: Optional[str] = None,
    max_comparisons: Optional[int] = None,
    feedback_key: str = "answer_consistency",
) -> SimpleEvaluator:
    """
    Create an evaluator that measures how consistently a system answers the same input across
    repeated runs. Needs no reference outputs: an LLM judge compares pairs of runs.

    Identical runs (ignoring case and whitespace) are scored 1.0 without a judge call, and
    replies the judge formats badly are retried.

    `outputs` must be a list of at least two runs. `inputs` (optional) is the input the runs
    answered, shown to the judge. For each pair the judge scores five dimensions of agreement
    from 1 to 5 (core facts, specifics, supporting detail, stance, format); the headline score
    is their weighted mean rescaled to [0, 1]. The mean score per dimension is also returned
    under `"<feedback_key>.<dimension>"` so you can see *what* varies between runs.

    Comparing all N(N-1)/2 pairs is expensive, so by default a random cycle of N pairs is judged
    (every run is compared at least twice).

    Args:
        judge: A LangChain chat model used as the judge.
        model: A model string for `init_chat_model`, used if `judge` is not given.
        max_comparisons: number of run pairs to judge. Defaults to the number of runs.
        feedback_key: key of the headline result.

    Returns:
        EvaluatorResult: the mean agreement in [0, 1] (1 = identical). Returned as a list when
        per-dimension results are included.
    """
    metric = _answer_metric(judge, model, max_comparisons)

    def wrapped_evaluator(
        *, inputs: Optional[Any] = None, outputs: Any, **kwargs: Any
    ) -> Union[EvaluatorResult, list[EvaluatorResult]]:
        runs = [_as_text(r) for r in _runs(outputs)]

        def get_score(**kwargs: Any) -> dict:
            result = metric.evaluate(runs, context={"query": _query(inputs)})
            return _results(feedback_key, result)

        return _run_evaluator(
            run_name="answer_consistency",
            scorer=get_score,
            feedback_key=feedback_key,
            inputs=inputs,
            outputs=outputs,
        )

    return wrapped_evaluator  # type: ignore


def create_async_answer_consistency_evaluator(
    *,
    judge: Optional[BaseChatModel] = None,
    model: Optional[str] = None,
    max_comparisons: Optional[int] = None,
    feedback_key: str = "answer_consistency",
) -> SimpleAsyncEvaluator:
    """Async version of `create_answer_consistency_evaluator`."""
    metric = _answer_metric(judge, model, max_comparisons)

    async def wrapped_evaluator(
        *, inputs: Optional[Any] = None, outputs: Any, **kwargs: Any
    ) -> Union[EvaluatorResult, list[EvaluatorResult]]:
        runs = [_as_text(r) for r in _runs(outputs)]

        async def get_score(**kwargs: Any) -> dict:
            result = await metric.aevaluate(runs, context={"query": _query(inputs)})
            return _results(feedback_key, result)

        return await _arun_evaluator(
            run_name="answer_consistency",
            scorer=get_score,
            feedback_key=feedback_key,
            inputs=inputs,
            outputs=outputs,
        )

    return wrapped_evaluator  # type: ignore
