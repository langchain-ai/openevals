__all__ = ["_run_evaluator"]

from langsmith import testing as t
from langsmith.testing._internal import _TEST_CASE
from typing import Callable

from evaluators.types import EvaluatorResult


def _run_evaluator(
    *, run_name: str, evaluator_fn: Callable, feedback_key: str
) -> EvaluatorResult:
    if _TEST_CASE.get():
        with t.trace_feedback(name=run_name):
            score = evaluator_fn()
            t.log_feedback(key=feedback_key, score=score)
    else:
        score = evaluator_fn()
    return EvaluatorResult(key=feedback_key, score=score)
