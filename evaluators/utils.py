__all__ = ["_run_evaluator"]

from langsmith import testing as t
from langsmith.testing._internal import _TEST_CASE
from typing import Any, Callable

from evaluators.types import EvaluatorResult


def _run_evaluator(
    *, run_name: str, scorer: Callable, feedback_key: str, **kwargs: Any
) -> EvaluatorResult | list[EvaluatorResult]:
    reasoning = None
    if _TEST_CASE.get():
        with t.trace_feedback(name=run_name):
            score = scorer(**kwargs)
            if isinstance(score, dict):
                for feedback_key in score:
                    t.log_feedback(key=feedback_key, score=score[feedback_key])
            else:
                if isinstance(score, tuple):
                    score, reasoning = score
                t.log_feedback(key=feedback_key, score=score, comment=reasoning)
    else:
        score = scorer(**kwargs)
    
        if isinstance(score, dict):
            results = []
            for feedback_key in score:
                results.append(EvaluatorResult(key=feedback_key, score=score[feedback_key]))
            return results
        else:
            if isinstance(score, tuple):
                score, reasoning = score
            return EvaluatorResult(key=feedback_key, score=score, comment=reasoning)
