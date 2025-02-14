__all__ = ["_run_evaluator"]

from langsmith import testing as t
from langsmith.testing._internal import _TEST_CASE
from typing import Any, Callable

from evaluators.types import EvaluatorResult


def _run_evaluator(
    *, run_name: str, scorer: Callable, feedback_key: str, **kwargs: Any
) -> EvaluatorResult | list[EvaluatorResult]:
    reasoning = None
    score = scorer(**kwargs)
    if _TEST_CASE.get():
        with t.trace_feedback(name=run_name):
            if isinstance(score, dict):
                for individual_feedback_key in score:
                    if isinstance(score[individual_feedback_key], dict):
                        if set(score[individual_feedback_key].keys()) == {"score", "reasoning"}:
                            key_score = score[individual_feedback_key]["score"]
                            reasoning = score[individual_feedback_key]["reasoning"]
                        else:
                            raise ValueError(
                                f"Expected a dictionary with keys 'score' and 'reasoning', but got {score[individual_feedback_key]}"
                            )
                    else:
                        key_score = score[individual_feedback_key]
                        reasoning = None
                    t.log_feedback(key=individual_feedback_key, score=key_score, comment=reasoning)
            else:
                if isinstance(score, tuple):
                    score, reasoning = score
                t.log_feedback(key=feedback_key, score=score, comment=reasoning)
    # Always return the feedback, even if not logging
    if isinstance(score, dict):
        results = []
        for individual_feedback_key in score:
            if isinstance(score[individual_feedback_key], dict):
                if set(score[individual_feedback_key].keys()) == {"score", "reasoning"}:
                    key_score = score[individual_feedback_key]["score"]
                    reasoning = score[individual_feedback_key]["reasoning"]
                else:
                    raise ValueError(
                        f"Expected a dictionary with keys 'score' and 'reasoning', but got {score[individual_feedback_key]}"
                    )
            else:
                key_score = score[individual_feedback_key]
                reasoning = None
            results.append(EvaluatorResult(key=individual_feedback_key, score=key_score, comment=reasoning))
        return results
    else:
        if isinstance(score, tuple):
            score, reasoning = score
        return EvaluatorResult(key=feedback_key, score=score, comment=reasoning)
