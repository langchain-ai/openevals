"""Jev-as-a-judge evaluators using TypeSafe's System One model.

Jev evaluates state against typed questions (noul, score, choice)
and returns calibrated probabilities — faster and cheaper than
LLM-as-a-judge for bounded classification tasks.

Requires ``typesafe-sdk``: ``pip install typesafe-sdk``
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from openevals.types import EvaluatorResult
from openevals.utils import _arun_evaluator, _run_evaluator


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _build_state(
    *,
    inputs: Any = None,
    outputs: Any = None,
    reference_outputs: Any = None,
    state_builder: Callable[..., str] | None = None,
    **kwargs: Any,
) -> str:
    if state_builder is not None:
        return state_builder(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )
    parts = []
    if inputs is not None:
        parts.append(f"[Inputs]\n{_stringify(inputs)}")
    if outputs is not None:
        parts.append(f"[Outputs]\n{_stringify(outputs)}")
    if reference_outputs is not None:
        parts.append(f"[Reference Outputs]\n{_stringify(reference_outputs)}")
    for k, v in kwargs.items():
        if v is not None:
            parts.append(f"[{k}]\n{_stringify(v)}")
    return "\n\n".join(parts)


def _parse_results(
    response: Any,
    questions: dict,
) -> dict[str, dict]:
    """Extract score dicts per question from the Jev response.

    Returns a dict of ``{name: {"score": ..., "reasoning": ..., "metadata": ...}}``
    compatible with openevals ``_process_score``.
    """
    results: dict[str, dict] = {}
    for name in questions:
        answer = response.answers.get(name)
        if answer is None:
            results[name] = {
                "score": False,
                "reasoning": f"No answer for question '{name}'",
            }
            continue

        answer_type = answer.type
        if answer_type == "noul":
            prob = answer.noul
            results[name] = {
                "score": prob,
                "metadata": {"probability": prob, "type": "noul"},
            }
        elif answer_type == "choice":
            chosen = answer.choice
            conf = answer.confidence
            probs = dict(answer.probabilities) if answer.probabilities else {}
            results[name] = {
                "score": conf,
                "reasoning": f"choice={chosen}",
                "metadata": {
                    "choice": chosen,
                    "confidence": conf,
                    "probabilities": probs,
                    "type": "choice",
                },
            }
        elif answer_type == "score":
            score_val = answer.score
            conf = answer.confidence
            probs = dict(answer.probabilities) if answer.probabilities else {}
            results[name] = {
                "score": score_val,
                "metadata": {
                    "score": score_val,
                    "confidence": conf,
                    "probabilities": probs,
                    "type": "score",
                },
            }
        else:
            results[name] = {
                "score": False,
                "reasoning": f"Unknown answer type: {answer_type}",
            }

    return results


def create_jev_evaluator(
    *,
    questions: dict[str, Any],
    model: str = "jev-latest",
    api_key: str | None = None,
    base_url: str | None = None,
    state_builder: Callable[..., str] | None = None,
    feedback_key: str | None = None,
) -> Callable[..., EvaluatorResult | list[EvaluatorResult]]:
    """Create an evaluator that uses Jev (TypeSafe System One) as a judge.

    Unlike LLM-as-judge, Jev returns typed answers with calibrated probabilities
    instead of generating text. All questions are evaluated in parallel in a
    single request.

    Args:
        questions: Mapping of feedback key names to Jev question objects
            (``Noul``, ``Score``, or ``Choice`` from ``typesafe_sdk``),
            or plain dicts matching the TypeSafe API schema.
        model: Jev model name. Defaults to ``"jev-latest"``.
        api_key: TypeSafe API key. Falls back to ``TYPESAFE_API_KEY`` env var.
        base_url: Optional base URL override.
        state_builder: Optional callable that builds the Jev state string from
            ``inputs``, ``outputs``, ``reference_outputs``, and extra kwargs.
            If not provided, a default builder concatenates all non-None params.
        feedback_key: Override the feedback key prefix. If ``None``, each
            question name becomes its own feedback key.

    Returns:
        An evaluator function compatible with OpenEvals and LangSmith.

    Example::

        from typesafe_sdk import Noul, Score
        from openevals.jev import create_jev_evaluator

        evaluator = create_jev_evaluator(
            questions={
                "correctness": Noul(instructions="Is the output factually correct?"),
                "helpfulness": Score(criteria=["Unhelpful", "Somewhat helpful", "Very helpful"]),
            },
        )
        result = evaluator(inputs="What is 2+2?", outputs="4")
    """
    try:
        from typesafe_sdk import TypeSafeClient
    except ImportError:
        raise ImportError(
            "typesafe-sdk is required for Jev evaluators. "
            "Install it with: pip install typesafe-sdk"
        )

    client_kwargs: dict[str, Any] = {}
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    client = TypeSafeClient(**client_kwargs)

    def evaluator(
        *,
        inputs: Any | None = None,
        outputs: Any,
        reference_outputs: Any | None = None,
        **kwargs: Any,
    ) -> EvaluatorResult | list[EvaluatorResult]:
        state = _build_state(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            state_builder=state_builder,
            **kwargs,
        )

        def scorer():
            response = client.system_one(
                state=state,
                questions=questions,
                model=model,
            )
            return _parse_results(response, questions)

        return _run_evaluator(
            run_name="jev_evaluator",
            scorer=scorer,
            feedback_key=feedback_key or "jev",
        )

    return evaluator


def create_async_jev_evaluator(
    *,
    questions: dict[str, Any],
    model: str = "jev-latest",
    api_key: str | None = None,
    base_url: str | None = None,
    state_builder: Callable[..., str] | None = None,
    feedback_key: str | None = None,
) -> Callable[..., Any]:
    """Async version of :func:`create_jev_evaluator`."""
    try:
        from typesafe_sdk import AsyncTypeSafeClient
    except ImportError:
        raise ImportError(
            "typesafe-sdk is required for Jev evaluators. "
            "Install it with: pip install typesafe-sdk"
        )

    client_kwargs: dict[str, Any] = {}
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    client = AsyncTypeSafeClient(**client_kwargs)

    async def evaluator(
        *,
        inputs: Any | None = None,
        outputs: Any,
        reference_outputs: Any | None = None,
        **kwargs: Any,
    ) -> EvaluatorResult | list[EvaluatorResult]:
        state = _build_state(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            state_builder=state_builder,
            **kwargs,
        )

        async def scorer():
            response = await client.system_one(
                state=state,
                questions=questions,
                model=model,
            )
            return _parse_results(response, questions)

        return await _arun_evaluator(
            run_name="jev_evaluator",
            scorer=scorer,
            feedback_key=feedback_key or "jev",
        )

    return evaluator
