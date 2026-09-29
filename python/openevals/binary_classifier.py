import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from openevals.types import EvaluatorResult, SimpleAsyncEvaluator, SimpleEvaluator
from openevals.utils import _arun_evaluator, _run_evaluator

_DEFAULT_POSITIVE_LABELS: list[str] = ["good", "pass", "true", "1"]
_DEFAULT_NEGATIVE_LABELS: list[str] = ["bad", "fail", "false", "0"]


def _normalize_label(
    label: Any,
    positive_labels: list[Any],
    negative_labels: list[Any],
) -> bool:
    if isinstance(label, bool):
        return label

    normalized_positive = {str(item).strip().lower() for item in positive_labels}
    normalized_negative = {str(item).strip().lower() for item in negative_labels}

    if isinstance(label, (int, float)) and float(label).is_integer():
        normalized_label = str(int(label))
    else:
        normalized_label = str(label).strip().lower()

    if normalized_label in normalized_positive:
        return True
    if normalized_label in normalized_negative:
        return False
    raise ValueError(
        f"Classifier returned an unrecognized label: {label!r}. Expected one of "
        f"{sorted(normalized_positive)} (positive) or {sorted(normalized_negative)} "
        "(negative). Pass `positive_labels`/`negative_labels` to support custom labels."
    )


def _validate_disjoint_labels(
    positive_labels: list[Any], negative_labels: list[Any]
) -> None:
    normalized_positive = {str(item).strip().lower() for item in positive_labels}
    normalized_negative = {str(item).strip().lower() for item in negative_labels}
    overlap = normalized_positive & normalized_negative
    if overlap:
        raise ValueError(
            f"`positive_labels` and `negative_labels` overlap: {sorted(overlap)}. "
            "Each label must be classified as either positive or negative, not both."
        )


def create_binary_classifier_evaluator(
    *,
    classifier: Callable[..., Any],
    feedback_key: str = "binary_classification",
    positive_labels: list[Any] | None = None,
    negative_labels: list[Any] | None = None,
) -> SimpleEvaluator:
    """
    Create an evaluator that wraps a user-provided classifier function and turns its
    label into a standard OpenEvals boolean score.

    This is not an LLM-as-judge: `classifier` can be a deterministic rule, a call out
    to your own model, or anything else that returns a label. It is normalized against
    `positive_labels`/`negative_labels` (which default to good/pass/true/1 vs.
    bad/fail/false/0, matched case-insensitively) into a boolean score.

    Args:
        classifier: A function that accepts `inputs`, `outputs`, `reference_outputs`,
            and any additional keyword arguments, and returns a label (e.g. "good",
            "bad", True, 1, or a custom bucket name).
        feedback_key: Key used to store the evaluation result, defaults to
            "binary_classification".
        positive_labels: Optional list of labels that should be treated as a positive
            (True) score. Matched case-insensitively. Defaults to
            ["good", "pass", "true", "1"].
        negative_labels: Optional list of labels that should be treated as a negative
            (False) score. Matched case-insensitively. Defaults to
            ["bad", "fail", "false", "0"].

    Returns:
        An evaluator function that takes inputs, outputs, and reference_outputs,
        calls `classifier` on them, and returns an EvaluatorResult with a boolean
        score.

    Example:
        ```python
        from openevals.binary_classifier import create_binary_classifier_evaluator

        evaluator = create_binary_classifier_evaluator(
            classifier=lambda *, outputs, **kwargs: "good" if outputs else "bad",
        )
        result = evaluator(outputs="Paris", reference_outputs="Paris")
        ```
    """
    positive = (
        positive_labels if positive_labels is not None else _DEFAULT_POSITIVE_LABELS
    )
    negative = (
        negative_labels if negative_labels is not None else _DEFAULT_NEGATIVE_LABELS
    )
    _validate_disjoint_labels(positive, negative)

    def wrapped_evaluator(
        *,
        inputs: Any | None = None,
        outputs: Any,
        reference_outputs: Any | None = None,
        **kwargs: Any,
    ) -> EvaluatorResult:
        def get_score():
            raw_label = classifier(
                inputs=inputs,
                outputs=outputs,
                reference_outputs=reference_outputs,
                **kwargs,
            )
            score = _normalize_label(raw_label, positive, negative)
            return (score, f"Classified as {raw_label}.")

        res = _run_evaluator(
            run_name="binary_classifier",
            scorer=get_score,
            feedback_key=feedback_key,
        )
        if isinstance(res, list):
            return res[0]
        return res

    return wrapped_evaluator  # type: ignore


def create_async_binary_classifier_evaluator(
    *,
    classifier: Callable[..., Any | Awaitable[Any]],
    feedback_key: str = "binary_classification",
    positive_labels: list[Any] | None = None,
    negative_labels: list[Any] | None = None,
) -> SimpleAsyncEvaluator:
    """
    Async variant of `create_binary_classifier_evaluator`. `classifier` may be a sync
    or async function.

    Args:
        classifier: A sync or async function that accepts `inputs`, `outputs`,
            `reference_outputs`, and any additional keyword arguments, and returns a
            label (e.g. "good", "bad", True, 1, or a custom bucket name).
        feedback_key: Key used to store the evaluation result, defaults to
            "binary_classification".
        positive_labels: Optional list of labels that should be treated as a positive
            (True) score. Matched case-insensitively. Defaults to
            ["good", "pass", "true", "1"].
        negative_labels: Optional list of labels that should be treated as a negative
            (False) score. Matched case-insensitively. Defaults to
            ["bad", "fail", "false", "0"].

    Returns:
        An async evaluator function that takes inputs, outputs, and
        reference_outputs, calls `classifier` on them, and returns an
        EvaluatorResult with a boolean score.
    """
    positive = (
        positive_labels if positive_labels is not None else _DEFAULT_POSITIVE_LABELS
    )
    negative = (
        negative_labels if negative_labels is not None else _DEFAULT_NEGATIVE_LABELS
    )
    _validate_disjoint_labels(positive, negative)

    async def wrapped_evaluator(
        *,
        inputs: Any | None = None,
        outputs: Any,
        reference_outputs: Any | None = None,
        **kwargs: Any,
    ) -> EvaluatorResult:
        async def get_score():
            raw_label = classifier(
                inputs=inputs,
                outputs=outputs,
                reference_outputs=reference_outputs,
                **kwargs,
            )
            if inspect.isawaitable(raw_label):
                raw_label = await raw_label
            score = _normalize_label(raw_label, positive, negative)
            return (score, f"Classified as {raw_label}.")

        res = await _arun_evaluator(
            run_name="binary_classifier",
            scorer=get_score,
            feedback_key=feedback_key,
        )
        if isinstance(res, list):
            return res[0]
        return res

    return wrapped_evaluator  # type: ignore
