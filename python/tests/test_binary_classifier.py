import pytest

from openevals.binary_classifier import (
    create_async_binary_classifier_evaluator,
    create_binary_classifier_evaluator,
)
from openevals.types import EvaluatorResult


@pytest.mark.langsmith
def test_binary_classifier_good_bad():
    evaluator = create_binary_classifier_evaluator(
        classifier=lambda **kwargs: "good",
    )
    assert evaluator(outputs="Paris", reference_outputs="Paris") == EvaluatorResult(
        key="binary_classification",
        score=True,
        comment="Classified as good.",
        metadata=None,
    )


@pytest.mark.langsmith
def test_binary_classifier_bad():
    evaluator = create_binary_classifier_evaluator(
        classifier=lambda **kwargs: "bad",
    )
    res = evaluator(outputs="London", reference_outputs="Paris")
    assert res["key"] == "binary_classification"
    assert res["score"] is False
    assert res["comment"] == "Classified as bad."


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "label,expected",
    [
        ("pass", True),
        ("fail", False),
        ("true", True),
        ("false", False),
        ("TRUE", True),
        ("  Good ", True),
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
        (True, True),
        (False, False),
    ],
)
def test_binary_classifier_label_normalization(label, expected):
    evaluator = create_binary_classifier_evaluator(classifier=lambda **kwargs: label)
    res = evaluator(outputs="anything")
    assert res["score"] is expected


@pytest.mark.langsmith
def test_binary_classifier_custom_labels():
    evaluator = create_binary_classifier_evaluator(
        classifier=lambda **kwargs: "acceptable",
        positive_labels=["acceptable"],
        negative_labels=["unacceptable"],
    )
    res = evaluator(outputs="anything")
    assert res["score"] is True


@pytest.mark.langsmith
def test_binary_classifier_unrecognized_label_raises():
    evaluator = create_binary_classifier_evaluator(
        classifier=lambda **kwargs: "maybe",
    )
    with pytest.raises(ValueError):
        evaluator(outputs="anything")


@pytest.mark.langsmith
def test_binary_classifier_overlapping_labels_raises_at_creation():
    with pytest.raises(ValueError):
        create_binary_classifier_evaluator(
            classifier=lambda **kwargs: "good",
            positive_labels=["good", "ambiguous"],
            negative_labels=["bad", "AMBIGUOUS"],
        )


@pytest.mark.langsmith
def test_binary_classifier_does_not_call_classifier_until_invoked():
    calls = []

    def classifier(**kwargs):
        calls.append(1)
        return "good"

    evaluator = create_binary_classifier_evaluator(classifier=classifier)
    assert calls == []
    evaluator(outputs="anything")
    assert calls == [1]


@pytest.mark.langsmith
def test_binary_classifier_receives_inputs_outputs_reference_outputs():
    received = {}

    def classifier(*, inputs, outputs, reference_outputs, **kwargs):
        received["inputs"] = inputs
        received["outputs"] = outputs
        received["reference_outputs"] = reference_outputs
        return "good"

    evaluator = create_binary_classifier_evaluator(classifier=classifier)
    evaluator(
        inputs="What is the capital of France?",
        outputs="Paris",
        reference_outputs="Paris",
    )
    assert received == {
        "inputs": "What is the capital of France?",
        "outputs": "Paris",
        "reference_outputs": "Paris",
    }


@pytest.mark.langsmith
def test_binary_classifier_custom_feedback_key():
    evaluator = create_binary_classifier_evaluator(
        classifier=lambda **kwargs: "good",
        feedback_key="my_custom_key",
    )
    res = evaluator(outputs="anything")
    assert res["key"] == "my_custom_key"


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_binary_classifier_sync_classifier():
    evaluator = create_async_binary_classifier_evaluator(
        classifier=lambda **kwargs: "pass",
    )
    res = await evaluator(outputs="Paris", reference_outputs="Paris")
    assert res == EvaluatorResult(
        key="binary_classification",
        score=True,
        comment="Classified as pass.",
        metadata=None,
    )


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_binary_classifier_async_classifier():
    async def classifier(**kwargs):
        return "fail"

    evaluator = create_async_binary_classifier_evaluator(classifier=classifier)
    res = await evaluator(outputs="London", reference_outputs="Paris")
    assert res["score"] is False
    assert res["comment"] == "Classified as fail."
