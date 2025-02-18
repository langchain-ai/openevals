from evaluators.string.levenshtein import (
    levenshtein_distance,
    levenshtein_distance_async,
)

import pytest


@pytest.mark.langsmith
def test_levenshtein():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 2}
    res = levenshtein_distance(outputs=outputs, reference_outputs=reference_outputs)
    assert res["key"] == "levenshtein_distance"
    assert res["score"] == 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
def test_levenshtein_with_different_values():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    res = levenshtein_distance(outputs=outputs, reference_outputs=reference_outputs)
    assert res["key"] == "levenshtein_distance"
    assert res["score"] < 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_levenshtein_async():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 2}
    res = await levenshtein_distance_async(
        outputs=outputs, reference_outputs=reference_outputs
    )
    assert res["key"] == "levenshtein_distance"
    assert res["score"] == 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_levenshtein_with_different_values_async():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    res = await levenshtein_distance_async(
        outputs=outputs, reference_outputs=reference_outputs
    )
    assert res["key"] == "levenshtein_distance"
    assert res["score"] < 1.0
    assert res["comment"] is None
