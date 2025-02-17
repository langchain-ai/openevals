from evaluators.string.levenshtein import (
    levenshtein_distance,
    levenshtein_distance_async,
)

import pytest


@pytest.mark.langsmith
def test_levenshtein():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    res = levenshtein_distance(inputs=inputs, outputs=outputs)
    assert res["key"] == "levenshtein_distance"
    assert res["score"] == 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
def test_levenshtein_with_different_values():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 3}
    res = levenshtein_distance(inputs=inputs, outputs=outputs)
    assert res["key"] == "levenshtein_distance"
    assert res["score"] < 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_levenshtein_async():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    res = await levenshtein_distance_async(inputs=inputs, outputs=outputs)
    assert res["key"] == "levenshtein_distance"
    assert res["score"] == 1.0
    assert res["comment"] is None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_levenshtein_with_different_values_async():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 3}
    res = await levenshtein_distance_async(inputs=inputs, outputs=outputs)
    assert res["key"] == "levenshtein_distance"
    assert res["score"] < 1.0
    assert res["comment"] is None
