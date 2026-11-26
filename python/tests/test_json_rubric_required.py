import pytest

import openevals.json.match as json_match
from openevals.json.match import (
    _prepare_parameters,
    create_async_json_match_evaluator,
    create_json_match_evaluator,
)


@pytest.mark.parametrize("use_reasoning", [False, True])
def test_rubric_keys_are_required_in_judge_schema(use_reasoning):
    _, _, json_schema, _, _, _ = _prepare_parameters(
        outputs={"a": 1, "b": 2, "c": 3},
        reference_outputs={"a": 1, "b": 2, "c": 3},
        rubric={"a": "Is a correct?", "b": "Is b correct?"},
        exclude_keys=[],
        use_reasoning=use_reasoning,
    )
    assert list(json_schema["properties"]) == ["a", "b"]
    assert json_schema["required"] == ["a", "b"]


def test_rubric_keys_are_required_for_list_outputs():
    _, _, json_schema, _, _, _ = _prepare_parameters(
        outputs=[{"a": 1}, {"a": 2}],
        reference_outputs=[{"a": 1}, {"a": 2}],
        rubric={"a": "Is a correct?"},
        exclude_keys=[],
        use_reasoning=False,
        list_match_mode="ordered",
    )
    assert sorted(json_schema["required"]) == sorted(json_schema["properties"])
    assert sorted(json_schema["required"]) == ["a_0", "a_1"]


@pytest.mark.parametrize("use_reasoning", [False, True])
@pytest.mark.parametrize("aggregator", ["all", "average"])
def test_aggregator_judge_schema_requires_rubric_keys(
    monkeypatch, aggregator, use_reasoning
):
    seen = {}
    if use_reasoning:
        reply = {
            "a": {"score": True, "reasoning": "ok"},
            "b": {"score": False, "reasoning": "no"},
        }
    else:
        reply = {"a": True, "b": False}

    def fake_judge_scorer(*, system, prompt, schema, judge, model):
        seen["schema"] = schema
        return lambda **kwargs: reply

    monkeypatch.setattr(json_match, "_create_llm_as_judge_scorer", fake_judge_scorer)
    evaluator = create_json_match_evaluator(
        aggregator=aggregator,
        rubric={"a": "Is a correct?", "b": "Is b correct?"},
        model="openai:gpt-4.1-mini",
        use_reasoning=use_reasoning,
    )
    evaluator(outputs={"a": 1, "b": 2}, reference_outputs={"a": 1, "b": 3})
    assert seen["schema"]["required"] == ["a", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("aggregator", ["all", "average"])
async def test_async_aggregator_judge_schema_requires_rubric_keys(
    monkeypatch, aggregator
):
    seen = {}

    def fake_judge_scorer(*, system, prompt, schema, judge, model):
        seen["schema"] = schema

        async def scorer(**kwargs):
            return {"a": True, "b": False}

        return scorer

    monkeypatch.setattr(
        json_match, "_create_async_llm_as_judge_scorer", fake_judge_scorer
    )
    evaluator = create_async_json_match_evaluator(
        aggregator=aggregator,
        rubric={"a": "Is a correct?", "b": "Is b correct?"},
        model="openai:gpt-4.1-mini",
        use_reasoning=False,
    )
    await evaluator(outputs={"a": 1, "b": 2}, reference_outputs={"a": 1, "b": 3})
    assert seen["schema"]["required"] == ["a", "b"]
