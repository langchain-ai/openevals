"""Tests for the consistency evaluators. They use a fake chat model, so no API key is needed."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from openevals.consistency import (
    create_answer_consistency_evaluator,
    create_async_answer_consistency_evaluator,
    create_async_retrieval_consistency_evaluator,
    create_retrieval_consistency_evaluator,
)

DIMENSIONS = ["core_facts", "specifics", "supporting_detail", "stance", "format"]


def judge_reply(**over) -> str:
    scores = {d: 5 for d in DIMENSIONS}
    scores.update(over)
    return json.dumps({"reason": "judge says so", "scores": scores})


def by_key(results):
    return {r["key"]: r for r in results}


def test_retrieval_id_mode():
    evaluator = create_retrieval_consistency_evaluator(mode="id", max_comparisons=3)
    res = evaluator(outputs=[["a", "b", "c"], ["c", "b", "a"], ["a", "b", "d"]])
    assert res["key"] == "retrieval_consistency"
    assert res["score"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert res["metadata"]["n_comparisons"] == 3
    assert "runs" in res["comment"]


def test_retrieval_ngram_mode_survives_rechunking():
    text = "backup jobs run nightly and retention is thirty days for gold tier"
    words = text.split()
    outputs = [[text], [" ".join(words[:6]), " ".join(words[5:])], [text]]
    ngram = create_retrieval_consistency_evaluator(mode="ngram", ngram_size=2)
    ids = create_retrieval_consistency_evaluator(mode="id")
    assert ngram(outputs=outputs)["score"] > 0.9
    assert ids(outputs=outputs)["score"] < 0.5


def test_validation():
    evaluator = create_retrieval_consistency_evaluator(mode="id")
    with pytest.raises(ValueError, match="at least two"):
        evaluator(outputs=[["a"]])
    with pytest.raises(ValueError, match="at least two"):
        evaluator(outputs="not a list")
    with pytest.raises(ValueError, match="Unsupported mode"):
        create_retrieval_consistency_evaluator(mode="bogus")  # type: ignore
    with pytest.raises(ValueError, match="judge"):
        create_answer_consistency_evaluator()


@pytest.mark.asyncio
async def test_retrieval_async():
    evaluator = create_async_retrieval_consistency_evaluator(mode="id")
    res = await evaluator(outputs=[["a"], ["a"], ["a"]])
    assert res["score"] == 1.0


def test_answer_consistency_returns_headline_and_dimensions():
    judge = FakeListChatModel(responses=[judge_reply(format=1)] * 10)
    evaluator = create_answer_consistency_evaluator(judge=judge)
    results = by_key(
        evaluator(
            inputs="How long are gold backups retained?",
            outputs=["30 days.", "- 30 days\n- see policy", "Thirty days."],
        )
    )
    headline = results["answer_consistency"]
    assert headline["score"] == pytest.approx(0.9)  # format (weight 0.1) scored 0
    assert headline["comment"].endswith("judge says so")
    assert results["answer_consistency.format"]["score"] == 0.0
    assert results["answer_consistency.core_facts"]["score"] == 1.0


@pytest.mark.asyncio
async def test_answer_consistency_async():
    judge = FakeListChatModel(responses=[judge_reply()] * 10)
    evaluator = create_async_answer_consistency_evaluator(judge=judge)
    results = by_key(await evaluator(outputs=["a", "b", "c"]))
    assert results["answer_consistency"]["score"] == 1.0


def test_non_string_runs_are_serialised():
    judge = FakeListChatModel(responses=[judge_reply()] * 4)
    evaluator = create_answer_consistency_evaluator(judge=judge, max_comparisons=1)
    results = by_key(evaluator(outputs=[{"answer": 1}, {"answer": 1}]))
    assert results["answer_consistency"]["score"] == 1.0


def test_identical_runs_skip_the_judge():
    judge = FakeListChatModel(responses=["this reply must never be requested"])
    evaluator = create_answer_consistency_evaluator(judge=judge)
    results = by_key(evaluator(outputs=["30 days.", "30  Days.", "30 days."]))
    assert results["answer_consistency"]["score"] == 1.0
    assert judge.i == 0  # FakeListChatModel advances i on every call


def test_judge_retries_a_badly_formatted_reply():
    judge = FakeListChatModel(responses=["I think they agree.", judge_reply()])
    evaluator = create_answer_consistency_evaluator(judge=judge, max_comparisons=1)
    results = by_key(evaluator(outputs=["a", "b"]))
    assert results["answer_consistency"]["score"] == 1.0
