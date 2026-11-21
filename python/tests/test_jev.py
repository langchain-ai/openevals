"""Tests for Jev-as-a-judge evaluators."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure typesafe_sdk is importable even when not installed.
if "typesafe_sdk" not in sys.modules:
    _mod = ModuleType("typesafe_sdk")
    _mod.TypeSafeClient = MagicMock  # type: ignore[attr-defined]
    _mod.AsyncTypeSafeClient = MagicMock  # type: ignore[attr-defined]
    _mod.Noul = MagicMock  # type: ignore[attr-defined]
    _mod.Choice = MagicMock  # type: ignore[attr-defined]
    _mod.Score = MagicMock  # type: ignore[attr-defined]
    sys.modules["typesafe_sdk"] = _mod

from openevals.jev import _build_state, _parse_results


def _make_noul_answer(prob):
    a = MagicMock()
    a.type = "noul"
    a.noul = prob
    return a


def _make_choice_answer(choice, confidence, probs):
    a = MagicMock()
    a.type = "choice"
    a.choice = choice
    a.confidence = confidence
    a.probabilities = probs
    return a


def _make_score_answer(score, confidence, probs):
    a = MagicMock()
    a.type = "score"
    a.score = score
    a.confidence = confidence
    a.probabilities = probs
    return a


def _make_response(answers):
    r = MagicMock()
    r.answers = answers
    return r


# ── _build_state ──


def test_build_state_all_params():
    state = _build_state(inputs="q", outputs="a", reference_outputs="ref")
    assert "[Inputs]" in state and "q" in state
    assert "[Outputs]" in state and "a" in state
    assert "[Reference Outputs]" in state and "ref" in state


def test_build_state_outputs_only():
    state = _build_state(outputs="answer")
    assert "[Outputs]" in state
    assert "[Inputs]" not in state


def test_build_state_dict_input():
    state = _build_state(inputs={"question": "2+2?"}, outputs="4")
    assert "2+2?" in state


def test_build_state_custom_builder():
    def builder(*, inputs=None, outputs=None, **kw):
        return f"Q:{inputs} A:{outputs}"

    assert _build_state(inputs="x", outputs="y", state_builder=builder) == "Q:x A:y"


def test_build_state_extra_kwargs():
    state = _build_state(outputs="out", context="ctx")
    assert "[context]" in state and "ctx" in state


# ── _parse_results ──


def test_parse_noul():
    r = _parse_results(_make_response({"q": _make_noul_answer(0.95)}), {"q": {}})
    assert r["q"]["score"] == 0.95
    assert r["q"]["metadata"]["type"] == "noul"


def test_parse_choice():
    r = _parse_results(
        _make_response({"q": _make_choice_answer("a", 0.8, {"a": 0.8, "b": 0.2})}),
        {"q": {}},
    )
    assert r["q"]["metadata"]["choice"] == "a"


def test_parse_score():
    r = _parse_results(
        _make_response({"q": _make_score_answer(0.7, 0.85, {"0": 0.3, "1": 0.7})}),
        {"q": {}},
    )
    assert r["q"]["score"] == 0.7
    assert r["q"]["metadata"]["confidence"] == 0.85


def test_parse_missing():
    r = _parse_results(_make_response({}), {"q": {}})
    assert r["q"]["score"] is False


def test_parse_multiple():
    r = _parse_results(
        _make_response({"a": _make_noul_answer(0.9), "b": _make_noul_answer(0.1)}),
        {"a": {}, "b": {}},
    )
    assert len(r) == 2


# ── create_jev_evaluator (mocked SDK) ──


@pytest.mark.langsmith
def test_jev_evaluator_noul():
    mock_client = MagicMock()
    mock_client.system_one.return_value = _make_response(
        {"is_correct": _make_noul_answer(0.92)}
    )
    with patch("typesafe_sdk.TypeSafeClient", return_value=mock_client):
        from openevals.jev import create_jev_evaluator
        from typesafe_sdk import Noul

        evaluator = create_jev_evaluator(
            questions={"is_correct": Noul(instructions="Is the answer correct?")},
        )
        result = evaluator(inputs="2+2?", outputs="4")

    mock_client.system_one.assert_called_once()
    if isinstance(result, list):
        scores = {r["key"]: r["score"] for r in result}
        assert scores["is_correct"] == 0.92
    else:
        assert result["score"] == 0.92


@pytest.mark.langsmith
def test_jev_evaluator_multi_question():
    mock_client = MagicMock()
    mock_client.system_one.return_value = _make_response(
        {
            "correct": _make_noul_answer(0.95),
            "helpful": _make_noul_answer(0.88),
        }
    )
    with patch("typesafe_sdk.TypeSafeClient", return_value=mock_client):
        from openevals.jev import create_jev_evaluator
        from typesafe_sdk import Noul

        evaluator = create_jev_evaluator(
            questions={
                "correct": Noul(instructions="Correct?"),
                "helpful": Noul(instructions="Helpful?"),
            },
        )
        result = evaluator(inputs="test", outputs="answer")

    assert isinstance(result, list)
    keys = {r["key"] for r in result}
    assert keys == {"correct", "helpful"}


@pytest.mark.langsmith
def test_jev_evaluator_with_reference():
    mock_client = MagicMock()
    mock_client.system_one.return_value = _make_response(
        {"matches": _make_noul_answer(0.99)}
    )
    with patch("typesafe_sdk.TypeSafeClient", return_value=mock_client):
        from openevals.jev import create_jev_evaluator
        from typesafe_sdk import Noul

        evaluator = create_jev_evaluator(
            questions={"matches": Noul(instructions="Match?")},
        )
        evaluator(inputs="Capital?", outputs="Paris", reference_outputs="Paris")

    state_arg = mock_client.system_one.call_args
    state = state_arg.kwargs.get("state") or state_arg[0][0]
    assert "Paris" in state and "Reference" in state


@pytest.mark.langsmith
def test_jev_evaluator_custom_state_builder():
    mock_client = MagicMock()
    mock_client.system_one.return_value = _make_response({"ok": _make_noul_answer(1.0)})
    with patch("typesafe_sdk.TypeSafeClient", return_value=mock_client):
        from openevals.jev import create_jev_evaluator
        from typesafe_sdk import Noul

        def my_builder(*, inputs=None, outputs=None, **kw):
            return f"Q:{inputs} A:{outputs}"

        evaluator = create_jev_evaluator(
            questions={"ok": Noul(instructions="OK?")},
            state_builder=my_builder,
        )
        evaluator(inputs="Hi", outputs="Hello")

    state = (
        mock_client.system_one.call_args.kwargs.get("state")
        or mock_client.system_one.call_args[0][0]
    )
    assert state == "Q:Hi A:Hello"


@pytest.mark.langsmith
def test_jev_evaluator_choice():
    mock_client = MagicMock()
    mock_client.system_one.return_value = _make_response(
        {"intent": _make_choice_answer("billing", 0.9, {"billing": 0.9, "tech": 0.1})}
    )
    with patch("typesafe_sdk.TypeSafeClient", return_value=mock_client):
        from openevals.jev import create_jev_evaluator
        from typesafe_sdk import Choice

        evaluator = create_jev_evaluator(
            questions={"intent": Choice(criteria={"billing": "Pay", "tech": "Bug"})},
        )
        result = evaluator(outputs="Invoice wrong")

    r = result[0] if isinstance(result, list) else result
    assert r["metadata"]["choice"] == "billing"


@pytest.mark.langsmith
def test_jev_evaluator_score():
    mock_client = MagicMock()
    mock_client.system_one.return_value = _make_response(
        {"quality": _make_score_answer(0.75, 0.85, {"0": 0.05, "1": 0.2, "2": 0.75})}
    )
    with patch("typesafe_sdk.TypeSafeClient", return_value=mock_client):
        from openevals.jev import create_jev_evaluator
        from typesafe_sdk import Score

        evaluator = create_jev_evaluator(
            questions={"quality": Score(criteria=["Poor", "OK", "Great"])},
        )
        result = evaluator(outputs="42")

    r = result[0] if isinstance(result, list) else result
    assert r["score"] == 0.75


@pytest.mark.langsmith
def test_jev_evaluator_import_error():
    with patch.dict("sys.modules", {"typesafe_sdk": None}):
        from openevals.jev import create_jev_evaluator

        with pytest.raises(ImportError, match="typesafe-sdk"):
            create_jev_evaluator(questions={"q": {}})


# ── Async ──


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_jev_evaluator():
    mock_client = AsyncMock()
    mock_client.system_one.return_value = _make_response(
        {"ok": _make_noul_answer(0.88)}
    )
    with patch("typesafe_sdk.AsyncTypeSafeClient", return_value=mock_client):
        from openevals.jev import create_async_jev_evaluator
        from typesafe_sdk import Noul

        evaluator = create_async_jev_evaluator(
            questions={"ok": Noul(instructions="OK?")},
        )
        await evaluator(inputs="test", outputs="answer")

    mock_client.system_one.assert_called_once()
