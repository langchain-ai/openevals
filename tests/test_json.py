from evaluators.json.json import json_match_evaluator
import pytest

@pytest.mark.langsmith
def test_json_match_base():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 2}
    evaluator = json_match_evaluator()
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert len(result) == 2
    assert result[0]["key"] == "a"
    assert result[0]["score"] == 1.0
    assert result[1]["key"] == "b"
    assert result[1]["score"] == 1.0

@pytest.mark.langsmith
def test_json_match_average():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    evaluator = json_match_evaluator(aggregator="average")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0.5

@pytest.mark.langsmith
def test_json_match_exclude():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    evaluator = json_match_evaluator(aggregator="average", exclude_keys=["b"])
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 1

@pytest.mark.langsmith
def test_json_match_all():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    evaluator = json_match_evaluator(aggregator="all")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0

@pytest.mark.langsmith
def test_json_match_rubric():
    outputs = {"name": "Harrison Chase", "description": "CEO of LangChain, used to work at Kensho + Robust Intelligence."}
    reference_outputs = {"name": "Harrison Chase", "description": "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence."}
    evaluator = json_match_evaluator(
        aggregator="all",
        judge_rubric={
            "description": "Is the correct title and company mentioned, as well as all previous companies?"
        }
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 1

@pytest.mark.langsmith
def test_json_match_rubric_wrong():
    outputs = {"name": "Harrison Chase", "description": "CEO of LangChain, used to work at Kensho."}
    reference_outputs = {"name": "Harrison Chase", "description": "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence."}
    evaluator = json_match_evaluator(
        aggregator="all",
        judge_rubric={
            "description": "Is the correct title and company mentioned, as well as all previous companies?"
        }
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0