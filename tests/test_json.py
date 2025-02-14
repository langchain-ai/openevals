from evaluators.json import json_match_evaluator
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
def test_json_match_mix():
    outputs = {"a": "Mango, Bananas", "b": 2}
    reference_outputs = {"a": "Bananas, Mango", "b": 3}
    evaluator = json_match_evaluator(
        rubric={"a": "Does the answer mention all the fruits in the reference answer?"}, 
        aggregator="average"
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0.5

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
    evaluator = json_match_evaluator(aggregator="conjunction")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0

@pytest.mark.langsmith
def test_json_match_rubric():
    outputs = {"name": "Harrison Chase", "description": "CEO of LangChain, used to work at Kensho + Robust Intelligence."}
    reference_outputs = {"name": "Harrison Chase", "description": "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence."}
    evaluator = json_match_evaluator(
        aggregator="conjunction",
        rubric={
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
        aggregator="conjunction",
        rubric={
            "description": "Is the correct title and company mentioned, as well as all previous companies?"
        }
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0

@pytest.mark.langsmith
def test_json_match_rubric_with_reasoning():
    outputs = {"description": "CEO of LangChain, used to work at Kensho."}
    reference_outputs = {"description": "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence."}
    evaluator = json_match_evaluator(
        rubric={
            "description": "Is the correct title and company mentioned, as well as all previous companies?"
        }
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0
    assert result["comment"] is not None

@pytest.mark.langsmith
def test_json_match_rubric_without_reasoning():
    outputs = {"description": "CEO of LangChain, used to work at Kensho."}
    reference_outputs = {"description": "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence."}
    evaluator = json_match_evaluator(
        aggregator="conjunction",
        rubric={
            "description": "Is the correct title and company mentioned, as well as all previous companies?"
        },
        use_reasoning=False
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0
    assert result["comment"] is None

@pytest.mark.langsmith
def test_json_match_rubric_with_reasoning_individual_key():
    outputs = {"name": "Harrison Chase", "description": "CEO of LangChain, used to work at Kensho."}
    reference_outputs = {"name": "Harrison Chase", "description": "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence."}
    evaluator = json_match_evaluator(
        rubric={
            "description": "Is the correct title and company mentioned, as well as all previous companies?"
        }
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert len(result) == 2
    assert result[0]["key"] == "name"
    assert result[0]["score"] == 1.0
    assert result[1]["key"] == "description"
    assert result[1]["score"] == 0
    assert result[1]["comment"] is not None

@pytest.mark.langsmith
def test_json_match_list_all_none():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    evaluator = json_match_evaluator()
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    result = sorted(result, key=lambda x: x["key"])
    assert len(result) == 2
    assert result[0]["key"] == "a"
    assert result[0]["score"] == 1.0
    assert result[1]["key"] == "b"
    assert result[1]["score"] == 1.0

@pytest.mark.langsmith
def test_json_match_list_average_none():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 3},
    ]
    evaluator = json_match_evaluator(list_aggregator="average")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    result = sorted(result, key=lambda x: x["key"])
    assert len(result) == 2
    assert result[0]["key"] == "a"
    assert result[0]["score"] == 1.0
    assert result[1]["key"] == "b"
    assert result[1]["score"] == 0.5

@pytest.mark.langsmith
def test_json_match_list_all_all():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    evaluator = json_match_evaluator(aggregator="conjunction")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 1

@pytest.mark.langsmith
def test_json_match_list_average_all():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 3},
    ]
    evaluator = json_match_evaluator(list_aggregator="average",aggregator="conjunction")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0

@pytest.mark.langsmith
def test_json_match_list_all_average():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 2, "b": 2},
    ]
    evaluator = json_match_evaluator(aggregator="average")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0.5

@pytest.mark.langsmith
def test_json_match_list_average_average():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 3},
    ]
    evaluator = json_match_evaluator(list_aggregator="average",aggregator="average")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0.75


@pytest.mark.langsmith
def test_json_match_list_mismatch_all_none():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = json_match_evaluator()
    results = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    results = sorted(results, key=lambda x: x["key"])
    assert len(results) == 4
    assert results[0]["key"] == "a"
    assert results[0]["score"] == 1.0
    assert results[1]["key"] == "b"
    assert results[1]["score"] == 0
    assert results[2]["key"] == "c"
    assert results[2]["score"] == 0
    assert results[3]["key"] == "d"
    assert results[3]["score"] == 0


@pytest.mark.langsmith
def test_json_match_list_mismatch_average_none():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = json_match_evaluator(list_aggregator="average")
    results = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    results = sorted(results, key=lambda x: x["key"])
    assert len(results) == 4
    assert results[0]["key"] == "a"
    assert results[0]["score"] == 1.0
    assert results[1]["key"] == "b"
    assert results[1]["score"] == 0.5
    assert results[2]["key"] == "c"
    assert results[2]["score"] == 0
    assert results[3]["key"] == "d"
    assert results[3]["score"] == 0

@pytest.mark.langsmith
def test_json_match_list_mismatch_all_all():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = json_match_evaluator(aggregator="conjunction")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0

@pytest.mark.langsmith
def test_json_match_list_mismatch_average_all():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = json_match_evaluator(list_aggregator="average", aggregator="conjunction")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0

@pytest.mark.langsmith
def test_json_match_list_mismatch_all_average():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = json_match_evaluator(aggregator="average")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0.25

@pytest.mark.langsmith
def test_json_match_list_mismatch_average_average():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = json_match_evaluator(list_aggregator="average", aggregator="average")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0.375


@pytest.mark.langsmith
def test_json_match_list_rubric():
    outputs = [
        {"a": "Strawberries, Melons, Bananas"}
    ]
    reference_outputs = [
        {"a": "Bananas, Strawberries, Melons"}
    ]
    evaluator = json_match_evaluator(
        rubric={"a": "Does the answer mention all the fruits in the reference answer?"},
        list_aggregator="average"
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 1


@pytest.mark.langsmith
def test_json_match_list_mismatch_output_missing():
    outputs = [
        {"a": 1, "b": 2, "d": 3},
        {"a": 1, "b": 2, "c": 3},
        {"a": 1, "b": 2, "d": 3},
    ]
    reference_outputs = [
        {"a": 1, "b": 2, "d": 3},
        {"a": 1, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = json_match_evaluator(list_aggregator="average", aggregator="average")
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    assert result["score"] == 0.75