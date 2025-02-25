from openevals.json import create_async_json_match_evaluator
import pytest
from langsmith import Client


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_base():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 2}
    evaluator = create_async_json_match_evaluator()
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert len(result) == 2
    assert result[0]["key"] == "json_match:a"
    assert result[0]["score"] == 1.0
    assert result[1]["key"] == "json_match:b"
    assert result[1]["score"] == 1.0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_mix():
    outputs = {"a": "Mango, Bananas", "b": 2}
    reference_outputs = {"a": "Bananas, Mango", "b": 3}
    evaluator = create_async_json_match_evaluator(
        model="openai:o3-mini",
        rubric={"a": "Does the answer mention all the fruits in the reference answer?"},
        aggregator="average",
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 0.5


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_average():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    evaluator = create_async_json_match_evaluator(aggregator="average")
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 0.5


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_exclude():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    evaluator = create_async_json_match_evaluator(
        aggregator="average", exclude_keys=["b"]
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 1


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_all():
    outputs = {"a": 1, "b": 2}
    reference_outputs = {"a": 1, "b": 3}
    evaluator = create_async_json_match_evaluator(aggregator="all")
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:all"
    assert result[0]["score"] == 0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_rubric():
    outputs = {
        "name": "Harrison Chase",
        "description": "CEO of LangChain, used to work at Kensho + Robust Intelligence.",
    }
    reference_outputs = {
        "name": "Harrison Chase",
        "description": "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence.",
    }
    evaluator = create_async_json_match_evaluator(
        model="openai:o3-mini",
        aggregator="all",
        rubric={
            "description": "Is the correct job title and company mentioned, as well as previous companies?"
        },
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:all"
    assert result[0]["score"] == 1


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_rubric_wrong():
    outputs = {
        "name": "Harrison Chase",
        "description": "CEO of LangChain, used to work at Kensho.",
    }
    reference_outputs = {
        "name": "Harrison Chase",
        "description": "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence.",
    }
    evaluator = create_async_json_match_evaluator(
        model="openai:o3-mini",
        aggregator="all",
        rubric={
            "description": "Is the correct job title and company mentioned, as well as previous companies?"
        },
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:all"
    assert result[0]["score"] == 0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_rubric_with_reasoning():
    outputs = {"description": "CEO of LangChain, used to work at Kensho."}
    reference_outputs = {
        "description": "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence."
    }
    evaluator = create_async_json_match_evaluator(
        model="openai:o3-mini",
        rubric={
            "description": "Is the correct job title and company mentioned, as well as previous companies?"
        },
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:description"
    assert result[0]["score"] == 0
    assert result[0]["comment"] is not None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_rubric_without_reasoning():
    outputs = {"description": "CEO of LangChain, used to work at Kensho."}
    reference_outputs = {
        "description": "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence."
    }
    evaluator = create_async_json_match_evaluator(
        model="openai:o3-mini",
        aggregator="all",
        rubric={
            "description": "Is the correct job title and company mentioned, as well as previous companies?"
        },
        use_reasoning=False,
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:all"
    assert result[0]["score"] == 0
    assert result[0]["comment"] is None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_rubric_with_reasoning_individual_key():
    outputs = {
        "name": "Harrison Chase",
        "description": "CEO of LangChain, used to work at Kensho.",
    }
    reference_outputs = {
        "name": "Harrison Chase",
        "description": "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence.",
    }
    evaluator = create_async_json_match_evaluator(
        model="openai:o3-mini",
        rubric={
            "description": "Is the correct job title and company mentioned, as well as previous companies?"
        },
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert len(result) == 2
    assert result[0]["key"] == "json_match:name"
    assert result[0]["score"] == 1.0
    assert result[1]["key"] == "json_match:description"
    assert result[1]["score"] == 0
    assert result[1]["comment"] is not None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_all_none():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    evaluator = create_async_json_match_evaluator()
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    result = sorted(result, key=lambda x: x["key"])
    assert len(result) == 2
    assert result[0]["key"] == "json_match:a"
    assert result[0]["score"] == 1.0
    assert result[1]["key"] == "json_match:b"
    assert result[1]["score"] == 1.0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_average_none():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 3},
    ]
    evaluator = create_async_json_match_evaluator(list_aggregator="average")
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    result = sorted(result, key=lambda x: x["key"])
    assert len(result) == 2
    assert result[0]["key"] == "json_match:a"
    assert result[0]["score"] == 1.0
    assert result[1]["key"] == "json_match:b"
    assert result[1]["score"] == 0.5


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_all_all():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    evaluator = create_async_json_match_evaluator(aggregator="all")
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:all"
    assert result[0]["score"] == 1


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_average_all():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 3},
    ]
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average", aggregator="all"
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:all"
    assert result[0]["score"] == 0.5


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_all_average():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 2, "b": 2},
    ]
    evaluator = create_async_json_match_evaluator(aggregator="average")
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_average_average():
    outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 3},
    ]
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average", aggregator="average"
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 0.75


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_mismatch_all_none():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = create_async_json_match_evaluator()
    results = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    results = sorted(results, key=lambda x: x["key"])
    assert len(results) == 4
    assert results[0]["key"] == "json_match:a"
    assert results[0]["score"] == 1.0
    assert results[1]["key"] == "json_match:b"
    assert results[1]["score"] == 0
    assert results[2]["key"] == "json_match:c"
    assert results[2]["score"] == 0
    assert results[3]["key"] == "json_match:d"
    assert results[3]["score"] == 0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_mismatch_average_none():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = create_async_json_match_evaluator(list_aggregator="average")
    results = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    results = sorted(results, key=lambda x: x["key"])
    assert len(results) == 4
    assert results[0]["key"] == "json_match:a"
    assert results[0]["score"] == 1.0
    assert results[1]["key"] == "json_match:b"
    assert results[1]["score"] == 0.5
    assert results[2]["key"] == "json_match:c"
    assert results[2]["score"] == 0
    assert results[3]["key"] == "json_match:d"
    assert results[3]["score"] == 0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_mismatch_all_all():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = create_async_json_match_evaluator(aggregator="all")
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:all"
    assert result[0]["score"] == 0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_mismatch_average_all():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average", aggregator="all"
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:all"
    assert result[0]["score"] == 0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_mismatch_all_average():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = create_async_json_match_evaluator(aggregator="average")
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_mismatch_average_average():
    outputs = [
        {"a": 1, "d": 2},
        {"a": 1, "b": 2},
    ]
    reference_outputs = [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2, "c": 3},
    ]
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average", aggregator="average"
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 0.5


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_rubric():
    outputs = [{"a": "Strawberries, Melons, Bananas"}]
    reference_outputs = [{"a": "Bananas, Strawberries, Melons"}]
    evaluator = create_async_json_match_evaluator(
        model="openai:o3-mini",
        rubric={"a": "Does the answer mention all the fruits in the reference answer?"},
        list_aggregator="average",
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:a"
    assert result[0]["score"] == 1


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_list_mismatch_output_missing():
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
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average", aggregator="average"
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 5 / 6


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_mode_exact_extra_reference():
    outputs = [{"a": 1}, {"a": 1}]
    reference_outputs = [{"a": 1}, {"a": 1}, {"a": 1}]
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average", aggregator="average"
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 2 / 3


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_mode_exact_extra_output():
    outputs = [{"a": 1}, {"a": 1}, {"a": 1}]
    reference_outputs = [
        {"a": 1},
        {"a": 1},
    ]
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average", aggregator="average"
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 2 / 3


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_mode_exact_unordered():
    outputs = [{"a": 1, "d": 2, "e": 2}, {"b": 1}, {"c": 1}]
    reference_outputs = [{"b": 1, "d": 2, "e": 2}, {"a": 1}, {"c": 1}]
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average",
        aggregator="average",
        exclude_keys=["d", "e"],
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 1


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_mode_subset_outputs():
    outputs = [{"a": 1}, {"b": 1}, {"c": 1}]
    reference_outputs = [
        {"b": 1},
        {"a": 1},
    ]
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average", aggregator="average", list_match_mode="superset"
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 1


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_mode_subset_reference():
    outputs = [
        {"a": 1},
        {"b": 1},
    ]
    reference_outputs = [{"b": 1}, {"c": 1}, {"a": 1}]
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average", aggregator="average", list_match_mode="subset"
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 1


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_mode_order_wrong():
    outputs = [
        {"a": 1},
        {"b": 1},
    ]
    reference_outputs = [{"b": 1}, {"a": 1}]
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average", aggregator="average", list_match_mode="ordered"
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_json_match_mode_order():
    outputs = [
        {"a": 1},
        {"b": 1},
        {"c": 1},
    ]
    reference_outputs = [
        {"a": 1},
        {"b": 1},
        {"d": 1},
    ]
    evaluator = create_async_json_match_evaluator(
        list_aggregator="average",
        aggregator="average",
        list_match_mode="ordered",
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result[0]["key"] == "json_match:average"
    assert result[0]["score"] == 2 / 3


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_works_with_aevaluate():
    client = Client()
    evaluator = create_async_json_match_evaluator()

    async def target(x):
        return x

    res = await client.aevaluate(target, data="json", evaluators=[evaluator])
    async for r in res:
        assert r["evaluation_results"]["results"][0].score is not None


@pytest.mark.langsmith
def test_error_no_rubric():
    with pytest.raises(ValueError):
        create_async_json_match_evaluator(model="openai:o3-mini")


@pytest.mark.langsmith
def test_error_no_model():
    with pytest.raises(ValueError):
        create_async_json_match_evaluator(rubric={"a": "foo"})
