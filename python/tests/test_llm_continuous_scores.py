"""Offline public-API coverage for the default continuous scoring contract."""

import asyncio
import json
from typing import Any

import httpx
import pytest
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from openevals.llm import create_async_llm_as_judge, create_llm_as_judge


@pytest.fixture(autouse=True)
def disable_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


async def evaluate_response(
    response: dict,
    *,
    backend: str,
    use_async: bool,
    **options: Any,
) -> tuple[dict, dict]:
    requests = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline-evaluation",
                "object": "chat.completion",
                "created": 0,
                "model": "offline-judge",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(response),
                            "refusal": None,
                        },
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as sync_http:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as async_http:
            if backend == "langchain":
                judge = ChatOpenAI(
                    model="offline-judge",
                    api_key="offline-not-a-key",
                    http_client=sync_http,
                    http_async_client=async_http,
                    max_retries=0,
                )
            elif use_async:
                judge = AsyncOpenAI(
                    api_key="offline-not-a-key",
                    http_client=async_http,
                    max_retries=0,
                )
            else:
                judge = OpenAI(
                    api_key="offline-not-a-key",
                    http_client=sync_http,
                    max_retries=0,
                )

            params = dict(
                prompt="Rate this response: {outputs}",
                judge=judge,
                model="offline-judge",
                **options,
            )
            if use_async:
                result = await create_async_llm_as_judge(**params)(outputs="example")
            else:
                result = create_llm_as_judge(**params)(outputs="example")

    assert len(requests) == 1
    return result, requests[0]["response_format"]["json_schema"]["schema"]


@pytest.mark.parametrize("backend", ["openai", "langchain"])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("use_reasoning", [False, True])
@pytest.mark.parametrize(
    "score",
    [-1.0, 2.0, float("nan"), float("inf"), -float("inf"), True, "0.5", None],
    ids=["negative", "above-one", "nan", "inf", "negative-inf", "bool", "str", "null"],
)
def test_continuous_judge_rejects_invalid_scores(
    backend: str, use_async: bool, use_reasoning: bool, score: Any
) -> None:
    with pytest.raises(ValueError, match="Continuous judge score"):
        asyncio.run(
            evaluate_response(
                {"score": score, "reasoning": "Offline reason"},
                backend=backend,
                use_async=use_async,
                use_reasoning=use_reasoning,
                continuous=True,
            )
        )


@pytest.mark.parametrize("backend", ["openai", "langchain"])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("use_reasoning", [False, True])
@pytest.mark.parametrize("score", [0, 0.5, 1])
def test_continuous_judge_preserves_valid_scores(
    backend: str, use_async: bool, use_reasoning: bool, score: float
) -> None:
    result, schema = asyncio.run(
        evaluate_response(
            {"score": score, "reasoning": "Offline reason"},
            backend=backend,
            use_async=use_async,
            use_reasoning=use_reasoning,
            continuous=True,
        )
    )
    assert result["score"] == score
    assert result["comment"] == ("Offline reason" if use_reasoning else None)
    assert schema["properties"]["score"]["minimum"] == 0
    assert schema["properties"]["score"]["maximum"] == 1


@pytest.mark.parametrize("backend", ["openai", "langchain"])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("use_reasoning", [False, True])
def test_custom_choices_keep_their_score_scale(
    backend: str, use_async: bool, continuous: bool, use_reasoning: bool
) -> None:
    result, schema = asyncio.run(
        evaluate_response(
            {"score": -2.0, "reasoning": "Offline reason"},
            backend=backend,
            use_async=use_async,
            use_reasoning=use_reasoning,
            continuous=continuous,
            choices=[-2.0, 0.0, 2.0],
        )
    )
    assert result["score"] == -2.0
    assert schema["properties"]["score"]["enum"] == [-2.0, 0.0, 2.0]
    assert "minimum" not in schema["properties"]["score"]


@pytest.mark.parametrize("backend", ["openai", "langchain"])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("use_reasoning", [False, True])
def test_custom_output_schema_keeps_its_score_scale(
    backend: str, use_async: bool, use_reasoning: bool
) -> None:
    output_schema = {
        "title": "custom_score",
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["score", "reasoning"],
        "additionalProperties": False,
    }
    response = {"score": -2.0, "reasoning": "Offline reason"}
    result, schema = asyncio.run(
        evaluate_response(
            response,
            backend=backend,
            use_async=use_async,
            use_reasoning=use_reasoning,
            continuous=True,
            output_schema=output_schema,
        )
    )
    assert result == response
    assert "minimum" not in schema["properties"]["score"]


@pytest.mark.parametrize("backend", ["openai", "langchain"])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("use_reasoning", [False, True])
@pytest.mark.parametrize("score", [False, True])
def test_default_binary_judge_preserves_boolean_scores(
    backend: str, use_async: bool, use_reasoning: bool, score: bool
) -> None:
    result, schema = asyncio.run(
        evaluate_response(
            {"score": score, "reasoning": "Offline reason"},
            backend=backend,
            use_async=use_async,
            use_reasoning=use_reasoning,
        )
    )
    assert result["score"] is score
    assert schema["properties"]["score"]["type"] == "boolean"
