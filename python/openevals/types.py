from typing import (
    Any,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from typing_extensions import NotRequired, TypedDict
from langchain_core.messages import BaseMessage


ScoreType = Union[float, bool]


class EvaluatorResult(TypedDict):
    key: str
    score: ScoreType
    comment: Optional[str]
    metadata: Optional[dict]


class SimpleEvaluator(Protocol):
    def __call__(
        self,
        *,
        inputs: Optional[Any] = None,
        outputs: Any,
        reference_outputs: Optional[Any] = None,
        **kwargs,
    ) -> EvaluatorResult | list[EvaluatorResult]: ...


class SimpleAsyncEvaluator(Protocol):
    async def __call__(
        self,
        *,
        inputs: Optional[Any] = None,
        outputs: Any,
        reference_outputs: Optional[Any] = None,
        **kwargs,
    ) -> EvaluatorResult | list[EvaluatorResult]: ...


class ChatCompletionMessage(TypedDict):
    id: NotRequired[Optional[str]]
    content: Union[str, list[dict]]
    role: str
    tool_calls: NotRequired[Optional[list[dict]]]


class ChatCompletion(TypedDict):
    choices: list[dict]


# Few shot example type for evaluator prompts
class FewShotExample(TypedDict):
    inputs: Any
    outputs: Any
    score: float | bool
    reasoning: Optional[str]


class MessagesDict(TypedDict):
    messages: list[Union[ChatCompletionMessage, BaseMessage]]


@runtime_checkable
class ChatCompletionsClient(Protocol):
    def create(self, **kwargs) -> ChatCompletion: ...


@runtime_checkable
class ModelClient(Protocol):
    @property
    def chat(self) -> type[ChatCompletionsClient]: ...


@runtime_checkable
class RunnableLike(Protocol):
    """@deprecated: Use langchain_core.runnables.Runnable instead."""

    def invoke(self, inputs: Any, **kwargs) -> Any: ...

    async def ainvoke(self, inputs: Any, **kwargs) -> Any: ...
