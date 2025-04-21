from typing import (
    Any,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from typing_extensions import NotRequired, TypedDict
from langchain_core.messages import BaseMessage, BaseMessageChunk


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
    ) -> Union[EvaluatorResult, list[EvaluatorResult]]: ...


class SimpleAsyncEvaluator(Protocol):
    async def __call__(
        self,
        *,
        inputs: Optional[Any] = None,
        outputs: Any,
        reference_outputs: Optional[Any] = None,
        **kwargs,
    ) -> Union[EvaluatorResult, list[EvaluatorResult]]: ...


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
    score: Union[float, bool]
    reasoning: Optional[str]


Messages = Union[ChatCompletionMessage, BaseMessage, BaseMessageChunk]


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


MultiturnSimulatorTrajectory = dict[str, Union[list[Messages], Any]]

MultiturnSimulatorTrajectoryUpdate = dict[str, Union[list[Messages], Messages, Any]]


class MultiturnSimulatorResult(TypedDict):
    evaluator_results: list[EvaluatorResult]
    trajectory: MultiturnSimulatorTrajectory
