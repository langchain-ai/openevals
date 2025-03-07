from typing import (
    Any,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from typing_extensions import NotRequired, TypedDict


ScoreType = Union[float, bool]


class EvaluatorResult(TypedDict):
    key: str
    score: ScoreType
    comment: Optional[str]


class SimpleEvaluatorNormal(Protocol):
    def __call__(
        self,
        *,
        inputs: Optional[Any] = None,
        outputs: Any,
        reference_outputs: Optional[Any] = None,
        **kwargs,
    ) -> EvaluatorResult | list[EvaluatorResult]: ...


class SimpleEvaluatorJson(Protocol):
    def __call__(
        self,
        *,
        outputs: Any,
        reference_outputs: Any,
        **kwargs,
    ) -> EvaluatorResult | list[EvaluatorResult]: ...


SimpleEvaluator = Union[SimpleEvaluatorNormal, SimpleEvaluatorJson]


class SimpleAsyncEvaluatorNormal(Protocol):
    async def __call__(
        self,
        *,
        inputs: Optional[Any] = None,
        outputs: Any,
        reference_outputs: Optional[Any] = None,
        **kwargs,
    ) -> EvaluatorResult | list[EvaluatorResult]: ...


class SimpleAsyncEvaluatorJson(Protocol):
    async def __call__(
        self,
        *,
        outputs: Any,
        reference_outputs: Any,
        **kwargs,
    ) -> EvaluatorResult | list[EvaluatorResult]: ...


SimpleAsyncEvaluator = Union[SimpleAsyncEvaluatorNormal, SimpleAsyncEvaluatorJson]


class ChatCompletionMessage(TypedDict):
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


@runtime_checkable
class ChatCompletionsClient(Protocol):
    def create(self, **kwargs) -> ChatCompletion: ...


@runtime_checkable
class ModelClient(Protocol):
    @property
    def chat(self) -> type[ChatCompletionsClient]: ...


@runtime_checkable
class RunnableLike(Protocol):
    def invoke(self, **kwargs) -> Any: ...

    async def ainvoke(self, **kwargs) -> Any: ...
