from typing import (
    Any,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)


class EvaluatorResult(TypedDict):
    key: str
    score: Union[float, bool]
    comment: Optional[str]


class SimpleEvaluator(Protocol):
    def __call__(
        self,
        *,
        inputs: Any,
        outputs: Any,
        reference_outputs: Optional[Any] = None,
        **kwargs,
    ) -> EvaluatorResult | list[EvaluatorResult]: ...


class SimpleAsyncEvaluator(Protocol):
    async def __call__(
        self,
        *,
        inputs: Any,
        outputs: Any,
        reference_outputs: Optional[Any] = None,
        **kwargs,
    ) -> EvaluatorResult | list[EvaluatorResult]: ...


class ChatCompletionMessage(TypedDict):
    content: list[Union[str, dict]]
    role: str
    tool_calls: Optional[list[dict]]


class ChatCompletion(TypedDict):
    choices: list[dict]


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
    def invoke(self, **kwargs) -> ChatCompletion: ...


@runtime_checkable
class LangChainLikeModel(Protocol):
    def with_structured_output(self, **kwargs) -> RunnableLike: ...
