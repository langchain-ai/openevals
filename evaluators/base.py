from typing import Any, Optional, Protocol, TypedDict, Union


class EvaluatorResult(TypedDict):
    key: str
    score: Union[int, bool]
    comment: Optional[str]


class SimpleEvaluator(Protocol):
    def __call__(
        self,
        *,
        inputs: Any,
        outputs: Any,
        reference_outputs: Optional[Any] = None,
        **kwargs,
    ) -> bool: ...
