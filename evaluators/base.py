from typing import TypedDict, Union, Optional, Protocol


class EvaluatorResult(TypedDict):
    key: str
    score: Union[int, bool]
    comment: Optional[str]


class SimpleEvaluator(Protocol):
    def __call__(
        self,
        *,
        inputs: dict,
        outputs: dict,
        reference_outputs: Optional[dict] = None,
        **kwargs,
    ) -> bool: ...
