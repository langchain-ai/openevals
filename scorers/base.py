from typing import TypedDict, Union, Optional, Protocol


class ScorerResult(TypedDict):
    key: str
    score: Union[int, bool]
    comment: Optional[str]


class SimpleScorerCallable(Protocol):
    def __call__(
        self, *, inputs: dict, outputs: dict, reference_outputs: Optional[dict] = None
    ) -> bool: ...
