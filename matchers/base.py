from typing import TypedDict, Union, Optional, Callable

class MatchResult(TypedDict):
    key: str
    score: Union[int, bool]
    comment: Optional[str]

class MatcherKwargs(TypedDict):
    inputs: dict
    outputs: dict
    reference_outputs: Optional[dict]

MatcherFunction = Callable[
    [MatcherKwargs],
    MatchResult
]
