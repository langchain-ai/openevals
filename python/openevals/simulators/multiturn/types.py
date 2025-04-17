from typing import Any, Union
from typing_extensions import TypedDict

from openevals.types import Messages, EvaluatorResult


TrajectoryDict = dict[str, Union[list[Messages], Any]]

TrajectoryDictUpdate = dict[str, Union[list[Messages], Messages, Any]]


class MultiturnSimulatorResult(TypedDict):
    evaluator_results: list[EvaluatorResult]
    trajectory: TrajectoryDict
