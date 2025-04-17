from typing_extensions import TypedDict

from openevals.types import MessagesDict, EvaluatorResult


class MultiturnSimulatorResult(TypedDict):
    evaluator_results: list[EvaluatorResult]
    trajectory: MessagesDict
