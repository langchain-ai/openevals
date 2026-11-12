"""Evaluators built as a graph of nodes. Experimental."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

__all__ = [
    "Run",
    "Thread",
    "Trajectory",
    "Example",
    "node",
    "LLM",
    "DecisionModel",
    "Noul",
    "Score",
    "Choice",
    "Graph",
]


class _Ref:
    def __init__(self, root: Any, *path: str) -> None:
        self._root, self._path = root, path

    def __getattr__(self, name: str) -> "_Ref":
        if name.startswith("_"):
            raise AttributeError(name)
        return _Ref(self._root, *self._path, name)


Run = _Ref("run")
Thread = _Ref("thread")
Trajectory = _Ref("trajectory")
Example = _Ref("example")


class _Node:
    def __getattr__(self, name: str) -> _Ref:
        if name.startswith("_"):
            raise AttributeError(name)
        return _Ref(self, name)


def node(fn: Optional[Callable[..., Any]] = None, **options: Any) -> Any:
    raise NotImplementedError


class LLM(_Node):
    def __init__(
        self,
        *,
        model: str,
        prompt: Union[str, list[tuple[str, str]]],
        inputs: Optional[dict[str, _Ref]] = None,
        output_schema: Optional[Any] = None,
        feedback_key: str = "score",
    ) -> None:
        raise NotImplementedError


class DecisionModel(_Node):
    def __init__(
        self,
        *,
        questions: dict[str, dict[str, Any]],
        inputs: Optional[dict[str, _Ref]] = None,
        model: str = "typesafe:jev-latest",
    ) -> None:
        raise NotImplementedError


def Noul(instructions: str) -> dict[str, Any]:
    raise NotImplementedError


def Score(instructions: str, levels: list[Any]) -> dict[str, Any]:
    raise NotImplementedError


def Choice(
    instructions: str, options: Union[list[str], dict[str, Optional[str]]]
) -> dict[str, Any]:
    raise NotImplementedError


class Graph:
    def __init__(self, nodes: list[_Node]) -> None:
        raise NotImplementedError

    def run(self, ctx: Any) -> dict[Any, Any]:
        raise NotImplementedError
