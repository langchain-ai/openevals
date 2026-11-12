"""Composable evaluators: an evaluator built as a graph of steps.

Experimental: the API may change in any release.

Each step is a node: a Python function marked with ``@node``, an ``LLM`` judge,
or a ``DecisionModel``. A node reads the run being evaluated (``run``,
``thread``, ``trajectory``, ``example``) or the outputs of other nodes, and
``Graph([...])`` runs the nodes in dependency order. LangSmith reads this
structure from the evaluator's source to draw it as a graph, so the names and
call shapes here are part of that contract.
"""

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
    """A path into the evaluated context (``Run.outputs.answer``) or a node's outputs."""

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
    """A step in a graph. Any public attribute is a reference to one of its outputs."""

    def __getattr__(self, name: str) -> _Ref:
        if name.startswith("_"):
            raise AttributeError(name)
        return _Ref(self, name)


def node(fn: Optional[Callable[..., Any]] = None, **options: Any) -> Any:
    """Marks a function as a code node.

    Parameters named ``run``, ``thread``, ``trajectory``, ``example`` or ``ctx``,
    or annotated with ``Run``, ``Thread``, ``Trajectory`` or ``Example``, read
    that context. A parameter named after another node reads its outputs.
    """
    raise NotImplementedError


class LLM(_Node):
    """An LLM-as-judge node whose prompt reads named inputs as ``{{name}}``."""

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
    """A node that asks a decision model the given questions about its inputs."""

    def __init__(
        self,
        *,
        questions: dict[str, dict[str, Any]],
        inputs: Optional[dict[str, _Ref]] = None,
        model: str = "typesafe:jev-latest",
    ) -> None:
        raise NotImplementedError


def Noul(instructions: str) -> dict[str, Any]:
    """A yes/no question; the answer is the probability of yes."""
    raise NotImplementedError


def Score(instructions: str, levels: list[Any]) -> dict[str, Any]:
    """A scale; the answer is a fractional index into ``levels``, lowest first."""
    raise NotImplementedError


def Choice(
    instructions: str, options: Union[list[str], dict[str, Optional[str]]]
) -> dict[str, Any]:
    """One of ``options``; the answer is the chosen option."""
    raise NotImplementedError


class Graph:
    """Nodes run in dependency order, whatever order they are listed in."""

    def __init__(self, nodes: list[_Node]) -> None:
        raise NotImplementedError

    def run(self, ctx: Any) -> dict[Any, Any]:
        """Runs every node against ``ctx`` and returns each node's outputs."""
        raise NotImplementedError
