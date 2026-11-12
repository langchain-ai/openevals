"""Evaluators built as a graph of nodes. Experimental."""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Optional, Union

from langsmith import traceable

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
    """A path into the evaluated context (`Run.outputs.answer`) or a node's outputs."""

    def __init__(self, root: Any, *path: str) -> None:
        self._root, self._path = root, path

    def __getattr__(self, name: str) -> "_Ref":
        if name.startswith("_"):
            raise AttributeError(name)
        return _Ref(self._root, *self._path, name)

    def _resolve(self, ctx: Any, results: dict) -> Any:
        value = (
            results[self._root] if self._root in results else _context(ctx, self._root)
        )
        for key in self._path:
            value = value[key]
        return value


Run, Thread, Trajectory = _Ref("run"), _Ref("thread"), _Ref("trajectory")
Example = _Ref("example")
_CONTEXT_NAMES = {"run", "thread", "trajectory", "example", "ctx"}


def _context(ctx: Any, name: str) -> Any:
    return ctx if name == "ctx" else getattr(ctx, name)


class _Node:
    _outputs: Optional[set] = None
    _name = "node"
    _run_type = "chain"

    def __getattr__(self, name: str) -> _Ref:
        if name.startswith("_"):
            raise AttributeError(name)
        return _Ref(self, name)

    def _sources(self):
        return ((name, ref._root) for name, ref in self._inputs.items())

    def _values(self, ctx: Any, results: dict) -> dict:
        return {name: ref._resolve(ctx, results) for name, ref in self._inputs.items()}

    def _traced_run(self, ctx: Any, results: dict) -> Any:
        return traceable(name=self._name, run_type=self._run_type)(self._run)(
            ctx, results
        )


class _CodeNode(_Node):
    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn, self._inputs, self._name = fn, {}, fn.__name__

    def _sources(self):
        for name, param in inspect.signature(self._fn).parameters.items():
            # Parameters with defaults are options; Python fills them in.
            if param.default is not param.empty:
                continue
            if param.annotation in (Run, Thread, Trajectory, Example):
                yield name, param.annotation._root
            elif name in _CONTEXT_NAMES:
                yield name, name
            else:
                # A parameter named after another node reads that node's outputs.
                yield name, self._fn.__globals__.get(name, name)

    def _run(self, ctx: Any, results: dict) -> Any:
        args = {
            name: results[src] if isinstance(src, _Node) else _context(ctx, src)
            for name, src in self._sources()
        }
        return self._fn(**args)


def node(fn: Optional[Callable[..., Any]] = None, **_: Any) -> Any:
    return _CodeNode(fn) if fn else _CodeNode


class LLM(_Node):
    _run_type = "llm"

    def __init__(
        self,
        *,
        model: str,
        prompt: Union[str, list[tuple[str, str]]],
        inputs: Optional[dict[str, _Ref]] = None,
        output_schema: Optional[Any] = None,
        feedback_key: str = "score",
        name: str = "LLM",
        **_: Any,
    ) -> None:
        self._model, self._prompt, self._inputs = model, prompt, inputs or {}
        self._output_schema, self._feedback_key = output_schema, feedback_key
        self._name = name
        if output_schema is None:
            self._outputs = {feedback_key}
        elif isinstance(output_schema, dict):
            self._outputs = set(output_schema["properties"])
        else:
            self._outputs = set(output_schema.__annotations__)

    def _messages(self, ctx: Any, results: dict) -> list[dict]:
        values = self._values(ctx, results)

        def fill(text: str) -> str:
            return re.sub(r"\{\{(\w+)\}\}", lambda m: str(values.get(m[1], m[0])), text)

        prompt = self._prompt
        if isinstance(prompt, str):
            prompt = [("user", prompt)]
        return [{"role": role, "content": fill(text)} for role, text in prompt]

    def _run(self, ctx: Any, results: dict) -> Any:
        from langchain.chat_models import init_chat_model

        schema = self._output_schema or {
            "title": "result",
            "type": "object",
            "properties": {self._feedback_key: {"type": "number"}},
            "required": [self._feedback_key],
        }
        if isinstance(schema, dict):
            schema = {"title": "result", **schema}
        result = (
            init_chat_model(self._model)
            .with_structured_output(schema)
            .invoke(self._messages(ctx, results))
        )
        # A Pydantic schema gives a model instance; TypedDicts and dicts give a dict.
        return result.model_dump() if hasattr(result, "model_dump") else result


def Noul(instructions: str) -> dict[str, Any]:
    """A yes/no question; the answer is the probability of yes."""
    return {"type": "noul", "instructions": instructions}


def Score(instructions: str, levels: list[Any]) -> dict[str, Any]:
    """A scale; the answer is a fractional index into `levels`, lowest first."""
    if not 2 <= len(levels) <= 10:
        raise ValueError("Score needs 2 to 10 levels")
    return {
        "type": "score",
        "instructions": instructions,
        "levels": [str(level) for level in levels],
    }


def Choice(
    instructions: str, options: Union[list[str], dict[str, Optional[str]]]
) -> dict[str, Any]:
    """One of `options`; the answer is the chosen option."""
    if len(options) < 2:
        raise ValueError("Choice needs at least 2 options")
    options = options if isinstance(options, dict) else dict.fromkeys(options)
    return {"type": "choice", "instructions": instructions, "options": options}


def _ask_typesafe(model: str, state: dict, questions: dict) -> dict:
    from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

    native = {
        "noul": lambda q: Noul(instructions=q["instructions"]),
        "score": lambda q: Score(instructions=q["instructions"], criteria=q["levels"]),
        "choice": lambda q: Choice(
            instructions=q["instructions"], criteria=q["options"]
        ),
    }
    response = TypeSafeClassifier(model=model).invoke(
        {
            "state": state,
            "questions": {k: native[q["type"]](q) for k, q in questions.items()},
        }
    )
    answers = {k: a.noul for k, a in response.nouls.items()}
    answers.update({k: a.score for k, a in response.scores.items()})
    answers.update({k: a.choice for k, a in response.choices.items()})
    return answers


def _ask_openai(model: str, state: dict, questions: dict) -> dict:
    from langchain_openai.decisions import Choice, OpenAIDecisions, Predicate, Score

    native = {
        "noul": lambda q: Predicate(instructions=q["instructions"]),
        "score": lambda q: Score(instructions=q["instructions"], levels=q["levels"]),
        "choice": lambda q: Choice(
            instructions=q["instructions"], choices=q["options"]
        ),
    }
    response = OpenAIDecisions(model=model).invoke(
        {
            "input": state,
            "questions": {k: native[q["type"]](q) for k, q in questions.items()},
        }
    )
    answers = {k: a.probability for k, a in response.predicates.items()}
    answers.update({k: a.score for k, a in response.scores.items()})
    answers.update({k: a.choice for k, a in response.choices.items()})
    return answers


class DecisionModel(_Node):
    """TypeSafe's Jev by default, or OpenAI Decisions with an `openai:` model."""

    _run_type = "llm"

    def __init__(
        self,
        *,
        questions: dict[str, dict[str, Any]],
        inputs: Optional[dict[str, _Ref]] = None,
        model: str = "typesafe:jev-latest",
        name: str = "DecisionModel",
        **_: Any,
    ) -> None:
        self._model, self._questions, self._inputs = model, questions, inputs or {}
        self._outputs, self._name = set(questions), name

    def _run(self, ctx: Any, results: dict) -> dict:
        provider, _, name = self._model.rpartition(":")
        ask = _ask_openai if provider == "openai" else _ask_typesafe
        return ask(name, self._values(ctx, results), self._questions)


class Graph:
    def __init__(self, nodes: list[_Node]) -> None:
        # Checked here so a mistake fails before any model is called.
        for step in nodes:
            for name, src in step._sources():
                if src not in nodes and src not in _CONTEXT_NAMES:
                    raise ValueError(f"{name!r} is not a context or a listed node")
            for ref in step._inputs.values():
                keys = ref._root._outputs if isinstance(ref._root, _Node) else None
                if keys is not None and ref._path and ref._path[0] not in keys:
                    raise ValueError(f"No output {ref._path[0]!r} in {sorted(keys)}")
        self.order: list[_Node] = []

        def place(step: _Node, path: tuple = ()) -> None:
            if step in path:
                raise ValueError("Graph([...]) has a cycle")
            if step not in self.order:
                for _, src in step._sources():
                    if isinstance(src, _Node):
                        place(src, (*path, step))
                self.order.append(step)

        for step in nodes:
            place(step)

    def run(self, ctx: Any) -> dict[_Node, Any]:
        """Runs each node after the nodes it reads; returns each node's outputs."""
        results: dict[_Node, Any] = {}
        for step in self.order:
            results[step] = step._traced_run(ctx, results)
        return results
