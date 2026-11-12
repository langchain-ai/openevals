"""Evaluators built as a graph of nodes. Experimental."""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace
from typing import Any, Callable, Optional, Union

from langchain_core.decisions import BaseDecisionModel, Choice, Noul, Question, Score
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
    if name == "ctx":
        return ctx
    value = getattr(ctx, name)
    # LangSmith passes `Run` and `Example` objects to local evaluators, dicts to uploaded ones.
    return value.model_dump() if hasattr(value, "model_dump") else value


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


class DecisionModel(_Node):
    """Asks a decision model typed questions about its inputs.

    `model` is a `BaseDecisionModel`, or a `typesafe:` model name like
    `typesafe:jev-latest`.
    """

    _run_type = "llm"

    def __init__(
        self,
        *,
        model: Union[str, BaseDecisionModel],
        questions: dict[str, Question],
        inputs: Optional[dict[str, _Ref]] = None,
        name: str = "DecisionModel",
        **_: Any,
    ) -> None:
        self._model, self._questions, self._inputs = model, questions, inputs or {}
        self._outputs, self._name = set(questions), name

    def _decision_model(self) -> BaseDecisionModel:
        if isinstance(self._model, BaseDecisionModel):
            return self._model
        provider, _, name = self._model.partition(":")
        if provider != "typesafe":
            raise ValueError(
                f"Unknown decision model {self._model!r}; pass a BaseDecisionModel"
            )
        try:
            from langchain_typesafe import TypeSafeClassifier
        except ImportError as e:
            raise ImportError(
                "typesafe: models need langchain-typesafe; "
                'install it with `pip install "openevals[typesafe]"`'
            ) from e
        return TypeSafeClassifier(model=name)

    def _run(self, ctx: Any, results: dict) -> dict:
        response = self._decision_model().invoke(
            {"state": self._values(ctx, results), "questions": self._questions}
        )
        return {key: getattr(a, a.type) for key, a in response.answers.items()}


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

    def run(self, ctx: Any = None, /, **context: Any) -> dict[_Node, Any]:
        """Runs each node after the nodes it reads; returns each node's outputs.

        Pass the context as keywords, like `graph.run(run=run, example=example)`,
        or as one object with those attributes.
        """
        if ctx is None:
            ctx = SimpleNamespace(**context)
        results: dict[_Node, Any] = {}
        for step in self.order:
            results[step] = step._traced_run(ctx, results)
        return results

    def as_evaluator(
        self, node: _Node, *, key: Optional[str] = None
    ) -> Callable[..., dict[str, Any]]:
        """A LangSmith evaluator that runs the graph and reports `node`'s outputs.

        The feedback is `node`'s `score` (and `comment`, if any), or its only
        output. Without `key`, LangSmith names the feedback: after the
        evaluator in LangSmith, after `node` in local `evaluate()`.
        """
        if node not in self.order:
            raise ValueError("as_evaluator needs a node in this graph")

        def evaluator(run: Any, example: Any = None) -> dict[str, Any]:
            outputs = self.run(run=run, example=example)[node]
            if "score" in outputs:
                score = outputs["score"]
            elif len(outputs) == 1:
                [score] = outputs.values()
            else:
                raise ValueError(
                    f"{node._name!r} returns {sorted(outputs)}; "
                    "return a 'score' to use it as the evaluator's result"
                )
            result = {"score": score}
            if "comment" in outputs:
                result["comment"] = outputs["comment"]
            if key is not None:
                result["key"] = key
            return result

        evaluator.__name__ = node._name
        return evaluator
