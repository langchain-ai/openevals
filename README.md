# LangSmith Evaluators

Much like unit tests in traditional software, evals are a hugely important part of bringing LLM applications to production.
The goal of this package is to help provide a starting point for you to write evals for your LLM applications. It is not
intended to be a highly customizable framework, but rather a jumping off point before you get into more complicated evals specific to your application.
To learn more about how to write more custom evals, please check out this [documentation](https://docs.smith.langchain.com/evaluation/how_to_guides/custom_evaluator).

## Installation

You can install the package as follows:

```bash
$ uv add langsmith-evaluators
```

For LLM-as-judge evaluators, you will also need an LLM client. You can either use [LangChain chat models](https://python.langchain.com/docs/integrations/chat/):

```bash
$ uv add langchain langchain_openai
```

Or an OpenAI client directly:

```bash
$ uv add openai
```

## LLM-as-Judge Evaluators

The core of this package is the `create_llm_as_judge` function, which creates an evaluator that uses an LLM to evaluate the quality of the outputs.

To use the `create_llm_as_judge` function, you need to provide a prompt and a model. Here's an example of a simple evaluator that uses an LLM to evaluate how funny an output is:

```python
from langsmith.evaluators.llm_as_judge import create_llm_as_judge

funny_evaluator = create_llm_as_judge(
    prompt="Is this text funny? {outputs}",
    metric="hilariousness",
    # Optional, forces a binary output
    threshold=0.5,
    # Default, requires langchain and langchain_openai installation
    model="openai:o3-mini",
    # You can also pass an OpenAI client instance and model name
    # model="o3-mini",
    # judge=OpenAI(api_key="...")
)
```

The prompt may be a string, LangChain prompt or a function that returns a string.

The model can either be:

- a string of `PROVIDER:MODEL` such as `openai:o3-mini`, in which case the package will [attempt to import and initialize a LangChain chat model instance](https://python.langchain.com/docs/how_to/chat_models_universal_init/)
- a LangChain chat model instance such as `ChatOpenAI(model="o3-mini")`
- a string along with a `judge` parameter set to an OpenAI client instance

You may also omit the `model` parameter entirely, in which case the package will attempt to import `langchain_openai` and use OpenAI's `o3-mini` model.

You can then [set up LangSmith's pytest runner](https://docs.smith.langchain.com/evaluation/how_to_guides/pytest) and run a simple eval:

```python
import pytest

@pytest.mark.langsmith
def test_funniness():
    inputs = "Tell me a joke"
    # These are fake outputs, in reality you would run your LLM-based system and get real outputs
    outputs = "Why did the chicken cross the road? To get to the other side!"
    eval_result = funny_evaluator(outputs=outputs)
    assert eval_result["score"] == 1
    assert eval_result["reasoning"] is not None
```

You can also use your created evaluator with LangSmith's [`evaluate`](https://docs.smith.langchain.com/evaluation#8-run-and-view-results) function:

```python
from langsmith import Client

client = Client()

experiment_results = client.evaluate(
    # This is a dummy target function, replace with your actual LLM-based system
    lambda inputs: "Why did the chicken cross the road? To get to the other side!",
    data="Sample dataset",
    evaluators=[
        funny_evaluator
    ]
)
```

## Prebuilt Evaluators

In addition to providing a generator function for evaluators, LangSmith also provides a set of pre-built evaluators. These evaluators are meant
to be used out of the box for sanity checks during development, and should provide a good starting point for you to write your own custom evals.

### Prebuilt LLM-as-Judge's

We offer a variety of prebuilt LLM-as-a-judge evaluators in the form of prompts. You can import and use them as follows:

```python
from langsmith.evaluators.prompts import CONCISENESS_PROMPT

conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    metric="conciseness",
)
```

Notably, these prompts are simple strings. You can log them and make edits to them as needed.

The above example also omits the `threshold` parameter, in which case the evaluator will return a float between 0 and 1.

If a prompt requires additional environment variables, you can pass them in by name when calling the created evaluator.

### Prebuilt Trajectory Evaluators

LangSmith also offers pre-built evaluators for evaluating the trajectory of a conversation against an expected trajectory. Here's an example of how to use the `exact_trajectory_match` evaluator:

```python
from langsmith.evaluators.trajectory.exact import exact_trajectory_match

@pytest.mark.langsmith
def test_exact_trajectory_match():
    inputs = {}
    outputs = [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
        {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
    ]
    reference_outputs = [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in San Francisco."},
        {"role": "assistant", "content": "The weather in SF is 80˚ and sunny."},
    ]
    assert exact_trajectory_match(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key="trajectory_match", score=1.0)
```

There are other evaluators for checking partial trajectory matches (ensuring that a trajectory contains a subset and superset of tool calls compared to a reference trajectory),
as well as an LLM-as-judge trajectory evaluator that uses an LLM to evaluate the trajectory:

```python
@pytest.mark.langsmith
def test_trajectory_match():
    evaluator = create_trajectory_llm_as_judge(prompt=DEFAULT_PROMPT)
    inputs = {}
    outputs = [
        {"role": "user", "content": "What is the weather in SF?"},
        {"role": "assistant", "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
        {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
    ]
    reference_outputs = [
        {"role": "user", "content": "What is the weather in SF?"},
        {"role": "assistant", "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in San Francisco."},
        {"role": "assistant", "content": "The weather in SF is 80˚ and sunny."},
    ]
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs,
    )
    assert eval_result["key"] == "trajectory_accuracy"
    assert eval_result["score"] == 1.0
```

### Prebuilt Metric Evaluators

LangSmith also provides a variety of prebuilt evaluators for calculating common metrics such as Levenshtein distance, exact match, etc. You can import and use them as follows:

#### Exact Match

```python
from langsmith.evaluators.exact import exact_match

@pytest.mark.langsmith
def test_exact_matcher():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    assert exact_match(inputs=inputs, outputs=outputs) == EvaluatorResult(
        key="equal", score=1.0
    )
```

#### Levenshtein Distance

```python
from langsmith.evaluators.levenshetein import levenshtein_distance

@pytest.mark.langsmith
def test_levenshtein_distance():
    outputs = "The correct answer"
    reference_outputs = "The correct answer"
    eval_result = levenshtein_distance(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    assert eval_result["score"] == 0
```

### Prebuilt Extraction/Tool Call evaluators

Two very common use cases for LLMs are extracting structured output from documents and tool calling. Both of these require the LLM
to respond in a structured format. LangSmith provides a prebuilt evaluator to help you evaluate these use cases, and is flexible
to work for a variety of extraction/tool calling use cases.

#### Evaluating a single structured output

Here is a code example of how to evaluate a single structured output, with comments explaining every parameter:

```python
import pytest
from langsmith.evaluators.json import json_match_evaluator

@pytest.mark.langsmith
def test_json_match_mix():
    outputs = {"a": "Mango, Bananas", "b": 2, "c": [1,2,3]}
    reference_outputs = {"a": "Bananas, Mango", "b": 3, "c": [1,2,3]}
    evaluator = json_match_evaluator(
        # How to aggregate the feedback keys. Can be "average", "all", or None
        # If None, feedback chips for each key (in this case "a" and "b") will be returned, else a single feedback chip will be returned with the key "structured_match_score"
        aggregator="average",
        # The criteria for the LLM judge to use for each key you want evaluated by the LLM
        rubric={
            "a": "Does the answer mention all the fruits in the reference answer?"
        }, 
        # The keys to ignore during evaluation. Any key not passed here or in `rubric` will be evaluated using an exact match comparison to the reference outputs
        exclude_keys=["c"],
        # The provider and name of the model to use, defaults to openai:o3-mini
        model="openai:o3-mini",
        # Whether to force the model to reason about the keys in `rubric`. Defaults to True
        use_reasoning=True
    )
    # Invoke the evaluator with the outputs and reference outputs
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    # "b" will be 0, "a" will be 1
    assert result["score"] == 0.5
```

Here is a code example of how to evaluate a list of structured outputs, with comments explaining every parameter:

```python
import pytest
from langsmith.evaluators.json import json_match_evaluator

@pytest.mark.langsmith
def test_json_match_list_all_average():
    outputs = [
        {"a": "Mango, Bananas", "b": 2},
        {"a": "Apples", "b": 2, "c": [1,2,3]},
    ]
    reference_outputs = [
        {"a": "Bananas, Mango", "b": 2, "d": "Not in outputs"},
        {"a": "Apples, Strawberries", "b": 2},
    ]
    evaluator = json_match_evaluator(
        # How to aggregate the feedback keys across elements of the list. Can be "average" or "all". Defaults to "all". If "all", the score for each key will be a combined and statement of the scores for that key across all elements of the list. If "average", the score for each key will be the average of the scores for that key across all elements of the list
        list_aggregator="all",
        # How to aggregate the feedback keys for each object in the list. Can be "average", "all", or None
        # If None, feedback chips for each key (in this case "a" and "b") will be returned, else a single feedback chip will be returned with the key "structured_match_score"
        aggregator="average",
        # The criteria for the LLM judge to use for each key you want evaluated by the LLM
        rubric={
            "a": "Does the answer mention all the fruits in the reference answer?"
        }, 
        # The keys to ignore during evaluation. Any key not passed here or in `rubric` will be evaluated using an exact match comparison to the reference outputs
        exclude_keys=["c"],
        # The provider and name of the model to use, defaults to openai:o3-mini
        model="openai:o3-mini",
        # Whether to force the model to reason about the keys in `rubric`. Defaults to True
        use_reasoning=True
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    # "a" will be 0 since the reference answer doesn't mention all the fruits in the output for the second list element, "b" will be 1 since it exact matches in all elements of the list, and "d" will be 0 since it is missing from the outputs.
    assert result["score"] == 1/3
```
