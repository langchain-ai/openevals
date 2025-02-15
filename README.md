# LangSmith Evaluators

Much like unit tests in traditional software, evals are a hugely important part of bringing LLM applications to production.
The goal of this package is to help provide a starting point for you to write evals for your LLM applications. It is not
intended to be a highly customizable framework, but rather a jumping off point before you get into more complicated evals specific to your application.
To learn more about how to write more custom evals, please check out this [documentation](https://docs.smith.langchain.com/evaluation/how_to_guides/custom_evaluator).

## LLM-as-Judge Evaluators

The core of this package is the `create_llm_as_judge` function, which creates a simple evaluator that uses an LLM to evaluate the quality of the outputs. The scoring is done
from 0 to 1, where 1 indicates a pass and 0 indicates a fail.

To use the `create_llm_as_judge` function, you need to provide a prompt and a model. The prompt is a string or a function that returns a string. The model can either
be a string of `PROVIDER:MODEL` such as `openai:gpt-4o`, a LangChain-like model such as `ChatOpenAI(model="gpt-4o")`, or an OpenAI client along with a model name.

The following example creates a simple evaluator that uses an LLM to evaluate how funny the outputs are:

```python
from langsmith.evaluators.llm_as_judge import create_llm_as_judge

funny_evaluator = create_llm_as_judge(
    # IMPORTANT: Your prompt can only contain the variables `inputs`, `outputs`, and `reference_outputs`
    prompt="Is this text funny? {outputs}",
    model="openai:gpt-4o",
    # reasoning defaults to True, and forces the model to use CoT
    reasoning=True,
    # Set a threshold to make the output binary
    threshold=0.5,
)
```

You can then write a simple unit test with the evaluator as such:

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

### Aside: Using with Evaluate

You can also use the evaluator with the [`evaluate`](https://docs.smith.langchain.com/evaluation#8-run-and-view-results) function from LangSmith:

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

We offer a variety of prebuilt LLM-as-Judge evaluator prompts. You can import and use them as follows:

```python
from langsmith.evaluators.prompts import *

conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    model="openai:gpt-4o",
)
```

### Prebuilt Trajectory Evaluator

LangSmith also offers a pre-built evaluator for evaluating the trajectory of a conversation against an expected trajectory. You can import and use it as follows:

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
