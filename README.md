# LangSmith Evaluators

Much like unit tests in traditional software, evals are a hugely important part of bringing LLM applications to production.
The goal of this package is to help provide a starting point for you to write evals for your LLM applications, from which
you can write more custom evals specific to your application.

To learn more about how to write more custom evals, please check out this [documentation](https://docs.smith.langchain.com/evaluation/how_to_guides/custom_evaluator).

## Setup

Install this package like this:

```bash
$ uv add langsmith-evaluators
```

For LLM-as-judge evaluators, you will also need an LLM client. You can use [LangChain chat models](https://python.langchain.com/docs/integrations/chat/):

```bash
$ uv add langchain langchain_openai
```

Or an OpenAI client directly:

```bash
$ uv add openai
```

It is also helpful to be familiar with [evaluation concepts in LangSmith](https://docs.smith.langchain.com/evaluation/concepts) and
LangSmith's pytest integration for running evals, which is documented [here](https://docs.smith.langchain.com/evaluation/how_to_guides/pytest).

## Prebuilt evaluators

### LLM-as-judge

One common way to evaluate an LLM app's outputs is to use another LLM as a judge. This is generally a good starting point for evals.

This package contains the `create_llm_as_judge` function, which takes a prompt and a model as input, and returns an evaluator function
that handles formatting inputs, parsing the judge LLM's outputs into a score, and LangSmith tracing and result logging.

To use the `create_llm_as_judge` function, you need to provide a prompt and a model. For prompts, LangSmith has some prebuilt prompts
in the `langsmith.evaluators.prompts` module that you can use out of the box. Here's an example:

```python
from langsmith.evaluators.llm_as_judge import create_llm_as_judge
from langsmith.evaluators.prompts import CONCISENESS_PROMPT

conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    metric="conciseness",
)
```

Note that `CONCISENESS_PROMPT` is a simple f-string that you can log and edit as needed for your specific use case:

```python
print(CONCISENESS_PROMPT)
```

```
You are an expert data labeler evaluating model outputs for conciseness. Your task is to assign a score between 0 and 1, where:
- 1.0 represents perfect conciseness
- 0.0 represents extreme verbosity

<Rubric>
  A perfectly concise answer (score = 1.0):
  ...
```

You can then [set up LangSmith's pytest runner](https://docs.smith.langchain.com/evaluation/how_to_guides/pytest) and run a simple eval:

```python
import pytest

@pytest.mark.langsmith
def test_conciseness():
    inputs = "What color is the sky?"
    # These are fake outputs, in reality you would run your LLM-based system and get real outputs
    outputs = "Blue."
    # outputs are formatted directly into the prompt
    eval_result = conciseness_evaluator(outputs=outputs)
    assert eval_result["score"] == 1
    assert eval_result["comment"] is not None
```

Or use your created evaluator with LangSmith's [`evaluate`](https://docs.smith.langchain.com/evaluation#8-run-and-view-results) function:

```python
from langsmith import Client

client = Client()

experiment_results = client.evaluate(
    # This is a dummy target function, replace with your actual LLM-based system
    lambda inputs: "What color is the sky?",
    data="Sample dataset",
    evaluators=[
        conciseness_evaluator
    ]
)
```

Prompts can also require additional inputs. In this case, you would pass extra kwargs when calling your evaluator function.

#### Customization

The `prompt` may be an f-string, LangChain prompt template, or a function that returns a string.

If you don't populate a `model` or `judge` parameter, the `create_llm_as_judge` function will default to OpenAI's `o3-mini` model
through LangChain's `ChatOpenAI` class, which requires you to install the `langchain_openai` package. Alternatively, you can:

- pass a string formatted as `PROVIDER:MODEL` (e.g. `model=anthropic:claude-3-5-sonnet-latest`) as the `model`, in which case the package will [attempt to import and initialize a LangChain chat model instance](https://python.langchain.com/docs/how_to/chat_models_universal_init/). This requires you to install the appropriate LangChain integration package installed
- directly pass a LangChain chat model instance as `judge` (e.g. `judge=ChatAnthropic(model="claude-3-5-sonnet-latest")`)
- pass a model name as `model` and a `judge` parameter set to an OpenAI client instance (e.g. `model="gpt-4o-mini", judge=OpenAI()`)

There are some additional fields that you can set as well:

- `threshold`: a float between 0 and 1 that sets the threshold for the evaluator. If set, changes the evaluator to only return 0 or 1 depending on whether the true evaluator score is above or below the threshold
- `use_reasoning`: a boolean that sets whether the evaluator should also output a `reasoning` field with the evaluator's reasoning behind assigning a score. Defaults to `True`.
- `few_shot_examples`: a list of example dicts that are appended to the end of the prompt. This is useful for providing the judge model with examples of good and bad outputs.

### Agent trajectory evaluators

LangSmith also offers pre-built evaluators for evaluating the trajectory of an agent's execution against an expected one.
You can format your agent's trajectory as a list of OpenAI format dicts or as a list of LangChain `BaseMessage` classes.

Here's an example of how to use the `trajectory_strict_match` evaluator, which compares two trajectories and
ensures that they contain the same messages in the same order with the same tool calls. It allows for differences in message content and tool call args
to allow for some variation in model outputs.

```python
from langsmith.evaluators.trajectory.strict import trajectory_strict_match

@pytest.mark.langsmith
def test_trajectory_strict_match():
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
        {"role": "user", "content": "What is the weather in San Francisco?"},
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
    result = trajectory_strict_match(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    assert result["score"] == 1.0
```

There are other evaluators for checking partial trajectory matches (ensuring that a trajectory contains a subset and superset of tool calls compared to a reference trajectory),
as well as an LLM-as-judge trajectory evaluator that uses an LLM to evaluate the trajectory. This allows for more flexibility in the trajectory comparison:

```python
from langsmith.evaluators.trajectory.llm import create_trajectory_llm_as_judge, DEFAULT_PROMPT

@pytest.mark.langsmith
def test_trajectory_llm_as_judge():
    # Also defaults to using OpenAI's o3-mini model through LangChain's ChatOpenAI class
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
    assert eval_result["score"] == 1.0
```

`create_trajectory_llm_as_judge` takes the same parameters as `create_llm_as_judge`, so you can customize the prompt, model, and judge as needed.

### Prebuilt metric evaluators

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
