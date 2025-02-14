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
    # continuous defaults to False, and forces the model to return either 0 or 1 instead of a score between 0 and 1
    continuous=False,
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
        ChatCompletionMessage(role="user", content="What is the weather in SF?"),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant", content="The weather in SF is 80 degrees andsunny."
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(role="user", content="What is the weather in SF?"),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 80 degrees and sunny in San Francisco."
        ),
        ChatCompletionMessage(
            role="assistant", content="The weather in SF is 80˚ and sunny."
        ),
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
        # How to aggregate the feedback keys. Can be "average", "conjunction", or None
        # If None, feedback chips for each key (in this case "a" and "b") will be returned, else a single feedback chip will be returned with the key "structured_match_score"
        aggregator="average",
        # The criteria for the LLM judge to use for each key you want evaluated by the LLM
        rubric={
            "a": "Does the answer mention all the fruits in the reference answer?"
        }, 
        # The keys to ignore during evaluation. Any key not passed here or in `rubric` will be evaluated using an exact match comparison to the reference outputs
        exclude_keys=["c"],
        # The provider and name of the model to use, defaults to openai:o3-mini
        model="openai:gpt-4o",
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
        # How to aggregate the feedback keys across elements of the list. Can be "average" or "conjunction". Defaults to "conjunction". If "conjunction", the score for each key will be the conjunction of the scores for that key across all elements of the list. If "average", the score for each key will be the average of the scores for that key across all elements of the list
        list_aggregator="conjunction",
        # How to aggregate the feedback keys for each object in the list. Can be "average", "conjunction", or None
        # If None, feedback chips for each key (in this case "a" and "b") will be returned, else a single feedback chip will be returned with the key "structured_match_score"
        aggregator="average",
        # The criteria for the LLM judge to use for each key you want evaluated by the LLM
        rubric={
            "a": "Does the answer mention all the fruits in the reference answer?"
        }, 
        # The keys to ignore during evaluation. Any key not passed here or in `rubric` will be evaluated using an exact match comparison to the reference outputs
        exclude_keys=["c"],
        # The provider and name of the model to use, defaults to openai:o3-mini
        model="openai:gpt-4o",
        # Whether to force the model to reason about the keys in `rubric`. Defaults to True
        use_reasoning=True
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "structured_match_score"
    # "a" will be 0 since the reference answer doesn't mention all the fruits in the output for the second list element, "b" will be 1 since it exact matches in all elements of the list, and "d" will be 0 since it is missing from the outputs.
    assert result["score"] == 1/3
```