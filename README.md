# ⚖️ OpenEvals

Much like tests in traditional software, evals are a hugely important part of bringing LLM applications to production.
The goal of this package is to help provide a starting point for you to write evals for your LLM applications, from which
you can write more custom evals specific to your application.

If you are looking for evals specific to evaluating LLM agents, please check out [`agentevals`](https://github.com/langchain-ai/agentevals).

## Quickstart

**Note:** If you'd like to follow along with a video walkthrough, click the image below:

[![Video quickstart](https://img.youtube.com/vi/J-F30jRyhoA/0.jpg)](https://www.youtube.com/watch?v=J-F30jRyhoA)

To get started, install `openevals`:

<details open>
<summary>Python</summary>

```bash
pip install openevals
```
</details>

<details>
<summary>TypeScript</summary>

```bash
npm install openevals @langchain/core
```
</details>

This quickstart will use an evaluator powered by OpenAI's `o3-mini` model to judge your results, so you'll need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Once you've done this, you can run your first eval:

<details open>
<summary>Python</summary>

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="openai:o3-mini",
)

inputs = "How much has the price of doodads changed in the past year?"
# These are fake outputs, in reality you would run your LLM-based system to get real outputs
outputs = "Doodads have increased in price by 10% in the past year."
reference_outputs = "The price of doodads has decreased by 50% in the past year."
# When calling an LLM-as-judge evaluator, parameters are formatted directly into the prompt
eval_result = correctness_evaluator(
  inputs=inputs,
  outputs=outputs,
  reference_outputs=reference_outputs
)

print(eval_result)
```

```
{
    'key': 'score',
    'score': False,
    'comment': 'The provided answer stated that doodads increased in price by 10%, which conflicts with the reference output...'
}
```
</details>

<details>
<summary>TypeScript</summary>

```ts
import { createLLMAsJudge, CORRECTNESS_PROMPT } from "openevals";

const correctnessEvaluator = createLLMAsJudge({
  prompt: CORRECTNESS_PROMPT,
  model: "openai:o3-mini",
});

const inputs = "How much has the price of doodads changed in the past year?"
// These are fake outputs, in reality you would run your LLM-based system to get real outputs
const outputs = "Doodads have increased in price by 10% in the past year."
const referenceOutputs = "The price of doodads has decreased by 50% in the past year."

// When calling an LLM-as-judge evaluator, parameters are formatted directly into the prompt
const evalResult = await correctnessEvaluator({
  inputs,
  outputs,
  referenceOutputs,
});

console.log(evalResult);
```

```
{
    key: 'score',
    score: false,
    comment: '...'
}
```
</details>

By default, LLM-as-judge evaluators will return a score of `True` or `False`. See the [LLM-as-judge](#llm-as-judge) section for more information on how to customize the [scoring](#customizing-output-scores), [model](#customizing-the-model), and [prompt](#customizing-prompts)!

## Table of Contents

- [Installation](#installation)
- [Evaluators](#evaluators)
  - [LLM-as-Judge](#llm-as-judge)
    - [Prebuilt prompts](#prebuilt-prompts)
      - [Correctness](#correctness)
      - [Conciseness](#conciseness)
      - [Hallucination](#hallucination)
    - [Customizing prompts](#customizing-prompts)
    - [Customizing the model](#customizing-the-model)
    - [Customizing output scores](#customizing-output-scores)
  - [Extraction and tool calls](#extraction-and-tool-calls)
    - [Evaluating structured output with exact match](#evaluating-structured-output-with-exact-match)
    - [Evaluating structured output with LLM-as-a-Judge](#evaluating-structured-output-with-llm-as-a-judge)
  - [Code](#code)
    - [Pyright (Python-only)](#pyright-python-only)
    - [Mypy (Python-only)](#mypy-python-only)
    - [TypeScript (TypeScript-only)](#typescript-typescript-only)
    - [LLM-as-judge for code](#llm-as-judge-for-code)
  - [Other](#other)
    - [Exact Match](#exact-match)
    - [Levenshtein Distance](#levenshtein-distance)
    - [Embedding Similarity](#embedding-similarity)
  - [Agent evals](#agent-evals)
  - [Creating your own](#creating-your-own)
- [Python Async Support](#python-async-support)
- [LangSmith Integration](#langsmith-integration)
  - [Pytest or Vitest/Jest](#pytest-or-vitestjest)
  - [Evaluate](#evaluate)

## Installation

You can install `openevals` like this:

<details open>
<summary>Python</summary>

```bash
pip install openevals
```
</details>

<details>
<summary>TypeScript</summary>

```bash
npm install openevals @langchain/core
```
</details>

For LLM-as-judge evaluators, you will also need an LLM client. By default, `openevals` will use [LangChain chat model integrations](https://python.langchain.com/docs/integrations/chat/) and comes with `langchain_openai` installed by default. However, if you prefer, you may use the OpenAI client directly:

<details open>
<summary>Python</summary>

```bash
pip install openai
```
</details>

<details>
<summary>TypeScript</summary>

```bash
npm install openai
```
</details>

It is also helpful to be familiar with some [evaluation concepts](https://docs.smith.langchain.com/evaluation/concepts).

## Evaluators

### LLM-as-judge

One common way to evaluate an LLM app's outputs is to use another LLM as a judge. This is generally a good starting point for evals.

This package contains the `create_llm_as_judge` function, which takes a prompt and a model as input, and returns an evaluator function
that handles formatting inputs, parsing the judge LLM's outputs into a score, and LangSmith tracing and result logging.

To use the `create_llm_as_judge` function, you need to provide a prompt and a model. For prompts, LangSmith has some prebuilt prompts
in the `openevals.prompts` module that you can use out of the box. Here's an example:

<details open>
<summary>Python</summary>

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="openai:o3-mini",
)
```
</details>

<details>
<summary>TypeScript</summary>

```ts
import { createLLMAsJudge, CORRECTNESS_PROMPT } from "openevals";

const correctnessEvaluator = createLLMAsJudge({
  prompt: CORRECTNESS_PROMPT,
  model: "openai:o3-mini",
});
```

</details>

Note that `CORRECTNESS_PROMPT` is a simple f-string that you can log and edit as needed for your specific use case:

<details open>
<summary>Python</summary>

```python
print(CORRECTNESS_PROMPT)
```

```
You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score based on the following rubric:

<Rubric>
  A correct answer:
  - Provides accurate and complete information
  ...
<input>
{inputs}
</input>

<output>
{outputs}
</output>
...
```
</details>

<details>
<summary>TypeScript</summary>

```ts
console.log(CORRECTNESS_PROMPT);
```

```
You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score based on the following rubric:

<Rubric>
  A correct answer:
  - Provides accurate and complete information
  ...
<input>
{inputs}
</input>

<output>
{outputs}
</output>
...
```

</details>

By convention, we generally suggest sticking to `inputs`, `outputs`, and `reference_outputs` as the names of the parameters for LLM-as-judge evaluators, but these will be directly formatted into the prompt so you can use any variable names you want.

### Prebuilt prompts

#### Correctness

`openevals` includes a prebuilt prompt for `create_llm_as_judge` that scores the correctness of an LLM's output. It takes `inputs`, `outputs`, and optionally, `reference_outputs` as parameters.

<details open>
<summary>Python</summary>

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:o3-mini",
)

inputs = "How much has the price of doodads changed in the past year?"
outputs = "Doodads have increased in price by 10% in the past year."
reference_outputs = "The price of doodads has decreased by 50% in the past year."

eval_result = correctness_evaluator(
  inputs=inputs,
  outputs=outputs,
  reference_outputs=reference_outputs
)

print(eval_result)
```

```
{
    'key': 'correctness',
    'score': False,
    'comment': '...'
}
```
</details>

<details>
<summary>TypeScript</summary>

```ts
import { createLLMAsJudge, CORRECTNESS_PROMPT } from "openevals";

const correctnessEvaluator = createLLMAsJudge({
  prompt: CORRECTNESS_PROMPT,
  feedbackKey: "correctness",
  model: "openai:o3-mini",
});

const inputs = "How much has the price of doodads changed in the past year?"
const outputs = "Doodads have increased in price by 10% in the past year."
const referenceOutputs = "The price of doodads has decreased by 50% in the past year."

const evalResult = await correctnessEvaluator({
  inputs,
  outputs,
  referenceOutputs,
});

console.log(evalResult);
```

```
{
    key: 'correctness',
    score: false,
    comment: '...'
}
```
</details>

#### Conciseness

`openevals` includes a prebuilt prompt for `create_llm_as_judge` that scores the conciseness of an LLM's output. It takes `inputs` and `outputs` as parameters.

<details open>
<summary>Python</summary>

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CONCISENESS_PROMPT

inputs = "How is the weather in San Francisco?"
outputs = "Thanks for asking! The current weather in San Francisco is sunny and 90 degrees."

llm_as_judge = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    feedback_key="conciseness",
    model="openai:o3-mini",
)

eval_result = llm_as_judge(inputs=inputs, outputs=outputs)

print(eval_result)
```

```
{
    'key': 'conciseness',
    'score': False,
    'comment': '...'
}
```
</details>

<details>
<summary>TypeScript</summary>

```ts
import { createLLMAsJudge, CONCISENESS_PROMPT } from "openevals";

const concisenessEvaluator = createLLMAsJudge({
  prompt: CONCISENESS_PROMPT,
  feedbackKey: "conciseness",
  model: "openai:o3-mini",
});

const inputs = "How is the weather in San Francisco?"
const outputs = "Thanks for asking! The current weather in San Francisco is sunny and 90 degrees."

const evalResult = await concisenessEvaluator({
  inputs,
  outputs,
});

console.log(evalResult);
```

```
{
    key: 'conciseness',
    score: false,
    comment: '...'
}
```

</details>

#### Hallucination

`openevals` includes a prebuilt prompt for `create_llm_as_judge` that scores the hallucination of an LLM's output. It takes `inputs`, `outputs`, and optionally, `context` as parameters.

<details open>
<summary>Python</summary>

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import HALLUCINATION_PROMPT

inputs = "What is a doodad?"
outputs = "I know the answer. A doodad is a kitten."
context = "A doodad is a self-replicating swarm of nanobots. They are extremely dangerous and should be avoided at all costs. Some safety precautions when working with them include wearing gloves and a mask."

llm_as_judge = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    feedback_key="hallucination",
    model="openai:o3-mini",
)

eval_result = llm_as_judge(inputs=inputs, outputs=outputs, context=context)
```

```
{
    'key': 'hallucination',
    'score': False,
    'comment': '...'
}
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { createLLMAsJudge, HALLUCINATION_PROMPT } from "openevals";

const hallucinationEvaluator = createLLMAsJudge({
  prompt: HALLUCINATION_PROMPT,
  feedbackKey: "hallucination",
  model: "openai:o3-mini",
});

const inputs = "What is a doodad?"
const outputs = "I know the answer. A doodad is a kitten."
const context = "A doodad is a self-replicating swarm of nanobots. They are extremely dangerous and should be avoided at all costs. Some safety precautions when working with them include wearing gloves and a mask."

const evalResult = await hallucinationEvaluator({
  inputs,
  outputs,
  context,
});

console.log(evalResult);
```

```
{
    key: 'hallucination',
    score: false,
    comment: '...'
}
```
</details>

#### Customizing prompts

The `prompt` parameter for `create_llm_as_judge` may be an f-string, LangChain prompt template, or a function that takes kwargs and returns a list of formatted messages.

Though we suggest sticking to conventional names (`inputs`, `outputs`, and `reference_outputs`) as prompt variables, you can also require additional variables. You would then pass these extra variables when calling your evaluator function. Here's an example:

<details open>
<summary>Python</summary>

```python
from openevals.llm import create_llm_as_judge

MY_CUSTOM_PROMPT = """
Use the following context to help you evaluate for hallucinations in the output:

<context>
{context}
</context>

<input>
{inputs}
</input>

<output>
{outputs}
</output>
"""

custom_prompt_evaluator = create_llm_as_judge(
    prompt=MY_CUSTOM_PROMPT,
    model="openai:o3-mini",
)

custom_prompt_evaluator(
    inputs="What color is the sky?",
    outputs="The sky is red.",
    context="It is early evening.",
)
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { createLLMAsJudge } from "openevals";

const MY_CUSTOM_PROMPT = `
Use the following context to help you evaluate for hallucinations in the output:

<context>
{context}
</context>

<input>
{inputs}
</input>

<output>
{outputs}
</output>
`;

const customPromptEvaluator = createLLMAsJudge({
  prompt: MY_CUSTOM_PROMPT,
  model: "openai:o3-mini",
});

const inputs = "What color is the sky?"
const outputs = "The sky is red."

const evalResult = await customPromptEvaluator({
  inputs,
  outputs,
});
```
</details>


For convenience, the following options are also available:

- `system`: a string that sets a system prompt for the judge model by adding a `system` message before other parts of the prompt.
- `few_shot_examples`: a list of example dicts that are appended to the end of the prompt. This is useful for providing the judge model with examples of good and bad outputs. The required structure looks like this:

<details open>
<summary>Python</summary>

```python
few_shot_examples = [
    {
        "inputs": "What color is the sky?",
        "outputs": "The sky is red.",
        "reasoning": "The sky is red because it is early evening.",
        "score": 1,
    }
]
```

</details>

<details>
<summary>TypeScript</summary>

```ts
const fewShotExamples = [
    {
        inputs: "What color is the sky?",
        outputs: "The sky is red.",
        reasoning: "The sky is red because it is early evening.",
        score: 1,
    }
]
```
</details>

These will be appended to the end of the final user message in the prompt.

#### Customizing the model

There are a few ways you can customize the model used for evaluation. You can pass a string formatted as `PROVIDER:MODEL` (e.g. `model=anthropic:claude-3-5-sonnet-latest`) as the `model`, in which case the package will [attempt to import and initialize a LangChain chat model instance](https://python.langchain.com/docs/how_to/chat_models_universal_init/). This requires you to install the appropriate LangChain integration package installed. Here's an example:

<details open>
<summary>Python</summary>

```bash
pip install langchain-anthropic
```

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

anthropic_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="anthropic:claude-3-5-sonnet-latest",
)
```

</details>

<details>
<summary>TypeScript</summary>

```bash
npm install @langchain/anthropic
```

```ts
import { createLLMAsJudge, CORRECTNESS_PROMPT } from "openevals";

const anthropicEvaluator = createLLMAsJudge({
  prompt: CORRECTNESS_PROMPT,
  model: "anthropic:claude-3-5-sonnet-latest",
});
```
</details>

You can also directly pass a LangChain chat model instance as `judge`. Note that your chosen model must support [structured output](https://python.langchain.com/docs/integrations/chat/):

<details open>
<summary>Python</summary>

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from langchain_anthropic import ChatAnthropic

anthropic_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    judge=ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.5),
)
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { createLLMAsJudge, CORRECTNESS_PROMPT } from "openevals";
import { ChatAnthropic } from "@langchain/anthropic";

const anthropicEvaluator = createLLMAsJudge({
  prompt: CORRECTNESS_PROMPT,
  judge: new ChatAnthropic({ model: "claude-3-5-sonnet-latest", temperature: 0.5 }),
});
```
</details>

This is useful in scenarios where you need to initialize your model with specific parameters, such as `temperature` or alternate URLs if using models through a service like Azure.

Finally, you can pass a model name as `model` and a `judge` parameter set to an OpenAI client instance:

<details open>
<summary>Python</summary>

```bash
pip install openai
```

```python
from openai import OpenAI

from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

openai_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="gpt-4o-mini",
    judge=OpenAI(),
)
```

</details>

<details>
<summary>TypeScript</summary>

```bash
npm install openai
```

```ts
import { OpenAI } from "openai";
import { createLLMAsJudge, CORRECTNESS_PROMPT } from "openevals";

const openaiEvaluator = createLLMAsJudge({
  prompt: CORRECTNESS_PROMPT,
  model: "gpt-4o-mini",
  judge: new OpenAI(),
});
```
</details>

#### Customizing output scores

There are two fields you can set to customize the output of your evaluator:

- `continuous`: a boolean that sets whether the evaluator should return a float score somewhere between 0 and 1 instead of a binary score. Defaults to `False`.
- `choices`: a list of floats that sets the possible scores for the evaluator.

These parameters are mutually exclusive. When using either of them, you should make sure that your prompt is grounded in information on what specific scores mean - the prebuilt ones in this repo do not have this information!

For example, here's an example of how to define a less harsh definition of correctness that only penalizes incorrect answers by 50% if they are on-topic:

<details open>
<summary>Python</summary>

```python
from openevals.llm import create_llm_as_judge

MY_CUSTOM_PROMPT = """
You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score based on the following rubric:

<Rubric>
  Assign a score of 0, .5, or 1 based on the following criteria:
  - 0: The answer is incorrect and does not mention doodads
  - 0.5: The answer mentions doodads but is otherwise incorrect
  - 1: The answer is correct and mentions doodads
</Rubric>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

<reference_outputs>
{reference_outputs}
</reference_outputs>
"""

evaluator = create_llm_as_judge(
    prompt=MY_CUSTOM_PROMPT,
    choices=[0.0, 0.5, 1.0],
    model="openai:o3-mini",
)

result = evaluator(
    inputs="What is the current price of doodads?",
    outputs="The price of doodads is $10.",
    reference_outputs="The price of doodads is $15.",
)

print(result)
```

```
{
    'key': 'score',
    'score': 0.5,
    'comment': 'The provided answer mentioned doodads but was incorrect.'
}
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { createLLMAsJudge } from "openevals";

const MY_CUSTOM_PROMPT = `
You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score based on the following rubric:

<Rubric>
  Assign a score of 0, .5, or 1 based on the following criteria:
  - 0: The answer is incorrect and does not mention doodads
  - 0.5: The answer mentions doodads but is otherwise incorrect
  - 1: The answer is correct and mentions doodads
</Rubric>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

<reference_outputs>
{reference_outputs}
</reference_outputs>
`;

const customEvaluator = createLLMAsJudge({
  prompt: MY_CUSTOM_PROMPT,
  choices: [0.0, 0.5, 1.0],
  model: "openai:o3-mini",
});

const result = await customEvaluator({
  inputs: "What is the current price of doodads?",
  outputs: "The price of doodads is $10.",
  reference_outputs: "The price of doodads is $15.",
});

console.log(result);
```

```
{
    'key': 'score',
    'score': 0.5,
    'comment': 'The provided answer mentioned doodads but was incorrect.'
}
```
</details>

Finally, if you would like to disable justifications for a given score, you can set `use_reasoning=False` when creating your evaluator.

### Extraction and tool calls

Two very common use cases for LLMs are extracting structured output from documents and tool calling. Both of these require the LLM
to respond in a structured format. This package provides a prebuilt evaluator to help you evaluate these use cases, and is flexible
to work for a variety of extraction/tool calling use cases.

You can use the `create_json_match_evaluator` evaluator in two ways:
1. To perform an exact match of the outputs to reference outputs
2. Using LLM-as-a-judge to evaluate the outputs based on a provided rubric.

Note that this evaluator may return multiple scores based on key and aggregation strategy, so the result will be an array of scores rather than a single one.

#### Evaluating structured output with exact match

Use exact match evaluation when there is a clear right or wrong answer. A common scenario is text extraction from images or PDFs where you expect specific values.

<details open>
<summary>Python</summary>

```python
from openevals.json import create_json_match_evaluator

outputs = [
    {"a": "Mango, Bananas", "b": 2},
    {"a": "Apples", "b": 2, "c": [1,2,3]},
]
reference_outputs = [
    {"a": "Mango, Bananas", "b": 2},
    {"a": "Apples", "b": 2, "c": [1,2,4]},
]
evaluator = create_json_match_evaluator(
    # How to aggregate feedback keys in each element of the list: "average", "all", or None
    # "average" returns the average score. "all" returns 1 only if all keys score 1; otherwise, it returns 0. None returns individual feedback chips for each key
    aggregator="all",
    # Remove if evaluating a single structured output. This aggregates the feedback keys across elements of the list. Can be "average" or "all". Defaults to "all". "all" returns 1 if each element of the list is 1; if any score is not 1, it returns 0. "average" returns the average of the scores from each element. 
    list_aggregator="average",
    exclude_keys=["a"],
)
# Invoke the evaluator with the outputs and reference outputs
result = evaluator(outputs=outputs, reference_outputs=reference_outputs)

print(result)
```

For the first element, "b" will be 1 and the aggregator will return a score of 1
For the second element, "b" will be 1, "c" will be 0 and the aggregator will return a score of 0
Therefore, the list aggregator will return a final score of 0.5.

```
[
  {
    'key': 'json_match:all',
    'score': 0.5,
    'comment': None,
  }
]
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { createJsonMatchEvaluator } from "openevals";
import { OpenAI } from "openai";

const outputs = [
    {a: "Mango, Bananas", b: 2},
    {a: "Apples", b: 2, c: [1,2,3]},
]
const reference_outputs = [
    {a: "Mango, Bananas", b: 2},
    {a: "Apples", b: 2, c: [1,2,4]},
]

const client = new OpenAI();

const evaluator = createJsonMatchEvaluator({
    // How to aggregate feedback keys in each element of the list: "average", "all", or None
    // "average" returns the average score. "all" returns 1 only if all keys score 1; otherwise, it returns 0. None returns individual feedback chips for each key
    aggregator="all",
    // Remove if evaluating a single structured output. This aggregates the feedback keys across elements of the list. Can be "average" or "all". Defaults to "all". "all" returns 1 if each element of the list is 1; if any score is not 1, it returns 0. "average" returns the average of the scores from each element. 
    list_aggregator="average",
    // The keys to ignore during evaluation. Any key not passed here or in `rubric` will be evaluated using an exact match comparison to the reference outputs
    exclude_keys=["a"],
    // The provider and name of the model to use
    judge: client,
    model: "openai:o3-mini",
})

// Invoke the evaluator with the outputs and reference outputs
const result = await evaluator({
    outputs,
    reference_outputs,
})

console.log(result)
```

For the first element, "b" will be 1 and the aggregator will return a score of 1
For the second element, "b" will be 1, "c" will be 0 and the aggregator will return a score of 0
Therefore, the list aggregator will return a final score of 0.5.

```
[
  {
    'key': 'json_match:all',
    'score': 0.5,
    'comment': None,
  }
]
```
</details>

#### Evaluating structured output with LLM-as-a-Judge

Use LLM-as-a-judge to evaluate structured output or tools calls when the criteria is more subjective (for example the output is a kind of fruit or mentions all the fruits). 


<details open>
<summary>Python</summary>

```python
from openevals.json import create_json_match_evaluator

outputs = [
    {"a": "Mango, Bananas", "b": 2},
    {"a": "Apples", "b": 2, "c": [1,2,3]},
]
reference_outputs = [
    {"a": "Bananas, Mango", "b": 2, "d": "Not in outputs"},
    {"a": "Apples, Strawberries", "b": 2},
]
evaluator = create_json_match_evaluator(
    # How to aggregate feedback keys in each element of the list: "average", "all", or None
    # "average" returns the average score. "all" returns 1 only if all keys score 1; otherwise, it returns 0. None returns individual feedback chips for each key
    aggregator="average",
    # Remove if evaluating a single structured output. This aggregates the feedback keys across elements of the list. Can be "average" or "all". Defaults to "all". "all" returns 1 if each element of the list is 1; if any score is not 1, it returns 0. "average" returns the average of the scores from each element. 
    list_aggregator="all",
    rubric={
        "a": "Does the answer mention all the fruits in the reference answer?"
    },
    # The provider and name of the model to use
    model="openai:o3-mini",
    # Whether to force the model to reason about the keys in `rubric`. Defaults to True
    # Note that this is not currently supported if there is an aggregator specified 
    use_reasoning=True
)
result = evaluator(outputs=outputs, reference_outputs=reference_outputs)

print(result)
```

For the first element, "a" will be 1  since both Mango and Bananas are in the reference output, "b" will be 1 and "d" will be 0. The aggregator will return an average score of 0.6. 
For the second element, "a" will be 0 since the reference output doesn't mention all the fruits in the output,  "b" will be 1. The aggregator will return a score of 0.5. 
Therefore, the list aggregator will return a final score of 0. 

```
[
  {
    'key': 'json_match:a',
    'score': 0,
    'comment': None
  }
]
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { createJsonMatchEvaluator } from "openevals";
import { OpenAI } from "openai";

const outputs = [
    {a: "Mango, Bananas", b: 2},
    {a: "Apples", b: 2, c: [1,2,3]},
]
const reference_outputs = [
    {a: "Bananas, Mango", b: 2},
    {a: "Apples, Strawberries", b: 2},
]

const client = new OpenAI();

const evaluator = createJsonMatchEvaluator({
    // How to aggregate feedback keys in each element of the list: "average", "all", or None
    // "average" returns the average score. "all" returns 1 only if all keys score 1; otherwise, it returns 0. None returns individual feedback chips for each key
    aggregator="average",
    // Remove if evaluating a single structured output. This aggregates the feedback keys across elements of the list. Can be "average" or "all". Defaults to "all". "all" returns 1 if each element of the list is 1; if any score is not 1, it returns 0. "average" returns the average of the scores from each element. 
    list_aggregator="all",
    // The criteria for the LLM judge to use for each key you want evaluated by the LLM
    rubric={
        a: "Does the answer mention all the fruits in the reference answer?"
    },
    // The keys to ignore during evaluation. Any key not passed here or in `rubric` will be evaluated using an exact match comparison to the reference outputs
    exclude_keys=["c"],
    // The provider and name of the model to use
    judge: client,
    model: "openai:o3-mini",
    // Whether to use reasoning to reason about the keys in `rubric`. Defaults to True
    useReasoning: true
})

// Invoke the evaluator with the outputs and reference outputs
const result = await evaluator({
    outputs,
    reference_outputs,
})

console.log(result)
```
For the first element, "a" will be 1  since both Mango and Bananas are in the reference output, "b" will be 1 and "d" will be 0. The aggregator will return an average score of 0.6. 
For the second element, "a" will be 0 since the reference output doesn't mention all the fruits in the output,  "b" will be 1. The aggregator will return a score of 0.5. 
Therefore, the list aggregator will return a final score of 0. 

```
{
  'key': 'json_match:a',
  'score': 0,
  'comment': None
}
```

</details>

### Code

OpenEvals contains some useful prebuilt evaluators for evaluating generated code:

- Type-checking generated code with [Pyright](https://github.com/microsoft/pyright) and [Mypy](https://github.com/python/mypy) (Python-only) or TypeScript's built-in type checker (JavaScript only)
- LLM-as-a-judge for code

Since LLM outputs with code may contain other text (for example, interleaved explanations with code), OpenEvals code evaluators share some built-in extraction methods for identifying just the code from of LLM outputs.

For any of the evaluators in this section, you can either pass a `code_extraction_strategy` param set to `llm`, which will use an `llm` with a default prompt to directly extract code, or `markdown_code_blocks`, which will extract anything in markdown code blocks (triple backticks). You can alternatively pass a `code_extractor` param set to a function that takes an LLM output and returns a string of code. The default is to leave the output content untouched (`"none"`).

If using `code_extraction_strategy="llm"`, you can also pass a `model` string or a `client` to the evaluator to set which evaluator the model uses for code extraction.
If you would like to customize the prompt, you should use the `code_extractor` param instead.

All evaluators in this section accept `outputs` as a string, an object with a key `"messages"` that contains a list of messages, or a message-like object with a key `"content"` that contains a string.

#### Pyright (Python-only)

For Pyright, you will need to install the `pyright` CLI on your system:

```bash
pip install pyright
```

You can find full installation instructions [here](https://microsoft.github.io/pyright/#/installation?id=command-line).

Then, you can use it as follows:

```python
from openevals.code.pyright import create_pyright_evaluator

evaluator = create_pyright_evaluator(
    code_extraction_strategy="markdown_code_blocks"
)

CODE = """
Sure! I can create a function that adds two numbers together. Here's the code:

\`\`\`python
def sum_of_two_numbers(a, b): return a + b
\`\`\`
"""

result = evaluator(outputs=CODE)

print(result)
```

```
{
    'key': 'pyright_succeeded',
    'score': True,
    'comment': None,
}
```

The evaluator will ignore `reportMissingImports` errors.

You can also pass `pyright_cli_args` to the evaluator to customize the arguments passed to the `pyright` CLI:

```python
evaluator = create_pyright_evaluator(
    pyright_cli_args=["--flag"]
)
```

For a full list of supported arguments, see the [pyright CLI documentation](https://microsoft.github.io/pyright/#/command-line).

#### Mypy (Python-only)

For Mypy, you will need to install `mypy` on your system:

```bash
pip install mypy
```

You can find full installation instructions [here](https://mypy.readthedocs.io/en/stable/getting_started.html).

Then, you can use it as follows:

```python
from openevals.code.mypy import create_mypy_evaluator

evaluator = create_mypy_evaluator(
    code_extraction_strategy="markdown_code_blocks"
)

CODE = """
Sure! I can create a function that adds two numbers together. Here's the code:

\`\`\`python
def sum_of_two_numbers(a, b): return a + b
\`\`\`
"""

result = evaluator(outputs=CODE)

print(result)
```

```
{
    'key': 'mypy_succeeded',
    'score': True,
    'comment': None,
}
```

By default, this evaluator will run with the following arguments:

```

mypy --no-incremental --disallow-untyped-calls --disallow-incomplete-defs --ignore-missing-imports
```

But you can pass `mypy_cli_args` to the evaluator to customize the arguments passed to the `mypy` CLI. This will override the default arguments:

```python
evaluator = create_mypy_evaluator(
    mypy_cli_args=["--flag"]
)
```

#### TypeScript (TypeScript-only)

// TODO

#### LLM-as-judge for code

OpenEvals includes a prebuilt LLM-as-a-judge evaluator for code. The primary differentiator between this one and the more generic [LLM-as-judge evaluator](#llm-as-judge) is that it will perform the extraction steps detailed above - otherwise it takes the same arguments, including a prompt.

You can run an LLM-as-a-judge evaluator for code as follows:

<details open>
<summary>Python</summary>

```python
from openevals.code.llm import create_code_llm_as_judge, CODE_CORRECTNESS_PROMPT

llm_as_judge = create_code_llm_as_judge(
    prompt=CODE_CORRECTNESS_PROMPT,
    model="openai:o3-mini",
    code_extraction_strategy="markdown_code_blocks",
)

CODE = """
Sure! Here's the code:

\`\`\`python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
\`\`\`
"""

eval_result = llm_as_judge(inputs="Generate a working web server in Python with FastAPI.", outputs=CODE)

print(eval_result)
```

```
{
    'key': 'code_correctness',
    'score': True,
    'comment': None,
}
```
</details>

### Other

This package also contains prebuilt evaluators for calculating common metrics such as Levenshtein distance, exact match, etc. You can import and use them as follows:

#### Exact match

<details open>
<summary>Python</summary>

```python
from openevals.exact import exact_match

outputs = {"a": 1, "b": 2}
reference_outputs = {"a": 1, "b": 2}
result = exact_match(outputs=outputs, reference_outputs=reference_outputs)

print(result)
```

```
{
    'key': 'equal',
    'score': True,
}
```
</details>

<details>
<summary>TypeScript</summary>

```ts
import { exactMatch } from "openevals";

const outputs = { a: 1, b: 2 };
const referenceOutputs = { a: 1, b: 2 };
const result = exactMatch(outputs, referenceOutputs);

console.log(result);
```

```
{
    key: "equal",
    score: true,
}
```
</details>

#### Levenshtein distance

<details open>
<summary>Python</summary>

```python
from openevals.string.levenshtein import levenshtein_distance

outputs = "The correct answer"
reference_outputs = "The correct answer"
result = levenshtein_distance(
    outputs=outputs, reference_outputs=reference_outputs,
)

print(result)
```

```
{
    'key': 'levenshtein_distance',
    'score': 0.0,
    'comment': None,
}
```
</details>

<details>
<summary>TypeScript</summary>

```ts
import { levenshteinDistance } from "openevals";

const outputs = "The correct answer";
const referenceOutputs = "The correct answer";
const result = levenshteinDistance(outputs, referenceOutputs);

console.log(result);
```

```
{
    key: "levenshtein_distance",
    score: 0,
}
```
</details>

#### Embedding similarity

This evaluator uses LangChain's [`init_embedding`](https://python.langchain.com/api_reference/langchain/embeddings/langchain.embeddings.base.init_embeddings.html) method (for Python) or takes a LangChain embeddings client directly (for TypeScript) and calculates distance between two strings using cosine similarity.

<details open>
<summary>Python</summary>

```python
from openevals.string.embedding_similarity import create_embedding_similarity_evaluator

evaluator = create_embedding_similarity_evaluator()

result = evaluator(
    outputs="The weather is nice!",
    reference_outputs="The weather is very nice!",
)

print(result)
```

```
{
    'key': 'embedding_similarity',
    'score': 0.9147273943905653,
    'comment': None,
}
```
</details>

<details>
<summary>TypeScript</summary>

```ts
import { createEmbeddingSimilarityEvaluator } from "openevals";
import { OpenAIEmbeddings } from "@langchain/openai";

const evaluator = createEmbeddingSimilarityEvaluator({
  embeddings: new OpenAIEmbeddings({ model: "text-embedding-3-small" }),
});

const result = await evaluator(
    outputs="The weather is nice!",
    referenceOutputs="The weather is very nice!",
);

console.log(result);
```

```
{
    key: "embedding_similarity",
    score: 0.9147273943905653,
}
```
</details>

### Agent evals

If you are building an agent, the evals in this repo are useful for evaluating specific outputs from your agent against references.

However, if you want to get started with more in-depth evals that take into account the entire trajectory of an agent, please check out the [`agentevals`](https://github.com/langchain-ai/agentevals) package.

### Creating your own

If there are metrics that you want to evaluate that are not covered by any of the above, you can create your own evaluator as well that interacts well with the rest of the `openevals` ecosystem.

#### Evaluator interface

The first thing to note that all evaluators should accept a subset of the following parameters:

- `inputs`: The inputs to your app.
- `outputs`: The outputs from your app.
- `reference_outputs` (Python) or `referenceOutputs` (TypeScript): The reference outputs to evaluate against.

These parameters can be any value, but should always accept a dict of some kind.

Not all evaluators will use all of these parameters, but they are there to ensure consistency across all evaluators.
Your evaluator may take more parameters as well (e.g. for LLM-as-judge evaluators whose prompts can require additional variables), but for simplicity it's best to stick to the three listed above.

If your evaluator requires additional configuration, you should use a factory function to create your evaluator. These should be named `create_<evaluator_name>` (for example, `create_llm_as_judge`).

The return values should be a dict (or, if your evaluator evaluates multiple metrics, an list of dicts) with the following keys:

- `key`: A string representing the name of the metric you are evaluating.
- `score`: A boolean or number representing the score for the given key.
- `comment`: A string representing the comment for the given key.

And that's it! Those are the only restrictions.

#### Logging to LangSmith

If you are using LangSmith to track experiments, you should also wrap the internals of your evaluator in the `_run_evaluator`/`_arun_evaluator` (Python) or `runEvaluator` (TypeScript) method. This ensures that the evaluator results are logged to LangSmith properly for supported runners.

This method takes a `scorer` function as part of its input that returns either:

- A single boolean or number, representing the score for the given key.
- A tuple that contains the score as its first element and a `comment` justifying the score as its second element.

#### Example

Here's an example of how you might define a very simple custom evaluator. It only takes into account the outputs of your app and compares them against a regex pattern. It uses a factory function to create the evaluator, since `regex` is an extra param.

<details open>
<summary>Python</summary>

```python
import json
import re
from typing import Any

from openevals.types import (
    EvaluatorResult,
    SimpleEvaluator,
)
from openevals.utils import _run_evaluator


def create_regex_evaluator(
    *, regex: str
) -> SimpleEvaluator:
    """
    Matches a regex pattern against the output.

    Args:
        regex (str): The regex pattern to match against the output.

    Returns:
        EvaluatorResult
    """

    regex = re.compile(regex)

    # Tolerate `inputs` and `reference_outputs` as kwargs, though they're unused
    def wrapped_evaluator(
        *, outputs: Any, **kwargs: Any
    ) -> EvaluatorResult:

        # Tolerate `outputs` being a dict, but convert to string for regex matching
        if not isinstance(outputs, str):
            outputs = json.dumps(outputs)

        def get_score():
            return regex.match(outputs) is not None

        res = _run_evaluator(
            run_name="regex_match",
            scorer=get_score,
            feedback_key="regex_match",
        )
        return res

    return wrapped_evaluator
```

```python
evaluator = create_regex_evaluator(regex=r"some string")
result = evaluator(outputs="this contains some string")
```

```
{
    'key': 'regex_match',
    'score': True,
    'comment': None,
}
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { EvaluatorResult } from "openevals/types";
import { _runEvaluator } from "openevals/utils";

/**
 * Creates an evaluator that compares the actual output and reference output for similarity by text embedding distance.
 * @param {Object} options - The configuration options
 * @param {Embeddings} options.embeddings - The embeddings model to use for similarity comparison
 * @param {('cosine'|'dot_product')} [options.algorithm='cosine'] - The algorithm to use for embedding similarity
 * @returns An evaluator that returns a score representing the embedding similarity
 */
export const createRegexEvaluator = ({
  regex,
}: {
  regex: RegExp;
}) => {
  return async (params: {
    outputs: string | Record<string, unknown>;
  }): Promise<EvaluatorResult> => {
    const { outputs } = params;

    // Tolerate `outputs` being an object, but convert to string for regex matching
    const outputString =
      typeof outputs === "string" ? outputs : JSON.stringify(outputs);

    const getScore = async (): Promise<boolean> => {
      return regex.test(outputString);
    };

    return _runEvaluator(
      "regex_match",
      getScore,
      "regex_match"
    );
  };
};
```

```ts
const evaluator = createRegexEvaluator({
  regex: /some string/,
});

const result = await evaluator({ outputs: "this text contains some string" });
```

```
{
  key: "regex_match",
  score: true,
}
```
</details>

## Python Async Support

All `openevals` evaluators support Python [asyncio](https://docs.python.org/3/library/asyncio.html). As a convention, evaluators that use a factory function will have `async` put immediately after `create_` in the function name (for example, `create_async_llm_as_judge`), and evaluators used directly will end in `async` (e.g. `exact_match_async`).

Here's an example of how to use the `create_async_llm_as_judge` evaluator asynchronously:

```python
from openevals.llm import create_async_llm_as_judge

evaluator = create_async_llm_as_judge(
    prompt="What is the weather in {inputs}?",
    model="openai:o3-mini",
)

result = await evaluator(inputs="San Francisco")
```

If you are using the OpenAI client directly, remember to pass in `AsyncOpenAI` as the `judge` parameter:

```python
from openai import AsyncOpenAI

evaluator = create_async_llm_as_judge(
    prompt="What is the weather in {inputs}?",
    judge=AsyncOpenAI(),
    model="o3-mini",
)

result = await evaluator(inputs="San Francisco")
```

## LangSmith Integration

For tracking experiments over time, you can log evaluator results to [LangSmith](https://smith.langchain.com/), a platform for building production-grade LLM applications that includes tracing, evaluation, and experimentation tools.

LangSmith currently offers two ways to run evals: a [pytest](https://docs.smith.langchain.com/evaluation/how_to_guides/pytest) (Python) or [Vitest/Jest](https://docs.smith.langchain.com/evaluation/how_to_guides/vitest_jest) integration and the `evaluate` function. We'll give a quick example of how to run evals using both.

### Pytest or Vitest/Jest

First, follow [these instructions](https://docs.smith.langchain.com/evaluation/how_to_guides/pytest) to set up LangSmith's pytest runner,
or these to set up [Vitest or Jest](https://docs.smith.langchain.com/evaluation/how_to_guides/vitest_jest), setting appropriate environment variables:

```bash
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGSMITH_TRACING="true"
```

<details open>
<summary>Python</summary>

Then, set up a file named `test_correctness.py` with the following contents:

```python
import pytest

from langsmith import testing as t

from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:o3-mini",
)

@pytest.mark.langsmith
def test_correctness():
    inputs = "How much has the price of doodads changed in the past year?"
    outputs = "Doodads have increased in price by 10% in the past year."
    reference_outputs = "The price of doodads has decreased by 50% in the past year."
    t.log_inputs({"question": inputs})
    t.log_outputs({"answer": outputs})
    t.log_reference_outputs({"answer": reference_outputs})

    correctness_evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )
```

Note that when creating the evaluator, we've added a `feedback_key` parameter. This will be used to name the feedback in LangSmith.

Now, run the eval with pytest:

```bash
pytest test_correctness.py --langsmith-output
```
</details>

<details>
<summary>TypeScript</summary>

Then, set up a file named `test_correctness.eval.ts` with the following contents:

```ts
import * as ls from "langsmith/vitest";
// import * as ls from "langsmith/jest";

import { createLLMAsJudge, CORRECTNESS_PROMPT } from "openevals";

const correctnessEvaluator = createLLMAsJudge({
  prompt: CORRECTNESS_PROMPT,
  feedbackKey: "correctness",
  model: "openai:o3-mini",
});


ls.describe("Correctness", () => {
  ls.test("incorrect answer", {
    inputs: {
      question: "How much has the price of doodads changed in the past year?"
    },
    referenceOutputs: {
      answer: "The price of doodads has decreased by 50% in the past year."
    }
  }, async ({ inputs, referenceOutputs }) => {
    const outputs = "Doodads have increased in price by 10% in the past year.";
    ls.logOutputs({ answer: outputs });

    await correctnessEvaluator({
      inputs,
      outputs,
      referenceOutputs,
    });
  });
});
```
Note that when creating the evaluator, we've added a `feedback_key` parameter. This will be used to name the feedback in LangSmith.

Now, run the eval with your runner of choice:

```bash
vitest run test_correctness.eval.ts
```
</details>

Feedback from the prebuilt evaluator will be automatically logged in LangSmith as a table of results like this in your terminal (if you've set up your reporter):

![Terminal results](/static/img/pytest_output.png)

And you should also see the results in the experiment view in LangSmith:

![LangSmith results](/static/img/langsmith_results.png)

### Evaluate

Alternatively, you can [create a dataset in LangSmith](https://docs.smith.langchain.com/evaluation/concepts#dataset-curation) and use your created evaluators with LangSmith's [`evaluate`](https://docs.smith.langchain.com/evaluation#8-run-and-view-results) function:

<details open>
<summary>Python</summary>

```python
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CONCISENESS_PROMPT

client = Client()

conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    feedback_key="conciseness",
    model="openai:o3-mini",
)

experiment_results = client.evaluate(
    # This is a dummy target function, replace with your actual LLM-based system
    lambda inputs: "What color is the sky?",
    data="Sample dataset",
    evaluators=[
        conciseness_evaluator
    ]
)
```

</details>

<details>
<summary>TypeScript</summary>

```ts
import { evaluate } from "langsmith/evaluation";
import { createLLMAsJudge, CONCISENESS_PROMPT } from "openevals";

const concisenessEvaluator = createLLMAsJudge({
  prompt: CONCISENESS_PROMPT,
  feedbackKey: "conciseness",
  model: "openai:o3-mini",
});

await evaluate(
  (inputs) => "What color is the sky?",
  {
    data: datasetName,
    evaluators: [concisenessEvaluator],
  }
);
```
</details>

## Thank you!

We hope that `openevals` helps make evaluating your LLM apps easier!

If you have any questions, comments, or suggestions, please open an issue or reach out to us on X [@LangChainAI](https://x.com/langchainai).
