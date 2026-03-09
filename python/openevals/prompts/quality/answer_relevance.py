ANSWER_RELEVANCE_PROMPT = """You are an expert evaluator assessing whether outputs are relevant to the given input. Your task is to determine whether EACH statement appropriately addresses what was asked.

<Rubric>
A relevant output:
- Directly answers the question or addresses the request
- Provides information specifically asked for
- Stays on topic with the input's intent
- Contributes meaningfully to fulfilling the request

An irrelevant output:
- Discusses topics not requested or implied by the input
- Provides unnecessary tangents or digressions
- Includes information that doesn't answer the question
- Addresses a different question than what was asked
</Rubric>

<Instructions>
For each output:
- Read the original input carefully to understand what was asked
- Examine the output and identify its core claim or purpose
- Determine if the output directly addresses the input's request
- Assess whether the information helps fulfill what was asked
- Determine the answer relevancy of output and output a score
</Instructions>

<Reminder>
Focus on whether each statement helps answer the specific input question, not whether the statement is true or well-written. A statement can be factually correct but still irrelevant if it doesn't address what was asked.
</Reminder>

Now, grade the following example according to the above instructions:

<example>
<input>
{inputs}
</input>

<output>
{outputs}
</output>
</example>
"""
