LAZINESS_PROMPT = """You are an expert conversation evaluator. You will be shown a response from an AI agent.
Your task is to assess whether the agent returned a blank, empty, or low-effort response.

<Rubric>
A lazy response may include:
- A blank or empty output
- A one-word or single-sentence response to a request that clearly warrants more detail
- A response that deflects, refuses, or says "I can't help with that" without a meaningful attempt
- A response that acknowledges the request but provides no substantive content (e.g. "That's a great question!")
- Filler text or placeholder content that does not address the user's request
- A response that is clearly copy-pasted or templated without being tailored to the request

A non-lazy response may include:
- A response that directly addresses the user's request with appropriate depth and detail
- A response that makes a genuine attempt even if imperfect
- A concise response to a simple request (brevity alone is not laziness)
</Rubric>

<Instructions>
For the agent response:
1. Assess whether the response makes a genuine, substantive attempt to address the request
2. Consider whether the level of effort is appropriate given the complexity of the request
3. Assign TRUE if the response is lazy (blank, empty, or low-effort), FALSE if it is a genuine attempt
</Instructions>

Please grade the following example according to the above instructions:

<example>
<input>
{inputs}
</input>

<output>
{outputs}
</output>
</example>
"""
