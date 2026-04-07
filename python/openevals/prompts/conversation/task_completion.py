TASK_COMPLETION_PROMPT = """You are an expert conversation evaluator. You will be shown a full conversation between a human user and an AI agent.
Your task is to track every request made by the human and determine whether each one was fully completed by the agent.

<Rubric>
Fully completed requests may include:
- The agent addressed all parts of the request in its response
- The human did not re-ask, restate, or follow up on the same request
- The human explicitly confirms the response was sufficient

Incomplete or failed requests may include:
- The agent partially addressed or missed part of the request
- The human had to re-ask or restate the same request
- The human had to correct or clarify because the AI misunderstood
- The human asked the AI to redo, fix, or finish something from a prior response
</Rubric>

<Instructions>
For each conversation:
1. List every request made by the human throughout the conversation
2. For each request, determine whether the AI fully completed it
3. Check whether the human re-asked or followed up on any prior request
4. Return TRUE if all requests were fully completed, FALSE if any request was missed, incomplete, or required a follow-up
</Instructions>

Please grade the following conversation according to the above instructions:

<conversation>
{outputs}
</conversation>
"""
