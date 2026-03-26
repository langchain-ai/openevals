export const USER_INTERRUPTS_PROMPT = `You are an expert conversation evaluator. You will be provided with a voice recording of a conversation between a user and an AI agent.
Your task is to assess whether the agent handled user interruptions gracefully.

<Rubric>
Graceful handling of user interrupts:
- The agent stops speaking promptly when the user interjects
- The agent acknowledges the interruption and incorporates the user's new direction
- The conversation recovers naturally without the agent losing track or repeating itself
- The agent adapts its response based on what the user said during the interrupt

Poor handling of user interrupts:
- The agent continues speaking over the user after being interrupted
- The agent ignores or fails to incorporate the user's interjection
- The agent loses track of context after the interruption and repeats itself or goes off course
- The agent requires the user to repeat themselves because the interrupt was not processed
</Rubric>

<Instructions>
For each voice recording:
1. Identify any instances where the user interrupts or interjects while the agent is speaking
2. Assess whether the agent stopped and responded appropriately to each interrupt
3. Determine whether the conversation recovered naturally after each interruption
4. Assign TRUE if the agent handled user interrupts gracefully, FALSE otherwise
</Instructions>

Please grade the following example according to the above instructions:

<example>
<input>
{inputs}
</input>

<output>
{outputs}
</output>

<audio>
{attachments}
</audio>
</example>
`;
