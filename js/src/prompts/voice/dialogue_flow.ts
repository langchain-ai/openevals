console.warn(
  "[openevals] Voice prompts are in beta and may change in future releases."
);

export const DIALOGUE_FLOW_PROMPT = `You are an expert conversation evaluator. You will be provided with a voice recording of a conversation between a user and an AI agent.
Your task is to assess whether the dialogue flow felt natural or whether it became awkward due to overlapping speech or frequent interruptions.

<Rubric>
Natural dialogue flow may include:
- Clean exchanges where one speaker finishes before the other begins
- Occasional interruptions that are handled smoothly without derailing the conversation
- The agent adapting naturally when the user redirects or interjects
- The conversation maintaining coherence and rhythm even after interruptions

Unnatural dialogue flow may include:
- Frequent instances of the user and agent speaking at the same time
- User interruptions that cause the agent to lose track or repeat itself
- Overlapping speech that makes the exchange difficult to follow
- The conversation never finding a natural rhythm due to repeated cross-talk
- Long awkward silences or abrupt restarts after interruptions
</Rubric>

<Instructions>
For each voice recording:
1. Listen for any instances of overlapping speech, cross-talk, or interruptions
2. Assess how frequently these occur and whether they disrupt the overall conversation
3. Determine whether the conversation recovered naturally or remained stilted throughout
4. Assign TRUE if dialogue flow was natural, FALSE if overlapping speech or interruptions made it awkward
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
