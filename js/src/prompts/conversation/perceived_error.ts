export const PERCEIVED_ERROR_PROMPT = `You are an expert conversation evaluator. You will be shown a full multi-turn conversation between a human user and an AI agent.
Your task is to assess whether the user's responses suggest the agent made a mistake during the conversation.

<Rubric>
Signs the user perceived an error:
- The user explicitly corrects the agent (e.g. "No, that's wrong", "You misunderstood me", "That's not what I asked")
- The user repeats or rephrases the same request, suggesting the agent did not fulfill it
- The user expresses frustration, confusion, or dissatisfaction directed at the agent's response
- The user asks the agent to redo, undo, or fix something it just did
- The conversation stalls or backtracks due to an agent misstep

Signs the user did not perceive an error:
- The user acknowledges or accepts the agent's response (e.g. "Thanks", "Perfect", "Got it")
- The user continues the conversation naturally without correcting or repeating themselves
- The user's follow-up questions build on the agent's response rather than challenging it
</Rubric>

<Instructions>
For each conversation:
1. Read the full conversation, paying close attention to how the user responds after each agent turn
2. Identify any signals in the user's language suggesting dissatisfaction, correction, or repetition
3. Assess whether these signals indicate the agent made a mistake the user noticed
4. Assign TRUE if there is evidence the user perceived an error, FALSE otherwise
</Instructions>

Please grade the following conversation according to the above instructions:

<conversation>
{outputs}
</conversation>
`;
