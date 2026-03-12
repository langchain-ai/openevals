export const USER_SATISFACTION_PROMPT = `You are an expert conversation evaluator. You will be shown a full conversation between a human user and an AI assistant.
Your task is to judge overall user satisfaction throughout the duration of this conversation.

<Rubric>
Satisfied responses may include:
- Gratitude or appreciation (e.g. "thank you", "that's helpful", "perfect")
- Resolution indicators (e.g. "works now", "that makes sense", "got it")
- Engagement that signals trust or enthusiasm (e.g. follow-up questions building on the answer)
- Neutral acknowledgment — treat as satisfied (e.g. "okay", "cool", "sure")

Unsatisfied responses may include:
- Explicit dissatisfaction or frustration (e.g. "this is wrong", "not helpful", "that's not what I asked")
- Continued problem restatement (e.g. "still doesn't work", "same issue")
- Implied negativity without explicit words (e.g. "I'll just figure it out myself", "sure, whatever", "never mind")
- Repeated clarifications suggesting the AI failed to understand
</Rubric>

<Instructions>
For each conversation:
1. Read the full conversation and identify shifts in user tone over time
2. Pay particular attention to the final human message
3. Assess whether the user's core need was met by the end of the conversation
4. Assign TRUE if the user appears satisfied, FALSE if unsatisfied
</Instructions>

Please grade the following conversation according to the above instructions:

<conversation>
{outputs}
</conversation>
`;
