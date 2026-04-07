export const WINS_PROMPT = `You are an expert conversation evaluator. You will be shown a conversation between a human user and an AI assistant.
Your task is to assess whether the user praised, thanked, or complimented the assistant during the conversation.

<Rubric>
A win may include:
- Explicit praise or compliments (e.g. "That's exactly what I needed", "Great answer", "You're amazing")
- Gratitude directed at the assistant (e.g. "Thank you so much", "Thanks, this really helped")
- Expressions of delight or positive surprise (e.g. "Wow, I didn't expect that to work so well")
- The user sharing the outcome positively (e.g. "I showed this to my team and they loved it")

Not considered a win:
- Neutral acknowledgment without positive sentiment (e.g. "Okay", "Got it", "Sure")
- Politeness that is part of a correction or follow-up request (e.g. "Thanks but that's not quite right")
- Sarcastic or backhanded praise
</Rubric>

<Instructions>
For the conversation:
1. Read the user's messages for any expressions of praise, gratitude, or compliments toward the assistant
2. Distinguish genuine positive sentiment from neutral acknowledgment or polite corrections
3. Assign TRUE if the user praised, thanked, or complimented the assistant, FALSE otherwise
</Instructions>

Please grade the following conversation according to the above instructions:

<conversation>
{outputs}
</conversation>
`;
