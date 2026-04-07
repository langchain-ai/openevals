export const LANGUAGE_DETECTION_PROMPT = `You are an expert conversation evaluator. You will be shown a conversation between a human user and an AI assistant.
Your task is to identify the primary language used by the human throughout the conversation.

<Instructions>
For each conversation:
1. Identify the primary language used by the human user
2. If the human switches languages, identify the language used most frequently
3. Return the detected language as a standard language name in English (e.g. "English", "Spanish", "French", "Chinese")
</Instructions>

Please grade the following conversation according to the above instructions:

<conversation>
{outputs}
</conversation>
`;
