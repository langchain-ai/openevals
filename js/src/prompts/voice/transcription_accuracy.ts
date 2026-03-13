export const TRANSCRIPTION_ACCURACY_PROMPT = `You are an expert transcription evaluator. You will be provided with an audio transcription alongside a reference transcript.
Your task is to determine whether the transcription accurately reflects the spoken content.

<Rubric>
Transcription inaccuracies include:
- Substituted words that change the meaning of the spoken content
- Missing words, phrases, or sentences present in the audio
- Inserted words not present in the audio
- Incorrect proper nouns, technical terms, or named entities
- Punctuation or formatting errors that misrepresent spoken structure or intent

Not considered inaccuracies:
- Minor punctuation differences that do not affect meaning
- Stylistic formatting differences (capitalization, spacing)
- Filler words (um, uh) omitted from a clean transcript
</Rubric>

<Instructions>
For each transcription:
1. Compare the transcription against the reference for substitutions, omissions, and insertions
2. Pay particular attention to proper nouns, technical terms, and meaning-bearing words
3. Assess whether any inaccuracies would meaningfully mislead or confuse a reader
4. Assign TRUE if the transcription is accurate, FALSE if inaccuracies are detected
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
