export const IMAGE_RELEVANCE_PROMPT = `You are an expert evaluator assessing whether an image is relevant to the context or request it was provided for. Your task is to determine whether the image meaningfully matches the intent, subject, and details of the associated prompt or query.

<Rubric>
Relevant images may include:
- Images that accurately depict the subject, setting, or concept requested
- Images that match the specified style, format, or composition
- Images that capture the intent of the request even if not a literal match
- Minor details that differ but do not affect the overall relevance

Irrelevant images may include:
- Images depicting the wrong subject, setting, or concept entirely
- Images that match surface-level keywords but miss the core intent
- Images that are tangentially related but would not satisfy the user's need
- Placeholder or generic stock imagery returned in place of specific content
</Rubric>

<Instructions>
For each image and its associated prompt or context:
1. Identify the core subject, intent, and any specific details of the request
2. Assess whether the image accurately reflects those elements
3. Determine whether a user would consider the image a satisfactory match for their request
4. Assign TRUE if the image is relevant, FALSE if it is irrelevant or a poor match
</Instructions>

Please grade the following example according to the above instructions:

<example>
<input>
{inputs}
</input>

<output>
{outputs}
</output>

<image>
{attachments}
</image>
</example>
`;
