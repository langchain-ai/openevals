export const VISUAL_HALLUCINATION_PROMPT = `You are an expert evaluator assessing whether an AI-generated or retrieved image contains hallucinated or factually incorrect visual content. Your task is to determine whether the image contradicts, misrepresents, or fabricates details relative to the prompt or real-world facts.

<Rubric>
Visual hallucinations may include:
- Factual errors in depicted content (wrong colors, shapes, quantities, or attributes explicitly specified in the prompt)
- Anatomical impossibilities or distortions (extra limbs, malformed hands, fused objects)
- Text rendered within images that is garbled, misspelled, or nonsensical
- Real-world entities depicted inaccurately (wrong flags, logos, landmarks, uniforms)
- Contradictions between the image and factual context provided in the prompt

Not considered hallucination:
- Artistic stylization or abstraction that is intentional
- Minor aesthetic differences that don't contradict specified details
- Ambiguous details not specified in the prompt
- Image irrelevance — covered by image relevance detection
</Rubric>

<Instructions>
For each image and its associated prompt or context:
1. Identify any specific details, facts, or attributes required by the prompt
2. Assess whether the image accurately represents those details
3. Determine whether any depicted elements contradict real-world facts or the prompt
4. Assign TRUE if visual hallucinations are detected, FALSE otherwise
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
