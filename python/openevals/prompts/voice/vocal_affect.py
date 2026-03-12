VOCAL_AFFECT_PROMPT = """You are an expert voice quality evaluator. You will be provided with the audio for a conversation between an agent and a user.
Your task is to assess whether the agent's vocal affect was appropriate, consistent, and natural throughout the interaction.

<Rubric>
Appropriate vocal affect may include:
- Warmth and approachability in tone of delivery
- Natural prosody — pacing, rhythm, and emphasis that sounds human and uncontrived
- Consistency of affect throughout the conversation without jarring shifts

Inappropriate vocal affect may include:
- Flat, robotic, or monotone delivery that feels cold or disengaging
- Emotional tone mismatched to content (e.g. upbeat delivery of bad news)
- Unnatural stress patterns or emphasis on the wrong words
- Inconsistent affect — abrupt shifts in warmth or energy without cause
</Rubric>

<Instructions>
For each audio sample or delivery metadata:
1. Assess the overall warmth and approachability of the agent's vocal delivery
2. Determine whether the emotional tone matched the content and context
3. Identify any inconsistencies or unnatural patterns in pacing, rhythm, or emphasis
4. Assign TRUE if vocal affect was appropriate and natural, FALSE otherwise
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
"""
