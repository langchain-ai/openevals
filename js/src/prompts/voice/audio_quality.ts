export const AUDIO_QUALITY_PROMPT = `You are an expert audio quality evaluator. You will be provided with metadata or signals from an audio recording.
Your task is to determine whether the audio contains quality issues that would degrade the listener's experience.

<Rubric>
Audio quality issues include:
- Clipping: audio peaks that exceed maximum amplitude, causing distortion at loud moments
- Distortion: warping, crackling, or degradation of the audio signal
- Glitches: sudden dropouts, stutters, pops, or unexpected interruptions in the audio stream

Not considered audio quality issues:
- Natural variation in speaking volume
- Brief pauses or silence that are intentional
- Mild background noise that does not interfere with intelligibility
</Rubric>

<Instructions>
For each audio sample:
1. Identify any clipping, distortion, or glitch events present
2. Assess the severity and frequency of any issues detected and if they would meaningfully degrade the listener's experience
3. Assign TRUE if audio quality issues are detected, FALSE otherwise
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
