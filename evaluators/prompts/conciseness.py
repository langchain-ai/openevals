CONCISENESS_PROMPT = """
You are evaluating model outputs for conciseness. Your task is to assign a score between 0 and 1, where:
- 1.0 represents perfect conciseness
- 0.0 represents extreme verbosity

<Rubric>
  A perfectly concise answer (score = 1.0):
  - Contains only the exact information requested.
  - Uses the minimum number of words necessary to convey the complete answer.
  - Omits pleasantries, hedging language, and unnecessary context.
  - Excludes meta-commentary about the answer or the model's capabilities.
  - Avoids redundant information or restatements.
  - Does not include explanations unless explicitly requested.
  <Penalties>
    Deduct points for:
    - Introductory phrases like "I believe," "I think," or "The answer is."
    - Hedging language like "probably," "likely," or "as far as I know."
    - Unnecessary context or background information.
    - Explanations when not requested.
    - Follow-up questions or offers for more information.
    - Redundant information or restatements.
    - Polite phrases like "hope this helps" or "let me know if you need anything else."
  </Penalties>
  <ScoringGuide>
    <Score value="1.0">Perfect conciseness. Only essential information included.</Score>
    <Score value="0.75">Mostly concise with a few minor unnecessary words.</Score>
    <Score value="0.5">Contains some unnecessary context or hedging language.</Score>
    <Score value="0.25">Includes multiple unnecessary elements like context, explanations, and hedging.</Score>
    <Score value="0">Very verbose with extensive unnecessary information.</Score>
  </ScoringGuide>
</Rubric>

<Examples>
  <Example>
    <Input>What is the capital of France?</Input>
    <Response score="1.0">Paris</Response>
    <Response score="0.75">The capital is Paris</Response>
    <Response score="0.5">I believe the capital of France is Paris</Response>
    <Response score="0.25">The capital of France is Paris, which has been the capital city since 1792</Response>
    <Response score="0">Let me help you with that. The capital of France is Paris, which is one of the most beautiful cities in Europe. It's been the capital since 1792 and is known for...</Response>
  </Example>
  <Example>
    <Input>How many planets are in our solar system?</Input>
    <Response score="1.0">8</Response>
    <Response score="0.75">There are 8 planets</Response>
    <Response score="0.5">Our solar system contains 8 planets</Response>
    <Response score="0.25">I can tell you that our solar system has 8 planets. These planets orbit around the Sun.</Response>
    <Response score="0">Great question! Our solar system is home to 8 planets, though this number has changed over time as our definition of planets has evolved...</Response>
  </Example>
</Examples>

<Instructions>
  - Carefully read the input and output.
  - Check for any unnecessary elements listed in the <Penalties> section in the <Rubric> above.
  - Compare the response to the scoring guide and examples.
  - Assign a score between 0 and 1, using decimals for precision.
  - The score should reflect how close the response comes to containing only the essential information requested based on the rubric above.
</Instructions>

<Reminder>
  The goal is to reward responses that provide complete answers with absolutely no extraneous information.
</Reminder>

<input>
{inputs}
</input>

<output>
{outputs}
</output>
"""
