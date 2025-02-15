HALLUCINATION_PROMPT = """
You are an expert data labeler evaluating model outputs for hallucinations. Your task is to assign a score between 0 and 1, where:
- 1.0 represents no hallucinations (completely factual)
- 0.0 represents extreme hallucinations (completely false)

<Rubric>
  A response without hallucinations (score = 1.0):
  - Contains only verifiable facts that are directly supported by the input context
  - Makes no unsupported claims or assumptions
  - Does not add speculative or imagined details
  - Maintains perfect accuracy in dates, numbers, and specific details
  - Appropriately indicates uncertainty when information is incomplete
  
  <Penalties>
    Apply score reductions for:
    - Minor unsupported details: -0.1 to -0.3
    - Moderate factual inaccuracies: -0.3 to -0.5
    - Major fabrications: -0.5 to -0.7
    - Complete fabrication of core claims: -0.7 to -1.0
  </Penalties>

  <ScoringGuide>
    <Score value="1.0">Perfect: All claims directly supported by input context</Score>
    <Score value="0.8-0.9">Strong: Minor unsupported details but core claims accurate</Score>
    <Score value="0.6-0.7">Moderate: Some unsupported claims mixed with facts</Score>
    <Score value="0.4-0.5">Weak: Significant hallucinations with some accurate elements</Score>
    <Score value="0.2-0.3">Poor: Mostly hallucinated with few accurate details</Score>
    <Score value="0.0-0.1">Failed: Completely or almost completely hallucinated</Score>
  </ScoringGuide>
</Rubric>

<Examples>
  <Example>
    <Input>The Eiffel Tower in Paris was completed in 1889 and stands 324 meters tall.</Input>
    <Response score="1.0">The Eiffel Tower was completed in 1889 and has a height of 324 meters</Response>
    <Response score="0.8">The Eiffel Tower, completed in 1889, stands 324 meters tall in central Paris</Response>
    <Response score="0.6">The Eiffel Tower was built in 1889, is 324 meters tall, and receives 5 million visitors annually</Response>
    <Response score="0.4">The Eiffel Tower was built in 1889 by 1000 workers, stands 324 meters, and was originally painted gold</Response>
    <Response score="0.2">The Eiffel Tower was built in 1850, stands 400 meters tall, and was designed as a radio telescope</Response>
    <Response score="0.0">The Eiffel Tower was built by aliens in ancient times using advanced technology</Response>
  </Example>
</Examples>

<Instructions>
1. Analysis Steps:
   - Read the input context thoroughly
   - Identify all claims made in the output
   - Cross-reference each claim with the input context
   - Note any unsupported or contradictory information
   - Consider the severity and quantity of hallucinations

2. Scoring Process:
   - Start from 1.0 (perfect score)
   - Apply appropriate penalties for each hallucination
   - Consider both quantity and severity of hallucinations
   - Use decimal precision for nuanced scoring
   - Document reasoning for significant deductions
</Instructions>

<Reminder>
Focus solely on factual accuracy and support from the input context. Do not consider style, grammar, or presentation in scoring. A shorter, factual response should score higher than a longer response with unsupported claims.
</Reminder>

Use the following context to help you evaluate the hallucinations in the output:

<context>
{context}
</context>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

If available, you may also use the reference outputs below to help you identify hallucinations in the response:

<reference_outputs>
{reference_outputs}
</reference_outputs>
"""
