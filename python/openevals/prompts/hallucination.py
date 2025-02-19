HALLUCINATION_PROMPT = """You are an expert data labeler evaluating model outputs for hallucinations. Your task is to assign a score based on the following rubric:

<Rubric>
  A response without hallucinations:
  - Contains only verifiable facts that are directly supported by the input context
  - Makes no unsupported claims or assumptions
  - Does not add speculative or imagined details
  - Maintains perfect accuracy in dates, numbers, and specific details
  - Appropriately indicates uncertainty when information is incomplete
</Rubric>

<Instructions>
  - Read the input context thoroughly
  - Identify all claims made in the output
  - Cross-reference each claim with the input context
  - Note any unsupported or contradictory information
  - Consider the severity and quantity of hallucinations
</Instructions>

<Reminder>
  Focus solely on factual accuracy and support from the input context. Do not consider style, grammar, or presentation in scoring. A shorter, factual response should score higher than a longer response with unsupported claims.
</Reminder>

Use the following context to help you evaluate for hallucinations in the output:

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
