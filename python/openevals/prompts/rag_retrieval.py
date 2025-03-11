RETRIEVAL_HELPFULNESS_PROMPT = """
You are an expert data labeler evaluating outputs for relevance to the input. Your task is to assign a score based on the following rubric:

<Rubric>
Retrieved outputs that are relevant:
- Contain information that could help answer the input
- Can also contain superfluous information, but it should still be somewhat related
</Rubric>

<Instruction>
- Read and understand the full meaning of the input (including edge cases)
- Formulate a list of facts that would be needed to respond to the input
- Identify all the information contained in the outputs
- Match each fact needed to answer the input with the information in the outputs
- Note any facts that are not found
- Note any outputs that are completely irrelevant
</Instruction>

<Reminder>
Focus solely on whether the information in the outputs can answer the input. Explain thoroughly why this is or isn't the case. Make sure to use partial credit.
</Reminder>

<input>
{input}
</input>

<outputs>
{outputs}
</outputs>
"""