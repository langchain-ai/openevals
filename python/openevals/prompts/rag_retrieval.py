RETRIEVAL_HELPFULNESS_PROMPT = """You are an expert data labeler evaluating outputs for relevance to the input. Your task is to assign a score based on the following rubric:

<Rubric>
- Relevant retrieved outputs:  
  - Contain information that could help answer the input.  
  - May include superfluous information, but it should still be somewhat related to the input.  
- Irrelevant retrieved outputs:  
  - Contain no useful information for answering the input.  
  - Are entirely unrelated to the input.
</Rubric>

<Instruction>
- Read and understand the full meaning of the input (including edge cases)
- Formulate a list of facts and relevant context that would be needed to respond to the input
- Analyze each retrieved document to identify:
    - Information directly relevant to answering the query
    - Information partially relevant or contextually helpful
    - Information completely irrelevant to the query
- For each piece of information need identified in the previous step, determine:
    - Whether it is fully addressed by the retrieved documents (cite specific text)
    - Whether it is partially addressed (cite specific text)
    - Whether it is not addressed at all
- Note any facts needed to answer the input that are not found
- Note any outputs that are completely irrelevant, i.e. contain no relevant facts for answering the input
</Instruction>

<Reminder>  
- Focus solely on whether the retrieved outputs provide useful information for answering the input.  
- Think deeply about why each output is or isn’t relevant.  
- Use partial credit where applicable, recognizing outputs that are somewhat helpful even if incomplete.  
</Reminder> 

<inputs>
{inputs}
</inputs>

<outputs>
{outputs}
</outputs>
"""
