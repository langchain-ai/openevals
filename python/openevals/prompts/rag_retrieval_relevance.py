RAG_RETRIEVAL_RELEVANCE_PROMPT = """You are an expert data labeler evaluating retrieved context for relevance to the input. Your task is to assign a score based on the following rubric:

<Rubric>
- Relevant retrieved context:
  - Contain information that could help answer the input, even if incomplete.
  - May include superfluous information, but it should still be somewhat related to the input.

- Irrelevant retrieved context:
  - Contain no useful information for answering the input.
  - Are entirely unrelated to the input.
  - Contains misleading or incorrect information
  - Contains only tangentially related information with no practical utility
</Rubric>

<Instruction>
- Read and understand the full meaning of the input (including edge cases)
- Formulate a list of facts and relevant context that would be needed to respond to the input
- Analyze the retrieved context to identify:
  - Information directly relevant to answering the query
  - Information partially relevant or contextually helpful
  - Information completely irrelevant to the query
- For each piece of information need identified in the previous step, determine:
  - Whether it is fully addressed by the retrieved documents (cite specific text)
  - Whether it is partially addressed (cite specific text)
  - Whether it is not addressed at all
- Note any facts needed to answer the input that are not found
- Note any context that are completely irrelevant, i.e. contain no relevant facts for answering the input
</Instruction>

<Reminder>  
- Focus solely on whether the retrieved context provides useful information for answering the input.
- Think deeply about why the context is or isn’t relevant.
- Use partial credit where applicable, recognizing context that is somewhat helpful even if incomplete.
</Reminder> 

<inputs>
{inputs}
</inputs>

<retrieved_context>
{context}
</retrieved_context>
"""
