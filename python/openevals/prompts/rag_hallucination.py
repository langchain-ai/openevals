RAG_HALLUCATION_PROMPT = """You are an expert data labeler evaluating model outputs for relevance to some context. Your task is to assign a score based on the following rubric:

<Rubric>
Relevant outputs:
- Only contains basic facts (e.g. the sky is blue, or 2+2=4) and information from the context
- Do not stray too far from the information contained in the context
- Do not hallucinate or make up facts without supporting evidence
- Do not misrepresent, alter, or obfuscate what is stated in the context.
</Rubric>

<Instruction>
- Read the context
- Construct a list of all the facts/opinions in the context
- Extract all the facts/opinions from the model output
- For all the facts/opinions outputted by the model determine if it is:
    - Directly supported by the retrieved documents
    - A widely-known basic fact requiring no citation (e.g., "water is liquid at room temperature")
    - Not supported by the documents and not a basic fact
- Penalize any facts/opinions in the model output that are not basic facts or supported by the retrieved documents
</Instruction>

<outputs>
{outputs}
</outputs>

<context>
{context}
</context>
"""
