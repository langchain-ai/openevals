GROUNDED_IN_CONTEXT_PROMPT = """
You are an expert data labeler evaluating model outputs for relevance to some context. Your task is to assign a score based on the following rubric:

<Rubric>
Relevant outputs:
- Only contains basic facts (i.e. the sky is blue) and information from the context
- Do not stray too far from the information contained in the context
- Do not hallucinate information
</Rubric>

<Instruction>
- Read the context
- Construct a list of all the facts/opinions in the context
- Extract all the facts/opinions from the model output
- Cross reference both these lists
- Penalize any facts/opinions in the model output that are not basic facts or mentioned in the context
</Instruction>

<outputs>
{outputs}
</outputs>

<context>
{context}
</context>
"""