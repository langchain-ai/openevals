from prompts.prompts import CONCISENESS_PROMPT, TOXIC_PROMPT

from evaluators.llm import create_llm_as_judge
from functools import partial

ConcisenessEvaluator = partial(
    create_llm_as_judge,
    prompt=CONCISENESS_PROMPT,
    key="conciseness",
)
ToxicityEvaluator = partial(
    create_llm_as_judge,
    prompt=TOXIC_PROMPT,
    key="toxicity",
)