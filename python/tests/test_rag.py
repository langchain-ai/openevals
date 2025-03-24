import pytest

from openevals.llm import create_llm_as_judge
from openevals.prompts.rag_retrieval import RETRIEVAL_HELPFULNESS_PROMPT
from openevals.prompts.rag_hallucination import RAG_HALLUCATION_PROMPT


@pytest.mark.langsmith
def test_llm_as_judge_rag_hallucination():
    inputs = {
        "question": "Who was the first president of the Star Republic of Oiewjoie?",
    }
    outputs = {"answer": "Bzkeoei Ahbeijo"}
    llm_as_judge = create_llm_as_judge(
        prompt=RAG_HALLUCATION_PROMPT,
        feedback_key="hallucination",
        model="openai:o3-mini",
    )
    context = "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo."
    with pytest.raises(KeyError):
        eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    eval_result = llm_as_judge(
        inputs=inputs, outputs=outputs, context=context, reference_outputs=""
    )
    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_rag_hallucination_not_correct():
    inputs = {
        "question": "Who was the first president of the Star Republic of Oiewjoie?",
    }
    outputs = {"answer": "John Adams"}
    llm_as_judge = create_llm_as_judge(
        prompt=RAG_HALLUCATION_PROMPT,
        feedback_key="hallucination",
        model="openai:o3-mini",
    )
    context = "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo."
    eval_result = llm_as_judge(
        inputs=inputs, outputs=outputs, context=context, reference_outputs=""
    )
    assert not eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_rag_retrieval():
    inputs = {
        "question": "Where was the first president of foobarland born?",
    }
    outputs = [
        {
            "title": "foobarland president",
            "content": "the first president of foobarland was bagatur",
        },
        {"title": "bagatur bio", "content": "bagutur was born in langchainland"},
    ]
    llm_as_judge = create_llm_as_judge(
        prompt=RETRIEVAL_HELPFULNESS_PROMPT,
        feedback_key="hallucination",
        model="openai:o3-mini",
    )

    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_rag_retrIEVAL_not_correct():
    inputs = {
        "question": "Where was the first president of foobarland born?",
    }
    outputs = [
        {
            "title": "foobarland president",
            "content": "the first president of foobarland was bagatur",
        },
        {"title": "bagatur bio", "content": "bagutur is a big fan of PR reviews"},
    ]
    llm_as_judge = create_llm_as_judge(
        prompt=RETRIEVAL_HELPFULNESS_PROMPT,
        feedback_key="hallucination",
        model="openai:o3-mini",
    )

    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)

    assert not eval_result["score"]
