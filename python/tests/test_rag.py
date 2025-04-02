import pytest

from langsmith import testing as t

from openevals.llm import create_llm_as_judge
from openevals.prompts.rag_helpfulness import RAG_HELPFULNESS_PROMPT
from openevals.prompts.rag_groundedness import RAG_GROUNDEDNESS_PROMPT
from openevals.prompts.rag_retrieval_relevance import RAG_RETRIEVAL_RELEVANCE_PROMPT

# @pytest.mark.langsmith
# def test_llm_as_judge_rag_hallucination():
#     inputs = {
#         "question": "Who was the first president of the Star Republic of Oiewjoie?",
#     }
#     outputs = {"answer": "Bzkeoei Ahbeijo"}
#     t.log_inputs(inputs)
#     t.log_outputs(outputs)
#     llm_as_judge = create_llm_as_judge(
#         prompt=RAG_HALLUCATION_PROMPT,
#         feedback_key="hallucination",
#         model="openai:o3-mini",
#     )
#     context = "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo."
#     with pytest.raises(KeyError):
#         eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
#     eval_result = llm_as_judge(
#         inputs=inputs, outputs=outputs, context=context, reference_outputs=""
#     )
#     assert eval_result["score"]


# @pytest.mark.langsmith
# def test_llm_as_judge_rag_hallucination_not_correct():
#     inputs = {
#         "question": "Who was the first president of the Star Republic of Oiewjoie?",
#     }
#     outputs = {"answer": "John Adams"}
#     t.log_inputs(inputs)
#     t.log_outputs(outputs)
#     llm_as_judge = create_llm_as_judge(
#         prompt=RAG_HALLUCATION_PROMPT,
#         feedback_key="hallucination",
#         model="openai:o3-mini",
#     )
#     context = "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo."
#     eval_result = llm_as_judge(
#         inputs=inputs, outputs=outputs, context=context, reference_outputs=""
#     )
#     assert not eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_rag_helpfulnesss():
    inputs = {
        "question": "Where was the first president of foobarland born?",
    }
    outputs = {
        "answer": "The first president of foobarland was born in langchainland. His name was Bagatur Askaryan."
    }
    t.log_inputs(inputs)
    t.log_outputs(outputs)
    llm_as_judge = create_llm_as_judge(
        prompt=RAG_HELPFULNESS_PROMPT,
        feedback_key="helpfulness",
        model="openai:o3-mini",
    )

    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_rag_helpfulness_not_correct():
    inputs = {
        "question": "Where was the first president of foobarland born?",
    }
    outputs = {"answer": "The first president of foobarland was bagatur"}
    t.log_inputs(inputs)
    t.log_outputs(outputs)
    llm_as_judge = create_llm_as_judge(
        prompt=RAG_HELPFULNESS_PROMPT,
        feedback_key="helpfulness",
        model="openai:o3-mini",
    )

    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)

    assert not eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_rag_groundedness():
    retrieval_evaluator = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        model="openai:o3-mini",
    )

    context = {
        "documents": [
            "FoobarLand is a new country located on the dark side of the moon",
            "Space dolphins are native to FoobarLand",
            "FoobarLand is a constitutional democracy whose first president was Bagatur Askaryan",
            "The current weather in FoobarLand is 80 degrees and clear.",
        ],
    }

    outputs = {
        "answer": "The first president of FoobarLand was Bagatur Askaryan.",
    }

    eval_result = retrieval_evaluator(
        context=context,
        outputs=outputs,
    )

    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_rag_retrieval_relevance():
    retrieval_relevance_evaluator = create_llm_as_judge(
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
        feedback_key="retrieval_relevance",
        model="openai:o3-mini",
    )

    inputs = {
        "question": "Where was the first president of FoobarLand born?",
    }

    context = {
        "documents": [
            "FoobarLand is a new country located on the dark side of the moon",
            "Space dolphins are native to FoobarLand",
            "FoobarLand is a constitutional democracy whose first president was Bagatur Askaryan",
            "The current weather in FoobarLand is 80 degrees and clear.",
        ],
    }

    eval_result = retrieval_relevance_evaluator(
        inputs=inputs,
        context=context,
    )

    assert not eval_result["score"]
