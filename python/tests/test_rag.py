import pytest

from langsmith import testing as t

from openevals.llm import create_llm_as_judge
from openevals.prompts.rag.helpfulness import RAG_HELPFULNESS_PROMPT
from openevals.prompts.rag.groundedness import RAG_GROUNDEDNESS_PROMPT
from openevals.prompts.rag.retrieval_relevance import RAG_RETRIEVAL_RELEVANCE_PROMPT


@pytest.mark.langsmith
def test_llm_as_judge_rag_helpfulnesss():
    inputs = {
        "question": "What is the boiling point of water?",
    }
    outputs = {
        "answer": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."
    }
    t.log_inputs(inputs)
    t.log_outputs(outputs)
    llm_as_judge = create_llm_as_judge(
        prompt=RAG_HELPFULNESS_PROMPT,
        feedback_key="helpfulness",
        model="openai:gpt-5-mini",
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
        model="openai:gpt-5-mini",
    )

    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)

    assert not eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_rag_groundedness():
    retrieval_evaluator = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        model="openai:gpt-5-mini",
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
        model="openai:gpt-5-mini",
    )

    inputs = {
        "question": "Where was the first president of FoobarLand born?",
    }

    context = {
        "documents": [
            "The Eiffel Tower was constructed between 1887 and 1889 in Paris, France.",
            "Photosynthesis is the process by which plants convert sunlight into energy.",
            "The Amazon River is the largest river in the world by discharge volume.",
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        ],
    }

    eval_result = retrieval_relevance_evaluator(
        inputs=inputs,
        context=context,
    )

    assert not eval_result["score"]
