import pytest
from langsmith import testing as t
from openevals.llm import create_llm_as_judge
from openevals.prompts.trajectory import (
    TASK_COMPLETION_PROMPT,
    USER_SATISFACTION_PROMPT,
    AGENT_TONE_PROMPT,
    KNOWLEDGE_RETENTION_PROMPT,
    LANGUAGE_DETECTION_PROMPT,
)


# ── TASK_COMPLETION_PROMPT ─────────────────────────────────────────────────────

@pytest.mark.langsmith
def test_task_completion_all_requests_fulfilled():
    evaluator = create_llm_as_judge(
        prompt=TASK_COMPLETION_PROMPT,
        feedback_key="task_completion",
        model="openai:gpt-5-mini",
    )
    conversation = [
        {"role": "user", "content": "Can you summarize this article for me?"},
        {"role": "assistant", "content": "Sure! The article discusses climate change impacts on coastal cities, rising sea levels, and mitigation strategies."},
        {"role": "user", "content": "Thanks, that was exactly what I needed."},
    ]
    t.log_inputs({"conversation": conversation})
    t.log_reference_outputs({"score": True})
    result = evaluator(outputs=conversation)
    t.log_outputs({"score": result["score"]})
    assert result["score"]


@pytest.mark.langsmith
def test_task_completion_request_not_fulfilled():
    evaluator = create_llm_as_judge(
        prompt=TASK_COMPLETION_PROMPT,
        feedback_key="task_completion",
        model="openai:gpt-5-mini",
    )
    conversation = [
        {"role": "user", "content": "Can you book a flight from NYC to Paris?"},
        {"role": "assistant", "content": "I can provide information about flights, but I cannot actually book them for you."},
        {"role": "user", "content": "I asked you to book it, not just give me info. Can you please just do it?"},
        {"role": "assistant", "content": "I understand your frustration but I'm unable to make bookings."},
    ]
    t.log_inputs({"conversation": conversation})
    t.log_reference_outputs({"score": False})
    result = evaluator(outputs=conversation)
    t.log_outputs({"score": result["score"]})
    assert not result["score"]


# ── USER_SATISFACTION_PROMPT ───────────────────────────────────────────────────

@pytest.mark.langsmith
def test_user_satisfaction_happy_user():
    evaluator = create_llm_as_judge(
        prompt=USER_SATISFACTION_PROMPT,
        feedback_key="user_satisfaction",
        model="openai:gpt-5-mini",
    )
    conversation = [
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "Go to Settings > Account > Reset Password and follow the steps."},
        {"role": "user", "content": "Perfect, that worked! Thank you so much."},
    ]
    t.log_inputs({"conversation": conversation})
    t.log_reference_outputs({"score": True})
    result = evaluator(outputs=conversation)
    t.log_outputs({"score": result["score"]})
    assert result["score"]


@pytest.mark.langsmith
def test_user_satisfaction_frustrated_user():
    evaluator = create_llm_as_judge(
        prompt=USER_SATISFACTION_PROMPT,
        feedback_key="user_satisfaction",
        model="openai:gpt-5-mini",
    )
    conversation = [
        {"role": "user", "content": "I need a refund for my order."},
        {"role": "assistant", "content": "Please contact our billing department."},
        {"role": "user", "content": "I already did that three times. This is useless. I'll just dispute it with my bank."},
    ]
    t.log_inputs({"conversation": conversation})
    t.log_reference_outputs({"score": False})
    result = evaluator(outputs=conversation)
    t.log_outputs({"score": result["score"]})
    assert not result["score"]


# ── AGENT_TONE_PROMPT ──────────────────────────────────────────────────────────

@pytest.mark.langsmith
def test_agent_tone_appropriate():
    evaluator = create_llm_as_judge(
        prompt=AGENT_TONE_PROMPT,
        feedback_key="agent_tone",
        model="openai:gpt-5-mini",
    )
    conversation = [
        {"role": "user", "content": "I'm really confused about how this works."},
        {"role": "assistant", "content": "No worries at all — it can be tricky at first! Let me walk you through it step by step."},
        {"role": "user", "content": "Oh I see, that makes much more sense now."},
        {"role": "assistant", "content": "Great! Let me know if you have any other questions."},
    ]
    t.log_inputs({"conversation": conversation})
    t.log_reference_outputs({"score": True})
    result = evaluator(outputs=conversation)
    t.log_outputs({"score": result["score"]})
    assert result["score"]


@pytest.mark.langsmith
def test_agent_tone_condescending():
    evaluator = create_llm_as_judge(
        prompt=AGENT_TONE_PROMPT,
        feedback_key="agent_tone",
        model="openai:gpt-5-mini",
    )
    conversation = [
        {"role": "user", "content": "I don't understand this feature."},
        {"role": "assistant", "content": "This is extremely basic. It's explained clearly in the documentation. Did you even read it?"},
        {"role": "user", "content": "I did but it wasn't clear to me."},
        {"role": "assistant", "content": "Well, most users find it obvious. Maybe try reading more carefully."},
    ]
    t.log_inputs({"conversation": conversation})
    t.log_reference_outputs({"score": False})
    result = evaluator(outputs=conversation)
    t.log_outputs({"score": result["score"]})
    assert not result["score"]


# ── KNOWLEDGE_RETENTION_PROMPT ─────────────────────────────────────────────────

@pytest.mark.langsmith
def test_knowledge_retention_good():
    evaluator = create_llm_as_judge(
        prompt=KNOWLEDGE_RETENTION_PROMPT,
        feedback_key="knowledge_retention",
        model="openai:gpt-5-mini",
    )
    conversation = [
        {"role": "user", "content": "My name is Alex and I'm planning a trip to Japan in March."},
        {"role": "assistant", "content": "That sounds exciting, Alex! March is a great time — cherry blossom season."},
        {"role": "user", "content": "What should I pack?"},
        {"role": "assistant", "content": "For Japan in March, bring layers since temperatures can vary. A light jacket is essential for cherry blossom viewing, Alex."},
    ]
    t.log_inputs({"conversation": conversation})
    t.log_reference_outputs({"score": True})
    result = evaluator(outputs=conversation)
    t.log_outputs({"score": result["score"]})
    assert result["score"]


@pytest.mark.langsmith
def test_knowledge_retention_forgot_context():
    evaluator = create_llm_as_judge(
        prompt=KNOWLEDGE_RETENTION_PROMPT,
        feedback_key="knowledge_retention",
        model="openai:gpt-5-mini",
    )
    conversation = [
        {"role": "user", "content": "My name is Sarah and I'm vegetarian."},
        {"role": "assistant", "content": "Nice to meet you, Sarah!"},
        {"role": "user", "content": "Can you recommend a restaurant for dinner?"},
        {"role": "assistant", "content": "I'd suggest a great steakhouse — they have excellent grilled meats and a fantastic ribeye."},
    ]
    t.log_inputs({"conversation": conversation})
    t.log_reference_outputs({"score": False})
    result = evaluator(outputs=conversation)
    t.log_outputs({"score": result["score"]})
    assert not result["score"]


# ── LANGUAGE_DETECTION_PROMPT ──────────────────────────────────────────────────

@pytest.mark.langsmith
def test_language_detection_spanish():
    evaluator = create_llm_as_judge(
        prompt=LANGUAGE_DETECTION_PROMPT,
        feedback_key="language_detection",
        model="openai:gpt-5-mini",
        output_schema={
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "score": {"type": "string", "description": "The detected language name in English"},
            },
            "required": ["reasoning", "score"],
            "additionalProperties": False,
        },
    )
    conversation = [
        {"role": "user", "content": "Hola, ¿cómo estás?"},
        {"role": "assistant", "content": "¡Hola! Estoy bien, gracias. ¿En qué puedo ayudarte?"},
        {"role": "user", "content": "Necesito ayuda con mi cuenta."},
    ]
    t.log_inputs({"conversation": conversation})
    t.log_reference_outputs({"score": "Spanish"})
    result = evaluator(outputs=conversation)
    t.log_outputs({"score": result["score"]})
    assert result["score"].lower() == "spanish"


@pytest.mark.langsmith
def test_language_detection_french():
    evaluator = create_llm_as_judge(
        prompt=LANGUAGE_DETECTION_PROMPT,
        feedback_key="language_detection",
        model="openai:gpt-5-mini",
        output_schema={
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "score": {"type": "string", "description": "The detected language name in English"},
            },
            "required": ["reasoning", "score"],
            "additionalProperties": False,
        },
    )
    conversation = [
        {"role": "user", "content": "Bonjour, j'ai besoin d'aide."},
        {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider?"},
        {"role": "user", "content": "Je ne comprends pas cette fonctionnalité."},
    ]
    t.log_inputs({"conversation": conversation})
    t.log_reference_outputs({"score": "French"})
    result = evaluator(outputs=conversation)
    t.log_outputs({"score": result["score"]})
    assert result["score"].lower() == "french"
