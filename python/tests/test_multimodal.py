import base64
import urllib.request

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langsmith import testing as t
from openevals.utils import _attachment_to_content_block
from openevals.llm import create_llm_as_judge
from openevals.prompts.image import IMAGE_RELEVANCE_PROMPT
from openevals.prompts.voice import (
    AUDIO_QUALITY_PROMPT,
    TRANSCRIPTION_ACCURACY_PROMPT,
    VOCAL_AFFECT_PROMPT,
    DIALOGUE_FLOW_PROMPT,
)

# Tiny 1×1 white PNG as base64 data URI — usable in unit tests without a real image
TINY_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)

PUBLIC_IMAGE_URL = "https://images.everydayhealth.com/images/2025/fruits-with-protein-help-boost-intake-pomegranate-1440x810.jpg"

# Minimal silent WAV (44-byte header, no samples) — satisfies {attachments} template.
TINY_WAV_DATA_URI = (
    "data:audio/wav;base64,"
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="
)


@pytest.fixture(scope="module")
def fruit_image_b64():
    """Fetch the fruit image once per test session and encode as base64."""
    req = urllib.request.Request(PUBLIC_IMAGE_URL, headers={"User-Agent": "openevals-test/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return "data:image/jpeg;base64," + base64.b64encode(resp.read()).decode()
    except Exception as e:
        pytest.skip(f"Could not fetch test image: {e}")


# ── image/* ────────────────────────────────────────────────────────────────────

def test_image_png_data_uri():
    block = _attachment_to_content_block({"mime_type": "image/png", "data": TINY_PNG_DATA_URI})
    assert block == {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URI}}


def test_image_jpeg_data_uri():
    data = "data:image/jpeg;base64,/9j/abc123"
    block = _attachment_to_content_block({"mime_type": "image/jpeg", "data": data})
    assert block == {"type": "image_url", "image_url": {"url": data}}


def test_image_url_string():
    url = "https://example.com/photo.jpg"
    block = _attachment_to_content_block(url)
    assert block == {"type": "image_url", "image_url": {"url": url}}


# ── application/pdf ────────────────────────────────────────────────────────────

def test_pdf_with_name():
    data = "data:application/pdf;base64,JVBERi0x"
    block = _attachment_to_content_block({"mime_type": "application/pdf", "data": data, "name": "invoice.pdf"})
    assert block == {"type": "file", "file": {"filename": "invoice.pdf", "file_data": data}}


def test_pdf_without_name_defaults_to_attachment():
    data = "data:application/pdf;base64,JVBERi0x"
    block = _attachment_to_content_block({"mime_type": "application/pdf", "data": data})
    assert block == {"type": "file", "file": {"filename": "attachment.pdf", "file_data": data}}


# ── audio/* ────────────────────────────────────────────────────────────────────

def test_audio_mp3_strips_data_prefix():
    raw_base64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGFtZQ=="
    data = f"data:audio/mp3;base64,{raw_base64}"
    block = _attachment_to_content_block({"mime_type": "audio/mp3", "data": data})
    assert block == {"type": "input_audio", "input_audio": {"data": raw_base64, "format": "mp3"}}


def test_audio_wav_no_prefix():
    raw_base64 = "UklGRiQAAABXQVZFZm10"
    block = _attachment_to_content_block({"mime_type": "audio/wav", "data": raw_base64})
    assert block == {"type": "input_audio", "input_audio": {"data": raw_base64, "format": "wav"}}


def test_audio_mpeg_format():
    raw_base64 = "SUQzBAA"
    data = f"data:audio/mpeg;base64,{raw_base64}"
    block = _attachment_to_content_block({"mime_type": "audio/mpeg", "data": data})
    assert block == {"type": "input_audio", "input_audio": {"data": raw_base64, "format": "mp3"}}


# ── pre-formatted content block passthrough ────────────────────────────────────

def test_pre_formatted_dict_passthrough():
    content_block = {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
    assert _attachment_to_content_block(content_block) == content_block


def test_pre_formatted_input_audio_passthrough():
    content_block = {"type": "input_audio", "input_audio": {"data": "abc", "format": "mp3"}}
    assert _attachment_to_content_block(content_block) == content_block


# ── error cases ────────────────────────────────────────────────────────────────

def test_unsupported_mime_type_raises():
    with pytest.raises(ValueError, match="Unsupported attachment MIME type"):
        _attachment_to_content_block({"mime_type": "video/mp4", "data": "data:video/mp4;base64,abc"})


def test_non_string_non_dict_raises():
    with pytest.raises(ValueError, match="Unsupported attachment type"):
        _attachment_to_content_block(12345)


# ── integration: message structure ────────────────────────────────────────────

def test_single_attachment_becomes_list_content():
    """Without calling an LLM, verify the scorer builds the right message shape."""
    from openevals.llm import _create_llm_as_judge_scorer
    from unittest.mock import MagicMock, patch

    scorer = _create_llm_as_judge_scorer(
        prompt="Evaluate this image: {outputs}\n{attachments}",
        model="openai:gpt-5-mini",
    )

    with patch("openevals.llm.init_chat_model") as mock_init:
        mock_judge = MagicMock(spec=BaseChatModel)
        mock_judge.with_structured_output.return_value.invoke.return_value = {
            "score": True,
            "reasoning": "looks good"
        }
        mock_init.return_value = mock_judge

        scorer(
            outputs="a fruit bowl",
            attachments={"mime_type": "image/png", "data": TINY_PNG_DATA_URI},
        )

    call_args = mock_judge.with_structured_output.return_value.invoke.call_args
    messages = call_args[0][0]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == TINY_PNG_DATA_URI


def test_multiple_attachments_all_appended():
    """Multiple attachments each become their own content block."""
    from openevals.llm import _create_llm_as_judge_scorer
    from unittest.mock import MagicMock, patch

    scorer = _create_llm_as_judge_scorer(
        prompt="Evaluate: {outputs}\n{attachments}",
        model="openai:gpt-5-mini",
    )

    with patch("openevals.llm.init_chat_model") as mock_init:
        mock_judge = MagicMock(spec=BaseChatModel)
        mock_judge.with_structured_output.return_value.invoke.return_value = {
            "score": True, "reasoning": "ok"
        }
        mock_init.return_value = mock_judge

        scorer(
            outputs="two images",
            attachments=[
                {"mime_type": "image/png", "data": TINY_PNG_DATA_URI},
                {"mime_type": "image/jpeg", "data": "data:image/jpeg;base64,/9j/"},
            ],
        )

    call_args = mock_judge.with_structured_output.return_value.invoke.call_args
    messages = call_args[0][0]
    content = messages[0]["content"]
    assert len(content) == 3  # text + 2 images
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "image_url"


def test_no_attachment_content_is_plain_string():
    """When no attachments, content stays a plain string (no performance regression)."""
    from openevals.llm import _create_llm_as_judge_scorer
    from unittest.mock import MagicMock, patch

    scorer = _create_llm_as_judge_scorer(
        prompt="Evaluate: {outputs}",
        model="openai:gpt-5-mini",
    )

    with patch("openevals.llm.init_chat_model") as mock_init:
        mock_judge = MagicMock(spec=BaseChatModel)
        mock_judge.with_structured_output.return_value.invoke.return_value = {
            "score": True, "reasoning": "ok"
        }
        mock_init.return_value = mock_judge

        scorer(outputs="some output")

    call_args = mock_judge.with_structured_output.return_value.invoke.call_args
    messages = call_args[0][0]
    assert isinstance(messages[0]["content"], str)


# ── LLM integration (requires API key + vision model) ─────────────────────────

@pytest.mark.langsmith
def test_image_relevance_with_real_image(fruit_image_b64):
    evaluator = create_llm_as_judge(
        prompt=IMAGE_RELEVANCE_PROMPT,
        feedback_key="image_relevance",
        model="openai:gpt-5-mini",
    )
    t.log_inputs({"inputs": "Show me a picture of fruits", "outputs": "Here is an image of various fruits"})
    t.log_reference_outputs({"score": True})
    result = evaluator(
        inputs="Show me a picture of fruits",
        outputs="Here is an image of various fruits",
        attachments={"mime_type": "image/jpeg", "data": fruit_image_b64},
    )
    t.log_outputs({"score": result["score"]})
    assert result["score"]


@pytest.mark.langsmith
def test_image_relevance_irrelevant_image(fruit_image_b64):
    evaluator = create_llm_as_judge(
        prompt=IMAGE_RELEVANCE_PROMPT,
        feedback_key="image_relevance",
        model="openai:gpt-5-mini",
    )
    t.log_inputs({"inputs": "Show me a photo of a sports car", "outputs": "Here is a red Ferrari"})
    t.log_reference_outputs({"score": False})
    result = evaluator(
        inputs="Show me a photo of a sports car",
        outputs="Here is a red Ferrari",
        attachments={"mime_type": "image/jpeg", "data": fruit_image_b64},
    )
    t.log_outputs({"score": result["score"]})
    assert not result["score"]


# ── LLM integration: voice (requires API key + audio model) ───────────────────

@pytest.mark.langsmith
def test_audio_quality_issues_detected():
    evaluator = create_llm_as_judge(
        prompt=AUDIO_QUALITY_PROMPT,
        feedback_key="audio_quality",
        model="google_genai:gemini-2.0-flash",
    )
    inputs = "Customer service call recording"
    outputs = "Audio exhibits severe clipping throughout with loud pops and dropouts every few seconds"
    t.log_inputs({"inputs": inputs, "outputs": outputs})
    t.log_reference_outputs({"score": True})
    result = evaluator(
        inputs=inputs,
        outputs=outputs,
        attachments={"mime_type": "audio/wav", "data": TINY_WAV_DATA_URI},
    )
    t.log_outputs({"score": result["score"]})
    assert result["score"]


@pytest.mark.langsmith
def test_audio_quality_clean():
    evaluator = create_llm_as_judge(
        prompt=AUDIO_QUALITY_PROMPT,
        feedback_key="audio_quality",
        model="google_genai:gemini-2.0-flash",
    )
    inputs = "Customer service call recording"
    outputs = "Audio is clear with no distortion, clipping, or glitches detected"
    t.log_inputs({"inputs": inputs, "outputs": outputs})
    t.log_reference_outputs({"score": False})
    result = evaluator(
        inputs=inputs,
        outputs=outputs,
        attachments={"mime_type": "audio/wav", "data": TINY_WAV_DATA_URI},
    )
    t.log_outputs({"score": result["score"]})
    assert not result["score"]


@pytest.mark.langsmith
def test_transcription_accurate():
    evaluator = create_llm_as_judge(
        prompt=TRANSCRIPTION_ACCURACY_PROMPT,
        feedback_key="transcription_accuracy",
        model="google_genai:gemini-2.0-flash",
    )
    inputs = "Please reschedule the meeting with Dr. Johnson to Thursday at 3pm."
    outputs = "Please reschedule the meeting with Dr. Johnson to Thursday at 3pm."
    t.log_inputs({"inputs": inputs, "outputs": outputs})
    t.log_reference_outputs({"score": True})
    result = evaluator(
        inputs=inputs,
        outputs=outputs,
        attachments={"mime_type": "audio/wav", "data": TINY_WAV_DATA_URI},
    )
    t.log_outputs({"score": result["score"]})
    assert result["score"]


@pytest.mark.langsmith
def test_transcription_inaccurate():
    evaluator = create_llm_as_judge(
        prompt=TRANSCRIPTION_ACCURACY_PROMPT,
        feedback_key="transcription_accuracy",
        model="google_genai:gemini-2.0-flash",
    )
    inputs = "Please reschedule the meeting with Dr. Johnson to Thursday at 3pm."
    outputs = "Please cancel the meeting with Dr. Jackson to Friday at 2pm."
    t.log_inputs({"inputs": inputs, "outputs": outputs})
    t.log_reference_outputs({"score": False})
    result = evaluator(
        inputs=inputs,
        outputs=outputs,
        attachments={"mime_type": "audio/wav", "data": TINY_WAV_DATA_URI},
    )
    t.log_outputs({"score": result["score"]})
    assert not result["score"]


@pytest.mark.langsmith
def test_vocal_affect_appropriate():
    evaluator = create_llm_as_judge(
        prompt=VOCAL_AFFECT_PROMPT,
        feedback_key="vocal_affect",
        model="google_genai:gemini-2.0-flash",
    )
    inputs = "User asks for help with a billing issue"
    outputs = "Agent speaks with warm, empathetic tone throughout, using natural pacing and appropriate emphasis"
    t.log_inputs({"inputs": inputs, "outputs": outputs})
    t.log_reference_outputs({"score": True})
    result = evaluator(
        inputs=inputs,
        outputs=outputs,
        attachments={"mime_type": "audio/wav", "data": TINY_WAV_DATA_URI},
    )
    t.log_outputs({"score": result["score"]})
    assert result["score"]


@pytest.mark.langsmith
def test_vocal_affect_inappropriate():
    evaluator = create_llm_as_judge(
        prompt=VOCAL_AFFECT_PROMPT,
        feedback_key="vocal_affect",
        model="google_genai:gemini-2.0-flash",
    )
    inputs = "User explains they just lost their job and need urgent financial help"
    outputs = "Agent responds with cheerful, upbeat tone as if reading a promotional script, with rapid delivery"
    t.log_inputs({"inputs": inputs, "outputs": outputs})
    t.log_reference_outputs({"score": False})
    result = evaluator(
        inputs=inputs,
        outputs=outputs,
        attachments={"mime_type": "audio/wav", "data": TINY_WAV_DATA_URI},
    )
    t.log_outputs({"score": result["score"]})
    assert not result["score"]


@pytest.mark.langsmith
def test_dialogue_flow_natural():
    evaluator = create_llm_as_judge(
        prompt=DIALOGUE_FLOW_PROMPT,
        feedback_key="dialogue_flow",
        model="google_genai:gemini-2.0-flash",
    )
    inputs = "Support call between agent and customer"
    outputs = "Clean turn-taking throughout; each speaker waited for the other to finish before responding"
    t.log_inputs({"inputs": inputs, "outputs": outputs})
    t.log_reference_outputs({"score": True})
    result = evaluator(
        inputs=inputs,
        outputs=outputs,
        attachments={"mime_type": "audio/wav", "data": TINY_WAV_DATA_URI},
    )
    t.log_outputs({"score": result["score"]})
    assert result["score"]


@pytest.mark.langsmith
def test_dialogue_flow_unnatural():
    evaluator = create_llm_as_judge(
        prompt=DIALOGUE_FLOW_PROMPT,
        feedback_key="dialogue_flow",
        model="google_genai:gemini-2.0-flash",
    )
    inputs = "Support call between agent and customer"
    outputs = "Frequent cross-talk with agent and user speaking simultaneously throughout, causing confusion and requiring constant repetition"
    t.log_inputs({"inputs": inputs, "outputs": outputs})
    t.log_reference_outputs({"score": False})
    result = evaluator(
        inputs=inputs,
        outputs=outputs,
        attachments={"mime_type": "audio/wav", "data": TINY_WAV_DATA_URI},
    )
    t.log_outputs({"score": result["score"]})
    assert not result["score"]
