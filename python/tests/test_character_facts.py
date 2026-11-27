import json

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from openevals.character_facts import format_character_facts
from openevals.code.llm import (
    create_async_code_llm_as_judge,
    create_code_llm_as_judge,
)
from openevals.prompts import CODE_CORRECTNESS_PROMPT


@pytest.mark.parametrize("text", ["", " 42 \t\r\n", "café 中文 Ελληνικά 😀"])
def test_no_metadata_for_ordinary_text(text):
    assert format_character_facts(text) == ""


@pytest.mark.parametrize(
    "character, expected",
    [
        ("\u00a0", "U+00A0 NO-BREAK SPACE"),
        ("\u202f", "U+202F NARROW NO-BREAK SPACE"),
        ("\u200b", "U+200B ZERO WIDTH SPACE"),
        ("\u2060", "U+2060 WORD JOINER"),
        ("\ufeff", "U+FEFF ZERO WIDTH NO-BREAK SPACE"),
        ("\u202e", "U+202E RIGHT-TO-LEFT OVERRIDE"),
    ],
)
def test_character_names_and_escaped_context(character, expected):
    source = f"a{character}b"
    panel = format_character_facts(source)
    assert expected in panel
    assert "escaped context: " + json.dumps(source) in panel
    assert panel.isascii()


def test_positions_use_code_points_and_lf_lines():
    panel = format_character_facts("😀\u00a0\r\n\u2028x\u200b")
    assert "line 1, column 2: U+00A0" in panel
    assert "line 2, column 1: U+2028" in panel
    assert "line 2, column 3: U+200B" in panel


def test_actual_character_differs_from_literal_escape():
    assert format_character_facts(r"\u00a0") == ""
    assert "U+00A0" in format_character_facts("\u00a0")


def test_bounded_entries_and_context_report_omissions():
    panel = format_character_facts(
        "\u00a0" * 10_000, max_occurrences=2, context_radius=0
    )
    assert panel.count("- line ") == 2
    assert panel.count('escaped context: "\\u00a0"') == 2
    assert "Omitted 9998 additional matching characters." in panel
    assert len(panel) < 500


def test_context_does_not_include_distant_text():
    panel = format_character_facts(
        "not-context--ab\u00a0cd--not-context", context_radius=2
    )
    assert 'escaped context: "ab\\u00a0cd"' in panel
    assert "not-context" not in panel
    assert "Omitted" not in panel


def test_legitimate_format_characters_are_descriptive_only():
    panel = format_character_facts("👩\u200d💻", label='answer\n"é')
    assert "U+200D ZERO WIDTH JOINER" in panel
    assert 'Character facts for "answer\\n\\"\\u00e9"' in panel
    assert "not a grading verdict" in panel
    assert panel.isascii()


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"max_occurrences": 0}, ValueError),
        ({"max_occurrences": -1}, ValueError),
        ({"context_radius": -1}, ValueError),
        ({"max_occurrences": 1.5}, TypeError),
        ({"context_radius": 1.5}, TypeError),
        ({"label": None}, TypeError),
    ],
)
def test_invalid_options(kwargs, error):
    with pytest.raises(error):
        format_character_facts("", **kwargs)


def test_non_text_input_rejected():
    with pytest.raises(TypeError):
        format_character_facts(None)


def character_aware_prompt(*, inputs, outputs, **kwargs):
    content = CODE_CORRECTNESS_PROMPT.format(inputs=inputs, outputs=outputs)
    facts = format_character_facts(outputs)
    if facts:
        content += "\n\n" + facts
    return [{"role": "user", "content": content}]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("extraction", ["none", "markdown_code_blocks"])
@pytest.mark.asyncio
async def test_judge_composition_preserves_code_and_keeps_facts_outside_answer(
    use_async, extraction
):
    # Capture the real SDK request; this test does not simulate model accuracy.
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": '{"score": true}'}}
                ]
            },
        )

    transport = httpx.MockTransport(handle)
    if use_async:
        client = AsyncOpenAI(
            api_key="test", http_client=httpx.AsyncClient(transport=transport)
        )
        create = create_async_code_llm_as_judge
    else:
        client = OpenAI(api_key="test", http_client=httpx.Client(transport=transport))
        create = create_code_llm_as_judge

    code = 'assert accepts("\u00a042\u00a0") is False\n'
    original = f"```python\n{code}```" if extraction == "markdown_code_blocks" else code
    evaluator = create(
        prompt=character_aware_prompt,
        judge=client,
        model="test",
        use_reasoning=False,
        code_extraction_strategy=extraction,
    )
    try:
        result = evaluator(inputs="Reject non-ASCII spaces", outputs=original)
        if use_async:
            await result
    finally:
        if use_async:
            await client.close()
        else:
            client.close()

    content = requests[0]["messages"][0]["content"]
    assert f"<output>\n{code}\n</output>" in content
    assert content.index("Character facts") > content.index("</output>")
    assert "line 1, column 17: U+00A0" in content
