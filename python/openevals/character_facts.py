"""Optional, descriptive Unicode metadata for text supplied to an LLM judge."""

import json
import unicodedata

__all__ = ["format_character_facts"]


def format_character_facts(
    text: str,
    *,
    label: str = "outputs",
    max_occurrences: int = 20,
    context_radius: int = 20,
) -> str:
    """Describe non-ASCII whitespace and Unicode format (Cf) characters.

    Returns an ASCII-only sidecar with code points, Unicode names when available,
    positions, and JSON-escaped context. It never normalizes or replaces ``text``.
    Returns an empty string when there are no matching characters.

    Include the result separately from the original answer in a judge prompt.
    Character presence is descriptive evidence, not a correctness verdict. This
    helper does not detect homoglyphs or every invisible character, and does not
    guarantee that a judge will interpret the evidence correctly.

    Args:
        text: The original text to inspect.
        label: The evidence field being described, JSON-escaped in the header.
        max_occurrences: Maximum number of entries to emit; must be positive.
            Additional matches are counted and reported as omitted.
        context_radius: Maximum number of Unicode code points to include on each
            side of a match; must be nonnegative. Context can span lines, with
            line endings escaped. Zero includes only the matching character.

    Positions are one-based Unicode code-point positions, not byte offsets or
    grapheme columns. Only LF (U+000A) advances the line number. When called after
    code extraction, positions refer to the extracted text, not the original
    response. Metadata cannot recover characters removed by earlier processing.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    if not isinstance(max_occurrences, int) or not isinstance(context_radius, int):
        raise TypeError("max_occurrences and context_radius must be integers")
    if max_occurrences < 1:
        raise ValueError("max_occurrences must be positive")
    if context_radius < 0:
        raise ValueError("context_radius must be nonnegative")

    entries: list[str] = []
    total = 0
    line, column = 1, 1
    for index, character in enumerate(text):
        if ord(character) > 127 and (
            character.isspace() or unicodedata.category(character) == "Cf"
        ):
            total += 1
            if len(entries) < max_occurrences:
                context = text[
                    max(0, index - context_radius) : index + context_radius + 1
                ]
                name = unicodedata.name(character, "UNNAMED")
                entries.append(
                    f"- line {line}, column {column}: "
                    f"U+{ord(character):04X} {name}; "
                    f"escaped context: {json.dumps(context, ensure_ascii=True)}"
                )
        if character == "\n":
            line, column = line + 1, 1
        else:
            column += 1

    if not entries:
        return ""

    header = (
        f"Character facts for {json.dumps(label, ensure_ascii=True)} "
        "(descriptive metadata, not a grading verdict).\n"
        "Positions are one-based Unicode code points; lines are separated by LF."
    )
    if total > len(entries):
        entries.append(
            f"Omitted {total - len(entries)} additional matching characters."
        )
    return "\n".join([header, *entries])
