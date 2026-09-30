import itertools
import json

import pytest

from openevals.string.levenshtein import _scorer


def _matrix_score(left, right):
    """Full-matrix reference for checking the rolling-row implementation."""
    matrix = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        matrix[i][0] = i
    for j in range(len(right) + 1):
        matrix[0][j] = j
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + (a != b),
            )
    length = max(len(left), len(right))
    return 1 - matrix[-1][-1] / length if length else 1.0


def test_levenshtein_matches_full_matrix():
    strings = [
        "".join(chars)
        for length in range(4)
        for chars in itertools.product("ab", repeat=length)
    ]
    strings += ["kitten", "sitting", "你好", "你好吗", "🙂a", "🙂"]
    for left, right in itertools.product(strings, repeat=2):
        assert _scorer(left, right) == _matrix_score(left, right), (left, right)


@pytest.mark.parametrize("left,right", [("a", "b" * 512), ("", "abc"), ("abc", "abc")])
def test_levenshtein_unequal_lengths_and_symmetry(left, right):
    expected = _matrix_score(left, right)
    assert _scorer(left, right) == expected
    assert _scorer(right, left) == expected


@pytest.mark.parametrize("left,right", [({"a": 1}, {"a": 2}), ([1, 2], []), (False, 0)])
def test_levenshtein_preserves_json_serialization(left, right):
    assert _scorer(left, right) == _matrix_score(json.dumps(left), json.dumps(right))


@pytest.mark.parametrize("left,right", [(None, "a"), ("a", None)])
def test_levenshtein_requires_both_outputs(left, right):
    with pytest.raises(ValueError, match="requires both"):
        _scorer(left, right)
