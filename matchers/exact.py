from matchers.base import MatchResult
import json


def exact_matcher(*, inputs: dict, outputs: dict) -> MatchResult:
    """
    Performs exact string matching between input and output values.

    Args:
        inputs (dict): Input dictionary to compare
        outputs (dict): Output dictionary to compare

    Returns:
        MatchResult: Contains match result with score 1.0 for exact match, 0.0 otherwise
    """
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise TypeError("Both inputs and outputs must be dictionaries")

    # Convert both to JSON strings for deep comparison
    inputs_json = json.dumps(inputs, sort_keys=True)
    outputs_json = json.dumps(outputs, sort_keys=True)
    return MatchResult(key="equal", score=1.0 if inputs_json == outputs_json else 0.0)
