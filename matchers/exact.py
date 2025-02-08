from matchers.base import MatchResult, MatcherKwargs
import json

def exact_match(data: MatcherKwargs) -> MatchResult:
    """
    Performs exact string matching between query and target strings.
    
    Args:
        data (dict): Dictionary containing 'input' and 'output' strings to compare
        
    Returns:
        MatchResult: Contains match result with score True for exact match, False otherwise
    """

    # Convert both to JSON strings for deep comparison
    inputs_json = json.dumps(data.inputs, sort_keys=True)
    outputs_json = json.dumps(data.outputs, sort_keys=True)
    return {
        'key': "equal",
        'score': True if inputs_json == outputs_json else False,
    }
