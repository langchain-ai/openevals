"""Tests for the React code evaluator."""

import pytest
from openevals.code.react import create_react_ui_llm_as_judge


# Sample React code with a button component
SAMPLE_REACT_CODE = """
function Button() {
  return (
    <button style={{ padding: '10px 20px', background: 'blue', color: 'white', borderRadius: '5px' }}>
      Click Me
    </button>
  );
}
"""

# Sample React code with an error (missing closing tag)
SAMPLE_ERROR_CODE = """
function Button() {
  return (
    <button style={{ padding: '10px 20px', background: 'blue', color: 'white', borderRadius: '5px' }}>
      Click Me
  );
}
"""

# Sample React code in markdown
SAMPLE_MARKDOWN_CODE = """
Here's a React button component:

```jsx
function Button() {
  return (
    <button style={{ padding: '10px 20px', background: 'blue', color: 'white', borderRadius: '5px' }}>
      Click Me
    </button>
  );
}
```
"""

# Sample text with code that needs LLM extraction
SAMPLE_TEXT_WITH_CODE = """
To create a blue button in React, you would write something like:

function Button() {
  return (
    <button style={{ padding: '10px 20px', background: 'blue', color: 'white', borderRadius: '5px' }}>
      Click Me
    </button>
  );
}

This creates a nice looking button with rounded corners.
"""


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "code_extraction_strategy, outputs, description",
    [
        (
            "none",
            SAMPLE_REACT_CODE,
            "A blue button with rounded corners that says 'Click Me'",
        ),
        (
            "none",
            SAMPLE_ERROR_CODE,
            "A blue button with rounded corners that says 'Click Me'",
        ),
        (
            "markdown_code_blocks",
            SAMPLE_MARKDOWN_CODE,
            "A blue button with rounded corners that says 'Click Me'",
        ),
        (
            "llm",
            SAMPLE_TEXT_WITH_CODE,
            "A blue button with rounded corners that says 'Click Me'",
        ),
    ],
)
def test_react_evaluator(code_extraction_strategy, outputs, description):
    """Test the React code evaluator with different extraction strategies."""

    # Create a real evaluator without any mocks
    evaluator = create_react_ui_llm_as_judge(
        code_extraction_strategy=code_extraction_strategy,
        model="openai:gpt-4o",
        feedback_key="react_test",
    )

    # Run evaluation with real API calls
    result = evaluator(outputs=outputs, description=description)

    # Verify the result
    assert result["key"] == "react_test"
    assert isinstance(result["score"], bool)
    assert isinstance(result["comment"], str)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "code_extraction_strategy, code_extractor, outputs, description",
    [
        (
            "none",
            lambda x: x.strip(),  # Simple custom extractor that just strips whitespace
            f"\n{SAMPLE_REACT_CODE}\n",
            "A blue button with rounded corners that says 'Click Me'",
        ),
    ],
)
def test_react_evaluator_with_custom_extractor(
    code_extraction_strategy,
    code_extractor,
    outputs,
    description,
):
    """Test the React code evaluator with a custom code extractor."""

    # Create evaluator with real API and custom extractor
    evaluator = create_react_ui_llm_as_judge(
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model="openai:gpt-4o",
        feedback_key="react_test",
    )

    # Run evaluation
    result = evaluator(outputs=outputs, description=description)

    # Verify the result
    assert result["key"] == "react_test"
    assert isinstance(result["score"], bool)
    assert isinstance(result["comment"], str)
