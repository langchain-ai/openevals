"""Tests for the React code evaluator."""

import pytest
from openevals.code.reacteval import create_react_evaluator


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
    "extraction_strategy, outputs, description, mock_success, mock_comment",
    [
        (
            "none",
            SAMPLE_REACT_CODE,
            "A blue button with rounded corners that says 'Click Me'",
            True,
            "The component matches the description perfectly. It's a blue button with rounded corners and 'Click Me' text.",
        ),
        (
            "none",
            SAMPLE_ERROR_CODE,
            "A blue button with rounded corners that says 'Click Me'",
            False,
            "There was an error rendering the component due to missing closing tag.",
        ),
        (
            "markdown_code_blocks",
            SAMPLE_MARKDOWN_CODE,
            "A blue button with rounded corners that says 'Click Me'",
            True,
            "The component matches the description perfectly. It's a blue button with rounded corners and 'Click Me' text.",
        ),
        (
            "llm",
            SAMPLE_TEXT_WITH_CODE,
            "A blue button with rounded corners that says 'Click Me'",
            True,
            "The component matches the description perfectly. It's a blue button with rounded corners and 'Click Me' text.",
        ),
    ],
)
def test_react_evaluator(
    extraction_strategy, outputs, description, mock_success, mock_comment
):
    """Test the React code evaluator with different extraction strategies."""
    
    # Create a real evaluator without any mocks
    evaluator = create_react_evaluator(
        extraction_strategy=extraction_strategy,
        extraction_model="openai:gpt-4o" if extraction_strategy == "llm" else None,
        evaluation_model="openai:gpt-4o",
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
    "extraction_strategy, code_extractor, outputs, description, mock_success, mock_comment",
    [
        (
            "none",
            lambda x: x.strip(),  # Simple custom extractor that just strips whitespace
            f"\n{SAMPLE_REACT_CODE}\n",
            "A blue button with rounded corners that says 'Click Me'",
            True,
            "The component matches the description perfectly.",
        ),
    ],
)
def test_react_evaluator_with_custom_extractor(
    extraction_strategy, code_extractor, outputs, description, mock_success, mock_comment
):
    """Test the React code evaluator with a custom code extractor."""
    
    # Create evaluator with real API and custom extractor
    evaluator = create_react_evaluator(
        extraction_strategy=extraction_strategy,
        code_extractor=code_extractor,
        evaluation_model="openai:gpt-4o",
        feedback_key="react_test",
    )
    
    # Run evaluation
    result = evaluator(outputs=outputs, description=description)
    
    # Verify the result
    assert result["key"] == "react_test"
    assert isinstance(result["score"], bool)
    assert isinstance(result["comment"], str)

@pytest.mark.langsmith
def test_react_evaluator_missing_description():
    """Test the React code evaluator when the description field is missing."""
    
    evaluator = create_react_evaluator(extraction_strategy="none")
    
    # Run evaluation without providing a description
    result = evaluator(outputs=SAMPLE_REACT_CODE)
    
    # Verify the result indicates an error due to missing description
    assert result["key"] == "react_succeeded"
    assert result["score"] is False
    assert "Missing required description field" in result["comment"]

@pytest.mark.langsmith
def test_react_evaluator_custom_description_field():
    """Test the React code evaluator with a custom description field name."""
    
    evaluator = create_react_evaluator(
        extraction_strategy="none",
        description_field="custom_desc",
        evaluation_model="openai:gpt-4o",
    )
    
    # Run evaluation with the custom description field
    result = evaluator(
        outputs=SAMPLE_REACT_CODE,
        custom_desc="A blue button with 'Click Me' text",
    )
    
    # Verify the result
    assert result["key"] == "react_succeeded"
    # Can't assert exact score since it depends on the actual API response
    assert isinstance(result["score"], bool)
    assert isinstance(result["comment"], str)