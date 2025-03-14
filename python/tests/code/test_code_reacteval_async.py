"""Tests for the async React code evaluator."""

import pytestw
from openevals.code.reacteval import create_async_react_evaluator


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extraction_strategy, outputs, description",
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
async def test_async_react_evaluator(
    extraction_strategy, outputs, description
):
    """Test the async React code evaluator with different extraction strategies."""
    
    # Create evaluator with real API calls
    evaluator = await create_async_react_evaluator(
        extraction_strategy=extraction_strategy,
        extraction_model="openai:gpt-4" if extraction_strategy == "llm" else None,
        evaluation_model="openai:gpt-4o",
        feedback_key="react_test",
    )
    
    # Run evaluation
    result = await evaluator(outputs=outputs, description=description)
    
    # Verify the result
    assert result["key"] == "react_test"
    assert isinstance(result["score"], bool)
    assert isinstance(result["comment"], str)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extraction_strategy, code_extractor, outputs, description",
    [
        (
            "none",
            lambda x: x.strip(),  # Simple custom extractor that just strips whitespace
            f"\n{SAMPLE_REACT_CODE}\n",
            "A blue button with rounded corners that says 'Click Me'",
        ),
    ],
)
async def test_async_react_evaluator_with_custom_extractor(
    extraction_strategy, code_extractor, outputs, description
):
    """Test the async React code evaluator with a custom code extractor."""
    
    # Create evaluator with real API and custom extractor
    evaluator = await create_async_react_evaluator(
        extraction_strategy=extraction_strategy,
        code_extractor=code_extractor,
        evaluation_model="openai:gpt-4o",
        feedback_key="react_test",
    )
    
    # Run evaluation
    result = await evaluator(outputs=outputs, description=description)
    
    # Verify the result
    assert result["key"] == "react_test"
    assert isinstance(result["score"], bool)
    assert isinstance(result["comment"], str)


@pytest.mark.asyncio
async def test_async_react_evaluator_missing_description():
    """Test the async React code evaluator when the description field is missing."""
    
    evaluator = await create_async_react_evaluator(extraction_strategy="none")
    
    # Run evaluation without providing a description
    result = await evaluator(outputs=SAMPLE_REACT_CODE)
    
    # Verify the result indicates an error due to missing description
    assert result["key"] == "react_succeeded"
    assert result["score"] is False
    assert "Missing required description field" in result["comment"]


@pytest.mark.asyncio
async def test_async_react_evaluator_custom_description_field():
    """Test the async React code evaluator with a custom description field name."""
    
    evaluator = await create_async_react_evaluator(
        extraction_strategy="none",
        description_field="custom_desc",
        evaluation_model="openai:gpt-4o",
    )
    
    # Run evaluation with the custom description field
    result = await evaluator(
        outputs=SAMPLE_REACT_CODE,
        custom_desc="A blue button with 'Click Me' text",
    )
    
    # Verify the result
    assert result["key"] == "react_succeeded"
    assert isinstance(result["score"], bool)
    assert isinstance(result["comment"], str)