"""
Script for setting up a LangSmith dataset to test tool calling functionality.

This script provides utilities to:
1. Create and manage LangSmith datasets for tool calling evaluation
2. Process input messages and extract tool calls from AI responses
3. Run evaluations using existing openevals utilities

The input should be a list of messages, and the output should be a list of 
tool calls and their arguments (not AI messages with tool calls).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Union

from langsmith import Client
from langchain_core.messages import BaseMessage

from openevals.types import ChatCompletionMessage, EvaluatorResult
from openevals.utils import _normalize_to_openai_messages_list, _run_evaluator


def extract_tool_calls_from_messages(
    messages: List[ChatCompletionMessage],
) -> List[Dict[str, Any]]:
    """
    Extract tool calls from a list of messages.
    
    Args:
        messages: List of chat completion messages
        
    Returns:
        List of tool calls with their names and arguments
    """
    tool_calls = []
    
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                function_info = tool_call.get("function", {})
                tool_call_data = {
                    "id": tool_call.get("id", str(uuid.uuid4())),
                    "name": function_info.get("name", ""),
                    "arguments": function_info.get("arguments", "")
                }
                
                # Parse arguments if they're a JSON string
                if isinstance(tool_call_data["arguments"], str):
                    try:
                        tool_call_data["arguments"] = json.loads(tool_call_data["arguments"])
                    except json.JSONDecodeError:
                        # Keep as string if not valid JSON
                        pass
                
                tool_calls.append(tool_call_data)
    
    return tool_calls


def create_tool_calling_dataset(
    dataset_name: str,
    examples: List[Dict[str, Any]],
    description: Optional[str] = None,
    client: Optional[Client] = None,
) -> str:
    """
    Create a LangSmith dataset for tool calling evaluation.
    
    Args:
        dataset_name: Name of the dataset
        examples: List of examples, each containing 'inputs' and 'expected_outputs'
        description: Optional description of the dataset
        client: Optional LangSmith client (will create one if not provided)
        
    Returns:
        Dataset ID
    """
    if client is None:
        client = Client()
    
    # Create the dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description or f"Tool calling evaluation dataset: {dataset_name}",
    )
    
    # Add examples to the dataset
    for example in examples:
        inputs = example.get("inputs", {})
        expected_outputs = example.get("expected_outputs", [])
        
        # Normalize inputs to ensure they're in the right format
        if "messages" in inputs:
            normalized_messages = _normalize_to_openai_messages_list(inputs["messages"])
            inputs["messages"] = normalized_messages
        
        client.create_example(
            dataset_id=dataset.id,
            inputs=inputs,
            outputs={"expected_tool_calls": expected_outputs},
        )
    
    return dataset.id


def process_messages_to_tool_calls(
    messages: Union[List[ChatCompletionMessage], List[BaseMessage], List[dict]],
) -> List[Dict[str, Any]]:
    """
    Process input messages and extract tool calls.
    
    Args:
        messages: List of messages in various formats
        
    Returns:
        List of tool calls with their names and arguments
    """
    # Normalize messages to OpenAI format
    normalized_messages = _normalize_to_openai_messages_list(messages)
    
    # Extract tool calls
    return extract_tool_calls_from_messages(normalized_messages)


def tool_calling_evaluator(
    inputs: Dict[str, Any],
    outputs: Any,
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> EvaluatorResult:
    """
    Evaluate tool calling accuracy by comparing expected vs actual tool calls.
    
    Args:
        inputs: Input data containing messages
        outputs: Actual outputs from the system
        reference_outputs: Expected outputs containing tool calls
        
    Returns:
        EvaluatorResult with score and reasoning
    """
    if reference_outputs is None:
        return EvaluatorResult(
            key="tool_calling_accuracy",
            score=0.0,
            comment="No reference outputs provided",
        )
    
    expected_tool_calls = reference_outputs.get("expected_tool_calls", [])
    
    # Extract actual tool calls from outputs
    if isinstance(outputs, dict) and "messages" in outputs:
        actual_tool_calls = process_messages_to_tool_calls(outputs["messages"])
    elif isinstance(outputs, list):
        actual_tool_calls = process_messages_to_tool_calls(outputs)
    else:
        actual_tool_calls = []
    
    # Compare tool calls
    if len(expected_tool_calls) == 0 and len(actual_tool_calls) == 0:
        return EvaluatorResult(
            key="tool_calling_accuracy",
            score=1.0,
            comment="No tool calls expected or generated - correct",
        )
    
    if len(expected_tool_calls) != len(actual_tool_calls):
        return EvaluatorResult(
            key="tool_calling_accuracy",
            score=0.0,
            comment=f"Expected {len(expected_tool_calls)} tool calls, got {len(actual_tool_calls)}",
        )
    
    # Check each tool call
    matches = 0
    for expected, actual in zip(expected_tool_calls, actual_tool_calls):
        if (expected.get("name") == actual.get("name") and 
            expected.get("arguments") == actual.get("arguments")):
            matches += 1
    
    score = matches / len(expected_tool_calls) if expected_tool_calls else 0.0
    
    return EvaluatorResult(
        key="tool_calling_accuracy",
        score=score,
        comment=f"Matched {matches}/{len(expected_tool_calls)} tool calls correctly",
    )


def run_tool_calling_evaluation(
    dataset_name: str,
    target_function: callable,
    client: Optional[Client] = None,
) -> Any:
    """
    Run tool calling evaluation on a dataset.
    
    Args:
        dataset_name: Name of the dataset to evaluate
        target_function: Function to evaluate (should take inputs and return outputs)
        client: Optional LangSmith client
        
    Returns:
        Evaluation results
    """
    if client is None:
        client = Client()
    
    # Create evaluator using the existing utilities
    evaluator = lambda **kwargs: _run_evaluator(
        run_name="tool_calling_evaluation",
        scorer=tool_calling_evaluator,
        feedback_key="tool_calling_accuracy",
        **kwargs,
    )
    
    # Run evaluation
    return client.evaluate(
        target_function,
        data=dataset_name,
        evaluators=[evaluator],
        experiment_prefix="tool_calling_eval",
    )


# Example usage and configuration
if __name__ == "__main__":
    # Example 1: Creating a simple tool calling dataset
    def example_create_dataset():
        """Example of creating a tool calling dataset with sample data."""
        print("=== Example 1: Creating a Tool Calling Dataset ===")
        
        # Sample examples for the dataset
        examples = [
            {
                "inputs": {
                    "messages": [
                        {"role": "user", "content": "What's the weather like in San Francisco?"},
                        {
                            "role": "assistant",
                            "content": "I'll check the weather for you.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "San Francisco", "unit": "celsius"}'
                                    }
                                }
                            ]
                        }
                    ]
                },
                "expected_outputs": [
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "arguments": {"location": "San Francisco", "unit": "celsius"}
                    }
                ]
            },
            {
                "inputs": {
                    "messages": [
                        {"role": "user", "content": "Send an email to john@example.com with subject 'Meeting' and body 'Let's meet tomorrow'"},
                        {
                            "role": "assistant",
                            "content": "I'll send that email for you.",
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "send_email",
                                        "arguments": '{"to": "john@example.com", "subject": "Meeting", "body": "Let\'s meet tomorrow"}'
                                    }
                                }
                            ]
                        }
                    ]
                },
                "expected_outputs": [
                    {
                        "id": "call_2",
                        "name": "send_email",
                        "arguments": {
                            "to": "john@example.com",
                            "subject": "Meeting",
                            "body": "Let's meet tomorrow"
                        }
                    }
                ]
            },
            {
                "inputs": {
                    "messages": [
                        {"role": "user", "content": "Calculate 15 * 23 and then convert the result to binary"},
                        {
                            "role": "assistant",
                            "content": "I'll calculate that for you and convert to binary.",
                            "tool_calls": [
                                {
                                    "id": "call_3a",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": '{"expression": "15 * 23"}'
                                    }
                                },
                                {
                                    "id": "call_3b",
                                    "type": "function",
                                    "function": {
                                        "name": "convert_to_binary",
                                        "arguments": '{"number": 345}'
                                    }
                                }
                            ]
                        }
                    ]
                },
                "expected_outputs": [
                    {
                        "id": "call_3a",
                        "name": "calculate",
                        "arguments": {"expression": "15 * 23"}
                    },
                    {
                        "id": "call_3b",
                        "name": "convert_to_binary",
                        "arguments": {"number": 345}
                    }
                ]
            }
        ]
        
        try:
            # Create the dataset
            dataset_id = create_tool_calling_dataset(
                dataset_name="tool_calling_examples",
                examples=examples,
                description="Example dataset for testing tool calling functionality"
            )
            print(f"✅ Created dataset with ID: {dataset_id}")
            return dataset_id
        except Exception as e:
            print(f"❌ Error creating dataset: {e}")
            return None
    
    
    # Example 2: Processing messages to extract tool calls
    def example_process_messages():
        """Example of processing messages to extract tool calls."""
        print("
        
        # Sample messages with tool calls
        messages = [
            {"role": "user", "content": "Book a flight to NYC"},
            {
                "role": "assistant",
                "content": "I'll help you book a flight to NYC.",
                "tool_calls": [
                    {
                        "id": "flight_1",
                        "type": "function",
                        "function": {
                            "name": "book_flight",
                            "arguments": '{"destination": "NYC", "departure_date": "2024-01-15"}'
                        }
                    }
                ]
            }
        ]
        
        # Extract tool calls
        tool_calls = process_messages_to_tool_calls(messages)
        print(f"Extracted tool calls: {json.dumps(tool_calls, indent=2)}")
        
        return tool_calls
    
    
    # Example 3: Running evaluation with a mock target function
    def example_run_evaluation():
        """Example of running tool calling evaluation."""
        print("
        
        # Mock target function that simulates an AI system
        def mock_ai_system(inputs):
            """Mock AI system that returns tool calls based on input."""
            messages = inputs.get("messages", [])
            
            # Simple logic to return tool calls based on user input
            user_message = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "").lower()
                    break
            
            # Return mock responses based on keywords
            if "weather" in user_message:
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I'll check the weather for you.",
                            "tool_calls": [
                                {
                                    "id": "weather_call",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "San Francisco", "unit": "celsius"}'
                                    }
                                }
                            ]
                        }
                    ]
                }
            elif "email" in user_message:
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I'll send that email for you.",
                            "tool_calls": [
                                {
                                    "id": "email_call",
                                    "type": "function",
                                    "function": {
                                        "name": "send_email",
                                        "arguments": '{"to": "john@example.com", "subject": "Meeting", "body": "Let\'s meet tomorrow"}'
                                    }
                                }
                            ]
                        }
                    ]
                }
            else:
                return {"messages": [{"role": "assistant", "content": "I can help with that."}]}
        
        print("Mock AI system created. In a real scenario, you would:")
        print("1. Create a dataset using create_tool_calling_dataset()")
        print("2. Run evaluation using run_tool_calling_evaluation()")
        print("3. Example call: run_tool_calling_evaluation('your_dataset_name', your_ai_function)")
        
        # Demonstrate the evaluator directly
        sample_inputs = {"messages": [{"role": "user", "content": "What's the weather?"}]}
        sample_outputs = mock_ai_system(sample_inputs)
        sample_reference = {
            "expected_tool_calls": [
                {
                    "id": "weather_call",
                    "name": "get_weather",
                    "arguments": {"location": "San Francisco", "unit": "celsius"}
                }
            ]
        }
        
        result = tool_calling_evaluator(sample_inputs, sample_outputs, sample_reference)
        print(f"Evaluation result: {result}")
    
    
    # Example 4: Configuration options
    def example_configuration():
        """Example of different configuration options."""
        print("
        
        print("Configuration options for tool calling evaluation:")
        print("1. Dataset Creation:")
        print("   - dataset_name: Unique name for your dataset")
        print("   - examples: List of input/output pairs")
        print("   - description: Optional description")
        print("   - client: Optional LangSmith client (auto-created if not provided)")
        
        print("
        print("   - Supports multiple input formats:")
        print("     * ChatCompletionMessage (OpenAI format)")
        print("     * BaseMessage (LangChain format)")
        print("     * Dict format")
        
        print("
        print("   - Compares tool call names and arguments")
        print("   - Returns scores from 0.0 to 1.0")
        print("   - Provides detailed feedback in comments")
        
        print("
        print("   - Set LANGSMITH_API_KEY environment variable")
        print("   - Optionally set LANGSMITH_PROJECT for organization")
    
    
    # Run all examples
    print("🚀 Tool Calling Evaluation Script Examples")
    print("=" * 50)
    
    # Run examples
    dataset_id = example_create_dataset()
    example_process_messages()
    example_run_evaluation()
    example_configuration()
    
    print("
    print("✅ All examples completed!")
    print("
    print("1. Import the functions you need")
    print("2. Set up your LangSmith API key")
    print("3. Create your dataset with examples")
    print("4. Run evaluation with your AI system")

