import pytest
from e2b_code_interpreter import Sandbox

from openevals.code.e2b.execution import create_e2b_execution_evaluator


@pytest.fixture
def sandbox():
    sandbox = Sandbox("OpenEvalsPython")
    yield sandbox


@pytest.mark.langsmith(output_keys=["outputs"])
@pytest.mark.parametrize(
    "inputs, outputs, expected_result",
    [
        (
            "Generate an app with a bad import.",
            """
from tyjfieowjiofewjiooiing import foo

foo()
""",
            False,
        ),
        (
            "Generate a bad LangGraph app.",
            """
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)
builder.add_node("start", lambda state: state)
builder.compile()

builder.invoke({})
""",
            False,
        ),
        (
            "Generate a good LangGraph app.",
            """
import json
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)
builder.add_node("start", lambda state: state)
builder.add_edge(START, "start")
graph = builder.compile()

graph.invoke({})
""",
            True,
        ),
    ],
)
def test_e2b_execution_evaluator(inputs, outputs, expected_result, sandbox):
    evaluator = create_e2b_execution_evaluator(
        sandbox=sandbox,
    )
    eval_result = evaluator(inputs=inputs, outputs=outputs)
    assert eval_result["score"] == expected_result


@pytest.mark.langsmith
def test_e2b_execution_evaluator_with_custom_env(sandbox):
    outputs = """
import unittest
from unittest.mock import patch, MagicMock
import os

class TestReactAgent(unittest.TestCase):
    def setUp(self):
        # Define the agent inline for testing
        from langgraph.prebuilt import create_react_agent

        def search(query: str):
            \"\"\"Call to surf the web.\"\"\"
            if "sf" in query.lower() or "san francisco" in query.lower():
                return "It's 60 degrees and foggy."
            return "It's 90 degrees and sunny."

        self.search_function = search
        # We'll patch the actual creation of the agent in each test
        # rather than creating it here, to allow for proper mocking

    @patch("langgraph.prebuilt.create_react_agent")
    def test_search_function_sf(self, mock_create_agent):
        \"\"\"Test that search function returns correct response for SF queries\"\"\"
        result = self.search_function("What's the weather in SF?")
        self.assertEqual(result, "It's 60 degrees and foggy.")

        result = self.search_function("Tell me about San Francisco weather")
        self.assertEqual(result, "It's 60 degrees and foggy.")

    @patch("langgraph.prebuilt.create_react_agent")
    def test_search_function_other_location(self, mock_create_agent):
        \"\"\"Test that search function returns default response for non-SF queries\"\"\"
        result = self.search_function("What's the weather in LA?")
        self.assertEqual(result, "It's 90 degrees and sunny.")

        result = self.search_function("Tell me about Chicago weather")
        self.assertEqual(result, "It's 90 degrees and sunny.")

    @patch("langgraph.prebuilt.create_react_agent")
    def test_agent_creation(self, mock_create_agent):
        \"\"\"Test that agent is created with the correct parameters\"\"\"
        # Create a mock agent instance
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        # Define the search function again to ensure it's in scope
        def search(query: str):
            \"\"\"Call to surf the web.\"\"\"
            if "sf" in query.lower() or "san francisco" in query.lower():
                return "It's 60 degrees and foggy."
            return "It's 90 degrees and sunny."

        # Create the agent
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent("anthropic:claude-3-7-sonnet-latest", tools=[search])

        # Verify that create_react_agent was called with the correct parameters
        mock_create_agent.assert_called_once_with(
            "anthropic:claude-3-7-sonnet-latest", 
            tools=[search]
        )

        # Verify that the returned agent is what we expect
        self.assertEqual(agent, mock_agent)

    @patch("langgraph.prebuilt.create_react_agent")
    def test_agent_invocation(self, mock_create_agent):
        \"\"\"Test that agent can be invoked and returns expected results\"\"\"
        # Create a mock agent that returns a predetermined response
        mock_agent = MagicMock()
        expected_response = {"output": "The weather in San Francisco is 60 degrees and foggy."}
        mock_agent.invoke.return_value = expected_response
        mock_create_agent.return_value = mock_agent

        # Define search function and create agent
        def search(query: str):
            \"\"\"Call to surf the web.\"\"\"
            if "sf" in query.lower() or "san francisco" in query.lower():
                return "It's 60 degrees and foggy."
            return "It's 90 degrees and sunny."

        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent("anthropic:claude-3-7-sonnet-latest", tools=[search])

        # Invoke the agent
        result = agent.invoke({"input": "What's the weather in San Francisco?"})

        # Verify the agent was called correctly
        mock_agent.invoke.assert_called_once_with({"input": "What's the weather in San Francisco?"})

        # Check the result
        self.assertEqual(result, expected_response)

    @patch("langgraph.prebuilt.create_react_agent")
    def test_integration_with_environment_variables(self, mock_create_agent):
        \"\"\"Test that the agent uses the environment variables correctly\"\"\"
        # This test would check if the anthropic API key is properly used
        # For a unit test, we'll verify that the environment variable exists
        # and that create_react_agent is called with the correct model string

        # Check if ANTHROPIC_API_KEY is set
        self.assertIn("ANTHROPIC_API_KEY", os.environ, 
                      "ANTHROPIC_API_KEY environment variable must be set")

        # Create a mock agent
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        # Define search function and create agent
        def search(query: str):
            \"\"\"Call to surf the web.\"\"\"
            if "sf" in query.lower() or "san francisco" in query.lower():
                return "It's 60 degrees and foggy."
            return "It's 90 degrees and sunny."

        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent("anthropic:claude-3-7-sonnet-latest", tools=[search])

        # Verify that create_react_agent was called with the correct model
        mock_create_agent.assert_called_once_with(
            "anthropic:claude-3-7-sonnet-latest", 
            tools=[search]
        )

if __name__ == "__main__":
    unittest.main()
"""

    evaluator = create_e2b_execution_evaluator(
        sandbox=sandbox,
        environment_variables={"ANTHROPIC_API_KEY": "foo"},
    )
    eval_result = evaluator(outputs=outputs)
    assert eval_result["score"] == True
