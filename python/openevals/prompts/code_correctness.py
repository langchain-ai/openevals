CODE_CORRECTNESS_PROMPT = """You are an expert code reviewer evaluating code for correctness. Your task is to assign a score based on the following rubric:

<Rubric>
  A correct code solution:
  - Solves the problem completely as specified in the input
  - Should contain only valid code without any additional text
  - Handles all edge cases appropriately
  - Contains absolutely no bugs or logical errors
  - Uses efficient and appropriate algorithms/data structures
  - Follows language-specific best practices
  - Has correct syntax and would compile/run without errors

  When scoring, you should penalize:
  - Logical errors or bugs that would cause incorrect behavior
  - Missing edge case handling
  - Overly inefficient implementations when better approaches exist
  - Incomplete solutions that don't address all requirements
  - Syntax errors that would prevent compilation/execution
  - Security vulnerabilities or unsafe practices
  - Additional text that is not code
</Rubric>

<Instructions>
  - Carefully analyze both the output code and the initial input query
  - Meticulously check for functional correctness and completeness
  - Focus on whether the code would work correctly rather than style preferences
</Instructions>

<Reminder>
  The goal is to evaluate whether the code correctly solves the given problem.
</Reminder>

<input>
{inputs}
</input>

<output>
{outputs}
</output>
"""

CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS = """You are an expert code reviewer evaluating code for correctness. Your task is to assign a score based on the following rubric:

<Rubric>
  A correct code solution:
  - Solves the problem completely as specified in the input
  - Should contain only valid code without any additional text
  - Handles all edge cases appropriately
  - Contains absolutely no bugs or logical errors
  - Uses efficient and appropriate algorithms/data structures
  - Follows language-specific best practices
  - Has correct syntax and would compile/run without errors

  When scoring, you should penalize:
  - Logical errors or bugs that would cause incorrect behavior
  - Missing edge case handling
  - Overly inefficient implementations when better approaches exist
  - Incomplete solutions that don't address all requirements
  - Syntax errors that would prevent compilation/execution
  - Security vulnerabilities or unsafe practices
  - Additional text that is not code
</Rubric>

<Instructions>
  - Carefully analyze both the output code and the initial input query
  - Meticulously check for functional correctness and completeness
  - Focus on whether the code would work correctly rather than style preferences
  - Compare the output with the reference output to verify correctness
  - The reference output represents the expected behavior or result
  - Code that produces results matching the reference output should be scored higher
  - Consider edge cases where the code might produce correct results for the given examples but fail in other scenarios
</Instructions>

<Reminder>
  The goal is to evaluate whether the code correctly solves the given problem and produces output that matches the reference.
</Reminder>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

<reference_output>
{reference_outputs}
</reference_output>
"""
