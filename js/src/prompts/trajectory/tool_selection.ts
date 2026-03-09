export const TOOL_SELECTION_PROMPT = `You are an expert data labeler. Your task is to grade the accuracy of an AI agent's tool selection during the resolution of a user query.

<Rubric>
Accurate tool selection:
- Uses the most appropriate tool for each step given the context
- Avoids unnecessary or redundant tool calls
- Uses tools in a logical order where dependencies exist
- Is semantically equivalent to the provided reference tool sequence, if present
</Rubric>

<Instructions>
1. Grade the following thread, evaluating whether the agent selected the right tools in the right order to resolve the user's query efficiently.
2. Evaluate both the choice of tools and whether any tools were unnecessary, missing, or could have been replaced with a more appropriate alternative
</Instructions>

Please grade the following trajectory according to the above instructions:

<trajectory>
{outputs}
</trajectory>
`;
