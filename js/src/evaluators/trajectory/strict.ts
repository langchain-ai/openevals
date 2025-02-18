import { BaseMessage } from "@langchain/core/messages";
import { ChatCompletionMessage, EvaluatorResult } from "../types.js";
import { _normalizeToOpenAIMessagesList, _runEvaluator } from "../utils.js";

function _scorer(params: {
  outputs:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
  referenceOutputs:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
}): boolean {
  const { outputs, referenceOutputs } = params;
  const normalizedOutputs = _normalizeToOpenAIMessagesList(outputs);
  const normalizedReferenceOutputs =
    _normalizeToOpenAIMessagesList(referenceOutputs);

  if (!normalizedOutputs || !normalizedReferenceOutputs) {
    throw new Error(
      "Strict trajectory match requires both outputs and reference_outputs"
    );
  }

  if (normalizedOutputs.length !== normalizedReferenceOutputs.length) {
    return false;
  }

  let exactMatch = true;
  for (let i = 0; i < normalizedOutputs.length; i++) {
    const output = normalizedOutputs[i];
    const referenceOutput = normalizedReferenceOutputs[i];

    if (output.role !== referenceOutput.role) {
      exactMatch = false;
      break;
    }

    const outputHasToolCalls = output.tool_calls != null;
    const referenceHasToolCalls = referenceOutput.tool_calls != null;

    if (outputHasToolCalls !== referenceHasToolCalls) {
      exactMatch = false;
      break;
    }

    if (outputHasToolCalls) {
      if (output.tool_calls!.length !== referenceOutput.tool_calls!.length) {
        exactMatch = false;
        break;
      }

      for (let j = 0; j < output.tool_calls!.length; j++) {
        if (
          output.tool_calls![j].function.name !==
          referenceOutput.tool_calls![j].function.name
        ) {
          exactMatch = false;
          break;
        }
      }
    }
  }

  return exactMatch;
}

export async function trajectoryStrictMatch(params: {
  outputs:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
  referenceOutputs:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
}): Promise<EvaluatorResult> {
  /**
   * Evaluate whether an input agent trajectory and called tools strictly matches a reference trajectory.
   * This means that at each step, the agent called the same tools in the same order as specified in the reference trajectory.
   *
   * @param outputs - Actual trajectory the agent followed. May be a list of OpenAI messages,
   *                 a list of LangChain messages, or a dictionary containing a "messages" key with one of the above.
   * @param referenceOutputs - Ideal reference trajectory the agent should have followed. May be a list of OpenAI messages,
   *                          a list of LangChain messages, or a dictionary containing a "messages" key with one of the above.
   * @returns EvaluatorResult containing a score of true if trajectory (including called tools) matches, false otherwise
   */
  return _runEvaluator(
    "trajectory_strict_match",
    _scorer,
    "trajectory_strict_match",
    params
  );
}
