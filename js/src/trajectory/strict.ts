import { BaseMessage } from "@langchain/core/messages";
import {
  ChatCompletionMessage,
  FlexibleChatCompletionMessage,
  EvaluatorResult,
  ToolArgsMatchMode,
  ToolArgsMatchOverrides,
} from "../types.js";
import { _normalizeToOpenAIMessagesList, _runEvaluator } from "../utils.js";
import { _isTrajectorySuperset } from "./utils.js";

export async function _scorer(params: {
  outputs: ChatCompletionMessage[];
  referenceOutputs: ChatCompletionMessage[];
  toolArgsMatchMode: ToolArgsMatchMode;
  toolArgsMatchOverrides?: ToolArgsMatchOverrides;
}): Promise<boolean> {
  const {
    outputs,
    referenceOutputs,
    toolArgsMatchMode,
    toolArgsMatchOverrides,
  } = params;

  if (!outputs || !referenceOutputs) {
    throw new Error(
      "Strict trajectory match requires both outputs and reference_outputs"
    );
  }

  if (outputs.length !== referenceOutputs.length) {
    return false;
  }

  for (let i = 0; i < outputs.length; i++) {
    const output = outputs[i];
    const referenceOutput = referenceOutputs[i];

    if (output.role !== referenceOutput.role) {
      return false;
    }

    const outputHasToolCalls = output.tool_calls != null;
    const referenceHasToolCalls = referenceOutput.tool_calls != null;

    if (outputHasToolCalls !== referenceHasToolCalls) {
      return false;
    }

    if (outputHasToolCalls) {
      if (output.tool_calls!.length !== referenceOutput.tool_calls!.length) {
        return false;
      }
      // Same number of calls, so a superset is a one-to-one match
      if (
        !(await _isTrajectorySuperset(
          [output],
          [referenceOutput],
          toolArgsMatchMode,
          toolArgsMatchOverrides
        ))
      ) {
        return false;
      }
    }
  }

  return true;
}

/**
 * @deprecated Use `createTrajectoryMatchEvaluator` with `trajectoryMatchMode: "strict"` instead.
 */
export async function trajectoryStrictMatch(params: {
  outputs:
    | ChatCompletionMessage[]
    | FlexibleChatCompletionMessage[]
    | BaseMessage[]
    | {
        messages: (
          BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage
        )[];
      };
  referenceOutputs:
    | ChatCompletionMessage[]
    | FlexibleChatCompletionMessage[]
    | BaseMessage[]
    | {
        messages: (
          BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage
        )[];
      };
  toolCallArgsExactMatch: boolean;
}): Promise<EvaluatorResult> {
  const normalizedOutputs = _normalizeToOpenAIMessagesList(params.outputs);
  const normalizedReferenceOutputs = _normalizeToOpenAIMessagesList(
    params.referenceOutputs
  );

  return _runEvaluator(
    "trajectory_strict_match",
    _scorer,
    "trajectory_strict_match",
    {
      outputs: normalizedOutputs,
      referenceOutputs: normalizedReferenceOutputs,
      toolArgsMatchMode: params.toolCallArgsExactMatch ? "exact" : "ignore",
    }
  );
}
