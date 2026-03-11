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

export const _scorer = async (params: {
  outputs: ChatCompletionMessage[];
  referenceOutputs: ChatCompletionMessage[];
  toolArgsMatchMode: ToolArgsMatchMode;
  toolArgsMatchOverrides?: ToolArgsMatchOverrides;
}): Promise<boolean> => {
  // superset: outputs contains all tool calls from referenceOutputs
  const isSuperset = await _isTrajectorySuperset(
    params.outputs,
    params.referenceOutputs,
    params.toolArgsMatchMode,
    params.toolArgsMatchOverrides
  );
  return isSuperset;
};

/**
 * @deprecated Use `createTrajectoryMatchEvaluator` with `trajectoryMatchMode: "superset"` instead.
 */
export async function trajectorySuperset(params: {
  outputs:
    | FlexibleChatCompletionMessage[]
    | BaseMessage[]
    | {
        messages: (
          | BaseMessage
          | ChatCompletionMessage
          | FlexibleChatCompletionMessage
        )[];
      };
  referenceOutputs:
    | FlexibleChatCompletionMessage[]
    | BaseMessage[]
    | {
        messages: (
          | BaseMessage
          | ChatCompletionMessage
          | FlexibleChatCompletionMessage
        )[];
      };
}): Promise<EvaluatorResult> {
  const { outputs, referenceOutputs } = params;
  const outputsList = _normalizeToOpenAIMessagesList(outputs);
  const referenceOutputsList = _normalizeToOpenAIMessagesList(referenceOutputs);

  return _runEvaluator("trajectory_superset", _scorer, "trajectory_superset", {
    ...params,
    outputs: outputsList,
    referenceOutputs: referenceOutputsList,
    toolArgsMatchMode: "ignore",
  });
}
