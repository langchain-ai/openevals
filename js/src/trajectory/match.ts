import type { BaseMessage } from "@langchain/core/messages";
import {
  ChatCompletionMessage,
  FlexibleChatCompletionMessage,
  ToolArgsMatchMode,
  ToolArgsMatchOverrides,
} from "../types.js";
import { _normalizeToOpenAIMessagesList, _runEvaluator } from "../utils.js";
import { _scorer as trajectoryStrictScorer } from "./strict.js";
import { _scorer as trajectoryUnorderedScorer } from "./unordered.js";
import { _scorer as trajectorySubsetScorer } from "./subset.js";
import { _scorer as trajectorySuperset } from "./superset.js";

export type TrajectoryMatchMode = "strict" | "unordered" | "subset" | "superset";

type TrajectoryInput =
  | ChatCompletionMessage[]
  | FlexibleChatCompletionMessage[]
  | BaseMessage[]
  | {
      messages: (
        | BaseMessage
        | ChatCompletionMessage
        | FlexibleChatCompletionMessage
      )[];
    };

/**
 * Creates an evaluator that compares trajectories between model outputs and reference outputs.
 *
 * @param options.trajectoryMatchMode - The mode for matching trajectories: "strict", "unordered", "subset", or "superset"
 * @param options.toolArgsMatchMode - Mode for matching tool arguments. Defaults to "exact".
 * @param options.toolArgsMatchOverrides - Dict containing custom overrides for tool argument matching.
 * @returns An async function that evaluates trajectory matches between outputs and references
 */
export function createTrajectoryMatchEvaluator({
  trajectoryMatchMode = "strict",
  toolArgsMatchMode = "exact",
  toolArgsMatchOverrides,
}: {
  trajectoryMatchMode?: TrajectoryMatchMode;
  toolArgsMatchMode?: ToolArgsMatchMode;
  toolArgsMatchOverrides?: ToolArgsMatchOverrides;
} = {}) {
  let scorer:
    | typeof trajectoryStrictScorer
    | typeof trajectoryUnorderedScorer
    | typeof trajectorySubsetScorer
    | typeof trajectorySuperset;

  if (trajectoryMatchMode === "strict") {
    scorer = trajectoryStrictScorer;
  } else if (trajectoryMatchMode === "unordered") {
    scorer = trajectoryUnorderedScorer;
  } else if (trajectoryMatchMode === "subset") {
    scorer = trajectorySubsetScorer;
  } else if (trajectoryMatchMode === "superset") {
    scorer = trajectorySuperset;
  } else {
    throw new Error(
      `Invalid trajectory match type: \`${trajectoryMatchMode}\`. Must be one of \`strict\`, \`unordered\`, \`subset\`, or \`superset\`.`
    );
  }

  if (!["exact", "ignore", "subset", "superset"].includes(toolArgsMatchMode)) {
    throw new Error(
      `Invalid tool args match mode: \`${toolArgsMatchMode}\`. Must be either \`exact\`, \`ignore\`, \`subset\`, or \`superset\`.`
    );
  }

  return async ({
    outputs,
    referenceOutputs,
    ...kwargs
  }: {
    outputs: TrajectoryInput;
    referenceOutputs: TrajectoryInput;
    [key: string]: unknown;
  }) => {
    const normalizedOutputs = _normalizeToOpenAIMessagesList(outputs);
    const normalizedReferenceOutputs =
      _normalizeToOpenAIMessagesList(referenceOutputs);

    return _runEvaluator(
      `trajectory_${trajectoryMatchMode}_match`,
      scorer,
      `trajectory_${trajectoryMatchMode}_match`,
      {
        outputs: normalizedOutputs,
        referenceOutputs: normalizedReferenceOutputs,
        toolArgsMatchMode,
        toolArgsMatchOverrides,
        ...kwargs,
      }
    );
  };
}
