import type { BaseMessage } from "@langchain/core/messages";
import { _createLLMAsJudgeScorer } from "../llm.js";

import { _runEvaluator, _normalizeToOpenAIMessagesList } from "../utils.js";
import { _chatCompletionMessagesToString } from "./utils.js";
import {
  ChatCompletionMessage,
  FlexibleChatCompletionMessage,
  EvaluatorResult,
} from "../types.js";
import {
  TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
} from "../prompts/trajectory.js";

export { TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE } from "../prompts/trajectory.js";
export { TRAJECTORY_ACCURACY_PROMPT } from "../prompts/trajectory.js";

// Re-export createLLMAsJudge param types for convenience
type TrajectoryLLMAsJudgeParams = Parameters<
  typeof _createLLMAsJudgeScorer
>[0] & {
  feedbackKey?: string;
};

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

type TrajectoryEvaluatorFunction = (params: {
  outputs: TrajectoryInput;
  referenceOutputs?: TrajectoryInput;
  [key: string]: unknown;
}) => Promise<EvaluatorResult>;

function _formatInputs(params: {
  outputs: TrajectoryInput;
  referenceOutputs?: TrajectoryInput;
}): [string, string] {
  const { outputs, referenceOutputs } = params;
  const normalizedOutputs = _normalizeToOpenAIMessagesList(outputs);
  const normalizedReferenceOutputs = _normalizeToOpenAIMessagesList(
    referenceOutputs ?? []
  );

  const formattedReferenceOutputs = normalizedReferenceOutputs?.length
    ? _chatCompletionMessagesToString(normalizedReferenceOutputs)
    : "";

  const formattedOutputs = _chatCompletionMessagesToString(normalizedOutputs);

  return [formattedOutputs, formattedReferenceOutputs];
}

/**
 * Creates an evaluator that uses an LLM to judge agent trajectories.
 *
 * @param options.prompt - The evaluation prompt. Defaults to TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE.
 * @param options.feedbackKey - Key used to store the evaluation result. Defaults to "trajectory_accuracy".
 * @param options.model - Model identifier to use.
 * @param options.judge - The LLM used for evaluation.
 * @param options.continuous - If true, score will be a float between 0 and 1. Defaults to false.
 * @param options.choices - Optional list of specific float values the score must be chosen from.
 * @param options.useReasoning - If true, includes explanation for the score. Defaults to true.
 * @param options.fewShotExamples - Optional list of example evaluations.
 * @returns A function that evaluates agent trajectories using the configured LLM judge.
 */
export const createTrajectoryLLMAsJudge = ({
  prompt = TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
  feedbackKey = "trajectory_accuracy",
  ...rest
}: TrajectoryLLMAsJudgeParams = {}): TrajectoryEvaluatorFunction => {
  const scorer = _createLLMAsJudgeScorer({
    prompt,
    ...rest,
  });

  const wrappedEvaluator = async ({
    inputs,
    outputs,
    referenceOutputs,
    ...extra
  }: {
    outputs: TrajectoryInput;
    referenceOutputs?: TrajectoryInput;
    [key: string]: unknown;
  }): Promise<EvaluatorResult> => {
    const [formattedOutputs, formattedReferenceOutputs] = _formatInputs({
      outputs,
      referenceOutputs,
    });

    return _runEvaluator(`llm_as_${feedbackKey}_judge`, scorer, feedbackKey, {
      inputs,
      outputs: formattedOutputs,
      referenceOutputs: formattedReferenceOutputs,
      ...extra,
    });
  };
  return wrappedEvaluator;
};
