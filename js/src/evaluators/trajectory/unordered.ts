import { BaseMessage } from "@langchain/core/messages";
import { ChatCompletionMessage, EvaluatorResult } from "../types.js";
import { _normalizeToOpenAIMessagesList, _runEvaluator } from "../utils.js";
import { _isTrajectorySuperset } from "./utils.js";

/**
 * Evaluate whether an input agent trajectory and called tools contains all the tools used in a reference trajectory.
 * This accounts for some differences in an LLM's reasoning process in a case-by-case basis.
 *
 * @param params - The parameters for trajectory unordered match evaluation
 * @param params.outputs - Actual trajectory the agent followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @param params.reference_outputs - Ideal reference trajectory the agent should have followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @returns EvaluatorResult containing a score of true if trajectory (including called tools) matches, false otherwise
 */
export async function trajectoryUnorderedMatch(params: {
  outputs:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
  referenceOutputs:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
}): Promise<EvaluatorResult> {
  const { outputs, referenceOutputs } = params;
  const outputsList = _normalizeToOpenAIMessagesList(outputs);
  const referenceOutputsList = _normalizeToOpenAIMessagesList(referenceOutputs);

  const getScore = async () => {
    if (outputsList == null || referenceOutputsList == null) {
      throw new Error(
        "Trajectory unordered match requires both outputs and reference_outputs"
      );
    }
    const unorderedMatch =
      _isTrajectorySuperset(outputsList, referenceOutputsList) &&
      _isTrajectorySuperset(referenceOutputsList, outputsList);
    return unorderedMatch;
  };
  return _runEvaluator(
    "trajectory_unordered_match",
    getScore,
    "trajectory_unordered_match",
    params
  );
}
