import { BaseMessage } from "@langchain/core/messages";
import { ChatCompletionMessage, EvaluatorResult } from "../types.js";
import { _normalizeToOpenAIMessagesList, _runEvaluator } from "../utils.js";
import { _isTrajectorySuperset } from "./utils.js";

/**
 * Evaluate whether an agent trajectory and called tools is a superset of a reference trajectory and called tools.
 * This means the agent called a superset of the tools specified in the reference trajectory.
 *
 * @param params - The parameters for trajectory superset evaluation
 * @param params.outputs - Actual trajectory the agent followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @param params.reference_outputs - Ideal reference trajectory the agent should have followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @returns EvaluatorResult containing a score of true if trajectory (including called tools) matches, false otherwise
 */
export async function trajectorySuperset(params: {
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
        "Trajectory superset match requires both outputs and reference_outputs"
      );
    }
    const isSuperset = _isTrajectorySuperset(outputsList, referenceOutputsList);
    return isSuperset;
  };
  return _runEvaluator(
    "trajectory_superset",
    getScore,
    "trajectory_superset",
    params
  );
}
