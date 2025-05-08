import { v4 as uuidv4 } from "uuid";
import { traceable } from "langsmith/traceable";
import type { BaseMessage } from "@langchain/core/messages";

import {
  ChatCompletionMessage,
  EvaluatorResult,
  SimpleEvaluator,
  Messages,
  MultiturnSimulatorResult,
} from "../types.js";
import {
  _convertToOpenAIMessage,
  _normalizeToOpenAIMessagesList,
} from "../utils.js";

export { MultiturnSimulatorResult };

type MultiturnSimulatorTrajectory = Record<string, unknown> & {
  trajectory: ChatCompletionMessage[];
};

function _wrap<T extends Record<string, unknown>>(
  app: (
    params: T
  ) =>
    | ChatCompletionMessage
    | BaseMessage
    | Promise<ChatCompletionMessage | BaseMessage>,
  runName: string,
  threadId: string
) {
  const wrapper = (params: T) => {
    return app({ ...params, threadId });
  };
  return traceable(wrapper, { name: runName });
}

function _coerceAndAssignIdToMessage(
  message: ChatCompletionMessage | BaseMessage
): ChatCompletionMessage {
  const convertedMessage = _convertToOpenAIMessage(message);
  if (convertedMessage.id === undefined) {
    return {
      ...convertedMessage,
      id: uuidv4(),
    };
  }
  return convertedMessage;
}

function _trajectoryReducer(
  currentTrajectory: MultiturnSimulatorTrajectory | null,
  newUpdate: ChatCompletionMessage,
  updateSource: "app" | "user",
  turnCounter: number
): MultiturnSimulatorTrajectory {
  function _combineMessages(
    left: Messages[] | Messages,
    right: Messages[] | Messages
  ): ChatCompletionMessage[] {
    // Coerce to list
    if (!Array.isArray(left)) {
      // eslint-disable-next-line no-param-reassign
      left = [left];
    }
    if (!Array.isArray(right)) {
      // eslint-disable-next-line no-param-reassign
      right = [right];
    }

    // Coerce to message
    const coercedLeft: ChatCompletionMessage[] = left.map((msg) =>
      _coerceAndAssignIdToMessage(msg)
    );

    const coercedRight: ChatCompletionMessage[] = right.map((msg) =>
      _coerceAndAssignIdToMessage(msg)
    );

    // Merge
    const merged = [...coercedLeft];
    const mergedById: Record<string, number> = {};
    merged.forEach((m, i) => {
      if (m.id) {
        mergedById[m.id] = i;
      }
    });

    for (const m of coercedRight) {
      if (m.id && mergedById[m.id] === undefined) {
        mergedById[m.id] = merged.length;
        merged.push(m);
      }
    }

    return merged;
  }

  if (currentTrajectory == null) {
    // eslint-disable-next-line no-param-reassign
    currentTrajectory = { trajectory: [] };
  }

  let coercedNewUpdate;
  try {
    coercedNewUpdate = _normalizeToOpenAIMessagesList(newUpdate);
  } catch {
    throw new Error(
      `Received unexpected trajectory update from '${updateSource}': ${JSON.stringify(newUpdate)}. Expected a message, list of messages, or dictionary with a 'messages' key containing messages.`
    );
  }
  return {
    trajectory: _combineMessages(
      currentTrajectory?.trajectory,
      coercedNewUpdate
    ),
    turnCounter,
  };
}

function _createStaticSimulatedUser(
  staticResponses: (string | Messages)[]
): (params: {
  trajectory: ChatCompletionMessage[];
  turnCounter: number;
  threadId: string;
}) => ChatCompletionMessage {
  return function _returnNextMessage(params: {
    trajectory: ChatCompletionMessage[];
    turnCounter: number;
    threadId: string;
  }): ChatCompletionMessage {
    const turns = params.turnCounter;
    if (turns === undefined || typeof turns !== "number") {
      throw new Error(
        "Internal error: Turn counter must be an integer in the trajectory."
      );
    }

    // First conversation turn is satisfied by the initial input
    if (turns >= staticResponses.length) {
      throw new Error(
        "Number of conversation turns is greater than the number of static user responses. Please reduce the number of turns or provide more responses."
      );
    }

    const nextResponse = staticResponses[turns];
    if (typeof nextResponse === "string") {
      return { role: "user", content: nextResponse, id: uuidv4() };
    }
    return _coerceAndAssignIdToMessage(nextResponse);
  };
}

/**
 * Run a simulation for multi-turn conversations between an application and a simulated user.
 *
 * This function runs a simulation between an app and
 * either a dynamic user simulator or a list of static user responses. The simulation supports
 * evaluation of conversation trajectories and customizable stopping conditions.
 *
 * Conversation trajectories are represented as a dict containing a key named "messages" whose
 * value is a list of message objects with "role" and "content" keys. The "app" param you provide
 * will receives the next message in sequence as an input, and should return a message. The simulator
 * will dedupe these messages by id and merge them into the complete trajectory.
 *
 * Once "maxTurns" is reached or a provided stopping condition is met, the final trajectory
 * will be passed to provided trajectory evaluators, which will receive the final trajectory
 * as an "outputs" param.
 *
 * When calling the created simulator, you may also provide a "referenceOutputs" param,
 * which will be passed directly through to the provided evaluators.
 *
 * @param {Object} params - Configuration parameters for the simulator
 * @param {(params: {inputs: ChatCompletionMessage, threadId: string}) => ChatCompletionMessage | Promise<ChatCompletionMessage>} params.app - Your application. Can be either a LangChain Runnable or a
 *        callable that takes the current conversation trajectory and returns
 *        a message.
 * @param {(params: {trajectory: ChatCompletionMessage[], turnCounter: number, threadId: string}) => ChatCompletionMessage | Promise<ChatCompletionMessage> | (string | Messages)[]} params.user - The simulated user. Can be:
 *        - A function that takes the current conversation trajectory and returns a message.
 *        - A list of strings or Messages representing static user responses
 * @param {number} [params.maxTurns] - Maximum number of conversation turns to simulate
 * @param {SimpleEvaluator[]} [params.trajectoryEvaluators] - Optional list of evaluator functions that assess the conversation
 *        trajectory. Each evaluator will receive the final trajectory of the conversation as
 *        a param named "outputs" and a param named "referenceOutputs" if provided.
 * @param {(params: {trajectory: ChatCompletionMessage[], turnCounter: number, threadId: string}) => boolean | Promise<boolean>} [params.stoppingCondition] - Optional callable that determines if the simulation should end early.
 *        Takes the current trajectory as input and returns a boolean.
 * @param {unknown} [params.referenceOutputs] - Optional reference outputs for evaluation
 * @param {string} [params.threadId] - Optional thread ID. If not provided, a random one will be generated.
 *
 * @returns Returns a Promise that resolves to a MultiturnSimulatorResult containing:
 *     - evaluator_results: List of results from trajectory evaluators
 *     - trajectory: The complete conversation trajectory
 *
 * @example
 * ```typescript
 * import { runMultiturnSimulation } from "openevals";
 *
 * // Create a simulator with static user responses
 * const result = runMultiturnSimulation({
 *   app: myChatApp,
 *   user: ["Hello!", "How are you?", "Goodbye"],
 *   maxTurns: 3,
 *   trajectoryEvaluators: [myEvaluator]
 * });
 * ```
 */
export const runMultiturnSimulation = traceable(
  (params: {
    app: (params: {
      inputs: ChatCompletionMessage;
      threadId: string;
    }) =>
      | ChatCompletionMessage
      | BaseMessage
      | Promise<ChatCompletionMessage | BaseMessage>;
    user:
      | ((params: {
          trajectory: ChatCompletionMessage[];
          turnCounter: number;
        }) =>
          | ChatCompletionMessage
          | BaseMessage
          | Promise<ChatCompletionMessage | BaseMessage>)
      | (string | Messages)[];
    maxTurns?: number;
    trajectoryEvaluators?: SimpleEvaluator[];
    stoppingCondition?: (params: {
      trajectory: ChatCompletionMessage[];
      turnCounter: number;
      threadId: string;
    }) => boolean | Promise<boolean>;
    referenceOutputs?: unknown;
    threadId?: string;
  }) => {
    const simulator = _createMultiturnSimulator(params);
    return simulator({
      referenceOutputs: params.referenceOutputs,
      threadId: params.threadId ?? uuidv4(),
    });
  },
  { name: "multiturn_simulator" }
);

export function _createMultiturnSimulator({
  app,
  user,
  maxTurns,
  trajectoryEvaluators = [],
  stoppingCondition,
}: {
  app: (params: {
    inputs: ChatCompletionMessage;
    threadId: string;
  }) =>
    | ChatCompletionMessage
    | BaseMessage
    | Promise<ChatCompletionMessage | BaseMessage>;
  user:
    | ((params: {
        trajectory: ChatCompletionMessage[];
        turnCounter: number;
      }) =>
        | ChatCompletionMessage
        | BaseMessage
        | Promise<ChatCompletionMessage | BaseMessage>)
    | (string | Messages)[];
  maxTurns?: number;
  trajectoryEvaluators?: SimpleEvaluator[];
  stoppingCondition?: (params: {
    trajectory: ChatCompletionMessage[];
    turnCounter: number;
    threadId: string;
  }) => boolean | Promise<boolean>;
}): (params: {
  referenceOutputs?: unknown;
  threadId: string;
  [key: string]: unknown;
}) => Promise<MultiturnSimulatorResult> {
  if (maxTurns === undefined && stoppingCondition === undefined) {
    throw new Error(
      "At least one of maxTurns or stoppingCondition must be provided."
    );
  }

  const _runSimulator = async ({
    referenceOutputs,
    threadId,
  }: {
    referenceOutputs?: unknown;
    threadId: string;
    [key: string]: unknown;
  }): Promise<MultiturnSimulatorResult> => {
    let turnCounter = 0;
    let currentReducedTrajectory: MultiturnSimulatorTrajectory = {
      trajectory: [],
      turnCounter: 0,
    };
    const wrappedApp = _wrap(app, "app", threadId);

    let wrappedSimulatedUser;
    if (Array.isArray(user)) {
      const staticResponses = user;
      const simulatedUser = _createStaticSimulatedUser(staticResponses);
      wrappedSimulatedUser = _wrap(simulatedUser, "simulated_user", threadId);
    } else {
      wrappedSimulatedUser = _wrap(user, "simulated_user", threadId);
    }

    // eslint-disable-next-line no-constant-condition
    while (true) {
      if (maxTurns !== undefined && turnCounter >= maxTurns) {
        break;
      }
      const rawInputs = await wrappedSimulatedUser({
        trajectory: currentReducedTrajectory.trajectory,
        turnCounter,
        threadId,
      });

      const currentInputs = _coerceAndAssignIdToMessage(rawInputs);

      currentReducedTrajectory = _trajectoryReducer(
        currentReducedTrajectory,
        currentInputs,
        "user",
        turnCounter
      );

      const rawOutputs = await wrappedApp({
        inputs: currentInputs,
        threadId,
      });

      const currentOutputs = _coerceAndAssignIdToMessage(rawOutputs);

      turnCounter += 1;

      currentReducedTrajectory = _trajectoryReducer(
        currentReducedTrajectory,
        currentOutputs,
        "app",
        turnCounter
      );

      if (
        stoppingCondition !== undefined &&
        (await stoppingCondition({
          trajectory: currentReducedTrajectory.trajectory,
          turnCounter,
          threadId,
        }))
      ) {
        break;
      }
    }

    const results: EvaluatorResult[] = [];
    delete currentReducedTrajectory.turnCounter;

    for (const trajectoryEvaluator of trajectoryEvaluators || []) {
      try {
        const trajectoryEvalResults = await trajectoryEvaluator({
          outputs: currentReducedTrajectory.trajectory,
          referenceOutputs,
        });

        if (Array.isArray(trajectoryEvalResults)) {
          results.push(...trajectoryEvalResults);
        } else {
          results.push(trajectoryEvalResults);
        }
      } catch (e) {
        console.error(`Error in trajectory evaluator: ${e}`);
      }
    }

    return {
      trajectory: currentReducedTrajectory.trajectory,
      evaluatorResults: results,
    };
  };

  return _runSimulator;
}
