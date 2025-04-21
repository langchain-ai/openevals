import { v4 as uuidv4 } from "uuid";
import {
  Runnable,
  type RunnableConfig,
  RunnableLambda,
} from "@langchain/core/runnables";
import { traceable } from "langsmith/traceable";

import {
  ChatCompletionMessage,
  EvaluatorResult,
  SimpleEvaluator,
  MultiturnSimulatorTrajectory,
  MultiturnSimulatorTrajectoryUpdate,
  Messages,
  MultiturnSimulatorResult,
} from "../types.js";
import { convertToOpenAIMessage } from "../utils.js";

export {
  MultiturnSimulatorTrajectory,
  MultiturnSimulatorTrajectoryUpdate,
  MultiturnSimulatorResult,
};

function _wrap(
  app:
    | Runnable<MultiturnSimulatorTrajectory, MultiturnSimulatorTrajectoryUpdate>
    | ((
        trajectory: MultiturnSimulatorTrajectory
      ) =>
        | MultiturnSimulatorTrajectoryUpdate
        | Promise<MultiturnSimulatorTrajectoryUpdate>),
  runName: string
) {
  if (Runnable.isRunnable(app)) {
    return app;
  } else {
    return RunnableLambda.from(app).withConfig({ runName });
  }
}

function _isInternalMessage(message: ChatCompletionMessage): boolean {
  return Boolean(
    message.role !== "user" &&
      (message.role !== "assistant" || (message.tool_calls ?? []).length > 0)
  );
}

function _trajectoryReducer(
  currentTrajectory: MultiturnSimulatorTrajectory | null,
  newUpdate: MultiturnSimulatorTrajectoryUpdate,
  updateSource: "app" | "user",
  turnCounter: number
): MultiturnSimulatorTrajectory {
  function _combineMessages(
    left: Messages[] | Messages,
    right: Messages[] | Messages
  ): Messages[] {
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
    const coercedLeft: ChatCompletionMessage[] = left
      .map((msg) => convertToOpenAIMessage(msg))
      .filter((m) => !_isInternalMessage(m));

    const coercedRight: ChatCompletionMessage[] = right
      .map((msg) => convertToOpenAIMessage(msg))
      .filter((m) => !_isInternalMessage(m));

    // Assign missing ids
    for (const m of coercedLeft) {
      if (!m.id) {
        m.id = uuidv4();
      }
    }
    for (const m of coercedRight) {
      if (!m.id) {
        m.id = uuidv4();
      }
    }

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
    currentTrajectory = { messages: [] };
  }

  if (typeof newUpdate === "object" && newUpdate.messages) {
    return {
      ...currentTrajectory,
      ...newUpdate,
      messages: _combineMessages(
        currentTrajectory.messages,
        newUpdate.messages
      ),
      turnCounter,
    };
  } else {
    throw new Error(
      `Received unexpected trajectory update from ${updateSource}: ${JSON.stringify(newUpdate)}. Expected a dictionary with a 'messages' key.`
    );
  }
}

function _createStaticSimulatedUser(
  staticResponses: (string | Messages)[]
): (
  trajectory: MultiturnSimulatorTrajectory
) => MultiturnSimulatorTrajectoryUpdate {
  return function _returnNextMessage(
    trajectory: MultiturnSimulatorTrajectory
  ): MultiturnSimulatorTrajectoryUpdate {
    const turns = trajectory.turnCounter;
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
      return { messages: { role: "user", content: nextResponse } };
    }
    return { messages: nextResponse };
  };
}

/**
 * Creates a simulator for multi-turn conversations between an application and a simulated user.
 *
 * This function generates a simulator that can run conversations between an app and
 * either a dynamic user simulator or a list of static user responses. The simulator supports
 * evaluation of conversation trajectories and customizable stopping conditions.
 *
 * Conversation trajectories are represented as a dict containing a key named "messages" whose
 * value is a list of message objects with "role" and "content" keys. The "app" and "user"
 * params you provide will both receive this trajectory as an input, and should return a
 * trajectory update dict with a new message or new messages under the "messages" key. The simulator
 * will dedupe these messages by id and merge them into the complete trajectory.
 *
 * Additional fields are also permitted as part of the trajectory dict, which allows you to pass
 * additional information between the app and user if needed.
 *
 * Once "maxTurns" is reached or a provided stopping condition is met, the final trajectory
 * will be passed to provided trajectory evaluators, which will receive the final trajectory
 * as an "outputs" param.
 *
 * When calling the created simulator, you may also provide a "referenceOutputs" param,
 * which will be passed directly through to the provided evaluators.
 *
 * @param {Object} params - Configuration parameters for the simulator
 * @param {Runnable<MultiturnSimulatorTrajectory, MultiturnSimulatorTrajectoryUpdate> | ((trajectory: MultiturnSimulatorTrajectory) => MultiturnSimulatorTrajectoryUpdate | Promise<MultiturnSimulatorTrajectoryUpdate>)} params.app - Your application. Can be either a LangChain Runnable or a
 *        callable that takes the current conversation trajectory dict and returns
 *        a trajectory update dict with new messages under the "messages" key (and optionally other fields).
 * @param {Runnable<MultiturnSimulatorTrajectory, MultiturnSimulatorTrajectoryUpdate> | ((trajectory: MultiturnSimulatorTrajectory) => MultiturnSimulatorTrajectoryUpdate | Promise<MultiturnSimulatorTrajectoryUpdate>) | (string | Messages)[]} params.user - The simulated user. Can be:
 *        - A LangChain Runnable or a callable that takes the current conversation trajectory
 *          and returns a trajectory update dict with new messages under the "messages" key (and optionally other fields).
 *        - A list of strings or Messages representing static user responses
 * @param {number} [params.maxTurns] - Maximum number of conversation turns to simulate
 * @param {SimpleEvaluator[]} [params.trajectoryEvaluators] - Optional list of evaluator functions that assess the conversation
 *        trajectory. Each evaluator will receive the final trajectory of the conversation as
 *        a param named "outputs" and a param named "referenceOutputs" if provided.
 * @param {(trajectory: MultiturnSimulatorTrajectory) => boolean | Promise<boolean>} [params.stoppingCondition] - Optional callable that determines if the simulation should end early.
 *        Takes the current trajectory as input and returns a boolean.
 *
 * @returns A function that runs the simulation when invoked. The function accepts the following params:
 *          - initialTrajectory: Initial input to start the conversation
 *          - referenceOutputs: Optional reference outputs for evaluation
 *          - runnableConfig: Optional config that will be passed in if using LangChain Runnable components.
 *          Returns a Promise that resolves to a MultiturnSimulatorResult containing:
 *          - evaluator_results: List of results from trajectory evaluators
 *          - trajectory: The complete conversation trajectory
 *
 * @example
 * ```typescript
 * import { createMultiturnSimulator } from "openevals";
 *
 * // Create a simulator with static user responses
 * const simulator = createMultiturnSimulator({
 *   app: myChatApp,
 *   user: ["Hello!", "How are you?", "Goodbye"],
 *   maxTurns: 3,
 *   trajectoryEvaluators: [myEvaluator]
 * });
 *
 * // Run the simulation
 * const result = await simulator({
 *   initialTrajectory: {messages: [{role: "user", content: "Start"}]}
 * });
 * ```
 */
export function createMultiturnSimulator({
  app,
  user,
  maxTurns,
  trajectoryEvaluators = [],
  stoppingCondition,
}: {
  app:
    | Runnable<MultiturnSimulatorTrajectory, MultiturnSimulatorTrajectoryUpdate>
    | ((
        trajectory: MultiturnSimulatorTrajectory
      ) =>
        | MultiturnSimulatorTrajectoryUpdate
        | Promise<MultiturnSimulatorTrajectoryUpdate>);
  user:
    | Runnable<MultiturnSimulatorTrajectory, MultiturnSimulatorTrajectoryUpdate>
    | ((
        trajectory: MultiturnSimulatorTrajectory
      ) =>
        | MultiturnSimulatorTrajectoryUpdate
        | Promise<MultiturnSimulatorTrajectoryUpdate>)
    | (string | Messages)[];
  maxTurns?: number;
  trajectoryEvaluators?: SimpleEvaluator[];
  stoppingCondition?: (
    trajectory: MultiturnSimulatorTrajectory
  ) => boolean | Promise<boolean>;
}): (params: {
  initialTrajectory: MultiturnSimulatorTrajectory;
  referenceOutputs?: unknown;
  runnableConfig?: RunnableConfig;
  [key: string]: unknown;
}) => Promise<MultiturnSimulatorResult> {
  if (maxTurns === undefined && stoppingCondition === undefined) {
    throw new Error(
      "At least one of maxTurns or stoppingCondition must be provided."
    );
  }

  const _runSimulator = traceable(
    async ({
      initialTrajectory,
      referenceOutputs = undefined,
      runnableConfig = undefined,
    }: {
      initialTrajectory: MultiturnSimulatorTrajectory;
      referenceOutputs?: unknown;
      runnableConfig?: RunnableConfig;
      [key: string]: unknown;
    }): Promise<MultiturnSimulatorResult> => {
      let turnCounter = 0;
      let currentReducedTrajectory: MultiturnSimulatorTrajectory = {
        messages: [],
      };
      const wrappedApp = _wrap(app, "app");

      let wrappedSimulatedUser;
      if (Array.isArray(user)) {
        const staticResponses = user;
        const simulatedUser = _createStaticSimulatedUser(staticResponses);
        wrappedSimulatedUser = _wrap(simulatedUser, "simulated_user");
      } else {
        wrappedSimulatedUser = _wrap(user, "simulated_user");
      }

      // eslint-disable-next-line no-constant-condition
      while (true) {
        if (maxTurns !== undefined && turnCounter >= maxTurns) {
          break;
        }

        const currentInputs =
          turnCounter === 0
            ? initialTrajectory
            : await wrappedSimulatedUser.invoke(
                currentReducedTrajectory,
                runnableConfig
              );

        currentReducedTrajectory = _trajectoryReducer(
          currentReducedTrajectory,
          currentInputs,
          "user",
          turnCounter
        );

        const currentOutputs = await wrappedApp.invoke(
          currentReducedTrajectory,
          runnableConfig
        );

        currentReducedTrajectory = _trajectoryReducer(
          currentReducedTrajectory,
          currentOutputs,
          "app",
          turnCounter
        );

        turnCounter += 1;

        if (
          stoppingCondition !== undefined &&
          stoppingCondition(currentReducedTrajectory)
        ) {
          break;
        }
      }

      const results: EvaluatorResult[] = [];
      delete currentReducedTrajectory.turnCounter;

      for (const trajectoryEvaluator of trajectoryEvaluators || []) {
        try {
          const trajectoryEvalResults = await trajectoryEvaluator({
            outputs: currentReducedTrajectory,
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
        trajectory: currentReducedTrajectory,
        evaluatorResults: results,
      };
    },
    { name: "multiturn_simulator" }
  );

  return _runSimulator;
}
