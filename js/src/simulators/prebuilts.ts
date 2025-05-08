import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { initChatModel } from "langchain/chat_models/universal";
import { v4 as uuidv4 } from "uuid";
import { _convertToOpenAIMessage } from "../utils.js";
import type { ChatCompletionMessage } from "../types.js";

// Exported for testing only
export function _isInternalMessage(message: ChatCompletionMessage): boolean {
  return Boolean(
    message.role !== "user" &&
      (message.role !== "assistant" || (message.tool_calls ?? []).length > 0)
  );
}

/**
 * Creates a simulated user powered by a language model for multi-turn conversations.
 *
 * This function generates a simulator that can be used with the runMultiturnSimulation method to create
 * dynamic, LLM-powered user responses in a conversation. The simulator automatically handles message
 * role conversion to maintain proper conversation flow, where user messages become assistant messages
 * and vice versa when passed to the underlying LLM.
 *
 * @param {Object} params - The parameters for creating the simulated user
 * @param {string} params.system - System prompt that guides the LLM's behavior as a simulated user
 * @param {string} [params.model] - Optional name of the language model to use. Must be provided if client is not.
 * @param {BaseChatModel} [params.client] - Optional LangChain chat model instance. Must be provided if model is not.
 * @param {(string | ChatCompletionMessage)[]} [params.fixedResponses] - Optional list of fixed responses to use for the simulated user.
 *
 * @returns A callable simulator function that takes a MultiturnSimulatorTrajectory containing conversation messages
 *          and returns a MultiturnSimulatorTrajectoryUpdate with the simulated user's response
 *
 * @example
 * ```typescript
 * import { runMultiturnSimulation, createLLMSimulatedUser } from "openevals";
 *
 * // Create a simulated user with GPT-4.1-mini
 * const simulatedUser = createLLMSimulatedUser({
 *   system: "You are a helpful customer service representative",
 *   model: "openai:gpt-4.1-mini"
 * });
 *
 * // Use with runMultiturnSimulation
 * const simulator = runMultiturnSimulation({
 *   app: myChatApp,
 *   user: simulatedUser,
 *   maxTurns: 5
 * });
 * ```
 *
 * Notes:
 * - The simulator automatically converts message roles to maintain proper conversation flow:
 *   * User messages become assistant messages when sent to the LLM
 *   * Assistant messages (without tool calls) become user messages when sent to the LLM
 * - The system prompt is prepended to each conversation to maintain consistent behavior
 * - The simulator returns responses in the format expected by createMultiturnSimulator
 */
export function createLLMSimulatedUser({
  system,
  model,
  client,
  fixedResponses,
}: {
  system: string;
  model?: string;
  client?: BaseChatModel;
  fixedResponses?: (string | ChatCompletionMessage)[];
}): (params: {
  trajectory: ChatCompletionMessage[];
  turnCounter: number;
}) => Promise<ChatCompletionMessage> {
  if (!client && !model) {
    throw new Error("Either client or model must be provided");
  } else if (client && model) {
    throw new Error("Only one of client or model should be provided");
  }
  return async function _simulator(params: {
    trajectory: ChatCompletionMessage[];
    turnCounter: number;
  }): Promise<ChatCompletionMessage> {
    if (model) {
      // eslint-disable-next-line no-param-reassign
      client = await initChatModel(model);
    }
    if (fixedResponses && params.turnCounter < fixedResponses.length) {
      const res = fixedResponses[params.turnCounter];
      if (typeof res === "string") {
        return {
          role: "user",
          content: res,
          id: uuidv4(),
        };
      } else {
        return res;
      }
    }

    const messages = [];
    for (const msg of params.trajectory) {
      const convertedMessage = _convertToOpenAIMessage(msg);
      if (_isInternalMessage(convertedMessage)) {
        continue;
      }
      if (convertedMessage.role === "user") {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (convertedMessage.role as any) = "assistant";
        messages.push(convertedMessage);
      } else if (
        convertedMessage.role === "assistant" &&
        !convertedMessage.tool_calls
      ) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (convertedMessage.role as any) = "user";
        messages.push(convertedMessage);
      }
    }

    if (messages.length === 0) {
      messages.push({
        role: "user",
        content:
          "Generate an initial query to start a conversation based on your instructions. Do not respond with other text.",
        id: uuidv4(),
      });
    }

    if (system) {
      messages.unshift({ role: "system", content: system });
    }

    const response = await client!.invoke(messages);

    return {
      role: "user",
      content: response.content,
      id: response.id,
    };
  };
}
