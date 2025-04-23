import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { initChatModel } from "langchain/chat_models/universal";

import { _convertToOpenAIMessage } from "../utils.js";
import type {
  MultiturnSimulatorTrajectory,
  MultiturnSimulatorTrajectoryUpdate,
  ChatCompletionMessage,
} from "../types.js";

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
 * This function generates a simulator that can be used with the createMultiturnSimulator to create
 * dynamic, LLM-powered user responses in a conversation. The simulator automatically handles message
 * role conversion to maintain proper conversation flow, where user messages become assistant messages
 * and vice versa when passed to the underlying LLM.
 *
 * @param {Object} params - The parameters for creating the simulated user
 * @param {string} params.system - System prompt that guides the LLM's behavior as a simulated user
 * @param {string} [params.model] - Optional name of the language model to use. Must be provided if client is not.
 * @param {BaseChatModel} [params.client] - Optional LangChain chat model instance. Must be provided if model is not.
 *
 * @returns A callable simulator function that takes a MultiturnSimulatorTrajectory containing conversation messages
 *          and returns a MultiturnSimulatorTrajectoryUpdate with the simulated user's response
 *
 * @example
 * ```typescript
 * import { createMultiturnSimulator, createLLMSimulatedUser } from "openevals";
 *
 * // Create a simulated user with GPT-4
 * const simulatedUser = createLLMSimulatedUser({
 *   system: "You are a helpful customer service representative",
 *   model: "openai:gpt-4o-mini"
 * });
 *
 * // Use with createMultiturnSimulator
 * const simulator = createMultiturnSimulator({
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
}: {
  system: string;
  model?: string;
  client?: BaseChatModel;
}): (
  inputs: MultiturnSimulatorTrajectory
) => Promise<MultiturnSimulatorTrajectoryUpdate> {
  if (!client && !model) {
    throw new Error("Either client or model must be provided");
  } else if (client && model) {
    throw new Error("Only one of client or model should be provided");
  }
  return async function _simulator(
    inputs: MultiturnSimulatorTrajectory
  ): Promise<MultiturnSimulatorTrajectoryUpdate> {
    if (!inputs || !inputs.messages || !Array.isArray(inputs.messages)) {
      throw new Error(
        "Simulated user inputs must be a dict with a 'messages' key containing a list of messages"
      );
    }
    if (model) {
      // eslint-disable-next-line no-param-reassign
      client = await initChatModel(model);
    }

    const messages = [];
    for (const msg of inputs.messages) {
      const convertedMessage = _convertToOpenAIMessage(msg);
      if (_isInternalMessage(convertedMessage)) {
        continue;
      }
      if (convertedMessage.role === "user") {
        convertedMessage.role = "assistant";
        messages.push(convertedMessage);
      } else if (
        convertedMessage.role === "assistant" &&
        !convertedMessage.tool_calls
      ) {
        convertedMessage.role = "user";
        messages.push(convertedMessage);
      }
    }

    if (system) {
      messages.unshift({ role: "system", content: system });
    }

    const response = await client!.invoke(messages);

    return {
      messages: [
        {
          role: "user",
          content: response.content,
          id: response.id,
        },
      ],
    };
  };
}
