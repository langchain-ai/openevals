import { BaseMessage, isBaseMessage } from "@langchain/core/messages";
import * as openAIImports from "@langchain/openai";
import { wrapEvaluator, isInTestContext } from "langsmith/utils/jestlike";
import { traceable } from "langsmith/traceable";

import {
  ChatCompletionMessage,
  FlexibleChatCompletionMessage,
  MultiResultScorerReturnType,
  SingleResultScorerReturnType,
  EvaluatorResult,
} from "./types.js";

const {
  // @ts-expect-error Shim for older versions of @langchain/openai
  _convertMessagesToOpenAIParams,
  convertMessagesToCompletionsMessageParams,
} = openAIImports;

function _convertMessagesShim(message: BaseMessage) {
  if (typeof _convertMessagesToOpenAIParams === "function") {
    return _convertMessagesToOpenAIParams([
      message,
    ])[0] as ChatCompletionMessage;
  }
  return convertMessagesToCompletionsMessageParams({
    messages: [message],
  })[0] as ChatCompletionMessage;
}

export const _convertToOpenAIMessage = (
  message: BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage
): ChatCompletionMessage => {
  if (isBaseMessage(message)) {
    const converted = _convertMessagesShim(message);
    if (message.id && !converted.id) {
      converted.id = message.id;
    }
    return converted;
  } else {
    return message as ChatCompletionMessage;
  }
};

export const _normalizeToOpenAIMessagesList: (
  messages:
    | (BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage)[]
    | { messages: (BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage)[] }
    | (BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage)
) => ChatCompletionMessage[] = (
  messages:
    | (BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage)[]
    | { messages: (BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage)[] }
    | (BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage)
): ChatCompletionMessage[] => {
  let messagesList: (BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage)[];
  if (!Array.isArray(messages)) {
    if ("messages" in messages && Array.isArray(messages.messages)) {
      messagesList = messages.messages;
    } else if ("content" in messages && "role" in messages) {
      messagesList = [messages as ChatCompletionMessage];
    } else {
      throw new Error(
        `If passing messages as an object, it must contain a "messages" key`
      );
    }
  } else {
    messagesList = messages;
  }
  return messagesList.map(_convertToOpenAIMessage);
};

function _normalizeAttachmentMimeType(mimeType: string): string {
  const normalized = mimeType.toLowerCase().trim();
  if (normalized === "audio/mpeg") return "audio/mp3";
  if (normalized === "audio/wave" || normalized === "audio/x-wav") return "audio/wav";
  return normalized;
}

/**
 * Convert an attachment to a content block for multimodal messages.
 *
 * Attachments should be passed in the multimodal trace format described at
 * https://docs.langchain.com/langsmith/log-multimodal-traces:
 * `{ mime_type: "image/png", data: "data:image/png;base64,..." }`.
 *
 * Also accepts plain image URL strings or pre-formatted content block objects.
 *
 * Supported MIME types:
 * - `image/*`: `{ type: "image_url", image_url: { url: data } }`
 * - `application/pdf`: `{ type: "file", file: { filename, file_data: data } }`
 * - `audio/*`: `{ type: "input_audio", input_audio: { data: base64, format } }`
 */
export function _attachmentToContentBlock(
  item: string | Record<string, unknown>
): Record<string, unknown> {
  if (typeof item === "string") {
    return { type: "image_url", image_url: { url: item } };
  }
  if (typeof item !== "object" || item === null) {
    throw new Error(
      `Unsupported attachment type: ${typeof item}. Expected a string URL or an object with mime_type and data.`
    );
  }

  const { mime_type: rawMimeType, data } = item as { mime_type?: string; data?: string };

  if (rawMimeType === undefined || data === undefined) {
    if (!("type" in item)) {
      const msg =
        "Attachment dict must contain either 'mime_type' and 'data' keys, " +
        "or a 'type' key for pre-formatted content blocks.";
      throw new Error(msg);
    }
    return item;
  }

  const mimeType = _normalizeAttachmentMimeType(rawMimeType);

  if (mimeType.startsWith("image/")) {
    return { type: "image_url", image_url: { url: data } };
  }
  if (mimeType === "application/pdf") {
    const filename = (item.name as string | undefined) ?? "attachment.pdf";
    return { type: "file", file: { filename, file_data: data } };
  }
  if (mimeType.startsWith("audio/")) {
    const base64Data = data.startsWith("data:") ? data.split(",")[1] : data;
    const fmt = mimeType.split("/")[1];
    return { type: "input_audio", input_audio: { data: base64Data, format: fmt } };
  }
  const msg = `Unsupported attachment MIME type: ${mimeType}. Supported types: image/*, application/pdf, audio/*`;
  throw new Error(msg);
}

export const processScore = (
  _: string,
  value:
    | boolean
    | number
    | {
        score: boolean | number;
        reasoning?: string;
        metadata?: Record<string, unknown>;
        sourceRunId?: string;
      }
) => {
  if (typeof value === "object") {
    if (value != null && "score" in value) {
      return [
        value.score,
        "reasoning" in value && typeof value.reasoning === "string"
          ? value.reasoning
          : undefined,
        value.metadata,
        value.sourceRunId,
      ] as const;
    } else {
      throw new Error(
        `Expected a dictionary with a "score" key, but got "${JSON.stringify(
          value,
          null,
          2
        )}"`
      );
    }
  }
  return [value] as const;
};

export type EvaluationResultType<O> = O extends
  | MultiResultScorerReturnType
  | Promise<MultiResultScorerReturnType>
  ? EvaluatorResult[]
  : EvaluatorResult;

export async function _runEvaluator<
  T extends Record<string, unknown>,
  O extends
    | SingleResultScorerReturnType
    | MultiResultScorerReturnType
    | Promise<SingleResultScorerReturnType | MultiResultScorerReturnType>,
>(
  runName: string,
  scorer: (params: T) => O,
  feedbackKey: string,
  extra?: T,
  ls_framework?: string
): Promise<EvaluationResultType<O>> {
  return _runEvaluatorUntyped(
    runName,
    scorer,
    feedbackKey,
    extra,
    ls_framework,
    false
  ) as Promise<EvaluationResultType<O>>;
}

export async function _runEvaluatorUntyped<
  T extends Record<string, unknown>,
  O extends
    | SingleResultScorerReturnType
    | MultiResultScorerReturnType
    | Promise<SingleResultScorerReturnType | MultiResultScorerReturnType>,
>(
  runName: string,
  scorer: (params: T) => O,
  feedbackKey: string,
  extra?: T,
  ls_framework?: string,
  returnRawOutputs?: true
): Promise<Record<string, unknown>>;

export async function _runEvaluatorUntyped<
  T extends Record<string, unknown>,
  O extends
    | SingleResultScorerReturnType
    | MultiResultScorerReturnType
    | Promise<SingleResultScorerReturnType | MultiResultScorerReturnType>,
>(
  runName: string,
  scorer: (params: T) => O,
  feedbackKey: string,
  extra?: T,
  ls_framework?: string,
  returnRawOutputs?: false | undefined
): Promise<EvaluationResultType<O>>;

export async function _runEvaluatorUntyped<
  T extends Record<string, unknown>,
  O extends
    | SingleResultScorerReturnType
    | MultiResultScorerReturnType
    | Promise<SingleResultScorerReturnType | MultiResultScorerReturnType>,
>(
  runName: string,
  scorer: (params: T) => O,
  feedbackKey: string,
  extra?: T,
  ls_framework?: string,
  returnRawOutputs?: boolean
): Promise<Record<string, unknown> | EvaluationResultType<O>>;

export async function _runEvaluatorUntyped<
  T extends Record<string, unknown>,
  O extends
    | SingleResultScorerReturnType
    | MultiResultScorerReturnType
    | Promise<SingleResultScorerReturnType | MultiResultScorerReturnType>,
>(
  runName: string,
  scorer: (params: T) => O,
  feedbackKey: string,
  extra?: T,
  ls_framework?: string,
  returnRawOutputs?: boolean
): Promise<Record<string, unknown> | EvaluationResultType<O>> {
  const runScorer = async (params: T) => {
    let score = await scorer(params);

    if (returnRawOutputs) {
      return score;
    }

    let reasoning;

    if (!Array.isArray(score) && typeof score === "object") {
      const results = [];
      for (const [key, value] of Object.entries(score)) {
        const [keyScore, reasoning, metadata, sourceRunId] = processScore(
          key,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          value as any
        );
        const result: EvaluatorResult = {
          key,
          score: keyScore,
          comment: reasoning,
          metadata,
        };
        if (sourceRunId !== undefined && typeof sourceRunId === "string") {
          result.sourceRunId = sourceRunId;
        }
        results.push(result);
      }
      return results;
    } else {
      let metadata;
      if (Array.isArray(score)) {
        metadata = score[2];
        reasoning = score[1];
        score = score[0] as Awaited<O>;
      }
      return {
        key: feedbackKey,
        score,
        comment: reasoning,
        metadata,
      } as EvaluatorResult;
    }
  };

  if (isInTestContext()) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const res = await wrapEvaluator(runScorer as any)(extra ?? ({} as T), {
      name: runName,
      metadata: {
        __ls_framework: ls_framework ?? "openevals",
        __ls_evaluator: runName,
        __ls_language: "js",
      },
    });
    if (returnRawOutputs) {
      // TODO: Fix LangSmith SDK types
      const rawResults = res as Record<string, unknown>;
      return rawResults;
    }
    return res as EvaluationResultType<O>;
  } else {
    const traceableRunScorer = traceable(runScorer, {
      name: runName,
      metadata: {
        __ls_framework: ls_framework ?? "openevals",
        __ls_evaluator: runName,
        __ls_language: "js",
      },
    }) as (params: T) => Promise<EvaluationResultType<O>>;
    const res = await traceableRunScorer(extra ?? ({} as T));
    return res as EvaluationResultType<O>;
  }
}

export function _normalizeOutputsAsString(
  outputs: string | Record<string, unknown>
): string {
  if (typeof outputs === "string") {
    return outputs;
  } else if (outputs !== null && typeof outputs === "object") {
    if ("content" in outputs) {
      return outputs.content as string;
    } else if (
      "messages" in outputs &&
      Array.isArray(outputs.messages) &&
      outputs.messages.length > 0
    ) {
      return outputs.messages[outputs.messages.length - 1].content as string;
    } else {
      throw new Error(
        `Expected a string, dictionary with a 'content' key or a 'messages' key with a list of messages, but got ${JSON.stringify(
          outputs,
          null,
          2
        )}`
      );
    }
  } else {
    throw new Error(`Expected string or object, got ${typeof outputs}`);
  }
}
