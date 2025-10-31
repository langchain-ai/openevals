import { BaseMessage, isBaseMessage } from "@langchain/core/messages";
import { _convertMessagesToOpenAIParams } from "@langchain/openai";
import { wrapEvaluator, isInTestContext } from "langsmith/utils/jestlike";
import { traceable } from "langsmith/traceable";

import {
  ChatCompletionMessage,
  MultiResultScorerReturnType,
  SingleResultScorerReturnType,
  EvaluatorResult,
} from "./types.js";

export const _convertToOpenAIMessage = (
  message: BaseMessage | ChatCompletionMessage
): ChatCompletionMessage => {
  if (isBaseMessage(message)) {
    const converted = _convertMessagesToOpenAIParams([
      message,
    ])[0] as ChatCompletionMessage;
    if (message.id && !converted.id) {
      converted.id = message.id;
    }
    return converted;
  } else {
    return message;
  }
};

export const _normalizeToOpenAIMessagesList = (
  messages:
    | (BaseMessage | ChatCompletionMessage)[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] }
    | (BaseMessage | ChatCompletionMessage)
): ChatCompletionMessage[] => {
  let messagesList: (BaseMessage | ChatCompletionMessage)[];
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
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const [keyScore, reasoning, metadata, sourceRunId] = processScore(
          key,
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
