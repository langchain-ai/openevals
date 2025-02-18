import { BaseMessage, isBaseMessage } from "@langchain/core/messages";
import { _convertMessagesToOpenAIParams } from "@langchain/openai";
import {
  wrapEvaluator,
  isInTestContext,
  SimpleEvaluationResult,
} from "langsmith/utils/jestlike";
import {
  ChatCompletionMessage,
  MultiResultScorerReturnType,
  SingleResultScorerReturnType,
} from "./types.js";

export const _convertToOpenAIMessage = (
  message: BaseMessage | ChatCompletionMessage
): ChatCompletionMessage => {
  if (isBaseMessage(message)) {
    return _convertMessagesToOpenAIParams([message])[0] as any;
  } else {
    return message;
  }
};

export const _normalizeToOpenAIMessagesList = (
  messages:
    | (BaseMessage | ChatCompletionMessage)[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] }
): ChatCompletionMessage[] => {
  let messagesList: (BaseMessage | ChatCompletionMessage)[];
  if (!Array.isArray(messages)) {
    if ("messages" in messages && Array.isArray(messages.messages)) {
      messagesList = messages.messages;
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
  value: boolean | number | { score: boolean | number; reasoning?: string }
) => {
  if (typeof value === "object") {
    if (value != null && "score" in value) {
      return [
        value.score,
        "reasoning" in value && typeof value.reasoning === "string"
          ? value.reasoning
          : undefined,
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

export type EvaluationResultType<O> = O extends MultiResultScorerReturnType
  ? SimpleEvaluationResult[]
  : SimpleEvaluationResult;

export const _runEvaluator = async <
  T extends Record<string, unknown>,
  O extends
    | SingleResultScorerReturnType
    | MultiResultScorerReturnType
    | Promise<SingleResultScorerReturnType | MultiResultScorerReturnType>,
>(
  runName: string,
  scorer: (params: T) => O,
  feedbackKey: string,
  extra?: T
): Promise<EvaluationResultType<O>> => {
  const runScorer = async (params: T) => {
    let score = await scorer(params);
    let reasoning;

    const results = [];
    if (!Array.isArray(score) && typeof score === "object") {
      for (const [key, value] of Object.entries(score)) {
        const [keyScore, reasoning] = processScore(key, value);
        results.push({ key, score: keyScore, comment: reasoning });
      }
    } else {
      if (Array.isArray(score)) {
        reasoning = score[1];
        score = score[0] as Awaited<O>;
      }
      results.push({ key: feedbackKey, score, comment: reasoning });
    }
    if (results.length === 1) {
      return results[0] as SimpleEvaluationResult;
    } else {
      return results as SimpleEvaluationResult[];
    }
  };

  if (isInTestContext()) {
    const res = await wrapEvaluator(runScorer)(extra ?? ({} as T), {
      name: runName,
    });
    return res as EvaluationResultType<O>;
  } else {
    const res = await runScorer(extra ?? ({} as T));
    return res as EvaluationResultType<O>;
  }
};
