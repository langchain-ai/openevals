import type { BaseMessage, BaseMessageChunk } from "@langchain/core/messages";

export type EvaluatorResult = {
  key: string;
  score: number | boolean;
  comment?: string;
  metadata?: Record<string, unknown>;
};

export type SimpleEvaluator = (params: {
  inputs?: unknown;
  outputs: unknown;
  reference_outputs?: unknown;
  [key: string]: unknown;
}) =>
  | Promise<EvaluatorResult | EvaluatorResult[]>
  | EvaluatorResult
  | EvaluatorResult[];

export type SingleResultScorerReturnType =
  | boolean
  | number
  | [boolean | number, string, Record<string, unknown>?]
  | readonly [boolean | number, string, Record<string, unknown>?];

export type MultiResultScorerReturnType = {
  [key: string]:
    | boolean
    | number
    | { score: boolean | number; reasoning?: string };
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ChatCompletionMessage = Record<string, any> &
  (
    | {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        content: any;
        role: "user" | "system" | "developer";
        id?: string;
      }
    | {
        role: "assistant";
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        content: any;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        tool_calls?: any[];
        id?: string;
      }
    | {
        role: "tool";
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        content: any;
        tool_call_id: string;
        id?: string;
      }
  );

export type ChatCompletion = {
  choices: { message: ChatCompletionMessage }[];
};

// Few shot example type for evaluator prompts
export type FewShotExample = {
  inputs: unknown;
  outputs: unknown;
  score: number | boolean;
  reasoning?: string;
};

export interface ChatCompletionsClient {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  create(params: Record<string, any>): Promise<ChatCompletion>;
}

export interface ModelChatClient {
  completions: ChatCompletionsClient;
}

export interface ModelClient {
  chat: ModelChatClient;
}

export type Messages = ChatCompletionMessage | BaseMessage | BaseMessageChunk;

export type MultiturnSimulationResult = {
  evaluatorResults: EvaluatorResult[];
  trajectory: ChatCompletionMessage[];
};
