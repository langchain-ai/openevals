import { RunnableInterface, Runnable } from "@langchain/core/runnables";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { BaseMessage } from "@langchain/core/messages";
import { initChatModel } from "langchain/chat_models/universal";
import { traceable } from "langsmith/traceable";

import { _runEvaluator, _normalizeToOpenAIMessagesList } from "./utils.js";
import {
  ChatCompletionMessage,
  FewShotExample,
  ModelClient,
  SingleResultScorerReturnType,
} from "./types.js";

function _isRunnableInterface(prompt: unknown): prompt is RunnableInterface {
  return Runnable.isRunnable(prompt);
}

function _isBaseChatModel(x: unknown): x is BaseChatModel {
  const model = x as BaseChatModel;
  return (
    x != null &&
    typeof x === "object" &&
    typeof model._modelType === "function" &&
    model._modelType() === "base_chat_model"
  );
}

function appendFewShotExamples({
  messages,
  fewShotExamples,
}: {
  messages: ChatCompletionMessage[];
  fewShotExamples: FewShotExample[];
}): ChatCompletionMessage[] {
  // Find the last user message to append examples to
  const lastUserMessageIdx = messages
    .slice()
    .reverse()
    .findIndex((msg) => msg.role === "user");

  if (lastUserMessageIdx === -1) {
    throw new Error(
      "Appending few-shot examples requires a user message in the provided prompt"
    );
  }

  const actualIdx = messages.length - 1 - lastUserMessageIdx;

  // eslint-disable-next-line no-param-reassign
  messages[actualIdx].content +=
    "\n\n" +
    fewShotExamples
      .map((example) => {
        let exampleStr = `<example>\n<input>${JSON.stringify(
          example.inputs
        )}</input>\n<output>${JSON.stringify(example.outputs)}</output>`;

        if (example.reasoning) {
          exampleStr += `\n<reasoning>${example.reasoning}</reasoning>`;
        }

        if (example.score !== undefined) {
          exampleStr += `\n<score>${example.score}</score>`;
        }

        exampleStr += "\n</example>";
        return exampleStr;
      })
      .join("\n");

  return messages;
}

function constructOutputSchema({
  schema,
  continuous,
  choices,
  useReasoning,
}: {
  schema?: Record<string, unknown>;
  continuous?: boolean;
  choices?: number[];
  useReasoning?: boolean;
}): [Record<string, unknown>, string] {
  const jsonSchema: Record<string, unknown> = schema ?? {
    type: "object",
    additionalProperties: false,
    strict: true,
  };

  let description: string;
  let scoreSchema: {
    type: string;
    description: string;
    enum?: number[];
  };

  if (choices) {
    description =
      "A number that represents the degree to which the criteria in the prompt are met.";
    scoreSchema = {
      type: "number",
      description,
      enum: choices,
    };
  } else if (continuous) {
    description =
      "A number that represents the degree to which the criteria in the prompt are met, from 0.0 to 1.0. 1.0 means the criteria are met perfectly. 0.0 means none of the criteria are met.";
    scoreSchema = {
      type: "number",
      description,
    };
  } else {
    description =
      "A score that is true if criteria in the prompt are met, and false otherwise.";
    scoreSchema = {
      type: "boolean",
      description,
    };
  }

  if (!schema) {
    if (useReasoning) {
      jsonSchema.properties = {
        reasoning: {
          type: "string",
          description:
            "A human-readable explanation of the score. You MUST end the reasoning with a sentence that says: Thus, the score should be: SCORE_YOU_ASSIGN.",
        },
        score: scoreSchema,
      };
      jsonSchema.required = ["reasoning", "score"];
    } else {
      jsonSchema.properties = {
        score: scoreSchema,
      };
      jsonSchema.required = ["score"];
    }
  }

  return [jsonSchema, description];
}

export const _createLLMAsJudgeScorer = (params: {
  prompt:
    | string
    | RunnableInterface
    | ((
        ...args: unknown[]
      ) => ChatCompletionMessage[] | Promise<ChatCompletionMessage[]>);
  system?: string;
  schema?: Record<string, unknown>;
  judge?: ModelClient | BaseChatModel;
  model?: string;
  continuous?: boolean;
  choices?: number[];
  useReasoning?: boolean;
  fewShotExamples?: FewShotExample[];
}) => {
  const {
    prompt,
    system,
    schema,
    model,
    continuous,
    choices,
    fewShotExamples,
  } = params;

  let judge = params.judge;
  const useReasoning = params.useReasoning ?? true;

  const getScore = async (params: {
    inputs: unknown;
    outputs: unknown;
    referenceOutputs?: unknown;
    [key: string]: unknown;
  }): Promise<SingleResultScorerReturnType> => {
    const { inputs, outputs, referenceOutputs, ...rest } = params;

    if (system && typeof prompt !== "string") {
      throw new Error(
        "`system` is only supported when `prompt` is a string template"
      );
    }

    let stringifiedInputs = inputs;
    let stringifiedOutputs = outputs;
    let stringifiedReferenceOutputs = referenceOutputs;
    if (inputs && typeof inputs !== "string") {
      stringifiedInputs = JSON.stringify(inputs);
    }
    if (outputs && typeof outputs !== "string") {
      stringifiedOutputs = JSON.stringify(outputs);
    }
    if (referenceOutputs && typeof referenceOutputs !== "string") {
      stringifiedReferenceOutputs = JSON.stringify(referenceOutputs);
    }
    const stringifiedRest = Object.fromEntries(
      Object.entries(rest).map(([key, value]) => [
        key,
        typeof value === "string" ? value : JSON.stringify(value),
      ])
    );

    let messages: (ChatCompletionMessage | BaseMessage)[] = [];

    if (_isRunnableInterface(prompt)) {
      const formattedPrompt = await prompt.invoke({
        inputs: stringifiedInputs,
        outputs: stringifiedOutputs,
        reference_outputs: stringifiedReferenceOutputs,
        ...stringifiedRest,
      });
      messages = formattedPrompt.messages;
    } else if (typeof prompt === "string") {
      const template = ChatPromptTemplate.fromTemplate(prompt);
      const formattedPrompt = await template.invoke({
        inputs: stringifiedInputs,
        outputs: stringifiedOutputs,
        reference_outputs: stringifiedReferenceOutputs,
        ...stringifiedRest,
      });
      messages = formattedPrompt.messages;
    } else {
      messages = await prompt({
        inputs,
        outputs,
        reference_outputs: referenceOutputs,
        ...rest,
      });
    }

    if (system) {
      messages = [{ role: "system", content: system }, ...messages];
    }

    let normalizedMessages = _normalizeToOpenAIMessagesList(messages);

    if (fewShotExamples) {
      normalizedMessages = appendFewShotExamples({
        messages: normalizedMessages,
        fewShotExamples,
      });
    }

    const [jsonSchema, description] = constructOutputSchema({
      schema,
      continuous,
      choices,
      useReasoning,
    });

    if (!judge) {
      judge = await initChatModel(model);
    }

    let response;
    if (_isBaseChatModel(judge)) {
      response = await judge
        .withStructuredOutput({
          title: "score",
          description,
          ...jsonSchema,
        })
        .invoke(normalizedMessages);
      if (schema === undefined) {
        if (useReasoning) {
          return [response.score, response.reasoning];
        }
        return response.score;
      } else {
        return response as SingleResultScorerReturnType;
      }
    } else {
      if (!model) {
        throw new Error("`model` is required for non-LangChain clients");
      }

      const params: Record<string, unknown> = {
        messages: normalizedMessages,
        model,
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "score",
            strict: true,
            schema: jsonSchema,
          },
        },
      };

      const invokeLlm = traceable(
        judge.chat.completions.create.bind(judge.chat.completions),
        {
          metadata: {
            ls_provider: "openai",
            ls_model_name: model,
            ls_model_type: "chat",
          },
          run_type: "llm",
          name: "OpenAI Chat Completion",
        }
      );
      const response = await invokeLlm(params);
      const parsed = JSON.parse(response.choices[0].message.content as string);
      if (schema === undefined) {
        if (useReasoning) {
          return [parsed.score, parsed.reasoning];
        }
        return parsed.score;
      }
      return parsed;
    }
  };
  return getScore;
};

/**
 * Create an evaluator that uses an LLM to assess output quality based on specified criteria.
 *
 * @param params Configuration object with the following properties:
 * @param params.prompt The evaluation prompt - can be a string template, LangChain prompt template,
 *                     or function that returns a list of chat messages
 * @param params.feedbackKey Key used to store the evaluation result, defaults to "score"
 * @param params.judge The LLM used for evaluation. Can be an OpenAI client or a LangChain model.
 *                     If using OpenAI client, must specify "model" parameter.
 *                     If omitted, "model" will be used to instantiate a LangChain model instance.
 * @param params.model Model identifier to use. Defaults to "openai:o3-mini".
 *                     If "judge" is an OpenAI client, this should be a model name directly.
 *                     If "judge" is omitted, must be a valid LangChain model identifier.
 * @param params.system Optional system message to prepend to the prompt
 * @param params.continuous If true, score will be a float between 0 and 1.
 *                         If false, score will be boolean. Defaults to false.
 * @param params.choices Optional list of specific float values the score must be chosen from
 * @param params.useReasoning If true, includes explanation for the score in the output.
 *                           Defaults to true.
 * @param params.fewShotExamples Optional list of example evaluations to append to the prompt
 *
 * @returns A function that takes inputs, outputs, reference_outputs, and other kwargs,
 *          formats them into a prompt, invokes the judge, and returns an evaluation result
 *
 * @example
 * ```typescript
 * import { createLLMAsJudge } from "openevals";
 *
 * const evaluator = createLLMAsJudge({
 *   prompt: "Rate the quality of this response from 0 to 1: {outputs}",
 *   continuous: true,
 * });
 * const result = await evaluator({
 *   inputs: { question: "What color is the sky?" },
 *   outputs: { response: "Blue" },
 * });
 * ```
 */
export const createLLMAsJudge = ({
  prompt,
  feedbackKey = "score",
  model = "openai:o3-mini",
  system,
  judge,
  continuous = false,
  choices,
  useReasoning = true,
  fewShotExamples,
}: {
  prompt:
    | string
    | RunnableInterface
    | ((
        ...args: unknown[]
      ) => ChatCompletionMessage[] | Promise<ChatCompletionMessage[]>);
  feedbackKey?: string;
  model?: string;
  system?: string;
  judge?: ModelClient | BaseChatModel;
  continuous?: boolean;
  choices?: number[];
  useReasoning?: boolean;
  fewShotExamples?: FewShotExample[];
}) => {
  const scorer = _createLLMAsJudgeScorer({
    prompt,
    judge,
    model,
    system,
    continuous,
    choices,
    useReasoning,
    fewShotExamples,
  });

  const _wrappedEvaluator = async (inputs: {
    inputs: unknown;
    outputs: unknown;
    referenceOutputs?: unknown;
    [key: string]: unknown;
  }) => {
    const runName =
      feedbackKey !== "score" ? "llm_as_judge" : `llm_as_${feedbackKey}_judge`;
    return _runEvaluator(runName, scorer, feedbackKey, inputs);
  };
  return _wrappedEvaluator;
};
