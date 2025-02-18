import {
  ChatCompletionMessage,
  EvaluatorResult,
  FewShotExample,
  ModelClient,
} from "../types.js";
import { BaseMessage } from "@langchain/core/messages";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { RunnableInterface } from "@langchain/core/runnables";

import { _createLLMAsJudgeScorer } from "../llm.js";
import { _runEvaluator } from "../utils.js";
import { _normalizeToOpenAIMessagesList } from "../utils.js";
import { _chatCompletionMessagesToString } from "./utils.js";

export const DEFAULT_PROMPT = `Grade the following agent trajectory:

<trajectory>
{outputs}
</trajectory>
{inputs}
{reference_outputs}
{rubric}
`;

function _formatInputs(params: {
  inputs?: Record<string, any>;
  outputs:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
  referenceOutputs?:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
  rubric?: string;
}): [string, string, string, string] {
  const { inputs, outputs, referenceOutputs, rubric } = params;
  const normalizedOutputs = _normalizeToOpenAIMessagesList(outputs);
  const normalizedReferenceOutputs = _normalizeToOpenAIMessagesList(
    referenceOutputs ?? []
  );

  const formattedReferenceOutputs = normalizedReferenceOutputs
    ? `\nUse the following trajectory as an example reference when grading:\n<reference_trajectory>\n${_chatCompletionMessagesToString(normalizedReferenceOutputs)}\n</reference_trajectory>\n`
    : "";

  const formattedInputs = inputs
    ? `\nThe agent generated the trajectory from the following input:\n<input>\n${JSON.stringify(inputs)}\n</input>\n`
    : "";

  const formattedOutputs =
    typeof outputs === "object" && !Array.isArray(outputs)
      ? outputs
      : _chatCompletionMessagesToString(normalizedOutputs);

  const formattedRubric = rubric
    ? `\nGrade the agent trajectory along the following rubric:\n<rubric>\n${rubric}\n</rubric>\n`
    : "";

  return [
    formattedOutputs as string,
    formattedReferenceOutputs,
    formattedInputs,
    formattedRubric,
  ];
}

/**
 * Creates an evaluator that uses an LLM to judge agent trajectories.
 *
 * @param options - Configuration options
 * @param options.prompt - The evaluation prompt. Can be a string template, LangChain prompt template,
 *                        or callable that returns a list of chat messages. Note that the default prompt
 *                        allows a rubric in addition to the typical "inputs", "outputs", and
 *                        "reference_outputs" parameters.
 * @param options.feedbackKey - Key used to store the evaluation result. Defaults to "trajectory_accuracy".
 * @param options.model - Model identifier to use. Defaults to "openai:o3-mini". If judge is an OpenAI client,
 *                       this should be a model name directly. If judge is omitted, must be a valid
 *                       LangChain model identifier.
 * @param options.system - Optional system message to prepend to the prompt.
 * @param options.judge - The LLM used for evaluation. Can be an OpenAI client or a LangChainLikeModel.
 *                       If an OpenAI client, must specify "model" as well. If omitted, "model" will be
 *                       used to instantiate a LangChain model instance by model string.
 * @param options.continuous - If true, score will be a float between 0 and 1. If false, score will be boolean.
 *                           Defaults to false.
 * @param options.choices - Optional list of specific float values the score must be chosen from.
 * @param options.useReasoning - If true, includes explanation for the score in the output. Defaults to true.
 * @param options.fewShotExamples - Optional list of example evaluations to append to the prompt.
 * @returns A function that evaluates agent trajectories using the configured LLM judge.
 */
export const createTrajectoryLLMAsJudge = ({
  prompt = DEFAULT_PROMPT,
  feedbackKey = "trajectory_accuracy",
  model = "openai:o3-mini",
  system,
  judge,
  continuous = false,
  choices,
  useReasoning = true,
  fewShotExamples,
}: {
  prompt?:
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

  const wrappedEvaluator = async ({
    inputs,
    outputs,
    referenceOutputs,
    rubric,
    ...extra
  }: {
    inputs?: Record<string, any>;
    outputs:
      | ChatCompletionMessage[]
      | BaseMessage[]
      | { messages: (BaseMessage | ChatCompletionMessage)[] };
    referenceOutputs?:
      | ChatCompletionMessage[]
      | BaseMessage[]
      | { messages: (BaseMessage | ChatCompletionMessage)[] };
    rubric?: string;
    [key: string]: unknown;
  }): Promise<EvaluatorResult> => {
    const [
      formattedOutputs,
      formattedReferenceOutputs,
      formattedInputs,
      formattedRubric,
    ] = _formatInputs({ inputs, outputs, referenceOutputs, rubric });

    return _runEvaluator(`llm_as_${feedbackKey}_judge`, scorer, feedbackKey, {
      outputs: formattedOutputs,
      referenceOutputs: formattedReferenceOutputs,
      inputs: formattedInputs,
      rubric: formattedRubric,
      ...extra,
    });
  };
  return wrappedEvaluator;
};
