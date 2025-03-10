import { BaseChatModel } from "@langchain/core/language_models/chat_models";

import { _createLLMAsJudgeScorer, createLLMAsJudge } from "../llm.js";
import { _createBaseCodeEvaluator } from "./base.js";
import { SingleResultScorerReturnType } from "../types.js";

export type CodeLLMAsJudgeConfig = Omit<
  Parameters<typeof createLLMAsJudge>[0],
  "prompt"
> & {
  prompt: string;
  codeExtractionStrategy?: "none" | "llm" | "markdown_code_blocks";
  codeExtractor?: (outputs: string | Record<string, unknown>) => string;
  model?: string;
  client?: BaseChatModel;
};

/**
 * Creates a code evaluator that uses an LLM as a judge to assess code quality or correctness.
 *
 * Accepts all the same arguments as the `createLLMAsJudge` function, but adds an extraction step
 * to extract code from the output dependent on the `codeExtractionStrategy`/`codeExtractor` argument.
 *
 * @param config - Configuration for the code LLM judge
 * @param config.prompt - The prompt template to use for evaluation
 * @param config.codeExtractionStrategy - Strategy for extracting code from outputs: 'none', 'llm', or 'markdown_code_blocks' (optional)
 * @param config.codeExtractor - Custom function to extract code from outputs (optional)
 * @param config.model - The model name to use
 * @param config.client - A BaseChatModel instance to use instead of creating a new one (optional)
 * @param config.feedbackKey - Key to use for feedback in the evaluation results (defaults to 'code_correctness')
 * @returns A code evaluator function that can assess code using the specified LLM
 */
export const createCodeLLMAsJudge = (config: CodeLLMAsJudgeConfig) => {
  const scorer = _createLLMAsJudgeScorer(config);
  return _createBaseCodeEvaluator<SingleResultScorerReturnType>({
    model: config.model,
    client: config.client,
    runName: "code_llm_as_judge",
    feedbackKey: config.feedbackKey ?? "code_correctness",
    scorer,
    codeExtractionStrategy: config.codeExtractionStrategy,
    codeExtractor: config.codeExtractor,
  });
};
