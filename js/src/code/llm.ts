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
