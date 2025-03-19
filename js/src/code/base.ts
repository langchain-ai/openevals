import { initChatModel } from "langchain/chat_models/universal";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

import { _normalizeOutputsAsString, _runEvaluator } from "../utils.js";
import {
  MultiResultScorerReturnType,
  SingleResultScorerReturnType,
} from "../types.js";

export const LLM_EXTRACTION_SYSTEM_PROMPT = `
You are an expert software auditor.

<Instructions>
  Your job is to extract code from a given text.

  - If there is code - extract it into a single script by calling the provided "ExtractCode" tool.
  - If there is no code to extract - call "NoCode".

  If you extract code, your response will be passed DIRECTLY into a code execution sandbox for further testing,
  so make sure to extract all code **without modifications**, even if it contains errors,
  since any modifications will ruin the integrity of the testing process.
  Omit installation instructions from extracted code.
</Instructions>
`;

export const LLM_EXTRACTION_USER_PROMPT = `
Extract code from the following:

<text>
{outputs}
</text>
`;

/**
 * Extract code from markdown code blocks in the provided text.
 *
 * Supports both triple backtick code blocks with or without language specifiers.
 *
 * @param text - The text containing markdown code blocks
 * @returns A string containing only the code extracted from code blocks, with blocks
 *          separated by newlines
 */
function _extractCodeFromMarkdownCodeBlocks(text: string): string {
  // Pattern to match code blocks with or without language specifier
  // (?s) enables dot to match newlines
  // (?:```(?:\w+)?\n(.*?)```) matches code blocks with optional language specifier
  const pattern = /```(?:\w+)?\n([\s\S]*?)```/g;

  // Find all code blocks
  const codeBlocks: string[] = [];

  let match = pattern.exec(text);
  while (match !== null) {
    codeBlocks.push(match[1]);
    match = pattern.exec(text);
  }

  if (codeBlocks.length === 0) {
    return text; // Return original text if no code blocks found
  }

  // Join all code blocks with newlines
  return codeBlocks.join("\n");
}

export function _createBaseCodeEvaluator<
  T extends SingleResultScorerReturnType | MultiResultScorerReturnType,
>(config: {
  scorer: (params: { inputs: unknown; outputs: string }) => T | Promise<T>;
  codeExtractionStrategy?: "none" | "llm" | "markdown_code_blocks";
  codeExtractor?: (outputs: string | Record<string, unknown>) => string;
  model?: string;
  client?: BaseChatModel;
  runName: string;
  feedbackKey: string;
}) {
  const {
    scorer,
    codeExtractionStrategy,
    codeExtractor,
    model,
    runName,
    feedbackKey,
  } = config;
  let client = config.client;

  if (codeExtractor && codeExtractionStrategy !== "none") {
    throw new Error(
      "`codeExtractor` and `codeExtractionStrategy` cannot both be provided"
    );
  }

  if (codeExtractionStrategy === "llm") {
    if (!model && !client) {
      throw new Error(`You must provide either a "model" string or a "client"`);
    }
  }

  const _wrappedEvaluator = async (params: {
    inputs?: unknown;
    outputs: string | Record<string, unknown>;
    [key: string]: unknown;
  }) => {
    const wrappedScorer = async (params: {
      inputs: unknown;
      outputs: string | Record<string, unknown>;
      [key: string]: unknown;
    }) => {
      let normalizedOutputs = _normalizeOutputsAsString(params.outputs);

      if (codeExtractor) {
        normalizedOutputs = codeExtractor(params.outputs);
      } else if (codeExtractionStrategy === "llm") {
        if (!client) {
          client = await initChatModel(model);
        }
        const llmUserPrompt = ChatPromptTemplate.fromTemplate(
          LLM_EXTRACTION_USER_PROMPT
        );

        if (client.bindTools === undefined) {
          throw new Error("You must pass a model that supports tool calling.");
        }

        const extractCodeTool = tool(() => {}, {
          name: "ExtractCode",
          description: "Tool to call if there is code to extract.",
          schema: z.object({
            code: z.string(),
          }),
        });

        const noCodeTool = tool(() => {}, {
          name: "NoCode",
          description: "Tool to call to indicate no code was found.",
          schema: z.object({
            no_code: z.boolean(),
          }),
        });

        const modelWithTools = client.bindTools([extractCodeTool, noCodeTool]);

        const res = await modelWithTools.invoke([
          {
            role: "system",
            content: LLM_EXTRACTION_SYSTEM_PROMPT,
          },
          {
            role: "user",
            content: await llmUserPrompt.format({ outputs: normalizedOutputs }),
          },
        ]);
        if (res.tool_calls?.[0]?.name === "ExtractCode") {
          normalizedOutputs = res.tool_calls[0].args.code;
        } else {
          return [
            false,
            "Code extraction failed",
            { code_extraction_failed: true },
          ] as const;
        }
      } else if (codeExtractionStrategy === "markdown_code_blocks") {
        normalizedOutputs =
          _extractCodeFromMarkdownCodeBlocks(normalizedOutputs);
      }

      return scorer({
        inputs: params.inputs ?? "",
        outputs: normalizedOutputs,
      });
    };

    return _runEvaluator(runName, wrappedScorer, feedbackKey, {
      ...params,
      inputs: params.inputs ?? "",
      outputs: params.outputs,
    });
  };

  return _wrappedEvaluator;
}
