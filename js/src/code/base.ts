import { initChatModel } from "langchain/chat_models/universal";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { ChatPromptTemplate } from "@langchain/core/prompts";

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
  Omit installation instructions and shell commands from any code you extract.
</Instructions>
`;

export const LLM_EXTRACTION_USER_PROMPT = `
Extract code from the following:

<text>
{outputs}
</text>
`;

const extractCodeToolSchema = {
  type: "function",
  function: {
    name: "ExtractCode",
    description:
      "Tool to call if there is code to extract. Omit installation instructions and shell commands.",
    parameters: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "The code to extract.",
        },
      },
      required: ["code"],
    },
  },
};
const noCodeToolSchema = {
  type: "function",
  function: {
    name: "NoCode",
    description: "Tool to call to indicate no code was found.",
    parameters: {
      type: "object",
      properties: {
        no_code: {
          type: "boolean",
          description: "Whether no code was found.",
        },
      },
      required: ["no_code"],
    },
  },
};

/**
 * Extract code from markdown code blocks in the provided text.
 *
 * Supports both triple backtick code blocks with or without language specifiers.
 *
 * @param text - The text containing markdown code blocks
 * @returns A string containing only the code extracted from code blocks, with blocks
 *          separated by newlines
 */
export function _extractCodeFromMarkdownCodeBlocks(
  text: string
): string | null {
  // Pattern to match code blocks with or without language specifier
  const pattern = /^(?<!`)```(\w*)\n([\s\S]*?)^(?<!`)```$/gm;

  // Find all code blocks
  const excludedLangs = new Set([
    "bash",
    "sh",
    "shell",
    "zsh",
    "fish",
    "console",
    "terminal",
    "json",
  ]);
  const codeBlocks: string[] = [];

  let match = pattern.exec(text);
  while (match !== null) {
    const lang = match[1].trim();
    if (!excludedLangs.has(lang)) {
      codeBlocks.push(match[2]);
    }
    match = pattern.exec(text);
  }

  if (codeBlocks.length === 0) {
    return null; // Return null if no valid code blocks found
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

        const modelWithTools = client.bindTools([
          extractCodeToolSchema,
          noCodeToolSchema,
        ]);

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
        const extractedCode =
          _extractCodeFromMarkdownCodeBlocks(normalizedOutputs);
        if (extractedCode == null) {
          return [
            false,
            "Code extraction failed",
            { code_extraction_failed: true },
          ] as const;
        }
        normalizedOutputs = extractedCode;
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
