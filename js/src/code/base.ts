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
  Your response will be passed DIRECTLY into a code execution sandbox for further testing,
  so make sure to extract all code **without modifications**, even if it contains errors,
  since any modifications will ruin the integrity of the testing process.
  Do not respond with any text other than the code you extract.
</Instructions>

<Examples>
  <Example>
    <Input>
      <text>
        Here is some perfectly written TypeScript code:
        
        \`\`\`ts
        print("Hello, world!")
        \`\`\`
      </text>
    </Input>
    <Output>
      print("Hello, world!")
    </Output>
  </Example>
  <Example>
    <Input>
      <text>
        The first thing you should do is import lodash:
        
        \`\`\`ts
        import _ from "lodash"
        \`\`\`
        
        Then, you should use lodash to create an array:
        
        \`\`\`ts
        const tail = _.tail([1, 2, 3, 4, 5]);
        \`\`\`
        
        And finally, you should print the array:
        
        \`\`\`ts
        console.log(tail);
        \`\`\`
        
      </text>
    </Input>
    <Output>
      import _ from "lodash"
      const tail = _.tail([1, 2, 3, 4, 5]);
      console.log(tail);
    </Output>
  </Example>
</Examples>
`;

export const LLM_EXTRACTION_USER_PROMPT = `
Extract code from the following text:

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
      throw new Error("Either model or client must be provided");
    }
  }

  const _wrappedEvaluator = async (params: {
    inputs?: unknown;
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

      const res = await client!.invoke([
        {
          role: "system",
          content: LLM_EXTRACTION_SYSTEM_PROMPT,
        },
        {
          role: "user",
          content: await llmUserPrompt.format({ outputs: normalizedOutputs }),
        },
      ]);
      if (typeof res.content !== "string") {
        throw new Error("Expected a string, got " + typeof res.content);
      }
      normalizedOutputs = res.content;
    } else if (codeExtractionStrategy === "markdown_code_blocks") {
      normalizedOutputs = _extractCodeFromMarkdownCodeBlocks(normalizedOutputs);
    }

    return _runEvaluator(runName, scorer, feedbackKey, {
      ...params,
      inputs: params.inputs ?? "",
      outputs: normalizedOutputs,
    });
  };

  return _wrappedEvaluator;
}
