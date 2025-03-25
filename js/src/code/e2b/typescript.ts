import { Sandbox } from "@e2b/code-interpreter";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { v4 as uuidv4 } from "uuid";

import { _createBaseCodeEvaluator } from "../base.js";
import { SingleResultScorerReturnType } from "../../types.js";
import {
  PACKAGE_JSON_FILE,
  TYPESCRIPT_EVALUATOR_FILE,
  EXTRACT_IMPORT_NAMES,
} from "./sandbox/files.js";

const E2B_COMMAND = [
  `echo '${PACKAGE_JSON_FILE}' > package.json`,
  `echo '${TYPESCRIPT_EVALUATOR_FILE}' > run_typescript.ts`,
  `echo '${EXTRACT_IMPORT_NAMES}' > extract_import_names.ts`,
  `npm i`,
  `npx tsx extract_import_names.ts | xargs -r -I {} npm install {}`,
].join("&&");

/**
 * Creates an evaluator that checks TypeScript code for syntax and type errors,
 * including those used in dependencies, using E2B's sandboxing.
 *
 * @param config - Optional configuration object
 * @param config.sandbox - E2B sandbox instance to use for the evaluation.
 * @param config.codeExtractionStrategy - Strategy to extract code from the output:
 *   - "none": Use the raw output as-is
 *   - "llm": Use an LLM to extract code from the output
 *   - "markdown_code_blocks": Extract code from markdown code blocks
 * @param config.codeExtractor - Custom function to extract code from the output. Will override `codeExtractionStrategy` if provided.
 * @param config.model - LLM model name (only allowed when codeExtractionStrategy is "llm")
 * @param config.client - LangChain chat model instance (only allowed when codeExtractionStrategy is "llm")
 * @returns A function that evaluates TypeScript code and returns whether it's valid
 */
export function createE2BTypeScriptEvaluator(config: {
  sandbox: Sandbox;
  codeExtractionStrategy?: "none" | "llm" | "markdown_code_blocks";
  codeExtractor?: (outputs: string | Record<string, unknown>) => string;
  model?: string;
  client?: BaseChatModel;
}) {
  if (
    config?.codeExtractionStrategy !== "llm" &&
    (config?.model || config?.client)
  ) {
    throw new Error(
      "model and client may only be passed if codeExtractionStrategy is 'llm'"
    );
  }
  const _scorer = async (params: { outputs: string }) => {
    const uuid = uuidv4();
    const sandbox = config.sandbox;
    try {
      await sandbox.files.write(`${uuid}/outputs.ts`, params.outputs);
      await sandbox.commands.run(E2B_COMMAND, {
        cwd: uuid,
      });
      const res = await sandbox.commands.run(`npx tsx run_typescript.ts`, {
        cwd: uuid,
      });
      return JSON.parse(res.stdout);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (e: any) {
      if (e.result && typeof e.result.stderr === "string") {
        return [false, e.result.stderr];
      }
      throw e;
    }
  };
  return _createBaseCodeEvaluator<SingleResultScorerReturnType>({
    scorer: _scorer,
    runName: "e2b_typescript_evaluator",
    feedbackKey: "typescript_succeeded",
    ...config,
  });
}
